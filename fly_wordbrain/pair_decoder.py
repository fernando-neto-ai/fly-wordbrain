"""Two independent linear forecasts from the same causal, frozen feature vector.

Both heads observe only X: neither receives the other target or prediction.
Each head fits its own training-only scaler and selects epoch/L2 on validation.
All selections in all arms finish before held-out test arrays are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from .decoder import Decoder, _summary, fit_decoder, load_split as load_single_split


SCHEMA_VERSION = 1
FEATURE_DIMENSIONS = 256
HORIZONS = ("next_1", "next_2")
INPUT_DESCRIPTIONS = {
    "brain": "standardized_frozen_pair_input_brain_features_only",
    "direct": "standardized_fixed_ordered_pair_sensory_projection_only",
    "single_word_brain": "standardized_frozen_single_word_input_brain_features_only",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def load_pair_split(path: Path, vocabulary_size: int) -> Dict[str, np.ndarray]:
    required = {"X", "X_direct", "y", "target_mask", "positions", "current_ids", "previous_ids", "story_ids"}
    with np.load(path, allow_pickle=False) as source:
        if not required.issubset(source.files):
            raise ValueError("Missing pair feature arrays: " + str(sorted(required - set(source.files))))
        data = {key: source[key].copy() for key in required}
    n = len(data["X"])
    if not n:
        raise ValueError("Pair feature split must not be empty")
    for key in ["X", "X_direct"]:
        if data[key].shape != (n, FEATURE_DIMENSIONS) or data[key].dtype != np.float32 or not np.isfinite(data[key]).all():
            raise ValueError(key + " must be finite float32 [n,256]")
    if data["y"].shape != (n, 2) or data["y"].dtype.kind not in "iu":
        raise ValueError("Pair targets must be integer [n,2]")
    if data["target_mask"].shape != (n, 2) or data["target_mask"].dtype != np.bool_:
        raise ValueError("target_mask must be boolean [n,2]")
    if not data["target_mask"][:, 0].all():
        raise ValueError("Every pair row must have a valid next-word target")
    if np.any(data["y"][~data["target_mask"]] != 0):
        raise ValueError("Missing future targets must use masked PAD=0 placeholders")
    for key in ["positions", "current_ids", "previous_ids"]:
        if data[key].shape != (n,) or data[key].dtype.kind not in "iu" or np.any(data[key] < 0):
            raise ValueError(key + " must be nonnegative integer [n]")
    for key in ["y", "current_ids", "previous_ids"]:
        if np.any(data[key] < 0) or np.any(data[key] >= vocabulary_size):
            raise ValueError(key + " exceeds vocabulary")
        data[key] = data[key].astype(np.int64)
    if data["story_ids"].shape != (n,) or data["story_ids"].dtype.kind != "U":
        raise ValueError("story_ids must be Unicode [n]")
    return data


def validate_alignment(data: Dict[str, np.ndarray], dataset: Dict[str, Any], split: str) -> None:
    """Require every source position exactly once and forbid cross-story targets."""
    records = dataset["splits"][split]
    stories = {str(story["id"]): story for story in records}
    if len(stories) != len(records):
        raise ValueError("Duplicate source story IDs")
    expected = set()
    for story_id, story in stories.items():
        ids = story["word_ids"]
        if not ids or ids[0] != 2 or len(story.get("words", [])) > 128:
            raise ValueError("Stories must begin with BOS and contain at most 128 lexical words")
        expected.update((story_id, p) for p in range(len(ids) - 1))
    observed = set()
    for i, (raw_id, position) in enumerate(zip(data["story_ids"], data["positions"])):
        story_id, p = str(raw_id), int(position)
        key = (story_id, p)
        if key not in expected or key in observed:
            raise ValueError("Unexpected or duplicate story-position in pair features")
        observed.add(key)
        story = stories[story_id]
        ids = story["word_ids"]
        mask = story.get("target_mask", [True] * len(ids))
        if len(mask) != len(ids):
            raise ValueError("Source target mask length differs from word IDs")
        previous = ids[p - 1] if p > 0 else 2
        if int(data["current_ids"][i]) != ids[p] or int(data["previous_ids"][i]) != previous:
            raise ValueError("Pair input alignment differs from source previous/current words")
        for head in range(2):
            target_position = p + 1 + head
            valid = target_position < len(ids) and bool(mask[target_position])
            target = ids[target_position] if valid else 0
            if bool(data["target_mask"][i, head]) != valid or int(data["y"][i, head]) != target:
                raise ValueError("Pair future-target alignment or mask differs from source")
    if observed != expected:
        raise ValueError("Pair features omit source story-positions")


def align_single_features(single: Dict[str, np.ndarray], pair: Dict[str, np.ndarray]) -> np.ndarray:
    """Comparisons use precisely the same rows in the same order."""
    for key in ["story_ids", "positions", "current_ids"]:
        if not np.array_equal(single[key], pair[key]):
            raise ValueError("Original single-word feature rows do not align: " + key)
    if not np.array_equal(single["y"], pair["y"][:, 0]):
        raise ValueError("Original single-word next targets do not align")
    if single["X"].shape != pair["X"].shape:
        raise ValueError("Original and paired brain feature shapes differ")
    return single["X"]


class PairDecoder:
    """Two independent saved Decoder instances, both receiving identical raw X."""

    def __init__(self, heads: Sequence[Decoder], config: Optional[Dict[str, Any]] = None):
        if len(heads) != 2 or heads[0].vocabulary != heads[1].vocabulary:
            raise ValueError("Pair decoder requires two heads with identical vocabulary")
        if len(heads[0].scaler.mean) != len(heads[1].scaler.mean):
            raise ValueError("Pair heads must observe the same feature dimensions")
        self.heads = list(heads)
        self.vocabulary = list(heads[0].vocabulary)
        self.config = dict(config or {})
        self.feature_manifest = None
        self.checkpoint_dir = None

    def logits(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        # Stacking immediately before the vocabulary axis handles both [D]
        # and [N,D]. No predicted/true first target enters the second head.
        return np.stack([head.logits(X, batch_size) for head in self.heads], axis=-2)

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        logits = self.logits(X, batch_size)
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return probabilities / probabilities.sum(axis=-1, keepdims=True)

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return self.logits(X, batch_size).argmax(axis=-1)

    def save(self, path: Path, manifest: Optional[Dict[str, Any]] = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for number, head in enumerate(self.heads):
            head.save(path / ("head-%d.pt" % number))
        head_hashes = {"head-%d.pt" % number: file_sha256(path / ("head-%d.pt" % number)) for number in range(2)}
        write_json(path / "pair.json", {"schema_version": SCHEMA_VERSION,
            "architecture": "two_independent_linear_softmax", "heads": ["head-0.pt", "head-1.pt"],
            "head_sha256": head_hashes, "config": self.config, "vocabulary": self.vocabulary})
        if manifest is not None:
            write_json(path / "feature-manifest.json", manifest)

    @classmethod
    def load(cls, path: Path) -> "PairDecoder":
        path = Path(path)
        seen = set()
        for _ in range(5):
            record_path = path / "pair.json" if path.is_dir() else path
            resolved = record_path.resolve()
            if resolved in seen:
                raise ValueError("Cyclic pair checkpoint reference")
            seen.add(resolved)
            record = json.loads(record_path.read_text())
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported pair checkpoint schema")
            if "primary" in record:
                path = record_path.parent / record["primary"]
                continue
            directory = record_path.parent
            if record.get("heads") != ["head-0.pt", "head-1.pt"]:
                raise ValueError("Unexpected pair checkpoint head layout")
            for name in record["heads"]:
                if file_sha256(directory / name) != record["head_sha256"][name]:
                    raise ValueError("Pair checkpoint head checksum mismatch")
            result = cls([Decoder.load(directory / name) for name in record["heads"]], record["config"])
            if result.vocabulary != record["vocabulary"]:
                raise ValueError("Pair metadata vocabulary differs from heads")
            result.checkpoint_dir = directory.resolve()
            manifest_path = directory / "feature-manifest.json"
            if manifest_path.exists():
                result.feature_manifest = json.loads(manifest_path.read_text())
            return result
        raise ValueError("Too many pair checkpoint references")


def fit_pair(train: Dict[str, np.ndarray], val: Dict[str, np.ndarray], vocabulary: Sequence[str], *,
             feature_key: str = "X", seed: int = 0, l2_values: Sequence[float] = (0., 1e-4, .01),
             epochs: int = 30, batch_size: int = 256, learning_rate: float = .01,
             patience: int = 8) -> Tuple[PairDecoder, List[Dict[str, Any]]]:
    """Fit independent heads on valid rows only. No test argument is accepted."""
    heads, curves = [], []
    for number in range(2):
        datasets = []
        for data in [train, val]:
            mask = data["target_mask"][:, number]
            if not mask.any():
                raise ValueError("Each horizon needs valid training and validation targets")
            datasets.append({"X": data[feature_key][mask], "y": data["y"][mask, number]})
        head, curve = fit_decoder(datasets[0], datasets[1], vocabulary, seed=seed,
            l2_values=l2_values, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, patience=patience)
        head.config.update(horizon=number + 1, head_conditioning="same_current_feature_vector; no future word")
        heads.append(head)
        curves.extend({**candidate, "horizon": number + 1} for candidate in curve)
    return PairDecoder(heads, {"seed": seed, "feature_dimensions": len(heads[0].scaler.mean),
        "head_conditioning": "Both independent heads receive only the same current raw feature vector."}), curves


def summarize_forecasts(losses: np.ndarray, predictions: np.ndarray, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    mask, y = data["target_mask"], data["y"]
    horizons = {}
    for number, name in enumerate(HORIZONS):
        valid = mask[:, number]
        metric = _summary(losses[valid, number], predictions[valid, number], y[valid, number])
        metric["interpretation"] = "Next-word conditional metrics, comparable to original next-word readout." if number == 0 else "Second-future-word forecast from the same current state; exp(CE) is not next-word sequence perplexity."
        horizons[name] = metric
    complete = mask.all(axis=1)
    known_pair = complete & (y >= 4).all(axis=1)
    exact = (predictions == y).all(axis=1)
    known = mask & (y >= 4)
    average = float(losses[mask].mean())
    return {"horizons": horizons, "forecast_average": {
        "valid_forecasts": int(mask.sum()), "cross_entropy": average,
        "exp_cross_entropy": math.exp(average) if average < 709 else None,
        "accuracy": float((predictions[mask] == y[mask]).mean()),
        "known_lexical_forecasts": int(known.sum()),
        "known_lexical_cross_entropy": float(losses[known].mean()) if known.any() else None,
        "known_lexical_accuracy": float((predictions[known] == y[known]).mean()) if known.any() else None,
        "interpretation": "Token-weighted mean over valid overlapping horizon forecasts, NOT standard sequence perplexity."},
        "complete_pair": {"examples": int(complete.sum()),
            "exact_accuracy": float(exact[complete].mean()) if complete.any() else None,
            "known_lexical_pairs": int(known_pair.sum()),
            "known_lexical_exact_accuracy": float(exact[known_pair].mean()) if known_pair.any() else None}}


def evaluate_pair(decoder: PairDecoder, X: np.ndarray, data: Dict[str, np.ndarray], batch_size: int = 256) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    losses = np.full(data["y"].shape, np.nan, dtype=np.float64)
    predictions = np.full(data["y"].shape, -1, dtype=np.int64)
    for number, head in enumerate(decoder.heads):
        indices = np.flatnonzero(data["target_mask"][:, number])
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            logits = torch.from_numpy(head.logits(X[chosen], batch_size))
            targets = torch.from_numpy(data["y"][chosen, number])
            losses[chosen, number] = F.cross_entropy(logits, targets, reduction="none").double().numpy()
            predictions[chosen, number] = logits.argmax(dim=-1).numpy()
    if not np.isfinite(losses[data["target_mask"]]).all():
        raise RuntimeError("Nonfinite valid forecast loss")
    arrays = {key: data[key] for key in ["y", "target_mask", "story_ids", "positions", "current_ids", "previous_ids"]}
    arrays.update(losses=losses, predictions=predictions)
    return summarize_forecasts(losses, predictions, data), arrays


class HorizonCounts:
    """One forecast horizon's counts; context never includes another future target."""

    def __init__(self, previous: np.ndarray, current: np.ndarray, target: np.ndarray,
                 vocabulary_size: int, alpha: float = 1., prior_mass: float = 1.):
        if not len(target) or alpha <= 0 or prior_mass <= 0:
            raise ValueError("Counts require nonempty targets and positive smoothing")
        self.alpha, self.prior_mass = float(alpha), float(prior_mass)
        self.unigram_counts = np.bincount(target, minlength=vocabulary_size)
        self.unigram = (self.unigram_counts + alpha) / (len(target) + alpha * vocabulary_size)
        self.current_counts = defaultdict(Counter)
        self.pair_counts = defaultdict(Counter)
        for prev, word, future in zip(previous, current, target):
            self.current_counts[int(word)][int(future)] += 1
            self.pair_counts[(int(prev), int(word))][int(future)] += 1
        self.current_totals = {key: sum(counts.values()) for key, counts in self.current_counts.items()}
        self.pair_totals = {key: sum(counts.values()) for key, counts in self.pair_counts.items()}

    def probability(self, previous: int, current: int, target: int, kind: str = "ordered_pair") -> float:
        probability = float(self.unigram[target])
        if kind == "unigram":
            return probability
        probability = (self.current_counts.get(current, {}).get(target, 0) + self.prior_mass * probability) / (self.current_totals.get(current, 0) + self.prior_mass)
        if kind == "current_word":
            return probability
        if kind != "ordered_pair":
            raise ValueError("Unknown count reference")
        pair = (previous, current)
        return (self.pair_counts.get(pair, {}).get(target, 0) + self.prior_mass * probability) / (self.pair_totals.get(pair, 0) + self.prior_mass)

    def predictions(self, previous: np.ndarray, current: np.ndarray, targets: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray]:
        losses = np.empty(len(targets), dtype=np.float64)
        predicted = np.empty(len(targets), dtype=np.int64)
        common = int(self.unigram.argmax())
        cache = {}
        for i, (p, c, target) in enumerate(zip(previous, current, targets)):
            p, c, target = int(p), int(c), int(target)
            key = (p, c)
            if key not in cache:
                candidates = {common}
                if kind != "unigram":
                    candidates.update(self.current_counts.get(c, {}))
                if kind == "ordered_pair":
                    candidates.update(self.pair_counts.get(key, {}))
                cache[key] = min(candidates, key=lambda t: (-self.probability(p, c, t, kind), t))
            predicted[i] = cache[key]
            losses[i] = -math.log(self.probability(p, c, target, kind))
        return losses, predicted

    def record(self) -> Dict[str, Any]:
        return {"unigram_counts": self.unigram_counts.tolist(), "alpha": self.alpha,
            "prior_mass": self.prior_mass,
            "current_counts": {str(key): dict(counts) for key, counts in sorted(self.current_counts.items())},
            "pair_counts": [{"previous": key[0], "current": key[1], "targets": dict(counts)}
                            for key, counts in sorted(self.pair_counts.items())]}


class PairCountReferences:
    def __init__(self, train: Dict[str, np.ndarray], vocabulary_size: int):
        self.heads = []
        for number in range(2):
            valid = train["target_mask"][:, number]
            self.heads.append(HorizonCounts(train["previous_ids"][valid], train["current_ids"][valid],
                                           train["y"][valid, number], vocabulary_size))

    def evaluate(self, data: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
        metrics, outputs = {}, {}
        for kind in ["unigram", "current_word", "ordered_pair"]:
            losses = np.full(data["y"].shape, np.nan, dtype=np.float64)
            predictions = np.full(data["y"].shape, -1, dtype=np.int64)
            for number, counts in enumerate(self.heads):
                valid = data["target_mask"][:, number]
                losses[valid, number], predictions[valid, number] = counts.predictions(
                    data["previous_ids"][valid], data["current_ids"][valid], data["y"][valid, number], kind)
            metrics[kind] = summarize_forecasts(losses, predictions, data)
            outputs[kind] = {key: data[key] for key in ["y", "target_mask", "story_ids", "positions", "current_ids", "previous_ids"]}
            outputs[kind].update(losses=losses, predictions=predictions)
        return metrics, outputs

    def record(self) -> Dict[str, Any]:
        return {"fit_split": "train", "horizons": [head.record() for head in self.heads],
            "conditioning": "Each horizon separately conditions on observed (previous,current); never on either true future word.",
            "formula": "unigram=(count(target)+1)/(N+V); current=(count(current,target)+unigram)/(count(current)+1); pair=(count(previous,current,target)+current)/(count(previous,current)+1)"}


def load_manifest(features: Path, dataset_sha256: str) -> Dict[str, Any]:
    path = features / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("completed") is not True or manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Feature manifest is incomplete or belongs to another dataset")
    if set(manifest.get("split_hashes", {})) != {"train", "val", "test"}:
        raise ValueError("Feature manifest must hash all three splits")
    brain = manifest.get("brain", {})
    if not brain.get("graph_sha256") or manifest.get("final_graph_sha256") != brain["graph_sha256"]:
        raise ValueError("Extraction manifest does not preserve a frozen graph")
    return manifest


def _check_split_hash(features: Path, split: str, manifest: Dict[str, Any]) -> None:
    if file_sha256(features / (split + ".npz")) != manifest["split_hashes"][split]:
        raise ValueError("Feature split hash differs from manifest: " + split)


def _aggregate_repeats(repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = {}
    for split in ["train", "val", "test"]:
        aggregate[split] = {}
        for horizon in HORIZONS:
            aggregate[split][horizon] = {}
            for metric in ["cross_entropy", "accuracy", "known_lexical_cross_entropy", "known_lexical_accuracy"]:
                values = [record["metrics"][split]["horizons"][horizon][metric] for record in repeats]
                if all(value is not None for value in values):
                    aggregate[split][horizon][metric] = {"mean": float(np.mean(values)),
                        "std": float(np.std(values)), "values": values}
        values = [record["metrics"][split]["forecast_average"]["cross_entropy"] for record in repeats]
        aggregate[split]["forecast_average_cross_entropy"] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
    return aggregate


def run(features: Path, dataset_path: Path, output: Path, *, single_features: Optional[Path] = None,
        seeds: Sequence[int] = (0, 1, 2), l2_values: Sequence[float] = (0., 1e-4, .01),
        epochs: int = 30, batch_size: int = 256, learning_rate: float = .01,
        patience: int = 8, threads: int = 1) -> Dict[str, Any]:
    features, dataset_path, output = Path(features), Path(dataset_path), Path(output)
    single_features = Path(single_features) if single_features is not None else None
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty pair-decoder output directory")
    if not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Provide one to three distinct nonnegative seeds")
    if threads < 1:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(threads)
    dataset = json.loads(dataset_path.read_text())
    vocabulary = dataset["vocabulary"]
    if not isinstance(vocabulary, list) or vocabulary[:4] != ["<pad>", "<unk>", "<bos>", "<eos>"] or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Expected a unique vocabulary with PAD/UNK/BOS/EOS IDs 0/1/2/3")
    story_ids = [{str(story["id"]) for story in dataset["splits"][split]} for split in ["train", "val", "test"]]
    if any(story_ids[i] & story_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Dataset story IDs overlap between splits")
    dataset_sha = file_sha256(dataset_path)
    manifest = load_manifest(features, dataset_sha)
    single_manifest = load_manifest(single_features, dataset_sha) if single_features is not None else None
    if single_manifest is not None:
        for key in ["graph_sha256", "dt_ms", "word_ms", "warmup_ms", "projection_sha256", "feature_dimensions", "neurons", "edges"]:
            if key in manifest["brain"] and manifest["brain"][key] != single_manifest["brain"].get(key):
                raise ValueError("Single-word and pair extraction comparison changes " + key)
    data = {}
    for split in ["train", "val"]:
        _check_split_hash(features, split, manifest)
        data[split] = load_pair_split(features / (split + ".npz"), len(vocabulary))
        validate_alignment(data[split], dataset, split)
        if single_features is not None:
            _check_split_hash(single_features, split, single_manifest)
            old = load_single_split(single_features / (split + ".npz"), len(vocabulary))
            data[split]["X_single"] = align_single_features(old, data[split])
    arms = {"brain": "X", "direct": "X_direct"}
    if single_features is not None:
        arms["single_word_brain"] = "X_single"
    selections, curves = {}, []
    for arm, key in arms.items():
        selections[arm] = []
        for seed in seeds:
            decoder, history = fit_pair(data["train"], data["val"], vocabulary, feature_key=key,
                seed=seed, l2_values=l2_values, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, patience=patience)
            selections[arm].append(decoder)
            curves.extend({**candidate, "arm": arm} for candidate in history)
            print(json.dumps({"phase": "validation_selection_complete", "arm": arm, "seed": seed,
                              "heads": [head.config for head in decoder.heads]}), flush=True)
    # No test NPZ data or target is accessed until EVERY arm/seed/head is selected.
    _check_split_hash(features, "test", manifest)
    data["test"] = load_pair_split(features / "test.npz", len(vocabulary))
    validate_alignment(data["test"], dataset, "test")
    if single_features is not None:
        _check_split_hash(single_features, "test", single_manifest)
        old = load_single_split(single_features / "test.npz", len(vocabulary))
        data["test"]["X_single"] = align_single_features(old, data["test"])
    output.mkdir(parents=True, exist_ok=True)
    arm_records = {}
    for arm, key in arms.items():
        repeats = []
        source_manifest = single_manifest if arm == "single_word_brain" else manifest
        source_features = single_features if arm == "single_word_brain" else features
        for decoder in selections[arm]:
            seed = decoder.config["seed"]
            directory = output / arm / ("seed-%d" % seed)
            decoder.config.update({"arm": arm, "feature_input": key, "input": INPUT_DESCRIPTIONS[arm],
                "dataset_sha256": dataset_sha, "feature_sha256": source_manifest["split_hashes"],
                "paired_feature_sha256": manifest["split_hashes"],
                "feature_manifest_sha256": file_sha256(source_features / "manifest.json"),
                "feature_manifest_file": "feature-manifest.json", "brain": source_manifest["brain"],
                "parameters": sum(parameter.numel() for head in decoder.heads for parameter in head.model.parameters()),
                "seed_policy": "Independent repeat; no best-seed selection.",
                "selection": "Each head independently selects lowest validation CE across epochs/L2."})
            for number, head in enumerate(decoder.heads):
                head.config.update({"arm": arm, "input": INPUT_DESCRIPTIONS[arm], "horizon": number + 1,
                    "dataset_sha256": dataset_sha, "feature_sha256": source_manifest["split_hashes"]})
            decoder.config["selected_heads"] = [dict(head.config) for head in decoder.heads]
            decoder.save(directory, source_manifest)
            # Preserve the extraction file's bytes, not only its JSON meaning.
            (directory / "feature-manifest.json").write_bytes((source_features / "manifest.json").read_bytes())
            split_metrics = {}
            for split in ["train", "val", "test"]:
                split_metrics[split], arrays = evaluate_pair(decoder, data[split][key], data[split], batch_size)
                if split == "test":
                    np.savez_compressed(directory / "test-predictions.npz", **arrays)
            repeat = {"seed": seed, "selected_heads": [head.config for head in decoder.heads], "metrics": split_metrics}
            repeats.append(repeat)
            write_json(directory / "metrics.json", repeat)
        arm_records[arm] = {"repeats": repeats, "aggregate": _aggregate_repeats(repeats)}
        write_json(output / arm / "pair.json", {"schema_version": SCHEMA_VERSION, "primary": "seed-%d/pair.json" % seeds[0]})
        write_json(output / arm / "metrics.json", arm_records[arm])
    references = PairCountReferences(data["train"], len(vocabulary))
    reference_metrics = {}
    for split in ["train", "val", "test"]:
        reference_metrics[split], arrays = references.evaluate(data[split])
        if split == "test":
            for kind, values in arrays.items():
                directory = output / "references"
                directory.mkdir(exist_ok=True)
                np.savez_compressed(directory / (kind + "-test-predictions.npz"), **values)
    config = {"schema_version": SCHEMA_VERSION, "seeds": list(seeds), "primary_seed": seeds[0],
        "arms": list(arms), "feature_dimensions": FEATURE_DIMENSIONS, "vocabulary_size": len(vocabulary),
        "parameters_per_two_head_model": 2 * (FEATURE_DIMENSIONS + 1) * len(vocabulary),
        "l2_values": list(l2_values), "max_epochs": epochs, "patience": patience,
        "batch_size": batch_size, "learning_rate": learning_rate, "cpu_threads": threads,
        "dataset_sha256": dataset_sha, "feature_sha256": manifest["split_hashes"],
        "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in ["pair_decoder.py", "decoder.py"]},
        "torch_version": str(torch.__version__),
        "selection": "All arm/head/seed selections use validation only and finish before loading any test arrays.",
        "standardization": "Each head uses only its valid training rows for mean/std; no shared learned trunk.",
        "target_contract": "Input previous/current; forecast next_1 and next_2 from SAME features. No teacher-forced future word in head2.",
        "loss_contract": "Horizon CE separate; mean across valid overlapping forecasts is NOT sequence perplexity.",
        "masked_outputs": "Unscored horizon losses are NaN and predictions -1 in NPZ; all metrics use target_mask.",
        "limitations": ["Brain dynamics and all synapses remain frozen; only readout weights/biases train.",
            "Independent head2 training cannot improve head1 representations because no learned trunk is shared.",
            "Direct arm measures decodability of engineered ordered sensory input without the brain.",
            "Overlapping forecasts are correlated; use story-level paired analysis, not independent-token significance."]}
    result = {"config": config, "arms": arm_records, "references": reference_metrics}
    write_json(output / "pair.json", {"schema_version": SCHEMA_VERSION, "primary": "brain/seed-%d/pair.json" % seeds[0]})
    write_json(output / "config.json", config)
    write_json(output / "metrics.json", result)
    write_json(output / "training-curves.json", curves)
    write_json(output / "references" / "counts.json", references.record())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-features", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--l2", default="0,0.0001,0.01")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    result = run(args.features, args.dataset, args.output, single_features=args.single_features,
        seeds=[int(seed) for seed in args.seeds.split(",")], l2_values=[float(value) for value in args.l2.split(",")],
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.learning_rate, threads=args.threads)
    print(json.dumps({"output": str(args.output.resolve()),
        "arms": {arm: record["aggregate"]["test"] for arm, record in result["arms"].items()},
        "references": result["references"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
