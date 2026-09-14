"""CPU linear readout of a frozen connectome; no neural gradients or token features.

Feature files are train.npz, val.npz and test.npz. Each contains X, y,
story_ids, positions and current_ids. Only X is supplied to the classifier.
Fit scaling on train, choose epoch/L2 on validation, and load test only after
all choices are frozen. Multiple seeds are repeats, never a seed-selection sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FORMAT_VERSION = 1
UNK_ID = 1


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        X = _features(X)
        if not len(X):
            raise ValueError("Cannot fit scaling without training examples")
        mean = X.mean(axis=0, dtype=np.float64)
        scale = X.std(axis=0, dtype=np.float64)
        scale[scale < 1e-8] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _features(X)
        if X.shape[1] != len(self.mean):
            raise ValueError("Feature dimension does not match trained decoder")
        result = (X - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("Standardized features must be finite")
        return np.ascontiguousarray(result, dtype=np.float32)


def _features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("Features must be a finite [examples, features] matrix")
    return X


def load_split(path: Path, vocabulary_size: int) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        required = {"X", "y", "story_ids", "positions", "current_ids"}
        if not required.issubset(source.files):
            raise ValueError("Missing feature arrays: " + str(sorted(required - set(source.files))))
        data = {key: source[key].copy() for key in required}
    X = _features(data["X"])
    if not len(X):
        raise ValueError("Every split must have at least one scored target")
    for key in required - {"X"}:
        if data[key].shape != (len(X),):
            raise ValueError("Feature metadata lengths do not match: " + key)
    for key in ["y", "current_ids", "positions"]:
        if data[key].dtype.kind not in "iu" or np.any(data[key] < 0):
            raise ValueError(key + " must contain nonnegative integers")
    for key in ["y", "current_ids"]:
        if np.any(data[key] >= vocabulary_size):
            raise ValueError(key + " exceeds vocabulary")
    if data["story_ids"].dtype.kind not in "US":
        raise ValueError("story_ids must be Unicode/string arrays, never objects")
    data["X"] = X
    data["y"] = data["y"].astype(np.int64)
    data["current_ids"] = data["current_ids"].astype(np.int64)
    return data


def validate_split_membership(data: Dict[str, np.ndarray], dataset: Dict[str, Any], split: str) -> None:
    stories = {str(story["id"]): story for story in dataset["splits"][split]}
    allowed = set(stories)
    observed = {str(value) for value in data["story_ids"]}
    if not observed.issubset(allowed):
        raise ValueError("Feature story IDs do not belong to dataset split " + split)
    for story_id, position, current, target in zip(data["story_ids"], data["positions"], data["current_ids"], data["y"]):
        story = stories[str(story_id)]
        ids = story["word_ids"]
        p = int(position)
        if p + 1 >= len(ids) or int(current) != ids[p] or int(target) != ids[p + 1]:
            raise ValueError("Feature input/target alignment differs from source story")
        if not story.get("target_mask", [True] * len(ids))[p + 1]:
            raise ValueError("Feature file includes a masked source target")


def _summary(losses: np.ndarray, predictions: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    ce = float(np.mean(losses))
    known = y != UNK_ID
    known_ce = float(np.mean(losses[known])) if known.any() else None
    lexical = y >= 4
    lexical_ce = float(np.mean(losses[lexical])) if lexical.any() else None
    return {
        "examples": len(y), "cross_entropy": ce,
        "perplexity": math.exp(ce) if ce < 709 else None,
        "accuracy": float(np.mean(predictions == y)),
        "known_target_examples": int(known.sum()),
        "known_target_cross_entropy": known_ce,
        "known_target_perplexity": math.exp(known_ce) if known_ce is not None and known_ce < 709 else None,
        "known_target_accuracy": float(np.mean(predictions[known] == y[known])) if known.any() else None,
        "unknown_target_fraction": float(np.mean(~known)),
        "known_lexical_examples": int(lexical.sum()),
        "known_lexical_cross_entropy": lexical_ce,
        "known_lexical_perplexity": math.exp(lexical_ce) if lexical_ce is not None and lexical_ce < 709 else None,
        "known_lexical_accuracy": float(np.mean(predictions[lexical] == y[lexical])) if lexical.any() else None,
    }


class Decoder:
    """Saved linear softmax readout. Prediction always takes raw brain features."""

    def __init__(self, model: nn.Linear, scaler: Standardizer, vocabulary: Sequence[str], config: Dict[str, Any]):
        self.model = model.cpu().eval()
        self.scaler = scaler
        self.vocabulary = list(vocabulary)
        self.config = dict(config)

    def logits(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        raw = np.asarray(X)
        single = raw.ndim == 1
        scaled = self.scaler.transform(raw[None, :] if single else raw)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(scaled), batch_size):
                outputs.append(self.model(torch.from_numpy(scaled[start:start + batch_size])).numpy())
        result = np.concatenate(outputs) if outputs else np.empty((0, len(self.vocabulary)), np.float32)
        return result[0] if single else result

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        logits = self.logits(X, batch_size)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=-1, keepdims=True)

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return np.asarray(self.logits(X, batch_size).argmax(axis=-1), dtype=np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> Dict[str, Any]:
        scaled = self.scaler.transform(X)
        losses, predictions = [], []
        with torch.inference_mode():
            for start in range(0, len(scaled), batch_size):
                logits = self.model(torch.from_numpy(scaled[start:start + batch_size]))
                target = torch.from_numpy(np.asarray(y[start:start + batch_size], dtype=np.int64))
                losses.append(F.cross_entropy(logits, target, reduction="none").double().numpy())
                predictions.append(logits.argmax(dim=1).numpy())
        return _summary(np.concatenate(losses), np.concatenate(predictions), y)

    def save(self, path: Path) -> None:
        payload = {"format_version": FORMAT_VERSION, "vocabulary": self.vocabulary,
                   "config": self.config, "mean": torch.from_numpy(self.scaler.mean.copy()),
                   "scale": torch.from_numpy(self.scaler.scale.copy()),
                   "state_dict": {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "Decoder":
        path = Path(path)
        if path.is_dir():
            path = path / "checkpoint.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported decoder checkpoint version")
        mean, scale = payload["mean"].numpy(), payload["scale"].numpy()
        if mean.ndim != 1 or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid saved standardizer")
        model = nn.Linear(len(mean), len(payload["vocabulary"]))
        model.load_state_dict(payload["state_dict"])
        return cls(model, Standardizer(mean, scale), payload["vocabulary"], payload["config"])


def fit_decoder(train: Dict[str, np.ndarray], val: Dict[str, np.ndarray], vocabulary: Sequence[str], *,
                seed: int = 0, l2_values: Sequence[float] = (0.0, 1e-4, 1e-2),
                epochs: int = 30, batch_size: int = 256, learning_rate: float = 0.01,
                patience: int = 8) -> Tuple[Decoder, List[Dict[str, Any]]]:
    """Fit using train and validation only; intentionally has no test argument."""
    if not 1 <= epochs <= 50 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("Require 1--50 epochs, positive batch size, learning rate and patience")
    if not 1 <= len(l2_values) <= 3 or any(not math.isfinite(float(v)) or v < 0 for v in l2_values):
        raise ValueError("Choose 1--3 finite, nonnegative L2 penalties")
    scaler = Standardizer.fit(train["X"])
    X = torch.from_numpy(scaler.transform(train["X"]))
    y = torch.from_numpy(train["y"])
    best = None
    best_ce = float("inf")
    curves = []
    for l2 in l2_values:
        # Identical initialization/order within a seed isolates the L2 choice.
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        model = nn.Linear(X.shape[1], len(vocabulary)).cpu()
        optimizer = torch.optim.Adam([{"params": [model.weight], "weight_decay": float(l2)},
                                      {"params": [model.bias], "weight_decay": 0.0}], lr=learning_rate)
        candidate_best = float("inf")
        stale = 0
        curve = {"seed": seed, "l2": float(l2), "epochs": []}
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(X), generator=generator)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(X[indices]), y[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite decoder training loss")
                loss.backward()
                optimizer.step()
            decoder = Decoder(model, scaler, vocabulary, {})
            train_metrics = decoder.evaluate(train["X"], train["y"], batch_size)
            val_metrics = decoder.evaluate(val["X"], val["y"], batch_size)
            ce = val_metrics["cross_entropy"]
            curve["epochs"].append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            if ce < best_ce:
                best_ce = ce
                saved_model = nn.Linear(X.shape[1], len(vocabulary))
                saved_model.load_state_dict({key: value.detach().clone() for key, value in model.state_dict().items()})
                best = Decoder(saved_model, scaler, vocabulary, {"seed": seed, "l2": float(l2),
                    "selected_epoch": epoch, "validation_cross_entropy": ce,
                    "learning_rate": learning_rate, "batch_size": batch_size,
                    "architecture": "linear_softmax", "input": "standardized_frozen_brain_features_only"})
            if ce < candidate_best - 1e-8:
                candidate_best = ce
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        curves.append(curve)
    if best is None:
        raise RuntimeError("No finite validation checkpoint selected")
    return best, curves


class CountReferences:
    """Train-only unigram and bigram with a smoothed unigram backoff prior."""

    def __init__(self, train: Dict[str, np.ndarray], vocabulary_size: int, alpha: float = 1.0, prior_mass: float = 1.0):
        self.vocabulary_size = vocabulary_size
        self.alpha, self.prior_mass = alpha, prior_mass
        self.unigram_counts = np.bincount(train["y"], minlength=vocabulary_size)
        self.unigram = (self.unigram_counts + alpha) / (len(train["y"]) + alpha * vocabulary_size)
        self.bigrams = defaultdict(Counter)
        for current, target in zip(train["current_ids"], train["y"]):
            self.bigrams[int(current)][int(target)] += 1
        self.totals = {current: sum(counts.values()) for current, counts in self.bigrams.items()}

    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        y = data["y"]
        unigram_pred = int(self.unigram.argmax())
        predictions = np.empty(len(y), dtype=np.int64)
        probabilities = np.empty(len(y), dtype=np.float64)
        cached = {}
        for i, (current, target) in enumerate(zip(data["current_ids"], y)):
            current, target = int(current), int(target)
            counts = self.bigrams.get(current, {})
            probabilities[i] = (counts.get(target, 0) + self.prior_mass * self.unigram[target]) / (self.totals.get(current, 0) + self.prior_mass)
            if current not in cached:
                # All unobserved continuations have only their unigram prior.
                candidates = set(counts) | {unigram_pred}
                cached[current] = min(candidates, key=lambda t: (-(counts.get(t, 0) + self.prior_mass * self.unigram[t]), t))
            predictions[i] = cached[current]
        return {
            "unigram": _summary(-np.log(self.unigram[y]), np.full(len(y), unigram_pred), y),
            "smoothed_bigram": _summary(-np.log(probabilities), predictions, y),
        }

    def record(self) -> Dict[str, Any]:
        return {"fit_split": "train", "unigram_alpha": self.alpha, "bigram_prior_mass": self.prior_mass,
                "formula": "P(t|c)=(count_train(c,t)+prior_mass*P_unigram(t))/(count_train(c)+prior_mass)",
                "unigram_counts": self.unigram_counts.tolist(),
                "bigram_counts": {str(k): dict(v) for k, v in sorted(self.bigrams.items())}}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(features: Path, dataset_path: Path, output: Path, *, seeds: Sequence[int] = (0,),
        l2_values: Sequence[float] = (0.0, 1e-4, 1e-2), epochs: int = 30,
        batch_size: int = 256, learning_rate: float = 0.01, patience: int = 8,
        threads: int = 1) -> Dict[str, Any]:
    features, dataset_path, output = Path(features), Path(dataset_path), Path(output)
    if not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Provide 1--3 distinct nonnegative seeds")
    if threads < 1:
        raise ValueError("threads must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty decoder output directory")
    torch.set_num_threads(threads)
    dataset = json.loads(dataset_path.read_text())
    vocabulary = dataset["vocabulary"]
    if not isinstance(vocabulary, list) or len(vocabulary) < 2 or not all(isinstance(token, str) for token in vocabulary):
        raise ValueError("Dataset vocabulary must be a list of at least two strings")
    source_ids = [{str(story["id"]) for story in dataset["splits"][split]} for split in ["train", "val", "test"]]
    if any(source_ids[i] & source_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Dataset story IDs overlap between train/val/test")
    data = {split: load_split(features / (split + ".npz"), len(vocabulary)) for split in ["train", "val"]}
    for split in data:
        validate_split_membership(data[split], dataset, split)
    selections, curves = [], []
    for seed in seeds:
        decoder, curve = fit_decoder(data["train"], data["val"], vocabulary, seed=seed,
            l2_values=l2_values, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
        selections.append(decoder)
        curves.extend(curve)
    # First load/access of held-out test arrays occurs after every selection.
    data["test"] = load_split(features / "test.npz", len(vocabulary))
    validate_split_membership(data["test"], dataset, "test")
    references = CountReferences(data["train"], len(vocabulary))
    repeats = [{"selected": decoder.config,
                "metrics": {split: decoder.evaluate(d["X"], d["y"], batch_size) for split, d in data.items()}}
               for decoder in selections]
    reference_metrics = {split: references.evaluate(d) for split, d in data.items()}
    aggregate = {split: {metric: {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
                        for metric in ["cross_entropy", "perplexity", "accuracy", "known_target_cross_entropy", "known_lexical_cross_entropy"]
                        for values in [[r["metrics"][split][metric] for r in repeats]] if all(v is not None for v in values)}
                 for split in data}
    config = {"format_version": FORMAT_VERSION, "seeds": list(seeds), "primary_seed": seeds[0],
              "seed_policy": "Independent repeats; no selection between seeds. checkpoint.pt uses the first seed.",
              "l2_values": list(l2_values), "max_epochs": epochs, "patience": patience,
              "batch_size": batch_size, "learning_rate": learning_rate, "cpu_threads": threads,
              "feature_dimension": int(data["train"]["X"].shape[1]), "vocabulary_size": len(vocabulary),
              "selection": "Lowest validation CE across L2 and epochs, separately for each seed; ties keep first.",
              "gradient_scope": "Only linear readout weight and bias; connectome features are immutable arrays.",
              "standardization": "Training-only per-feature mean/std; std below 1e-8 replaced with 1.",
              "known_target": "Exclude only UNK ID 1; EOS and all other scored labels remain.",
              "known_lexical": "Only IDs >=4, excluding all reserved tokens including EOS and UNK.",
              "loss_units": "Natural logarithms, token-weighted mean; perplexity=exp(cross_entropy).",
              "torch_version": str(torch.__version__), "dataset_sha256": _sha256(dataset_path),
              "feature_sha256": {split: _sha256(features / (split + ".npz")) for split in data}}
    metrics = {"config": config, "repeats": repeats, "aggregate": aggregate, "references": reference_metrics,
               "primary": repeats[0], "limitations": ["Frozen-circuit decoder training does not train the neural wiring.",
               "Repeated seeds share the same corpus and connectome; they are not independent biological samples."]}
    output.mkdir(parents=True, exist_ok=True)
    for decoder in selections:
        decoder.config.update({"dataset_sha256": config["dataset_sha256"], "feature_sha256": config["feature_sha256"]})
        decoder.save(output / ("checkpoint-seed-%d.pt" % decoder.config["seed"]))
    selections[0].save(output / "checkpoint.pt")
    _write_json(output / "config.json", config)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "training-curves.json", curves)
    _write_json(output / "references.json", references.record())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0", help="One to three comma-separated seeds; no best-seed selection")
    parser.add_argument("--l2", default="0,0.0001,0.01", help="Up to three comma-separated weight penalties")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    metrics = run(args.features, args.dataset, args.output, seeds=[int(x) for x in args.seeds.split(",")],
        l2_values=[float(x) for x in args.l2.split(",")], epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience, threads=args.threads)
    print(json.dumps({"output": str(args.output.resolve()), "primary": metrics["primary"],
                      "test_references": metrics["references"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
