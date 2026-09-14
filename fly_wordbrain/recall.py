"""Supplemental delayed-name decodability from final frozen-brain state.

This trains a separate four-way linear diagnostic. It is not the text decoder
and does not measure language understanding. No word IDs reach the classifier.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .brain import FrozenBrain, array_digest
from .decoder import fit_decoder


CONTEXTS = (8, 16, 32, 64, 128)
SPLITS = ("train", "val", "test")
WORKER = None


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_arrays(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def validate_probes(dataset, probes):
    """Fail before expensive simulation if the controlled contrast is invalid."""
    labels = probes["labels"]
    vocabulary = {word: i for i, word in enumerate(dataset["vocabulary"])}
    if len(labels) != 4 or len(set(labels)) != 4 or any(name not in vocabulary for name in labels):
        raise ValueError("Require four distinct recall names present in the text vocabulary")
    if dataset["vocabulary"][:4] != ["<pad>", "<unk>", "<bos>", "<eos>"]:
        raise ValueError("Unexpected special-token mapping")
    if not probes["metadata"]["usable_with_text_word_encoding"]:
        raise ValueError("Probe was marked invalid for the text vocabulary")
    seen_prompts, seen_ids, templates = set(), set(), {}
    expected_counts = {"train": 80, "val": 40, "test": 40}
    for split in SPLITS:
        items = probes["splits"][split]
        if len(items) != expected_counts[split]:
            raise ValueError("This bounded probe expects exactly 80/40/40 items")
        groups, balance = defaultdict(list), defaultdict(Counter)
        templates[split] = set()
        for item in items:
            words = item["words"]
            context = item["context_words"]
            label = item["target_class"]
            if context not in CONTEXTS or len(words) != context:
                raise ValueError("Context length must equal actual preceding word count")
            if label not in range(4) or labels[label] != item["target_word"]:
                raise ValueError("Target class does not match name")
            expected_ids = [2] + [vocabulary.get(word, 1) for word in words]
            if item["word_ids"] != expected_ids or item["target_id"] != vocabulary[item["target_word"]]:
                raise ValueError("Probe encoding differs from the main text vocabulary")
            if item["cue_word_index"] != 1 or words[1] != item["target_word"]:
                raise ValueError("Recall cue must be at fixed word position 1")
            if sum(word in labels for word in words) != 1 or set(words[-5:]) & set(labels):
                raise ValueError("Target leaks outside its one early cue")
            if item["intervening_words"] != context - 2:
                raise ValueError("Intervening word count differs from prompt")
            if item["id"] in seen_ids or tuple(words) in seen_prompts:
                raise ValueError("Duplicate probe identity or prompt across splits")
            seen_ids.add(item["id"])
            seen_prompts.add(tuple(words))
            templates[split].add(item["template_id"])
            groups[item["pair_group"]].append(item)
            balance[context][label] += 1
        if set(balance) != set(CONTEXTS):
            raise ValueError("Missing context rung")
        for counts in balance.values():
            if set(counts) != set(range(4)) or len(set(counts.values())) != 1:
                raise ValueError("Names must be balanced within each split/context")
        for paired in groups.values():
            if len(paired) != 4 or {item["target_class"] for item in paired} != set(range(4)):
                raise ValueError("Each paired group must contain all four names")
            stems = {tuple(item["words"][:1] + ["NAME"] + item["words"][2:]) for item in paired}
            if len(stems) != 1:
                raise ValueError("Paired prompts differ outside the name cue")
    if any(templates[a] & templates[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Probe templates must be held out across splits")


def init_worker(config):
    global WORKER
    WORKER = FrozenBrain(**config)


def prompt_features(item):
    """Observe only the final word, after BOS and all C lexical inputs."""
    brain = WORKER
    if brain is None:
        raise RuntimeError("Worker has no frozen brain")
    brain.reset()
    started = time.perf_counter()
    kernel_seconds, total_spikes = 0., 0
    feature = None
    for word_id in item["word_ids"]:
        feature = brain.step(word_id)
        kernel_seconds += brain.last_diagnostics["kernel_seconds"]
        total_spikes += brain.last_diagnostics["all_spikes"]
    if feature is None or feature.shape != (256,) or not np.isfinite(feature).all():
        raise RuntimeError("Expected one finite 256-dimensional final observation")
    feature = np.array(feature, dtype=np.float32, copy=True)
    return {
        "X": feature[None, :], "y": np.array([item["target_class"]], np.int64),
        "context_words": np.array([item["context_words"]], np.int64),
        "item_ids": np.array([item["id"]]), "pair_groups": np.array([item["pair_group"]]),
        "diagnostics": {
            "item_id": item["id"], "word_inputs_including_bos": len(item["word_ids"]),
            "wall_seconds": time.perf_counter() - started,
            "kernel_seconds": kernel_seconds, "all_spikes": total_spikes,
            "frozen_graph_sha256": brain.verify_frozen(),
            "feature_sha256": array_digest([feature]),
        },
    }


def extract_recall(dataset_path, probes_path, output, workers=4):
    dataset_path, probes_path, output = Path(dataset_path), Path(probes_path), Path(output)
    if not 1 <= workers <= 4:
        raise ValueError("Require one to four workers for this bounded study")
    dataset, probes = json.loads(dataset_path.read_text()), json.loads(probes_path.read_text())
    validate_probes(dataset, probes)
    output.mkdir(parents=True, exist_ok=True)
    config = {"vocab_size": len(dataset["vocabulary"]), "word_ms": 20.,
              "warmup_ms": 100., "high": .02, "bins": 128, "seed": 1729}
    brain = FrozenBrain(**config)
    manifest = {
        "schema_version": 1, "task": "supplemental delayed-name decodability",
        "dataset_sha256": file_hash(dataset_path), "probes_sha256": file_hash(probes_path),
        "brain": brain.metadata(), "workers": workers,
        "code_sha256": {filename: file_hash(Path(__file__).with_name(filename))
                        for filename in ("brain.py", "recall.py", "decoder.py")},
        "feature_contract": "Reset per prompt; present BOS and all lexical words; classifier sees only final 256-dimensional neural observation",
        "scope": "Same full frozen graph and fixed word encoding as text pilot; independently trained four-way diagnostic",
    }
    del brain
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["identity"] = identity
    manifest_path = output / "feature_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("identity") != identity:
        raise RuntimeError("Recall cache source/config/code differs; use a new output directory")
    atomic_json(manifest_path, manifest)
    progress = {"state": "extracting", "identity": identity, "completed": 0, "total": 160}
    atomic_json(output / "progress.json", progress)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(config,)) as pool:
        for split in SPLITS:
            items = probes["splits"][split]
            chunks = output / "chunks" / split
            chunks.mkdir(parents=True, exist_ok=True)
            pending = {}
            for index, item in enumerate(items):
                path = chunks / f"{index:05d}.npz"
                receipt_path = path.with_suffix(".json")
                if path.exists() and receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt["item_id"] != item["id"] or receipt["npz_sha256"] != file_hash(path):
                        raise RuntimeError("Recall item cache failed source/hash verification")
                    if receipt["frozen_graph_sha256"] != manifest["brain"]["graph_sha256"]:
                        raise RuntimeError("Cached feature used a different graph")
                    progress["completed"] += 1
                else:
                    pending[pool.submit(prompt_features, item)] = path
            for future in as_completed(pending):
                path = pending[future]
                result = future.result()
                receipt = result.pop("diagnostics")
                if receipt["frozen_graph_sha256"] != manifest["brain"]["graph_sha256"]:
                    raise RuntimeError("Worker changed the frozen graph")
                save_arrays(path, **result)
                receipt["npz_sha256"] = file_hash(path)
                atomic_json(path.with_suffix(".json"), receipt)
                progress.update(completed=progress["completed"] + 1, split=split,
                                elapsed_seconds=time.perf_counter() - started, latest=receipt)
                atomic_json(output / "progress.json", progress)
                if progress["completed"] % 8 == 0:
                    print(json.dumps(progress), flush=True)
            arrays = {key: [] for key in ("X", "y", "context_words", "item_ids", "pair_groups")}
            for index in range(len(items)):
                with np.load(chunks / f"{index:05d}.npz", allow_pickle=False) as chunk:
                    for key in arrays:
                        arrays[key].append(chunk[key])
            save_arrays(output / f"{split}.npz", **{key: np.concatenate(value) for key, value in arrays.items()})
    verify = FrozenBrain(**config)
    manifest["final_graph_sha256"] = verify.verify_frozen()
    if manifest["final_graph_sha256"] != manifest["brain"]["graph_sha256"]:
        raise RuntimeError("Graph changed during recall extraction")
    manifest["split_hashes"] = {split: file_hash(output / f"{split}.npz") for split in SPLITS}
    manifest["completed"] = True
    manifest["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(manifest_path, manifest)
    progress.update(state="features_complete", elapsed_seconds=manifest["elapsed_seconds"])
    atomic_json(output / "progress.json", progress)
    return manifest


def load_recall_split(path, items):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in ("X", "y", "context_words", "item_ids", "pair_groups")}
    n = len(items)
    if data["X"].shape != (n, 256) or not np.isfinite(data["X"]).all():
        raise ValueError("Bad recall feature shape or numerical values")
    for key in ("y", "context_words", "item_ids", "pair_groups"):
        if data[key].shape != (n,):
            raise ValueError("Feature metadata length mismatch")
    for index, item in enumerate(items):
        if (data["item_ids"][index] != item["id"] or data["y"][index] != item["target_class"]
                or data["context_words"][index] != item["context_words"]
                or data["pair_groups"][index] != item["pair_group"]):
            raise ValueError("Cached feature item alignment differs from probe")
    data["X"] = data["X"].astype(np.float32)
    data["y"] = data["y"].astype(np.int64)
    return data


def class_metrics(logits, targets):
    """Four real classes: there is no PAD, UNK, BOS or EOS target mask."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.shape != (len(targets), 4) or not len(targets) or not np.isfinite(logits).all():
        raise ValueError("Expected finite four-way logits and at least one target")
    if np.any(targets < 0) or np.any(targets >= 4):
        raise ValueError("Name classes must be 0..3")
    shifted = logits - logits.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(targets)), targets]
    predictions = logits.argmax(axis=1)
    return {"examples": len(targets), "cross_entropy": float(losses.mean()),
            "accuracy": float(np.mean(predictions == targets)),
            "correct": int(np.sum(predictions == targets)), "chance_accuracy": .25,
            "uniform_cross_entropy": float(np.log(4.)),
            "class_counts": np.bincount(targets, minlength=4).tolist()}


def evaluate_recall(decoder, data):
    logits = decoder.logits(data["X"])
    overall = class_metrics(logits, data["y"])
    by_context = {}
    for context in CONTEXTS:
        mask = data["context_words"] == context
        by_context[str(context)] = class_metrics(logits[mask], data["y"][mask])
    return {"overall": overall, "by_context_words": by_context}


def clean_learning_curves(curves):
    # fit_decoder selects using ordinary full-class CE. Its generic word-model
    # diagnostics additionally label class1 UNK; none of those fields apply here.
    result = []
    for curve in curves:
        clean = {"seed": curve["seed"], "l2": curve["l2"], "epochs": []}
        for epoch in curve["epochs"]:
            clean["epochs"].append({"epoch": epoch["epoch"], **{
                split: {key: epoch[split][key] for key in ("examples", "cross_entropy", "accuracy")}
                for split in ("train", "val")}})
        result.append(clean)
    return result


def train_recall(dataset_path, probes_path, output, manifest):
    output = Path(output)
    dataset, probes = json.loads(Path(dataset_path).read_text()), json.loads(Path(probes_path).read_text())
    validate_probes(dataset, probes)
    if file_hash(dataset_path) != manifest["dataset_sha256"] or file_hash(probes_path) != manifest["probes_sha256"]:
        raise RuntimeError("Data source changed since neural feature extraction")
    for split in SPLITS:
        if file_hash(output / f"{split}.npz") != manifest["split_hashes"][split]:
            raise RuntimeError("Feature artifact hash mismatch")
    train = load_recall_split(output / "train.npz", probes["splits"]["train"])
    val = load_recall_split(output / "val.npz", probes["splits"]["val"])
    models, runs = [], []
    # Finish every validation-only selection before loading held-out test values.
    for seed in (0, 1, 2):
        model, curves = fit_decoder(train, val, probes["labels"], seed=seed,
                                   l2_values=(0., 1e-4, 1e-2), epochs=30,
                                   batch_size=32, learning_rate=.01, patience=8)
        model.config.update(task="four_way_delayed_name_decodability", feature_identity=manifest["identity"])
        checkpoint = output / f"decoder_seed_{seed}.pt"
        model.save(checkpoint)
        atomic_json(output / f"learning_curves_seed_{seed}.json", clean_learning_curves(curves))
        models.append(model)
        runs.append({"seed": seed, "selection": model.config,
                     "checkpoint": checkpoint.name, "checkpoint_sha256": file_hash(checkpoint),
                     "train": evaluate_recall(model, train), "val": evaluate_recall(model, val)})
    atomic_json(output / "selection_frozen.json", {
        "seeds": [run["selection"] for run in runs], "test_used_for_selection": False,
        "feature_identity": manifest["identity"],
    })
    test = load_recall_split(output / "test.npz", probes["splits"]["test"])
    for model, run in zip(models, runs):
        run["test"] = evaluate_recall(model, test)
    aggregate = {}
    for context in CONTEXTS:
        metrics = [run["test"]["by_context_words"][str(context)] for run in runs]
        aggregate[str(context)] = {"test_examples_per_seed": metrics[0]["examples"],
            "accuracy_mean": float(np.mean([metric["accuracy"] for metric in metrics])),
            "accuracy_std_across_decoder_seeds": float(np.std([metric["accuracy"] for metric in metrics])),
            "cross_entropy_mean": float(np.mean([metric["cross_entropy"] for metric in metrics])),
            "cross_entropy_std_across_decoder_seeds": float(np.std([metric["cross_entropy"] for metric in metrics])),
            "chance_accuracy": .25}
    report = {
        "schema_version": 1, "task": "delayed-name decodability from frozen neural state",
        "feature_identity": manifest["identity"], "brain": manifest["brain"],
        "dataset_sha256": manifest["dataset_sha256"], "probes_sha256": manifest["probes_sha256"],
        "feature_split_hashes": manifest["split_hashes"], "labels": probes["labels"],
        "architecture": "256 neural features -> affine four-way softmax; no IDs/history supplied to classifier",
        "trainable_parameters": 256 * 4 + 4,
        "training_protocol": "Train-only feature scaling; epoch/L2 selection on validation; report all seeds0/1/2; no test selection",
        "context_definition": "Actual lexical words preceding answer; name cue at word index1; C-2 intervening words; BOS excluded",
        "runs": runs, "test_by_context_words": aggregate,
        "interpretation": "Separate trained memory diagnostic, not the next-word decoder and not a test of language understanding. Held-out query/filler templates; each context has only8 test examples in2 paired groups. Decoder seeds share the same neural features and test items, so seed spread is not a population confidence interval.",
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "progress.json", {"state": "complete", "identity": manifest["identity"],
                                           "completed": 160, "total": 160,
                                           "report_sha256": file_hash(output / "report.json")})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = extract_recall(args.dataset, args.probes, args.output, args.workers)
    report = train_recall(args.dataset, args.probes, args.output, manifest)
    print(json.dumps(report["test_by_context_words"], indent=2), flush=True)


if __name__ == "__main__":
    main()
