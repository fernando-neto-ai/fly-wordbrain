"""Audit saved paired-word artifacts without loading or advancing a neural brain."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from fly_wordbrain.decoder import Decoder
from fly_wordbrain.pair_brain import DirectProjection, OrderedPairEncoder
from fly_wordbrain.pair_decoder import PairDecoder
from fly_wordbrain.pair_extract import story_rows

SPLITS = ("train", "val", "test")
ARMS = ("brain", "direct", "single_word_brain")


class AuditFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def source_path(name):
    path = Path(name)
    return ROOT / path if len(path.parts) > 1 else ROOT / "fly_wordbrain" / path


def check_sources(run, manifest):
    checked = {}
    for name, expected in manifest["code_sha256"].items():
        actual = digest(source_path(name))
        require(actual == expected, f"Current extraction source differs: {name}")
        checked[name] = actual
        snapshot = run / "source" / Path(name).name
        if snapshot.exists():
            require(digest(snapshot) == expected, f"Saved extraction snapshot differs: {snapshot}")
    locks = []
    for lock in (run / "extraction-source-lock.json", run / "source-lock.json",
                 run / "source" / "extraction-lock.json"):
        if lock.exists():
            for name, expected in read_json(lock).items():
                require(digest(source_path(name)) == expected, f"Current source differs from {lock}: {name}")
            locks.append({"path": str(lock), "sha256": digest(lock)})
    for name, expected in manifest["brain"].get("upstream_python_sha256", {}).items():
        require(digest(ROOT / "vendor" / "doomfly" / "doom" / name) == expected,
                f"Upstream source differs from extraction: {name}")
    return {"current_source_sha256": checked, "verified_source_locks": locks}


def validate_rows(data, dataset, split):
    expected = {(story["id"], row["position"]): row
                for story in dataset["splits"][split] for row in story_rows(story)}
    keys = [(str(story), int(position)) for story, position in zip(data["story_ids"], data["positions"])]
    require(len(keys) == len(set(keys)) == len(expected) and set(keys) == set(expected),
            f"{split}: duplicate, missing or unexpected causal rows")
    n = len(keys)
    require(data["X"].shape == data["X_direct"].shape == (n, 256), f"{split}: incorrect feature dimensions")
    require(np.isfinite(data["X"]).all() and np.isfinite(data["X_direct"]).all(), f"{split}: nonfinite features")
    require(data["y"].shape == data["target_mask"].shape == (n, 2), f"{split}: incorrect two-horizon shape")
    for i, key in enumerate(keys):
        row = expected[key]
        require(int(data["previous_ids"][i]) == row["previous_id"]
                and int(data["current_ids"][i]) == row["current_id"]
                and tuple(data["y"][i]) == tuple(row["targets"])
                and tuple(data["target_mask"][i]) == tuple(row["target_mask"]),
                f"{split}: causal input/target/mask mismatch at {key}")
    require(np.all(data["target_mask"][:, 0]), f"{split}: unexpected masked first horizon")
    require(np.all(data["y"][~data["target_mask"]] == 0), f"{split}: missing targets must carry PAD0")
    return {"rows": n, "targets_by_horizon": data["target_mask"].sum(axis=0).tolist(),
            "max_lexical_context": int(data["positions"].max()),
            "complete_unique_causal_rows": True,
            "nonconstant_brain_columns": int((data["X"].std(axis=0) > 1e-8).sum()),
            "nonconstant_direct_columns": int((data["X_direct"].std(axis=0) > 1e-8).sum())}


def train_scaler(X):
    mean = X.mean(axis=0, dtype=np.float64)
    scale = X.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.
    return mean.astype(np.float32), scale.astype(np.float32)


def numpy_forecasts(model, X, y, mask):
    """Independent affine/logsumexp computation; never call Decoder.evaluate."""
    losses = np.full(y.shape, np.nan, np.float64)
    predictions = np.full(y.shape, -1, np.int64)
    errors = []
    for horizon, head in enumerate(model.heads):
        weight = head.model.weight.detach().cpu().numpy().astype(np.float64)
        bias = head.model.bias.detach().cpu().numpy().astype(np.float64)
        max_error = 0.
        for start in range(0, len(X), 256):
            end = min(start + 256, len(X))
            # Match the saved transform's float32 rounding, then evaluate the
            # affine map independently with NumPy float64 arithmetic.
            scaled = ((X[start:end].astype(np.float32) - head.scaler.mean) / head.scaler.scale).astype(np.float32)
            direct = scaled.astype(np.float64) @ weight.T + bias
            runtime = head.logits(X[start:end]).astype(np.float64)
            difference = float(np.max(np.abs(runtime - direct)))
            max_error = max(max_error, difference)
            require(np.allclose(runtime, direct, atol=1e-5, rtol=1e-5),
                    f"Horizon {horizon + 1}: runtime logits disagree with NumPy affine map")
            chosen = mask[start:end, horizon]
            if not chosen.any():
                continue
            selected = direct[chosen]
            shifted = selected - selected.max(axis=1, keepdims=True)
            target = y[start:end, horizon][chosen]
            values = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(selected)), target]
            indices = np.arange(start, end)[chosen]
            losses[indices, horizon] = values
            predictions[indices, horizon] = selected.argmax(axis=1)
        errors.append(max_error)
    return losses, predictions, errors


def manual_metrics(losses, predictions, y, mask):
    result = {}
    for h in range(2):
        valid = mask[:, h]
        known = valid & (y[:, h] != 1)
        lexical = valid & (y[:, h] >= 4)
        result[f"next_{h + 1}"] = {
            "examples": int(valid.sum()), "cross_entropy": float(losses[valid, h].mean()),
            "accuracy": float(np.mean(predictions[valid, h] == y[valid, h])),
            "known_target_cross_entropy": float(losses[known, h].mean()) if known.any() else None,
            "known_lexical_cross_entropy": float(losses[lexical, h].mean()) if lexical.any() else None,
            "known_lexical_accuracy": float(np.mean(predictions[lexical, h] == y[lexical, h])) if lexical.any() else None,
            "unknown_fraction": float(np.mean(y[valid, h] == 1)),
        }
    result["overlapping_forecast_average_cross_entropy"] = float(losses[mask].mean())
    return result


def audit(run, dataset_path, single_run, report):
    torch.set_num_threads(1)
    output = run / "decoder"
    dataset = read_json(dataset_path)
    dataset_hash = digest(dataset_path)
    require(len(dataset["vocabulary"]) == 1024, "This audit is bound to the 1024-word pilot")
    pair_manifest = read_json(run / "features" / "manifest.json")
    single_manifest = read_json(single_run / "features" / "manifest.json")
    for name, base, manifest in (("pair", run, pair_manifest), ("single", single_run, single_manifest)):
        require(manifest["completed"] is True, f"{name}: incomplete extraction")
        require(manifest["dataset_sha256"] == dataset_hash, f"{name}: dataset hash mismatch")
        require(manifest["final_graph_sha256"] == manifest["brain"]["graph_sha256"], f"{name}: extraction graph changed")
        report["sources"][name] = check_sources(base, manifest)
        for split in SPLITS:
            require(digest(base / "features" / f"{split}.npz") == manifest["split_hashes"][split],
                    f"{name}/{split}: feature artifact hash mismatch")
    unchanged = ("upstream_commit", "upstream_python_sha256", "sensory_arrays_sha256", "superclasses_sha256",
                 "neurons", "edges", "synaptic_contacts", "graph_sha256", "dt_ms", "word_ms", "warmup_ms",
                 "retinal_inputs", "readout_population", "readout_neurons", "feature_dimensions", "features",
                 "projection_sha256", "encoder_high", "kernel", "plasticity", "lamina_bias", "decoder_temporal_memory")
    for key in unchanged:
        require(key in pair_manifest["brain"] and key in single_manifest["brain"]
                and pair_manifest["brain"][key] == single_manifest["brain"][key],
                f"Pair comparison changes graph/dynamics/projection metadata: {key}")
    report["unchanged_graph_dynamics_fields"] = list(unchanged)
    metrics = read_json(output / "metrics.json")
    for name, expected in metrics["config"]["source_sha256"].items():
        require(digest(source_path(name)) == expected, f"Decoder training source changed: {name}")
    report["sources"]["decoder"] = metrics["config"]["source_sha256"]
    require(metrics["config"]["dataset_sha256"] == dataset_hash
            and metrics["config"]["feature_sha256"] == pair_manifest["split_hashes"], "Decoder source identity mismatch")
    brain_meta = pair_manifest["brain"]
    encoder = OrderedPairEncoder(len(dataset["vocabulary"]), brain_meta["retinal_inputs"],
                                 seed=brain_meta["encoder_seed"], high=brain_meta["encoder_high"])
    projection = DirectProjection(brain_meta["retinal_inputs"], brain_meta["direct_control"]["dimensions"],
                                  brain_meta["direct_control"]["seed"])
    require(encoder.metadata() == brain_meta["input_encoding"], "Reconstructed ordered stimulus metadata differs")
    require(encoder.sha256 == brain_meta["encoder_sha256"], "Ordered stimulus digest differs from brain manifest")
    require(projection.metadata() == brain_meta["direct_control"], "Reconstructed direct projection metadata differs")
    data, single_data, direct_cache = {}, {}, {}
    for split in SPLITS:
        data[split] = read_arrays(run / "features" / f"{split}.npz")
        single_data[split] = read_arrays(single_run / "features" / f"{split}.npz")
        report["counts"][split] = validate_rows(data[split], dataset, split)
        for key in ("story_ids", "positions", "current_ids"):
            require(np.array_equal(single_data[split][key], data[split][key]), f"{split}: original single-word rows differ")
        require(np.array_equal(single_data[split]["y"], data[split]["y"][:, 0]), f"{split}: original targets differ")
        require(single_data[split]["X"].shape == data[split]["X"].shape, f"{split}: original feature shape differs")
        for i, pair in enumerate(zip(data[split]["previous_ids"], data[split]["current_ids"])):
            pair = tuple(map(int, pair))
            if pair not in direct_cache:
                direct_cache[pair] = projection(encoder(pair))
            require(np.array_equal(direct_cache[pair], data[split]["X_direct"][i]),
                    f"{split}, row{i}: direct control is not exact projected retinal stimulus")
    report["direct_control"] = {"all_rows_recomputed": True, "unique_ordered_pairs": len(direct_cache),
                                "encoder_sha256": encoder.sha256, "projection_sha256": projection.sha256}
    original = Decoder.load(single_run / "decoder" / "checkpoint-seed-0.pt")
    for arm in ARMS:
        model_path = output / arm / "seed-0"
        model = PairDecoder.load(model_path)
        require(model.vocabulary == dataset["vocabulary"], f"{arm}: checkpoint vocabulary differs")
        count = sum(parameter.numel() for head in model.heads for parameter in head.model.parameters())
        require(count == 526336 == model.config["parameters"], f"{arm}: incorrect trainable parameter count")
        source_manifest = single_manifest if arm == "single_word_brain" else pair_manifest
        source_base = single_run if arm == "single_word_brain" else run
        require(model.config["dataset_sha256"] == dataset_hash
                and model.config["feature_sha256"] == source_manifest["split_hashes"]
                and model.config["paired_feature_sha256"] == pair_manifest["split_hashes"], f"{arm}: checkpoint data hashes differ")
        expected_manifest_hash = digest(source_base / "features" / "manifest.json")
        require(model.config["feature_manifest_sha256"] == expected_manifest_hash
                and digest(model_path / "feature-manifest.json") == expected_manifest_hash,
                f"{arm}: saved extraction manifest bytes differ")
        require(model.config["brain"] == source_manifest["brain"], f"{arm}: saved neural metadata differs")
        feature_key = "X_direct" if arm == "direct" else "X"
        features = {split: (single_data if arm == "single_word_brain" else data)[split][feature_key] for split in SPLITS}
        arm_report = {"parameters": count, "seed": 0, "splits": {}, "checkpoint_head_sha256": {
            f"head-{h}.pt": digest(model_path / f"head-{h}.pt") for h in range(2)}}
        report["arms"][arm] = arm_report
        for h, head in enumerate(model.heads):
            mean, scale = train_scaler(features["train"][data["train"]["target_mask"][:, h]])
            require(np.array_equal(mean, head.scaler.mean) and np.array_equal(scale, head.scaler.scale),
                    f"{arm}/head{h}: scaler does not equal valid training-only mean/std")
        arm_report["train_only_scalers_exact"] = True
        if arm == "single_word_brain":
            head = model.heads[0]
            for key, value in original.model.state_dict().items():
                require(torch.equal(value, head.model.state_dict()[key]),
                        f"TRIAGE: unchanged single-word head0 weights differ from original: {key}")
            require(np.array_equal(head.scaler.mean, original.scaler.mean)
                    and np.array_equal(head.scaler.scale, original.scaler.scale),
                    "TRIAGE: unchanged single-word head0 scaler differs from original")
            selected_fields = ("seed", "l2", "selected_epoch", "validation_cross_entropy", "learning_rate", "batch_size", "architecture")
            for key in selected_fields:
                require(head.config[key] == original.config[key],
                        f"TRIAGE: unchanged single-word head0 selected configuration differs: {key}")
            arm_report["original_nextword_checkpoint_exact"] = True
            arm_report["original_selected_fields"] = {key: original.config[key] for key in selected_fields}
        saved = read_arrays(model_path / "test-predictions.npz")
        for key in ("y", "target_mask", "story_ids", "positions", "current_ids", "previous_ids"):
            require(np.array_equal(saved[key], data["test"][key]), f"{arm}: prediction receipt row metadata differs: {key}")
        require(np.isnan(saved["losses"][~saved["target_mask"]]).all()
                and np.all(saved["predictions"][~saved["target_mask"]] == -1),
                f"{arm}: invalid forecast slots have incorrect sentinels")
        saved_metrics = read_json(model_path / "metrics.json")["metrics"]
        for split in SPLITS:
            target, mask = data[split]["y"], data[split]["target_mask"]
            losses, predictions, logits_errors = numpy_forecasts(model, features[split], target, mask)
            manual = manual_metrics(losses, predictions, target, mask)
            for horizon in ("next_1", "next_2"):
                require(abs(manual[horizon]["cross_entropy"] - saved_metrics[split]["horizons"][horizon]["cross_entropy"]) <= 1e-5,
                        f"{arm}/{split}/{horizon}: manually recomputed CE differs from report")
            if split == "test":
                error = float(np.max(np.abs(losses[mask] - saved["losses"][mask])))
                require(error <= 1e-5, f"{arm}: per-target stored loss disagrees with NumPy by {error:.8g}")
                require(np.array_equal(predictions[mask], saved["predictions"][mask]),
                        f"{arm}: per-target stored argmax disagrees with NumPy")
                arm_report["test_receipt_loss_max_absolute_error"] = error
            arm_report["splits"][split] = {"manual_metrics": manual, "logits_max_absolute_errors": logits_errors}
    report.update(passed=True, dataset_sha256=dataset_hash,
                  graph_sha256=pair_manifest["brain"]["graph_sha256"],
                  scope="Saved artifacts and seed0 of all three arms; no NativeBrain instantiated or neural trajectory executed. Graph fidelity uses immutable extraction start/end receipts plus identical dynamics/source metadata.",
                  numeric_tolerances={"logits_atol": 1e-5, "logits_rtol": 1e-5,
                                      "per_target_loss_absolute": 1e-5, "reported_mean_ce_absolute": 1e-5})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--single-run", type=Path, required=True)
    args = parser.parse_args()
    report = {"passed": False, "sources": {}, "counts": {}, "arms": {}}
    output = args.run / "checkpoint-audit.json"
    try:
        audit(args.run, args.dataset, args.single_run, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"passed": False, "audit": str(output), "error": report["error"]}), file=sys.stderr)
        raise SystemExit(1) from exc
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "audit": str(output), "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
