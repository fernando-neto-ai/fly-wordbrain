"""Summarize completed frozen-memory readouts with nuisance-group uncertainty.

Usage: python -m scripts.summarize_memory_probe --output path/to/probe
No model fitting, checkpoint selection, or feature extraction occurs here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(group_values, *, seed=0, resamples=10000):
    """Percentile interval for the equal-weight mean across sampled groups."""
    values = np.asarray(group_values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Require finite, nonempty group-level values")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("Bootstrap resamples must be a positive integer")
    draw = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    low, high = np.quantile(values[draw].mean(1), [.025, .975])
    return {"estimate": float(values.mean()), "low": float(low), "high": float(high),
            "confidence": .95, "groups": len(values), "resamples": resamples, "seed": seed,
            "method": "Percentile bootstrap, resampling entire nuisance groups with replacement"}


def _group_results(result, task_rows):
    """Resolve evaluator task-local prediction indices against source rows."""
    predicted = result["predictions"]["test"]
    indices, labels, predictions = (predicted[key] for key in ("row_indices", "labels", "predicted_labels"))
    expected = {i for i, row in enumerate(task_rows) if row["split"] == "test"}
    if (not len(indices) or len(indices) != len(labels) or len(indices) != len(predictions)
            or any(type(i) is not int for i in indices) or len(set(indices)) != len(indices)
            or set(indices) != expected):
        raise ValueError("Test prediction indices do not match the complete task-local test split")
    classes = result["num_classes"]
    groups = defaultdict(list)
    for index, label, prediction in zip(indices, labels, predictions):
        row = task_rows[index]
        if label != row["label"] or type(prediction) is not int or not 0 <= prediction < classes:
            raise ValueError("Prediction labels or class IDs disagree with aligned rows")
        groups[row["group_id"]].append((row, int(label == prediction)))
    records = []
    for group_id, values in sorted(groups.items()):
        if sorted(row["label"] for row, _ in values) != list(range(classes)):
            raise ValueError("Each held-out nuisance group must contain every class exactly once")
        source_ids = {row["source_story_id"] for row, _ in values}
        if len(source_ids) != 1:
            raise ValueError("A nuisance group contains multiple source stories")
        records.append({"group_id": group_id, "source_story_id": next(iter(source_ids)),
                        "count": len(values), "correct": sum(correct for _, correct in values),
                        "accuracy": sum(correct for _, correct in values) / len(values)})
    actual = sum(row["correct"] for row in records) / len(indices)
    reported = result["metrics"]["test"]
    if reported["count"] != len(indices) or not math.isclose(actual, reported["accuracy"], abs_tol=1e-6):
        raise ValueError("Reported test accuracy disagrees with prediction arrays")
    if not math.isfinite(reported["cross_entropy"]):
        raise ValueError("Reported test cross entropy is not finite")
    return records


def summarize(readouts, rows, protocol=None, *, seed=0, resamples=10000):
    if readouts.get("status") != "completed" or not readouts.get("results"):
        raise ValueError("Readouts must be completed before summarization")
    if type(seed) is not int or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    protocol = protocol or {}
    task_rows = defaultdict(list)
    seen_rows, group_owners, source_owners = set(), {}, {}
    for row in rows:
        if row["id"] in seen_rows:
            raise ValueError("Duplicate probe row ID")
        seen_rows.add(row["id"])
        owner = (row["task"], row["split"])
        if row["group_id"] in group_owners and group_owners[row["group_id"]] != owner:
            raise ValueError("A nuisance group crosses tasks or splits")
        source_owner = owner + (row["group_id"],)
        if row["source_story_id"] in source_owners and source_owners[row["source_story_id"]] != source_owner:
            raise ValueError("A source story crosses nuisance groups, probe tasks or splits")
        group_owners[row["group_id"]] = owner
        source_owners[row["source_story_id"]] = source_owner
        task_rows[row["task"]].append(row)
    sample = {}
    for task, values in sorted(task_rows.items()):
        sample[task] = {split: {"groups": len({row["group_id"] for row in values if row["split"] == split}),
                               "rows": sum(row["split"] == split for row in values)}
                        for split in ("train", "val", "test")}
    results = {}
    for key, result in sorted(readouts["results"].items()):
        pieces = key.split("/")
        if len(pieces) != 3 or pieces[0] not in task_rows:
            raise ValueError("Malformed or unknown task/feature/architecture key")
        task, feature, variant = pieces
        records = _group_results(result, task_rows[task])
        metric = result["metrics"]["test"]
        results[key] = {"task": task, "feature": feature, "architecture": result["architecture"],
            "variant": variant, "shuffled_train_labels": result["receipt"]["shuffle_train_labels"],
            "classes": result["num_classes"], "parameter_count": result["trainable_parameters"],
            "selected_validation_epoch": result["selected_epoch"],
            "active_feature_dimensions": result["receipt"]["normalization"]["active_dimensions"],
            "effective_std_floor": result["receipt"]["normalization"]["effective_std_floor"],
            "test_accuracy": metric["accuracy"], "test_cross_entropy": metric["cross_entropy"],
            "chance_accuracy": metric["chance_accuracy"], "majority_accuracy": metric["majority_accuracy"],
            "test_rows": metric["count"], "test_groups": len(records),
            "group_accuracy": records,
            "group_accuracy_interval": interval([row["accuracy"] for row in records], seed=seed, resamples=resamples)}
    comparisons = {}
    for task in sorted(task_rows):
        for architecture in ("linear", "mlp"):
            for first, second in (("on_cells_final", "on_pooled_final"),
                                  ("on_cells_temporal", "on_cells_final")):
                left_key, right_key = (task + "/" + feature + "/" + architecture for feature in (first, second))
                if left_key not in results or right_key not in results:
                    continue
                left = {row["group_id"]: row for row in results[left_key]["group_accuracy"]}
                right = {row["group_id"]: row for row in results[right_key]["group_accuracy"]}
                if set(left) != set(right):
                    raise ValueError("Paired comparison has different nuisance groups")
                differences = []
                for group in sorted(left):
                    if (left[group]["count"] != right[group]["count"]
                            or left[group]["source_story_id"] != right[group]["source_story_id"]):
                        raise ValueError("Paired comparison groups do not refer to the same examples")
                    differences.append({"group_id": group, "accuracy_difference": left[group]["accuracy"] - right[group]["accuracy"]})
                key = task + "/" + first + "-minus-" + second + "/" + architecture
                comparisons[key] = {"task": task, "architecture": architecture, "first": left_key, "second": right_key,
                    "group_differences": differences,
                    "paired_accuracy_difference_interval": interval([row["accuracy_difference"] for row in differences],
                        seed=seed, resamples=resamples)}
    return {"schema": 1, "kind": "frozen_memory_probe_summary", "status": "completed",
        "sample": sample, "probe_data": protocol.get("probe_data"), "snapshot": protocol.get("snapshot"),
        "results": results, "paired_comparisons": comparisons,
        "bootstrap": {"unit": "whole held-out nuisance/source-story group", "seed": seed,
            "resamples": resamples, "confidence": .95, "group_weighting": "equal",
            "independent_word_samples_assumed": False},
        "caveats": ["One frozen checkpoint and one readout seed; no checkpoint-seed or training-seed uncertainty is estimated.",
            "The tasks manipulate earlier input identity/order; they do not measure natural-language prediction or generation.",
            "Temporal features retain earlier activity externally; their advantage does not establish memory in the final neural state.",
            "All nuisance contexts come from the original language-training split; probe test groups are held out only from probe fitting.",
            "Eight held-out nuisance groups per task provide a small context sample; bootstrap intervals are descriptive, may be degenerate, and do not establish broad generalization.",
            "Intervals are pointwise and unadjusted across the displayed comparisons; no winner or additional training configuration is selected.",
            "No matched randomized-connectome control is present, so these probes cannot establish an advantage of biological geometry."]}


def _percent(value):
    return "{:.1f}%".format(100 * value)


def add_scale_sensitivity(summary, primary_readouts, sensitivity, rows, protocol=None, *, seed=0, resamples=10000):
    """Attach a post hoc floor check while preserving all primary result fields."""
    if sensitivity.get("kind") != "posthoc_numerical_scale_sensitivity":
        raise ValueError("Unexpected scale-sensitivity artifact kind")
    primary_keys = {key for key, value in primary_readouts["results"].items() if not value["receipt"]["shuffle_train_labels"]}
    alternate_keys = {key for key, value in sensitivity.get("results", {}).items() if not value["receipt"]["shuffle_train_labels"]}
    if alternate_keys != primary_keys:
        raise ValueError("Scale sensitivity must retain every primary unshuffled task/readout, without selection")
    for key, alternate in sensitivity["results"].items():
        matching_key = key if key in primary_readouts["results"] else key.removesuffix("_shuffled")
        if matching_key not in primary_readouts["results"]:
            raise ValueError("Exploratory cached feature/readout has no primary counterpart: " + key)
        primary = primary_readouts["results"][matching_key]
        for field in ("input_features_sha256", "input_labels_sha256", "split_row_indices", "seed"):
            if field not in alternate["receipt"] or alternate["receipt"][field] != primary["receipt"][field]:
                raise ValueError("Scale sensitivity changed cached inputs, labels, splits, or seed: " + key + "/" + field)
        if alternate["receipt"]["shuffle_train_labels"]:
            controls = [value for original_key, value in primary_readouts["results"].items()
                        if original_key.split("/")[0] == key.split("/")[0]
                        and value["architecture"] == alternate["architecture"] and value["receipt"]["shuffle_train_labels"]]
            if not controls or any(alternate["receipt"]["fitting_train_labels_sha256"] != value["receipt"]["fitting_train_labels_sha256"] for value in controls):
                raise ValueError("Exploratory control changed the fitting-label permutation")
        elif alternate["receipt"]["fitting_train_labels_sha256"] != primary["receipt"]["fitting_train_labels_sha256"]:
            raise ValueError("Exploratory readout changed its fitting labels")
        if alternate["architecture"] != primary["architecture"] or alternate["num_classes"] != primary["num_classes"]:
            raise ValueError("Scale sensitivity changed readout architecture or classes")
    alternate_summary = summarize(sensitivity, rows, protocol, seed=seed, resamples=resamples)
    summary["scale_sensitivity"] = {"kind": sensitivity["kind"], "status": "completed",
        "exploratory": True, "posthoc_after_test_inspection": True, "same_test_reused": True,
        "new_confirmatory_run": False, "winner_selected": False, "same_cached_features_verified": True,
        "rationale": sensitivity.get("rationale"), "config": sensitivity.get("config"),
        "changed_control_keys": {"primary_only": sorted(set(primary_readouts["results"]) - set(sensitivity["results"])),
                                 "exploratory_only": sorted(set(sensitivity["results"]) - set(primary_readouts["results"]))},
        "results": alternate_summary["results"], "paired_comparisons": alternate_summary["paired_comparisons"],
        "bootstrap": alternate_summary["bootstrap"],
        "interpretation": "Normalization sensitivity checked after inspecting primary test results; the same cached features and test groups were reused. This is exploratory evidence, not an independent confirmation or a selected winning model."}
    return summary


def markdown(summary):
    lines = ["# Frozen circuit memory probes", "",
        "Every fitted readout is shown below. Accuracy intervals resample complete held-out nuisance groups, keeping all label variants of a group together.", ""]
    snapshot = summary.get("snapshot") or {}
    if snapshot:
        lines.extend(["Frozen checkpoint step: **{}**; readout seed: **0**.".format(snapshot.get("checkpoint_step", "unknown")), ""])
    for task, splits in summary["sample"].items():
        example = next(result for result in summary["results"].values() if result["task"] == task)
        lines.extend(["**{}:** {} classes; {} train, {} validation, and {} test nuisance groups. The {} test rows are balanced label variants within {} contexts, not {} independent words. Uniform chance is {}.".format(
            task, example["classes"], splits["train"]["groups"], splits["val"]["groups"], splits["test"]["groups"],
            splits["test"]["rows"], splits["test"]["groups"], splits["test"]["rows"], _percent(example["chance_accuracy"])), ""])
    metadata = summary.get("probe_data") or {}
    if metadata.get("identity", {}).get("label_words"):
        lines.extend(["Identity labels: " + ", ".join("`" + word.replace("`", "'") + "`" for word in metadata["identity"]["label_words"]) + ".", ""])
    lines.extend(["| Task | Features | Readout | Test accuracy | Group bootstrap 95% interval | Test CE | Parameters | Selected val epoch | Active features |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in summary["results"].values():
        bounds = row["group_accuracy_interval"]
        lines.append("| {} | {} | {} | {} | {}–{} | {:.4f} | {} | {} | {} |".format(
            row["task"], row["feature"], row["variant"], _percent(row["test_accuracy"]),
            _percent(bounds["low"]), _percent(bounds["high"]), row["test_cross_entropy"],
            row["parameter_count"], row["selected_validation_epoch"], row["active_feature_dimensions"]))
    lines.extend(["", "CE is cross entropy in nats. Epochs are selected using validation CE only; epoch 0 is the initialized head. `_shuffled` readouts use a fixed permutation of training labels and true evaluation labels. `on` and `off` enable or disable temporary weights.", "",
        "The paired differences below use the same held-out groups for both feature sets, with temporary weights enabled. Positive values mean higher accuracy for the first feature set. Values are percentage points.", "",
        "| Task | Readout | First minus second | Difference | Paired group bootstrap 95% interval |",
        "|---|---|---|---:|---:|"])
    for row in summary["paired_comparisons"].values():
        bounds = row["paired_accuracy_difference_interval"]
        lines.append("| {} | {} | {} − {} | {:+.1f} | {:+.1f} to {:+.1f} |".format(
            row["task"], row["architecture"], row["first"].split("/")[1], row["second"].split("/")[1],
            100 * bounds["estimate"], 100 * bounds["low"], 100 * bounds["high"]))
    lines.extend(["", "Normalization uses only probe-training moments. Dimensions at or below the recorded standard-deviation floor are masked; the exact floor and every group's accuracy are retained in `summary.json`.", ""])
    lines.extend("- " + caveat for caveat in summary["caveats"])
    exploratory = summary.get("scale_sensitivity")
    if exploratory:
        config = exploratory.get("config") or {}
        lines.extend(["", "## Exploratory normalization sensitivity", "",
            "This check was added **after inspecting the primary test results**. It reuses the same cached features, labels, and test groups with absolute standard-deviation floor `{}` and relative floor `{}`. The primary results above remain the original analysis. This is a post hoc diagnostic, not a new confirmatory run or a selected winner.".format(
                config.get("absolute_std_floor", "see receipt"), config.get("relative_std_floor", "see receipt")), "",
            "The primary floor masked all early pooled identity features. The focused table compares earlier pooled activity with final pooled, final selected-cell, and temporal selected-cell features while temporary weights are enabled. All {} exploratory records, including disabled-weight and shuffled-label controls, are retained separately in `summary.json`. Intervals again resample the same small set of nuisance groups and do not account for the post hoc choice of normalization.".format(len(exploratory["results"])), "",
            "The exploratory shuffled-label controls use early pooled activity; the primary shuffled-label controls use temporal selected-cell activity. This changed control is recorded explicitly in the JSON.", "",
            "| Task | Features | Readout | Primary accuracy | Exploratory accuracy | Group bootstrap 95% interval | Exploratory CE | Active primary → exploratory | Selected val epoch |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"])
        focused = {"on_pooled_early", "on_pooled_final", "on_cells_final", "on_cells_temporal"}
        for key, row in exploratory["results"].items():
            if row["feature"] not in focused or row["shuffled_train_labels"]:
                continue
            original = summary["results"][key]
            bounds = row["group_accuracy_interval"]
            lines.append("| {} | {} | {} | {} | {} | {}–{} | {:.4f} | {} → {} | {} |".format(
                row["task"], row["feature"], row["variant"], _percent(original["test_accuracy"]),
                _percent(row["test_accuracy"]), _percent(bounds["low"]), _percent(bounds["high"]),
                row["test_cross_entropy"], original["active_feature_dimensions"], row["active_feature_dimensions"],
                row["selected_validation_epoch"]))
        lines.extend(["", "A lower floor can expose small signals and can amplify numerical residue. Recovery from earlier activity addresses when information is accessible; it does not demonstrate retention in the final state. Temporal features still require external storage, and these source contexts remain part of the original language-training corpus."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    readout_file, row_file = args.output / "readouts.json", args.output / "rows.json"
    readouts, rows = json.loads(readout_file.read_text()), json.loads(row_file.read_text())
    if readouts.get("rows_sha256") != file_sha256(row_file):
        raise ValueError("Rows file differs from the readout receipt")
    protocol_file = args.output / "protocol.json"
    protocol = json.loads(protocol_file.read_text()) if protocol_file.exists() else {}
    summary = summarize(readouts, rows, protocol, seed=args.seed, resamples=args.resamples)
    summary["source_sha256"] = {"readouts.json": file_sha256(readout_file), "rows.json": file_sha256(row_file),
        "scripts/summarize_memory_probe.py": file_sha256(__file__)}
    if protocol_file.exists():
        summary["source_sha256"]["protocol.json"] = file_sha256(protocol_file)
    sensitivity_file = args.output / "scale-sensitivity.json"
    if sensitivity_file.exists():
        sensitivity = json.loads(sensitivity_file.read_text())
        add_scale_sensitivity(summary, readouts, sensitivity, rows, protocol, seed=args.seed, resamples=args.resamples)
        summary["source_sha256"]["scale-sensitivity.json"] = file_sha256(sensitivity_file)
    for name, content in (("summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n"),
                          ("report.md", markdown(summary))):
        temporary = args.output / (name + ".partial")
        temporary.write_text(content)
        temporary.replace(args.output / name)
    print(json.dumps({"status": "completed", "readouts": len(summary["results"]),
                      "paired_comparisons": len(summary["paired_comparisons"]), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
