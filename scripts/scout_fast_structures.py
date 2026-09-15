"""Matched, bounded train-only scouting of existing-edge fast-weight selections.

Probe stories never supply calibration statistics or optimizer updates. The
count proposal model uses all training stories with entire-current-story LOO,
including on the probes; these are gradient-held-out training probes, not a
validation estimate. No validation/test split is indexed and no checkpoint is
written. Run from the project root with ``python -m scripts.scout_fast_structures``.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.action_train import ActionMetrics, utc_now, write_json
from fly_wordbrain.feedback_data import collate_feedback, make_windows, windows_identity
from fly_wordbrain.feedback_diagnostics import measure_action_effect, measure_group_state
from fly_wordbrain.feedback_model import FeedbackActionBrain, FeedbackSelector
from fly_wordbrain.feedback_train import (calibrate, clip_gradients, final_batch,
                                        make_optimizer, source_hashes, window_loss)
from fly_wordbrain.pair_decoder import file_sha256


ARMS = ("original347", "expanded8192_shared", "expanded8192_susceptibility")


def source_receipt():
    return {**source_hashes(), "scripts/scout_fast_structures.py": file_sha256(__file__)}


def synchronize(device):
    if torch.device(device).type == "mps":
        torch.mps.synchronize()
    elif torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def select_windows(stories, dataset_sha, *, batch_size=16, updates=32,
                   calibration_count=256, probe_count=128):
    """Four source-disjoint subsets with fixed order and one seeded shuffle."""
    if len(stories) < 176:
        raise ValueError("Scout requires at least 176 training stories")
    partitions = {"calibration": make_windows(stories[:32]),
                  "updates": make_windows(stories[32:160]),
                  "probe_a": make_windows(stories[160:168]),
                  "probe_b": make_windows(stories[168:176])}
    random.Random(0).shuffle(partitions["updates"])
    required = {"calibration": calibration_count, "updates": batch_size * updates,
                "probe_a": probe_count, "probe_b": probe_count}
    for name, rows in partitions.items():
        if not rows or (name in ("calibration", "updates") and len(rows) < required[name]):
            raise ValueError("Insufficient eight-word windows for " + name)
        partitions[name] = rows[:required[name]]
    ids = {name: {row.story_id for row in rows} for name, rows in partitions.items()}
    for index, name in enumerate(ids):
        for other in list(ids)[index + 1:]:
            if ids[name] & ids[other]:
                raise ValueError("Scout source stories overlap")
    identities = {name: windows_identity(rows, dataset_sha, "train")
                  for name, rows in partitions.items()}
    return partitions, identities


def group_gradients(brain):
    """Shared-rule gradients per anatomical group, before gradient clipping."""
    sums = np.zeros(brain.group_count, dtype=np.float64)
    maxima = np.zeros(brain.group_count, dtype=np.float64)
    entries = 0
    present = False
    for name in brain.rule_parameter_names:
        if name == "slow_susceptibility":
            continue
        parameter = getattr(brain, name)
        entries += parameter.numel() // brain.group_count
        if parameter.grad is not None:
            values = parameter.grad.detach().cpu().double().numpy().reshape(brain.group_count, -1)
            sums += np.square(values).sum(axis=1)
            maxima = np.maximum(maxima, np.abs(values).max(axis=1))
            present = True
    return [{"group": group, "name": name, "shared_parameter_count": entries,
             "gradient_l2": float(np.sqrt(sums[group])) if present else None,
             "gradient_rms": float(np.sqrt(sums[group] / entries)) if present else None,
             "gradient_max_abs": float(maxima[group]) if present else None}
            for group, name in enumerate(brain.group_names)]


def _gradient_norm(parameters):
    squares = [float(p.grad.detach().cpu().double().square().sum())
               for p in parameters if p.grad is not None]
    return float(np.sqrt(sum(squares))) if squares else None


def _with_names(summary, brain):
    for row in summary["groups"]:
        row["name"] = brain.group_names[row["group"]]
    return summary


def _counterfactual_features(model, batch, observations):
    """Intervene on writes only, holding all original sensory inputs fixed."""
    state = model.initial_state(len(batch.identity))
    for position in range(8):
        features, state = model.brain.predict(batch.previous[:, position], batch.current[:, position],
            batch.candidates[:, position], batch.probabilities[:, position], state, plasticity=True)
        if position < 7:
            state = model.brain.observe(state, batch.candidates[:, position], batch.probabilities[:, position],
                                        observations[:, position], position + 1)
    return features, state


def probe(model, windows, proposals, *, batch_size=16, shuffled_feedback=False, progress=None):
    """Paired, whole-story-excluded probes; labels enter only metric reporting."""
    started = time.perf_counter()
    metrics = {name: ActionMetrics() for name in ("fast_on", "fast_off", "count_baseline")}
    if shuffled_feedback:
        metrics["shuffled_write_feedback"] = ActionMetrics()
    logits_all = {name: [] for name in (*metrics, "replay")}
    fast_all, feature_difference, feature_noise = [], [], []
    shuffled_feature_difference, shuffled_fast_difference = [], []
    feedback_changes = 0
    if shuffled_feedback:
        original = np.asarray([window.word_ids8[:7] for window in windows], dtype=np.int64)
        shuffled = original[np.random.default_rng(0).permutation(len(windows))]
        feedback_changes = int(np.count_nonzero(original != shuffled))
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            for offset in range(0, len(windows), batch_size):
                batch = collate_feedback(windows[offset:offset + batch_size], proposals,
                    device=model.brain.device, exclude_own_story=True)
                features, state = model.features(batch, plasticity=True)
                disabled, disabled_state = model.features(batch, plasticity=False)
                repeated, _ = model.features(batch, plasticity=True)
                if bool(torch.count_nonzero(disabled_state.fast).cpu()):
                    raise RuntimeError("Disabled probe wrote fast weights")
                prior = batch.probabilities[:, 7].log()
                logits = {"fast_on": prior + model.correction(features),
                          "fast_off": prior + model.correction(disabled),
                          "count_baseline": prior,
                          "replay": prior + model.correction(repeated)}
                fast_all.append(state.fast.detach().cpu())
                feature_difference.append(((features - disabled) / model.feature_std).detach().cpu())
                feature_noise.append(((features - repeated) / model.feature_std).detach().cpu())
                if shuffled_feedback:
                    changed, changed_state = _counterfactual_features(model, batch,
                        torch.as_tensor(shuffled[offset:offset + batch_size], device=model.brain.device))
                    logits["shuffled_write_feedback"] = prior + model.correction(changed)
                    shuffled_feature_difference.append(((features - changed) / model.feature_std).detach().cpu())
                    shuffled_fast_difference.append((state.fast - changed_state.fast).detach().cpu())
                final = final_batch(batch)
                for name, values in logits.items():
                    logits_all[name].append(values.detach().cpu())
                    if name in metrics:
                        metrics[name].update(values, final)
                if progress:
                    progress(min(offset + batch_size, len(windows)), len(windows))
    finally:
        for module, mode in modes:
            module.training = mode
    combined = {name: torch.cat(chunks) for name, chunks in logits_all.items()}
    difference, noise = torch.cat(feature_difference).double(), torch.cat(feature_noise).double()
    result = {"window_count": len(windows), "metrics": {name: value.summary() for name, value in metrics.items()},
        "fast_action_effect": measure_action_effect(combined["fast_on"], combined["fast_off"]),
        "action_replay_noise": measure_action_effect(combined["fast_on"], combined["replay"]),
        "standardized_feature_difference_max": float(difference.abs().max()),
        "standardized_feature_difference_rms": float(difference.square().mean().sqrt()),
        "standardized_feature_replay_noise_max": float(noise.abs().max()),
        "group_fast_state": _with_names(measure_group_state(torch.cat(fast_all), model.brain.candidate_group.cpu()), model.brain),
        "timestamp_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started}
    if shuffled_feedback:
        result.update(shuffled_feedback_action_effect=measure_action_effect(
            combined["fast_on"], combined["shuffled_write_feedback"]),
            shuffled_feedback_changed_observations=feedback_changes,
            shuffled_feedback_observations=len(windows) * 7,
            shuffled_feedback_standardized_feature_difference_max=float(torch.cat(shuffled_feature_difference).abs().max()),
            shuffled_feedback_fast_difference_max=float(torch.cat(shuffled_fast_difference).abs().max()))
    return result


def run_scout(dataset_path, graph_path, structure_path, output, *, arms=ARMS, device="mps",
              batch_size=16, updates=32, calibration_count=256, probe_count=128, threads=4,
              internal_steps=8, global_scale=.002, shuffled_feedback=False, strict_full_graph=True):
    """Execute each matched arm serially; persist progress and failures atomically."""
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Choose unique known scout arms")
    for name, value in (("batch_size", batch_size), ("updates", updates), ("calibration_count", calibration_count),
                        ("probe_count", probe_count), ("threads", threads), ("internal_steps", internal_steps)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if updates > 32 or batch_size * updates > 512 or calibration_count > 256 or probe_count > 128:
        raise ValueError("Scout exceeds the bounded 32-update/512-window budget")
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a scout receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    graph_file = Path(graph_path) / "graph.npz" if Path(graph_path).is_dir() else Path(graph_path)
    structure_path = Path(structure_path) if structure_path is not None else None
    if any(arm != "original347" for arm in arms) and structure_path is None:
        raise ValueError("Expanded arms require a verified structure sidecar")
    dataset_sha = file_sha256(dataset_path)
    # Deliberately access only this split: do not call validation-aware helpers.
    data = json.loads(Path(dataset_path).read_text())
    stories, vocabulary_size = data["splits"]["train"], len(data["vocabulary"])
    partitions, identities = select_windows(stories, dataset_sha, batch_size=batch_size, updates=updates,
        calibration_count=calibration_count, probe_count=probe_count)
    receipt = {"schema": 1, "kind": "fast_structure_scout", "status": "running", "training_only": True,
        "validation_accessed": False, "test_accessed": False, "checkpoint_written": False,
        "started_utc": utc_now(), "dataset_sha256": dataset_sha, "graph_sha256": file_sha256(graph_file),
        "structure_sha256": file_sha256(structure_path) if structure_path is not None else None,
        "source_sha256": source_receipt(), "device": device, "torch_version": str(torch.__version__),
        "subsets": identities, "seed": 0, "arms_requested": list(arms), "arms": {},
        "config": {"batch_size": batch_size, "updates": updates, "head_lr": .0003, "head_eps": 1e-8,
                   "rule_lr": .003, "rule_eps": 1e-12, "separate_gradient_clipping": True,
                   "fast_feature_calibration": True, "internal_steps": internal_steps,
                   "global_scale": global_scale, "threads": threads, "shuffled_feedback": shuffled_feedback},
        "count_model": "All training stories, subtract the entire current story for every calibration/update/probe proposal",
        "probe_scope": "Disjoint training source stories absent from calibration and optimizer updates; not validation",
        "label_loss": "Conditional candidate cross entropy; missing candidates remain accuracy misses",
        "comparison": "Same trained head with temporary weights on/off; not separately trained ablation heads"}
    started = time.perf_counter()
    def persist(phase, **fields):
        receipt.update(phase=phase, elapsed_seconds=time.perf_counter() - started,
                       updated_utc=utc_now(), progress=fields)
        write_json(output, receipt)
    persist("building_training_proposals")
    torch.set_num_threads(threads)
    proposals = ActionCandidates(stories, vocabulary_size, top_k=10)
    try:
        for arm in arms:
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            row = {"status": "running", "started_utc": utc_now(), "steps": [], "probes": {}}
            receipt["arms"][arm] = row
            arm_started = time.perf_counter()
            persist("loading_brain", arm=arm)
            brain = FeedbackActionBrain(graph_path, vocabulary_size, global_scale=global_scale,
                top_k=10, internal_steps=internal_steps, leak=.5, input_high=.02, seed=0,
                device=device, strict_full_graph=strict_full_graph,
                structure_path=structure_path if arm != "original347" else None,
                trainable_susceptibility=arm == "expanded8192_susceptibility")
            model = FeedbackSelector(brain)
            if strict_full_graph and brain.candidates != (347 if arm == "original347" else 8192):
                raise ValueError("Arm name does not match the selected existing-edge count")
            initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
            row.update(model=model.metadata(), base_graph_before=brain.base_graph_fingerprint())
            row["calibration"] = calibrate(model, partitions["calibration"], proposals, batch_size,
                fast_features=True, progress=lambda done, total: persist("calibration", arm=arm,
                    windows_completed=done, total_windows=total))
            for stage in ("initial", "posttrain"):
                if stage == "posttrain":
                    model.train()
                    optimizer = make_optimizer(model, lr=.0003, rule_lr=.003, rule_eps=1e-12)
                    row["optimizer_groups"] = [{"name": group["name"], "lr": group["lr"], "eps": group["eps"],
                        "parameter_count": sum(p.numel() for p in group["params"])} for group in optimizer.param_groups]
                    for offset in range(0, len(partitions["updates"]), batch_size):
                        step_started = time.perf_counter()
                        batch = collate_feedback(partitions["updates"][offset:offset + batch_size], proposals,
                            device=brain.device, exclude_own_story=True)
                        optimizer.zero_grad(set_to_none=True)
                        features, state = model.features(batch, plasticity=True)
                        logits = batch.probabilities[:, 7].log() + model.correction(features)
                        loss, covered = window_loss(logits, batch)
                        if covered:
                            loss.backward()
                        susceptibility_grad = (brain.slow_susceptibility.grad
                            if brain.slow_susceptibility is not None else None)
                        step = {"step": len(row["steps"]) + 1, "window_count": len(batch.identity),
                            "covered_target_count": covered, "optimizer_updated": bool(covered),
                            "conditional_cross_entropy": float(loss.detach().cpu()) if covered else None,
                            "head_gradient_norm": _gradient_norm(model.readout.parameters()),
                            "rule_gradient_norm": _gradient_norm(brain.parameters()),
                            "shared_rule_group_gradients": group_gradients(brain),
                            "group_fast_state": _with_names(measure_group_state(state.fast,
                                brain.candidate_group, susceptibility_grad), brain)}
                        if covered:
                            clip_gradients(model, separate=True)
                            optimizer.step()
                        synchronize(device)
                        step["elapsed_seconds"] = time.perf_counter() - step_started
                        row["steps"].append(step)
                        persist("training", arm=arm, updates_completed=len(row["steps"]), total_updates=updates)
                        del features, state, logits, loss, batch, susceptibility_grad
                    row["optimizer_updates"] = sum(step["optimizer_updated"] for step in row["steps"])
                    model.zero_grad(set_to_none=True)
                    del optimizer
                row["probes"][stage] = {}
                for scope in ("probe_a", "probe_b"):
                    row["probes"][stage][scope] = probe(model, partitions[scope], proposals,
                        batch_size=batch_size, shuffled_feedback=shuffled_feedback,
                        progress=lambda done, total: persist("probe", arm=arm, stage=stage, scope=scope,
                            windows_completed=done, total_windows=total))
                    persist("probe_completed", arm=arm, stage=stage, scope=scope)
            row["parameter_max_deltas"] = {name: float((p.detach().cpu() - initial[name]).abs().max())
                                           for name, p in model.named_parameters() if p.requires_grad}
            row["base_graph_after"] = brain.base_graph_fingerprint()
            row["frozen_fingerprint"] = brain.verify_frozen()
            if row["base_graph_before"] != row["base_graph_after"]:
                raise RuntimeError("Scout changed the canonical base graph")
            row.update(status="completed", elapsed_seconds=time.perf_counter() - arm_started,
                       completed_utc=utc_now())
            persist("arm_completed", arm=arm)
            del model, brain, initial
            gc.collect()
            if torch.device(device).type == "mps":
                torch.mps.empty_cache()
        receipt["source_sha256_after"] = source_receipt()
        receipt["graph_sha256_after"] = file_sha256(graph_file)
        if (receipt["source_sha256_after"] != receipt["source_sha256"]
                or receipt["graph_sha256_after"] != receipt["graph_sha256"]):
            raise RuntimeError("Source or canonical graph changed during scout")
        receipt["status"] = "completed"
        persist("completed")
    except BaseException as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        persist("failed")
        raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--global-scale", type=float, default=.002)
    parser.add_argument("--internal-steps", type=int, default=8)
    parser.add_argument("--shuffled-feedback", action="store_true")
    args = parser.parse_args()
    result = run_scout(args.dataset, args.graph, args.structure, args.output, arms=args.arms,
        device=args.device, threads=args.threads, global_scale=args.global_scale,
        internal_steps=args.internal_steps, shuffled_feedback=args.shuffled_feedback)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
