"""Bounded train-only full-graph MPS checks for the new candidate sensory input."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.action_model import CandidateActionBrain, FlyActionSelector
from fly_wordbrain.plastic_train import causal_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-scale", type=float, default=.002)
    parser.add_argument("--stories", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    start = time.monotonic()
    dataset = json.loads(args.dataset.read_text())
    proposals = ActionCandidates(dataset["splits"]["train"], len(dataset["vocabulary"]), top_k=args.top_k)
    stories = dataset["splits"]["train"][:args.stories]
    brain = CandidateActionBrain(args.graph, len(dataset["vocabulary"]), args.global_scale,
        internal_steps=8, leak=.5, seed=0, device="mps", input_high=.02, top_k=args.top_k)
    assert brain.neurons == 166700 and brain.edges == 25582938
    graph_before = brain.verify_frozen()
    records, features_all = [], []
    example = None
    with torch.no_grad():
        for story in stories:
            state = brain.initial_state(1)
            rows = list(causal_rows(story))
            for index, (previous, current, target, _, _) in enumerate(rows):
                ids, probabilities = proposals.candidates(previous, current, exclude_story_id=story["id"])
                p = torch.tensor([previous], device="mps")
                c = torch.tensor([current], device="mps")
                ids = torch.tensor(ids[None], device="mps")
                probabilities = torch.tensor(probabilities[None], dtype=torch.float32, device="mps")
                old = state
                base, state = brain.step(p, c, ids, probabilities, old)
                features_all.append(base.cpu().numpy()[0])
                if index in (1, 32, 64, 96, len(rows) - 1):
                    repeat, _ = brain.step(p, c, ids, probabilities, old)
                    changed, _ = brain.step(p, c, ids.roll(1, dims=1), probabilities, old)
                    records.append({"story_id": story["id"], "position": index,
                        "base": base.cpu().numpy()[0], "repeat": repeat.cpu().numpy()[0],
                        "changed": changed.cpu().numpy()[0],
                        "activity_saturated_fraction": float((state.h.abs() >= .99).float().mean().cpu())})
                if example is None and bool(torch.any(ids == target)):
                    example = (p, c, ids, probabilities, old, target)
    array = np.asarray(features_all, dtype=np.float64)
    std = np.maximum(array.std(axis=0), 1e-6)
    varying = int(np.count_nonzero(array.std(axis=0) > 1e-10))
    visible_late = []
    for record in records:
        difference = float(np.max(np.abs(record["base"] - record["changed"]) / std))
        repeat_noise = float(np.max(np.abs(record["base"] - record["repeat"]) / std))
        threshold = max(10 * repeat_noise, 1e-5)
        record.update(candidate_effect=difference, repeat_noise=repeat_noise,
                      required_effect=threshold, visible=difference > threshold)
        for key in ("base", "repeat", "changed"):
            del record[key]
        if record["position"] >= 64:
            visible_late.append(record["visible"])
    assert example is not None
    model = FlyActionSelector(brain, array.mean(axis=0), std)
    p, c, ids, probabilities, state, target = example
    logits, _ = model.step(p, c, ids, probabilities, state)
    baseline_exact = bool(torch.equal(logits, probabilities.log()))
    label = (ids == target).long().argmax(dim=1)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    gradient_norm = float(torch.sqrt(sum(p.grad.square().sum() for p in model.readout.parameters())).cpu())
    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=.001)
    optimizer.step()
    changed_head = any(bool(torch.any(p != 0)) for p in model.readout.parameters())
    graph_after = brain.verify_frozen()
    checks = {"finite_features": bool(np.isfinite(array).all()), "varying_features": varying == 256,
        "candidate_identity_visible_late": bool(visible_late) and all(visible_late),
        "initial_count_prior_exact": baseline_exact,
        "finite_nonzero_head_gradient": bool(np.isfinite(gradient_norm) and gradient_norm > 0),
        "head_updated": changed_head, "frozen_graph_unchanged": graph_before == graph_after,
        "trainable_parameters_match": model.trainable_parameter_count() == 257 * args.top_k,
        "mps_execution": model.readout.weight.device.type == "mps"}
    result = {"schema": 1, "passed": all(checks.values()), "checks": checks,
        "global_scale": args.global_scale, "top_k": args.top_k,
        "trainable_parameters": model.trainable_parameter_count(),
        "story_ids": [s["id"] for s in stories],
        "training_only": True, "whole_story_exclusion": True, "positions": len(array),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "graph_fingerprint": graph_before, "encoder": brain.encoder_metadata,
        "varying_features": varying, "gradient_norm": gradient_norm,
        "conditional_action_loss": float(loss.detach().cpu()), "observations": records,
        "elapsed_seconds": time.monotonic() - start,
        "torch_version": torch.__version__, "train_started": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
