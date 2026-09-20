#!/usr/bin/env python3
"""Does the brain's state carry task identity, or just settle depth?

The router head reaches 100% within a couple of hundred updates, which is far too fast to
mean much. It has an obvious shortcut: language chunks settle for 32 steps and chess
positions for 5, so the two states differ in how long the recurrence has been running
before anything about words or boards is considered. A probe that reads depth would look
exactly like a probe that reads content.

So this measures the question directly, at **matched settle depth**, and carries its own
floor:

    trained router      what the trained head scores on matched-depth states
    fresh probe         a linear probe fitted on held-out states and scored on a
                        disjoint split -- "is task identity linearly decodable"
    shuffled control    the same probe with the labels permuted

The shuffled control is not decoration. The settled state has 49,393 dimensions and the
probe sees a few thousand samples, so *any* two sets are linearly separable on the training
split; only the held-out gap between the real and shuffled probes means anything.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_connectorch as trainer
from fly_wordbrain.chess_encoding import features_from_packed
from fly_wordbrain.chess_model import settle
from fly_wordbrain.unified_model import CHESS, LANGUAGE, SENTIMENT, UnifiedFly

TASK_NAMES = {LANGUAGE: "language", CHESS: "chess", SENTIMENT: "sentiment"}


@torch.no_grad()
def language_states(brain, rows, steps, device, batch_size=8, pad_id=0):
    """Settled state after exactly `steps` tokens, so depth matches the chess side."""
    collected = []
    for start in range(0, len(rows), batch_size):
        group = [r for r in rows[start:start + batch_size] if len(r["ids"]) > steps]
        if not group:
            continue
        inputs, _, mask = trainer.batch_tensors(group, device, pad_id)
        out = brain(inputs[:, :steps], attention_mask=mask[:, :steps], use_cache=False,
                    return_dict=True)
        collected.append(out.last_hidden_state[:, -1].float().cpu())
    return torch.cat(collected)


@torch.no_grad()
def chess_states(unified, arrays, span, steps, device, limit, chunk=128):
    stop = min(span[1], span[0] + limit)
    boards, extras = arrays["boards"][span[0]:stop], arrays["extras"][span[0]:stop]
    collected = []
    for start in range(0, boards.shape[0], chunk):
        features = torch.from_numpy(features_from_packed(boards[start:start + chunk],
                                                         extras[start:start + chunk])).to(device)
        drive = unified.board_drive(features).t().contiguous()
        collected.append(settle(unified.brain, drive, steps).float().cpu())
    return torch.cat(collected)


def linear_probe(states, labels, seed=1729, epochs=600, weight_decay=1e-2, rate=5e-2):
    """Fit on half, score the other half. Returns (train, held-out) accuracy.

    Initialised at zero rather than from the ambient RNG: logistic regression is convex, so
    a fixed start needs no symmetry breaking and the result stops depending on whatever
    else has drawn from the global generator. A probe whose answer moves between calls
    cannot be used as a control.
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(states.shape[0], generator=generator)
    states, labels = states[order], labels[order]
    # Standardise on the training half only.
    split = states.shape[0] // 2
    mean, deviation = states[:split].mean(0), states[:split].std(0).clamp_min(1e-6)
    states = (states - mean) / deviation
    train, test = (states[:split], labels[:split]), (states[split:], labels[split:])
    model = torch.nn.Linear(states.shape[1], 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(train[0]), train[1]).backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = lambda pair: float((model(pair[0]).argmax(-1) == pair[1]).float().mean())
        return accuracy(train), accuracy(test)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="A trained unified arm's directory")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chess-corpus", type=Path, required=True)
    parser.add_argument("--config", default="U32unifiedfixed")
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 32])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    specification = json.loads((ROOT / "experiments/configs" / (args.config + ".json")).read_text())
    training = specification["training"]
    reference = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                                                     local_files_only=True, torch_dtype=torch.float32)
    groups, _ = trainer.load_groups(args.groups)
    from argparse import Namespace
    build_args = Namespace(d_embed=training["d_embed"], plasticity=training["plasticity"],
                           readout_rank=training["readout_rank"],
                           history_length=training["history_length"], seed=training["seed"],
                           leak=training.get("leak", "fixed"))
    model = trainer.build_model(reference, build_args, groups)
    # Rebuild the arm in the form it was trained in. Defaults describe the two-task arms,
    # which predate both the sentiment range and the task cue.
    form = json.loads((ROOT / "experiments/configs" / (args.config + ".json")).read_text()) \
        .get("unified", {})
    space = form.get("output_space", {})
    unified = UnifiedFly(model.brain, settle_steps=5, readout_rank=training["readout_rank"],
                         tokens=space.get("language_tokens", 1024),
                         moves=space.get("chess_moves", 1968),
                         classes=space.get("sentiment_classes", 0),
                         toxicity_classes=space.get("toxicity_classes", 0),
                         tasks=len(form.get("tasks", ["language", "chess"])),
                         task_cue=form.get("task_cue", False),
                         sentiment_pooling=form.get("sentiment_pooling", "mean"))
    model.unified = unified

    saved = torch.load(args.run / args.checkpoint, map_location="cpu", weights_only=False)
    trainer.restore_parameters(model, saved["parameters"])
    model = model.to(args.device).eval()

    data = json.loads(args.data.read_text())
    arrays = np.load(args.chess_corpus / "corpus.npz")
    report = {"run": str(args.run), "checkpoint": args.checkpoint,
              "updates": saved["cursor"]["updates"],
              # The cap asked for, not the count achieved. Language states come one per
              # validation story, so the split size bounds it; each depth records what it
              # actually used.
              "samples_per_task_requested": args.samples,
              "note": "Language states are taken after exactly `steps` tokens so that both "
                      "tasks have run the recurrence the same number of times.",
              "by_settle_depth": {}}

    for steps in args.steps:
        language = language_states(model.brain, data["validation"], steps, args.device)
        chess = chess_states(unified, arrays, arrays["span_validation"], steps, args.device,
                             args.samples)
        count = min(language.shape[0], chess.shape[0], args.samples)
        states = torch.cat([language[:count], chess[:count]])
        labels = torch.cat([torch.full((count,), LANGUAGE), torch.full((count,), CHESS)])

        with torch.no_grad():
            trunk = unified.trunk(unified.ln(states.to(args.device)))
            routed = unified.router(trunk).argmax(-1).cpu()
        router_accuracy = float((routed == labels).float().mean())
        # Accuracy alone cannot say *how* a router fails. A three-task router has somewhere
        # else to put a state, and language and sentiment are both token streams, so a
        # language state read as sentiment is a different failure from one read as chess.
        confusion = {}
        for task in (LANGUAGE, CHESS):
            predicted = routed[labels == task]
            confusion[TASK_NAMES[task]] = {
                TASK_NAMES[choice]: int((predicted == choice).sum())
                for choice in sorted(TASK_NAMES) if int((predicted == choice).sum())}

        # A perfect probe is not yet an interesting result. If the two tasks simply drive
        # the brain to different magnitudes, a reader of "how big is this vector" scores
        # perfectly while the state represents nothing task-like. So the norm is measured
        # on its own, and the probe is repeated on direction alone.
        norms = states.norm(dim=1, keepdim=True)
        norm_train, norm_test = linear_probe(norms, labels)
        direction_train, direction_test = linear_probe(states / norms.clamp_min(1e-6), labels)
        real_train, real_test = linear_probe(states, labels)
        generator = torch.Generator().manual_seed(4242)
        shuffled = labels[torch.randperm(labels.shape[0], generator=generator)]
        fake_train, fake_test = linear_probe(states, shuffled)

        report["by_settle_depth"][str(steps)] = {
            "samples_per_task_used": count,
            "language_states_available": int(language.shape[0]),
            "trained_router_accuracy": router_accuracy,
            "router_predictions_by_true_task": confusion,
            "linear_probe_train_accuracy": real_train,
            "linear_probe_heldout_accuracy": real_test,
            "state_norm_only_heldout_accuracy": norm_test,
            "direction_only_heldout_accuracy": direction_test,
            "mean_state_norm": {"language": float(norms[labels == LANGUAGE].mean()),
                                "chess": float(norms[labels == CHESS].mean())},
            "shuffled_label_probe_train_accuracy": fake_train,
            "shuffled_label_probe_heldout_accuracy": fake_test,
            "probe_margin_over_shuffled": real_test - fake_test,
        }
        print(f"settle depth {steps}: router {router_accuracy:.3f} | probe held-out "
              f"{real_test:.3f} (train {real_train:.3f}) | shuffled held-out {fake_test:.3f} "
              f"(train {fake_train:.3f}) | norm alone {norm_test:.3f} | direction alone "
              f"{direction_test:.3f}", flush=True)

    # Read every depth, not just the deepest. Summarising only the last one hid a router
    # that scored 1.000 at depth 32 and 0.515 at depth 5 behind a clean verdict.
    scored = {depth: body["trained_router_accuracy"]
              for depth, body in report["by_settle_depth"].items()}
    weakest_depth = min(scored, key=scored.get)
    weakest = scored[weakest_depth]
    if weakest > .9:
        reading = ("The router holds at every settle depth measured "
                   f"({', '.join(f'{d}: {a:.3f}' for d, a in sorted(scored.items()))}), "
                   "so it is reading task content rather than recurrence length.")
    elif max(scored.values()) > .9:
        reading = (f"The router splits by settle depth ("
                   f"{', '.join(f'{d}: {a:.3f}' for d, a in sorted(scored.items()))}). It is "
                   f"not simply reading how long the recurrence has run -- that would fail "
                   f"everywhere -- but it is not depth-invariant either, so its accuracy is "
                   f"conditional on the depth a task is normally run at. See "
                   f"router_predictions_by_true_task at depth {weakest_depth} for where the "
                   f"mass goes.")
    else:
        reading = ("The router collapses once settle depth is matched, so its training "
                   "accuracy was reading how long the recurrence had been running, not "
                   "what it was running on.")
    report["reading"] = reading
    report["router_accuracy_by_settle_depth"] = scored
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
