#!/usr/bin/env python3
"""Route by hypothesis testing: which task cue lets the brain best explain this input?

The trained gate reads the input once and guesses. An active-inference gate tries each
task hypothesis and keeps the one under which its own surprise is lowest -- the task is the
explanation that minimises free energy. No label is needed and nothing is circular: the
model is scored on its own predictions.

Surprise has to be the same quantity under every hypothesis or the comparison is rigged.
So under each cue the brain is run over the window and scored on how well it predicts the
window's own next tokens in the language range -- identical in form for every cue. A cue
trained for pooled classification may well make the brain a poor generative model of its
inputs; if so this routing fails, and that failure is the finding.

A second variant routes by self-consistency instead: under cue c, how much probability
mass lands outside task c's own range, normalised by the arm's typical leakage for that
task. Reported alongside, not instead.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from compare_unified_arms import rebuild
from fly_wordbrain.unified_model import LANGUAGE, SENTIMENT, TOXICITY
from train_task_gate import windows
NAMES = {LANGUAGE: "language", SENTIMENT: "sentiment", TOXICITY: "toxicity"}


@torch.no_grad()
def surprise_under(unified, brain, ids, mask, cue, device):
    ids, mask = ids.to(device), mask.to(device)
    hidden = brain(input_ids=ids, inputs_embeds=unified.embed(ids, cue), attention_mask=mask,
                   use_cache=False, return_dict=True).last_hidden_state
    logits = unified.read(hidden)[0]
    lang = logits[:, :-1, :unified.tokens].float()
    target, valid = ids[:, 1:], mask[:, 1:].bool()
    ce = F.cross_entropy(lang.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
    per_row = (ce * valid).sum(1) / valid.sum(1).clamp_min(1)
    start, stop = unified.ranges()[cue]
    probs = logits.float().softmax(-1)
    leak = 1 - probs[..., start:stop].sum(-1)
    leak_row = (leak * mask).sum(1) / mask.sum(1).clamp_min(1)
    return per_row.cpu(), leak_row.cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True); p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--model", type=Path, required=True); p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--language", type=Path, required=True); p.add_argument("--sentiment", type=Path, required=True)
    p.add_argument("--toxicity", type=Path); p.add_argument("--rows", type=int, default=200)
    p.add_argument("--width", type=int, default=32); p.add_argument("--device", default="mps")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    model, unified, _ = rebuild(a.config, a.model, a.groups, a.run, a.checkpoint, a.device)
    brain = model.brain
    g = np.random.default_rng(1729)
    sources = [(LANGUAGE, json.loads(a.language.read_text())["validation"]),
               (SENTIMENT, json.loads(a.sentiment.read_text())["audit"])]
    if a.toxicity and unified.toxicity_classes:
        sources.append((TOXICITY, json.loads(a.toxicity.read_text())["audit"]))
    cues = [t for t, _ in sources]
    surprise, leakage, labels = [], [], []
    for task, rows in sources:
        ids, mask = windows(rows, a.width, limit=a.rows, generator=g)
        s_cols, l_cols = [], []
        for c in cues:
            s, l = surprise_under(unified, brain, ids, mask, c, a.device)
            s_cols.append(s); l_cols.append(l)
        surprise.append(torch.stack(s_cols, 1)); leakage.append(torch.stack(l_cols, 1))
        labels.append(torch.full((ids.shape[0],), cues.index(task)))
    S, Lk, y = torch.cat(surprise), torch.cat(leakage), torch.cat(labels)
    # Normalise leakage per cue by its own median so 1e-4 and 4e-3 baselines are comparable.
    Ln = Lk / Lk.median(0).values.clamp_min(1e-9)
    report = {"arm": a.config, "checkpoint": a.checkpoint, "cues": [NAMES[c] for c in cues],
              "rows_per_task": int(a.rows), "width": a.width}
    for name, score in (("route_by_min_surprise", S), ("route_by_min_normalised_leakage", Ln)):
        pred = score.argmin(1)
        acc = float((pred == y).float().mean())
        conf = {}
        for i, t in enumerate(cues):
            row = pred[y == i]
            conf[NAMES[t]] = {NAMES[cues[int(c)]]: int((row == c).sum()) for c in row.unique()}
        recalls = [float((pred[y == i] == i).float().mean()) for i in range(len(cues))]
        report[name] = {"accuracy": acc, "balanced_accuracy": float(np.mean(recalls)),
                        "recall": dict(zip([NAMES[c] for c in cues], recalls)), "confusion": conf}
    # What the surprise matrix looks like: mean CE of each source under each cue.
    report["mean_surprise_by_source_and_cue"] = {
        NAMES[t]: {NAMES[c]: float(S[y == i, j].mean()) for j, c in enumerate(cues)}
        for i, t in enumerate(cues)}
    report["majority_class"] = float(torch.bincount(y).max() / y.numel())
    report["learned_gate_reference"] = "99.59% (2-way, Y-era) / 93.32% (3-way) on the same 32-token windows"
    print(json.dumps(report, indent=2))
    if a.output: a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
