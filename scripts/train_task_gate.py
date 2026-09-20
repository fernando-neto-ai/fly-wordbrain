#!/usr/bin/env python3
"""Train and score the input-side task gate, against the controls that make it mean something.

The gate answers the question the router cannot: given an input and no task label, which
task is this? The router reads the trunk *after* the task cue has been injected, so using
it to pick a task is circular. This sees the raw input.

Evaluated exactly as the pipeline would run it. Language arrives as 32-token chunks,
because that is the chunk size the trainer uses, not as whole stories -- whole rows are
76-319 tokens against SST-2's 3-116, so a gate given rows scores in the nineties by
reading length. Chess is excluded from the reported accuracy entirely: a board is 780
floats and separating it from a token sequence is a type check, so including it would
inflate the figure with a free class.

Three numbers are reported together and the gate's is only meaningful beside the other two:

    gate            the model, over pooled token content
    length only     the same decision from the valid-token count alone
    majority class  the trivial floor for this split
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fly_wordbrain.task_gate import LengthOnlyGate, TaskGate


def windows(rows, width, pad=0, limit=None, generator=None):
    """One window per row, taken where the pipeline would take it.

    A language row is longer than the chunk, so a window is a random interior slice --
    which is what the trainer feeds the brain. A sentiment row is shorter, so it is the
    whole row, padded. That asymmetry is real and is exactly why the length control exists.
    """
    rows = rows[:limit] if limit else rows
    ids = np.full((len(rows), width), pad, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.float32)
    for index, row in enumerate(rows):
        body = row["ids"]
        if len(body) > width:
            start = int(generator.integers(0, len(body) - width)) if generator else 0
            body = body[start:start + width]
        ids[index, :len(body)] = body
        mask[index, :len(body)] = 1.0
    return torch.from_numpy(ids), torch.from_numpy(mask)


def build(language, sentiment, width, limit, seed):
    generator = np.random.default_rng(seed)
    li, lm = windows(language, width, limit=limit, generator=generator)
    si, sm = windows(sentiment, width, limit=limit, generator=generator)
    ids = torch.cat([li, si])
    mask = torch.cat([lm, sm])
    labels = torch.cat([torch.zeros(li.shape[0], dtype=torch.long),
                        torch.ones(si.shape[0], dtype=torch.long)])
    order = torch.randperm(ids.shape[0], generator=torch.Generator().manual_seed(seed))
    return ids[order], mask[order], labels[order]


def fit(model, data, epochs, learning_rate, batch=256, seed=42):
    ids, mask, labels = data
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(ids.shape[0], generator=generator)
        for start in range(0, ids.shape[0], batch):
            pick = order[start:start + batch]
            loss = F.cross_entropy(model(ids[pick], mask[pick]), labels[pick])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


@torch.no_grad()
def score(model, data):
    ids, mask, labels = data
    model.eval()
    predicted = model(ids, mask).argmax(-1)
    correct = (predicted == labels)
    out = {"rows": int(labels.numel()), "accuracy": float(correct.float().mean())}
    for name, value in (("language", 0), ("sentiment", 1)):
        pick = labels == value
        out[f"{name}_recall"] = float(correct[pick].float().mean())
        out[f"{name}_rows"] = int(pick.sum())
    return out


def majority(labels):
    counts = torch.bincount(labels)
    return float(counts.max()) / float(counts.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", type=Path, required=True)
    parser.add_argument("--sentiment", type=Path, required=True)
    parser.add_argument("--width", type=int, default=32, help="pipeline chunk size")
    parser.add_argument("--train-rows", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    language = json.loads(args.language.read_text())
    sentiment = json.loads(args.sentiment.read_text())
    train = build(language["train"], sentiment["train"], args.width, args.train_rows, args.seed)
    # Held out: the language validation split and the official SST-2 validation set.
    held = build(language["validation"], sentiment["audit"], args.width, None, args.seed + 1)

    report = {"width": args.width, "train_rows": int(train[2].numel()),
              "held_out_rows": int(held[2].numel()),
              "note": "Chess is excluded: it is dispatched by shape, not classified."}

    gate = fit(TaskGate(width=64), train, args.epochs, args.learning_rate, seed=args.seed)
    report["gate"] = score(gate, held)
    control = fit(LengthOnlyGate(), train, args.epochs, args.learning_rate, seed=args.seed)
    report["length_only"] = score(control, held)
    report["majority_class"] = majority(held[2])
    report["gate_margin_over_length_only"] = (report["gate"]["accuracy"]
                                              - report["length_only"]["accuracy"])

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save({"state": gate.state_dict(), "width": args.width},
                   args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
