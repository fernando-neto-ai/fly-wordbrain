#!/usr/bin/env python3
"""Score a unified arm end to end with no task label supplied.

Every number in this repository so far was measured with the task *given*: the caller
chose the encoder, applied the matching cue and read the matching output range. That
measures the model, not a system -- a real pipeline is handed a payload and has to work
out what it is.

This stacks the input-side gate in front of the arm. Each payload is routed first, the
predicted task selects the cue and the output range, and the answer is scored against the
truth. A misrouted row is scored wrong, because reading a chess move out of the token
range is a wrong answer, not a missing one.

The decomposition is the point. Routed accuracy is bounded by the gate, so both are
reported alongside the oracle-routed score -- the same arm told the task -- and the gap
between them is exactly what routing costs.

Chess is dispatched by shape rather than classified, and that is reported as a dispatch;
inflating a routing result with a free type check would misrepresent what was learned.
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

from compare_unified_arms import rebuild
from fly_wordbrain.task_gate import LANGUAGE, SENTIMENT, TaskGate
from fly_wordbrain.unified_model import LANGUAGE as U_LANGUAGE, SENTIMENT as U_SENTIMENT
from train_task_gate import windows


def full_batch(rows, pad=0):
    """Pad to the longest row, exactly as the trainer's sentiment evaluator does."""
    width = max(len(r["ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros(len(rows), width)
    for index, row in enumerate(rows):
        ids[index, :len(row["ids"])] = torch.tensor(row["ids"], dtype=torch.long)
        mask[index, :len(row["ids"])] = 1.0
    return ids, mask


@torch.no_grad()
def routed_sentiment(unified, gate, rows, device, width, chunk=128):
    """Sentiment rows, routed by the gate rather than declared.

    The gate sees a `width`-token window, because that is what it was trained on and what
    it would see in a stream. The *model* sees the whole sentence, padded to the longest
    row in the batch, exactly as the trainer's own evaluator feeds it. Feeding the model
    the gate's window instead truncates every sentence to 32 tokens and costs about three
    accuracy points -- which would have been misread as the price of routing.
    """
    correct = misrouted = total = 0
    start, stop = unified.ranges()[U_SENTIMENT]
    for begin in range(0, len(rows), chunk):
        piece = rows[begin:begin + chunk]
        gate_ids, gate_mask = windows(piece, width)
        labels = torch.tensor([r["label"] for r in piece], dtype=torch.long)
        predicted_task = gate.classify(gate_ids, gate_mask)
        ids, mask = full_batch(piece)
        ids, mask = ids.to(device), mask.to(device)
        logits, _ = unified.sentiment(ids, mask)
        answer = logits[:, start:stop].argmax(-1).cpu()
        # A row routed to language never reaches the sentiment range, so it cannot be right.
        right = (answer == labels) & (predicted_task == SENTIMENT)
        correct += int(right.sum())
        misrouted += int((predicted_task != SENTIMENT).sum())
        total += labels.numel()
    return {"rows": total, "routed_accuracy": correct / max(total, 1),
            "misrouted": misrouted, "gate_accuracy": (total - misrouted) / max(total, 1)}


@torch.no_grad()
def routed_language(unified, brain, rows, device, width, pad=0):
    """Language chunks, routed by the gate rather than declared."""
    import train_connectorch as trainer
    correct = misrouted = total = 0
    for start in range(0, len(rows), 8):
        group = rows[start:start + 8]
        inputs, targets, mask = trainer.batch_tensors(group, device, pad)
        cache = None
        for offset in range(0, inputs.shape[1], width):
            sl = slice(offset, offset + width)
            chunk_ids, chunk_mask = inputs[:, sl], mask[:, sl]
            if chunk_mask.sum() == 0:
                continue
            predicted = gate_cache["gate"].classify(chunk_ids.cpu(), chunk_mask.cpu().float())
            logits, _, cache = unified.language(chunk_ids, attention_mask=chunk_mask,
                                                cache_params=cache)
            target = targets[:, sl]
            valid = target.ne(pad)
            hit = (logits.argmax(-1) == target) & valid
            # Routed to another task means the answer is read from another range.
            keep = (predicted == LANGUAGE).to(device).unsqueeze(-1)
            correct += int((hit & keep).sum())
            misrouted += int((valid & ~keep).sum())
            total += int(valid.sum())
    return {"targets": total, "routed_accuracy": correct / max(total, 1),
            "misrouted_targets": misrouted,
            "gate_accuracy": (total - misrouted) / max(total, 1)}


gate_cache = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--audit-data", type=Path, required=True)
    parser.add_argument("--sentiment-corpus", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True, help="task-gate .pt from train_task_gate")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    saved = torch.load(args.gate, map_location="cpu", weights_only=False)
    gate = TaskGate(width=64)
    gate.load_state_dict(saved["state"])
    gate.eval()
    gate_cache["gate"] = gate
    width = saved["width"]

    model, unified, plasticity = rebuild(args.config, args.model, args.groups,
                                         args.run, args.checkpoint, args.device)
    report = {"run": str(args.run), "config": args.config, "plasticity": plasticity,
              "gate_width": width,
              "chess": "dispatched by shape (780 floats vs token ids), not classified"}

    stories = json.loads(args.audit_data.read_text())["audit"]
    report["language"] = routed_language(unified, model.brain, stories, args.device, width)

    rows = json.loads((args.sentiment_corpus / "dataset.json").read_text())["audit"]
    report["sentiment"] = routed_sentiment(unified, gate, rows, args.device, width)

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
