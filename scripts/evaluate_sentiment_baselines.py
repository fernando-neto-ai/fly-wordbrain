#!/usr/bin/env python3
"""The floor a fly has to clear on sentiment before the task is worth GPU time.

Majority class is not that floor -- it is 51% and any model beats it. The floor that
matters is **bag of tokens**: a logistic regression over the same pinned 1,024-entry
tokenizer, which has no word order, no composition and no recurrence at all. If the brain
cannot beat a model that only counts which tokens appeared, then nothing about the fly is
being measured and the task should not be run.

Reported on the official SST-2 validation split, which is what the literature reports on,
so the numbers sit on a scale with published work rather than only with each other.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def features(rows, vocabulary, binary=True):
    matrix = np.zeros((len(rows), vocabulary), dtype=np.float32)
    for index, row in enumerate(rows):
        ids = np.asarray(row["ids"][1:-1], dtype=np.int64)     # drop BOS and EOS
        if ids.size == 0:
            continue
        if binary:
            matrix[index, np.unique(ids)] = 1.0
        else:
            np.add.at(matrix[index], ids, 1.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-6, None)


def fit_logistic(train_x, train_y, epochs=800, rate=0.2, weight_decay=1e-4, seed=1729):
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], 2)
    with torch.no_grad():
        model.weight.zero_(); model.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(train_x), train_y).backward()
        optimizer.step()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", default="audit")
    parser.add_argument("--vocabulary", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads((args.corpus / "dataset.json").read_text())
    train, held = data["train"], data[args.split]
    train_y = torch.tensor([r["label"] for r in train])
    held_y = torch.tensor([r["label"] for r in held])

    report = {"corpus": str(args.corpus), "split": args.split,
              "train_rows": len(train), "held_out_rows": len(held)}

    majority = int(train_y.bincount().argmax())
    report["majority_class"] = {"label": majority,
                                "accuracy": float((held_y == majority).float().mean())}

    for name, binary in (("bag_of_tokens_binary", True), ("bag_of_tokens_counts", False)):
        train_x = torch.from_numpy(features(train, args.vocabulary, binary))
        held_x = torch.from_numpy(features(held, args.vocabulary, binary))
        model = fit_logistic(train_x, train_y)
        with torch.no_grad():
            accuracy = float((model(held_x).argmax(-1) == held_y).float().mean())
            train_accuracy = float((model(train_x).argmax(-1) == train_y).float().mean())
        report[name] = {"held_out_accuracy": accuracy, "train_accuracy": train_accuracy}
        print(f"  {name:22s} held-out {accuracy:.4f}  (train {train_accuracy:.4f})", flush=True)

    best = max(report[k]["held_out_accuracy"] for k in
               ("bag_of_tokens_binary", "bag_of_tokens_counts"))
    report["floor_to_beat"] = best
    report["note"] = ("A bag of tokens has no word order, no composition and no recurrence. "
                      "A recurrent model that does not clear this is not being measured on "
                      "anything the recurrence provides.")
    print(f"\n  majority class         {report['majority_class']['accuracy']:.4f}")
    print(f"  FLOOR TO BEAT          {best:.4f}")
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
