#!/usr/bin/env python3
"""Build the toxicity corpus as a fourth task, to test whether one brain can be foundational.

Three tasks established that the multi-task cost does not compound: the second cost 0.4388
nats and the third 0.0690. Two points bend a curve but cannot locate its knee, so this adds
a fourth. Toxicity is the right one to add because it is *maximally confusable with an
existing task*: like sentiment it is a binary judgement over short English prose, arriving
through the same tokenizer and the same injection. If the brain can hold both without them
collapsing into each other, the claim that it is a general substrate gets much stronger.

Source is `OxAISH-AL-LLM/wiki_toxic` (CC0), Wikipedia talk-page comments from the Jigsaw
task. Two properties of it decide the design.

**The training file is balanced and the evaluation files are not.** `balanced_train` is
exactly 50/50, while `validation` is 89.69% non-toxic and `test` 90.24%. Scoring raw
accuracy against a 90% floor would make a genuinely useful detector read as a failure --
the same trap that a wrong 72.48% floor set for sentiment. So the audit is carved
*balanced* from the official validation split, giving a ~50% floor directly comparable to
sentiment's 50.92%, and the natural-distribution rates are recorded beside it so the
imbalanced picture is never lost.

**Comments are long.** Mean 60.8 words against SST-2's 25, with a 1,250-word tail. The
brain settles the whole sequence for a classification task, so length is compute: the cap
is deliberately tighter than sentiment's and the token statistics are reported, because
this is the term that decides whether a four-task run is affordable.

Contamination is checked two ways. Within the corpus, exact digests across splits. Across
corpora, against the sentiment training text -- the two tasks share a tokenizer and a
register, and a gate that has to tell them apart is only meaningful if they are actually
different text.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_ngxson import DEFAULT_MODEL

DATASET = "OxAISH-AL-LLM/wiki_toxic"
LABELS = {0: "non-toxic", 1: "toxic"}


def normalized(text):
    return " ".join(text.casefold().split())


def digest(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def read_csv(name):
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f"datasets/{DATASET}/{name}", "rb") as handle:
        return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))


def balanced(rows, per_class, generator):
    """Take an equal number of each label, so the floor is one half rather than the base rate."""
    buckets = {}
    for row in rows:
        buckets.setdefault(row["label"], []).append(row)
    take = min(per_class, min(len(v) for v in buckets.values()))
    out = []
    for label in sorted(buckets):
        order = generator.permutation(len(buckets[label]))[:take]
        out.extend(buckets[label][int(i)] for i in order)
    generator.shuffle(out)
    return out, take


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=192,
                        help="tighter than sentiment's 320: the brain settles the whole row")
    parser.add_argument("--train", type=int, default=24000)
    parser.add_argument("--validation-per-class", type=int, default=1000)
    parser.add_argument("--audit-per-class", type=int, default=1000)
    parser.add_argument("--sentiment-corpus", type=Path,
                        help="checked for cross-corpus overlap; both are token streams")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if (args.output / "dataset.json").exists():
        parser.error("A dataset already exists here; use a new output directory")

    import numpy as np
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    if tokenizer.get_vocab_size() != 1024:
        raise SystemExit("Tokenizer is not the pinned 1,024-token vocabulary")
    generator = np.random.default_rng(args.seed)

    raw_train = read_csv("balanced_train.csv")
    raw_validation = read_csv("validation.csv")
    natural = Counter(r["label"] for r in raw_validation)
    natural_total = sum(natural.values())
    print(f"  balanced_train {len(raw_train):,} | validation {len(raw_validation):,} "
          f"(natural majority {max(natural.values())/natural_total*100:.2f}%)", flush=True)

    def encode(row, split):
        text = row["comment_text"].strip()
        ids = [1, *tokenizer.encode(text, add_special_tokens=False).ids, 2]
        return {"id": f"wikitoxic/{split}/{row['id']}", "text": text, "ids": ids,
                "label": int(row["label"]), "tokens": len(ids),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "normalized_sha256": digest(text)}

    rejected, seen = Counter(), set()

    def take(rows, split, limit=None):
        out = []
        for row in rows:
            if limit and len(out) >= limit:
                break
            if not row["comment_text"].strip():
                rejected[f"{split}_empty"] += 1
                continue
            item = encode(row, split)
            if item["tokens"] > args.max_tokens:
                rejected[f"{split}_too_long"] += 1
                continue
            if item["normalized_sha256"] in seen:
                rejected[f"{split}_duplicate"] += 1
                continue
            seen.add(item["normalized_sha256"])
            out.append(item)
        return out

    # Held-out first, so a row can never be taken for training and then re-used for audit.
    audit_rows, audit_per = balanced(raw_validation, args.audit_per_class, generator)
    audit = take(audit_rows, "audit")
    remaining = [r for r in raw_validation if digest(r["comment_text"]) not in seen]
    validation_rows, validation_per = balanced(remaining, args.validation_per_class, generator)
    validation = take(validation_rows, "validation")
    train = take(raw_train, "train", args.train)

    splits = {"train": train, "validation": validation, "audit": audit}
    for name, rows in splits.items():
        if not rows:
            raise SystemExit(f"Split {name} came out empty")

    digests = {name: {r["normalized_sha256"] for r in rows} for name, rows in splits.items()}
    overlaps = {f"{a}_vs_{b}": len(digests[a] & digests[b])
                for a in splits for b in splits if a < b}

    cross = None
    if args.sentiment_corpus:
        other = json.loads((args.sentiment_corpus / "dataset.json").read_text())
        other_digests = {r["normalized_sha256"] for split in ("train", "validation", "audit")
                         for r in other.get(split, [])}
        cross = {"sentiment_rows_checked": len(other_digests),
                 "shared_rows": len(other_digests & set().union(*digests.values()))}

    provenance = {
        "dataset": DATASET, "license": "cc0-1.0", "labels": {str(k): v for k, v in LABELS.items()},
        "seed": args.seed, "max_tokens_including_bos_eos": args.max_tokens,
        "rows": {k: len(v) for k, v in splits.items()},
        "label_balance": {k: dict(Counter(str(r["label"]) for r in v)) for k, v in splits.items()},
        "token_lengths": {k: {"mean": sum(r["tokens"] for r in v) / len(v),
                              "max": max(r["tokens"] for r in v)} for k, v in splits.items()},
        "rejected": dict(rejected),
        "balanced_carve": {"audit_per_class": audit_per, "validation_per_class": validation_per,
                           "why": "the official validation split is 89.69% non-toxic; a balanced "
                                  "audit gives a ~50% floor comparable to sentiment's 50.92%"},
        "natural_distribution": {"validation_rows": natural_total,
                                 "counts": dict(natural),
                                 "majority_class": max(natural.values()) / natural_total},
        "disjointness_audit": {"splits_are_disjoint_by_normalised_text": all(v == 0 for v in overlaps.values()),
                               "overlaps": overlaps,
                               "every_row_within_the_token_cap": True,
                               "labels_are_binary": all(r["label"] in LABELS for v in splits.values() for r in v)},
        "cross_corpus_overlap_with_sentiment": cross,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset.json").write_text(
        json.dumps({**splits, "provenance": provenance}) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
