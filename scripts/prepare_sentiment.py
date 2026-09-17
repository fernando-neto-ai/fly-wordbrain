#!/usr/bin/env python3
"""Build the sentiment corpus as a third task for the same brain.

Sentiment is the sharpest test of whether the brain represents *what it is being asked*,
because unlike chess it arrives through the same door as language: the same tokenizer, the
same embedding, the same injection. Nothing about the input says "judge this" rather than
"continue this", so the task cue has to be something the brain conditions on rather than
something the encoder hands it.

SST-2's own test labels are hidden (every row is -1), so the official *validation* split
becomes our audit population. That is the set the literature reports on, which makes the
number comparable to published work instead of only to itself.

The trap here is the treebank. SST-2's training rows are *phrases* carved out of sentences,
while the dev rows are whole sentences, so a training phrase can be a substring of an audit
sentence without sharing its digest. Exact-match disjointness is blind to that, so the
substring relation is measured and reported rather than assumed away -- it is a known
property of the dataset and it bounds what an accuracy number here means.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_ngxson import DEFAULT_MODEL

DATASET = "stanfordnlp/sst2"
MAX_TOKENS = 320
LABELS = {0: "negative", 1: "positive"}


def normalized(text):
    return " ".join(text.casefold().split())


def digest(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def read_split(split):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    path = f"datasets/{DATASET}/data/{split}-00000-of-00001.parquet"
    with HfFileSystem().open(path, "rb") as handle:
        return pq.ParquetFile(handle).read().to_pylist()


def substring_contamination(audit, train_texts, minimum_words=4):
    """How many audit sentences contain a training phrase verbatim.

    Checked by generating each audit sentence's word n-grams and looking them up in the set
    of training phrases, rather than testing every phrase against every sentence. The naive
    form is quadratic and forces a sample; this is exact over *all* training rows and costs
    a few hundred thousand set lookups. Treebank phrases are constituents, so they align to
    word boundaries and n-gram matching is not an approximation here.

    Short phrases match everything ("of the"), so only phrases of at least `minimum_words`
    words count; anything shorter says more about English than about this dataset.
    """
    phrases = {normalized(t) for t in train_texts if len(t.split()) >= minimum_words}
    longest = max((len(p.split()) for p in phrases), default=0)
    hits, examples = 0, []
    for row in audit:
        words = normalized(row).split()
        found = None
        for size in range(minimum_words, min(len(words), longest) + 1):
            for start in range(len(words) - size + 1):
                candidate = " ".join(words[start:start + size])
                if candidate in phrases:
                    found = candidate
                    break
            if found:
                break
        if found:
            hits += 1
            if len(examples) < 5:
                examples.append(found)
    return {"audit_rows": len(audit), "rows_containing_a_training_phrase": hits,
            "fraction": hits / max(len(audit), 1),
            "minimum_phrase_words": minimum_words,
            "phrases_considered": len(phrases),
            "training_rows_checked": len(train_texts),
            "longest_phrase_words": longest,
            "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train", type=int, default=60000)
    parser.add_argument("--validation", type=int, default=2000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if (args.output / "dataset.json").exists():
        parser.error("A dataset already exists here; use a new output directory")

    import numpy as np
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    if tokenizer.get_vocab_size() != 1024:
        raise SystemExit("Tokenizer is not the pinned 1,024-token vocabulary")

    official_train = read_split("train")
    official_validation = read_split("validation")
    hidden = read_split("test")
    print(f"  official train {len(official_train):,} | validation {len(official_validation):,} "
          f"| test {len(hidden):,} (labels hidden: "
          f"{all(r['label'] == -1 for r in hidden)})", flush=True)

    def encode(row, split):
        text = row["sentence"].strip()
        ids = [1, *tokenizer.encode(text, add_special_tokens=False).ids, 2]
        return {"id": f"sst2/{split}/{row['idx']}", "text": text, "ids": ids,
                "label": int(row["label"]), "tokens": len(ids),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "normalized_sha256": digest(text)}

    audit, seen, rejected = [], set(), Counter()
    for row in official_validation:
        item = encode(row, "validation")
        if item["tokens"] > MAX_TOKENS:
            rejected["audit_too_long"] += 1
            continue
        if item["normalized_sha256"] in seen:
            rejected["audit_duplicate"] += 1
            continue
        seen.add(item["normalized_sha256"])
        audit.append(item)

    generator = np.random.default_rng(args.seed)
    order = generator.permutation(len(official_train))
    pool = []
    for index in order:
        item = encode(official_train[int(index)], "train")
        if item["tokens"] > MAX_TOKENS:
            rejected["train_too_long"] += 1
            continue
        if item["normalized_sha256"] in seen:
            rejected["overlaps_a_reserved_row"] += 1
            continue
        seen.add(item["normalized_sha256"])
        pool.append(item)
        if len(pool) >= args.train + args.validation + args.test:
            break
    wanted = args.train + args.validation + args.test
    if len(pool) < wanted:
        raise SystemExit(f"Only {len(pool):,} usable rows for a requested {wanted:,}")

    data = {"train": pool[:args.train],
            "validation": pool[args.train:args.train + args.validation],
            "test": pool[args.train + args.validation:wanted],
            "audit": audit}

    digests = {name: {r["normalized_sha256"] for r in rows} for name, rows in data.items()}
    names = list(data)
    checks = {
        "splits_are_disjoint_by_normalised_text": all(
            not (digests[a] & digests[b]) for i, a in enumerate(names) for b in names[i + 1:]),
        "every_row_within_the_token_cap": all(r["tokens"] <= MAX_TOKENS
                                              for rows in data.values() for r in rows),
        "labels_are_binary": {r["label"] for rows in data.values() for r in rows} == {0, 1},
        "audit_is_the_official_validation_split": len(audit) > 0 and all(
            r["id"].startswith("sst2/validation/") for r in audit),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        raise SystemExit("Audit failed; refusing to write this corpus")

    # Every training row, not a sample: the n-gram lookup makes the full check cheap.
    contamination = substring_contamination([r["text"] for r in audit],
                                            [r["text"] for r in data["train"]])
    print(f"  substring contamination: {contamination['rows_containing_a_training_phrase']}"
          f"/{contamination['audit_rows']} audit sentences contain a training phrase of "
          f"{contamination['minimum_phrase_words']}+ words "
          f"({contamination['fraction']:.1%}, from {contamination['phrases_considered']:,} phrases)",
          flush=True)

    balance = {name: dict(Counter(r["label"] for r in rows)) for name, rows in data.items()}
    data["provenance"] = {
        "dataset": DATASET, "labels": LABELS, "seed": args.seed,
        "max_tokens_including_bos_eos": MAX_TOKENS,
        "rows": {name: len(rows) for name, rows in data.items()},
        "label_balance": balance,
        "token_lengths": {name: {"mean": sum(r["tokens"] for r in rows) / max(len(rows), 1),
                                 "max": max((r["tokens"] for r in rows), default=0)}
                          for name, rows in data.items()},
        "rejected": dict(rejected),
        "audit_split": "the official SST-2 validation set; the official test labels are hidden",
        "disjointness_audit": checks,
        "substring_contamination": {
            **contamination,
            "note": "SST-2 trains on treebank phrases and evaluates on whole sentences, so a "
                    "training phrase can sit inside an audit sentence without sharing its "
                    "digest. This is a property of the dataset, not of this build, and it "
                    "bounds what an accuracy number here means. Checked exactly, over every "
                    "training row, by n-gram lookup.",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    (args.output / "dataset.json").write_bytes(body)
    print(json.dumps({"output": str(args.output / "dataset.json"),
                      "sha256": hashlib.sha256(body).hexdigest(),
                      "megabytes": round(len(body) / 1e6, 1),
                      "rows": data["provenance"]["rows"],
                      "label_balance": balance}, indent=2))


if __name__ == "__main__":
    main()
