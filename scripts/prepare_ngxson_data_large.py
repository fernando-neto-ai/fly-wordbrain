#!/usr/bin/env python3
"""Build a larger training corpus while holding every evaluation set fixed.

The 1,000-story recipe leaves the 50.6M-parameter readout enough room to memorise
its training set, which is what the rank comparison exposed. To test whether the
low-rank advantage is a small-data regularisation artifact or something that
survives more data, the training corpus has to grow while nothing else moves.

So this changes exactly one thing. Validation and test rows are copied **verbatim**
from the existing dataset rather than reselected; the tokenizer, the 320-token cap
and the duplicate filter are the pinned ones. Training stories come from the
official *train* split, while validation, test and the separate audit population
all come from the official *validation* split, so growth cannot reach them.

Rows come from the dataset's parquet shards rather than the row API, which
rate-limits sustained paging. The two must agree for story IDs to stay comparable,
so that is verified rather than assumed: every cached row-API page found on disk is
checked against the parquet by index and by exact text.

Disjointness is likewise asserted — this refuses to write a corpus whose training
rows share an ID or a normalised text digest with any reserved set.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fly_wordbrain.data import DATASET_ID
from prepare_ngxson import DEFAULT_MODEL, REVISION

MAX_TOKENS = 320
# The default config's train split is the concatenation of these shards in order.
SHARDS = ["data/train-00000-of-00004-2d5a1467fff1081b.parquet",
          "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
          "data/train-00002-of-00004-a26307300439e943.parquet",
          "data/train-00003-of-00004-d243063613e5a057.parquet"]


def normalized_digest(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def verify_against_cached_pages(texts, cache_dir):
    """Prove the parquet ordering matches the row API the 1,000-story dataset used."""
    checked = 0
    for page_file in sorted(cache_dir.glob("train_*.json")):
        if page_file.name.endswith(".receipt.json"):
            continue
        page = json.loads(page_file.read_text())
        for row in page.get("rows", []):
            index = row["row_idx"]
            if index >= len(texts):
                continue
            if texts[index] != row["row"]["text"]:
                raise SystemExit(
                    f"Parquet row {index} differs from cached row-API page {page_file.name}. "
                    "Story IDs would not be comparable with the existing dataset.")
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=ROOT / "data/ngxson-tinystories-v1/dataset.json",
                        help="Dataset whose validation and test rows are reused unchanged")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/ngxson-quality-v1/dataset.json",
                        help="Audit population the training set must not touch")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stories", type=int, default=10_000)
    parser.add_argument("--page-cache", type=Path,
                        help="Cached row-API pages used to verify the parquet ordering")
    args = parser.parse_args()
    if (args.output / "dataset.json").exists():
        parser.error("A dataset already exists here; use a new output directory")

    from tokenizers import Tokenizer
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    if tokenizer.get_vocab_size() != 1024:
        raise SystemExit("Tokenizer is not the pinned 1,024-token vocabulary")

    existing = json.loads(args.existing.read_text())
    audit = json.loads(args.audit.read_text())["audit"] if args.audit.exists() else []
    blocked_ids, blocked_text = set(), set()
    for name, rows in (("validation", existing["validation"]), ("test", existing["test"]),
                       ("audit", audit)):
        for row in rows:
            blocked_ids.add(row["id"])
            blocked_text.add(row["normalized_sha256"])
        print(f"  reserved {name}: {len(rows)} stories", flush=True)

    texts, shard_info = [], []
    for shard in SHARDS:
        path = hf_hub_download(DATASET_ID, shard, repo_type="dataset")
        table = pq.read_table(path, columns=["text"])
        texts.extend(table.column("text").to_pylist())
        shard_info.append({"file": shard, "rows": table.num_rows,
                           "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
        print(f"  {shard.split('/')[-1]}: {table.num_rows:,} rows "
              f"(running total {len(texts):,})", flush=True)
        if len(texts) > args.stories * 4:
            break

    cache_dir = args.page_cache or (ROOT / "data/ngxson-tinystories-10k/source")
    verified_rows = verify_against_cached_pages(texts, cache_dir) if cache_dir.is_dir() else 0
    print(f"  parquet matches the row API on {verified_rows:,} independently cached rows", flush=True)

    selected, seen = [], set()
    examined = discarded_long = discarded_duplicate = blocked_hits = 0
    for index, text in enumerate(texts):
        if len(selected) >= args.stories:
            break
        examined += 1
        digest = normalized_digest(text)
        if not text.strip() or digest in seen:
            discarded_duplicate += 1
            continue
        story_id = f"tinystories/train/{index}"
        # Different official splits, so this should never fire. Count it anyway.
        if story_id in blocked_ids or digest in blocked_text:
            blocked_hits += 1
            continue
        ids = [1, *tokenizer.encode(text, add_special_tokens=False).ids, 2]
        if len(ids) > MAX_TOKENS:
            discarded_long += 1
            continue
        seen.add(digest)
        selected.append({"id": story_id, "text": text, "ids": ids,
                         "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "normalized_sha256": digest})
    if len(selected) != args.stories:
        raise SystemExit(f"Only {len(selected)} qualifying training stories available")

    data = {"train": selected, "validation": existing["validation"], "test": existing["test"]}
    train_ids = {r["id"] for r in selected}
    train_text = {r["normalized_sha256"] for r in selected}
    checks = {
        "train_ids_unique": len(train_ids) == len(selected),
        "train_texts_unique": len(train_text) == len(selected),
        "no_id_overlap_with_reserved": not (train_ids & blocked_ids),
        "no_text_overlap_with_reserved": not (train_text & blocked_text),
        "validation_unchanged": existing["validation"] == data["validation"],
        "test_unchanged": existing["test"] == data["test"],
        "every_story_within_token_cap": all(len(r["ids"]) <= MAX_TOKENS for r in selected),
        "parquet_matches_row_api": verified_rows > 0,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        raise SystemExit("Audit failed; refusing to write this corpus")

    data["provenance"] = {
        "dataset": DATASET_ID, "shards": shard_info, "model_revision": REVISION,
        "max_tokens_including_bos_eos": MAX_TOKENS,
        "token_counts": {k: sum(len(r["ids"]) - 1 for r in rows)
                         for k, rows in data.items() if k != "provenance"},
        "selection": f"First {args.stories} qualifying official training stories by parquet row "
                     f"order; validation and test copied verbatim from {args.existing.name}",
        "reused_from": str(args.existing),
        "reused_sha256": hashlib.sha256(args.existing.read_bytes()).hexdigest(),
        "source_stats": {"examined": examined, "selected": len(selected),
                         "discarded_long": discarded_long,
                         "discarded_duplicate_or_empty": discarded_duplicate,
                         "blocked_by_reserved_sets": blocked_hits,
                         "rows_verified_against_row_api": verified_rows},
        "disjointness_audit": checks,
        "reserved_populations": {"validation": len(existing["validation"]),
                                 "test": len(existing["test"]), "audit": len(audit)},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Only the training corpus differs from the 1,000-story dataset. Validation, test "
                "and the audit population are byte-identical, so scores are directly comparable.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    (args.output / "dataset.json").write_bytes(body)
    print(json.dumps({"output": str(args.output / "dataset.json"),
                      "sha256": hashlib.sha256(body).hexdigest(),
                      "megabytes": round(len(body) / 1e6, 1),
                      "stories": {k: len(v) for k, v in data.items() if k != "provenance"},
                      "next_token_targets": data["provenance"]["token_counts"]}, indent=2))


if __name__ == "__main__":
    main()
