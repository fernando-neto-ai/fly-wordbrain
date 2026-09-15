#!/usr/bin/env python3
"""Prepare a fresh, frozen audit set without consuming the reserved test split."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly_wordbrain.data import DATASET_ID, _fetch_json
from prepare_ngxson import DEFAULT_MODEL, FILES, REVISION, verify


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalized_digest(text):
    return digest_text(" ".join(text.casefold().split()))


def exclusions_from_existing(data):
    ids, exact, normalized = set(), set(), set()
    counts = {}
    for split in ("train", "validation", "test"):
        rows = data.get(split)
        if not rows:
            raise ValueError("Existing exclusion split is empty: " + split)
        counts[split] = len(rows)
        for row in rows:
            text = row["text"]
            raw_hash, normal_hash = digest_text(text), normalized_digest(text)
            if row.get("text_sha256") != raw_hash or row.get("normalized_sha256") != normal_hash:
                raise ValueError("Existing story text hashes do not match: " + row["id"])
            if row["id"] in ids:
                raise ValueError("Duplicate existing story ID: " + row["id"])
            ids.add(row["id"])
            exact.add(raw_hash)
            normalized.add(normal_hash)
    return ids, exact, normalized, counts


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(path)
    return hashlib.sha256(body).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=ROOT / "data/ngxson-tinystories-v1/dataset.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ngxson-quality-v1")
    parser.add_argument("--stories", type=int, default=200)
    parser.add_argument("--offset", type=int, default=1000)
    parser.add_argument("--max-source-rows", type=int, default=10000)
    args = parser.parse_args()
    if args.stories < 1 or args.offset < 1000 or args.max_source_rows < args.stories:
        parser.error("Require a positive story count, offset >= 1000, and enough bounded source rows")
    destination = args.output / "dataset.json"
    if destination.exists() or (args.output / "manifest.json").exists():
        parser.error("Audit data already exists; use a fresh output directory")

    existing_body = args.existing.read_bytes()
    existing_sha = hashlib.sha256(existing_body).hexdigest()
    original_manifest = json.loads(args.existing.with_name("manifest.json").read_text())
    if existing_sha != original_manifest["dataset_sha256"]:
        raise ValueError("Existing dataset SHA does not match its manifest")
    existing = json.loads(existing_body)
    revision = existing["provenance"]["revision"]
    if existing["provenance"]["dataset"] != DATASET_ID:
        raise ValueError("Unexpected source dataset")
    blocked_ids, blocked_raw, blocked_normal, blocked_counts = exclusions_from_existing(existing)
    if blocked_counts != original_manifest["splits"]:
        raise ValueError("Exclusion population differs from existing split manifest")
    tokenizer_receipt = verify(args.model / "tokenizer.json", FILES["tokenizer.json"])
    if tokenizer_receipt["sha256"] != existing["provenance"]["tokenizer"]["sha256"]:
        raise ValueError("Audit tokenizer differs from existing training tokenizer")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    if tokenizer.get_vocab_size() != 1024 or [tokenizer.token_to_id(x) for x in ("<pad>", "<s>", "</s>")] != [0, 1, 2]:
        raise ValueError("Unexpected tokenizer vocabulary or special tokens")

    selected, decisions, receipts = [], [], []
    seen_ids, seen_normal = set(), set()
    stats = {"attempted": 0, "selected": 0, "excluded_existing_id": 0,
             "excluded_existing_exact_text": 0, "excluded_existing_normalized_text": 0,
             "excluded_audit_duplicate": 0, "excluded_empty": 0, "too_long": 0}
    for offset in range(args.offset, args.offset + args.max_source_rows, 100):
        page_length = min(100, args.offset + args.max_source_rows - offset)
        url = "https://datasets-server.huggingface.co/rows?" + urlencode({
            "dataset": DATASET_ID, "config": "default", "split": "validation",
            "offset": offset, "length": page_length,
        })
        page, receipt = _fetch_json(url, args.output / "source" / f"validation_{offset:07d}.json")
        if receipt.get("x_revision") != revision:
            raise ValueError("Audit page revision differs from original training data: " + repr(receipt.get("x_revision")))
        receipts.append(receipt)
        if not page.get("rows"):
            break
        for source in page["rows"]:
            if len(selected) == args.stories:
                break
            source_index = source["row_idx"]
            if not isinstance(source_index, int) or not offset <= source_index < offset + page_length:
                raise ValueError("Source page returned an out-of-range row index")
            row_id = f"tinystories/validation/{source_index}"
            text = source["row"]["text"]
            raw_hash, normal_hash = digest_text(text), normalized_digest(text)
            decision = {"id": row_id, "text_sha256": raw_hash, "normalized_sha256": normal_hash}
            stats["attempted"] += 1
            if row_id in blocked_ids:
                reason = "excluded_existing_id"
            elif raw_hash in blocked_raw:
                reason = "excluded_existing_exact_text"
            elif normal_hash in blocked_normal:
                reason = "excluded_existing_normalized_text"
            elif row_id in seen_ids or normal_hash in seen_normal:
                reason = "excluded_audit_duplicate"
            elif not text.strip():
                reason = "excluded_empty"
            else:
                ids = [1, *tokenizer.encode(text, add_special_tokens=False).ids, 2]
                decision["tokens_including_bos_eos"] = len(ids)
                if len(ids) > 320:
                    reason = "too_long"
                else:
                    if len(ids) < 2 or any(token <= 0 or token >= 1024 for token in ids):
                        raise ValueError("Invalid audit token ID")
                    reason = "selected"
                    selected.append({**decision, "text": text, "ids": ids})
                    seen_ids.add(row_id)
                    seen_normal.add(normal_hash)
            decision["decision"] = reason
            decisions.append(decision)
            stats[reason] += 1
        if len(selected) == args.stories:
            break
    if len(selected) != args.stories:
        raise ValueError(f"Only {len(selected)} qualifying audit stories within source bound")
    audit_ids = {row["id"] for row in selected}
    audit_raw = {row["text_sha256"] for row in selected}
    audit_normal = {row["normalized_sha256"] for row in selected}
    intersections = {"ids": len(audit_ids & blocked_ids),
                     "exact_text_hashes": len(audit_raw & blocked_raw),
                     "normalized_text_hashes": len(audit_normal & blocked_normal)}
    if any(intersections.values()) or len(audit_ids) != args.stories or len(audit_normal) != args.stories:
        raise ValueError("Audit holdout overlaps existing data or contains duplicate stories")
    if stats["attempted"] != sum(value for key, value in stats.items() if key != "attempted"):
        raise ValueError("Audit selection accounting does not reconcile")
    predicted_tokens = sum(len(row["ids"]) - 1 for row in selected)
    if predicted_tokens <= 0:
        raise ValueError("Audit holdout has no scored targets")
    exclusion_sha = write_json(args.output / "exclusion-population.json", {
        "existing_dataset_sha256": existing_sha, "split_counts": blocked_counts,
        "ids": sorted(blocked_ids), "text_sha256": sorted(blocked_raw),
        "normalized_sha256": sorted(blocked_normal),
    })
    decisions_sha = write_json(args.output / "selection-decisions.json", decisions)
    provenance = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID, "revision": revision, "receipts": receipts,
        "model_revision": REVISION, "tokenizer": tokenizer_receipt,
        "existing_dataset_sha256": existing_sha, "excluded_split_counts": blocked_counts,
        "exclusion_population_sha256": exclusion_sha, "selection_decisions_sha256": decisions_sha,
        "selection": f"First {args.stories} qualifying official validation stories at source offset {args.offset} onward; no shuffling or truncation",
        "source_offset": args.offset, "max_source_rows": args.max_source_rows,
        "max_tokens_including_bos_eos": 320, "source_stats": stats,
        "overlap_counts": intersections, "token_counts": {"audit": predicted_tokens},
        "deduplication": "All existing train/validation/test IDs plus recomputed exact and whitespace-normalized casefolded full-text SHA256; audit stories also unique by ID and normalized full text",
        "purpose": "Fresh fixed checkpoint-quality audit; existing reserved test split remains unevaluated",
        "limitations": "No semantic deduplication. Author's exact training and tokenizer-fitting texts are unpublished, so overlap with the released checkpoint's original corpus remains unknown.",
        "preparation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    dataset_sha = write_json(destination, {"audit": selected, "provenance": provenance})
    manifest = {"dataset_sha256": dataset_sha, "splits": {"audit": len(selected)},
                "predicted_tokens": {"audit": predicted_tokens}, "source_stats": stats,
                "existing_dataset_sha256": existing_sha, "overlap_counts": intersections,
                "exclusion_population_sha256": exclusion_sha, "selection_decisions_sha256": decisions_sha}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"output": str(destination), **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
