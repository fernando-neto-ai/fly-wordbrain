"""Pinned TinyStories expansion that preserves every pilot split assignment.

The offline builder handles selection independently of network access. Existing
tokenization, sequence construction and vocabulary ranking come from data.py.
Published outputs are immutable; verified source-page caches are resumable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import random
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from . import data as original


SPLITS = ("train", "val", "test")
PROTECTED_OUTPUTS = ("dataset.json", "manifest.json", "recall_probes.json")


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def _pilot_revision(pilot):
    source = pilot["metadata"]["source"]
    if source.get("dataset_id") != original.DATASET_ID or not source.get("revision"):
        raise ValueError("Pilot must identify the author's TinyStories source and pinned revision")
    return source["revision"]


def _check_source(pilot, source_metadata):
    revision = _pilot_revision(pilot)
    if source_metadata.get("dataset_id") != original.DATASET_ID or source_metadata.get("revision") != revision:
        raise ValueError("Expanded source does not match the pilot's pinned dataset revision")
    for receipt in source_metadata.get("receipts", []):
        page_revision = receipt.get("x_revision")
        if page_revision is not None and page_revision != revision:
            raise ValueError("Source page receipt differs from pinned revision")
    return revision


def _record_description(record, max_words):
    if not isinstance(record.get("id"), str) or not record["id"] or not isinstance(record.get("text"), str):
        raise ValueError("Source records require a nonempty string id and string text")
    words = original.words_from_text(record["text"])
    return {"id": record["id"], "text": record["text"], "words": words[:max_words],
            "source_text_sha256": _hash(record["text"]),
            "normalized_story_sha256": _hash(" ".join(words)),
            "retained_words_sha256": _hash(" ".join(words[:max_words])),
            "original_word_count": len(words)}


def _coverage(rows, vocabulary):
    allowed = set(vocabulary)
    unknown = Counter(word for row in rows for word in row["words"] if word not in allowed)
    total = sum(len(row["words"]) for row in rows)
    count = sum(unknown.values())
    return {"stories": len(rows), "lexical_words": total, "unknown_words": count,
            "known_word_fraction": 1 - count / total if total else None,
            "unknown_word_types": len(unknown), "top_unknown_words": unknown.most_common(20),
            "truncated_stories": sum(not row["ended_naturally"] for row in rows),
            "supervised_targets": sum(sum(row["target_mask"]) for row in rows)}


def expanded_dataset_from_records(records, pilot, source_metadata, *, train_stories=8192,
                                  val_stories=1024, test_stories=1024, max_words=128,
                                  vocab_size=1024, seed=1729):
    """Pure deterministic expansion with anchor/content checks before splitting."""
    revision = _check_source(pilot, source_metadata)
    counts = dict(zip(SPLITS, (train_stories, val_stories, test_stories)))
    if any(not isinstance(n, int) or n < 1 for n in counts.values()):
        raise ValueError("All three expanded split counts must be positive integers")
    if max_words != pilot["metadata"]["max_words"] or not 1 <= max_words <= 128:
        raise ValueError("Expansion must retain the pilot word-window cap, within1..128")
    if vocab_size < 5:
        raise ValueError("Vocabulary must include four specials and lexical words")
    by_id = {}
    identical_id_duplicates = 0
    for record in records:
        row = _record_description(record, max_words)
        if row["id"] in by_id:
            if row["source_text_sha256"] != by_id[row["id"]]["source_text_sha256"]:
                raise ValueError(f"Conflicting source content for ID {row['id']}")
            identical_id_duplicates += 1
        else:
            by_id[row["id"]] = row
    anchors, seen_full, seen_prefix = {}, set(), set()
    selected_ids = {split: [] for split in SPLITS}
    for split in SPLITS:
        if counts[split] < len(pilot["splits"][split]):
            raise ValueError(f"Requested {split} count cannot retain all pilot anchors")
        for anchor in pilot["splits"][split]:
            identifier = anchor["id"]
            if identifier in anchors:
                raise ValueError("Pilot story ID occurs in more than one split")
            if identifier not in by_id:
                raise ValueError(f"Missing raw source for pilot anchor {identifier}")
            row = by_id[identifier]
            for key in ("source_text_sha256", "normalized_story_sha256", "retained_words_sha256",
                        "original_word_count", "words"):
                if row[key] != anchor[key]:
                    raise ValueError(f"Pilot anchor source/content mismatch: {identifier}: {key}")
            if row["normalized_story_sha256"] in seen_full or row["retained_words_sha256"] in seen_prefix:
                raise ValueError("Pilot anchors contain duplicate full stories or retained prefixes")
            anchors[identifier] = split
            selected_ids[split].append(identifier)
            seen_full.add(row["normalized_story_sha256"])
            seen_prefix.add(row["retained_words_sha256"])
    candidates = []
    empty_count, content_duplicates = 0, 0
    for identifier in sorted(by_id):
        if identifier in anchors:
            continue
        row = by_id[identifier]
        if not row["words"]:
            empty_count += 1
            continue
        if row["normalized_story_sha256"] in seen_full or row["retained_words_sha256"] in seen_prefix:
            content_duplicates += 1
            continue
        seen_full.add(row["normalized_story_sha256"])
        seen_prefix.add(row["retained_words_sha256"])
        candidates.append(identifier)
    needed_new = sum(counts.values()) - len(anchors)
    if len(candidates) < needed_new:
        raise ValueError(f"Need {needed_new} new unique stories after anchor-aware dedup; found {len(candidates)}")
    random.Random(seed).shuffle(candidates)
    cursor = 0
    for number, split in enumerate(SPLITS):
        take = counts[split] - len(selected_ids[split])
        selected_ids[split].extend(candidates[cursor:cursor + take])
        cursor += take
        # Keep split membership while mixing anchors through the larger split,
        # so a train[:N] calibration is not restricted to old pilot stories.
        random.Random(seed + number + 1).shuffle(selected_ids[split])

    # The audited builder supplies every record's hashes, natural EOS decision,
    # word boundaries and masks. Only its TRAIN pass supplies the final vocab.
    # Evaluation passes are schema construction only: their temporary IDs/vocab
    # are discarded and every item is encoded with the expanded training vocab.
    schemas = {}
    for split in SPLITS:
        schema = original.dataset_from_records(
            [{"id": identifier, "text": by_id[identifier]["text"]} for identifier in selected_ids[split]],
            train_stories=counts[split], val_stories=0, test_stories=0,
            max_words=max_words, vocab_size=vocab_size, seed=seed,
        )
        schemas[split] = schema
    vocabulary = schemas["train"]["vocabulary"]
    lookup = {word: i for i, word in enumerate(vocabulary)}
    splits = {}
    for split in SPLITS:
        normalized = {row["id"]: row for row in schemas[split]["splits"]["train"]}
        rows = [normalized[identifier] for identifier in selected_ids[split]]
        for row in rows:
            lexical = [lookup.get(word, original.UNK_ID) for word in row["words"]]
            row["word_ids"] = [original.BOS_ID] + lexical + ([original.EOS_ID] if row["ended_naturally"] else [])
            # Masks from audited sequence construction remain valid after
            # replacing lexical IDs; no lexical word can map to BOS or PAD.
            if len(row["target_mask"]) != len(row["word_ids"]):
                raise ValueError("Sequence schema changed during expanded-vocab encoding")
        splits[split] = rows
    metadata = dict(schemas["train"]["metadata"])
    metadata.update({
        "source": source_metadata, "pinned_revision": revision,
        "selection": "Preserve pilot split membership; lexicographic source-ID dedup; seed-shuffle new candidates; fill remaining train/val/test slots; independently shuffle each completed split",
        "split_shuffle_seeds": {split: seed + i + 1 for i, split in enumerate(SPLITS)},
        "anchor_policy": "All original pilot stories stay in original splits; order changes; no anchor content or prefix alias can cross splits",
        "pilot_anchor_counts": {split: len(pilot["splits"][split]) for split in SPLITS},
        "pilot_anchor_ids": {split: [row["id"] for row in pilot["splits"][split]] for split in SPLITS},
        "requested_split_counts": counts,
        "coverage": {split: _coverage(rows, vocabulary) for split, rows in splits.items()},
        "duplicate_source_ids_dropped": identical_id_duplicates,
        "duplicate_stories_dropped": content_duplicates, "empty_stories_dropped": empty_count,
        "unique_candidates": len(candidates) + len(anchors),
        "unique_source_ids": len(by_id),
        "vocabulary_fit_split": "train", "vocabulary_size": len(vocabulary),
        "recalibration_required": "Vocabulary is refitted on expanded training only; rebuild word codes, neural features and train-only calibration. Prior checkpoints are incompatible.",
        "limitations": "Expanded prefix sample from official TinyStories train split, with our anchored train/val/test division; not the official validation benchmark",
    })
    return {"schema_version": 1, "vocabulary": vocabulary, "splits": splits, "metadata": metadata}


def retry_delay(error, attempt, maximum=60.):
    retry_after = error.headers.get("Retry-After") if isinstance(error, HTTPError) and error.headers else None
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                date = parsedate_to_datetime(retry_after)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                seconds = 2. ** (attempt + 1)
    else:
        seconds = 2. ** (attempt + 1)
    return max(0., min(float(maximum), seconds))


def fetch_json_with_retries(url, cache_file, *, max_attempts=7, fetch=None, sleep=None):
    fetch = fetch or original._fetch_json
    sleep = sleep or time.sleep
    for attempt in range(max_attempts):
        try:
            return fetch(url, cache_file)
        except HTTPError as error:
            if error.code != 429 and not 500 <= error.code <= 599:
                raise
            last_error = error
        except (URLError, TimeoutError, ConnectionError, socket.timeout) as error:
            last_error = error
        if attempt + 1 == max_attempts:
            raise RuntimeError(f"Source fetch exhausted {max_attempts} attempts: {url}") from last_error
        delay = retry_delay(last_error, attempt)
        print(json.dumps({"event": "source_retry", "attempt": attempt + 1,
                          "delay_seconds": delay, "error": str(last_error), "url": url}), flush=True)
        sleep(delay)


def fetch_pinned_stories(output_dir, pilot, needed, max_words):
    """Resume verified 100-row pages; fail if any page has a different revision."""
    output_dir = Path(output_dir)
    cache = output_dir / "source"
    pinned = _pilot_revision(pilot)
    info, info_receipt = fetch_json_with_retries(
        f"https://huggingface.co/api/datasets/{original.DATASET_ID}", cache / "dataset_info.json")
    if info.get("sha") != pinned:
        raise ValueError("Dataset repository revision changed from pilot; refusing expansion")
    receipts, records = [info_receipt], []
    full_seen, prefix_seen = set(), set()
    offset, max_rows = 0, max(1000, needed * 10)
    while len(prefix_seen) < needed and offset < max_rows:
        length = min(100, max_rows - offset)
        url = "https://datasets-server.huggingface.co/rows?" + urlencode({
            "dataset": original.DATASET_ID, "config": "default", "split": "train",
            "offset": offset, "length": length,
        })
        page, receipt = fetch_json_with_retries(url, cache / f"train_rows_{offset:07d}_{length}.json")
        if receipt.get("x_revision") != pinned:
            raise ValueError("Dataset viewer page revision differs from pinned pilot source")
        rows = page.get("rows", [])
        if not rows:
            break
        receipts.append(receipt)
        for row in rows:
            text = row["row"]["text"]
            words = original.words_from_text(text)
            if words:
                full, prefix = " ".join(words), " ".join(words[:max_words])
                if full not in full_seen and prefix not in prefix_seen:
                    full_seen.add(full)
                    prefix_seen.add(prefix)
            records.append({"id": f"tinystories/train/{row['row_idx']}", "text": text})
        offset += len(rows)
        if offset % 1000 == 0 or len(prefix_seen) >= needed:
            print(json.dumps({"event": "source_progress", "rows_read": offset,
                              "unique_retained_stories": len(prefix_seen), "required": needed}), flush=True)
    if len(prefix_seen) < needed:
        raise ValueError(f"Insufficient unique stories within bounded source prefix: {len(prefix_seen)}/{needed}")
    return records, {"dataset_id": original.DATASET_ID, "dataset_url": original.DATASET_URL,
                     "revision": pinned, "license": "cdla-sharing-1.0", "source_split": "train",
                     "selection": "bounded source prefix with verified cached rows API pages",
                     "source_rows_read": offset, "receipts": receipts}


def build_expanded_dataset(output_dir, pilot_path, *, train_stories=8192, val_stories=1024,
                           test_stories=1024, max_words=128, vocab_size=1024, seed=1729):
    output, pilot_path = Path(output_dir), Path(pilot_path)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in PROTECTED_OUTPUTS):
        raise FileExistsError("Expanded outputs are immutable; choose a new output directory")
    lock = output / "build.lock"
    try:
        handle = lock.open("x")
    except FileExistsError as error:
        raise RuntimeError("An expanded-data builder is already active or left build.lock; inspect before resuming") from error
    try:
        with handle:
            handle.write("Expanded TinyStories builder active\n")
        pilot_bytes = pilot_path.read_bytes()
        pilot = json.loads(pilot_bytes)
        records, source = fetch_pinned_stories(output, pilot, train_stories + val_stories + test_stories, max_words)
        dataset = expanded_dataset_from_records(records, pilot, source,
            train_stories=train_stories, val_stories=val_stories, test_stories=test_stories,
            max_words=max_words, vocab_size=vocab_size, seed=seed)
        pilot_sha = hashlib.sha256(pilot_bytes).hexdigest()
        dataset["metadata"]["pilot_dataset_sha256"] = pilot_sha
        encoded = _json_bytes(dataset)
        manifest = {"schema_version": 1, "dataset_sha256": hashlib.sha256(encoded).hexdigest(),
                    "pilot_dataset_sha256": pilot_sha, "source_dataset": original.DATASET_ID,
                    "source_revision": _pilot_revision(pilot), "split_counts": dataset["metadata"]["requested_split_counts"],
                    "builder_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "base_data_source_sha256": hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
                    "source_receipts_sha256": hashlib.sha256(_json_bytes(source["receipts"])).hexdigest()}
        # Stage complete bytes before publishing either immutable artifact.
        stage = output / ".building"
        stage.mkdir(exist_ok=True)
        (stage / "dataset.json").write_bytes(encoded)
        (stage / "manifest.json").write_bytes(_json_bytes(manifest))
        if any((output / name).exists() for name in PROTECTED_OUTPUTS):
            raise FileExistsError("Expanded output appeared while building; refusing overwrite")
        (stage / "dataset.json").replace(output / "dataset.json")
        (stage / "manifest.json").replace(output / "manifest.json")
        stage.rmdir()
        return dataset
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-stories", type=int, default=8192)
    parser.add_argument("--val-stories", type=int, default=1024)
    parser.add_argument("--test-stories", type=int, default=1024)
    parser.add_argument("--max-words", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    dataset = build_expanded_dataset(args.output, args.pilot, train_stories=args.train_stories,
        val_stories=args.val_stories, test_stories=args.test_stories, max_words=args.max_words,
        vocab_size=args.vocab_size, seed=args.seed)
    print(json.dumps({"output": str(args.output), "vocabulary_size": len(dataset["vocabulary"]),
                      "coverage": dataset["metadata"]["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
