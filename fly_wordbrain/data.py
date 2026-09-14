"""Small, auditable word-level data for a frozen-connectome pilot.

No ML/data-library dependency is required. ``word_ids`` includes BOS and, only
for a complete story, EOS. ``words`` contains actual lexical words only. Train
with ``word_ids[:-1]`` predicting ``word_ids[1:]``, masking ``target_mask[1:]``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET_ID = "roneneldan/TinyStories"
DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)
# Letters only, with internal apostrophes. Digits and punctuation are excluded;
# hyphenated forms count as separate words. This is a word-level experiment.
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
RECALL_NAMES = ("lily", "tim", "tom", "sam")


def words_from_text(text: str) -> list[str]:
    return WORD_RE.findall(text.replace("\u2019", "'").replace("\u2018", "'").lower())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(_canonical_json(value))
    tmp.replace(path)


def _fetch_json(url: str, cache_file: Path) -> tuple[dict, dict]:
    receipt_file = cache_file.with_suffix(".receipt.json")
    if cache_file.exists() and receipt_file.exists():
        body = cache_file.read_bytes()
        receipt = json.loads(receipt_file.read_text())
        if receipt["url"] != url or receipt["sha256"] != _sha256(body):
            raise ValueError(f"Source cache verification failed: {cache_file}")
    else:
        request = Request(url, headers={"User-Agent": "fly-wordbrain/0.1"})
        with urlopen(request, timeout=60) as response:
            # Dataset viewer pages are deliberately bounded to <=100 rows.
            body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                raise ValueError("Unexpectedly large dataset API response")
            receipt = {
                "url": url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": _sha256(body),
                "bytes": len(body),
                "x_revision": response.headers.get("x-revision"),
                "etag": response.headers.get("etag"),
            }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(body)
        _write_json(receipt_file, receipt)
    return json.loads(body), receipt


def fetch_stories(output_dir: Path, needed: int, max_words: int) -> tuple[list[dict], dict]:
    """Read a bounded prefix of the author's dataset using HF's rows API.

    Both full normalized stories and truncated word sequences are deduplicated
    before splitting. Source receipts preserve the exact returned bytes.
    """
    cache = output_dir / "source"
    info, info_receipt = _fetch_json(
        f"https://huggingface.co/api/datasets/{DATASET_ID}", cache / "dataset_info.json"
    )
    revision = info["sha"]
    records, receipts = [], [info_receipt]
    unique_full, unique_prefix = set(), set()
    offset = 0
    # Prevent accidental unbounded downloading in the face of bad data.
    max_rows = max(1000, needed * 10)
    while len(unique_prefix) < needed and offset < max_rows:
        length = min(100, max_rows - offset)
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(
            {"dataset": DATASET_ID, "config": "default", "split": "train",
             "offset": offset, "length": length}
        )
        page, receipt = _fetch_json(url, cache / f"train_rows_{offset:07d}_{length}.json")
        if receipt.get("x_revision") != revision:
            raise ValueError("Dataset viewer revision does not match source repository receipt")
        receipts.append(receipt)
        page_rows = page.get("rows", [])
        if not page_rows:
            break
        for row in page_rows:
            text = row["row"]["text"]
            words = words_from_text(text)
            if not words:
                continue
            full = " ".join(words)
            prefix = " ".join(words[:max_words])
            if full not in unique_full and prefix not in unique_prefix:
                unique_full.add(full)
                unique_prefix.add(prefix)
            records.append({"id": f"tinystories/train/{row['row_idx']}", "text": text})
        offset += len(page_rows)
    if len(unique_prefix) < needed:
        raise ValueError(f"Needed {needed} unique stories; found {len(unique_prefix)} within {offset} rows")
    return records, {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "revision": revision,
        "license": "cdla-sharing-1.0",
        "source_split": "train",
        "selection": "bounded prefix of author dataset, then deterministic story shuffle",
        "source_rows_read": offset,
        "receipts": receipts,
    }


def dataset_from_records(
    records: Iterable[dict], *, train_stories: int = 128, val_stories: int = 32,
    test_stories: int = 32, max_words: int = 128, vocab_size: int = 1024,
    seed: int = 1729, source_metadata: dict | None = None,
) -> dict:
    """Pure construction function, also used by offline unit tests."""
    counts = {"train": train_stories, "val": val_stories, "test": test_stories}
    if train_stories < 1 or val_stories < 0 or test_stories < 0:
        raise ValueError("Train count must be positive and evaluation counts nonnegative")
    if not 1 <= max_words <= 128:
        raise ValueError("max_words must be within 1..128 for this pilot")
    if vocab_size < len(SPECIAL_TOKENS) + 1:
        raise ValueError("vocab_size must include specials and at least one lexical word")

    candidates, seen_full, seen_prefix, seen_ids = [], set(), set(), set()
    duplicate_count, empty_count = 0, 0
    for i, record in enumerate(records):
        text = record["text"]
        full_words = words_from_text(text)
        if not full_words:
            empty_count += 1
            continue
        full_digest = _sha256(" ".join(full_words).encode())
        retained_words = full_words[:max_words]
        prefix_digest = _sha256(" ".join(retained_words).encode())
        if full_digest in seen_full or prefix_digest in seen_prefix:
            duplicate_count += 1
            continue
        record_id = str(record.get("id", f"record/{i}"))
        if record_id in seen_ids:
            raise ValueError(f"Duplicate story id with different content: {record_id}")
        seen_ids.add(record_id)
        seen_full.add(full_digest)
        seen_prefix.add(prefix_digest)
        candidates.append({
            "id": record_id, "words": retained_words,
            "source_text_sha256": _sha256(text.encode()),
            "normalized_story_sha256": full_digest,
            "retained_words_sha256": prefix_digest,
            "original_word_count": len(full_words),
            "ended_naturally": len(full_words) <= max_words,
        })
    needed = sum(counts.values())
    if len(candidates) < needed:
        raise ValueError(f"Need {needed} unique stories after deduplication; got {len(candidates)}")
    random.Random(seed).shuffle(candidates)
    splits, offset = {}, 0
    for split, count in counts.items():
        splits[split] = candidates[offset:offset + count]
        offset += count
    word_counts = Counter(word for record in splits["train"] for word in record["words"])
    ranked = sorted(word_counts, key=lambda word: (-word_counts[word], word))
    vocabulary = list(SPECIAL_TOKENS) + ranked[:vocab_size - len(SPECIAL_TOKENS)]
    word_to_id = {word: i for i, word in enumerate(vocabulary)}
    coverage = {}
    for split, items in splits.items():
        unknown_words = Counter()
        total = 0
        for record in items:
            lexical_ids = [word_to_id.get(word, UNK_ID) for word in record["words"]]
            ids = [BOS_ID] + lexical_ids + ([EOS_ID] if record["ended_naturally"] else [])
            record["word_ids"] = ids
            record["target_mask"] = [token_id not in (PAD_ID, BOS_ID) for token_id in ids]
            unknown_words.update(w for w in record["words"] if w not in word_to_id)
            total += len(record["words"])
        unknown_total = sum(unknown_words.values())
        coverage[split] = {
            "stories": len(items), "lexical_words": total, "unknown_words": unknown_total,
            "known_word_fraction": 1 - unknown_total / total if total else None,
            "unknown_word_types": len(unknown_words),
            "top_unknown_words": unknown_words.most_common(20),
            "truncated_stories": sum(not r["ended_naturally"] for r in items),
            "supervised_targets": sum(sum(r["target_mask"]) for r in items),
        }
    return {
        "schema_version": 1, "vocabulary": vocabulary, "splits": splits,
        "metadata": {
            "seed": seed, "max_words": max_words, "requested_vocab_size": vocab_size,
            "vocabulary_size": len(vocabulary), "vocabulary_fit_split": "train",
            "word_definition": "lowercase [a-z]+ with internal apostrophes; punctuation and digits excluded; hyphens separate words",
            "special_token_ids": {token: i for i, token in enumerate(SPECIAL_TOKENS)},
            "sequence_contract": "word_ids = BOS + words + EOS only for untruncated complete stories; predict ids[1:] from ids[:-1], using target_mask[1:]",
            "context_contract": "Reset neural state per story; at most max_words actual lexical words, excluding BOS/EOS; no state crosses story boundaries",
            "loss_contract": "Never score BOS/PAD; UNK and genuine EOS are scored; report OOV coverage and lexical-only metrics separately",
            "deduplication": "normalized full lexical text AND retained lexical prefix before story-level split",
            "duplicate_stories_dropped": duplicate_count, "empty_stories_dropped": empty_count,
            "unique_candidates": len(candidates), "coverage": coverage,
            "source": source_metadata or {"description": "caller-supplied records"},
            "limitations": "Small prefix-sampled pilot; shuffled story split is not the official TinyStories validation split and does not establish broad language competence",
        },
    }


def build_recall_probes(vocabulary: list[str], seed: int = 1729) -> dict:
    """Counterfactual four-name probes, separate from natural text training.

    Context length counts all lexical words preceding the answer. The name is
    always word index 1 (zero-based), so C-2 words intervene before the answer.
    A group contains the exact same prompt with each of four names substituted.
    Templates are held out by split, making this a stringent OOD memory probe.
    """
    vocab = {word: i for i, word in enumerate(vocabulary)}
    configs = {
        "train": ("name", "now say the first name", 4,
                  "the soft rain falls near a small tree while birds play outside"),
        "val": ("meet", "now give the first name", 2,
                "a bright sun shines over the green field while children walk slowly"),
        "test": ("person", "now tell the first name", 2,
                 "the cool wind moves through a quiet garden while little animals rest"),
    }
    markers = ["today", "outside", "quietly", "again", "nearby", "slowly", "happily", "later"]
    rng = random.Random(seed)
    rng.shuffle(markers)
    splits = {}
    marker_idx = 0
    for split, (prefix, suffix, groups, filler_text) in configs.items():
        items = []
        filler_base, suffix_words = words_from_text(filler_text), words_from_text(suffix)
        for group in range(groups):
            marker = markers[marker_idx]
            marker_idx += 1
            rotated = filler_base[group:] + filler_base[:group]
            for context_words in (8, 16, 32, 64, 128):
                filler_length = context_words - 2 - len(suffix_words)
                filler = [marker] + [rotated[i % len(rotated)] for i in range(filler_length - 1)]
                pair_id = f"recall/{split}/template-{split}/variant-{group}/context-{context_words}"
                for label_id, name in enumerate(RECALL_NAMES):
                    words = [prefix, name] + filler + suffix_words
                    assert len(words) == context_words and words.count(name) == 1
                    items.append({
                        "id": f"{pair_id}/{name}", "pair_group": pair_id,
                        "template_id": f"template-{split}", "words": words,
                        "word_ids": [BOS_ID] + [vocab.get(w, UNK_ID) for w in words],
                        "context_words": context_words, "cue_word_index": 1,
                        "intervening_words": context_words - 2,
                        "target_word": name, "target_id": vocab.get(name, UNK_ID),
                        "target_class": label_id,
                    })
        splits[split] = items
    missing_labels = [name for name in RECALL_NAMES if name not in vocab]
    return {
        "schema_version": 1, "labels": list(RECALL_NAMES), "splits": splits,
        "metadata": {
            "seed": seed, "purpose": "supplemental controlled name-memory probe; never mixed into natural text corpus",
            "chance_accuracy": 0.25, "balanced_by": ["split", "context_words", "pair_group"],
            "pairing": "All four names occupy identical cue position in otherwise identical prompts",
            "split_contract": "Distinct cue/query templates and filler sequences by split; shared labels; 4/2/2 paired filler variants",
            "context_contract": "Exactly 8/16/32/64/128 lexical words precede answer; cue at zero-based index1; BOS excluded; C-2 intervening words",
            "target_leakage_check": "Each target appears once, at the cue; no labels occur in the five-word query suffix or filler",
            "missing_label_words_from_text_vocab": missing_labels,
            "usable_with_text_word_encoding": not missing_labels,
            "interpretation": "If any label maps to UNK, this probe is invalid with that word encoder. Failure on held-out query templates does not isolate forgetting from query comprehension.",
            "provenance": "Locally generated deterministic synthetic diagnostic, not TinyStories",
        },
    }


def build_dataset(
    output_dir: str | Path, train_stories: int = 128, val_stories: int = 32,
    test_stories: int = 32, max_words: int = 128, vocab_size: int = 1024,
    seed: int = 1729,
) -> dict:
    """Download the small source subset and write dataset, probes and receipts."""
    output_dir = Path(output_dir)
    needed = train_stories + val_stories + test_stories
    if needed < 1 or not 1 <= max_words <= 128:
        raise ValueError("Need positive total stories and max_words within1..128")
    records, provenance = fetch_stories(output_dir, needed, max_words)
    dataset = dataset_from_records(
        records, train_stories=train_stories, val_stories=val_stories,
        test_stories=test_stories, max_words=max_words, vocab_size=vocab_size,
        seed=seed, source_metadata=provenance,
    )
    dataset_path = output_dir / "dataset.json"
    _write_json(dataset_path, dataset)
    probes_path = output_dir / "recall_probes.json"
    _write_json(probes_path, build_recall_probes(dataset["vocabulary"], seed))
    _write_json(output_dir / "manifest.json", {
        "dataset_sha256": _sha256(dataset_path.read_bytes()),
        "recall_probes_sha256": _sha256(probes_path.read_bytes()),
        "source_revision": provenance["revision"],
        "source_dataset": DATASET_ID,
    })
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-stories", type=int, default=128)
    parser.add_argument("--val-stories", type=int, default=32)
    parser.add_argument("--test-stories", type=int, default=32)
    parser.add_argument("--max-words", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    dataset = build_dataset(**vars(args))
    print(json.dumps(dataset["metadata"]["coverage"], indent=2))


if __name__ == "__main__":
    main()
