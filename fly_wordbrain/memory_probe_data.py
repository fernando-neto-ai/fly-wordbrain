"""Controlled history probes, not examples of natural-language prediction.

Each nuisance group contains every balanced label and one fixed final bigram.
The identity task varies observed word 3; the order task swaps words 2 and 3
using one fixed pair across all groups. The eighth word is an inert constant:
probe labels live separately and must never be substituted into model inputs.

All source contexts and lexical frequency counts come from the ORIGINAL train
split. Probe validation/test hold out source stories from probe fitting, not
from the slow brain's earlier training. Call ``collate_feedback`` with
``exclude_own_story=True`` for ALL probe splits. Window identities deliberately
repeat within a group to retain this whole-source-story exclusion; use row
``id`` for extraction receipts, never ``windows_identity`` on these variants.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .feedback_data import Window, _lexical_ids, make_windows


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class MemoryProbeDataset:
    rows: List[dict]
    windows: List[Window]
    metadata: dict

    def to_dict(self) -> dict:
        """JSON-safe artifact; reconstruct aligned Windows with windows_from_rows."""
        return {"schema": 1, "kind": "controlled_history_probe",
                "metadata": self.metadata, "rows": self.rows}


def windows_from_rows(rows: Sequence[Mapping[str, Any]]) -> List[Window]:
    """Reconstruct source identities; synthetic label/row IDs are not inputs."""
    return [Window(row["source_story_id"], row["source_start"],
                   tuple(row["word_ids8"])) for row in rows]


def build_memory_probes(dataset: Mapping[str, Any], *, train_groups: int = 32,
                        val_groups: int = 8, test_groups: int = 8,
                        seed: int = 1729,
                        dataset_sha256: Optional[str] = None) -> MemoryProbeDataset:
    """Build 576 default windows without inspecting original validation/test.

    Ten most frequent known lexical IDs define identity labels (frequency ties
    use lower ID); the first two define order labels 0=(a,b), 1=(b,a). Nuisance
    positions contain none of those ten IDs, so the varied identity/pair is not
    repeated near the readout. One untouched eight-word source span supplies
    the nuisance words for each group. Every group uses a distinct source story
    across both tasks, and duplicate nuisance tuples within a task are skipped.
    """
    counts = {"train": train_groups, "val": val_groups, "test": test_groups}
    if any(type(value) is not int or value <= 0 for value in counts.values()):
        raise ValueError("Require positive integer nuisance-group counts")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    vocabulary = dataset["vocabulary"]
    if (not isinstance(vocabulary, list) or len(vocabulary) < 14
            or any(not isinstance(word, str) for word in vocabulary)):
        raise ValueError("Require a vocabulary with at least ten known lexical IDs")
    if dataset_sha256 is not None and (not isinstance(dataset_sha256, str)
            or len(dataset_sha256) != 64
            or any(c not in "0123456789abcdef" for c in dataset_sha256)):
        raise ValueError("dataset_sha256 must be a lowercase SHA256 hex digest")

    # Deliberately do not enumerate or serialize the enclosing split mapping.
    training = list(dataset["splits"]["train"])
    seen_ids, frequencies, source_sequences = set(), Counter(), []
    for story in training:
        story_id = str(story["id"])
        if story_id in seen_ids:
            raise ValueError("Duplicate training source story ID: " + story_id)
        seen_ids.add(story_id)
        lexical = _lexical_ids(story)
        if lexical and max(lexical) >= len(vocabulary):
            raise ValueError("Training lexical ID exceeds vocabulary")
        frequencies.update(value for value in lexical if value >= 4)
        source_sequences.append({"id": story_id, "word_ids": list(lexical),
                                 "words": list(story["words"])})
    label_ids = sorted(frequencies, key=lambda value: (-frequencies[value], value))[:10]
    if len(label_ids) != 10:
        raise ValueError("Require ten known lexical IDs occurring in training")
    label_set = set(label_ids)
    dummy_id = label_ids[-1]
    rng = random.Random(seed)
    ordered_stories = sorted(training, key=lambda story: str(story["id"]))
    rng.shuffle(ordered_stories)
    used_sources, rows = set(), []
    source_groups: Dict[str, Dict[str, List[str]]] = {}

    for task in ("identity", "order"):
        # Omit only the manipulated positions when checking nuisance leakage.
        nuisance_positions = (0, 1, 3, 4, 5, 6) if task == "identity" else (0, 3, 4, 5, 6)
        seen_nuisances = set()
        source_groups[task] = {}
        available = iter(ordered_stories)
        for split, group_count in counts.items():
            source_groups[task][split] = []
            for group_index in range(group_count):
                selected = None
                for story in available:
                    source_id = str(story["id"])
                    if source_id in used_sources:
                        continue
                    spans = make_windows([story])
                    rng.shuffle(spans)
                    for span in spans:
                        nuisance = tuple(span.word_ids8[position] for position in nuisance_positions)
                        if (any(value < 4 or value in label_set for value in nuisance)
                                or nuisance in seen_nuisances):
                            continue
                        selected = (story, span, nuisance)
                        break
                    if selected is not None:
                        break
                if selected is None:
                    raise ValueError("Insufficient distinct training stories/nuisance contexts for "
                                     + task + "/" + split + "; reduce group counts")
                story, span, nuisance = selected
                used_sources.add(span.story_id)
                seen_nuisances.add(nuisance)
                source_groups[task][split].append(span.story_id)
                group_id = "{}/{}/{:03d}".format(task, split, group_index)
                source_hash = _digest({"word_ids": list(_lexical_ids(story)),
                                       "words": list(story["words"])})
                for label in range(10 if task == "identity" else 2):
                    words = list(span.word_ids8)
                    if task == "identity":
                        words[2] = label_ids[label]
                    else:
                        words[1:3] = label_ids[:2] if label == 0 else label_ids[:2][::-1]
                    words[7] = dummy_id
                    rows.append({"id": group_id + "/" + str(label), "task": task,
                                 "split": split, "label": label, "group_id": group_id,
                                 "source_story_id": span.story_id, "source_start": span.start,
                                 "source_story_sha256": source_hash,
                                 "source_word_ids8": list(span.word_ids8),
                                 "word_ids8": words, "final_previous_id": words[5],
                                 "final_current_id": words[6]})

    metadata = {
        "schema": 1, "seed": seed, "dataset_sha256": dataset_sha256,
        "original_source_split": "train",
        "source_training_sha256": _digest(sorted(source_sequences, key=lambda row: row["id"])),
        "vocabulary_sha256": _digest(vocabulary),
        "window_words": 8, "observed_words": 7, "dummy_eighth_id": dummy_id,
        "probe_labels_separate_from_eighth_word": True,
        "exclude_own_story": True,
        "split_unit": "distinct original-training source story and complete nuisance group",
        "heldout_scope": "Held out from probe fitting only; original slow-brain training contexts",
        "interpretation": "Controlled neural-state identity/order decodability; not natural-language accuracy",
        "selection_rule": "Known training lexical frequency descending, token ID ascending for ties",
        "identity": {"classes": 10, "chance_accuracy": .1, "word_position_1based": 3,
                     "intervening_words_before_final_bigram": 2,
                     "label_word_ids": label_ids, "label_words": [vocabulary[i] for i in label_ids],
                     "training_frequencies": [frequencies[i] for i in label_ids]},
        "order": {"classes": 2, "chance_accuracy": .5, "positions_1based": [2, 3],
                  "fixed_pair_across_all_groups": True, "pair_word_ids": label_ids[:2],
                  "label_orders": [label_ids[:2], label_ids[:2][::-1]]},
        "nuisance_excludes_all_identity_words": True,
        "groups_per_task_split": counts,
        "rows_per_task_split": {task: {split: value * size for split, value in counts.items()}
                                for task, size in (("identity", 10), ("order", 2))},
        "source_story_ids": source_groups,
        "group_count": 2 * sum(counts.values()), "window_count": len(rows),
        "rows_sha256": _digest(rows),
    }
    return MemoryProbeDataset(rows, windows_from_rows(rows), metadata)
