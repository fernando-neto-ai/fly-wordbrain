"""Fresh controlled retention episodes and balanced two-task fitting batches.

This filters contexts only. Build ActionCandidates from ALL original training
stories, then collate every retention split with exclude_own_story=True. The
new probe's validation/test groups are held out from retention fitting, not
from the original language-model training. The separate probe label must never
replace the inert eighth word or otherwise enter the circuit input.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

from .feedback_data import _lexical_ids, make_windows
from .memory_probe_data import MemoryProbeDataset, _digest, build_memory_probes, windows_from_rows


NUISANCE_POSITIONS = {"identity": (0, 1, 3, 4, 5, 6), "order": (0, 3, 4, 5, 6)}


def _nuisance(task, word_ids8):
    return tuple(word_ids8[position] for position in NUISANCE_POSITIONS[task])


def _previous_contract(previous_rows, sources):
    """Recover and validate the actual old class mapping, not an assumed one."""
    if not previous_rows:
        raise ValueError("Require the previous probe rows to exclude its contexts")
    windows_from_rows(previous_rows)  # Existing audited eight-word/source-start validation.
    ids, grouped = set(), defaultdict(list)
    identity_mapping, orders, dummy_ids = {}, {}, set()
    nuisance = {task: set() for task in NUISANCE_POSITIONS}
    for row in previous_rows:
        if row["id"] in ids:
            raise ValueError("Duplicate previous probe row ID")
        ids.add(row["id"])
        task, label = row["task"], row["label"]
        if task not in NUISANCE_POSITIONS or type(label) is not int or not 0 <= label < (10 if task == "identity" else 2):
            raise ValueError("Malformed previous probe task/label")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Malformed previous probe split")
        source_id = str(row["source_story_id"])
        if source_id not in sources:
            raise ValueError("Previous probe source is absent from original TRAIN: " + source_id)
        source = sources[source_id]
        lexical = _lexical_ids(source)
        start = row["source_start"]
        if list(lexical[start:start + 8]) != row["source_word_ids8"]:
            raise ValueError("Previous source window differs from original TRAIN")
        source_hash = _digest({"word_ids": list(lexical), "words": list(source["words"])})
        if row["source_story_sha256"] != source_hash:
            raise ValueError("Previous source story hash differs from original TRAIN")
        words = row["word_ids8"]
        if any(words[p] != row["source_word_ids8"][p] for p in NUISANCE_POSITIONS[task]):
            raise ValueError("Previous nuisance differs from its source window")
        if (row["final_previous_id"], row["final_current_id"]) != tuple(words[5:7]):
            raise ValueError("Previous final bigram metadata mismatch")
        mapping = identity_mapping if task == "identity" else orders
        value = words[2] if task == "identity" else tuple(words[1:3])
        if label in mapping and mapping[label] != value:
            raise ValueError("Inconsistent previous class-to-word mapping")
        mapping[label] = value
        dummy_ids.add(words[7])
        nuisance[task].add(_nuisance(task, words))
        grouped[row["group_id"]].append(row)
    if set(identity_mapping) != set(range(10)) or set(orders) != {0, 1}:
        raise ValueError("Previous probe must contain all ten identity and both order labels")
    label_ids = [identity_mapping[label] for label in range(10)]
    if (len(set(label_ids)) != 10 or any(value < 4 for value in label_ids)
            or orders != {0: tuple(label_ids[:2]), 1: tuple(label_ids[:2][::-1])}
            or dummy_ids != {label_ids[-1]}):
        raise ValueError("Previous identity/order/dummy-word contract mismatch")
    for rows in grouped.values():
        task = rows[0]["task"]
        if (sorted(row["label"] for row in rows) != list(range(10 if task == "identity" else 2))
                or len({(row["task"], row["split"], str(row["source_story_id"]),
                         row["source_start"], _nuisance(row["task"], row["word_ids8"])) for row in rows}) != 1):
            raise ValueError("Previous nuisance groups must be complete, paired, and unsplit")
    return label_ids, nuisance


def build_retention_probes(dataset: Mapping[str, Any], previous_rows: Sequence[Mapping[str, Any]], *,
                           train_groups: int = 64, val_groups: int = 16, test_groups: int = 32,
                           seed: int = 2718, dataset_sha256: Optional[str] = None) -> MemoryProbeDataset:
    """Exclude previous sources/full nuisances, then reuse the audited builder.

    A whole extra source story is excluded if any nonoverlapping source span
    reproduces a previous task-specific full nuisance tuple. Final bigrams may
    be shared: every complete group balances all labels at that fixed bigram.
    Frequency ranking on the filtered pool must reproduce the old ten labels
    exactly, or construction fails rather than silently changing the task.
    """
    # No original validation/test indexing or split enumeration.
    original_training = list(dataset["splits"]["train"])
    sources = {}
    for story in original_training:
        source_id = str(story["id"])
        if source_id in sources:
            raise ValueError("Duplicate original training story ID: " + source_id)
        sources[source_id] = story
    previous_label_ids, previous_nuisance = _previous_contract(previous_rows, sources)
    previous_sources = {str(row["source_story_id"]) for row in previous_rows}
    excluded_nuisance_sources, filtered = set(), []
    for story in original_training:
        source_id = str(story["id"])
        if source_id in previous_sources:
            continue
        if any(_nuisance(task, window.word_ids8) in previous_nuisance[task]
               for window in make_windows([story]) for task in NUISANCE_POSITIONS):
            excluded_nuisance_sources.add(source_id)
        else:
            filtered.append(story)
    filtered_dataset = {"vocabulary": dataset["vocabulary"], "splits": {"train": filtered}}
    result = build_memory_probes(filtered_dataset, train_groups=train_groups, val_groups=val_groups,
                                 test_groups=test_groups, seed=seed, dataset_sha256=dataset_sha256)
    if result.metadata["identity"]["label_word_ids"] != previous_label_ids:
        raise ValueError("Filtered training frequencies changed the previous ten-class mapping")
    selected_sources = {row["source_story_id"] for row in result.rows}
    if selected_sources & previous_sources or any(
            _nuisance(row["task"], row["word_ids8"]) in previous_nuisance[row["task"]]
            for row in result.rows):
        raise AssertionError("Retention contexts overlap the previous probe")
    previous_bigrams = {tuple(row["word_ids8"][5:7]) for row in previous_rows}
    selected_bigrams = {tuple(row["word_ids8"][5:7]) for row in result.rows}
    result.metadata.update({
        "experiment": "fresh_controlled_retention_training",
        "heldout_scope": "Held out from retention fitting and previous probe; original slow-brain training contexts",
        "context_source_training_sha256": result.metadata["source_training_sha256"],
        "context_pool_filtered_only": True,
        "proposal_source": "ALL original TRAIN, excluding each entire original source story in every probe split",
        "original_training_story_ids_sha256": _digest(sorted(sources)),
        "original_training_story_count": len(original_training),
        "filtered_context_story_count": len(filtered),
        "previous_probe_rows_sha256": _digest(list(previous_rows)),
        "previous_probe_window_count": len(previous_rows),
        "excluded_previous_source_story_ids": sorted(previous_sources),
        "excluded_previous_source_story_count": len(previous_sources),
        "excluded_repeated_nuisance_source_story_ids": sorted(excluded_nuisance_sources),
        "excluded_previous_nuisance_counts": {task: len(values) for task, values in previous_nuisance.items()},
        "previous_label_word_ids": previous_label_ids,
        "previous_label_mapping_identical": True,
        "previous_source_overlap_count": 0,
        "previous_full_nuisance_overlap_count": 0,
        "final_bigram_overlap_with_previous": len(selected_bigrams & previous_bigrams),
        "final_bigram_policy": "May repeat across groups and splits; every group is label balanced",
        "training_batch_recipe": {"identity_rows": 8, "order_rows": 8,
            "shuffle": "Seeded per-task/per-epoch permutations, recycled as needed",
            "length": "ceil(max(task training row counts)/8) batches; all batches contain 16 rows",
            "validation_and_test_oversampled": False},
    })
    return result


def balanced_training_batches(rows: Sequence[Mapping[str, Any]], *, epoch: int = 0,
                               seed: int = 2718, per_task: int = 8):
    """Return aligned row indices, with per_task identity + per_task order.

    Each task is independently shuffled and recycled with a fresh permutation.
    One epoch covers the larger task once (padding its final batch if needed).
    For the default corpus this yields 80 batches: all 640 identity rows once,
    all 128 order rows five times. No validation/test row can enter the batches.
    """
    if type(epoch) is not int or epoch < 0 or type(seed) is not int or type(per_task) is not int or per_task <= 0:
        raise ValueError("Require nonnegative integer epoch, integer seed, and positive per_task")
    pools = {task: [] for task in NUISANCE_POSITIONS}
    seen = set()
    for index, row in enumerate(rows):
        if row["id"] in seen:
            raise ValueError("Duplicate retention row ID")
        seen.add(row["id"])
        if row["task"] not in pools or row["split"] not in ("train", "val", "test"):
            raise ValueError("Malformed retention task/split")
        if row["split"] == "train":
            pools[row["task"]].append(index)
    if any(not indices for indices in pools.values()):
        raise ValueError("Require training rows for both identity and order")
    steps = math.ceil(max(map(len, pools.values())) / per_task)
    drawn = {}
    for task, indices in pools.items():
        rng = random.Random(int(_digest([seed, epoch, task]), 16))
        values = []
        while len(values) < steps * per_task:
            cycle = indices.copy()
            rng.shuffle(cycle)
            values.extend(cycle)
        drawn[task] = values[:steps * per_task]
    return [drawn["identity"][start:start + per_task] + drawn["order"][start:start + per_task]
            for start in range(0, steps * per_task, per_task)]
