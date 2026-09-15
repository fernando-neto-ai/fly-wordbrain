"""Probe group splits and strict last-bigram controls, with no neural work."""
from collections import Counter, defaultdict
import copy
import json

import pytest
import torch

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.feedback_data import collate_feedback
from fly_wordbrain.memory_probe_data import build_memory_probes, windows_from_rows


def source(identifier, ids):
    return {"id": identifier, "words": ["word" + str(i) for i in ids],
            "word_ids": [2] + ids + [3], "target_mask": [False] + [True] * (len(ids) + 1)}


def dataset(n=110):
    # Frequent task words are 4..13; each nuisance story has unique contexts.
    train = [source("frequency-{:02d}".format(i), list(range(4, 14)) * 12) for i in range(11)]
    for i in range(n):
        train.append(source("source-{:03d}".format(i), [14 + i, 150, 151, 152, 153, 154, 155, 156]))
    return {"vocabulary": ["word" + str(i) for i in range(160)],
            "splits": {"train": train, "val": [], "test": []}}


def small(data=None, **kwargs):
    return build_memory_probes(data or dataset(), train_groups=2, val_groups=1,
                               test_groups=1, **kwargs)


def test_default_counts_group_balance_and_source_splits():
    data = dataset()
    built = build_memory_probes(data)
    assert len(built.rows) == len(built.windows) == 576
    assert built.metadata["group_count"] == 96
    by_group, split_sources = defaultdict(list), defaultdict(set)
    for row, window in zip(built.rows, built.windows):
        by_group[row["group_id"]].append(row)
        split_sources[row["split"]].add(row["source_story_id"])
        assert window.story_id == row["source_story_id"]
        assert window.start == row["source_start"]
        assert list(window.word_ids8) == row["word_ids8"]
    assert len({row["id"] for row in built.rows}) == 576
    assert len({row["source_story_id"] for row in built.rows}) == 96
    for split, expected in (("train", 64), ("val", 16), ("test", 16)):
        assert len(split_sources[split]) == expected
    assert not split_sources["train"] & split_sources["val"]
    assert not split_sources["train"] & split_sources["test"]
    assert not split_sources["val"] & split_sources["test"]
    for rows in by_group.values():
        assert sorted(row["label"] for row in rows) == list(range(10 if rows[0]["task"] == "identity" else 2))
        assert len({row["split"] for row in rows}) == 1
    assert built.metadata["rows_per_task_split"] == {
        "identity": {"train": 320, "val": 80, "test": 80},
        "order": {"train": 64, "val": 16, "test": 16}}


def test_exact_identity_order_and_all_final_inputs_constant_with_source_exclusion():
    data = dataset()
    built = small(data)
    proposals = ActionCandidates(data["splits"]["train"], len(data["vocabulary"]), top_k=10)
    by_group = defaultdict(list)
    for row in built.rows:
        by_group[row["group_id"]].append(row)
    label_ids = built.metadata["identity"]["label_word_ids"]
    for rows in by_group.values():
        windows = windows_from_rows(rows)
        batch = collate_feedback(windows, proposals, exclude_own_story=True)
        for field in ("previous", "current", "candidates", "probabilities"):
            final = getattr(batch, field)[:, -1]
            torch.testing.assert_close(final, final[0].expand_as(final), rtol=0, atol=0)
        assert len(set(batch.targets.tolist())) == 1
        for row in rows:
            assert row["word_ids8"][5:7] == row["source_word_ids8"][5:7]
            if row["task"] == "identity":
                assert row["word_ids8"][2] == label_ids[row["label"]]
                positions = (0, 1, 3, 4, 5, 6)
            else:
                assert Counter(rows[0]["word_ids8"][:7]) == Counter(row["word_ids8"][:7])
                assert row["word_ids8"][1:3] == label_ids[:2][::1 if row["label"] == 0 else -1]
                positions = (0, 3, 4, 5, 6)
            assert all(row["word_ids8"][p] == row["source_word_ids8"][p] for p in positions)
            assert all(row["word_ids8"][p] not in label_ids for p in positions)
        # With the changed words outside the two-word proposal context, all
        # final three proposal steps agree; any final neural distinction is history.
        for field in ("previous", "current", "candidates", "probabilities"):
            tail = getattr(batch, field)[:, 5:]
            torch.testing.assert_close(tail, tail[:1].expand_as(tail), rtol=0, atol=0)


def test_only_original_training_is_read_and_frequency_ranking_is_train_only():
    class TrainOnly(dict):
        def __getitem__(self, key):
            assert key == "train", "Original validation/test access forbidden"
            return super().__getitem__(key)
        def __iter__(self):
            raise AssertionError("Do not enumerate held-out splits")
    data = dataset()
    data["splits"] = TrainOnly(data["splits"])
    built = small(data)
    assert built.metadata["identity"]["label_word_ids"] == list(range(4, 14))
    assert built.metadata["identity"]["training_frequencies"] == [132] * 10
    assert built.metadata["original_source_split"] == "train"
    assert "not natural-language accuracy" in built.metadata["interpretation"]
    assert "probe fitting only" in built.metadata["heldout_scope"]


def test_reproducible_serializable_order_independent_and_labels_never_targets():
    data = dataset()
    original = copy.deepcopy(data)
    first = small(data, dataset_sha256="a" * 64)
    reversed_data = {**data, "splits": {**data["splits"], "train": data["splits"]["train"][::-1]}}
    second = small(reversed_data, dataset_sha256="a" * 64)
    assert first.to_dict() == second.to_dict()
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()
    assert windows_from_rows(first.rows) == first.windows
    assert data == original
    changed = small(data, seed=42)
    assert changed.metadata["rows_sha256"] != first.metadata["rows_sha256"]
    assert {window.word_ids8[-1] for window in first.windows} == {13}
    assert all(window.word_ids8[-1] != row["label"] for row, window in zip(first.rows, first.windows))


def test_invalid_sources_and_insufficient_independent_groups_fail_clearly():
    data = dataset(3)
    with pytest.raises(ValueError, match="Insufficient"):
        small(data)
    data = dataset()
    data["splits"]["train"].append(data["splits"]["train"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        small(data)
    with pytest.raises(ValueError, match="positive integer"):
        build_memory_probes(dataset(), train_groups=0)
    with pytest.raises(ValueError, match="SHA256"):
        small(dataset_sha256="not-a-hash")


def test_repeated_nuisance_cannot_appear_in_heldout_groups():
    data = dataset()
    for row in data["splits"]["train"][1:]:
        row.update(source(row["id"], [14, 150, 151, 152, 153, 154, 155, 156]))
    with pytest.raises(ValueError, match="Insufficient"):
        small(data)
