"""Fresh-source exclusion, unchanged tasks, and balanced training indices."""
from collections import Counter, defaultdict
import copy
import json

import pytest
import torch

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.feedback_data import collate_feedback
from fly_wordbrain.memory_probe_data import build_memory_probes, windows_from_rows
from fly_wordbrain.retention_data import build_retention_probes, balanced_training_batches


def source(identifier, ids):
    return {"id": identifier, "words": ["word" + str(i) for i in ids],
            "word_ids": [2] + ids + [3], "target_mask": [False] + [True] * (len(ids) + 1)}


def dataset(n=360):
    train = [source("frequency-{:02d}".format(i), list(range(4, 14)) * 12) for i in range(40)]
    train += [source("source-{:03d}".format(i), [14 + i, 500, 501, 502, 503, 504, 505, 506])
              for i in range(n)]
    return {"vocabulary": ["word" + str(i) for i in range(600)],
            "splits": {"train": train, "val": [], "test": []}}


def previous(data):
    return build_memory_probes(data)


def test_default_fresh_sources_counts_mapping_and_complete_split_groups():
    data = dataset()
    old = previous(data)
    new = build_retention_probes(data, old.rows)
    assert len(new.rows) == len(new.windows) == 1344
    assert new.metadata["rows_per_task_split"] == {
        "identity": {"train": 640, "val": 160, "test": 320},
        "order": {"train": 128, "val": 32, "test": 64}}
    assert new.metadata["excluded_previous_source_story_count"] == 96
    old_sources = {r["source_story_id"] for r in old.rows}
    new_sources = {r["source_story_id"] for r in new.rows}
    assert len(new_sources) == 224 and not old_sources & new_sources
    split_sources = {split: {r["source_story_id"] for r in new.rows if r["split"] == split}
                     for split in ("train", "val", "test")}
    assert [len(split_sources[s]) for s in ("train", "val", "test")] == [128, 32, 64]
    assert not split_sources["train"] & split_sources["val"]
    assert not split_sources["train"] & split_sources["test"]
    assert not split_sources["val"] & split_sources["test"]
    assert new.metadata["identity"]["label_word_ids"] == old.metadata["identity"]["label_word_ids"]
    assert new.metadata["order"] == old.metadata["order"]
    assert new.metadata["dummy_eighth_id"] == old.metadata["dummy_eighth_id"]
    assert new.metadata["previous_full_nuisance_overlap_count"] == 0
    assert "ALL original TRAIN" in new.metadata["proposal_source"]
    assert json.loads(json.dumps(new.to_dict())) == new.to_dict()


def test_duplicate_nuisance_under_new_story_id_is_excluded_without_filtering_proposals():
    data = dataset()
    old = previous(data)
    original = next(s for s in data["splits"]["train"] if s["id"] == old.rows[0]["source_story_id"])
    duplicate = copy.deepcopy(original)
    duplicate["id"] = "new-id-old-context"
    data["splits"]["train"].append(duplicate)
    new = build_retention_probes(data, old.rows)
    assert duplicate["id"] in new.metadata["excluded_repeated_nuisance_source_story_ids"]
    assert duplicate["id"] not in {row["source_story_id"] for row in new.rows}
    all_proposals = ActionCandidates(data["splits"]["train"], len(data["vocabulary"]), top_k=10)
    # Collation keeps original source IDs, so full-story exclusion works even
    # though context selection was filtered. The proposal table stays complete.
    assert set(all_proposals.training_story_ids) == {s["id"] for s in data["splits"]["train"]}
    groups = defaultdict(list)
    for row in new.rows:
        groups[row["group_id"]].append(row)
    for rows in list(groups.values())[:2]:
        batch = collate_feedback(windows_from_rows(rows), all_proposals, exclude_own_story=True)
        for field in ("previous", "current", "candidates", "probabilities"):
            x = getattr(batch, field)[:, -1]
            torch.testing.assert_close(x, x[:1].expand_as(x), rtol=0, atol=0)
        assert set(batch.targets.tolist()) == {new.metadata["dummy_eighth_id"]}


def test_never_accesses_original_validation_test_and_is_deterministic_without_mutation():
    data = dataset()
    old = previous(data)
    saved_data, saved_rows = copy.deepcopy(data), copy.deepcopy(old.rows)
    class TrainOnly(dict):
        def __getitem__(self, key):
            assert key == "train", "Do not access original validation/test"
            return super().__getitem__(key)
        def __iter__(self):
            raise AssertionError("Do not enumerate original held-out splits")
    data["splits"] = TrainOnly(data["splits"])
    first = build_retention_probes(data, old.rows)
    second = build_retention_probes(data, old.rows)
    assert first.to_dict() == second.to_dict()
    assert data == saved_data and old.rows == saved_rows


def test_rejects_inconsistent_previous_sources_mappings_and_changed_frequency_task():
    data = dataset()
    old = previous(data)
    bad = copy.deepcopy(old.rows)
    bad[0]["source_story_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source story hash"):
        build_retention_probes(data, bad)
    bad = copy.deepcopy(old.rows)
    bad[0]["word_ids8"][2] = 14
    with pytest.raises(ValueError, match="class-to-word mapping"):
        build_retention_probes(data, bad)
    with pytest.raises(ValueError, match="Duplicate previous"):
        build_retention_probes(data, old.rows + old.rows[:1])
    data["splits"]["train"] += [source("boost-" + str(i), [14] * 128) for i in range(5)]
    with pytest.raises(ValueError, match="changed the previous ten-class mapping"):
        build_retention_probes(data, old.rows)


def test_balanced_batches_cover_identity_once_repeat_order_five_times_and_never_holdout():
    data = dataset()
    new = build_retention_probes(data, previous(data).rows)
    batches = balanced_training_batches(new.rows)
    assert len(batches) == 80
    usage = Counter(i for batch in batches for i in batch)
    for batch in batches:
        assert len(batch) == 16
        assert Counter(new.rows[i]["task"] for i in batch) == {"identity": 8, "order": 8}
        assert all(new.rows[i]["split"] == "train" for i in batch)
    for i, row in enumerate(new.rows):
        assert usage[i] == (0 if row["split"] != "train" else 1 if row["task"] == "identity" else 5)
    assert balanced_training_batches(new.rows) == batches
    assert balanced_training_batches(new.rows, epoch=1) != batches
    assert balanced_training_batches(new.rows, seed=17) != batches


def test_small_batch_tail_is_explicitly_padded_and_invalid_inputs_fail():
    rows = [{"id": "i" + str(i), "task": "identity", "split": "train"} for i in range(10)]
    rows += [{"id": "o" + str(i), "task": "order", "split": "train"} for i in range(2)]
    rows += [{"id": "v", "task": "identity", "split": "val"}]
    batches = balanced_training_batches(rows)
    assert len(batches) == 2 and all(len(batch) == 16 for batch in batches)
    assert set(range(12)) <= {i for batch in batches for i in batch}
    assert 12 not in {i for batch in batches for i in batch}
    with pytest.raises(ValueError, match="both identity and order"):
        balanced_training_batches(rows[:10])
    with pytest.raises(ValueError, match="Duplicate"):
        balanced_training_batches(rows + rows[:1])
    with pytest.raises(ValueError, match="nonnegative"):
        balanced_training_batches(rows, epoch=-1)
