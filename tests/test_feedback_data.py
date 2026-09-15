"""Causal episode input construction and full-source-story exclusion."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.feedback_data import Window, collate_feedback, make_windows, windows_identity


def story(identifier, tokens, *, ended=True):
    ids = [2] + list(tokens) + ([3] if ended else [])
    return {"id": identifier, "words": [str(token) for token in tokens], "word_ids": ids,
            "target_mask": [False] + [True] * (len(ids) - 1)}


def sources():
    return [story("a", [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]),
            story("b", [5, 6, 7, 8, 9, 10, 11, 12]),
            story("c", [4, 5, 4, 5, 4, 5, 6, 7], ended=False)]


def test_nonoverlap_drops_short_tails_and_never_crosses_stories_or_specials():
    rows = [story("short", [4] * 7), story("a", list(range(4, 23))),
            story("b", list(range(30, 46)), ended=False)]
    windows = make_windows(rows)
    assert [(w.story_id, w.start) for w in windows] == [("a", 0), ("a", 8), ("b", 0), ("b", 8)]
    assert [w.word_ids8 for w in windows] == [tuple(range(4, 12)), tuple(range(12, 20)),
                                           tuple(range(30, 38)), tuple(range(38, 46))]
    assert all(len(w.word_ids8) == 8 and not set(w.word_ids8) & {0, 2, 3} for w in windows)
    assert windows[0].word_ids == windows[0].word_ids8
    # A natural EOS cannot turn a seven-word tail into an eighth lexical word.
    assert make_windows([story("seven", [4] * 7, ended=True)]) == []
    with pytest.raises(ValueError, match="nonoverlapping"):
        make_windows(rows, stride=1)


def test_final_target_mutation_cannot_change_any_model_input_or_batch_identity():
    rows = sources()
    proposals = ActionCandidates(rows, 16, top_k=10)
    window = make_windows(rows)[0]
    changed = replace(window, word_ids8=window.word_ids8[:-1] + (15,))
    for exclusion in (False, True):
        first = collate_feedback([window], proposals, exclude_own_story=exclusion)
        second = collate_feedback([changed], proposals, exclude_own_story=exclusion)
        for name in ("previous", "current", "candidates", "probabilities", "observed_ids"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
        assert first.identity == second.identity
        assert first.targets.tolist() == [11] and second.targets.tolist() == [15]
        assert first.previous.shape == first.current.shape == (1, 8)
        assert first.candidates.shape == first.probabilities.shape == (1, 8, 10)
        assert first.observed_ids.shape == (1, 7) and first.targets.shape == (1,)
        assert first.previous.dtype == first.observed_ids.dtype == torch.int64
        assert first.probabilities.dtype == torch.float32


def test_observed_word_changes_only_later_proposals_never_its_own_choice_set():
    rows = sources()
    proposals = ActionCandidates(rows, 16, top_k=10)
    window = make_windows(rows)[0]
    changed_ids = list(window.word_ids8)
    changed_ids[3] = 15
    first = collate_feedback([window], proposals, exclude_own_story=True)
    second = collate_feedback([replace(window, word_ids8=tuple(changed_ids))], proposals, exclude_own_story=True)
    for name in ("previous", "current", "candidates", "probabilities"):
        torch.testing.assert_close(getattr(first, name)[:, :4], getattr(second, name)[:, :4], rtol=0, atol=0)
    assert first.current[0, 4] != second.current[0, 4]
    assert first.previous[0, 5] != second.previous[0, 5]
    # Once the two-token count context no longer contains the changed word,
    # proposal identities/probabilities agree; neural feedback can still differ.
    for name in ("previous", "current", "candidates", "probabilities"):
        torch.testing.assert_close(getattr(first, name)[:, 6:], getattr(second, name)[:, 6:], rtol=0, atol=0)
    assert first.observed_ids[0, 3] == 7 and second.observed_ids[0, 3] == 15


def test_proposals_reset_to_bos_each_window_and_use_only_previous_two_observations():
    rows = sources()
    proposals = ActionCandidates(rows, 16, top_k=10)
    windows = make_windows(rows)[:2]
    batch = collate_feedback(windows, proposals, exclude_own_story=True)
    assert batch.previous.tolist() == [[2, 2, 4, 5, 6, 7, 8, 9], [2, 2, 12, 13, 14, 15, 4, 5]]
    assert batch.current.tolist() == [[2, 4, 5, 6, 7, 8, 9, 10], [2, 12, 13, 14, 15, 4, 5, 6]]
    torch.testing.assert_close(batch.candidates[0, 0], batch.candidates[1, 0])
    torch.testing.assert_close(batch.probabilities[0, 0], batch.probabilities[1, 0])
    assert batch.identity == (("a", 0), ("a", 8))
    assert batch.story_ids == ("a", "a") and batch.starts == (0, 8)


def test_training_excludes_entire_source_story_not_just_current_window():
    rows = sources()
    all_counts = ActionCandidates(rows, 16, top_k=10)
    remaining_counts = ActionCandidates(rows[1:], 16, top_k=10)
    windows = make_windows(rows)[:2]
    excluded = collate_feedback(windows, all_counts, exclude_own_story=True)
    fresh = collate_feedback(windows, remaining_counts)
    torch.testing.assert_close(excluded.candidates, fresh.candidates, rtol=0, atol=0)
    torch.testing.assert_close(excluded.probabilities, fresh.probabilities, rtol=0, atol=0)
    normal = collate_feedback(windows, all_counts)
    assert not torch.equal(excluded.probabilities, normal.probabilities)
    # Validation uses training counts without excluding an unknown validation ID.
    val = Window("val", 0, windows[0].word_ids8)
    collate_feedback([val], all_counts)
    with pytest.raises(ValueError, match="unknown training story"):
        collate_feedback([val], all_counts, exclude_own_story=True)


def test_identity_binds_labels_order_split_dataset_and_unique_window_positions():
    windows = make_windows(sources())
    identity = windows_identity(windows, "dataset-sha", "train")
    assert identity["window_count"] == identity["target_count"] == 4
    assert identity["story_count"] == 3 and identity["story_ids"] == ["a", "b", "c"]
    assert identity["window_size"] == identity["stride"] == 8
    assert identity["observed_words_per_window"] == 7
    assert identity == windows_identity(windows, "dataset-sha", "train")
    assert identity["subset_sha256"] != windows_identity(windows[::-1], "dataset-sha", "train")["subset_sha256"]
    assert identity["subset_sha256"] != windows_identity(windows, "other", "train")["subset_sha256"]
    assert identity["subset_sha256"] != windows_identity(windows, "dataset-sha", "val")["subset_sha256"]
    changed = [replace(windows[0], word_ids8=windows[0].word_ids8[:-1] + (15,))] + windows[1:]
    assert identity["subset_sha256"] != windows_identity(changed, "dataset-sha", "train")["subset_sha256"]
    with pytest.raises(ValueError, match="Repeated"):
        windows_identity(windows + [windows[0]], "dataset-sha", "train")


def test_unknown_words_remain_lexical_and_no_test_split_is_accessed():
    class NoTestAccess(dict):
        def __getitem__(self, key):
            assert key != "test", "Test split must not be read"
            return super().__getitem__(key)
    splits = NoTestAccess(train=[story("train", [1, 4, 5, 6, 7, 8, 9, 10])],
                          val=[story("val", [4] * 8)])
    windows = make_windows(splits["train"])
    assert windows[0].word_ids8[0] == 1
    proposals = ActionCandidates(splits["train"], 16, top_k=10)
    collate_feedback(windows, proposals, exclude_own_story=True)
    collate_feedback(make_windows(splits["val"]), proposals)


@pytest.mark.parametrize("mutation", [
    lambda s: s["word_ids"].__setitem__(0, 4),
    lambda s: s["word_ids"].__setitem__(2, 2),
    lambda s: s["word_ids"].__setitem__(2, 3),
    lambda s: s["target_mask"].__setitem__(2, False),
    lambda s: s["target_mask"].__setitem__(0, True),
    lambda s: s["words"].append("extra"),
])
def test_malformed_story_cannot_shift_episode_alignment(mutation):
    row = story("bad", [4] * 8)
    mutation(row)
    with pytest.raises(ValueError):
        make_windows([row])


def test_bad_window_batch_and_duplicate_source_ids_fail():
    rows = sources()
    with pytest.raises(ValueError, match="Duplicate"):
        make_windows([rows[0], rows[0]])
    with pytest.raises(ValueError, match="eight lexical"):
        Window("x", 0, (4,) * 7)
    with pytest.raises(ValueError, match="eight lexical"):
        Window("x", 0, (4,) * 7 + (3,))
    with pytest.raises(ValueError, match="multiple"):
        Window("x", 1, (4,) * 8)
    proposals = ActionCandidates(rows, 16, top_k=10)
    with pytest.raises(ValueError, match="empty"):
        collate_feedback([], proposals)
    with pytest.raises(ValueError, match="vocabulary"):
        collate_feedback([Window("x", 0, (16,) * 8)], proposals)
