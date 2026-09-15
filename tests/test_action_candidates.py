"""Exact proposal probabilities, training exclusion, and no label repair."""
from copy import deepcopy

import numpy as np
import pytest

from fly_wordbrain.action_candidates import ActionCandidates


def story(identifier, tokens, *, ended=True):
    ids = [2] + list(tokens) + ([3] if ended else [])
    return {"id": identifier, "word_ids": ids, "words": [str(t) for t in tokens],
            "target_mask": [False] + [True] * (len(ids) - 1)}


def dense_reference(stories, vocabulary_size, previous, current):
    # Deliberately build independent dense n-gram arrays. This catches context
    # shifts, smoothing-order mistakes, and cross-story transitions.
    uni = np.ones(vocabulary_size, dtype=np.float64)
    bi = np.zeros(vocabulary_size, dtype=np.int64)
    tri = np.zeros(vocabulary_size, dtype=np.int64)
    for row in stories:
        tokens = row["word_ids"]
        for index, target in enumerate(tokens[1:]):
            uni[target] += 1
            if tokens[index] == current:
                bi[target] += 1
                if (tokens[index - 1] if index else 2) == previous:
                    tri[target] += 1
    p_uni = uni / uni.sum()
    p_bi = (bi + p_uni) / (bi.sum() + 1.)
    return (tri + p_bi) / (tri.sum() + 1.)


def expected_top(probabilities, count):
    ids = sorted((i for i in range(len(probabilities)) if i not in (0, 2)),
                 key=lambda i: (-probabilities[i], i))[:count]
    return np.asarray(ids, dtype=np.int64), probabilities[ids]


@pytest.mark.parametrize("vocabulary_size", [9, 1024])
def test_exact_dense_smoothing_all_contexts_and_whole_story_exclusion(vocabulary_size):
    rows = [story("a", [4, 5, 4, 6]), story("b", [4, 5, 7], ended=False),
            story("c", [8, 5, 4, 5, 8])]
    model = ActionCandidates(rows, vocabulary_size)
    for excluded_id in (None, "a", "b", "c"):
        remaining = [s for s in rows if s["id"] != excluded_id]
        for previous, current in ((2, 2), (2, 4), (4, 5), (5, 4), (5, 8), (7, 8), (3, 2)):
            ids, probabilities = model.candidates(previous, current, excluded_id)
            dense = dense_reference(remaining, vocabulary_size, previous, current)
            wanted_ids, wanted_probs = expected_top(dense, 5)
            np.testing.assert_array_equal(ids, wanted_ids)
            np.testing.assert_allclose(probabilities, wanted_probs, rtol=1e-13, atol=1e-15)
            assert ids.dtype == np.int64 and probabilities.dtype == np.float64
            assert not set(ids) & {0, 2}
            assert probabilities.sum() < 1.  # Original probability mass retained.


def test_matches_existing_horizon_counts_without_exclusion():
    from fly_wordbrain.pair_decoder import HorizonCounts
    from fly_wordbrain.plastic_train import causal_rows

    stories = [story("a", [4, 5, 4, 6]), story("b", [4, 5, 7], ended=False)]
    causal = [row for source in stories for row in causal_rows(source)]
    counts = HorizonCounts(np.asarray([r[0] for r in causal]), np.asarray([r[1] for r in causal]),
                           np.asarray([r[2] for r in causal]), 1024)
    candidate_model = ActionCandidates(stories, 1024)
    for previous, current in ((2, 2), (4, 5), (7, 8)):
        dense = np.asarray([counts.probability(previous, current, t) for t in range(1024)])
        expected_ids, expected_probs = expected_top(dense, 5)
        ids, probabilities = candidate_model.candidates(previous, current)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(probabilities, expected_probs, rtol=1e-14, atol=0.)


def test_exclusion_removes_entire_story_including_future_counts_and_natural_eos():
    excluded = story("a", [4, 5, 4, 5, 4, 5])
    other = story("b", [4, 7], ended=False)
    model = ActionCandidates([excluded, other], 9, top_k=7)
    without = model.candidates(2, 4, exclude_story_id="a")
    fresh = ActionCandidates([other], 9, top_k=7).candidates(2, 4)
    for actual, expected in zip(without, fresh):
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    assert model.candidates(2, 4)[0][0] == 5
    assert without[0][0] == 7
    # No artificial EOS is fitted for a truncated story and no previous story
    # is joined to a following BOS. The dense reference checks these contexts.
    for context in ((4, 7), (3, 2), (5, 2)):
        ids, probabilities = model.candidates(*context, exclude_story_id="a")
        expected_ids, expected_probs = expected_top(dense_reference([other], 9, *context), 7)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(probabilities, expected_probs, rtol=1e-14, atol=1e-15)


def test_only_story_excluded_leaves_uniform_prior_and_id_ties():
    model = ActionCandidates([story("only", [8, 8])], 9)
    ids, probabilities = model.candidates(8, 8, exclude_story_id="only")
    np.testing.assert_array_equal(ids, [1, 3, 4, 5, 6])
    np.testing.assert_allclose(probabilities, np.full(5, 1 / 9), rtol=1e-14, atol=1e-15)


def test_top_five_miss_stays_missing_and_queries_cannot_fit_validation():
    training = [story("train-" + str(i), [4, 5], ended=False) for i in range(4)]
    original = deepcopy(training)
    validation = story("val", [4, 8], ended=False)
    test = story("test", [4, 7], ended=False)
    dataset = {"train": training, "val": [validation], "test": [test]}
    model = ActionCandidates(dataset["train"], 9)
    before = model.candidates(2, 4)
    assert validation["word_ids"][2] not in before[0]
    assert test["word_ids"][2] not in before[0]
    for _ in range(3):
        ids, probabilities = model.candidates(2, validation["word_ids"][1])
        np.testing.assert_array_equal(ids, before[0])
        np.testing.assert_array_equal(probabilities, before[1])
    assert training == original
    assert model.training_story_ids == tuple(row["id"] for row in training)
    assert model.training_target_count == 8
    with pytest.raises(ValueError, match="unknown training story"):
        model.candidates(2, 4, exclude_story_id="val")
    with pytest.raises(TypeError, match="training story sequence"):
        ActionCandidates(dataset, 9)


def test_caches_bounded_and_returned_arrays_do_not_mutate_counts(monkeypatch):
    monkeypatch.setattr(ActionCandidates, "CANDIDATE_CACHE_SIZE", 2)
    monkeypatch.setattr(ActionCandidates, "EXCLUSION_CACHE_SIZE", 1)
    rows = [story("a", [4, 5]), story("b", [4, 6]), story("c", [7, 8])]
    model = ActionCandidates(rows, 9)
    before = model.candidates(2, 2)
    returned_ids, returned_probs = model.candidates(2, 2)
    returned_ids[:] = 0
    returned_probs[:] = -10
    for actual, expected in zip(model.candidates(2, 2), before):
        np.testing.assert_array_equal(actual, expected)
    for row in rows:
        model.candidates(2, 2, exclude_story_id=row["id"])
    assert len(model._candidate_cache) <= 2 and len(model._exclusion_cache) <= 1
    for actual, expected in zip(model.candidates(2, 2), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mutation", [
    lambda row: row["word_ids"].__setitem__(0, 4),
    lambda row: row["word_ids"].__setitem__(1, 0),
    lambda row: row["word_ids"].__setitem__(1, 2),
    lambda row: row["word_ids"].__setitem__(1, 3),
    lambda row: row["word_ids"].__setitem__(1, 9),
    lambda row: row["target_mask"].__setitem__(1, False),
    lambda row: row["words"].extend(["x"] * 129),
])
def test_rejects_malformed_story_boundaries_and_masks(mutation):
    row = story("bad", [4, 5])
    mutation(row)
    with pytest.raises(ValueError):
        ActionCandidates([row], 9)


def test_rejects_empty_duplicate_and_out_of_range_configuration():
    with pytest.raises(ValueError, match="nonempty"):
        ActionCandidates([], 9)
    with pytest.raises(ValueError, match="Duplicate"):
        ActionCandidates([story("x", [4]), story("x", [5])], 9)
    with pytest.raises(ValueError, match="top_k"):
        ActionCandidates([story("x", [4])], 6)
    model = ActionCandidates([story("x", [4])], 9)
    for previous, current in ((-1, 4), (4, 9), (True, 4)):
        with pytest.raises(ValueError, match="Context token"):
            model.candidates(previous, current)
