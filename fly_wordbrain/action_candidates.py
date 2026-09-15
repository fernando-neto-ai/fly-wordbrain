"""Train-only word proposals for a five-action fly selector.

This is the next-word ``HorizonCounts(kind='ordered_pair')`` distribution:
add-one unigram smoothing, then current-word and ordered-pair backoff with
prior mass one. Returned probabilities retain their original full-vocabulary
mass; they are NOT renormalized over the selected actions. PAD/BOS cannot be
actions, but retain their smoothing mass in the underlying distribution.

Training examples should pass ``exclude_story_id=story['id']`` so the proposal
table never uses that story's targets, including its later words. Validation
uses all training stories and no exclusion. Only the supplied training stories
are fitted; querying a context never fits, inserts, or reads a future target.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from numbers import Integral
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


PAD_ID, BOS_ID, EOS_ID = 0, 2, 3


def _story_rows(story: Dict[str, Any], vocabulary_size: int) -> np.ndarray:
    """The first horizon of plastic_train.causal_rows, without importing torch."""
    ids, masks = story["word_ids"], story["target_mask"]
    if (len(ids) < 2 or ids[0] != BOS_ID or len(ids) != len(masks)
            or len(story["words"]) > 128):
        raise ValueError("Require BOS, aligned masks, and at most 128 lexical words")
    if (any(not isinstance(i, Integral) or isinstance(i, (bool, np.bool_))
            or not 0 <= i < vocabulary_size for i in ids)
            or any(i in (PAD_ID, BOS_ID) for i in ids[1:])
            or EOS_ID in ids[1:-1]
            or len(ids) != 1 + len(story["words"]) + int(ids[-1] == EOS_ID)):
        raise ValueError("Malformed bounded story token IDs")
    rows = []
    for position in range(len(ids) - 1):
        if position > 128 or ids[position] == EOS_ID or not masks[position + 1]:
            raise ValueError("Invalid recurrent position or interior masked target")
        rows.append((ids[position - 1] if position else BOS_ID,
                     ids[position], ids[position + 1]))
    result = np.asarray(rows, dtype=np.int64)
    result.flags.writeable = False
    return result


class ActionCandidates:
    """Deterministic top-K proposals, with optional whole-story exclusion.

    ``training_stories`` must be precisely the training split. The class accepts
    no validation/test dataset and never mutates the supplied story objects.
    A context is ``(previous_id, current_id)``; at a story's start it is
    ``(BOS_ID, BOS_ID)``. Ties are resolved by the lower token ID.

    Counts are sparse, stored source rows are linear in training tokens, and
    both caches are bounded. No context-by-vocabulary dense table is stored.
    """

    CANDIDATE_CACHE_SIZE = 4096
    # Keep a normal story batch resident while its words are interleaved.
    EXCLUSION_CACHE_SIZE = 64

    def __init__(self, training_stories: Iterable[Dict[str, Any]],
                 vocabulary_size: int, top_k: int = 5):
        if (not isinstance(vocabulary_size, Integral) or isinstance(vocabulary_size, bool)
                or vocabulary_size < 5):
            raise ValueError("vocabulary_size must include four specials and a lexical word")
        if (not isinstance(top_k, Integral) or isinstance(top_k, bool)
                or not 1 <= top_k <= vocabulary_size - 2):
            raise ValueError("top_k must fit the vocabulary excluding PAD/BOS")
        if isinstance(training_stories, dict):
            raise TypeError("Pass the training story sequence, not a dataset or split mapping")
        self.vocabulary_size, self.top_k = int(vocabulary_size), int(top_k)
        self._unigram_counts = np.zeros(self.vocabulary_size, dtype=np.int64)
        self._current_counts = defaultdict(Counter)
        self._pair_counts = defaultdict(Counter)
        self._story_rows = {}
        for story in training_stories:
            story_id = str(story["id"])
            if story_id in self._story_rows:
                raise ValueError("Duplicate training story ID: " + story_id)
            rows = _story_rows(story, self.vocabulary_size)
            self._story_rows[story_id] = rows
            for previous, current, target in rows.tolist():
                self._unigram_counts[target] += 1
                self._current_counts[current][target] += 1
                self._pair_counts[(previous, current)][target] += 1
        if not self._story_rows:
            raise ValueError("Require nonempty training stories")
        self.training_story_ids = tuple(self._story_rows)
        self.training_target_count = int(self._unigram_counts.sum())
        self._current_totals = {key: sum(value.values()) for key, value in self._current_counts.items()}
        self._pair_totals = {key: sum(value.values()) for key, value in self._pair_counts.items()}
        self._allowed_ids = np.asarray([i for i in range(self.vocabulary_size)
                                       if i not in (PAD_ID, BOS_ID)], dtype=np.int64)
        self._candidate_cache = OrderedDict()
        self._exclusion_cache = OrderedDict()

    def _exclusion(self, story_id: Optional[str]):
        if story_id is None:
            return None
        story_id = str(story_id)
        if story_id not in self._story_rows:
            raise ValueError("Cannot exclude unknown training story ID: " + story_id)
        if story_id in self._exclusion_cache:
            self._exclusion_cache.move_to_end(story_id)
            return self._exclusion_cache[story_id]
        unigram = np.zeros(self.vocabulary_size, dtype=np.int64)
        current_counts, pair_counts = defaultdict(Counter), defaultdict(Counter)
        for previous, current, target in self._story_rows[story_id].tolist():
            unigram[target] += 1
            current_counts[current][target] += 1
            pair_counts[(previous, current)][target] += 1
        result = (unigram, current_counts, pair_counts,
                  {key: sum(value.values()) for key, value in current_counts.items()},
                  {key: sum(value.values()) for key, value in pair_counts.items()})
        self._exclusion_cache[story_id] = result
        if len(self._exclusion_cache) > self.EXCLUSION_CACHE_SIZE:
            self._exclusion_cache.popitem(last=False)
        return result

    def candidates(self, previous_id: int, current_id: int,
                   exclude_story_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return int64 IDs[K] and float64 full-distribution probabilities[K].

        No future label is an argument. A correct word absent from the top K
        remains absent; callers must count that example as an oracle miss.
        """
        for token_id in (previous_id, current_id):
            if (not isinstance(token_id, Integral) or isinstance(token_id, (bool, np.bool_))
                    or not 0 <= token_id < self.vocabulary_size):
                raise ValueError("Context token ID outside vocabulary")
        previous_id, current_id = int(previous_id), int(current_id)
        excluded_id = None if exclude_story_id is None else str(exclude_story_id)
        # Validate even on cache hits; no unknown ID silently means no exclusion.
        if excluded_id is not None and excluded_id not in self._story_rows:
            raise ValueError("Cannot exclude unknown training story ID: " + excluded_id)
        key = (previous_id, current_id, excluded_id)
        if key in self._candidate_cache:
            self._candidate_cache.move_to_end(key)
            ids, probabilities = self._candidate_cache[key]
            return ids.copy(), probabilities.copy()
        excluded = self._exclusion(excluded_id)
        unigram = self._unigram_counts.copy()
        current_total = self._current_totals.get(current_id, 0)
        pair = (previous_id, current_id)
        pair_total = self._pair_totals.get(pair, 0)
        if excluded is not None:
            unigram -= excluded[0]
            current_total -= excluded[3].get(current_id, 0)
            pair_total -= excluded[4].get(pair, 0)
        probabilities = (unigram + 1.) / (int(unigram.sum()) + self.vocabulary_size)
        excluded_current = excluded[1].get(current_id, {}) if excluded is not None else {}
        for token_id, count in self._current_counts.get(current_id, {}).items():
            # Subtract integers before adding the prior, preserving exact ties
            # when exclusion removes all occurrences of an observed target.
            probabilities[token_id] += count - excluded_current.get(token_id, 0)
        probabilities /= current_total + 1.
        excluded_pair = excluded[2].get(pair, {}) if excluded is not None else {}
        for token_id, count in self._pair_counts.get(pair, {}).items():
            probabilities[token_id] += count - excluded_pair.get(token_id, 0)
        probabilities /= pair_total + 1.
        order = np.lexsort((self._allowed_ids, -probabilities[self._allowed_ids]))[:self.top_k]
        ids = self._allowed_ids[order].copy()
        selected_probabilities = probabilities[ids].copy()
        self._candidate_cache[key] = (ids, selected_probabilities)
        if len(self._candidate_cache) > self.CANDIDATE_CACHE_SIZE:
            self._candidate_cache.popitem(last=False)
        return ids.copy(), selected_probabilities.copy()
