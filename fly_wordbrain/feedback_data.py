"""Eight-word episodes: observe seven outcomes, then predict the eighth.

Windows are nonoverlapping lexical spans within one story. Each of the eight
proposal sets is computed before its corresponding word is observed. The final
word is carried only in ``FeedbackBatch.targets`` and never in model inputs.
Training proposals exclude the entire source story, not merely the window.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import torch

from .action_candidates import ActionCandidates


WINDOW_SIZE = 8
OBSERVED_WORDS = 7
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)


@dataclass(frozen=True)
class Window:
    story_id: str
    start: int
    word_ids8: Tuple[int, ...]

    def __post_init__(self):
        if (not isinstance(self.start, Integral) or isinstance(self.start, (bool, np.bool_))
                or self.start < 0 or self.start % WINDOW_SIZE):
            raise ValueError("Window start must be a nonnegative multiple of eight lexical words")
        ids = tuple(self.word_ids8)
        if len(ids) != WINDOW_SIZE or any(
                not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
                or value < 0 or value in (PAD_ID, BOS_ID, EOS_ID) for value in ids):
            raise ValueError("Window must contain exactly eight lexical IDs, including UNK if needed")
        object.__setattr__(self, "story_id", str(self.story_id))
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "word_ids8", tuple(int(value) for value in ids))

    @property
    def word_ids(self):
        """Readable alias for callers that use the corpus field name."""
        return self.word_ids8


@dataclass
class FeedbackBatch:
    previous: torch.Tensor             # int64 [B,8]
    current: torch.Tensor              # int64 [B,8]
    candidates: torch.Tensor           # int64 [B,8,K]
    probabilities: torch.Tensor        # float32 [B,8,K], original vocabulary mass
    observed_ids: torch.Tensor         # int64 [B,7]
    targets: torch.Tensor              # int64 [B]; loss/metrics only
    identity: Tuple[Tuple[str, int], ...]

    @property
    def story_ids(self):
        return tuple(story_id for story_id, _ in self.identity)

    @property
    def starts(self):
        return tuple(start for _, start in self.identity)


def _lexical_ids(story: Dict[str, Any]) -> Tuple[int, ...]:
    ids, mask = story["word_ids"], story["target_mask"]
    if (not len(ids) or ids[0] != BOS_ID or len(ids) != len(mask)
            or any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
                   or value < 0 for value in ids)):
        raise ValueError("Require integer story IDs beginning with BOS and aligned target masks")
    lexical = ids[1:-1] if ids[-1] == EOS_ID else ids[1:]
    if (len(lexical) != len(story["words"]) or len(lexical) > 128
            or any(value in (PAD_ID, BOS_ID, EOS_ID) for value in lexical)
            or bool(mask[0]) or not all(mask[1:])):
        raise ValueError("Malformed lexical sequence, special-token boundary, or target mask")
    return tuple(int(value) for value in lexical)


def make_windows(stories: Iterable[Dict[str, Any]], stride: int = 8) -> list:
    """Preserve source order, reset each story, and discard tails below 8 words."""
    if type(stride) is not int or stride != WINDOW_SIZE:
        raise ValueError("This experiment requires fixed nonoverlapping eight-word windows (stride=8)")
    windows, seen_ids = [], set()
    for story in stories:
        story_id = str(story["id"])
        if story_id in seen_ids:
            raise ValueError("Duplicate source story ID: " + story_id)
        seen_ids.add(story_id)
        lexical = _lexical_ids(story)
        for start in range(0, len(lexical) - WINDOW_SIZE + 1, stride):
            windows.append(Window(story_id, start, lexical[start:start + WINDOW_SIZE]))
    return windows


def collate_feedback(windows: Sequence[Window], proposals: ActionCandidates,
                     device="cpu", exclude_own_story: bool = False) -> FeedbackBatch:
    """Prepare causal inputs; the target for position seven enters targets only."""
    if not windows:
        raise ValueError("Cannot collate an empty episode batch")
    if type(exclude_own_story) is not bool:
        raise ValueError("exclude_own_story must be boolean")
    shape = (len(windows), WINDOW_SIZE)
    previous = np.full(shape, BOS_ID, dtype=np.int64)
    current = np.full(shape, BOS_ID, dtype=np.int64)
    candidates = np.empty(shape + (proposals.top_k,), dtype=np.int64)
    probabilities = np.empty(shape + (proposals.top_k,), dtype=np.float32)
    observed = np.empty((len(windows), OBSERVED_WORDS), dtype=np.int64)
    targets = np.empty(len(windows), dtype=np.int64)
    identities = []
    for batch_index, window in enumerate(windows):
        if not isinstance(window, Window):
            raise TypeError("collate_feedback requires Window objects")
        if max(window.word_ids8) >= proposals.vocabulary_size:
            raise ValueError("Window lexical ID exceeds the proposal vocabulary")
        known_words = window.word_ids8[:OBSERVED_WORDS]
        previous[batch_index, 2:] = known_words[:-1]
        current[batch_index, 1:] = known_words
        for position in range(WINDOW_SIZE):
            ids, probs = proposals.candidates(
                int(previous[batch_index, position]), int(current[batch_index, position]),
                exclude_story_id=window.story_id if exclude_own_story else None)
            candidates[batch_index, position], probabilities[batch_index, position] = ids, probs
        observed[batch_index] = known_words
        targets[batch_index] = window.word_ids8[-1]
        identities.append((window.story_id, window.start))
    tensors = [torch.as_tensor(array, device=device) for array in
               (previous, current, candidates, probabilities, observed, targets)]
    return FeedbackBatch(*tensors, identity=tuple(identities))


def windows_identity(windows: Sequence[Window], dataset_sha256: str, split: str) -> dict:
    """Bind ordered windows, including labels, without putting labels in inputs.

    The compact returned receipt names source stories. Its subset hash also
    includes every ordered start and all eight IDs, avoiding large repeated
    per-window inventories in validation JSONL.
    """
    if not isinstance(dataset_sha256, str) or not dataset_sha256 or not isinstance(split, str) or not split:
        raise ValueError("Require explicit dataset SHA and split identity")
    descriptors, story_ids, seen_stories, seen_windows = [], [], set(), set()
    for window in windows:
        if not isinstance(window, Window):
            raise TypeError("windows_identity requires Window objects")
        key = (window.story_id, window.start)
        if key in seen_windows:
            raise ValueError("Repeated story-window identity")
        seen_windows.add(key)
        if window.story_id not in seen_stories:
            story_ids.append(window.story_id)
            seen_stories.add(window.story_id)
        descriptors.append([window.story_id, window.start, list(window.word_ids8)])
    identity = {"dataset_sha256": dataset_sha256, "split": split,
                "window_size": WINDOW_SIZE, "stride": WINDOW_SIZE,
                "observed_words_per_window": OBSERVED_WORDS,
                "window_count": len(windows), "target_count": len(windows),
                "story_count": len(story_ids), "story_ids": story_ids}
    encoded = json.dumps({**identity, "windows": descriptors},
                         sort_keys=True, separators=(",", ":")).encode()
    return {**identity, "subset_sha256": hashlib.sha256(encoded).hexdigest()}
