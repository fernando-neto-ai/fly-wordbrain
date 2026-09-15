"""Numerical-floor screening and causality of independent probe replays."""
from typing import NamedTuple

import numpy as np
import pytest
import torch

from fly_wordbrain.plastic_probe import measure_fast_effect, numerical_effect_gate


class State(NamedTuple):
    h: torch.Tensor
    fast: torch.Tensor


class RecordingBrain:
    def __init__(self):
        self.initial = []
        self.origins = {}
        self.calls = []

    def initial_state(self, batch):
        state = State(torch.zeros(batch, 1), torch.zeros(batch, 1))
        self.origins[id(state)] = len(self.initial)
        self.initial.append(state)
        return state

    def step(self, previous, current, state, plasticity_override=None):
        origin = self.origins[id(state)]
        self.calls.append((origin, previous.tolist(), current.tolist(), plasticity_override, state.h.clone()))
        h = state.h + previous[:, None] * .1 + current[:, None] * .2
        fast = state.fast + .01 if plasticity_override else state.fast
        result = State(h, fast)
        self.origins[id(result)] = origin
        # Retain state objects so Python cannot reuse their IDs during a test.
        self.initial.append(result)
        return h + fast, result


def story():
    return {'id': 'training-story', 'word_ids': [2, 4, 5, 6, 3],
            'words': ['a', 'b', 'c'], 'target_mask': [False, True, True, True, True]}


def test_roundoff_sized_effect_no_longer_passes():
    gate = numerical_effect_gate(1.9e-6, 6e-7)
    assert gate['required_strict_lower_bound'] == 1e-5
    assert not gate['passed']
    assert not numerical_effect_gate(0., 0.)['passed']
    assert not numerical_effect_gate(1e-5, 0.)['passed']
    assert numerical_effect_gate(1.01e-5, 0.)['passed']


def test_measured_repeat_noise_can_raise_floor_and_boundary_is_strict():
    gate = numerical_effect_gate(1e-4, 2e-5)
    assert gate['required_strict_lower_bound'] == pytest.approx(2e-4)
    assert not gate['passed']
    assert not numerical_effect_gate(2e-4, 2e-5)['passed']
    assert numerical_effect_gate(2.01e-4, 2e-5)['passed']
    for difference in (-1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite nonnegative'):
            numerical_effect_gate(difference, 0.)


def test_replays_have_independent_states_and_exact_same_causal_inputs():
    model = RecordingBrain()
    report = measure_fast_effect(model, story(), 3, torch.ones(1))
    # The first three initial states all start at zero with distinct storage.
    assert len({state.h.data_ptr() for state in model.initial[:3]}) == 3
    assert len({state.fast.data_ptr() for state in model.initial[:3]}) == 3
    assert all(torch.count_nonzero(state.h) == 0 for state in model.initial[:3])
    for origin in range(3):
        calls = [call for call in model.calls if call[0] == origin]
        assert [(call[1], call[2]) for call in calls] == [([2], [2]), ([2], [4]), ([4], [5])]
        assert all(call[3] is (origin == 2) for call in calls)
    assert report['max_normalized_frozen_repeat_difference'] == 0.
    assert report['numerical_gate']['passed']
    assert report['positions'] == 3


def test_future_labels_cannot_affect_replay_of_an_earlier_prefix():
    first, changed = story(), story()
    # During the first two observations the model has seen only BOS and a.
    # Both next-word labels can change without entering that observation stream.
    changed['word_ids'][2:4] = [7, 8]
    changed['words'][1:] = ['changed', 'future']
    one = measure_fast_effect(RecordingBrain(), first, 2, torch.ones(1))
    two = measure_fast_effect(RecordingBrain(), changed, 2, torch.ones(1))
    assert one == two


def test_normalization_is_frozen_and_shared_across_three_streams():
    report = measure_fast_effect(RecordingBrain(), story(), 3, torch.tensor([2.]))
    expected = measure_fast_effect(RecordingBrain(), story(), 3, torch.tensor([1.]))
    np.testing.assert_allclose(report['fast_difference_by_position'],
                               np.array(expected['fast_difference_by_position']) / 2)
    for scale in (torch.tensor([0.]), torch.tensor([float('nan')])):
        with pytest.raises(ValueError, match='finite positive'):
            measure_fast_effect(RecordingBrain(), story(), 3, scale)
