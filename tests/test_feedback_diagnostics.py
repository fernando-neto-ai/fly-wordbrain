"""Read-only action-distribution and edge-group diagnostic semantics."""
import json
import math

import pytest
import torch

from fly_wordbrain.feedback_diagnostics import measure_action_effect, measure_group_state


def test_noop_and_per_example_common_shift_have_no_action_effect():
    logits = torch.tensor([[1., 2., -1.], [4., 0., 2.]], dtype=torch.float64, requires_grad=True)
    for shifted in (logits, logits + torch.tensor([[16.], [-8.]], dtype=torch.float64)):
        receipt = measure_action_effect(shifted, logits)
        assert receipt["examples"] == 2 and receipt["actions"] == 3
        for key in ("centered_logit_rms", "centered_logit_max", "p95_centered_logit_max",
                    "mean_total_variation", "max_total_variation", "changed_predictions", "changed_prediction_rate"):
            assert receipt[key] == 0
        json.dumps(receipt, allow_nan=False)
    assert logits.grad is None


def test_hand_computed_centered_effect_and_tie_breaking():
    off = torch.zeros((2, 2), dtype=torch.float64)
    on = torch.tensor([[1., -1.], [0., 2.]], dtype=torch.float64)
    receipt = measure_action_effect(on, off)
    assert receipt["centered_logit_rms"] == receipt["centered_logit_max"] == 1.
    assert receipt["p95_centered_logit_max"] == 1.
    expected_tv = 1 / (1 + math.exp(-2.)) - .5
    assert receipt["mean_total_variation"] == pytest.approx(expected_tv)
    assert receipt["max_total_variation"] == pytest.approx(expected_tv)
    assert receipt["changed_predictions"] == 1 and receipt["changed_prediction_rate"] == .5


def test_probability_change_can_exist_without_argmax_change():
    first = torch.log(torch.tensor([[.8, .1, .1]], dtype=torch.float64))
    second = torch.log(torch.tensor([[.6, .2, .2]], dtype=torch.float64))
    receipt = measure_action_effect(first, second)
    assert receipt["changed_predictions"] == 0
    assert receipt["mean_total_variation"] == pytest.approx(.2)
    assert receipt["centered_logit_rms"] > 0


def test_supports_noncontiguous_arbitrary_batch_action_dimensions_and_detaches():
    logits = torch.arange(80., dtype=torch.float32).reshape(10, 8).t().requires_grad_()
    before = logits.detach().clone()
    result = measure_action_effect(logits, logits.flip(1))
    assert result["examples"] == 8 and result["actions"] == 10
    assert result["changed_predictions"] == 8
    assert torch.equal(before, logits) and logits.grad is None
    assert 0 <= result["mean_total_variation"] <= 1


@pytest.mark.parametrize("on,off", [
    (torch.ones(2, 3), torch.ones(3, 3)), (torch.ones(0, 3), torch.ones(0, 3)),
    (torch.ones(2, 1), torch.ones(2, 1)), (torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3)),
    (torch.tensor([[float('nan'), 1.]]), torch.ones(1, 2)),
    (torch.tensor([[float('inf'), 1.]]), torch.ones(1, 2)),
])
def test_bad_logits_are_explicit_errors(on, off):
    with pytest.raises(ValueError):
        measure_action_effect(on, off)


def test_group_moments_include_batch_and_optional_gradients_once_per_edge():
    fast = torch.tensor([[1., -3., 2.], [-1., 3., 4.]], requires_grad=True)
    groups = torch.tensor([2, 2, 7], dtype=torch.int32)
    gradients = torch.tensor([2., -4., 0.], requires_grad=True)
    receipt = measure_group_state(fast, groups, gradients)
    assert (receipt["batch_size"], receipt["edge_count"], receipt["group_count"]) == (2, 3, 2)
    a, b = receipt["groups"]
    assert a["group"] == 2 and a["edge_count"] == 2
    assert a["fast_rms"] == pytest.approx(math.sqrt(5.)) and a["fast_mean_abs"] == 2.
    assert a["susceptibility_gradient_rms"] == pytest.approx(math.sqrt(10.))
    assert a["susceptibility_gradient_mean_abs"] == 3. and a["susceptibility_gradient_max_abs"] == 4.
    assert b["group"] == 7 and b["fast_rms"] == pytest.approx(math.sqrt(10.))
    assert b["susceptibility_gradient_rms"] == 0.
    absent = measure_group_state(fast, groups)
    assert all(row["susceptibility_gradient_rms"] is None for row in absent["groups"])
    assert fast.grad is gradients.grad is None
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("groups,grad", [
    (torch.tensor([0., 1.]), None), (torch.tensor([0, -1]), None),
    (torch.tensor([0]), None), (torch.tensor([0, 1]), torch.ones(1)),
    (torch.tensor([0, 1]), torch.tensor([float('nan'), 0.])),
])
def test_group_shape_type_and_finiteness_are_checked(groups, grad):
    with pytest.raises(ValueError):
        measure_group_state(torch.zeros(2, 2), groups, grad)
