"""Causal final-memory loss, matched cached control, and existing-rule gradients."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from fly_wordbrain.feedback_model import FeedbackActionBrain, FeedbackSelector
from fly_wordbrain.retention_model import RetentionModel, RetentionReadout, memory_loss
from test_memory_probe_features import batch, write_graph


@pytest.fixture
def selector(tmp_path):
    torch.set_num_threads(1)
    structure = write_graph(tmp_path, True)
    brain = FeedbackActionBrain(tmp_path, 20, .5, strict_full_graph=False, seed=0,
        input_gain=10., internal_steps=4, structure_path=structure, trainable_susceptibility=True)
    brain.set_activity_scales(.1, .2)
    model = FeedbackSelector(brain, feature_mean=torch.zeros(256), feature_std=torch.ones(256))
    return model.requires_grad_(False).eval()


def moments():
    return torch.zeros(5, 256), torch.full((5, 256), .1)


def test_only_existing_rules_and_new_heads_train_with_matched_seed(selector):
    rng = torch.get_rng_state().clone()
    treatment = RetentionModel(selector, *moments(), seed=7)
    control = RetentionReadout(*moments(), device="cpu", seed=7)
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(control, "selector") and not hasattr(control, "brain")
    for left, right in zip(treatment.head_parameters(), control.head_parameters()):
        assert torch.equal(left, right) and left.requires_grad and right.requires_grad
    assert treatment.identity_head.weight.abs().sum() > 0
    assert torch.all(treatment.identity_head.bias == 0)
    assert torch.all(treatment.order_head.bias == 0)
    assert {name for name, value in treatment.brain.named_parameters() if value.requires_grad} == set(treatment.brain.rule_parameter_names)
    assert all(not value.requires_grad for value in selector.readout.parameters())
    assert sum(value.numel() for value in control.parameters()) == 3084
    assert sum(value.numel() for value in treatment.rule_parameters()) == 34 * 6 + 64
    assert sum(value.numel() for value in treatment.parameters() if value.requires_grad) == 3352
    assert treatment.metadata()["trainable_head_parameters"] == 3084
    treatment.train()
    assert treatment.identity_head.training and treatment.order_head.training
    assert all(not module.training for module in treatment.selector.modules())


def test_causal_features_match_original_trajectory_and_never_read_task_labels(selector):
    model = RetentionModel(selector, *moments())
    class Guard:
        def __init__(self):
            self.__dict__.update(vars(batch()))
        @property
        def targets(self):
            raise AssertionError("Hidden eighth word accessed")
        @property
        def task_ids(self):
            raise AssertionError("Probe task entered neural inputs")
        @property
        def labels(self):
            raise AssertionError("Probe label entered neural inputs")
    inputs = Guard()
    actual, state = model.forward_features(inputs, return_state=True)
    assert actual.shape == (2, 5, 256) and actual.requires_grad
    previous, current, candidates, probabilities, observed = selector._inputs(inputs)
    with torch.no_grad():
        expected, manual = [], model.brain.initial_state(2)
        for t in range(8):
            feature, manual = model.brain.predict(previous[:, t], current[:, t], candidates[:, t],
                probabilities[:, t], manual)
            if t >= 3:
                expected.append(feature)
            if t < 7:
                manual = model.brain.observe(manual, candidates[:, t], probabilities[:, t], observed[:, t], t+1)
        final, original = selector.features(inputs)
    assert torch.equal(actual, torch.stack(expected, 1))
    assert torch.equal(actual[:, -1], final)
    assert torch.equal(state.h, original.h) and torch.equal(state.fast, original.fast)
    assert state.observations == 7 and state.pending
    # The same final row can use different FP32 GEMM kernels when batched
    # across time versus evaluated alone; its mathematical input is equal.
    torch.testing.assert_close(model.logits(actual, "identity")[:, -1],
                               model.logits(actual[:, -1], 0), rtol=1e-6, atol=1e-8)
    altered = actual.detach().clone()
    altered[:, :-1] += 100
    assert torch.equal(model.logits(actual, "order")[:, -1], model.logits(altered, 1)[:, -1])


def test_final_only_loss_balances_tasks_and_ignores_earlier_feature_slots():
    model = RetentionReadout(*moments())
    features = torch.randn(6, 5, 256, generator=torch.Generator().manual_seed(8), requires_grad=True)
    task_ids, labels = torch.tensor([0, 0, 0, 0, 1, 1]), torch.tensor([1, 2, 3, 4, 0, 1])
    actual = model.memory_loss(features, task_ids, labels, [0, 0, 0, 0, 1])
    expected = .5 * (F.cross_entropy(model.logits(features[:4, -1], 0), labels[:4])
                     + F.cross_entropy(model.logits(features[4:, -1], 1), labels[4:]))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.all(features.grad[:, :-1] == 0)
    assert features.grad[:, -1].abs().sum() > 0
    assert model.identity_head.weight.grad.abs().sum() > 0
    assert model.order_head.weight.grad.abs().sum() > 0
    assert torch.equal(memory_loss(model, features.detach(), task_ids, labels, [0, 0, 0, 0, 7]), actual.detach())


def test_first_final_loss_reaches_all_rule_parameter_tensors_and_preserves_graph(selector):
    model = RetentionModel(selector, *moments())
    graph_identity = model.brain.verify_frozen()
    old_head = [value.clone() for value in selector.readout.parameters()]
    normalized_buffers = [model.feature_means.clone(), model.feature_stds.clone()]
    features = model.forward_features(batch())
    model.memory_loss(features, [0, 1], [4, 1], [0, 0, 0, 0, 1]).backward()
    for name, value in model.brain.named_parameters():
        assert value.requires_grad and value.grad is not None, name
        assert bool(torch.isfinite(value.grad).all()) and value.grad.abs().sum() > 0, name
    for value in selector.readout.parameters():
        assert value.grad is None
    previous = {name: value.detach().clone() for name, value in model.brain.named_parameters()}
    optimizer = torch.optim.Adam([
        {"params": model.rule_parameters(), "lr": .003, "eps": 1e-12},
        {"params": model.head_parameters(), "lr": .003}])
    optimizer.step()
    assert any(not torch.equal(value, previous[name]) for name, value in model.brain.named_parameters())
    assert all(torch.equal(a, b) for a, b in zip(old_head, selector.readout.parameters()))
    assert torch.equal(normalized_buffers[0], model.feature_means)
    assert torch.equal(normalized_buffers[1], model.feature_stds)
    assert model.brain.verify_frozen() == graph_identity


def test_cached_control_has_identical_head_loss_gradient_and_update(selector):
    model = RetentionModel(selector, *moments(), train_rules=False, seed=11)
    control = RetentionReadout(*moments(), device="cpu", seed=11)
    assert model.rule_parameters() == []
    assert all(not p.requires_grad for p in selector.parameters())
    features = model.forward_features(batch())
    assert not features.requires_grad
    left = model.memory_loss(features, [0, 1], [5, 1], [0, 0, 0, 0, 1])
    right = control.memory_loss(features.detach().clone(), [0, 1], [5, 1], [0, 0, 0, 0, 1])
    assert torch.equal(left, right)
    left.backward(); right.backward()
    for a, b in zip(model.head_parameters(), control.head_parameters()):
        assert torch.equal(a.grad, b.grad)
    torch.optim.AdamW(model.head_parameters(), lr=.003).step()
    torch.optim.AdamW(control.head_parameters(), lr=.003).step()
    assert all(torch.equal(a, b) for a, b in zip(model.head_parameters(), control.head_parameters()))


def test_fast_erasure_occurs_after_observe_seven_and_preserves_h_and_eligibility(selector, monkeypatch):
    model = RetentionModel(selector, *moments())
    predict = model.brain.predict
    captured = []
    def trace(previous, current, ids, probabilities, state, plasticity=True):
        if state.observations == 7:
            captured.append(state)
        return predict(previous, current, ids, probabilities, state, plasticity=plasticity)
    monkeypatch.setattr(model.brain, "predict", trace)
    normal, normal_state = model.forward_features(batch(), return_state=True)
    erased, erased_state = model.forward_features(batch(), clear_fast_before_final=True, return_state=True)
    assert len(captured) == 2
    assert captured[0].fast.abs().sum() > 0 and torch.all(captured[1].fast == 0)
    assert torch.equal(captured[0].h, captured[1].h)
    assert torch.equal(captured[0].eligibility, captured[1].eligibility)
    assert not captured[1].pending and captured[1].observations == 7
    assert torch.equal(normal[:, :-1], erased[:, :-1])
    assert (normal[:, -1] - erased[:, -1]).abs().max() > 0
    assert normal_state.fast.abs().sum() > 0 and torch.all(erased_state.fast == 0)
    off = model.forward_features(batch(), plasticity=False)
    off_erased = model.forward_features(batch(), plasticity=False, clear_fast_before_final=True)
    assert torch.equal(off, off_erased)


def test_normalization_clones_inputs_and_floors_without_masking():
    mean = torch.zeros(5, 256)
    std = torch.full((5, 256), 1e-8)
    std[0, 0] = 0
    model = RetentionReadout(mean, std)
    assert torch.equal(model.feature_stds[1:], std[1:])
    assert model.feature_stds[0, 0] == torch.tensor(1e-12)
    mean.add_(1); std.add_(1)
    assert torch.all(model.feature_means == 0)
    feature = torch.zeros(1, 5, 256)
    changed = feature.clone(); changed[:, -1, 0] = 1e-8
    assert (model.logits(feature, 0) - model.logits(changed, 0)).abs().sum() > 0


@pytest.mark.parametrize("weights", [[0]*5, [-1, 0, 0, 0, 1], [1]*4, [0, 0, 0, 0, float("nan")]])
def test_invalid_temporal_objectives_are_rejected(weights):
    model = RetentionReadout(*moments())
    with pytest.raises(ValueError, match="Time weights"):
        model.memory_loss(torch.zeros(2, 5, 256), [0, 1], [0, 0], weights)


@pytest.mark.parametrize("tasks,labels", [([0, 2], [0, 0]), ([0, 1], [10, 0]), ([0, 1], [0, 2]),
                                       ([0., 1.], [0, 0]), ([True, False], [0, 0])])
def test_invalid_task_or_class_labels_are_rejected(tasks, labels):
    model = RetentionReadout(*moments())
    with pytest.raises(ValueError):
        model.memory_loss(torch.zeros(2, 5, 256), tasks, labels, [0, 0, 0, 0, 1])
