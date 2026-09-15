"""Delayed-feedback trainer tests on a small real graph and causal episodes."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.action_train import ActionMetrics
from fly_wordbrain.feedback_data import collate_feedback, make_windows
from fly_wordbrain.feedback_model import FeedbackActionBrain, FeedbackSelector
import fly_wordbrain.feedback_train as training
from test_action_train import make_graph, story


def dataset():
    return {"vocabulary": ["<pad>", "<unk>", "<bos>", "<eos>"] + [str(i) for i in range(4, 16)],
            "splits": {"train": [
                story("a", [4, 5, 6, 7, 8, 9, 10, 4, 11, 12, 13, 14, 15, 5, 4, 6]),
                story("b", [5, 6, 7, 8, 9, 10, 11, 5, 12, 13, 14, 15, 4, 6, 5, 7]),
                story("c", [6, 7, 8, 9, 10, 11, 12, 4, 13, 14, 15, 4, 5, 7, 6, 8]),
                story("d", [7, 8, 9, 10, 11, 12, 13, 5, 14, 15, 4, 5, 6, 8, 7, 9])],
                "val": [story("v", [4, 5, 6, 7, 8, 9, 10, 4]),
                        story("w", [6, 7, 8, 9, 10, 11, 12, 4])],
                "test": [{"deliberately_invalid": "never inspect this split"}]}}


@pytest.fixture
def model_data(tmp_path):
    torch.set_num_threads(1)
    make_graph(tmp_path)
    data = dataset()
    proposals = ActionCandidates(data["splits"]["train"], 16, top_k=10)
    model = FeedbackSelector(FeedbackActionBrain(tmp_path, 16, global_scale=.5, top_k=10,
        internal_steps=4, device="cpu", strict_full_graph=False))
    return model, proposals, data


def test_only_eighth_word_contributes_loss_and_missing_candidates_remain_misses():
    candidates = torch.tensor([list(range(4, 14))] * 3)[:, None].repeat(1, 8, 1)
    probabilities = torch.arange(10, 0, -1, dtype=torch.float32)
    probabilities = (probabilities / probabilities.sum() * .9).repeat(3, 8, 1)
    batch = SimpleNamespace(candidates=candidates, probabilities=probabilities, targets=torch.tensor([5, 15, 4]))
    logits = probabilities[:, 7].log().clone().requires_grad_()
    loss, covered = training.window_loss(logits, batch)
    assert covered == 2
    expected = torch.nn.functional.cross_entropy(logits[[0, 2]], torch.tensor([1, 0]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0].abs().sum() > 0 and logits.grad[2].abs().sum() > 0
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))
    with torch.no_grad():
        logits[0, 1] = 10  # Fix one count-model mistake without repairing the missing target.
    metrics = ActionMetrics()
    metrics.update(logits, training.final_batch(batch))
    summary = metrics.summary()
    assert summary["target_count"] == 3 and summary["topk_coverage"] == 2 / 3
    assert summary["accuracy"] == 2 / 3 and summary["conditional_accuracy"] == 1.
    assert summary["baseline_accuracy"] == 1 / 3
    assert summary["fixes"] == 1 and summary["regressions"] == 0
    batch.targets.fill_(15)
    empty_logits = probabilities[:, 7].log().clone().requires_grad_()
    zero, covered = training.window_loss(empty_logits, batch)
    assert covered == 0 and zero.item() == 0
    zero.backward()
    assert torch.equal(empty_logits.grad, torch.zeros_like(empty_logits.grad))


def test_zero_head_matches_count_baseline_before_and_after_calibration(model_data):
    model, proposals, data = model_data
    windows = make_windows(data["splits"]["val"])
    reference = training.evaluate(None, windows, proposals, batch_size=2, baseline_only=True)
    assert training.evaluate(model, windows, proposals, batch_size=2) == reference
    calibration = training.calibrate(model, make_windows(data["splits"]["train"]), proposals, batch_size=2)
    assert calibration["training_only"] and calibration["plasticity_enabled"] is False
    assert calibration["varying_features"] > 0
    assert training.evaluate(model, windows, proposals, batch_size=2) == reference
    assert reference["target_count"] == len(windows)
    assert reference["accuracy"] == reference["baseline_accuracy"]
    assert reference["fixes"] == reference["regressions"] == reference["overrides"] == 0
    with pytest.raises(ValueError, match="unknown training story"):
        training.calibrate(model, windows, proposals, batch_size=2)


def test_forward_does_not_even_read_target_tensor_and_evaluation_restores_modes(model_data):
    model, proposals, data = model_data
    windows = make_windows(data["splits"]["val"])
    batch = collate_feedback(windows, proposals)
    expected = model(batch)

    class UnreadableTarget:
        def __getattribute__(self, name):
            raise AssertionError("Forward accessed eighth-word target: " + name)

    batch.targets = UnreadableTarget()
    torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)
    model.train()
    model.readout.eval()
    before = [module.training for module in model.modules()]
    training.evaluate(model, windows, proposals, batch_size=2)
    assert [module.training for module in model.modules()] == before


def test_fast_off_evaluation_preserves_learning_and_next_training_update(model_data, monkeypatch):
    """An interleaved ablation must match a branch that never ran evaluation."""
    model, proposals, data = model_data
    train_windows = make_windows(data["splits"]["train"])
    validation = make_windows(data["splits"]["val"])
    training.calibrate(model, train_windows, proposals, batch_size=2)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    warmup = collate_feedback(train_windows[:2], proposals, exclude_own_story=True)
    # Populate real Adam momentum, including the slow rule after head warmup.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, covered = training.window_loss(model(warmup), warmup)
        assert covered > 0
        loss.backward()
        optimizer.step()
    assert any(optimizer.state[p]["exp_avg"].abs().sum() > 0 for p in model.brain.parameters())

    untouched = FeedbackSelector(FeedbackActionBrain(model.brain.graph_path, 16, global_scale=.5,
        top_k=10, internal_steps=4, device="cpu", strict_full_graph=False))
    untouched.load_state_dict(model.state_dict())
    untouched.train()
    untouched_optimizer = torch.optim.Adam(untouched.parameters(), lr=.001)
    untouched_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    parameters_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    gradients_before = {name: None if p.grad is None else p.grad.detach().clone()
                        for name, p in model.named_parameters()}
    scales_before = {name: value.clone() for name, value in (
        ("feature_mean", model.feature_mean), ("feature_std", model.feature_std),
        ("pre_scale", model.brain.pre_scale), ("post_scale", model.brain.post_scale))}
    optimizer_before = deepcopy(optimizer.state_dict())
    modes_before = [module.training for module in model.modules()]
    graph_before = model.brain.verify_frozen()

    def assert_tree_equal(actual, expected):
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected):
                assert_tree_equal(left, right)
        else:
            assert actual == expected

    forward_calls, initial_states, final_states = [], [], []
    original_forward, original_initial, original_features = model.forward, model.initial_state, model.features

    def traced_forward(batch, **kwargs):
        forward_calls.append({"plasticity": kwargs.get("plasticity", True),
                              "explicit_flag": "plasticity" in kwargs,
                              "grad_enabled": torch.is_grad_enabled()})
        return original_forward(batch, **kwargs)

    def traced_initial(batch_size):
        state = original_initial(batch_size)
        initial_states.append(state)
        assert torch.all(state.h == 0) and torch.all(state.fast == 0) and torch.all(state.eligibility == 0)
        assert state.observations == 0 and not state.pending
        return state

    def traced_features(batch, plasticity=True):
        features, state = original_features(batch, plasticity=plasticity)
        final_states.append((plasticity, state.fast.detach().clone()))
        return features, state

    monkeypatch.setattr(model, "forward", traced_forward)
    monkeypatch.setattr(model, "initial_state", traced_initial)
    monkeypatch.setattr(model, "features", traced_features)
    training.evaluate(model, validation, proposals, batch_size=1)
    assert [call["plasticity"] for call in forward_calls] == [True, False] * len(validation)
    assert all(not call["grad_enabled"] for call in forward_calls)
    assert all(torch.all(fast == 0) if not enabled else fast.abs().sum() > 0
               for enabled, fast in final_states)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters_before[name]), name
        if gradients_before[name] is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, gradients_before[name]), name
    for name, value in (("feature_mean", model.feature_mean), ("feature_std", model.feature_std),
                        ("pre_scale", model.brain.pre_scale), ("post_scale", model.brain.post_scale)):
        assert torch.equal(value, scales_before[name]), name
    assert_tree_equal(optimizer.state_dict(), optimizer_before)
    assert [module.training for module in model.modules()] == modes_before
    assert model.brain.verify_frozen() == graph_before

    # Compare the next real training window against the branch that skipped
    # evaluation entirely, including the ensuing momentum-based update.
    next_batch = collate_feedback(train_windows[2:4], proposals, exclude_own_story=True)
    optimizer.zero_grad(set_to_none=True)
    untouched_optimizer.zero_grad(set_to_none=True)
    actual, expected = model(next_batch), untouched(next_batch)
    assert forward_calls[-1] == {"plasticity": True, "explicit_flag": False, "grad_enabled": True}
    assert final_states[-1][0] is True and final_states[-1][1].abs().sum() > 0
    assert torch.equal(actual, expected)
    actual_loss, covered = training.window_loss(actual, next_batch)
    expected_loss, expected_covered = training.window_loss(expected, next_batch)
    assert covered == expected_covered > 0
    actual_loss.backward()
    expected_loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.brain.parameters() if p.grad is not None) > 0
    optimizer.step()
    untouched_optimizer.step()
    for (name, parameter), (reference_name, reference) in zip(model.named_parameters(), untouched.named_parameters()):
        assert name == reference_name and torch.equal(parameter, reference), name
    assert_tree_equal(optimizer.state_dict(), untouched_optimizer.state_dict())
    assert len(initial_states) == len(validation) * 2 + 1


def test_preflight_restores_parameters_and_preserves_graph(model_data):
    model, proposals, data = model_data
    windows = make_windows(data["splits"]["train"])
    training.calibrate(model, windows, proposals, batch_size=2)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    graph_identity = model.brain.verify_frozen()
    probe = training.preflight(model, windows, proposals, batch_size=2)
    assert probe["passed"], probe["checks"]
    for key in ("eighth_target_hidden", "feedback_changes_fast_memory", "disabled_fast_state_zero",
                "effective_weight_signs_and_bounds", "parameters_restored"):
        assert probe["checks"][key]
    assert probe["rule_gradient_norms"][0] == 0.
    assert all(value > 0 for value in probe["rule_gradient_norms"][1:])
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, initial[name], rtol=0, atol=0)
        assert p.grad is None
    assert model.brain.verify_frozen() == graph_identity


def test_cpu_training_uses_only_train_val_and_saves_only_learned_parameters(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    graph = tmp_path / "graph"
    graph.mkdir()
    make_graph(graph)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset()))
    output = tmp_path / "run"
    original_loads = json.loads

    class GuardedSplits(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("test split was accessed during training or selection")
            return super().__getitem__(key)

    def guarded_loads(*args, **kwargs):
        value = original_loads(*args, **kwargs)
        if isinstance(value, dict) and isinstance(value.get("splits"), dict):
            value["splits"] = GuardedSplits(value["splits"])
        return value

    monkeypatch.setattr(json, "loads", guarded_loads)
    result = training.train(dataset_path, graph, output, device="cpu", epochs=1, batch_size=2,
        monitor_stories=1, monitor_every=2, progress_every=1, calibration_windows=8,
        threads=1, strict_full_graph=False, global_scale=.5, internal_steps=4)
    assert result["global_step"] == 4 and result["train_windows"] == 8
    assert result["validation_windows"] == 2 and result["test_evaluated"] is False
    history = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert [(row["scope"], row["global_step"]) for row in history] == [
        ("full_validation", 0), ("monitor_subset", 0), ("monitor_subset", 1),
        ("monitor_subset", 2), ("monitor_subset", 4), ("full_validation", 4)]
    assert all(row["target_count"] == row["window_count"] for row in history)
    assert all(row["kind"] == "feedback_action" and row["top_k"] == 10 for row in history)
    for scope in ("monitor_subset", "full_validation"):
        rows = [row for row in history if row["scope"] == scope]
        assert len({row["baseline_accuracy"] for row in rows}) == 1
        assert len({row["topk_coverage"] for row in rows}) == 1
        assert len({row["subset_sha256"] for row in rows}) == 1
    best = torch.load(output / "best.pt", weights_only=True)
    last = torch.load(output / "last.pt", weights_only=True)
    assert best["selection_eligible"] and best["validation"]["scope"] == "full_validation"
    assert best["validation"]["accuracy"] == max(row["accuracy"] for row in history if row["scope"] == "full_validation")
    earliest_best_step = min(row["global_step"] for row in history
                             if row["scope"] == "full_validation" and row["accuracy"] == best["validation"]["accuracy"])
    assert best["global_step"] == earliest_best_step
    assert not last["selection_eligible"] and last["global_step"] == 4
    expected_model = FeedbackSelector(FeedbackActionBrain(graph, 16, global_scale=.5, top_k=10,
        internal_steps=4, device="cpu", strict_full_graph=False))
    expected_names = {name for name, parameter in expected_model.named_parameters() if parameter.requires_grad}
    assert expected_names == {"readout.weight", "readout.bias", "brain.write_strength", "brain.retention_logit",
        "brain.gate_bias", "brain.gate_pre", "brain.gate_post", "brain.feedback_modulation",
        "brain.eligibility_retention_logit"}
    assert set(last["parameters"]) == expected_names
    assert {"readout.weight", "readout.bias"}.issubset(expected_names)
    assert any(name.startswith("brain.") for name in expected_names)
    assert sum(value.numel() for value in last["parameters"].values()) == expected_model.trainable_parameter_count() == 2706
    assert not any("weight" in name and name.startswith("brain.") for name in last["parameters"])
    assert all(torch.isfinite(value).all() for value in last["parameters"].values())
    assert (output / "last.pt").stat().st_size < 500000
    with pytest.raises(ValueError, match="prior experiment"):
        training.train(dataset_path, graph, output, device="cpu")
