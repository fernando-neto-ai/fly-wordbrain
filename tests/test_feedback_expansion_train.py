"""Expanded fast-edge calibration, optimizer isolation, and real CPU training."""
import json

import numpy as np
import pytest
import torch

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.fast_structure import ARRAY_KEYS, ORIGINAL_GROUPS, array_digest, original_selection_sha256, save_structure
from fly_wordbrain.feedback_data import make_windows, collate_feedback
from fly_wordbrain.feedback_model import FeedbackActionBrain, FeedbackSelector
import fly_wordbrain.feedback_train as training
from test_action_train import make_graph
from test_feedback_train import dataset


def expanded_fixture(root):
    graph_path = root / "graph"
    graph_path.mkdir()
    make_graph(graph_path)
    with np.load(graph_path / "graph.npz", allow_pickle=False) as source:
        graph = {key: source[key] for key in source.files}
    ids = np.arange(16, dtype=np.int64)
    arrays = {"candidate_edge_ids": ids, "candidate_group": np.r_[np.arange(4), [4] * 6, [5] * 6].astype(np.int64),
        "candidate_pre": ids.copy(), "candidate_post": graph["post"][ids].astype(np.int64),
        "candidate_weight": graph["weight"][ids].copy()}
    metadata = {"schema": 1, "kind": "fast_structure", "graph_sha256": training.file_sha256(graph_path / "graph.npz"),
        "original_selection_sha256": original_selection_sha256(graph),
        "selection_array_sha256": array_digest([arrays[key] for key in ARRAY_KEYS]),
        "group_names": list(ORIGINAL_GROUPS) + ["toy_existing_inputs_a", "toy_existing_inputs_b"],
        "neurons": 64, "edges": 64, "candidate_edges": 16, "original_candidate_edges": 4}
    structure = root / "selected.npz"
    save_structure(structure, arrays, metadata)
    return graph_path, structure


def make_model(graph_path, structure):
    return FeedbackSelector(FeedbackActionBrain(graph_path, 16, global_scale=.5, top_k=10,
        internal_steps=4, device="cpu", strict_full_graph=False,
        structure_path=structure, trainable_susceptibility=True))


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    graph_path, structure = expanded_fixture(tmp_path)
    data = dataset()
    model = make_model(graph_path, structure)
    proposals = ActionCandidates(data["splits"]["train"], 16, top_k=10)
    return model, proposals, data, graph_path, structure


def frozen_moments(model, windows, proposals):
    pre, post, output = [], [], []
    with torch.no_grad():
        for offset in range(0, len(windows), 2):
            batch = collate_feedback(windows[offset:offset + 2], proposals, exclude_own_story=True)
            state = model.initial_state(len(batch.identity))
            for position in range(8):
                features, state = model.brain.predict(batch.previous[:, position], batch.current[:, position],
                    batch.candidates[:, position], batch.probabilities[:, position], state, plasticity=False)
                pre.append(state.h[:, model.brain.candidate_pre].numpy())
                post.append(state.h[:, model.brain.candidate_post].numpy())
                if position < 7:
                    state = model.brain.observe(state, batch.candidates[:, position], batch.probabilities[:, position],
                                               batch.observed_ids[:, position], position + 1)
            assert torch.count_nonzero(state.fast) == 0
            output.append(features.numpy())
    scales = [np.maximum(np.sqrt(np.mean(np.concatenate(values).astype(np.float64) ** 2, axis=0)), 1e-6).astype(np.float32)
              for values in (pre, post)]
    return scales, np.concatenate(output)


def test_calibration_first_frozen_activity_then_fast_features_on_same_train_only_windows(setup, monkeypatch):
    model, proposals, data, graph_path, structure = setup
    windows = make_windows(data["splits"]["train"])
    fresh = make_model(graph_path, structure)
    expected_scales, frozen_features = frozen_moments(fresh, windows, proposals)
    fresh.brain.set_activity_scales(*expected_scales)
    expected_features = []
    with torch.no_grad():
        for offset in range(0, len(windows), 2):
            batch = collate_feedback(windows[offset:offset + 2], proposals, exclude_own_story=True)
            features, _ = fresh.features(batch, plasticity=True)
            expected_features.append(features.numpy())
    expected_features = np.concatenate(expected_features).astype(np.float64)
    expected_std = expected_features.std(0)
    expected_std = np.maximum(expected_std, max(expected_std.max() * 1e-4, 1e-6)).astype(np.float32)
    calls = []
    original_candidates = proposals.candidates
    def guarded(previous, current, exclude_story_id=None):
        assert exclude_story_id in proposals.training_story_ids
        calls.append((previous, current, exclude_story_id))
        return original_candidates(previous, current, exclude_story_id)
    monkeypatch.setattr(proposals, "candidates", guarded)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    receipt = training.calibrate(model, windows, proposals, 2, fast_features=True)
    assert receipt["training_only"] and receipt["plasticity_enabled"]
    assert receipt["activity_scales_plasticity_enabled"] is False
    assert receipt["window_count"] == len(windows)
    first_pass_calls = len(windows) * 8
    assert len(calls) == 2 * first_pass_calls
    assert calls[:first_pass_calls] == calls[first_pass_calls:]
    np.testing.assert_array_equal(model.brain.pre_scale.detach().numpy(), expected_scales[0])
    np.testing.assert_array_equal(model.brain.post_scale.detach().numpy(), expected_scales[1])
    np.testing.assert_allclose(model.feature_mean.detach().numpy(), expected_features.mean(0), rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(model.feature_std.detach().numpy(), expected_std)
    assert not np.array_equal(expected_features.astype(np.float32), frozen_features)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    with pytest.raises((ValueError, AssertionError)):
        training.calibrate(model, make_windows(data["splits"]["val"]), proposals, 2, fast_features=True)


def test_optimizer_partitions_exact_parameters_lr_epsilon_and_separate_clipping(setup):
    model, _, _, _, _ = setup
    optimizer = training.make_optimizer(model, lr=.0003, rule_lr=.003, rule_eps=1e-12)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"plasticity", "action_head"}
    rules, head = groups["plasticity"], groups["action_head"]
    assert rules["lr"] == .003 and rules["eps"] == 1e-12
    assert head["lr"] == .0003 and head["eps"] == 1e-8
    assert sum(p.numel() for p in rules["params"]) == 34 * 6 + 16
    assert sum(p.numel() for p in head["params"]) == 2570
    assert not {id(p) for p in rules["params"]} & {id(p) for p in head["params"]}
    assert {id(p) for group in groups.values() for p in group["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad}
    for large_group, tiny_group in ((head, rules), (rules, head)):
        for p in large_group["params"]:
            p.grad = torch.full_like(p, 100.)
        for p in tiny_group["params"]:
            p.grad = torch.full_like(p, 1e-7)
        tiny_before = [p.grad.clone() for p in tiny_group["params"]]
        training.clip_gradients(model, separate=True)
        large_norm = torch.sqrt(sum(p.grad.square().sum() for p in large_group["params"]))
        assert .999 < large_norm <= 1.00001
        for p, expected in zip(tiny_group["params"], tiny_before):
            torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)


def test_expanded_strict_preflight_susceptibility_effect_and_original_graph_restoration(setup):
    model, proposals, data, _, _ = setup
    windows = make_windows(data["splits"]["train"])
    training.calibrate(model, windows, proposals, 2, fast_features=True)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    graph_before = model.brain.base_graph_fingerprint()
    probe = training.preflight(model, windows, proposals, 2, lr=.0003, rule_lr=.003, rule_eps=1e-12,
                               optimizer_steps=4, minimum_feature_effect=1e-5, minimum_action_tv=1e-10)
    assert probe["passed"], probe["checks"]
    assert probe["checks"]["fast_changes_action_distribution"]
    assert probe["parameter_max_deltas"]["brain.slow_susceptibility"] > 0
    assert probe["action_effect"]["mean_total_variation"] > 1e-10
    assert model.brain.base_graph_fingerprint() == graph_before
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        assert p.grad is None
    assert model.brain.group_count == 6 and model.brain.candidates == 16


def test_expanded_cpu_training_records_sidecar_calibration_optimizer_and_small_checkpoint(setup, tmp_path):
    _, _, data, graph_path, structure = setup
    data_path = tmp_path / "dataset.json"
    data_path.write_text(json.dumps(data))
    output = tmp_path / "run"
    graph_before = training.file_sha256(graph_path / "graph.npz")
    result = training.train(data_path, graph_path, output, device="cpu", epochs=1, batch_size=2,
        lr=.0003, rule_lr=.003, rule_eps=1e-12, monitor_stories=1, monitor_every=2, progress_every=1,
        calibration_windows=8, threads=1, global_scale=.5, internal_steps=4, strict_full_graph=False,
        structure_path=structure, trainable_susceptibility=True, probe_steps=4,
        minimum_feature_effect=1e-5, minimum_action_tv=1e-10)
    assert result["global_step"] == 4 and result["optimizer_steps"] > 0
    assert result["test_evaluated"] is False
    assert training.file_sha256(graph_path / "graph.npz") == graph_before
    calibration = json.loads((output / "calibration.json").read_text())
    assert calibration["plasticity_enabled"] and not calibration["activity_scales_plasticity_enabled"]
    assert set(calibration["story_ids"]) == {"a", "b", "c", "d"}
    payload = torch.load(output / "last.pt", weights_only=True)
    protocol = payload["protocol"]
    assert protocol["structure_path"] == str(structure.resolve())
    assert protocol["structure_sha256"] == training.file_sha256(structure)
    assert protocol["brain_trainable_parameters"] == 34 * 6 + 16
    assert protocol["trainable_parameters"] == 34 * 6 + 16 + 2570
    assert protocol["config"]["rule_lr"] == .003 and protocol["config"]["rule_eps"] == 1e-12
    assert protocol["config"]["lr"] == .0003 and protocol["config"]["feature_calibration_fast_enabled"]
    assert payload["parameters"]["brain.slow_susceptibility"].shape == (16,)
    assert payload["parameters"]["brain.feedback_modulation"].shape == (6, 28)
    assert payload["parameters"]["readout.weight"].shape == (10, 256)
    assert sum(t.numel() for t in payload["parameters"].values()) == 2790
    assert all(torch.isfinite(t).all() for t in payload["parameters"].values())
    assert (output / "last.pt").stat().st_size < 250000
    assert payload["inference_only"] and not payload["training_resume_supported"]
