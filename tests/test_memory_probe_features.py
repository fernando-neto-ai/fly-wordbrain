"""Checkpoint restoration and causal extraction, using a small real CSR graph."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fly_wordbrain.fast_structure import (ARRAY_KEYS, ORIGINAL_GROUPS, array_digest,
    file_sha256, original_selection_sha256, save_structure)
from fly_wordbrain.feedback_model import FeedbackActionBrain, FeedbackSelector
from fly_wordbrain.feedback_train import save_checkpoint, source_hashes
from fly_wordbrain.memory_probe_features import (candidate_neuron_indices,
    extract_batch, load_frozen_model)


def write_graph(directory, expanded):
    n = 64
    pre = np.arange(n, dtype=np.int64)
    post = 56 + pre % 8
    weight = (.2 + (pre % 7) * .1).astype(np.float32)
    weight[pre % 3 == 0] *= -1
    incoming = np.lexsort((pre, post))
    graph = dict(ids=np.arange(n), ptr=np.arange(n + 1), post=post, weight=weight,
        incoming_ptr=np.r_[0, np.cumsum(np.bincount(post, minlength=n))],
        incoming_pre=pre[incoming], incoming_weight=weight[incoming], incoming_edge_ids=incoming,
        retina=np.arange(56), descending=np.arange(56, n), candidate_edge_ids=np.arange(4),
        candidate_pre=pre[:4], candidate_post=post[:4], candidate_group=np.arange(4))
    np.savez(directory / "graph.npz", **graph)
    (directory / "metadata.json").write_text(json.dumps({"toy_graph": True}))
    if not expanded:
        return None
    arrays = dict(candidate_edge_ids=np.arange(n),
        candidate_group=np.r_[np.arange(4), 4 + np.arange(n-4) % 2],
        candidate_pre=pre, candidate_post=post, candidate_weight=weight)
    metadata = dict(schema=1, kind="fast_structure", graph_sha256=file_sha256(directory / "graph.npz"),
        original_selection_sha256=original_selection_sha256(graph),
        selection_array_sha256=array_digest([arrays[key] for key in ARRAY_KEYS]),
        group_names=list(ORIGINAL_GROUPS) + ["toy_extra_even", "toy_extra_odd"],
        neurons=n, edges=n, candidate_edges=n, original_candidate_edges=4)
    path = directory / "expanded.npz"
    save_structure(path, arrays, metadata)
    return path


def batch():
    observed = torch.tensor([[4, 5, 6, 7, 8, 9, 10], [11, 10, 9, 8, 7, 6, 5]])
    probability = .85 * torch.arange(10, 0, -1, dtype=torch.float32) / 55.
    return SimpleNamespace(previous=torch.cat((torch.full((2, 2), 2), observed[:, :6]), dim=1),
        current=torch.cat((torch.full((2, 1), 2), observed), dim=1),
        candidates=torch.tensor([1, 3, 4, 5, 6, 7, 8, 9, 10, 11]).repeat(2, 8, 1),
        probabilities=probability.repeat(2, 8, 1), observed_ids=observed)


@pytest.fixture(params=[False, True], ids=["original", "expanded"])
def checkpoint(tmp_path, request):
    torch.set_num_threads(1)
    expanded = request.param
    structure = write_graph(tmp_path, expanded)
    brain = FeedbackActionBrain(tmp_path, 20, .5, strict_full_graph=False, seed=0,
        input_gain=10., internal_steps=4, structure_path=structure, trainable_susceptibility=expanded)
    brain.set_activity_scales(.1, .2)
    model = FeedbackSelector(brain, feature_mean=torch.linspace(-.1, .1, 256),
                            feature_std=torch.linspace(.01, 1., 256))
    # Every persisted parameter differs from constructor initialization.
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.add_(.003 * (index + 1))
    protocol = dict(top_k=10, observed_words=7, window_words=8, seed=0,
        vocabulary=[str(i) for i in range(20)], graph_sha256=file_sha256(tmp_path / "graph.npz"),
        structure_sha256=file_sha256(structure) if expanded else None,
        source_sha256=source_hashes(), model={"metadata": model.metadata()},
        trainable_parameters=model.trainable_parameter_count(), dataset_sha256="toy")
    path = tmp_path / "frozen.pt"
    save_checkpoint(path, model, protocol, 8, 1, "monitor_only")
    return SimpleNamespace(path=path, graph=tmp_path, structure=structure, source=model,
                           count=2838 if expanded else 2706)


def restore(checkpoint):
    return load_frozen_model(checkpoint.path, checkpoint.graph, checkpoint.structure, "cpu")


def test_restore_every_parameter_and_calibration_exactly_then_freeze(checkpoint):
    frozen, receipt = restore(checkpoint)
    source = dict(checkpoint.source.named_parameters())
    assert set(dict(frozen.named_parameters())) == set(source)
    for name, value in frozen.named_parameters():
        assert torch.equal(value, source[name])
        assert not value.requires_grad
    for parent_name, name in ((None, "feature_mean"), (None, "feature_std"),
                              ("brain", "pre_scale"), ("brain", "post_scale")):
        loaded = frozen if parent_name is None else getattr(frozen, parent_name)
        original = checkpoint.source if parent_name is None else getattr(checkpoint.source, parent_name)
        assert torch.equal(getattr(loaded, name), getattr(original, name))
    assert all(not module.training for module in frozen.modules())
    assert frozen.features_calibrated
    assert receipt["parameter_count_restored"] == checkpoint.count
    assert receipt["trainable_parameters"] == 0
    assert receipt["checkpoint_step"] == 8 and receipt["checkpoint_role"] == "monitor_only"
    assert receipt["checkpoint_sha256"] == file_sha256(checkpoint.path)
    assert frozen.brain.verify_frozen() == checkpoint.source.brain.verify_frozen()


@pytest.mark.parametrize("plasticity", [True, False], ids=["fast_on", "fast_off"])
def test_extraction_matches_unmodified_features_and_state_without_labels(checkpoint, plasticity):
    frozen, _ = restore(checkpoint)
    class Guard:
        def __init__(self):
            self.__dict__.update(vars(batch()))
        @property
        def targets(self):
            raise AssertionError("Hidden target accessed during extraction")
    inputs = Guard()
    indices = candidate_neuron_indices(frozen, "all_nonretinal")[::-1]
    before_parameters = {name: value.clone() for name, value in frozen.named_parameters()}
    before_scales = [value.clone() for value in (frozen.feature_mean, frozen.feature_std,
                                               frozen.brain.pre_scale, frozen.brain.post_scale)]
    immutable_identity = frozen.brain.verify_frozen()
    actual = extract_batch(frozen, inputs, selected_indices=indices, plasticity=plasticity)
    with torch.no_grad():
        expected, state = frozen.features(inputs, plasticity=plasticity)
    np.testing.assert_array_equal(actual["final_pooled"], expected.numpy())
    np.testing.assert_array_equal(actual["final_activity"], state.h.numpy())
    np.testing.assert_array_equal(actual["final_selected"], state.h.numpy()[:, indices])
    np.testing.assert_array_equal(actual["selected_indices"], indices)
    assert actual["temporal_selected"].shape == (2, 8, 8)
    assert actual["temporal_pooled"].shape == (2, 8, 256)
    assert actual["metadata"]["final_observations"] == 7
    assert actual["metadata"]["final_fast_abs_max"] > 0 if plasticity else actual["metadata"]["final_fast_abs_max"] == 0
    replay = extract_batch(frozen, inputs, selected_indices=indices, plasticity=plasticity)
    for key in ("final_pooled", "final_activity", "temporal_pooled", "temporal_selected"):
        np.testing.assert_array_equal(actual[key], replay[key])
    for name, value in frozen.named_parameters():
        assert torch.equal(value, before_parameters[name])
    for old, new in zip(before_scales, (frozen.feature_mean, frozen.feature_std,
                                      frozen.brain.pre_scale, frozen.brain.post_scale)):
        assert torch.equal(old, new)
    assert frozen.brain.verify_frozen() == immutable_identity


def test_capture_uses_causal_predict_observe_api_under_no_grad(checkpoint, monkeypatch):
    frozen, _ = restore(checkpoint)
    predict, observe = frozen.brain.predict, frozen.brain.observe
    calls = []
    def traced_predict(previous, current, candidate_ids, probabilities, state, plasticity=True):
        assert not torch.is_grad_enabled()
        assert not state.pending
        if state.observations == 0:
            assert torch.all(state.h == 0) and torch.all(state.fast == 0) and torch.all(state.eligibility == 0)
        calls.append(("predict", state.observations, plasticity))
        return predict(previous, current, candidate_ids, probabilities, state, plasticity=plasticity)
    def traced_observe(state, ids, probabilities, observed, position):
        assert not torch.is_grad_enabled() and state.pending
        calls.append(("observe", position, state.writing_enabled))
        return observe(state, ids, probabilities, observed, position)
    monkeypatch.setattr(frozen.brain, "predict", traced_predict)
    monkeypatch.setattr(frozen.brain, "observe", traced_observe)
    extract_batch(frozen, batch(), plasticity=False)
    expected = []
    for position in range(8):
        expected.append(("predict", position, False))
        if position < 7:
            expected.append(("observe", position + 1, False))
    assert calls == expected


@pytest.mark.parametrize("corruption", ["missing_parameter", "extra_parameter", "wrong_shape",
    "nan_parameter", "wrong_dtype", "invalid_activity_scale", "invalid_feature_scale", "wrong_source",
    "wrong_graph", "wrong_structure", "wrong_count", "not_calibrated"])
def test_reject_checkpoint_corruption(checkpoint, corruption):
    payload = copy.deepcopy(torch.load(checkpoint.path, map_location="cpu", weights_only=False))
    parameters = payload["parameters"]
    if corruption == "missing_parameter":
        del parameters["readout.bias"]
    elif corruption == "extra_parameter":
        parameters["invented"] = torch.zeros(1)
    elif corruption == "wrong_shape":
        parameters["readout.bias"] = torch.zeros(11)
    elif corruption == "nan_parameter":
        parameters["readout.bias"][0] = float("nan")
    elif corruption == "wrong_dtype":
        parameters["readout.bias"] = parameters["readout.bias"].double()
    elif corruption == "invalid_activity_scale":
        payload["pre_scale"][0] = 0
    elif corruption == "invalid_feature_scale":
        payload["feature_std"][0] = 0
    elif corruption == "wrong_source":
        payload["protocol"]["source_sha256"]["feedback_model.py"] = "wrong"
    elif corruption == "wrong_graph":
        payload["protocol"]["graph_sha256"] = "wrong"
    elif corruption == "wrong_structure":
        payload["protocol"]["structure_sha256"] = "wrong"
    elif corruption == "wrong_count":
        payload["protocol"]["trainable_parameters"] += 1
    elif corruption == "not_calibrated":
        payload["protocol"]["model"]["metadata"]["features_calibrated"] = False
    path = checkpoint.graph / (corruption + ".pt")
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_frozen_model(path, checkpoint.graph, checkpoint.structure, "cpu")


def test_candidate_selection_excludes_driven_cells_and_preserves_requested_order(checkpoint):
    frozen, _ = restore(checkpoint)
    np.testing.assert_array_equal(candidate_neuron_indices(frozen, "all_nonretinal"), np.arange(56, 64))
    default = candidate_neuron_indices(frozen)
    assert not np.intersect1d(default, frozen.brain.retina.numpy()).size
    with pytest.raises(ValueError, match="Unknown"):
        candidate_neuron_indices(frozen, "invented")
    for indices in ([0], [56, 56], [], [64], [56.5], [-1]):
        with pytest.raises(ValueError):
            extract_batch(frozen, batch(), selected_indices=indices)
    output = extract_batch(frozen, batch(), selected_indices=[63, 56, 60])
    np.testing.assert_array_equal(output["selected_indices"], [63, 56, 60])
    np.testing.assert_array_equal(output["final_selected"], output["final_activity"][:, [63, 56, 60]])


def test_extraction_rejects_trainable_parameters_or_training_mode(checkpoint):
    with pytest.raises(ValueError, match="frozen parameters"):
        extract_batch(checkpoint.source, batch())
    frozen, _ = restore(checkpoint)
    frozen.train()
    with pytest.raises(ValueError, match="eval mode"):
        extract_batch(frozen, batch())
