"""Small real-graph tests for causal batching, BPTT, and compact checkpoints."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fly_wordbrain.plastic_brain import PlasticBrain
import fly_wordbrain.plastic_train as training
from test_plastic_brain import toy_graph


VOCABULARY = ["<pad>", "<unk>", "<bos>", "<eos>", "a", "b", "c", "d"]


def story(name="s", words=(4, 5, 6), eos=True):
    ids = [2] + list(words) + ([3] if eos else [])
    return {"id": name, "word_ids": ids, "target_mask": [False] + [True] * (len(ids) - 1),
            "words": [VOCABULARY[i] for i in words]}


@pytest.fixture
def graph(tmp_path):
    torch.set_num_threads(1)
    toy_graph(tmp_path)
    return tmp_path


def model(graph):
    torch.manual_seed(17)
    brain = PlasticBrain(graph, 8, .5, internal_steps=4, input_gain=20., strict_full_graph=False)
    brain.set_activity_scales(.1, .1)
    return training.PlasticLanguageModel(brain, 8)


def run(net, stories):
    batch = training.collate_stories(stories)
    return training.rollout_batch(net, batch, torch.zeros(256), torch.ones(256))


def bind_passed_probe(calibration, metadata):
    metadata["source_sha256"] = {name: training.file_sha256(Path(training.__file__).with_name(name))
                                  for name in training.CALIBRATION_SOURCES}
    probe = {"checks": {key: True for key in training.PROBE_CHECKS},
        "calibration_npz_sha256": training.file_sha256(calibration),
        "config": {key: metadata[key] for key in ("global_scale", "internal_steps", "leak", "input_gain", "input_center")},
        "provenance": {key: metadata[key] for key in ("graph_sha256", "graph_metadata_sha256", "dataset_sha256",
            "train_only", "story_ids", "source_sha256")}}
    path = calibration.with_name("probe.json")
    path.write_text(json.dumps(probe))
    metadata.update(probe_passed=True, probe_file="probe.json", probe_sha256=training.file_sha256(path))
    calibration.with_suffix(".json").write_text(json.dumps(metadata))


def test_two_future_targets_and_masked_tail_are_causal(graph):
    batch = training.collate_stories([story()])
    assert batch.previous.tolist() == [[2, 2, 4, 5]]
    assert batch.current.tolist() == [[2, 4, 5, 6]]
    assert batch.targets.tolist() == [[[4, 5], [5, 6], [6, 3], [3, 0]]]
    assert batch.target_mask.tolist() == [[[True, True], [True, True], [True, True], [True, False]]]
    net = model(graph)
    original = run(net, [story()])[1]
    changed = run(net, [story(words=(4, 7, 6))])[1]
    # The second label of row0 and first label of row1 changed. The network has
    # not observed that word yet, so BOTH same-row heads remain unchanged.
    torch.testing.assert_close(original[:, :2], changed[:, :2], rtol=0, atol=0)
    first = training.rollout_batch(net, batch, torch.zeros(256), torch.ones(256))[0]
    batch.targets[~batch.target_mask] = 7
    second = training.rollout_batch(net, batch, torch.zeros(256), torch.ones(256))[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_padding_freezes_activity_and_fast_state_and_preserves_valid_outputs(graph):
    net = model(graph)
    short, long = story("short", (4,)), story("long", (6, 4, 5, 7))
    _, batched, _, state = run(net, [short, long])
    _, alone, _, single_state = run(net, [short])
    torch.testing.assert_close(batched[0, :alone.shape[1]], alone[0])
    torch.testing.assert_close(state.h[0], single_state.h[0])
    torch.testing.assert_close(state.fast[0], single_state.fast[0])


def test_full_story_gradients_reach_early_fast_state(graph):
    net = model(graph)
    captured = []
    original_step = net.step

    def record(*args, **kwargs):
        features, state = original_step(*args, **kwargs)
        captured.append(state.fast)
        return features, state

    net.step = record
    _, logits, _, _ = run(net, [story(words=(4, 5, 6, 7))])
    gradient = torch.autograd.grad(logits[:, -1].square().sum(), captured[0])[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize("arm", training.ARMS)
def test_only_declared_parameters_update(graph, arm):
    net = model(graph)
    optimizer, counts = training.configure_arm(net, arm)
    assert counts == {"readout": 4112, "shared_rule": 20, "trainable": 4132 if arm == "learned_fast" else 4112}
    graph_hash = net.brain.verify_frozen()
    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    batch = training.collate_stories([story(), story("other", (7, 4, 5))])
    loss, _, _, _ = training.rollout_batch(net, batch, torch.zeros(256), torch.ones(256),
                                          plasticity_override=arm != "frozen")
    loss.backward()
    optimizer.step()
    assert not torch.equal(net.readout.weight, before["readout.weight"])
    rule_changed = []
    for name, p in net.named_parameters():
        if name.startswith("brain."):
            rule_changed.append(not torch.equal(p, before[name]))
            if arm != "learned_fast":
                assert p.grad is None
    assert any(rule_changed) == (arm == "learned_fast")
    assert net.brain.verify_frozen() == graph_hash


@pytest.mark.parametrize("arm", training.ARMS)
def test_compact_checkpoint_roundtrip_and_graph_binding(graph, arm):
    net = model(graph)
    training.configure_arm(net, arm)
    scales = {"feature_mean": np.zeros(256, np.float32), "feature_std": np.ones(256, np.float32),
              "pre_scale": np.full(4, .1, np.float32), "post_scale": np.full(4, .1, np.float32)}
    config = {"arm": arm, "model_kwargs": {"global_scale": .5, "internal_steps": 4,
        "input_gain": 20., "strict_full_graph": False}, "provenance": {
            "graph_sha256": training.file_sha256(graph / "graph.npz"),
            "graph_metadata_sha256": training.file_sha256(graph / "metadata.json"),
            "source_sha256": training._source_hashes()}}
    path = graph / "checkpoint.pt"
    training.save_checkpoint(path, net, scales, config, VOCABULARY, 3, 2.)
    restored, payload = training.load_checkpoint(path, graph)
    assert set(payload) == {"schema", "parameters", "scales", "config", "vocabulary", "epoch", "validation_head1_cross_entropy"}
    assert set(payload["parameters"]) == set(dict(net.named_parameters()))
    assert all(t.layout == torch.strided for t in payload["parameters"].values())
    torch.testing.assert_close(run(net, [story()])[1], run(restored, [story()])[1], rtol=0, atol=0)
    batch = training.collate_stories([story()])
    explicit = training.rollout_batch(restored, batch, torch.zeros(256), torch.ones(256),
                                      plasticity_override=arm != "frozen")
    default = run(restored, [story()])
    torch.testing.assert_close(default[1], explicit[1], rtol=0, atol=0)
    torch.testing.assert_close(default[3].fast, explicit[3].fast, rtol=0, atol=0)
    if arm == "frozen":
        assert torch.count_nonzero(default[3].fast) == 0
    (graph / "metadata.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="graph provenance"):
        training.load_checkpoint(path, graph)


def test_calibration_rejects_heldout_story_identity(graph):
    dataset = {"vocabulary": VOCABULARY, "splits": {s: [story(s)] for s in ("train", "val", "test")}}
    dataset_path = graph / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    path = graph / "calibration.npz"
    np.savez(path, feature_mean=np.zeros(256, np.float32), feature_std=np.ones(256, np.float32),
             pre_scale=np.ones(4, np.float32), post_scale=np.ones(4, np.float32))
    metadata = {"graph_sha256": training.file_sha256(graph / "graph.npz"),
        "graph_metadata_sha256": training.file_sha256(graph / "metadata.json"),
        "dataset_sha256": training.file_sha256(dataset_path), "calibration_npz_sha256": training.file_sha256(path),
        "train_only": True, "story_ids": ["test"], "global_scale": .5, "internal_steps": 4,
        "leak": .5, "input_gain": 20., "input_center": 0.}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="training stories only"):
        training.load_calibration(path, graph, dataset_path, dataset)


def test_complete_toy_training_locks_all_selections_before_test(graph, monkeypatch):
    dataset = {"vocabulary": VOCABULARY, "splits": {
        s: [story(s + "-1"), story(s + "-2", (7, 6, 4))] for s in ("train", "val", "test")}}
    dataset_path = graph / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    calibration = graph / "calibration.npz"
    np.savez(calibration, feature_mean=np.zeros(256, np.float32), feature_std=np.ones(256, np.float32),
             pre_scale=np.full(4, .1, np.float32), post_scale=np.full(4, .1, np.float32))
    bind_passed_probe(calibration, {
        "graph_sha256": training.file_sha256(graph / "graph.npz"),
        "graph_metadata_sha256": training.file_sha256(graph / "metadata.json"),
        "dataset_sha256": training.file_sha256(dataset_path),
        "calibration_npz_sha256": training.file_sha256(calibration),
        "train_only": True, "story_ids": ["train-1"], "global_scale": .5,
        "internal_steps": 4, "leak": .5, "input_gain": 20., "input_center": 0.})
    output = graph / "run"
    original = training.evaluate_stories
    test_calls = []

    def guarded(model, stories, *args, **kwargs):
        if stories[0]["id"].startswith("test"):
            locked = json.loads((output / "selections-locked.json").read_text())
            assert set(locked["selections"]) == set(training.ARMS)
            test_calls.append(kwargs.get("intervention"))
        return original(model, stories, *args, **kwargs)

    monkeypatch.setattr(training, "evaluate_stories", guarded)
    result = training.train(dataset_path, graph, calibration, output, device="cpu", epochs=1,
                            batch_size=2, threads=1, gradient_clip=.05, strict_full_graph=False)
    assert set(result["arms"]) == set(training.ARMS)
    assert result["protocol"]["threads"] == 1
    assert result["protocol"]["gradient_clip"] == .05
    first_history = json.loads((output / "frozen" / "history.json").read_text())
    assert first_history[0]["clipped_update_fraction"] > 0
    assert len(test_calls) == 6  # Three baselines + frozen activity + two learned interventions.
    for arm in training.ARMS:
        data = np.load(output / arm / "test-predictions.npz")
        assert data["target_mask"].sum(0).tolist() == [8, 6]
        assert result["arms"][arm]["test"]["second_half"]["horizons"]["next_1"]["examples"] == 4
        assert (output / arm / "best.pt").is_file()
        assert (output / arm / "latest.pt").is_file()


def test_calibration_source_drift_and_failed_probe_are_rejected(graph):
    dataset = {"vocabulary": VOCABULARY, "splits": {s: [story(s)] for s in ("train", "val", "test")}}
    dataset_path = graph / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    calibration = graph / "calibration.npz"
    np.savez(calibration, feature_mean=np.zeros(256, np.float32), feature_std=np.ones(256, np.float32),
             pre_scale=np.ones(4, np.float32), post_scale=np.ones(4, np.float32))
    metadata = {"graph_sha256": training.file_sha256(graph / "graph.npz"),
        "graph_metadata_sha256": training.file_sha256(graph / "metadata.json"),
        "dataset_sha256": training.file_sha256(dataset_path),
        "calibration_npz_sha256": training.file_sha256(calibration),
        "train_only": True, "story_ids": ["train"], "global_scale": .5, "internal_steps": 4,
        "leak": .5, "input_gain": 20., "input_center": 0.}
    bind_passed_probe(calibration, metadata)
    training.load_calibration(calibration, graph, dataset_path, dataset)
    metadata["source_sha256"]["plastic_brain.py"] = "0" * 64
    calibration.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Calibration source changed"):
        training.load_calibration(calibration, graph, dataset_path, dataset)
    bind_passed_probe(calibration, metadata)
    probe_path = graph / "probe.json"
    probe = json.loads(probe_path.read_text())
    probe["checks"]["finite_rule_gradients"] = False
    probe_path.write_text(json.dumps(probe))
    metadata["probe_sha256"] = training.file_sha256(probe_path)
    calibration.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="wiring checks did not all pass"):
        training.load_calibration(calibration, graph, dataset_path, dataset)
    bind_passed_probe(calibration, metadata)
    (graph / "probe.json").write_text('{}')
    with pytest.raises(ValueError, match="checksum-bound passed probe receipt"):
        training.load_calibration(calibration, graph, dataset_path, dataset)


def test_interventions_change_only_midpoint_and_later_forecasts(graph):
    net = model(graph)
    stories = [story(words=(4, 5, 6, 7))]
    baseline, rows = training.evaluate_stories(net, stories, torch.zeros(256), torch.ones(256), arm="learned_fast")
    intervention, changed = training.evaluate_stories(net, stories, torch.zeros(256), torch.ones(256),
        arm="learned_fast", intervention="reset_activity")
    prefix = ~rows["second_half"]
    np.testing.assert_array_equal(rows["losses"][prefix], changed["losses"][prefix])
    assert not np.array_equal(rows["losses"][~prefix], changed["losses"][~prefix])
    assert baseline["second_half"]["horizons"]["next_1"]["examples"] == intervention["second_half"]["horizons"]["next_1"]["examples"]
