"""Live validation is held-out inference and cannot change training/selection."""
import copy
import json
import random

import numpy as np
import pytest
import torch

import fly_wordbrain.plastic_train as training
from test_plastic_brain import toy_graph
from test_plastic_train import VOCABULARY, bind_passed_probe, story


def prepare(tmp_path, train_count=4, val_count=2):
    torch.set_num_threads(1)
    toy_graph(tmp_path)
    dataset = {"vocabulary": VOCABULARY, "splits": {
        split: [story(split + "-" + str(i), (4 + i % 3, 5, 6)) for i in range(count)]
        for split, count in (("train", train_count), ("val", val_count), ("test", 2))}}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    calibration = tmp_path / "calibration.npz"
    np.savez(calibration, feature_mean=np.zeros(256, np.float32), feature_std=np.ones(256, np.float32),
             pre_scale=np.full(4, .1, np.float32), post_scale=np.full(4, .1, np.float32))
    bind_passed_probe(calibration, {
        "graph_sha256": training.file_sha256(tmp_path / "graph.npz"),
        "graph_metadata_sha256": training.file_sha256(tmp_path / "metadata.json"),
        "dataset_sha256": training.file_sha256(path), "calibration_npz_sha256": training.file_sha256(calibration),
        "train_only": True, "story_ids": ["train-0"], "global_scale": .5,
        "internal_steps": 4, "leak": .5, "input_gain": 20., "input_center": 0.})
    return path, calibration, dataset


def test_fixed_schedule_and_ordered_validation_identity():
    assert [i for i in range(130) if training.monitor_due(i, 64)] == [0, 1, 64, 128]
    rows = [story("val-z"), story("val-a")]
    identity = training.validation_identity(rows, "dataset-sha")
    assert identity["story_ids"] == ["val-z", "val-a"]
    assert identity == training.validation_identity(rows, "dataset-sha")
    assert identity["subset_sha256"] != training.validation_identity(rows[::-1], "dataset-sha")["subset_sha256"]
    assert identity["subset_sha256"] != training.validation_identity(rows, "changed-dataset")["subset_sha256"]


@pytest.mark.parametrize("raises", [False, True])
def test_monitor_preserves_rng_modes_parameters_gradients_and_optimizer(monkeypatch, raises):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(.5))
    model.train()
    model[1].eval()  # Preserve mixed child modes, not just the root mode.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(2, 1)).sum().backward()
    optimizer.step()
    modes = [m.training for m in model.modules()]
    parameters = [p.detach().clone() for p in model.parameters()]
    gradients = [p.grad.clone() for p in model.parameters()]
    state = copy.deepcopy(optimizer.state_dict())
    cpu_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()

    def evaluation(model, *args, **kwargs):
        model.eval()
        torch.rand(5)
        np.random.random(5)
        random.random()
        if raises:
            raise RuntimeError("diagnostic failure")
        return {"real_evaluation": True}, {}

    monkeypatch.setattr(training, "evaluate_stories", evaluation)
    if raises:
        with pytest.raises(RuntimeError, match="diagnostic failure"):
            training.evaluate_preserving_training(model, [], torch.zeros(1), torch.ones(1), arm="frozen", batch_size=1)
    else:
        training.evaluate_preserving_training(model, [], torch.zeros(1), torch.ones(1), arm="frozen", batch_size=1)
    assert [m.training for m in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert random.getstate() == python_rng
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_rng[0] and current_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(current_numpy[1], numpy_rng[1])
    for p, value, gradient in zip(model.parameters(), parameters, gradients):
        torch.testing.assert_close(p, value, rtol=0, atol=0)
        torch.testing.assert_close(p.grad, gradient, rtol=0, atol=0)
    for key, values in state["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0)
    assert optimizer.state_dict()["param_groups"] == state["param_groups"]


def test_real_snapshots_fixed_subset_checkpoint_parity_and_no_early_test(tmp_path, monkeypatch):
    path, calibration, dataset = prepare(tmp_path)
    output = tmp_path / "monitored"
    original = training.evaluate_stories

    def guarded(model, rows, *args, **kwargs):
        if rows[0]["id"].startswith("test"):
            locked = json.loads((output / "selections-locked.json").read_text())
            assert set(locked["selections"]) == set(training.ARMS)
        return original(model, rows, *args, **kwargs)

    monkeypatch.setattr(training, "evaluate_stories", guarded)
    result = training.train(path, tmp_path, calibration, output, device="cpu", epochs=1,
        batch_size=2, threads=1, strict_full_graph=False, monitor_stories=1,
        monitor_every=2, progress_every=1)
    snapshots = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    subset = [r for r in snapshots if r["scope"] == "monitor_subset"]
    assert len({r["subset_sha256"] for r in subset}) == 1
    assert all(r["story_ids"] == ["val-0"] and not r["selection_eligible"] for r in subset)
    assert all(r["target_count"] == 7 and r["target_counts"] == {"next_1": 4, "next_2": 3} for r in subset)
    for arm in training.ARMS:
        history = json.loads((output / arm / "validation-history.json").read_text())
        assert [(r["scope"], r["global_step"]) for r in history] == [
            ("monitor_subset", 0), ("monitor_subset", 1), ("monitor_subset", 2), ("full_validation", 2)]
        partial = output / arm / "partial.pt"
        model, payload = training.load_checkpoint(partial, tmp_path)
        assert payload["global_step"] == 2 and payload["inference_only"]
        assert not payload["training_resume_supported"] and not payload["selection_eligible"]
        assert payload["validation_head1_cross_entropy"] is None
        assert history[2]["checkpoint_sha256"] == training.file_sha256(partial)
        actual, _ = original(model, dataset["splits"]["val"][:1], torch.zeros(256), torch.ones(256), arm=arm, batch_size=2)
        assert history[2]["metrics"] == actual
        assert history[2]["head1_cross_entropy"] == actual["horizons"]["next_1"]["cross_entropy"]
        assert history[3]["selection_eligible"] and history[3]["story_count"] == 2
    # The default path produces exactly the same learned parameters and selected
    # full-validation values, and creates none of the optional live artifacts.
    monkeypatch.setattr(training, "evaluate_stories", original)
    legacy = tmp_path / "legacy"
    baseline = training.train(path, tmp_path, calibration, legacy, device="cpu", epochs=1,
        batch_size=2, threads=1, strict_full_graph=False)
    assert not (legacy / "validation.jsonl").exists()
    for arm in training.ARMS:
        left = torch.load(output / arm / "best.pt", weights_only=True)
        right = torch.load(legacy / arm / "best.pt", weights_only=True)
        assert result["arms"][arm]["selection"]["validation_head1_cross_entropy"] == baseline["arms"][arm]["selection"]["validation_head1_cross_entropy"]
        for name in left["parameters"]:
            torch.testing.assert_close(left["parameters"][name], right["parameters"][name], rtol=0, atol=0)
        assert not (legacy / arm / "partial.pt").exists()


def test_subset_scores_cannot_select_checkpoint_or_reset_patience(tmp_path, monkeypatch):
    path, calibration, _ = prepare(tmp_path, train_count=2)
    output = tmp_path / "selection"
    original = training.evaluate_stories
    full_calls, subset_calls = [], []

    def controlled(model, rows, *args, **kwargs):
        summary, arrays = original(model, rows, *args, **kwargs)
        if rows[0]["id"].startswith("val") and not (output / "selections-locked.json").exists():
            if len(rows) == 1:
                subset_calls.append(1)
                summary["horizons"]["next_1"]["cross_entropy"] = 100. - len(subset_calls)
            else:
                full_calls.append(1)
                summary["horizons"]["next_1"]["cross_entropy"] = 1. + len(full_calls)
        return summary, arrays

    monkeypatch.setattr(training, "evaluate_stories", controlled)
    result = training.train(path, tmp_path, calibration, output, device="cpu", arms=("frozen",),
        epochs=4, patience=1, batch_size=2, threads=1, strict_full_graph=False,
        monitor_stories=1, monitor_every=1)
    assert len(full_calls) == 2 and len(subset_calls) == 3
    assert result["arms"]["frozen"]["selection"]["epoch"] == 1
    assert result["arms"]["frozen"]["selection"]["validation_head1_cross_entropy"] == 2.
