"""Leak-free small readouts, row alignment, numerical floors, and controls."""
import json

import numpy as np
import pytest
import torch

from fly_wordbrain.memory_probe_readouts import fit_probe


def classification_data(classes=2, copies=32):
    labels = np.tile(np.arange(classes), 3 * copies).astype(np.int64)
    features = np.zeros((len(labels), 256), dtype=np.float32)
    features[np.arange(len(labels)), labels] = 1
    splits = np.repeat(["train", "val", "test"], classes * copies)
    return features, labels, splits


def test_linear_ten_way_alignment_confusion_predictions_and_parameter_count():
    features, labels, splits = classification_data(10, 12)
    result = fit_probe(features, labels, splits, 10, max_epochs=45)
    assert result["trainable_parameters"] == 2570
    assert result["receipt"]["normalization"]["active_dimensions"] == 10
    for name, metric in result["metrics"].items():
        assert metric["accuracy"] == 1
        assert metric["chance_accuracy"] == .1
        assert metric["majority_accuracy"] == pytest.approx(.1)
        prediction = result["predictions"][name]
        np.testing.assert_array_equal(labels[prediction["row_indices"]], prediction["labels"])
        matrix = np.zeros((10, 10), dtype=np.int64)
        np.add.at(matrix, (prediction["labels"], prediction["predicted_labels"]), 1)
        np.testing.assert_array_equal(matrix, metric["confusion"])
        assert matrix.sum() == metric["count"]
        np.testing.assert_allclose(np.sum(prediction["probabilities"], axis=1), 1, atol=1e-6)
    json.dumps(result, allow_nan=False)


def test_test_changes_cannot_affect_normalization_weights_or_epoch_selection():
    x, y, split = classification_data()
    first = fit_probe(x, y, split, 2, max_epochs=15)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[split == "test"] = 1e4
    changed_y[split == "test"] = 1 - changed_y[split == "test"]
    second = fit_probe(changed_x, changed_y, split, 2, max_epochs=15)
    assert first["history"] == second["history"]
    assert first["selected_epoch"] == second["selected_epoch"]
    for key in ("normalization", "selected_parameters_sha256"):
        assert first["receipt"][key] == second["receipt"][key]
    assert first["predictions"]["val"] == second["predictions"]["val"]
    assert first["predictions"]["test"] != second["predictions"]["test"]


def test_train_only_floor_masks_constant_and_numerical_zero_even_when_heldout_varies():
    x, y, split = classification_data()
    x[:, 2] = 100
    x[:, 3] = y * 1e-9
    x[split != "train", 4] = y[split != "train"] * 100
    result = fit_probe(x, y, split, 2, max_epochs=2)
    stats = result["receipt"]["normalization"]
    expected = x[split == "train"].astype(np.float64)
    np.testing.assert_array_equal(stats["mean"], expected.mean(0))
    np.testing.assert_array_equal(stats["raw_std"], expected.std(0))
    assert stats["effective_std_floor"] == .5 * 1e-4
    assert stats["active_dimensions"] == 2
    assert not any(stats["active_mask"][2:])


def test_shuffled_train_label_control_is_fixed_and_preserves_true_validation_labels():
    x, y, split = classification_data(copies=64)
    real = fit_probe(x, y, split, 2, max_epochs=35)
    first = fit_probe(x, y, split, 2, max_epochs=35, shuffle_train_labels=True)
    second = fit_probe(x, y, split, 2, max_epochs=35, shuffle_train_labels=True)
    assert first == second
    assert first["receipt"]["changed_train_labels"] > 0
    permutation = first["receipt"]["train_label_permutation"]
    train_y = y[split == "train"]
    np.testing.assert_array_equal(np.bincount(train_y[permutation]), np.bincount(train_y))
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(first["predictions"][name]["labels"], y[split == name])
    assert real["metrics"]["test"]["accuracy"] == 1
    # With just two repeated feature vectors, a small accidental correlation
    # in the fixed shuffle can preserve argmax while destroying confidence.
    assert first["metrics"]["test"]["cross_entropy"] > real["metrics"]["test"]["cross_entropy"] + .2


def test_optimizer_receives_the_permuted_training_targets(monkeypatch):
    import fly_wordbrain.memory_probe_readouts as readouts
    x, y, split = classification_data()
    seen = []
    original = readouts.F.cross_entropy
    def inspect_training(logits, targets, *args, **kwargs):
        if torch.is_grad_enabled() and logits.requires_grad:
            seen.extend(targets.tolist())
        return original(logits, targets, *args, **kwargs)
    monkeypatch.setattr(readouts.F, "cross_entropy", inspect_training)
    result = fit_probe(x, y, split, 2, shuffle_train_labels=True, max_epochs=1, batch_size=16)
    train_labels = y[split == "train"][result["receipt"]["train_label_permutation"]]
    order = torch.randperm(len(train_labels), generator=torch.Generator().manual_seed(0)).numpy()
    np.testing.assert_array_equal(seen, train_labels[order])


def test_mlp_binary_nonlinear_history_control_and_seed_restoration():
    points = np.tile(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32), (48, 1))
    x = np.zeros((len(points), 256), dtype=np.float32)
    x[:, :2] = points
    y = (points[:, 0] != points[:, 1]).astype(np.int64)
    split = np.repeat(["train", "val", "test"], len(points) // 3)
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    result = fit_probe(x, y, split, 2, architecture="mlp", max_epochs=100)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert result["trainable_parameters"] == 4146
    assert result["metrics"]["test"]["accuracy"] == 1
    assert result["receipt"]["hidden_dimensions"] == 16


def test_early_stopping_preserves_initial_epoch_when_no_sufficient_improvement():
    x, y, split = classification_data()
    result = fit_probe(x, y, split, 2, min_delta=10, patience=3, max_epochs=20)
    assert result["selected_epoch"] == 0
    assert result["epochs_run"] == 3 and result["stopped_early"]
    assert result["metrics"]["val"]["cross_entropy"] == result["history"][0]["validation_cross_entropy"]


@pytest.mark.parametrize("problem", ["width", "nan", "float_labels", "bad_label", "missing_split", "unaligned", "epochs"])
def test_invalid_or_unaligned_data_is_rejected(problem):
    x, y, split = classification_data()
    kwargs = {}
    if problem == "width": x = x[:, :16]
    if problem == "nan": x[0, 0] = np.nan
    if problem == "float_labels": y = y.astype(float)
    if problem == "bad_label": y[0] = 2
    if problem == "missing_split": split[split == "test"] = "val"
    if problem == "unaligned": split = split[:-1]
    if problem == "epochs": kwargs["max_epochs"] = 151
    with pytest.raises(ValueError):
        fit_probe(x, y, split, 2, **kwargs)
