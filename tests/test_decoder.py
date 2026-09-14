"""Readout isolation, honest held-out evaluation, and inference round-trip."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fly_wordbrain.decoder import (
    CountReferences, Decoder, Standardizer, _summary,
    fit_decoder, load_split, run, validate_split_membership,
)


def examples(story_id="train-story", repetitions=20):
    ids = np.asarray([2] + [4, 5] * repetitions + [3], dtype=np.int64)
    y = ids[1:]
    # An explicitly synthetic separable signal, not extracted connectome data.
    X = np.column_stack([y == 4, y == 5, y == 3]).astype(np.float32)
    X = X * 2 - 1
    data = {"X": X, "y": y, "current_ids": ids[:-1],
            "positions": np.arange(len(y)), "story_ids": np.full(len(y), story_id)}
    story = {"id": story_id, "word_ids": ids.tolist(),
             "target_mask": [False] + [True] * len(y), "words": []}
    return data, story


VOCABULARY = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "red", "blue"]


def test_standardizer_fits_only_training_and_handles_constant_feature():
    X = np.asarray([[0, 3], [2, 3]], dtype=np.float32)
    scaler = Standardizer.fit(X)
    np.testing.assert_array_equal(scaler.mean, [1, 3])
    np.testing.assert_array_equal(scaler.scale, [1, 1])
    np.testing.assert_array_equal(scaler.transform(X), [[-1, 0], [1, 0]])
    # A shifted holdout must remain shifted, rather than being centered itself.
    np.testing.assert_array_equal(scaler.transform(np.asarray([[101, 9]])), [[100, 6]])
    np.testing.assert_array_equal(scaler.mean, [1, 3])


def test_readout_learns_features_without_using_current_token_ids(tmp_path):
    torch.set_num_threads(1)
    train, _ = examples()
    val, _ = examples("validation", 5)
    decoder, curves = fit_decoder(train, val, VOCABULARY, epochs=25, l2_values=[0], learning_rate=.05)
    metrics = decoder.evaluate(val["X"], val["y"])
    assert metrics["accuracy"] > .95
    assert curves[0]["epochs"][-1]["train"]["cross_entropy"] < curves[0]["epochs"][0]["train"]["cross_entropy"]
    checkpoint = tmp_path / "checkpoint.pt"
    decoder.save(checkpoint)
    loaded = Decoder.load(checkpoint)
    np.testing.assert_array_equal(loaded.logits(val["X"]), decoder.logits(val["X"]))
    np.testing.assert_allclose(loaded.predict_proba(val["X"]).sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(loaded.predict(val["X"][0]), loaded.predict(val["X"][:1])[0])
    assert loaded.config["selected_epoch"] <= 25
    # Metadata never reaches the classifier: scrambling current IDs, positions
    # and story IDs cannot change fitted weights given the same X and y.
    scrambled = {k: v.copy() for k, v in train.items()}
    scrambled["current_ids"][:] = 1
    scrambled["positions"][:] = 999
    scrambled["story_ids"][:] = "scrambled"
    alternative, _ = fit_decoder(scrambled, val, VOCABULARY, epochs=25, l2_values=[0], learning_rate=.05)
    np.testing.assert_array_equal(alternative.logits(val["X"]), decoder.logits(val["X"]))


def test_test_distribution_cannot_change_scaling_or_selected_checkpoint(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    dataset = {"vocabulary": VOCABULARY, "splits": {}}
    splits = {}
    for split in ["train", "val", "test"]:
        data, story = examples(split, 4)
        splits[split] = data
        dataset["splits"][split] = [story]
        np.savez(features / (split + ".npz"), **data)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    first = run(features, dataset_path, tmp_path / "a", epochs=3, l2_values=[0, .01])
    # Perturb only unseen test features; train/val choices and scaling must match.
    splits["test"]["X"] = -splits["test"]["X"] * 20
    np.savez(features / "test.npz", **splits["test"])
    second = run(features, dataset_path, tmp_path / "b", epochs=3, l2_values=[0, .01])
    a, b = Decoder.load(tmp_path / "a"), Decoder.load(tmp_path / "b")
    np.testing.assert_array_equal(a.scaler.mean, b.scaler.mean)
    np.testing.assert_array_equal(a.scaler.scale, b.scaler.scale)
    for name, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[name])
    assert a.config["selected_epoch"] == b.config["selected_epoch"]
    assert a.config["l2"] == b.config["l2"]
    assert first["primary"]["metrics"]["test"]["cross_entropy"] != second["primary"]["metrics"]["test"]["cross_entropy"]
    curves = json.loads((tmp_path / "a" / "training-curves.json").read_text())
    assert all("test" not in epoch for candidate in curves for epoch in candidate["epochs"])


def test_count_references_fit_train_and_back_off_for_unseen_context():
    train = {"current_ids": np.asarray([2, 4, 2, 4]), "y": np.asarray([4, 5, 4, 5])}
    references = CountReferences(train, 6)
    test = {"current_ids": np.asarray([0, 2, 4]), "y": np.asarray([1, 4, 5])}
    before = references.record()
    metrics = references.evaluate(test)
    assert references.record() == before
    assert metrics["smoothed_bigram"]["cross_entropy"] < metrics["unigram"]["cross_entropy"]
    assert metrics["smoothed_bigram"]["known_target_examples"] == 2
    unseen = {"current_ids": np.asarray([0]), "y": np.asarray([1])}
    unseen_metrics = references.evaluate(unseen)
    assert unseen_metrics["unigram"]["cross_entropy"] == unseen_metrics["smoothed_bigram"]["cross_entropy"]
    assert np.isfinite(unseen_metrics["smoothed_bigram"]["cross_entropy"])


def test_unknown_and_lexical_metrics_have_explicit_denominators():
    metrics = _summary(np.asarray([7., 2., 3.]), np.asarray([1, 3, 4]), np.asarray([1, 3, 4]))
    assert metrics["known_target_examples"] == 2
    assert metrics["known_target_cross_entropy"] == 2.5
    assert metrics["known_lexical_examples"] == 1
    assert metrics["known_lexical_cross_entropy"] == 3
    assert metrics["accuracy"] == 1


def test_alignment_and_mask_violations_rejected_before_training(tmp_path):
    data, story = examples()
    dataset = {"splits": {"train": [story]}}
    validate_split_membership(data, dataset, "train")
    bad = {key: value.copy() for key, value in data.items()}
    bad["y"][0] = 5
    with pytest.raises(ValueError, match="alignment"):
        validate_split_membership(bad, dataset, "train")
    story["target_mask"][1] = False
    with pytest.raises(ValueError, match="masked"):
        validate_split_membership(data, dataset, "train")
    path = tmp_path / "bad.npz"
    bad["y"] = bad["y"].astype(np.float32)
    np.savez(path, **bad)
    with pytest.raises(ValueError, match="integers"):
        load_split(path, len(VOCABULARY))


def test_run_rejects_overlapping_story_splits(tmp_path):
    _, story = examples()
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"vocabulary": VOCABULARY,
                               "splits": {split: [story] for split in ["train", "val", "test"]}}))
    with pytest.raises(ValueError, match="overlap"):
        run(tmp_path, path, tmp_path / "out", epochs=1, l2_values=[0])
