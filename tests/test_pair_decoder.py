"""Causal two-head targets, masked boundaries, independent fitting, and artifacts."""
import json

import numpy as np
import pytest
import torch

from fly_wordbrain import pair_decoder as pair
from fly_wordbrain.decoder import fit_decoder


VOCABULARY = ["<pad>", "<unk>", "<bos>", "<eos>", "red", "blue", "cat", "dog"]


def fixture_split(story_id="train"):
    ids = np.asarray([2, 4, 5, 6, 4, 7, 5, 4, 6, 7, 4, 3], dtype=np.int64)
    n = len(ids) - 1
    y = np.zeros((n, 2), dtype=np.int64)
    y[:, 0] = ids[1:]
    y[:-1, 1] = ids[2:]
    mask = np.ones((n, 2), dtype=bool)
    mask[-1, 1] = False
    rng = np.random.default_rng(33)
    X = rng.normal(size=(n, 256)).astype(np.float32)
    data = {"X": X, "X_direct": X[:, ::-1].copy(), "y": y, "target_mask": mask,
            "positions": np.arange(n, dtype=np.int64), "current_ids": ids[:-1],
            "previous_ids": np.r_[2, ids[:-2]], "story_ids": np.full(n, story_id)}
    story = {"id": story_id, "word_ids": ids.tolist(),
             "target_mask": [False] + [True] * n, "words": ["red"] * (len(ids) - 2)}
    return data, story


def test_masked_tail_never_enters_head2_fit_or_metrics():
    torch.set_num_threads(1)
    train, _ = fixture_split()
    val, _ = fixture_split("val")
    decoder, _ = pair.fit_pair(train, val, VOCABULARY, epochs=2, l2_values=[0])
    poisoned = {key: value.copy() for key, value in train.items()}
    poisoned["y"][-1, 1] = 10**6  # Invalid class must never reach cross_entropy.
    poisoned["X"][-1] = 1e6
    changed, _ = pair.fit_pair(poisoned, val, VOCABULARY, epochs=2, l2_values=[0])
    for key, value in decoder.heads[1].model.state_dict().items():
        assert torch.equal(value, changed.heads[1].model.state_dict()[key])
    np.testing.assert_array_equal(decoder.heads[1].scaler.mean, changed.heads[1].scaler.mean)
    test = {key: value.copy() for key, value in val.items()}
    test["y"][-1, 1] = 10**6
    metrics, outputs = pair.evaluate_pair(decoder, test["X"], test)
    assert metrics["horizons"]["next_2"]["examples"] == len(test["y"]) - 1
    assert np.isnan(outputs["losses"][-1, 1]) and outputs["predictions"][-1, 1] == -1
    assert metrics["forecast_average"]["valid_forecasts"] == 2 * len(test["y"]) - 1


def test_head1_is_independent_of_head2_labels_and_matches_original_readout():
    torch.set_num_threads(1)
    train, _ = fixture_split()
    val, _ = fixture_split("val")
    first, _ = pair.fit_pair(train, val, VOCABULARY, seed=1, epochs=3, l2_values=[0, .01])
    changed_train = {key: value.copy() for key, value in train.items()}
    changed_val = {key: value.copy() for key, value in val.items()}
    for data in [changed_train, changed_val]:
        data["y"][data["target_mask"][:, 1], 1] = 7
    second, _ = pair.fit_pair(changed_train, changed_val, VOCABULARY, seed=1, epochs=3, l2_values=[0, .01])
    original, _ = fit_decoder({"X": train["X"], "y": train["y"][:, 0]},
                             {"X": val["X"], "y": val["y"][:, 0]},
                             VOCABULARY, seed=1, epochs=3, l2_values=[0, .01])
    np.testing.assert_array_equal(first.heads[0].logits(val["X"]), second.heads[0].logits(val["X"]))
    np.testing.assert_array_equal(first.heads[0].logits(val["X"]), original.logits(val["X"]))
    for key in ["seed", "l2", "selected_epoch", "validation_cross_entropy"]:
        assert first.heads[0].config[key] == original.config[key]
    assert not np.array_equal(first.heads[1].logits(val["X"]), second.heads[1].logits(val["X"]))


def test_pair_wrapper_roundtrip_and_no_head_teacher_forcing(tmp_path):
    torch.set_num_threads(1)
    train, _ = fixture_split()
    decoder, _ = pair.fit_pair(train, train, VOCABULARY, epochs=2, l2_values=[0])
    directory = tmp_path / "brain" / "seed-0"
    decoder.save(directory, {"fixture": "manifest"})
    pair.write_json(tmp_path / "pair.json", {"schema_version": 1, "primary": "brain/seed-0/pair.json"})
    restored = pair.PairDecoder.load(tmp_path)
    expected = decoder.logits(train["X"])
    np.testing.assert_array_equal(restored.logits(train["X"]), expected)
    assert restored.logits(train["X"][0]).shape == (2, len(VOCABULARY))
    np.testing.assert_allclose(restored.predict_proba(train["X"]).sum(axis=-1), 1, atol=1e-6)
    assert restored.checkpoint_dir == directory.resolve()
    assert restored.feature_manifest == {"fixture": "manifest"}
    before_head2 = restored.logits(train["X"])[:, 1].copy()
    with torch.no_grad():
        restored.heads[0].model.weight.fill_(999)
        restored.heads[0].model.bias.fill_(123)
    np.testing.assert_array_equal(restored.logits(train["X"])[:, 1], before_head2)


def test_alignment_rejects_missing_positions_or_overlapping_future():
    data, story = fixture_split()
    dataset = {"splits": {"train": [story]}}
    pair.validate_alignment(data, dataset, "train")
    omitted = {key: value[:-1] for key, value in data.items()}
    with pytest.raises(ValueError, match="omit"):
        pair.validate_alignment(omitted, dataset, "train")
    copied_current = {key: value.copy() for key, value in data.items()}
    copied_current["y"][0, 0] = data["current_ids"][0]
    with pytest.raises(ValueError, match="future-target"):
        pair.validate_alignment(copied_current, dataset, "train")
    tail = {key: value.copy() for key, value in data.items()}
    tail["target_mask"][-1, 1] = True
    tail["y"][-1, 1] = 4
    with pytest.raises(ValueError, match="future-target"):
        pair.validate_alignment(tail, dataset, "train")


def test_count_forecasts_use_observed_pair_and_never_true_next1_for_head2():
    train = {"previous_ids": np.asarray([4, 4, 5, 5]), "current_ids": np.asarray([6, 6, 6, 6]),
             "y": np.asarray([[4, 7], [4, 7], [5, 6], [5, 6]]), "target_mask": np.ones((4, 2), bool)}
    counts = pair.PairCountReferences(train, 8)
    changed = {key: value.copy() for key, value in train.items()}
    changed["y"][:, 0] = [1, 2, 3, 7]
    new = pair.PairCountReferences(changed, 8)
    assert counts.heads[1].record() == new.heads[1].record()
    second = counts.heads[1]
    assert second.probability(4, 6, 7) > second.probability(5, 6, 7)
    for p, c in [(4, 6), (5, 6), (0, 0)]:
        assert sum(second.probability(p, c, target) for target in range(8)) == pytest.approx(1)
    assert second.probability(0, 0, 1) == second.unigram[1]
    # Scoring labels affect loss only. Predictions cannot use either future label.
    original_predictions = second.predictions(np.asarray([4]), np.asarray([6]), np.asarray([7]), "ordered_pair")[1]
    swapped_predictions = second.predictions(np.asarray([4]), np.asarray([6]), np.asarray([1]), "ordered_pair")[1]
    np.testing.assert_array_equal(original_predictions, swapped_predictions)


def write_inputs(directory, include_single=False):
    features = directory / "features"
    features.mkdir(parents=True)
    dataset = {"vocabulary": VOCABULARY, "splits": {}}
    data = {}
    for split in ["train", "val", "test"]:
        data[split], story = fixture_split(split)
        dataset["splits"][split] = [story]
        np.savez_compressed(features / (split + ".npz"), **data[split])
    dataset_path = directory / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    manifest = {"completed": True, "dataset_sha256": pair.file_sha256(dataset_path),
                "brain": {"graph_sha256": "fixture-graph", "dt_ms": .1, "word_ms": 20,
                          "warmup_ms": 100, "feature_dimensions": 256, "projection_sha256": "same"},
                "final_graph_sha256": "fixture-graph",
                "split_hashes": {split: pair.file_sha256(features / (split + ".npz")) for split in data}}
    # Deliberately use compact JSON to exercise byte-exact manifest copying.
    (features / "manifest.json").write_text(json.dumps(manifest))
    single = None
    if include_single:
        single = directory / "single"
        single.mkdir()
        for split, source in data.items():
            old = {key: source[key] for key in ["X", "positions", "current_ids", "story_ids"]}
            old["y"] = source["y"][:, 0]
            np.savez_compressed(single / (split + ".npz"), **old)
        old_manifest = dict(manifest)
        old_manifest["split_hashes"] = {split: pair.file_sha256(single / (split + ".npz")) for split in data}
        (single / "manifest.json").write_text(json.dumps(old_manifest))
    return features, dataset_path, data, single


def test_all_selections_before_test_and_artifact_compatibility(tmp_path, monkeypatch):
    features, dataset, data, single = write_inputs(tmp_path, include_single=True)
    events = []
    original_fit, original_load = pair.fit_pair, pair.load_pair_split
    def fit(*args, **kwargs):
        events.append("fit")
        return original_fit(*args, **kwargs)
    def load(path, *args, **kwargs):
        events.append(path.stem)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(pair, "fit_pair", fit)
    monkeypatch.setattr(pair, "load_pair_split", load)
    result = pair.run(features, dataset, tmp_path / "output", single_features=single,
                      seeds=[0], epochs=1, l2_values=[0])
    assert events.count("fit") == 3
    assert events.index("test") > max(i for i, event in enumerate(events) if event == "fit")
    root = tmp_path / "output"
    loaded = pair.PairDecoder.load(root)
    assert loaded.config["arm"] == "brain"
    assert pair.file_sha256(loaded.checkpoint_dir / "feature-manifest.json") == loaded.config["feature_manifest_sha256"]
    assert set(result["arms"]) == {"brain", "direct", "single_word_brain"}
    with np.load(loaded.checkpoint_dir / "test-predictions.npz") as saved:
        np.testing.assert_array_equal(saved["y"], data["test"]["y"])
        assert np.isnan(saved["losses"][-1, 1])
        assert saved["predictions"][-1, 1] == -1
        np.testing.assert_array_equal(saved["target_mask"], data["test"]["target_mask"])
    # Since these fixtures intentionally give both brain arms identical X,
    # their head0 checkpoints must be exactly identical numerically.
    old_arm = pair.PairDecoder.load(root / "single_word_brain")
    np.testing.assert_array_equal(loaded.heads[0].logits(data["test"]["X"]), old_arm.heads[0].logits(data["test"]["X"]))


def test_test_feature_shift_cannot_change_head_selection_or_scales(tmp_path):
    features, dataset, data, _ = write_inputs(tmp_path)
    pair.run(features, dataset, tmp_path / "first", seeds=[0], epochs=1, l2_values=[0])
    data["test"]["X"] *= -2
    data["test"]["X_direct"] *= 3
    np.savez_compressed(features / "test.npz", **data["test"])
    manifest = json.loads((features / "manifest.json").read_text())
    manifest["split_hashes"]["test"] = pair.file_sha256(features / "test.npz")
    (features / "manifest.json").write_text(json.dumps(manifest))
    pair.run(features, dataset, tmp_path / "second", seeds=[0], epochs=1, l2_values=[0])
    for arm in ["brain", "direct"]:
        first = pair.PairDecoder.load(tmp_path / "first" / arm)
        second = pair.PairDecoder.load(tmp_path / "second" / arm)
        for a, b in zip(first.heads, second.heads):
            np.testing.assert_array_equal(a.scaler.mean, b.scaler.mean)
            np.testing.assert_array_equal(a.scaler.scale, b.scaler.scale)
            for key, value in a.model.state_dict().items():
                assert torch.equal(value, b.model.state_dict()[key])
