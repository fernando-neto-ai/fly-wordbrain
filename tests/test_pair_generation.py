"""Causal pair-feedback and rolling-window tests without loading a connectome."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from fly_wordbrain import pair_generate as generation


VOCABULARY = ["<pad>", "<unk>", "<bos>", "<eos>", "alpha", "beta", "gamma", "delta"]


class StubBrain:
    def __init__(self):
        self.segments = []
        self.calls = 0

    def reset(self):
        self.segments.append([])

    def step(self, pair):
        pair = tuple(pair)
        self.segments[-1].append(pair)
        self.calls += 1
        return np.array([self.calls], np.float32)

    def verify_frozen(self):
        return "unchanged-graph"


class StubDecoder:
    vocabulary = VOCABULARY

    def __init__(self):
        self.observed_features = []

    def logits(self, features):
        self.observed_features.append(np.array(features, copy=True))
        return np.stack((np.arange(len(self.vocabulary)), np.arange(len(self.vocabulary))[::-1]))


class ScheduledRng:
    def __init__(self, brain, tokens):
        self.brain = brain
        self.tokens = iter(tokens)
        self.neural_calls_at_draw = []

    def choice(self, count, p):
        assert count == len(VOCABULARY)
        assert p[0] == p[2] == 0
        assert np.isclose(p.sum(), 1)
        self.neural_calls_at_draw.append(self.brain.calls)
        return next(self.tokens)


def test_both_heads_sample_before_either_word_is_fed_back():
    brain, decoder = StubBrain(), StubDecoder()
    rng = ScheduledRng(brain, [4, 5, 6, 7])
    result = generation.generate_sequence(brain, decoder, "alpha beta", rng, max_words=4)
    assert rng.neural_calls_at_draw == [3, 3, 5, 5]
    assert brain.segments == [[(2, 2), (2, 4), (4, 5), (5, 4), (4, 5), (5, 6), (6, 7)]]
    assert [float(value[0]) for value in decoder.observed_features] == [3, 5]
    assert result["generated_ids"] == [4, 5, 6, 7]
    assert result["samples"][0]["shared_feature_sha256"] == result["samples"][1]["shared_feature_sha256"]
    assert result["samples"][2]["shared_feature_sha256"] == result["samples"][3]["shared_feature_sha256"]
    assert result["samples"][0]["model_probability"] != result["samples"][0]["sample_probability"]


def test_first_head_eos_stops_without_sampling_or_feeding_second():
    brain, decoder = StubBrain(), StubDecoder()
    rng = ScheduledRng(brain, [3])
    result = generation.generate_sequence(brain, decoder, "alpha", rng)
    assert result["stopped"] == "eos"
    assert result["eos_future_word_offset"] == 1
    assert result["generated_ids"] == []
    assert result["sampled_ids_including_eos"] == [3]
    assert rng.neural_calls_at_draw == [2]
    assert brain.segments == [[(2, 2), (2, 4)]]
    assert result["pair_predictions"][0]["second_head_not_sampled_reason"] == "first_head_eos"


def test_second_head_eos_consumes_first_word_only_then_stops():
    brain, decoder = StubBrain(), StubDecoder()
    rng = ScheduledRng(brain, [5, 3])
    result = generation.generate_sequence(brain, decoder, "alpha", rng)
    assert result["stopped"] == "eos"
    assert result["eos_future_word_offset"] == 2
    assert result["generated_ids"] == [5]
    assert result["sampled_ids_including_eos"] == [5, 3]
    assert rng.neural_calls_at_draw == [2, 2]
    assert brain.segments == [[(2, 2), (2, 4), (4, 5)]]


def test_odd_limit_skips_unused_second_head_and_unk_is_literal():
    brain, decoder = StubBrain(), StubDecoder()
    rng = ScheduledRng(brain, [1, 5, 6])
    result = generation.generate_sequence(brain, decoder, "alpha", rng, max_words=3)
    assert result["generated_words"] == ["<unk>", "beta", "gamma"]
    assert rng.neural_calls_at_draw == [2, 2, 4]
    assert result["pair_predictions"][-1]["second_head_not_sampled_reason"] == "word_limit"
    assert brain.segments[-1][-3:] == [(4, 1), (1, 5), (5, 6)]


def test_rolling_replay_drops_prefix_from_state_and_previous_word_slot():
    brain, decoder = StubBrain(), StubDecoder()
    rng = ScheduledRng(brain, [5, 6])
    prompt = "delta " + " ".join(["alpha"] * 127)
    result = generation.generate_sequence(brain, decoder, prompt, rng, max_words=2)
    assert rng.neural_calls_at_draw == [129, 129]
    assert len(brain.segments) == 3
    for replay in brain.segments[1:]:
        assert len(replay) == 129
        assert replay[:2] == [(2, 2), (2, 4)]
        assert all(7 not in pair for pair in replay)
    assert brain.segments[-1][-2:] == [(4, 5), (5, 6)]
    assert result["final_context_lexical_words"] == 128
    assert [row["after_generated_words"] for row in result["rolling_context_replays"]] == [1, 2]


def test_logits_masking_retains_full_model_probabilities():
    logits = np.array([30., -2., 40., 0., 1., 2., 3., 4.])
    original_logits = logits.copy()
    probabilities, original = generation.sampling_probabilities(logits, .8)
    assert np.array_equal(logits, original_logits)
    assert probabilities[0] == probabilities[2] == 0
    assert original[0] > 0 and original[2] > 0
    assert probabilities[1] > 0 and probabilities[3] > 0
    with pytest.raises(ValueError, match="finite"):
        generation.sampling_probabilities(np.array([0., 0., 0., np.nan]), .8)


def test_provenance_rejects_different_arm_sources_or_brain(monkeypatch):
    hashes = {"dataset.json": "dataset", "manifest.json": "manifest",
              "brain.py": "brain-code", "pair_brain.py": "pair-code"}
    monkeypatch.setattr(generation, "sha256_file", lambda path: hashes[path.name])
    metadata = {"feature_dimensions": 256, "graph_sha256": "graph", "word_ms": 20.}
    manifest = {"dataset_sha256": "dataset", "completed": True,
                "split_hashes": {"train": "train", "val": "val", "test": "test"},
                "brain": metadata.copy(), "final_graph_sha256": "graph", "identity": "features",
                "code_sha256": {"brain.py": "brain-code", "pair_brain.py": "pair-code"}}
    heads = [SimpleNamespace(vocabulary=VOCABULARY, scaler=SimpleNamespace(mean=np.zeros(256)),
                             config={"arm": "brain", "horizon": number + 1,
                                     "dataset_sha256": "dataset", "feature_sha256": manifest["split_hashes"]})
             for number in range(2)]
    decoder = SimpleNamespace(vocabulary=VOCABULARY, heads=heads, config={
        "arm": "brain", "seed": 0, "dataset_sha256": "dataset",
        "feature_sha256": manifest["split_hashes"], "feature_manifest_sha256": "manifest",
        "brain": metadata.copy(), "feature_dimensions": 256})
    brain = SimpleNamespace(metadata=lambda: metadata.copy())
    args = (generation.Path("dataset.json"), {"vocabulary": VOCABULARY}, decoder,
            generation.Path("manifest.json"), manifest, brain)
    assert generation.check_provenance(*args)["feature_manifest_identity"] == "features"
    decoder.config["arm"] = "direct"
    with pytest.raises(ValueError, match="brain arm"):
        generation.check_provenance(*args)
    decoder.config["arm"] = "brain"
    heads[1].config["horizon"] = 1
    with pytest.raises(ValueError, match="head has different"):
        generation.check_provenance(*args)
    heads[1].config["horizon"] = 2
    hashes["pair_brain.py"] = "changed"
    with pytest.raises(ValueError, match="source changed"):
        generation.check_provenance(*args)
    hashes["pair_brain.py"] = "pair-code"
    metadata["word_ms"] = 40.
    with pytest.raises(ValueError, match="Generation brain differs"):
        generation.check_provenance(*args)


def test_saved_pair_decoder_and_cli_generation_provenance_integrate(tmp_path, monkeypatch):
    """Exercise the real checkpoint interface with a stub neural simulator."""
    import torch
    from fly_wordbrain import pair_brain
    from fly_wordbrain.decoder import Decoder, Standardizer
    from fly_wordbrain.pair_decoder import PairDecoder

    metadata = {"feature_dimensions": 256, "graph_sha256": "unchanged-graph", "word_ms": 20.}

    class StubPairedBrain(StubBrain):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def step(self, pair):
            super().step(pair)
            features = np.zeros(256, np.float32)
            features[0] = self.calls
            return features

        def metadata(self):
            return metadata.copy()

    monkeypatch.setattr(pair_brain, "PairedFrozenBrain", StubPairedBrain)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"vocabulary": VOCABULARY}))
    dataset_hash = generation.sha256_file(dataset_path)
    split_hashes = {"train": "train", "val": "val", "test": "test"}
    manifest = {"completed": True, "dataset_sha256": dataset_hash,
                "brain": metadata, "split_hashes": split_hashes,
                "identity": "pair-fixture", "final_graph_sha256": "unchanged-graph",
                "code_sha256": {name: generation.sha256_file(generation.Path(generation.__file__).with_name(name))
                                for name in ("brain.py", "pair_brain.py")}}
    torch.manual_seed(17)
    heads = [Decoder(torch.nn.Linear(256, len(VOCABULARY)),
                     Standardizer(np.zeros(256, np.float32), np.ones(256, np.float32)),
                     VOCABULARY, {"arm": "brain", "horizon": index + 1,
                                  "dataset_sha256": dataset_hash, "feature_sha256": split_hashes})
             for index in range(2)]
    model = PairDecoder(heads, {"arm": "brain", "seed": 0, "feature_dimensions": 256,
                               "brain": metadata, "dataset_sha256": dataset_hash,
                               "feature_sha256": split_hashes})
    checkpoint = tmp_path / "brain" / "seed-0"
    model.save(checkpoint, manifest)
    model.config["feature_manifest_sha256"] = generation.sha256_file(checkpoint / "feature-manifest.json")
    model.save(checkpoint, manifest)
    output = tmp_path / "generation.json"
    record = generation.generate(dataset_path, checkpoint, output)
    assert record["sampling_seed"] == 1729 and record["temperature"] == .8
    assert record["prompts"] == list(generation.PROMPTS)
    assert len(record["samples"]) == 3
    assert record["provenance"]["feature_manifest_identity"] == "pair-fixture"
    assert record["checkpoint_files_sha256"]["pair.json"] == generation.sha256_file(checkpoint / "pair.json")
    assert all(row["frozen_graph_sha256_after"] == "unchanged-graph" for row in record["samples"])
    assert json.loads(output.read_text())["frozen_graph"] is True
