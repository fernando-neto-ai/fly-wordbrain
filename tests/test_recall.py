"""Lightweight behavioral checks without loading or simulating the connectome."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from fly_wordbrain.data import SPECIAL_TOKENS, build_recall_probes
from fly_wordbrain import recall


class FakeBrain:
    def __init__(self):
        self.calls = []
        self.resets = 0
        self.last_diagnostics = {"kernel_seconds": .1, "all_spikes": 2}

    def reset(self):
        self.resets += 1

    def step(self, word_id):
        self.calls.append(word_id)
        return np.full(256, len(self.calls), np.float32)

    def verify_frozen(self):
        return "frozen-graph-hash"


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"vocabulary": list(SPECIAL_TOKENS) + ["lily", "tim", "tom", "sam"]}
        self.probes = build_recall_probes(self.dataset["vocabulary"])

    def test_valid_controlled_probe(self):
        recall.validate_probes(self.dataset, self.probes)

    def test_changed_encoding_and_leaked_target_rejected(self):
        changed = copy.deepcopy(self.probes)
        changed["splits"]["train"][0]["word_ids"][2] = 1
        with self.assertRaisesRegex(ValueError, "encoding"):
            recall.validate_probes(self.dataset, changed)
        changed = copy.deepcopy(self.probes)
        row = changed["splits"]["train"][0]
        row["words"][-1] = "lily"
        row["word_ids"][-1] = self.dataset["vocabulary"].index("lily")
        with self.assertRaisesRegex(ValueError, "leaks"):
            recall.validate_probes(self.dataset, changed)

    def test_full_prompt_is_presented_and_only_final_feature_is_returned(self):
        brain = FakeBrain()
        row = self.probes["splits"]["train"][0]
        with patch.object(recall, "WORKER", brain):
            result = recall.prompt_features(row)
        self.assertEqual(brain.calls, row["word_ids"])
        self.assertEqual(brain.resets, 1)
        self.assertEqual(result["X"].shape, (1, 256))
        np.testing.assert_array_equal(result["X"], np.full((1, 256), len(row["word_ids"])))
        self.assertEqual(result["y"].tolist(), [row["target_class"]])
        self.assertNotIn("word_ids", result)
        self.assertNotIn("current_ids", result)

    def test_all_four_classes_scored_without_unknown_mask(self):
        targets = np.arange(4)
        logits = np.eye(4) * 10
        logits[1] = [10, -10, -10, -10]
        result = recall.class_metrics(logits, targets)
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["examples"], 4)
        self.assertGreater(result["cross_entropy"], 4.9)
        self.assertNotIn("known_target_accuracy", result)
        uniform = recall.class_metrics(np.zeros((4, 4)), targets)
        self.assertAlmostEqual(uniform["cross_entropy"], np.log(4))

    def test_only_neural_features_reach_classifier_and_every_context_has_eight_items(self):
        items = self.probes["splits"]["test"]
        X = np.zeros((len(items), 256), np.float32)
        data = {"X": X, "y": np.array([item["target_class"] for item in items]),
                "context_words": np.array([item["context_words"] for item in items])}

        class FakeDecoder:
            def logits(self, features):
                self.seen = features
                return np.zeros((len(features), 4))

        decoder = FakeDecoder()
        report = recall.evaluate_recall(decoder, data)
        self.assertIs(decoder.seen, X)
        for metric in report["by_context_words"].values():
            self.assertEqual(metric["examples"], 8)
            self.assertEqual(metric["class_counts"], [2, 2, 2, 2])

    def test_word_specific_metrics_are_removed_from_learning_curves(self):
        metric = {"examples": 4, "cross_entropy": 1., "accuracy": .5,
                  "known_target_accuracy": 0., "unknown_target_fraction": .25}
        curves = [{"seed": 0, "l2": 0., "epochs": [{"epoch": 1, "train": metric, "val": metric}]}]
        cleaned = recall.clean_learning_curves(curves)
        self.assertEqual(set(cleaned[0]["epochs"][0]["val"]), {"examples", "cross_entropy", "accuracy"})


if __name__ == "__main__":
    unittest.main()
