"""Causal ordered-word-pair encoding and two-horizon target contracts.

These tests allocate word codes, not a connectome or neural simulator.
"""

import unittest
from unittest.mock import patch

import numpy as np

from fly_wordbrain.brain import FrozenBrain, OutputProjection
from fly_wordbrain.pair_brain import DirectProjection, OrderedPairEncoder, PairedFrozenBrain
from fly_wordbrain.pair_extract import story_rows


class PairEncodingTests(unittest.TestCase):
    def encoder(self, seed=1729):
        return OrderedPairEncoder(vocab_size=8, receptors=3335, seed=seed, high=.02)

    def test_equal_energy_deterministic_and_ordered(self):
        first, repeat = self.encoder(), self.encoder()
        for previous, current in ((2, 2), (4, 5), (5, 4), (7, 1), (0, 3)):
            stimulus = first((previous, current))
            self.assertEqual(stimulus.shape, (3335,))
            self.assertEqual(np.count_nonzero(stimulus), 834)
            np.testing.assert_allclose(stimulus[stimulus != 0], .02, rtol=1e-6)
            self.assertAlmostEqual(float(stimulus.sum()), 834 * .02, places=5)
            np.testing.assert_array_equal(stimulus, repeat((previous, current)))
        self.assertFalse(np.array_equal(first((4, 5)), first((5, 4))))
        self.assertFalse(np.array_equal(first((4, 5)), self.encoder(1731)((4, 5))))

    def test_changing_one_role_leaves_the_other_role_unchanged(self):
        encoder = self.encoder()
        base = encoder((4, 5))
        previous_changed = encoder((6, 5))
        current_changed = encoder((4, 7))
        previous_positions = base != previous_changed
        current_positions = base != current_changed
        self.assertTrue(previous_positions.any())
        self.assertTrue(current_positions.any())
        self.assertFalse(np.any(previous_positions & current_positions))
        # Effects add exactly because the roles occupy disjoint retinal banks.
        both_changed = encoder((6, 7))
        np.testing.assert_allclose(both_changed - base,
                                   previous_changed - base + current_changed - base,
                                   atol=1e-8, rtol=0)

    def test_role_changes_do_not_mutate_previously_returned_stimulus(self):
        encoder = self.encoder()
        first = encoder((4, 5))
        frozen_copy = first.copy()
        encoder((6, 7))
        np.testing.assert_array_equal(first, frozen_copy)

    def test_all_receptors_retained_in_disjoint_role_banks(self):
        encoder = self.encoder()
        previous, current = encoder.previous_bank, encoder.current_bank
        self.assertEqual((len(previous), len(current)), (1667, 1668))
        self.assertFalse(set(previous) & set(current))
        self.assertEqual(set(previous) | set(current), set(range(3335)))
        stimulus = encoder((4, 5))
        self.assertEqual(np.count_nonzero(stimulus[previous]), 417)
        self.assertEqual(np.count_nonzero(stimulus[current]), 417)
        np.testing.assert_array_equal(stimulus[previous], encoder.previous_codes[4])
        np.testing.assert_array_equal(stimulus[current], encoder.current_codes[5])

    def test_direct_control_is_fixed_linear_projection_of_exact_stimulus(self):
        encoder = self.encoder()
        projection = DirectProjection(receptors=3335, bins=256, seed=3187)
        repeat = DirectProjection(receptors=3335, bins=256, seed=3187)
        first, second = encoder((4, 5)), encoder((6, 7))
        first_features = projection(first)
        self.assertEqual(first_features.shape, (256,))
        self.assertTrue(np.isfinite(first_features).all())
        np.testing.assert_array_equal(first_features, repeat(first))
        np.testing.assert_allclose(projection(first + second), projection(first) + projection(second), atol=1e-7)
        np.testing.assert_array_equal(projection(np.zeros(3335, np.float32)), np.zeros(256))

    def test_paired_adapter_advances_original_step_once_with_exact_pair_stimulus(self):
        class TinyNative:
            retina = np.arange(16)
            v = np.full(20, -52., np.float32)

            def __init__(self):
                self.calls = []

            def step(self, stimulus, duration, sugar, lamina_bias):
                self.calls.append((stimulus.copy(), duration, sugar, lamina_bias))
                return np.arange(20, dtype=np.int32), .001

        def initialize_without_connectome(brain, *args):
            brain.native = TinyNative()
            brain.word_ms = 20.
            brain.projection = OutputProjection(np.arange(16, 20), bins=128)

        with patch.object(FrozenBrain, "__init__", initialize_without_connectome):
            brain = PairedFrozenBrain(vocab_size=8)
        self.assertIs(PairedFrozenBrain.step, FrozenBrain.step)
        pair = (4, 5)
        result = brain.step(pair)
        self.assertEqual(result.shape, (256,))
        self.assertEqual(len(brain.native.calls), 1)
        actual, duration, sugar, bias = brain.native.calls[0]
        np.testing.assert_array_equal(actual, brain.encoder(pair))
        self.assertEqual((duration, sugar, bias), (20., False, 12.))
        np.testing.assert_array_equal(brain.direct_features(pair), brain.direct_projection(actual))


class PairTargetTests(unittest.TestCase):
    def story(self, ids, masks=None):
        return {"id": "tiny-story", "word_ids": ids,
                "words": [f"word-{wid}" for wid in ids if wid not in (0, 2, 3)],
                "target_mask": masks if masks is not None else [False] + [True] * (len(ids) - 1)}

    def normalized(self, rows):
        return [(row["position"], row["previous_id"], row["current_id"],
                 tuple(row["targets"]), tuple(row["target_mask"])) for row in rows]

    def test_two_future_targets_and_real_final_eos(self):
        actual = self.normalized(story_rows(self.story([2, 5, 6, 7, 3])))
        self.assertEqual(actual, [
            (0, 2, 2, (5, 6), (True, True)),
            (1, 2, 5, (6, 7), (True, True)),
            (2, 5, 6, (7, 3), (True, True)),
            (3, 6, 7, (3, 0), (True, False)),
        ])

    def test_truncated_story_never_gains_a_false_eos(self):
        actual = self.normalized(story_rows(self.story([2, 5, 6, 7])))
        self.assertEqual(actual[-1], (2, 5, 6, (7, 0), (True, False)))
        for row in actual:
            self.assertNotIn(3, row[3])
        self.assertEqual(len(actual), 3)

    def test_one_word_and_two_word_boundaries(self):
        complete = self.normalized(story_rows(self.story([2, 5, 3])))
        self.assertEqual(complete, [(0, 2, 2, (5, 3), (True, True)),
                                    (1, 2, 5, (3, 0), (True, False))])
        truncated = self.normalized(story_rows(self.story([2, 5])))
        self.assertEqual(truncated, [(0, 2, 2, (5, 0), (True, False))])

    def test_future_content_does_not_enter_current_input_pair(self):
        original = list(story_rows(self.story([2, 5, 6, 7, 8, 3])))
        altered = list(story_rows(self.story([2, 5, 6, 99, 100, 3])))
        for p in range(3):
            for field in ("position", "previous_id", "current_id"):
                self.assertEqual(original[p][field], altered[p][field])
        self.assertNotEqual(tuple(original[2]["targets"]), tuple(altered[2]["targets"]))

    def test_128_word_context_does_not_count_bos_or_eos_as_words(self):
        complete = list(story_rows(self.story([2] + [5] * 128 + [3])))
        self.assertEqual(len(complete), 129)
        self.assertEqual(complete[-1]["position"], 128)
        self.assertEqual(tuple(complete[-1]["targets"]), (3, 0))
        truncated = list(story_rows(self.story([2] + [5] * 128)))
        self.assertEqual(len(truncated), 128)
        self.assertEqual(truncated[-1]["position"], 127)

    def test_second_source_mask_is_respected_and_unsupported_interior_mask_fails(self):
        rows = story_rows(self.story([2, 5, 6, 7, 3], [False, True, False, True, True]))
        first = next(rows)
        self.assertEqual(tuple(first["target_mask"]), (True, False))
        self.assertEqual(tuple(first["targets"]), (5, 0))
        with self.assertRaisesRegex(ValueError, "Interior masked"):
            next(rows)

    def test_overlong_story_is_rejected_before_extraction(self):
        with self.assertRaisesRegex(ValueError, "128"):
            list(story_rows(self.story([2] + [5] * 129 + [3])))


if __name__ == "__main__":
    unittest.main()
