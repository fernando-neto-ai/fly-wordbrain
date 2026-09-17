"""Invariants of the randomised-graph control.

The control's entire evidentiary value rests on one claim: it destroys *which neuron
connects to which* while leaving every connectivity statistic intact. If the shuffle
quietly changed a degree, a null result would be explained by the damage rather than by
the anatomy being uninformative. So the claim is tested, not asserted.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_connectorch_control import conflicts, control_indices


def make_graph(neurons=400, seed=7):
    """A simple directed graph in destination-major CSR, with skewed degrees."""
    generator = np.random.default_rng(seed)
    rows = []
    for destination in range(neurons):
        count = int(generator.integers(1, 20))
        choices = generator.choice([n for n in range(neurons) if n != destination],
                                   size=count, replace=False)
        rows.append(np.sort(choices))
    offsets = np.concatenate([[0], np.cumsum([r.size for r in rows])]).astype(np.int32)
    return offsets, np.concatenate(rows).astype(np.int32)


class GraphControlTests(unittest.TestCase):
    def setUp(self):
        self.offsets, self.source = make_graph()
        self.neurons = self.offsets.size - 1
        self.destination = np.repeat(np.arange(self.neurons, dtype=np.int64),
                                     np.diff(self.offsets))

    def test_shuffle_is_deterministic_in_its_seed(self):
        first, repairs = control_indices(self.offsets, self.source, "shuffle", 1729)
        second, again = control_indices(self.offsets, self.source, "shuffle", 1729)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(repairs, again)
        # The evaluator rebuilds the graph from the seed alone; a different seed must
        # not silently give the same graph.
        other, _ = control_indices(self.offsets, self.source, "shuffle", 1730)
        self.assertFalse(np.array_equal(first, other))

    def test_shuffle_does_not_touch_the_caller_s_array(self):
        before = self.source.copy()
        control_indices(self.offsets, self.source, "shuffle", 1729)
        self.assertTrue(np.array_equal(self.source, before))

    def test_shuffle_preserves_every_degree(self):
        shuffled, _ = control_indices(self.offsets, self.source, "shuffle", 1729)
        self.assertEqual(shuffled.size, self.source.size)
        # In-degree is the row length, and the row offsets are never returned or changed.
        self.assertTrue(np.array_equal(
            np.bincount(self.source, minlength=self.neurons),
            np.bincount(shuffled, minlength=self.neurons)), "out-degree changed")

    def test_shuffle_yields_a_simple_graph(self):
        shuffled, _ = control_indices(self.offsets, self.source, "shuffle", 1729)
        bad, _ = conflicts(shuffled, self.destination, self.neurons)
        # The sparse backend rejects duplicates rather than coalescing them, so a single
        # leftover is a hard failure at training time, not a rounding detail.
        self.assertEqual(bad, [], "self-loops or duplicate edges remain")

    def test_shuffle_actually_rewires(self):
        shuffled, _ = control_indices(self.offsets, self.source, "shuffle", 1729)
        retained = int((shuffled == self.source).sum())
        self.assertLess(retained / self.source.size, 0.05,
                        "the shuffle left too much of the original wiring in place")

    def test_zero_mode_keeps_the_topology(self):
        unchanged, repairs = control_indices(self.offsets, self.source, "zero", 1729)
        self.assertTrue(np.array_equal(unchanged, self.source))
        self.assertEqual(repairs, 0)

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(SystemExit):
            control_indices(self.offsets, self.source, "rotate", 1729)


if __name__ == "__main__":
    unittest.main()
