"""Comparability must be computed from the configs, not remembered.

Arm names grew one at a time and each encodes a different subset of what distinguishes it,
letters ran out and were reused, and reading a contrast therefore meant opening two configs
and holding them side by side. That is how a four-task arm was built carrying three-task
weights: the fourth weight was simply absent from the config it was copied from, and
nothing was watching.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_arm_registry as registry


class UnmodelledKeyTests(unittest.TestCase):
    """Every config key is either an axis or explicitly declared not-a-knob."""

    def test_the_generator_runs_clean(self):
        # It exits non-zero and names the keys if any config carries an unclassified field.
        done = subprocess.run([sys.executable, str(ROOT / "scripts/build_arm_registry.py")],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_a_new_unclassified_key_is_caught(self):
        config = {"training": {"readout_rank": 64}, "unified": {}, "a_brand_new_knob": 7}
        self.assertIn("a_brand_new_knob", registry.unmodelled(config))


class GraphAxisTests(unittest.TestCase):
    """The shuffled-graph control must never read as a clean contrast on something else."""

    def setUp(self):
        self.arms = registry.load()

    def test_the_rewired_arm_declares_a_different_graph(self):
        self.assertEqual(self.arms["I32rank64fixed10k"]["graph"], "measured")
        self.assertEqual(self.arms["K32rank64shuffled10k"]["graph"], "shuffle")

    def test_graph_is_the_only_axis_separating_the_control_from_its_reference(self):
        d = registry.differences(self.arms["I32rank64fixed10k"], self.arms["K32rank64shuffled10k"])
        self.assertEqual(d, ["graph"], "Stage 7's control must be a clean single-axis contrast")

    def test_an_arm_differing_in_graph_and_rank_is_not_a_clean_contrast(self):
        # Before the graph axis existed this pair was reported as a clean rank contrast.
        d = registry.differences(self.arms["J32fullfixed10k"], self.arms["K32rank64shuffled10k"])
        self.assertIn("graph", d)
        self.assertIn("readout_rank", d)
        self.assertGreater(len(d), 1)


class FourTaskArmTests(unittest.TestCase):
    """The arms queued to recover language must differ from Z on exactly one axis each."""

    def setUp(self):
        self.arms = registry.load()

    def test_every_four_task_arm_weights_all_four_tasks(self):
        for name, coordinates in self.arms.items():
            if len(coordinates["tasks"]) != 4:
                continue
            weights = coordinates["weights"] or {}
            for task in coordinates["tasks"]:
                self.assertIn(task, weights,
                              f"{name} trains {task} but its config carries no weight for it")

    def test_the_rank_arm_differs_from_the_four_task_baseline_only_in_rank(self):
        d = registry.differences(self.arms["Z32fourtasksbounded"],
                                 self.arms["R32fourtasksrank256bounded"])
        self.assertEqual(d, ["readout_rank"])

    def test_the_weight_arm_differs_only_in_weights(self):
        d = registry.differences(self.arms["Z32fourtasksbounded"],
                                 self.arms["Q32fourtasksheavylanguage"])
        self.assertEqual(d, ["weights"])

    def test_the_weight_arm_reallocates_without_changing_total_scale(self):
        # Raising language alone would raise total gradient magnitude on every shared
        # parameter, and the gain could not be told apart from a larger learning rate.
        for name in ("Z32fourtasksbounded", "Q32fourtasksheavylanguage"):
            w = self.arms[name]["weights"]
            total = sum(v for k, v in w.items() if k != "router")
            self.assertAlmostEqual(total, 4.0, places=6, msg=f"{name} task weights sum to {total}")
        heavy = self.arms["Q32fourtasksheavylanguage"]["weights"]
        base = self.arms["Z32fourtasksbounded"]["weights"]
        self.assertGreater(heavy["language"], base["language"])


if __name__ == "__main__":
    unittest.main()
