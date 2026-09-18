"""The paired arm comparison is only meaningful if it rescored the published models.

`compare_unified_arms.py` rebuilds each arm from its config and rescores it, rather than
reusing records, because the evaluator never wrote per-story records. That reintroduces the
risk the evaluator's own design avoided: a rebuild that drifts from the one which produced
the published numbers would report a clean, tight interval between two models nobody ever
shipped. `verify()` is the guard, so it is the part worth pinning.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RECORDS = ROOT / "experiments/records"


class VerifyGuardTests(unittest.TestCase):
    def setUp(self):
        import compare_unified_arms as module
        self.module = module
        self.published = {"language_range_only": {"cross_entropy": 3.388097881226581,
                                                  "top1_accuracy": 0.3119021727068954,
                                                  "tokens": 45059}}

    def reproduced(self, **overrides):
        base = dict(self.published["language_range_only"])
        base.update(overrides)
        return base

    def test_an_exact_reproduction_passes(self):
        self.module.verify(self.reproduced(), self.published, "U", "language_range_only")

    def test_a_drifted_cross_entropy_is_a_hard_failure(self):
        # A rebuild that loses the task cue or the third task shifts CE far more than this.
        with self.assertRaises(SystemExit) as caught:
            self.module.verify(self.reproduced(cross_entropy=3.3881), self.published,
                               "U", "language_range_only")
        self.assertIn("has drifted", str(caught.exception))

    def test_a_drifted_token_count_is_a_hard_failure(self):
        with self.assertRaises(SystemExit):
            self.module.verify(self.reproduced(tokens=45058), self.published,
                               "U", "language_range_only")

    def test_the_tolerance_is_tight_enough_to_catch_a_rounding_level_change(self):
        # 1e-9 must reject a difference in the ninth decimal, not merely a visible one.
        self.assertLess(self.module.TOLERANCE, 1e-8)
        with self.assertRaises(SystemExit):
            self.module.verify(self.reproduced(cross_entropy=3.388097881226581 + 1e-8),
                               self.published, "U", "language_range_only")


class PublishedRecordTests(unittest.TestCase):
    """The numbers quoted in the Stage 9 write-up must match the records on disk."""

    def test_the_paired_record_is_a_verified_rebuild(self):
        record = json.loads((RECORDS / "unified-v-minus-u.json").read_text())
        self.assertTrue(record["rebuild_verified_against_published_audit"])
        self.assertEqual(record["treatment"], "V32unifiedbounded")
        self.assertEqual(record["control"], "U32unifiedfixed")
        self.assertEqual(record["plasticity"],
                         {"V32unifiedbounded": "bounded10", "U32unifiedfixed": "fixed"})

    def test_adapting_the_connectome_helps_but_the_interval_is_small(self):
        record = json.loads((RECORDS / "unified-v-minus-u.json").read_text())["language_range_only"]
        low, high = record["ce_95_ci"]
        self.assertLess(high, 0.0, "the interval should exclude zero on the better side")
        self.assertGreater(low, -0.05, "the effect should be far smaller than the sharing penalty")
        # The accuracy interval straddles zero; the write-up says so.
        accuracy_low, accuracy_high = record["accuracy_95_ci"]
        self.assertLess(accuracy_low, 0.0)
        self.assertGreater(accuracy_high, 0.0)

    def test_the_sharing_penalty_dwarfs_the_plasticity_gain(self):
        sharing = json.loads((RECORDS / "unified-U32unifiedfixed-audit-language.json").read_text())
        penalty = sharing["versus_reference"]["language_range_only"]["delta_cross_entropy"]
        gain = -json.loads((RECORDS / "unified-v-minus-u.json").read_text()) \
            ["language_range_only"]["delta_cross_entropy"]
        self.assertGreater(penalty, 0.4)
        self.assertLess(gain / penalty, 0.05, "the write-up claims plasticity recovers ~3%")


if __name__ == "__main__":
    unittest.main()
