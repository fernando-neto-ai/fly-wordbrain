"""The cue-regime probe's verdict must need a margin, and the record must support the claim."""
import json, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import probe_cue_regime as probe

class VerdictTests(unittest.TestCase):
    def test_a_hair_of_difference_reads_as_comparable_not_as_a_winner(self):
        # The first version printed "cue moves it MORE" off a 0.0015 edge difference.
        self.assertIn("comparable", probe.read_regime(0.588, 0.620))
        self.assertIn("comparable", probe.read_regime(0.620, 0.588))
    def test_a_distinct_regime_needs_the_cue_to_move_far_more_than_content(self):
        self.assertIn("distinct regime", probe.read_regime(0.30, 0.62))
    def test_a_weak_bias_is_when_content_dominates(self):
        self.assertIn("weak bias", probe.read_regime(0.80, 0.62))

class RecordTests(unittest.TestCase):
    def test_the_published_probe_is_stable_across_window_counts(self):
        a = json.loads((ROOT / "experiments/records/cue-regime-Y32threetasksrank128bounded.json").read_text())
        b = json.loads((ROOT / "experiments/records/cue-regime-Y32threetasksrank128bounded-64.json").read_text())
        for key in ("same_input_different_cue", "same_cue_different_input"):
            self.assertAlmostEqual(a[key]["saturated_set_jaccard"], b[key]["saturated_set_jaccard"], delta=0.01)
    def test_the_edge_graph_is_nearly_task_invariant(self):
        r = json.loads((ROOT / "experiments/records/cue-regime-Y32threetasksrank128bounded-64.json").read_text())
        for key in ("same_input_different_cue", "same_cue_different_input", "chess_vs_language"):
            self.assertGreater(r[key]["transmitting_edge_jaccard"], 0.95)
    def test_the_cue_is_comparable_to_content_not_a_switch(self):
        r = json.loads((ROOT / "experiments/records/cue-regime-Y32threetasksrank128bounded-64.json").read_text())
        self.assertIn("comparable", r["reading"])

if __name__ == "__main__":
    unittest.main()
