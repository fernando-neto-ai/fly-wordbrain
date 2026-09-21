"""The two active-inference probes' records must support exactly the claims made from them."""
import json, unittest
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "experiments/records"

class FreeEnergyRoutingTests(unittest.TestCase):
    def test_generative_surprise_routes_everything_to_language(self):
        # Only the language cue is a generative model of its inputs; every other cue makes
        # the brain WORSE at predicting its own text, so min-surprise always says language.
        for name in ("fe-routing-Y32threetasksrank128bounded.json",
                     "fe-routing-Z32fourtasksbounded-partial8992.json"):
            r = json.loads((R / name).read_text())
            self.assertEqual(r["route_by_min_surprise"]["recall"]["language"], 1.0, name)
            self.assertLessEqual(r["route_by_min_surprise"]["recall"]["sentiment"], 0.05, name)
            m = r["mean_surprise_by_source_and_cue"]
            for source, row in m.items():
                self.assertEqual(min(row, key=row.get), "language",
                                 f"{name}: {source} text is best predicted under the language cue")
    def test_self_consistency_routing_beats_majority_on_two_tasks_only(self):
        y = json.loads((R / "fe-routing-Y32threetasksrank128bounded.json").read_text())
        self.assertGreater(y["route_by_min_normalised_leakage"]["accuracy"], y["majority_class"] + 0.1)
        z = json.loads((R / "fe-routing-Z32fourtasksbounded-partial8992.json").read_text())
        self.assertLess(z["route_by_min_normalised_leakage"]["accuracy"], z["majority_class"] + 0.05,
                        "on the sentiment/toxicity pair, leakage routing is chance")

class SurpriseGatedStateTests(unittest.TestCase):
    def test_the_brain_updates_uniformly_and_slightly_less_when_surprised(self):
        r = json.loads((R / "surprise-state-Y32threetasksrank128bounded.json").read_text())
        self.assertLess(abs(r["pearson_r"]), 0.2, "no meaningful surprise-gating")
        self.assertLess(r["pearson_r"], 0.0, "the small slope is NEGATIVE: surprising tokens move the state less")
        self.assertGreater(r["mean_relative_state_change"], 0.7, "leak 0.9 replaces most of the state every step")
        q = r["by_surprise_quartile"]
        self.assertGreater(q["q1"]["mean_relative_state_change"], q["q4"]["mean_relative_state_change"])

if __name__ == "__main__":
    unittest.main()


class GenerationMatchingTests(unittest.TestCase):
    """Two arms compared from `latest.pt` are matched only by luck; the record must say so."""

    def test_the_matched_record_is_flagged_matched(self):
        r = json.loads((R / "generations-L8998-vs-Z8991-matched.json").read_text())
        arms = [a["updates"] for a in r["arms"].values()]
        self.assertLessEqual(max(arms) - min(arms), 50)

    def test_the_confounded_record_is_not_presented_as_matched(self):
        r = json.loads((R / "generations-L10171-vs-Z8991-40prompts.json").read_text())
        arms = [a["updates"] for a in r["arms"].values()]
        self.assertGreater(max(arms) - min(arms), 1000,
                           "this record IS confounded and must not be read as matched")

    def test_generation_and_cross_entropy_disagree_at_the_matched_point(self):
        # The finding worth keeping: teacher-forced CE says the arms are identical while
        # free-running generation says they are not.
        g = json.loads((R / "generations-L8998-vs-Z8991-matched.json").read_text())
        arms = list(g["arms"].values())
        self.assertLess(arms[0]["mean_repeated_3gram_rate"] * 10, arms[1]["mean_repeated_3gram_rate"])
