"""A launcher must read every knob from the arm's config and hardcode none of them.

Two failures in this project came from a launcher disagreeing with a config, and they fail
in different ways. A hardcoded `--readout-rank 64` under a config declaring 128 produces a
checkpoint that cannot be restored, discovered at scoring time hours later. A hardcoded
`--toxicity-weight 1.0` under a config declaring 0.6 produces a checkpoint that restores
perfectly and answers a different question than the record claims -- that one never
surfaces on its own.

So the test is structural: for every training flag whose value the config also carries, the
launcher must pass a shell variable, not a literal.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts/launch"

# Flags whose value an arm's config declares. Passing a literal for any of these means the
# launcher and the config can disagree.
CONFIG_OWNED = (
    "--readout-rank", "--plasticity", "--language-weight", "--chess-weight",
    "--sentiment-weight", "--toxicity-weight", "--router-weight", "--toxicity-batch",
)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.launcher = LAUNCH / "launch_arm.sh"
        self.assertTrue(self.launcher.is_file(), "the live launcher is not in the repository")
        self.source = self.launcher.read_text()

    def test_every_config_owned_flag_is_passed_as_a_variable(self):
        offenders = []
        for flag in CONFIG_OWNED:
            for match in re.finditer(re.escape(flag) + r'\s+("?)([^\s\\]+)\1', self.source):
                value = match.group(2)
                if not value.startswith("$"):
                    offenders.append(f'{flag} {value}')
        self.assertEqual(offenders, [],
                         "launcher hardcodes a value the config also declares:\n  "
                         + "\n  ".join(offenders))

    def test_the_launcher_reads_the_config_it_launches(self):
        self.assertIn("experiments/configs/$ARM.json", self.source)

    def test_it_refuses_to_start_beside_another_worker(self):
        # Two arms sharing one MPS device make both timings and both arms incomparable.
        self.assertIn("WORKER PRESENT", self.source)

    def test_it_refuses_to_overwrite_an_existing_arm(self):
        self.assertIn("OUTPUT EXISTS", self.source)


class OrchestrationTests(unittest.TestCase):
    def test_the_audit_chain_is_in_the_repository(self):
        self.assertTrue((LAUNCH / "chain_audit.py").is_file())

    def test_the_queue_gates_each_arm_on_the_previous_reaching_its_budget(self):
        source = (LAUNCH / "queue_arms.py").read_text()
        self.assertIn("16800", source.replace("_", ""))
        self.assertIn("ABORT", source)

    def test_the_queue_stops_rather_than_training_against_a_short_reference(self):
        source = (LAUNCH / "queue_arms.py").read_text()
        self.assertIn("SystemExit", source)


if __name__ == "__main__":
    unittest.main()
