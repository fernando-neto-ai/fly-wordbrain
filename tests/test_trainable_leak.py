"""A trainable leak must start as the fixed leak, exactly, and only then be allowed to move.

The leak is the fraction of state replaced each step: one scalar, 0.9, for every neuron in
the pinned model, so nothing survives ~26 steps -- shorter than the 32-token chunk language
is scored over. Making it per-neuron and trainable is the one structural lever this project
has, and it is dangerous in a specific way: every published number was produced by the
fixed recurrence, and a refactor that touched the fixed path by a single ulp would move all
of them. So the fixed path must return the config scalar itself, and the trainable path
must be indistinguishable from it at initialisation.
"""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fly_wordbrain.chess_model import settle
from fly_wordbrain.connectorch_model import build_connectorch_model
from fly_wordbrain.metal_sparse_trainable import TrainableCSR
from test_chess_model import language_drive
from test_connectorch_model import reference_fixture


def make(leak):
    torch.manual_seed(1729)
    return build_connectorch_model(reference_fixture(), csr_factory=TrainableCSR,
                                   d_embed=4, leak=leak)


class FixedPathTests(unittest.TestCase):
    def test_fixed_returns_the_config_scalar_itself(self):
        brain = make("fixed").brain
        leak = brain.effective_leak()
        self.assertIsInstance(leak, float, "the fixed path must be the pinned arithmetic, not a tensor")
        self.assertEqual(leak, brain.config.leak)

    def test_fixed_carries_no_leak_parameter(self):
        model = make("fixed")
        self.assertIsNone(model.brain.leak_delta)
        self.assertNotIn("brain.leak_delta", dict(model.named_parameters()))


class ParityTests(unittest.TestCase):
    """Trainable at initialisation is the fixed model, on both recurrences."""

    def setUp(self):
        self.fixed, self.trainable = make("fixed"), make("trainable")
        self.trainable.load_state_dict(self.fixed.state_dict(), strict=False)
        self.ids = torch.randint(3, self.fixed.config.vocab_size, (2, 6))

    def test_language_forward_is_identical_at_init(self):
        mask = torch.ones_like(self.ids)
        a = self.fixed(self.ids, mask, return_dict=False)[0]
        b = self.trainable(self.ids, mask, return_dict=False)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=0), f"max |diff| {(a-b).abs().max()}")

    def test_chess_settle_is_identical_at_init(self):
        drive = language_drive(self.fixed.brain, self.ids)
        a = settle(self.fixed.brain, drive, 4, collect=True)
        b = settle(self.trainable.brain, drive, 4, collect=True)
        self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=0))

    def test_the_effective_leak_starts_at_the_pinned_value_everywhere(self):
        leak = self.trainable.brain.effective_leak().squeeze(1)
        self.assertEqual(tuple(leak.shape), (self.trainable.config.n_neurons,))
        self.assertTrue(torch.allclose(leak, torch.full_like(leak, self.fixed.config.leak), atol=1e-7))


class TrainablePathTests(unittest.TestCase):
    def setUp(self):
        self.model = make("trainable")

    def test_one_extra_value_per_neuron(self):
        n = self.model.config.n_neurons
        self.assertEqual(self.model.brain.leak_delta.numel(), n)
        counts = self.model.parameter_counts()
        self.assertEqual(counts["neuron_parameters"], 4 * n)

    def test_the_trainer_expects_the_same_count_it_finds(self):
        import numpy as np
        from argparse import Namespace
        import train_connectorch as trainer
        n = self.model.config.n_neurons
        args = Namespace(d_embed=4, plasticity="fixed", leak="trainable", readout_rank=0,
                         history_length=8, seed=1729)
        expected = trainer.expected_parameter_counts(self.model, args, np.zeros(1, dtype=int))
        found = trainer.parameter_counts(self.model)
        self.assertEqual(expected["neurons"], 4 * n)
        self.assertEqual(found["neurons"], expected["neurons"])

    def test_a_gradient_reaches_the_leak(self):
        ids = torch.randint(3, self.model.config.vocab_size, (2, 6))
        out = self.model(ids, torch.ones_like(ids), return_dict=False)[0]
        out.sum().backward()
        grad = self.model.brain.leak_delta.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_the_leak_stays_a_valid_fraction_however_far_it_moves(self):
        # A sigmoid keeps the fraction in (0, 1) in exact arithmetic; in float32 it
        # saturates to exactly 0.0 or 1.0 past roughly +/-17, which is bounded and finite
        # but not strictly open. Assert the open interval at a large-but-representable
        # offset, and finiteness plus the closed bound at an absurd one.
        with torch.no_grad():
            self.model.brain.leak_delta.fill_(8.0); high = self.model.brain.effective_leak()
            self.model.brain.leak_delta.fill_(-8.0); low = self.model.brain.effective_leak()
            self.model.brain.leak_delta.fill_(50.0); sat = self.model.brain.effective_leak()
        self.assertTrue(bool((high > 0).all() and (high < 1).all()))
        self.assertTrue(bool((low > 0).all() and (low < 1).all()))
        self.assertTrue(bool(torch.isfinite(sat).all() and (sat >= 0).all() and (sat <= 1).all()))

    def test_decay_pulls_toward_the_fixed_arm_not_toward_one_half(self):
        # delta = 0 is leak = 0.9, the fixed arm. A parameterisation whose zero was
        # leak = 0.5 would let weight decay silently drift every neuron away from the
        # pinned dynamics.
        with torch.no_grad():
            self.model.brain.leak_delta.zero_()
        pinned = self.model.config.leak  # the fixture pins 0.7; the real model pins 0.9
        leak = self.model.brain.effective_leak().squeeze(1)
        self.assertTrue(torch.allclose(leak, torch.full_like(leak, pinned), atol=1e-7))
        # The real model's offset origin, so the number in the write-up is checked somewhere.
        self.assertAlmostEqual(math.log(0.9 / 0.1), 2.1972, places=3)

    def test_the_leak_joins_the_neuron_optimizer_group(self):
        import train_connectorch as trainer
        opt = trainer.optimizer_for(self.model, 0)
        groups = {g["name"]: {id(p) for p in g["params"]} for g in opt.param_groups}
        self.assertIn(id(self.model.brain.leak_delta), groups["input_and_neurons"])


class ConfigTests(unittest.TestCase):
    def test_an_unknown_leak_mode_is_refused(self):
        with self.assertRaises(ValueError):
            make("adaptive")

    def test_the_arm_differs_from_the_four_task_baseline_only_in_leak(self):
        import build_arm_registry as registry
        arms = registry.load()
        d = registry.differences(arms["Z32fourtasksbounded"], arms["L32fourtaskstrainableleak"])
        self.assertEqual(d, ["leak"])


if __name__ == "__main__":
    unittest.main()
