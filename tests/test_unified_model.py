"""The unified head: one output space, one injection, and a router that only observes.

The claim this architecture makes is that both tasks go through the same parameters, so
the tests check that the shared pieces really are shared, that the two tasks occupy
disjoint halves of one output space, and that the router reads the brain rather than
steering it. A router wired into the output path would turn a measurement into an
architectural crutch, so `test_router_does_not_gate_the_output` pins that down.
"""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fly_wordbrain.connectorch_model import build_connectorch_model
from fly_wordbrain.metal_sparse_trainable import TrainableCSR
from fly_wordbrain.unified_model import CHESS, LANGUAGE, UnifiedFly, move_targets
from test_connectorch_model import reference_fixture


class UnifiedModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.brain = build_connectorch_model(reference_fixture(), csr_factory=TrainableCSR,
                                             d_embed=4).brain
        self.fly = UnifiedFly(self.brain, settle_steps=3, readout_rank=3, value_bins=8,
                              features=5, tokens=11, moves=7)

    def test_both_tasks_share_one_output_space(self):
        tokens, _, _ = self.fly.language(torch.tensor([[1, 3, 2]]))
        moves, _, _ = self.fly.chess(torch.randn(4, 5))
        self.assertEqual(tokens.shape[-1], 18)
        self.assertEqual(moves.shape[-1], 18)

    def test_chess_targets_sit_above_the_language_range(self):
        self.assertTrue(torch.equal(move_targets(torch.tensor([0, 6]), tokens=11),
                                    torch.tensor([11, 17])))

    def test_both_tasks_inject_through_the_same_projection(self):
        shared = self.brain.in_proj
        before = shared.detach().clone()
        logits, _, _ = self.fly.chess(torch.randn(3, 5))
        logits.square().mean().backward()
        self.assertIsNotNone(shared.grad, "the board never reached brain.in_proj")
        self.assertGreater(float(shared.grad.abs().sum()), 0.0)
        self.assertTrue(torch.equal(shared.detach(), before))

    def test_the_board_reaches_the_brains_own_dynamics(self):
        logits, _, _ = self.fly.chess(torch.randn(3, 5))
        logits.square().mean().backward()
        for name in ("gain", "rec_gain", "bias"):
            gradient = getattr(self.brain, name).grad
            self.assertIsNotNone(gradient, f"no gradient reached brain.{name}")
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_router_does_not_gate_the_output(self):
        """The router observes. If it steered the head, breaking it would move the logits."""
        features = torch.randn(4, 5)
        with torch.no_grad():
            before, _, _ = self.fly.chess(features)
            self.fly.router.weight.mul_(-7.0)
            after, _, _ = self.fly.chess(features)
        self.assertTrue(torch.equal(before, after))

    def test_leakage_is_the_mass_in_the_other_tasks_range(self):
        # A head that is certain about one language token leaks nothing.
        certain = torch.full((1, 18), -30.0)
        certain[0, 2] = 30.0
        self.assertLess(self.fly.leakage(certain, LANGUAGE), 1e-6)
        self.assertGreater(self.fly.leakage(certain, CHESS), 1 - 1e-6)
        # A uniform head leaks in proportion to how much of the space is the other task.
        uniform = torch.zeros(1, 18)
        self.assertAlmostEqual(self.fly.leakage(uniform, LANGUAGE), 7 / 18, places=5)

    def test_own_parameters_excludes_the_shared_brain(self):
        own = {id(p) for p in self.fly.own_parameters()}
        for name in ("gain", "rec_gain", "bias", "in_proj"):
            self.assertNotIn(id(getattr(self.brain, name)), own)
        self.assertIn(id(self.fly.head.weight), own)

    def test_the_brain_is_not_registered_twice(self):
        holder = torch.nn.Module()
        holder.brain, holder.unified = self.brain, self.fly
        self.assertFalse(any(n.startswith("unified.brain") for n in holder.state_dict()))


if __name__ == "__main__":
    unittest.main()
