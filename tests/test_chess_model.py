"""The chess head's recurrence must be the language model's recurrence, exactly.

`ChessFly.settle` re-implements the update loop because a board is held constant across
settling steps rather than arriving one token at a time. A re-implementation that drifts
from the pinned model would quietly turn "what can this brain do" into "what can some
other dynamical system do", and no downstream number would notice. So the loop is driven
with the same currents the pinned model builds for a real token sequence, and its output
is compared against that model's own forward pass step for step.
"""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fly_wordbrain.chess_model import ChessFly
from fly_wordbrain.connectorch_model import build_connectorch_model
from fly_wordbrain.metal_sparse_trainable import TrainableCSR
from test_connectorch_model import reference_fixture


def make_brain(**kwargs):
    return build_connectorch_model(reference_fixture(), csr_factory=TrainableCSR, **kwargs).brain


def language_drive(brain, input_ids):
    """Reproduce the currents the pinned model injects for a token sequence."""
    config = brain.config
    k = config.delay_k
    embeds = brain.wte(input_ids)
    previous = torch.full((input_ids.shape[0], k), config.pad_token_id, dtype=torch.long)
    extended = torch.cat([brain.wte(previous), embeds], dim=1)
    length = input_ids.shape[1]
    drive = torch.cat([extended[:, k - lag:k - lag + length] @ brain.in_proj[j]
                       for j, lag in enumerate(brain.lag_map)], dim=-1)
    return drive.permute(1, 2, 0).contiguous()


class ChessModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.brain = make_brain(d_embed=4)
        self.fly = ChessFly(self.brain, settle_steps=3, encoder_rank=2, readout_rank=2,
                            moves=7, value_bins=8, features=5)

    def test_settle_matches_the_pinned_models_own_forward_pass(self):
        ids = torch.tensor([[1, 3, 4, 2], [1, 8, 2, 5]])
        expected = self.brain(ids, use_cache=False).last_hidden_state
        drive = language_drive(self.brain, ids)
        actual = self.fly.settle(drive, collect=True)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0),
                        f"max deviation {(actual - expected).abs().max().item():.3e}")

    def test_a_held_current_is_the_same_as_repeating_it(self):
        current = torch.randn(self.brain.in_index.numel(), 3)
        held = self.fly.settle(current, steps=4)
        repeated = self.fly.settle(current[None].expand(4, -1, -1).contiguous(), collect=True)
        self.assertTrue(torch.allclose(held, repeated[:, -1], atol=1e-6, rtol=0))

    def test_settling_starts_from_rest_so_positions_are_independent(self):
        current = torch.randn(self.brain.in_index.numel(), 2)
        first = self.fly.settle(current)
        self.fly.settle(torch.randn(self.brain.in_index.numel(), 2))
        self.assertTrue(torch.equal(first, self.fly.settle(current)))

    def test_forward_shapes_and_value_decoding(self):
        features = torch.randn(6, 5)
        logits, value_logits = self.fly(features)
        self.assertEqual(logits.shape, (6, 7))
        self.assertEqual(value_logits.shape, (6, 8))
        decoded = self.fly.value_from_logits(value_logits)
        self.assertEqual(decoded.shape, (6,))
        self.assertTrue(bool(((decoded >= 0) & (decoded <= 1)).all()))

    def test_two_hot_value_targets_are_normalised_and_decode_back(self):
        values = torch.tensor([0.0, 0.125, 0.5, 0.77, 1.0])
        targets = self.fly.value_targets(values)
        self.assertTrue(torch.allclose(targets.sum(-1), torch.ones(5), atol=1e-6))
        recovered = (targets * self.fly.bin_centres).sum(-1)
        # Interior values round-trip through the two-hot split; the outermost half-bins
        # cannot, because no centre lies beyond them.
        self.assertTrue(torch.allclose(recovered[1:4], values[1:4], atol=1e-6))
        self.assertLessEqual(float((recovered[0] - values[0]).abs()), .5 / self.fly.value_bins)
        self.assertLessEqual(float((recovered[4] - values[4]).abs()), .5 / self.fly.value_bins)

    def test_gradients_reach_the_brains_own_parameters(self):
        logits, _ = self.fly(torch.randn(4, 5))
        logits.square().mean().backward()
        for name in ("gain", "rec_gain", "bias"):
            gradient = getattr(self.brain, name).grad
            self.assertIsNotNone(gradient, f"no gradient reached brain.{name}")
            self.assertGreater(float(gradient.abs().sum()), 0.0, f"brain.{name} got a zero gradient")


if __name__ == "__main__":
    unittest.main()
