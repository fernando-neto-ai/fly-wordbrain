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
from fly_wordbrain.unified_model import (CHESS, LANGUAGE, SENTIMENT, UnifiedFly,
                                        move_targets, sentiment_targets)
from test_connectorch_model import reference_fixture


class UnifiedModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.brain = build_connectorch_model(reference_fixture(), csr_factory=TrainableCSR,
                                             d_embed=4).brain
        self.fly = UnifiedFly(self.brain, settle_steps=3, readout_rank=3, value_bins=8,
                              features=5, tokens=11, moves=7, classes=2)

    def test_every_task_shares_one_output_space(self):
        width = 11 + 7 + 2
        tokens, _, _ = self.fly.language(torch.tensor([[1, 3, 2]]))
        moves, _, _ = self.fly.chess(torch.randn(4, 5))
        judgement, _ = self.fly.sentiment(torch.tensor([[1, 3, 2]]), torch.ones(1, 3))
        for logits in (tokens, moves, judgement):
            self.assertEqual(logits.shape[-1], width)

    def test_each_task_owns_a_disjoint_range_that_covers_the_space(self):
        ranges = self.fly.ranges()
        self.assertEqual(ranges[LANGUAGE], (0, 11))
        self.assertEqual(ranges[CHESS], (11, 18))
        self.assertEqual(ranges[SENTIMENT], (18, 20))
        covered = sorted(i for start, stop in ranges.values() for i in range(start, stop))
        self.assertEqual(covered, list(range(20)), "ranges must tile the output space exactly")

    def test_targets_land_in_their_own_range(self):
        self.assertTrue(torch.equal(move_targets(torch.tensor([0, 6]), tokens=11),
                                    torch.tensor([11, 17])))
        self.assertTrue(torch.equal(sentiment_targets(torch.tensor([0, 1]), tokens=11, moves=7),
                                    torch.tensor([18, 19])))

    def test_pooling_excludes_padding_from_both_sum_and_divisor(self):
        """A short row must be judged on its own tokens, not diluted by the batch width."""
        pooled = UnifiedFly(self.brain, settle_steps=3, readout_rank=3, value_bins=8,
                            features=5, tokens=11, moves=7, classes=2,
                            sentiment_pooling="mean")
        short, mask = torch.tensor([[1, 3, 2]]), torch.ones(1, 3)
        padded = torch.tensor([[1, 3, 2, 0, 0]])
        padded_mask = torch.tensor([[1., 1., 1., 0., 0.]])
        alone, _ = pooled.sentiment(short, mask)
        with_padding, _ = pooled.sentiment(padded, padded_mask)
        self.assertTrue(torch.allclose(alone, with_padding, atol=1e-6),
                        "padding leaked into the pooled summary")

    def test_pooling_uses_the_whole_sequence_not_just_the_end(self):
        """Changing an early token must move a pooled judgement. With leak 0.9 the last
        position barely remembers it, which is the reason pooling exists."""
        pooled = UnifiedFly(self.brain, settle_steps=3, readout_rank=3, value_bins=8,
                            features=5, tokens=11, moves=7, classes=2,
                            sentiment_pooling="mean")
        mask = torch.ones(1, 6)
        a = torch.tensor([[1, 3, 4, 5, 6, 2]])
        b = torch.tensor([[1, 9, 4, 5, 6, 2]])   # differs only at position 1
        first, _ = pooled.sentiment(a, mask)
        second, _ = pooled.sentiment(b, mask)
        self.assertFalse(torch.allclose(first, second, atol=1e-6),
                         "an early token had no effect on the pooled judgement")

    def test_last_token_pooling_is_still_available(self):
        last = UnifiedFly(self.brain, settle_steps=3, readout_rank=3, value_bins=8,
                          features=5, tokens=11, moves=7, classes=2,
                          sentiment_pooling="last")
        mean = UnifiedFly(self.brain, settle_steps=3, readout_rank=3, value_bins=8,
                          features=5, tokens=11, moves=7, classes=2,
                          sentiment_pooling="mean")
        mean.load_state_dict(last.state_dict())
        ids, mask = torch.tensor([[1, 3, 4, 5, 2]]), torch.ones(1, 5)
        self.assertFalse(torch.allclose(last.sentiment(ids, mask)[0],
                                        mean.sentiment(ids, mask)[0], atol=1e-6),
                         "the two poolings produced identical output")

    def test_an_unknown_pooling_is_refused(self):
        with self.assertRaises(ValueError):
            UnifiedFly(self.brain, readout_rank=3, features=5, tokens=11, moves=7,
                       classes=2, sentiment_pooling="first")

    def test_sentiment_reads_the_last_real_token_not_the_padding(self):
        """A short sequence in a padded batch must be judged where its text ends."""
        short = torch.tensor([[1, 3, 2]])
        padded = torch.tensor([[1, 3, 2, 0, 0]])
        last = UnifiedFly(self.brain, settle_steps=3, readout_rank=3, value_bins=8,
                          features=5, tokens=11, moves=7, classes=2,
                          sentiment_pooling="last")
        alone, _ = last.sentiment(short, torch.ones(1, 3))
        with_padding, _ = last.sentiment(padded, torch.tensor([[1., 1., 1., 0., 0.]]))
        self.assertTrue(torch.allclose(alone, with_padding, atol=1e-6),
                        "padding moved the read position")

    def test_the_delay_history_advances_across_chunks(self):
        """The cue must not cost the brain its eight-token memory.

        The brain builds its next cache from input_ids, not from the embeddings, so a
        readout that hands it embeddings alone freezes the delay history at padding and
        every chunk after the first is driven by tokens that were never read. Nothing
        downstream notices: the loss stays finite and the model still trains.
        """
        first, second = torch.tensor([[1, 3, 4, 2]]), torch.tensor([[5, 6, 7, 8]])
        _, _, cache = self.fly.language(first)
        self.assertFalse(bool((cache.last_tokens == 0).all()),
                         "the delay history never advanced past padding")
        expected = self.brain(first, use_cache=True, return_dict=True).cache_params
        self.assertTrue(torch.equal(cache.last_tokens, expected.last_tokens))
        # And it keeps advancing on the next chunk.
        _, _, after = self.fly.language(second, cache_params=cache)
        self.assertFalse(torch.equal(after.last_tokens, cache.last_tokens))

    def test_language_matches_the_pinned_path_when_no_cue_is_used(self):
        """With the cue off, the readout must see exactly the pinned model's hidden state."""
        plain = UnifiedFly(self.brain, settle_steps=3, readout_rank=3, value_bins=8,
                           features=5, tokens=11, moves=7, classes=0, tasks=2, task_cue=False)
        ids = torch.tensor([[1, 3, 4, 2]])
        expected = self.brain(ids, use_cache=True, return_dict=True)
        _, _, cache = plain.language(ids)
        self.assertTrue(torch.equal(cache.last_tokens, expected.cache_params.last_tokens))

    def test_the_task_cue_changes_what_the_brain_receives(self):
        ids = torch.tensor([[1, 3, 2]])
        self.assertFalse(torch.allclose(self.fly.embed(ids, LANGUAGE),
                                        self.fly.embed(ids, SENTIMENT)),
                         "language and sentiment reach the brain identically; nothing "
                         "distinguishes the two questions")

    def test_the_same_text_can_be_answered_two_ways(self):
        ids, mask = torch.tensor([[1, 3, 4, 2]]), torch.ones(1, 4)
        continued, _, _ = self.fly.language(ids, attention_mask=mask)
        judged, _ = self.fly.sentiment(ids, mask)
        self.assertFalse(torch.allclose(continued[:, -1], judged, atol=1e-6))

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

    def test_leakage_is_the_mass_outside_the_tasks_own_range(self):
        # A head certain about one language token leaks nothing as language, everything
        # as either other task.
        certain = torch.full((1, 20), -30.0)
        certain[0, 2] = 30.0
        self.assertLess(self.fly.leakage(certain, LANGUAGE), 1e-6)
        for task in (CHESS, SENTIMENT):
            self.assertGreater(self.fly.leakage(certain, task), 1 - 1e-6)
        # A uniform head leaks whatever share of the space is not its own.
        uniform = torch.zeros(1, 20)
        self.assertAlmostEqual(self.fly.leakage(uniform, LANGUAGE), 9 / 20, places=5)
        self.assertAlmostEqual(self.fly.leakage(uniform, CHESS), 13 / 20, places=5)
        self.assertAlmostEqual(self.fly.leakage(uniform, SENTIMENT), 18 / 20, places=5)

    def test_own_parameters_excludes_the_shared_brain(self):
        own = {id(p) for p in self.fly.own_parameters()}
        for name in ("gain", "rec_gain", "bias", "in_proj"):
            self.assertNotIn(id(getattr(self.brain, name)), own)
        self.assertIn(id(self.fly.head.weight), own)
        self.assertIn(id(self.fly.task_vector.weight), own)

    def test_the_brain_is_not_registered_twice(self):
        holder = torch.nn.Module()
        holder.brain, holder.unified = self.brain, self.fly
        self.assertFalse(any(n.startswith("unified.brain") for n in holder.state_dict()))


if __name__ == "__main__":
    unittest.main()
