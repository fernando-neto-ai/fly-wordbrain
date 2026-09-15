"""CPU toy-circuit oracles for the candidate-aware action prototype."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from fly_wordbrain.action_model import CandidateActionBrain, CandidateInput, FlyActionSelector
from fly_wordbrain.plastic_brain import PlasticBrain


def make_graph(path):
    n, r = 64, 56
    pre = np.arange(n, dtype=np.int32)
    post = 56 + pre % 8
    weights = (.2 + (pre % 7) * .1).astype(np.float32)
    weights[pre % 3 == 0] *= -1
    order = np.lexsort((pre, post))
    selected = np.array([0, 1, 2, 3], dtype=np.int64)
    np.savez(path / "graph.npz", ids=np.arange(n), ptr=np.arange(n + 1), post=post,
             weight=weights, incoming_ptr=np.r_[0, np.cumsum(np.bincount(post, minlength=n))],
             incoming_pre=pre[order], incoming_weight=weights[order], incoming_edge_ids=order,
             retina=np.arange(r), descending=np.arange(r, n), candidate_edge_ids=selected,
             candidate_pre=pre[selected], candidate_post=post[selected], candidate_group=np.arange(4))
    (path / "metadata.json").write_text(json.dumps({"toy_graph": True}))


class ActionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        make_graph(self.path)
        self.kwargs = dict(vocab_size=16, global_scale=.5, strict_full_graph=False,
                           internal_steps=4, input_gain=10.)
        self.brain = CandidateActionBrain(self.path, **self.kwargs)
        self.model = FlyActionSelector(self.brain)
        self.previous = torch.tensor([2, 4])
        self.current = torch.tensor([5, 6])
        self.candidates = torch.tensor([[4, 6, 7, 8, 9], [7, 8, 4, 5, 9]])
        self.probs = torch.tensor([[.4, .2, .1, .08, .02], [.1, .35, .2, .1, .05]])

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_baseline_logits_and_argmax_at_initialization(self):
        logits, state = self.model.step(self.previous, self.current, self.candidates,
                                        self.probs, self.model.initial_state(2))
        self.assertTrue(torch.equal(logits, self.probs.log()))
        self.assertTrue(torch.equal(logits.argmax(-1), self.probs.argmax(-1)))
        self.assertEqual(self.model.trainable_parameter_count(), 1285)
        self.assertEqual(self.brain.trainable_parameter_count(), 0)
        self.assertFalse(state.h.requires_grad)
        self.assertEqual([name for name, p in self.model.named_parameters() if p.requires_grad],
                         ["readout.weight", "readout.bias"])
        tied = torch.full((2, 5), .1)
        logits, _ = self.model.step(self.previous, self.current, self.candidates,
                                    tied, self.model.initial_state(2))
        self.assertEqual(logits.argmax(-1).tolist(), [0, 0])

    def test_single_affine_zero_initialization_has_nonzero_learning_signal(self):
        fingerprint = self.brain.verify_frozen()
        logits, _ = self.model.step(self.previous, self.current, self.candidates,
                                    self.probs, self.model.initial_state(2))
        loss = F.cross_entropy(logits, torch.tensor([2, 4]))
        loss.backward()
        for parameter in self.model.readout.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in self.brain.parameters()))
        torch.optim.SGD(self.model.readout.parameters(), lr=.1).step()
        changed, _ = self.model.step(self.previous, self.current, self.candidates,
                                     self.probs, self.model.initial_state(2))
        self.assertFalse(torch.equal(changed, logits))
        self.assertEqual(self.brain.verify_frozen(), fingerprint)

    def test_all_roles_have_disjoint_banks_and_exact_identity_probability_mapping(self):
        brain = self.brain
        current = brain.retinal_current(CandidateInput(self.previous, self.candidates, self.probs), self.current)
        ids = torch.cat((self.previous[:, None], self.current[:, None], self.candidates), dim=1)
        amplitudes = torch.cat((torch.ones(2, 2), .5 + .5 * self.probs), dim=1)
        self.assertEqual(len(brain.action_neurons.unique()), len(brain.retina))
        self.assertEqual(brain.action_codes.numel(), 16 * 56)
        self.assertFalse(hasattr(brain, "previous_codes"))
        for role, (start, end) in enumerate(zip(brain.bank_offsets[:-1], brain.bank_offsets[1:])):
            expected = brain.action_codes[ids[:, role], start:end] * amplitudes[:, role, None] * brain.input_gain
            torch.testing.assert_close(current[:, brain.action_neurons[start:end]], expected, rtol=0, atol=0)
        self.assertTrue(torch.all(current[:, brain.descending] == 0))
        self.assertTrue(torch.equal(brain.retina.sort().values, brain.action_neurons.sort().values))

    def test_candidate_identity_probability_and_order_reach_features(self):
        inputs = (self.previous, self.current, self.candidates, self.probs)
        base, _ = self.brain.step(*inputs, self.model.initial_state(2))
        changed_ids = self.candidates.clone()
        changed_ids[0, 2] = 15
        changed_probs = self.probs.clone()
        changed_probs[0, 2] *= .1
        permuted = self.candidates.flip(1)
        for candidate_ids, probs in ((changed_ids, self.probs), (self.candidates, changed_probs), (permuted, self.probs.flip(1))):
            features, _ = self.brain.step(self.previous, self.current, candidate_ids, probs, self.model.initial_state(2))
            self.assertTrue(torch.isfinite(features).all())
            self.assertGreater(float((features - base).abs().max()), 1e-7)

    def test_parent_graph_buffers_and_neural_update_are_exactly_preserved(self):
        original = PlasticBrain(self.path, **self.kwargs, plasticity=False)
        self.assertEqual(original.frozen_fingerprint(), self.brain.frozen_fingerprint())
        for name in ("raw_outgoing_ptr", "raw_outgoing_post", "raw_outgoing_weight", "candidate_edge_ids", "readout_bucket", "readout_sign"):
            self.assertTrue(torch.equal(getattr(original, name), getattr(self.brain, name)), name)
        current = self.brain.retinal_current(CandidateInput(self.previous, self.candidates, self.probs), self.current)
        features, result = self.brain.step(self.previous, self.current, self.candidates,
                                         self.probs, self.model.initial_state(2))
        h = torch.zeros_like(result.h)
        dense = original.incoming.to_dense()
        for _ in range(original.internal_steps):
            h = (1-original.leak)*h + original.leak*torch.tanh(original.global_scale*(h @ dense.T) + current)
        torch.testing.assert_close(result.h, h)
        torch.testing.assert_close(features, original.readout(h))

    def test_persistent_state_batch_independence_and_padding_mask(self):
        args = (self.previous, self.current, self.candidates, self.probs)
        _, first = self.model.step(*args, self.model.initial_state(2))
        saved = first.h.clone()
        _, second = self.model.step(*args, first, active=torch.tensor([True, False]))
        self.assertTrue(torch.equal(first.h, saved))
        self.assertTrue(torch.equal(second.h[1], first.h[1]))
        self.assertFalse(torch.equal(second.h[0], first.h[0]))
        for row in range(2):
            row_args = tuple(value[row:row+1] for value in args)
            _, alone = self.model.step(*row_args, self.model.initial_state(1))
            torch.testing.assert_close(alone.h[0], first.h[row])
        self.assertTrue(torch.equal(second.fast, first.fast))

    def test_validation_rejects_invalid_candidates_probabilities_and_masks(self):
        cases = [
            (self.candidates.float(), self.probs, "integer"),
            (self.candidates[:, :4], self.probs, "shape"),
            (torch.zeros_like(self.candidates), self.probs, "distinct"),
            (torch.tensor([[0, 6, 7, 8, 9], [7, 8, 4, 5, 9]]), self.probs, "PAD/BOS"),
            (torch.tensor([[2, 6, 7, 8, 9], [7, 8, 4, 5, 9]]), self.probs, "PAD/BOS"),
            (self.candidates + 16, self.probs, "vocabulary"),
            (self.candidates, torch.zeros_like(self.probs), "positive"),
            (self.candidates, torch.full_like(self.probs, float("nan")), "finite"),
            (self.candidates, torch.full_like(self.probs, .3), "at most one"),
        ]
        for ids, probs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.model.step(self.previous, self.current, ids, probs, self.model.initial_state(2))
        for mask in ([1, 0], [True], [[True], [False]]):
            with self.assertRaisesRegex(ValueError, "boolean"):
                self.model.step(self.previous, self.current, self.candidates, self.probs,
                                self.model.initial_state(2), active=mask)

    def test_calibration_and_encoder_are_fixed_and_auditable(self):
        mean = torch.ones(256, requires_grad=True)
        std = torch.full((256,), 2., requires_grad=True)
        calibrated = FlyActionSelector(self.brain, mean, std)
        self.assertFalse(calibrated.feature_mean.requires_grad)
        self.assertFalse(calibrated.feature_std.requires_grad)
        mean.detach().fill_(7.)
        self.assertTrue(torch.all(calibrated.feature_mean == 1))
        self.assertTrue(calibrated.metadata()["features_calibrated"])
        with self.assertRaisesRegex(ValueError, "both"):
            FlyActionSelector(self.brain, mean)
        with self.assertRaisesRegex(ValueError, "positive"):
            FlyActionSelector(self.brain, mean, torch.zeros(256))
        self.brain.action_codes[0, 0] += .1
        with self.assertRaisesRegex(RuntimeError, "interface"):
            self.brain.verify_frozen()

    def test_top_ten_keeps_encoder_size_and_expands_actions(self):
        brain = CandidateActionBrain(self.path, top_k=10, **self.kwargs)
        model = FlyActionSelector(brain)
        ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        probs = torch.tensor([[.16, .12, .10, .08, .07, .06, .05, .04, .03, .02]])
        logits, _ = model.step(torch.tensor([2]), torch.tensor([4]), ids, probs, model.initial_state(1))
        self.assertEqual(tuple(logits.shape), (1, 10))
        self.assertEqual(model.trainable_parameter_count(), 2570)
        self.assertEqual(brain.action_codes.numel(), self.brain.action_codes.numel())
        self.assertEqual(len(brain.bank_sizes), 12)
        self.assertEqual(model.metadata()["actions"], 10)
        torch.testing.assert_close(logits, probs.log(), rtol=0, atol=0)
        F.cross_entropy(logits, torch.tensor([9])).backward()
        self.assertGreater(float(model.readout.weight.grad.abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, "top_k"):
            CandidateActionBrain(self.path, top_k=2.5, **self.kwargs)


if __name__ == "__main__":
    unittest.main()
