"""Tiny graph oracles for sparse orientation, gradients and fast-state semantics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from fly_wordbrain.plastic_brain import PlasticBrain, PlasticState, frozen_sparse_mm, sorted_csr_columns


def toy_graph(path):
    n = 12
    pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11], np.int32)
    # The canonical row for presynaptic neuron8 is deliberately unsorted.
    post = np.array([8, 8, 9, 9, 10, 10, 11, 11, 10, 9, 10, 11, 8], np.int32)
    weights = np.array([.7, -.4, .6, .4, -.5, .8, .9, .3, .5, -.3, .6, .5, -.2], np.float32)
    ptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=n))]
    incoming_ids = np.argsort(post, kind="stable")
    incoming_ptr = np.r_[0, np.cumsum(np.bincount(post, minlength=n))]
    selected = np.array([0, 2, 4, 6], np.int64)
    np.savez(path / "graph.npz", ids=np.arange(n), ptr=ptr, post=post, weight=weights,
             incoming_ptr=incoming_ptr, incoming_pre=pre[incoming_ids],
             incoming_weight=weights[incoming_ids], incoming_edge_ids=incoming_ids,
             retina=np.arange(8), descending=np.arange(8, 12),
             candidate_edge_ids=selected, candidate_pre=pre[selected], candidate_post=post[selected],
             candidate_group=np.arange(4))
    (path / "metadata.json").write_text(json.dumps({"toy_graph": True}))
    dense = np.zeros((n, n), np.float32)
    np.add.at(dense, (post, pre), weights)
    return torch.from_numpy(dense)


class PlasticBrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.dense = toy_graph(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def brain(self, **kwargs):
        return PlasticBrain(self.path, vocab_size=8, global_scale=.5, strict_full_graph=False,
                            input_gain=20., internal_steps=4, **kwargs)

    def test_incoming_orientation_and_state_gradient_match_dense_oracle(self):
        brain = self.brain()
        h = torch.randn(3, 12, requires_grad=True)
        output = frozen_sparse_mm(h, brain.incoming, brain.outgoing)
        oracle = h @ self.dense.T
        torch.testing.assert_close(output, oracle)
        probe = torch.randn_like(output)
        gradient = torch.autograd.grad((output * probe).sum(), h)[0]
        expected = probe @ self.dense
        torch.testing.assert_close(gradient, expected)
        self.assertFalse(brain.incoming.requires_grad)
        self.assertFalse(brain.outgoing.requires_grad)

    def test_runtime_transpose_is_sorted_without_changing_canonical_edges(self):
        brain = self.brain()
        with np.load(self.path / "graph.npz") as source:
            np.testing.assert_array_equal(brain.raw_outgoing_ptr.numpy(), source["ptr"])
            np.testing.assert_array_equal(brain.raw_outgoing_post.numpy(), source["post"])
            np.testing.assert_array_equal(brain.raw_outgoing_weight.numpy(), source["weight"])
            np.testing.assert_array_equal(brain.candidate_weight.numpy(), source["weight"][source["candidate_edge_ids"]])
        self.assertEqual(brain.outgoing.crow_indices().dtype, brain.outgoing.col_indices().dtype)
        torch.testing.assert_close(brain.outgoing.to_dense(), self.dense.T)
        self.assertEqual(brain.outgoing._nnz(), 13)
        ptr, col = brain.outgoing.crow_indices().numpy(), brain.outgoing.col_indices().numpy()
        for start, end in zip(ptr[:-1], ptr[1:]):
            self.assertTrue(np.all(np.diff(col[start:end]) > 0))

    def test_parallel_edges_are_rejected_instead_of_coalesced(self):
        with self.assertRaisesRegex(ValueError, "cannot be coalesced"):
            sorted_csr_columns(np.array([0, 2, 2]), np.array([1, 1]), np.array([.1, .2]))

    def test_sparse_base_never_changes_endpoints_or_signs(self):
        brain = self.brain()
        before = brain.verify_frozen()
        values = brain.effective_candidate_weights(torch.tensor([[-1e6, -2., 2., 1e6]]))
        self.assertTrue(torch.equal(torch.sign(values[0]), torch.sign(brain.candidate_weight)))
        ratios = values / brain.candidate_weight
        self.assertTrue(torch.all(ratios >= .5) and torch.all(ratios <= 2.))
        _, state = brain.step([4, 5], [6, 7], brain.initial_state(2))
        brain.step([6, 7], [4, 5], state)
        self.assertEqual(brain.verify_frozen(), before)

    def test_batch_independence_and_prefix_causality(self):
        brain = self.brain()
        batch_features, batch_state = brain.step([2, 3], [4, 5], brain.initial_state(2))
        for i, (previous, current) in enumerate(((2, 4), (3, 5))):
            features, state = brain.step([previous], [current], brain.initial_state(1))
            torch.testing.assert_close(features[0], batch_features[i])
            torch.testing.assert_close(state.h[0], batch_state.h[i])
            torch.testing.assert_close(state.fast[0], batch_state.fast[i])
        untouched_h = batch_state.h.clone()
        untouched_fast = batch_state.fast.clone()
        brain.step([4, 5], [6, 7], batch_state)
        torch.testing.assert_close(batch_state.h, untouched_h)
        torch.testing.assert_close(batch_state.fast, untouched_fast)

    def test_activity_and_fast_resets_are_separate(self):
        brain = self.brain()
        state = PlasticState(torch.ones(2, 12), torch.ones(2, 4))
        activity_reset = brain.reset_activity(state)
        fast_reset = brain.reset_fast(state)
        self.assertTrue(torch.all(activity_reset.h == 0))
        self.assertIs(activity_reset.fast, state.fast)
        self.assertTrue(torch.all(fast_reset.fast == 0))
        self.assertIs(fast_reset.h, state.h)

    def test_fast_rule_gets_nonzero_gradients_and_only_twenty_parameters_train(self):
        brain = self.brain()
        brain.set_activity_scales(.1, .1)
        state = PlasticState(torch.full((2, 12), .2), torch.full((2, 4), .1))
        features, new_state = brain.step([4, 5], [6, 7], state)
        # Objective depends only on output features: gradients through a direct
        # fast-state penalty would pass even if fast weights never affect h.
        loss = features.square().sum()
        loss.backward()
        self.assertEqual(brain.trainable_parameter_count(), 20)
        for name, parameter in brain.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        self.assertEqual(brain.verify_frozen(), brain.initial_graph_fingerprint)

    def test_override_off_is_static_rate_oracle_and_does_not_write_fast_state(self):
        brain = self.brain()
        state = PlasticState(torch.full((1, 12), .2), torch.full((1, 4), 2.))
        features, result = brain.step([4], [5], state, plasticity_override=False)
        h = state.h
        current = brain.retinal_current([4], [5])
        for _ in range(4):
            h = .5 * h + .5 * torch.tanh(.5 * (h @ self.dense.T) + current)
        torch.testing.assert_close(result.h, h)
        torch.testing.assert_close(features, brain.readout(h))
        self.assertIs(result.fast, state.fast)
        static = self.brain(plasticity=False)
        self.assertEqual(static.trainable_parameter_count(), 0)
        static_features, _ = static.step([4], [5], state)
        torch.testing.assert_close(features, static_features)

    def test_bad_incoming_orientation_and_wrong_full_graph_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "166700"):
            PlasticBrain(self.path, vocab_size=8, global_scale=.5)
        graph_path = self.path / "graph.npz"
        with np.load(graph_path) as z:
            arrays = {key: z[key].copy() for key in z.files}
        arrays["incoming_weight"][0] += 1.
        np.savez(graph_path, **arrays)
        with self.assertRaisesRegex(ValueError, "orientation/weights"):
            self.brain()

    def test_frozen_verification_rejects_mutated_graph_and_interface(self):
        brain = self.brain()
        with torch.no_grad():
            brain.incoming.values()[0] += .1
        with self.assertRaisesRegex(RuntimeError, "connectome"):
            brain.verify_frozen()
        other = self.brain()
        with torch.no_grad():
            other.previous_codes[0, 0] += .1
        with self.assertRaisesRegex(RuntimeError, "interface"):
            other.verify_frozen()


if __name__ == "__main__":
    unittest.main()
