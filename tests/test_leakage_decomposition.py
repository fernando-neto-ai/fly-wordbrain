"""The identity that lets a unified arm be compared to a single-task one.

A unified arm predicts over 2,992 outputs while every earlier arm predicted over 1,024, so
its cross-entropy is not comparable as it stands. The report splits it:

    CE_full     = -z_t + log sum(exp(z))          over all 2,992
    CE_language = -z_t + log sum(exp(z[:1024]))   over the token range, renormalised
    difference  = log sum(all) - log sum(language) = -log(mass on the language range)

so the difference is exactly what the model pays for leaving probability outside the range
its answer lives in. That is an identity, not an approximation, and the write-up leans on
it -- so it is checked numerically rather than asserted.
"""
import unittest

import torch
import torch.nn.functional as F

TOKENS, MOVES = 1024, 1968


class LeakageDecompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.logits = torch.randn(64, TOKENS + MOVES) * 3
        self.targets = torch.randint(0, TOKENS, (64,))

    def difference(self):
        full = F.cross_entropy(self.logits, self.targets, reduction="none")
        language = F.cross_entropy(self.logits[:, :TOKENS], self.targets, reduction="none")
        return full - language

    def test_the_difference_is_minus_log_of_the_language_mass(self):
        mass = self.logits.softmax(-1)[:, :TOKENS].sum(-1)
        self.assertTrue(torch.allclose(self.difference(), -mass.log(), atol=1e-5))

    def test_it_is_never_negative(self):
        """Renormalising onto a subset can only lower the loss, so the cost has one sign."""
        self.assertTrue(bool((self.difference() >= -1e-6).all()))

    def test_no_leakage_means_no_cost(self):
        confident = self.logits.clone()
        confident[:, TOKENS:] = -1e4
        full = F.cross_entropy(confident, self.targets, reduction="none")
        language = F.cross_entropy(confident[:, :TOKENS], self.targets, reduction="none")
        self.assertTrue(torch.allclose(full, language, atol=1e-5))

    def test_a_uniform_head_pays_the_uniform_cost(self):
        flat = torch.zeros(8, TOKENS + MOVES)
        targets = torch.zeros(8, dtype=torch.long)
        full = F.cross_entropy(flat, targets, reduction="none")
        language = F.cross_entropy(flat[:, :TOKENS], targets, reduction="none")
        expected = torch.full((8,), -torch.tensor(TOKENS / (TOKENS + MOVES)).log().item())
        self.assertTrue(torch.allclose(full - language, expected, atol=1e-5))

    def test_the_cost_grows_as_mass_moves_to_the_other_range(self):
        costs = []
        for boost in (-6.0, 0.0, 6.0):
            shifted = self.logits.clone()
            shifted[:, TOKENS:] += boost
            full = F.cross_entropy(shifted, self.targets, reduction="none").mean()
            language = F.cross_entropy(shifted[:, :TOKENS], self.targets, reduction="none").mean()
            costs.append(float(full - language))
        self.assertTrue(costs[0] < costs[1] < costs[2], costs)


if __name__ == "__main__":
    unittest.main()
