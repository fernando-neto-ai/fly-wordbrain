"""The linear probe used to control the router, and the floor that makes it readable.

The settled state has far more dimensions than the probe has samples, so the training
split is separable no matter what the labels are. Everything therefore rests on the
held-out gap between a real-label probe and a shuffled-label one. If that control did not
work, a shortcut would read as a finding.
"""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_task_identity import TASK_NAMES, linear_probe, router_class_names


def two_clouds(count, width, separation, seed=1729):
    generator = torch.Generator().manual_seed(seed)
    offset = torch.zeros(width)
    offset[0] = separation
    a = torch.randn(count, width, generator=generator)
    b = torch.randn(count, width, generator=generator) + offset
    labels = torch.cat([torch.zeros(count, dtype=torch.long), torch.ones(count, dtype=torch.long)])
    return torch.cat([a, b]), labels


class ProbeTests(unittest.TestCase):
    def test_a_real_difference_is_found(self):
        # Two unit Gaussians offset by 3.0 in one of 40 dimensions. The Bayes-optimal
        # classifier uses that dimension alone and errs at Phi(-1.5), so the ceiling is
        # about 93.3%; the probe has to come near it, not exceed it.
        states, labels = two_clouds(300, 40, separation=3.0)
        train, test = linear_probe(states, labels)
        self.assertGreater(test, .85)
        self.assertLess(test, .95)
        self.assertGreater(train, .9)

    def test_pure_noise_does_not_generalise(self):
        states, labels = two_clouds(300, 40, separation=0.0)
        _, test = linear_probe(states, labels)
        self.assertLess(test, .65, "the probe claims signal in noise")

    def test_more_dimensions_than_samples_separates_the_training_split_regardless(self):
        """The reason the shuffled control exists, stated as a test."""
        states, labels = two_clouds(60, 4000, separation=0.0)
        train, test = linear_probe(states, labels)
        self.assertGreater(train, .95, "expected the wide regime to be trivially separable")
        self.assertLess(test, .70, "a wide noise probe must not generalise")

    def test_shuffling_the_labels_destroys_a_real_effect(self):
        states, labels = two_clouds(300, 40, separation=3.0)
        _, honest = linear_probe(states, labels)
        generator = torch.Generator().manual_seed(4242)
        shuffled = labels[torch.randperm(labels.shape[0], generator=generator)]
        _, control = linear_probe(states, shuffled)
        self.assertGreater(honest - control, .3,
                           "the shuffled control did not separate signal from fitting")

    def test_the_probe_is_deterministic(self):
        states, labels = two_clouds(200, 30, separation=2.0)
        self.assertEqual(linear_probe(states, labels), linear_probe(states, labels))


class RouterClassNames(unittest.TestCase):
    """A confusion table shorter than the router loses predictions silently.

    Measured on L32fourtaskstrainableleak: a three-name table read a four-class router
    and reported 44 of 100 language states, so the dropped 56 -- every state called
    toxicity -- never appeared, and chess looked like the failure mode when toxicity was.
    """

    @staticmethod
    def router(width, features=8):
        return torch.nn.Linear(features, width)

    def test_names_cover_every_class_a_four_task_router_emits(self):
        names = router_class_names(self.router(4))
        self.assertEqual(sorted(names), [0, 1, 2, 3])
        self.assertEqual(names[3], "toxicity")

    def test_a_narrower_router_is_named_to_its_own_width(self):
        self.assertEqual(sorted(router_class_names(self.router(2))), [0, 1])

    def test_an_unnameable_class_refuses_rather_than_dropping_it(self):
        with self.assertRaises(SystemExit):
            router_class_names(self.router(len(TASK_NAMES) + 1))

    def test_every_prediction_lands_in_a_named_column(self):
        names = router_class_names(self.router(4))
        predicted = torch.tensor([0, 1, 1, 2, 3, 3, 3])
        row = {names[choice]: int((predicted == choice).sum())
               for choice in sorted(names) if int((predicted == choice).sum())}
        self.assertEqual(sum(row.values()), predicted.shape[0])
        self.assertEqual(row["toxicity"], 3)


if __name__ == "__main__":
    unittest.main()
