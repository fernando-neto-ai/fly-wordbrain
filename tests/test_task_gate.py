"""The gate decides which task an input is, so its class mapping is load-bearing.

`UnifiedFly`'s router cannot make this decision: the task cue is added in embedding space
before the brain runs, so the trunk it reads already encodes the task the caller declared.
Using it to choose a task is circular. The gate sees the raw input instead, and whatever
it predicts selects the cue, the settle path and the output range -- so a mapping error
here does not raise, it silently answers the wrong question.
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fly_wordbrain.task_gate import (CHESS, LANGUAGE, SENTIMENT, TOKEN_TASKS, TOXICITY,
                                     LengthOnlyGate, TaskGate, route)
from fly_wordbrain import unified_model


class TaskIdentityTests(unittest.TestCase):
    def test_the_gate_and_the_model_agree_on_what_each_task_id_means(self):
        # Two modules carry these constants. If they ever drift, the gate would route a
        # sentiment input into the toxicity range and nothing would raise.
        self.assertEqual(LANGUAGE, unified_model.LANGUAGE)
        self.assertEqual(CHESS, unified_model.CHESS)
        self.assertEqual(SENTIMENT, unified_model.SENTIMENT)
        self.assertEqual(TOXICITY, unified_model.TOXICITY)

    def test_chess_is_not_a_gate_class(self):
        # It is dispatched by shape; including it would inflate any reported accuracy.
        self.assertNotIn(CHESS, TOKEN_TASKS)
        self.assertEqual(TOKEN_TASKS, (LANGUAGE, SENTIMENT, TOXICITY))


class ClassMappingTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.randint(0, 1024, (16, 32))
        self.mask = torch.ones(16, 32)

    def test_a_two_class_gate_never_predicts_toxicity(self):
        gate = TaskGate(classes=2)
        self.assertEqual(gate.task_ids.tolist(), [LANGUAGE, SENTIMENT])
        predicted = set(gate.classify(self.ids, self.mask).tolist())
        self.assertTrue(predicted <= {LANGUAGE, SENTIMENT})

    def test_a_three_class_gate_predicts_real_task_ids_not_class_indices(self):
        gate = TaskGate(classes=3)
        self.assertEqual(gate.task_ids.tolist(), [LANGUAGE, SENTIMENT, TOXICITY])
        predicted = set(gate.classify(self.ids, self.mask).tolist())
        self.assertTrue(predicted <= set(TOKEN_TASKS))
        # Class index 1 means SENTIMENT (2), not task 1 (chess). Off-by-one here would
        # route every sentiment row to the chess range.
        self.assertNotEqual(gate.task_ids.tolist(), [0, 1, 2])

    def test_the_mapping_travels_with_the_weights(self):
        gate = TaskGate(classes=3)
        self.assertIn("task_ids", gate.state_dict())

    def test_a_stale_two_class_gate_refuses_to_load_as_three(self):
        old = TaskGate(classes=2).state_dict()
        with self.assertRaises(RuntimeError):
            TaskGate(classes=3).load_state_dict(old)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.gate = TaskGate(classes=3)

    def test_board_features_are_dispatched_by_shape_and_labelled_as_such(self):
        task, how = route(torch.randn(5, 780), self.gate)
        self.assertEqual(task.tolist(), [CHESS] * 5)
        self.assertEqual(how, "shape", "a type check must not be reported as a classification")

    def test_token_streams_are_classified_and_labelled_as_such(self):
        ids = torch.randint(0, 1024, (5, 32))
        task, how = route(None, self.gate, ids, torch.ones(5, 32))
        self.assertEqual(how, "classified")
        self.assertTrue(set(task.tolist()) <= set(TOKEN_TASKS))

    def test_routing_without_an_input_is_an_error_not_a_guess(self):
        with self.assertRaises(ValueError):
            route(None, self.gate)


class LengthControlTests(unittest.TestCase):
    """The control must see only length, or it is not a control."""

    def test_the_length_gate_gives_identical_answers_for_different_content(self):
        control = LengthOnlyGate(classes=3)
        mask = torch.ones(4, 32)
        a = control(torch.randint(0, 1024, (4, 32)), mask)
        b = control(torch.randint(0, 1024, (4, 32)), mask)
        self.assertTrue(torch.allclose(a, b),
                        "the length control responded to token content")

    def test_the_length_gate_responds_to_length(self):
        control = LengthOnlyGate(classes=3)
        ids = torch.randint(0, 1024, (2, 32))
        short = torch.zeros(2, 32); short[:, :4] = 1.0
        long = torch.ones(2, 32)
        self.assertFalse(torch.allclose(control(ids, short), control(ids, long)))


if __name__ == "__main__":
    unittest.main()
