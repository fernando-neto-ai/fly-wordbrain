"""Gate the unified trainer through the seams it actually patches.

Every check here runs the real patched functions on real batch tensors, because the
failure this arm is most exposed to is silent: a loss that looks finite while the chess
term leaks into the validation number, or a parameter that quietly stops receiving
gradient. Both have already happened once in this project.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_connectorch as trainer
import train_unified
from fly_wordbrain.chess_encoding import pack_board
from fly_wordbrain.metal_sparse_trainable import TrainableCSR


NEURONS, DELAY, INPUTS, OUTPUTS, VOCAB = 16, 8, 16, 8, 11


def eight_slot_fixture():
    """A toy reference with the real delay structure.

    The trainer always builds an eight-entry lag map, so a fixture with a different
    delay_k cannot be driven through it at all -- which is the point of testing through
    the trainer rather than around it.
    """
    from types import SimpleNamespace
    from torch import nn
    generator = torch.Generator().manual_seed(1729)
    sources = []
    for destination in range(NEURONS):
        choices = [n for n in range(NEURONS) if n != destination]
        picked = torch.randperm(len(choices), generator=generator)[:2]
        sources.extend(sorted(choices[i] for i in picked.tolist()))
    offsets = torch.arange(0, 2 * NEURONS + 1, 2, dtype=torch.int32)
    config = SimpleNamespace(n_neurons=NEURONS, n_edges=2 * NEURONS, n_in=INPUTS, n_out=OUTPUTS,
                             d_embed=4, delay_k=DELAY, vocab_size=VOCAB, pad_token_id=0,
                             leak=.7, rec_target=5., use_cache=True, return_dict=True)
    model = nn.Module()
    model.config = config
    model.brain = nn.Module()
    brain = model.brain
    brain.wte = nn.Embedding(VOCAB, 4, padding_idx=0)
    brain.in_proj = nn.Parameter(torch.randn(DELAY, 4, INPUTS // DELAY, generator=generator) * .2)
    for name, value in (("gain", torch.linspace(.8, 1.2, NEURONS)),
                        ("rec_gain", torch.linspace(.7, 1.1, NEURONS)),
                        ("bias", torch.linspace(-.1, .1, NEURONS))):
        brain.register_parameter(name, nn.Parameter(value))
    for name, value in (("w_offsets", offsets),
                        ("w_indices", torch.tensor(sources, dtype=torch.int32)),
                        ("w_values", torch.linspace(-.5, .5, 2 * NEURONS)),
                        ("in_index", torch.arange(INPUTS)),
                        ("out_index", torch.arange(OUTPUTS))):
        brain.register_buffer(name, value.clone())
    model.ln = nn.LayerNorm(OUTPUTS)
    model.lm_head = nn.Linear(OUTPUTS, VOCAB, bias=False)
    return model


def tiny_corpus(directory, positions=24):
    generator = np.random.default_rng(1729)
    counts = generator.integers(2, 6, size=positions)
    arrays = {
        # Real corpora only ever hold piece codes 0..12; random bytes would not.
        "boards": pack_board(generator.integers(0, 13, size=(positions, 64)).astype(np.uint8)),
        "extras": np.zeros((positions, 2), np.uint8),
        "moves": generator.integers(0, 1968, size=positions).astype(np.uint16),
        "values": generator.random(positions).astype(np.float32),
        "span_train": np.array([0, 12]), "span_validation": np.array([12, 18]),
        "span_test": np.array([18, 21]), "span_audit": np.array([21, positions]),
    }
    for name, (start, stop) in (("validation", (12, 18)), ("test", (18, 21)),
                                ("audit", (21, positions))):
        sizes = counts[start:stop]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        indices = generator.integers(0, 1968, size=int(offsets[-1])).astype(np.uint16)
        # The target must be reachable, exactly as the real corpus guarantees.
        for row, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            indices[begin] = arrays["moves"][start + row]
        arrays[f"legal_offsets_{name}"] = offsets
        arrays[f"legal_indices_{name}"] = indices
    np.savez(Path(directory) / "corpus.npz", **arrays)
    return Path(directory)


class UnifiedTrainerTests(unittest.TestCase):
    def setUp(self):
        self.originals = {name: getattr(trainer, name) for name in
                          ("build_model", "expected_parameter_counts", "optimizer_for", "chunk_loss")}
        torch.manual_seed(1729)
        self.directory = tempfile.TemporaryDirectory()
        corpus = tiny_corpus(self.directory.name)
        self.args = Namespace(chess_corpus=corpus, chess_batch=4, settle_steps=2,
                              language_weight=1.0, chess_weight=1.0, router_weight=0.1,
                              sentiment_corpus=None, sentiment_batch=4, sentiment_weight=1.0,
                              sentiment_pooling="mean",
                              toxicity_corpus=None, toxicity_batch=4, toxicity_weight=1.0)
        train_unified.install(self.args)
        train_unified.STATE["weights"] = {"language": 1.0, "chess": 1.0, "router": 0.1,
                                          "sentiment": 0.0}
        train_unified.STATE["log"] = None
        train_unified.STATE["sentiment"] = None
        train_unified.STATE["batches"] = train_unified.ChessBatches(corpus, "cpu", 4, 42)
        self.training_args = Namespace(d_embed=4, plasticity="fixed", readout_rank=2,
                                       history_length=DELAY, seed=42)
        self.groups = np.arange(NEURONS) % 3
        reference = eight_slot_fixture()
        self.model = trainer.build_model(reference, self.training_args, self.groups)
        for module in self.model.modules():
            if hasattr(module, "_csr_factory") and module._csr_factory is None:
                module._csr_factory = TrainableCSR
        self.model.brain._csr_factory = TrainableCSR
        self.model.train()

    def tearDown(self):
        for name, value in self.originals.items():
            setattr(trainer, name, value)
        self.directory.cleanup()

    def batch(self):
        rows = [{"ids": [1, 3, 4, 2, 5]}, {"ids": [1, 8, 2, 0, 0]}]
        return trainer.batch_tensors(rows, "cpu", self.model.config.pad_token_id)

    def test_the_audit_the_real_run_performs_passes(self):
        counts = trainer.parameter_counts(self.model)
        expected = trainer.expected_parameter_counts(self.model, self.training_args, self.groups)
        self.assertEqual(counts, expected)

    def test_the_unused_language_readout_is_frozen_and_uncounted(self):
        self.assertFalse(self.model.lm_head[0].weight.requires_grad)
        self.assertEqual(trainer.parameter_counts(self.model)["readout"], 0)
        self.assertEqual(trainer.parameter_counts(self.model)["layernorm"], 0)

    def test_a_training_step_reaches_every_shared_and_private_parameter(self):
        inputs, targets, mask = self.batch()
        loss, count, correct, _ = trainer.chunk_loss(self.model, inputs, targets, mask, None,
                                                     self.model.config.pad_token_id)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(count, 0)
        loss.backward()
        unified = train_unified.STATE["unified"]
        for name, tensor in (("brain.gain", self.model.brain.gain),
                             ("brain.in_proj", self.model.brain.in_proj),
                             ("board_adapter", unified.board_adapter.weight),
                             ("head", unified.head.weight),
                             ("router", unified.router.weight),
                             ("value", unified.value.weight)):
            self.assertIsNotNone(tensor.grad, f"no gradient reached {name}")
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0, f"{name} got a zero gradient")

    def test_evaluation_draws_no_chess_batch_and_stays_pure_language(self):
        inputs, targets, mask = self.batch()
        self.model.eval()
        before = train_unified.STATE["batches"].cursor
        loss, count, correct, _ = trainer.chunk_loss(self.model, inputs, targets, mask, None,
                                                     self.model.config.pad_token_id)
        self.assertEqual(train_unified.STATE["batches"].cursor, before,
                         "evaluation consumed chess positions, so validation is not pure language")
        # A pure language loss over a 2,992-way head starts near log(2992), not near the
        # joint objective, which is what the two-head arm got wrong.
        value = float(loss.detach())
        self.assertLess(value, 9.5)
        self.assertGreater(value, 6.0)

    def test_trainer_evaluate_runs_end_to_end_on_the_patched_loss(self):
        rows = [{"ids": [1, 3, 4, 2]}, {"ids": [1, 8, 2, 5]}]
        report = trainer.evaluate(self.model, rows, "cpu", batch_size=2, chunk_size=2,
                                  pad_id=self.model.config.pad_token_id)
        self.assertGreater(report["tokens"], 0)
        self.assertTrue(np.isfinite(report["cross_entropy"]))

    def test_optimizer_groups_the_board_encoder_with_the_brain(self):
        optimizer = trainer.optimizer_for(self.model, 0)
        unified = train_unified.STATE["unified"]
        groups = {group["name"]: {id(p) for p in group["params"]} for group in optimizer.param_groups}
        self.assertIn(id(unified.board_adapter.weight), groups["input_and_neurons"])
        self.assertIn(id(unified.head.weight), groups["readout_and_norm"])
        self.assertIn(id(self.model.brain.in_proj), groups["input_and_neurons"])
        # Frozen parameters must not be handed to the optimizer at all.
        every = groups["input_and_neurons"] | groups["readout_and_norm"]
        self.assertNotIn(id(self.model.lm_head[0].weight), every)

    def test_the_record_decomposes_the_joint_objective(self):
        inputs, targets, mask = self.batch()
        total, language, count, correct, _, record = train_unified.unified_losses(
            train_unified.STATE["unified"], (inputs, targets, mask, None),
            train_unified.STATE["batches"].next(), train_unified.STATE["weights"],
            self.model.config.pad_token_id, None)
        for key in ("language_loss", "chess_policy_loss", "chess_value_loss", "router_loss",
                    "router_accuracy", "language_leakage", "chess_leakage"):
            self.assertIn(key, record)
        self.assertGreater(float(total), float(language),
                           "the joint objective must exceed its language term")
        # Leakage is a probability mass.
        for key in ("language_leakage", "chess_leakage"):
            self.assertGreaterEqual(record[key], 0.0)
            self.assertLessEqual(record[key], 1.0)

    def test_chess_evaluation_reports_masked_accuracy_within_the_move_range(self):
        report = train_unified.evaluate_chess(train_unified.STATE["unified"],
                                              train_unified.STATE["batches"], "audit", "cpu")
        # A two-task arm must stay two-task: no sentiment range, no task cue, 2-lane router.
        unified = train_unified.STATE["unified"]
        self.assertEqual(unified.classes, 0)
        self.assertIsNone(unified.task_vector)
        self.assertEqual(unified.router.out_features, 2)
        self.assertEqual(report["positions"], 3)
        for key in ("policy_cross_entropy", "top1_legal_masked", "value_mae", "router_accuracy",
                    "leakage_into_language"):
            self.assertIn(key, report)
        self.assertGreaterEqual(report["top1_legal_masked"], 0.0)
        self.assertLessEqual(report["top1_legal_masked"], 1.0)


if __name__ == "__main__":
    unittest.main()


def tiny_sentiment(directory, rows=24, vocab=11):
    """A minimal labelled corpus in the shape prepare_sentiment.py writes."""
    generator = np.random.default_rng(4242)
    def make(count, offset):
        out = []
        for index in range(count):
            length = int(generator.integers(3, 7))
            body = generator.integers(3, vocab, size=length).tolist()
            out.append({"id": f"toy/{offset + index}", "ids": [1, *body, 2],
                        "label": int(index % 2), "tokens": length + 2})
        return out
    data = {"train": make(rows, 0), "validation": make(6, 100),
            "test": make(6, 200), "audit": make(6, 300)}
    path = Path(directory) / "dataset.json"
    path.write_text(json.dumps(data))
    return Path(directory)


class ThreeTaskTrainerTests(unittest.TestCase):
    """The third task changes the model's shape, so the trainer has to build it that way."""

    def setUp(self):
        self.originals = {name: getattr(trainer, name) for name in
                          ("build_model", "expected_parameter_counts", "optimizer_for", "chunk_loss")}
        torch.manual_seed(1729)
        self.directory = tempfile.TemporaryDirectory()
        chess = tiny_corpus(self.directory.name)
        folder = Path(self.directory.name) / "sentiment"
        folder.mkdir(exist_ok=True)
        sentiment = tiny_sentiment(folder)
        self.args = Namespace(chess_corpus=chess, chess_batch=4, settle_steps=2,
                              language_weight=1.0, chess_weight=1.0, router_weight=0.1,
                              sentiment_corpus=sentiment, sentiment_batch=4,
                              sentiment_weight=1.0, sentiment_pooling="mean",
                              toxicity_corpus=None, toxicity_batch=4, toxicity_weight=1.0)
        train_unified.install(self.args)
        train_unified.STATE["weights"] = {"language": 1.0, "chess": 1.0, "router": 0.1,
                                          "sentiment": 1.0}
        train_unified.STATE["log"] = None
        train_unified.STATE["batches"] = train_unified.ChessBatches(chess, "cpu", 4, 42)
        train_unified.STATE["sentiment"] = train_unified.SentimentBatches(
            sentiment, "cpu", 4, 42)
        self.training_args = Namespace(d_embed=4, plasticity="fixed", readout_rank=2,
                                       history_length=DELAY, seed=42)
        self.groups = np.arange(NEURONS) % 3
        self.model = trainer.build_model(eight_slot_fixture(), self.training_args, self.groups)
        self.model.brain._csr_factory = TrainableCSR
        self.model.train()

    def tearDown(self):
        for name, value in self.originals.items():
            setattr(trainer, name, value)
        self.directory.cleanup()

    def test_the_model_is_built_in_three_task_form(self):
        unified = train_unified.STATE["unified"]
        self.assertEqual(unified.classes, 2)
        self.assertIsNotNone(unified.task_vector)
        self.assertEqual(unified.router.out_features, 3)
        self.assertEqual(unified.head.out_features, unified.tokens + unified.moves + 2)

    def test_the_audit_still_accounts_for_every_parameter(self):
        counts = trainer.parameter_counts(self.model)
        expected = trainer.expected_parameter_counts(self.model, self.training_args, self.groups)
        self.assertEqual(counts, expected)

    def test_a_step_decomposes_all_three_tasks_and_reaches_the_task_cue(self):
        rows = [{"ids": [1, 3, 4, 2, 5]}, {"ids": [1, 8, 2, 0, 0]}]
        inputs, targets, mask = trainer.batch_tensors(rows, "cpu", self.model.config.pad_token_id)
        loss, count, correct, _ = trainer.chunk_loss(self.model, inputs, targets, mask, None,
                                                     self.model.config.pad_token_id)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        unified = train_unified.STATE["unified"]
        gradient = unified.task_vector.weight.grad
        self.assertIsNotNone(gradient, "no gradient reached the task cue")
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_the_record_carries_the_sentiment_terms(self):
        rows = [{"ids": [1, 3, 4, 2, 5]}, {"ids": [1, 8, 2, 0, 0]}]
        inputs, targets, mask = trainer.batch_tensors(rows, "cpu", self.model.config.pad_token_id)
        _, _, _, _, _, record = train_unified.unified_losses(
            train_unified.STATE["unified"], (inputs, targets, mask, None),
            train_unified.STATE["batches"].next(), train_unified.STATE["weights"],
            self.model.config.pad_token_id, train_unified.STATE["sentiment"].next())
        for key in ("sentiment_loss", "sentiment_accuracy", "sentiment_leakage"):
            self.assertIn(key, record)
        self.assertGreaterEqual(record["sentiment_accuracy"], 0.0)
        self.assertLessEqual(record["sentiment_accuracy"], 1.0)

    def test_the_trainer_built_the_pooled_readout(self):
        self.assertEqual(train_unified.STATE["unified"].sentiment_pooling, "mean")

    def test_sentiment_is_scored_inside_its_own_range(self):
        report = train_unified.evaluate_sentiment(
            train_unified.STATE["unified"], train_unified.STATE["sentiment"], "audit", "cpu")
        self.assertEqual(report["rows"], 6)
        # Two classes: an argmax restricted to the sentiment range can only ever be 0 or 1,
        # so accuracy is a real rate rather than a near-zero one over 2,994 outputs.
        self.assertGreaterEqual(report["accuracy"], 0.0)
        self.assertLessEqual(report["accuracy"], 1.0)
        self.assertIn("leakage_out_of_range", report)

    def test_a_short_sequence_is_judged_where_its_text_ends(self):
        batches = train_unified.STATE["sentiment"]
        rows = [{"ids": [1, 3, 2], "label": 0}, {"ids": [1, 4, 5, 6, 2], "label": 1}]
        ids, mask, _ = batches.tensors(rows)
        self.assertEqual(ids.shape[1], 5)
        self.assertEqual(mask[0].tolist(), [1., 1., 1., 0., 0.])
        unified = train_unified.STATE["unified"]
        together, _ = unified.sentiment(ids, mask)
        alone, _ = unified.sentiment(*batches.tensors([rows[0]])[:2])
        self.assertTrue(torch.allclose(together[0], alone[0], atol=1e-6),
                        "padding changed where the short sequence was read")


class FourTaskTrainerTests(unittest.TestCase):
    """The fourth task widens the head again, and must not disturb the third.

    Sentiment and toxicity are both pooled binary judgements over token streams, which is
    exactly why they are easy to confuse in code: they share a forward pass, differ only
    in cue and output range, and a wrong offset would train toxicity on sentiment's two
    outputs without any shape error to reveal it.
    """

    def setUp(self):
        self.originals = {name: getattr(trainer, name) for name in
                          ("build_model", "expected_parameter_counts", "optimizer_for", "chunk_loss")}
        torch.manual_seed(1729)
        self.directory = tempfile.TemporaryDirectory()
        chess = tiny_corpus(self.directory.name)
        sentiment_dir = Path(self.directory.name) / "sentiment"
        sentiment_dir.mkdir(exist_ok=True)
        sentiment = tiny_sentiment(sentiment_dir)
        toxicity_dir = Path(self.directory.name) / "toxicity"
        toxicity_dir.mkdir(exist_ok=True)
        toxicity = tiny_sentiment(toxicity_dir)
        self.args = Namespace(chess_corpus=chess, chess_batch=4, settle_steps=2,
                              language_weight=1.0, chess_weight=1.0, router_weight=0.1,
                              sentiment_corpus=sentiment, sentiment_batch=4,
                              sentiment_weight=1.0, sentiment_pooling="mean",
                              toxicity_corpus=toxicity, toxicity_batch=4, toxicity_weight=1.0)
        train_unified.install(self.args)
        train_unified.STATE["weights"] = {"language": 1.0, "chess": 1.0, "router": 0.1,
                                          "sentiment": 1.0, "toxicity": 1.0}
        train_unified.STATE["log"] = None
        train_unified.STATE["batches"] = train_unified.ChessBatches(chess, "cpu", 4, 42)
        train_unified.STATE["sentiment"] = train_unified.SentimentBatches(sentiment, "cpu", 4, 42)
        train_unified.STATE["toxicity"] = train_unified.SentimentBatches(toxicity, "cpu", 4, 42)
        self.training_args = Namespace(d_embed=4, plasticity="fixed", readout_rank=2,
                                       history_length=DELAY, seed=42)
        self.groups = np.arange(NEURONS) % 3
        self.model = trainer.build_model(eight_slot_fixture(), self.training_args, self.groups)
        self.model.brain._csr_factory = TrainableCSR
        self.model.train()

    def tearDown(self):
        for name, value in self.originals.items():
            setattr(trainer, name, value)
        train_unified.STATE.clear()
        self.directory.cleanup()

    def test_the_head_and_router_widen_for_the_fourth_task(self):
        unified = self.model.unified
        self.assertEqual(unified.router.out_features, 4)
        self.assertEqual(unified.toxicity_classes, 2)
        self.assertEqual(unified.head.out_features,
                         unified.tokens + unified.moves + unified.classes + 2)

    def test_each_task_owns_a_disjoint_output_range(self):
        ranges = self.model.unified.ranges()
        spans = sorted(ranges.values())
        for (a_start, a_stop), (b_start, b_stop) in zip(spans, spans[1:]):
            self.assertLessEqual(a_stop, b_start, "task output ranges overlap")
        self.assertEqual(spans[-1][1], self.model.unified.head.out_features)

    def test_toxicity_targets_land_above_the_sentiment_range(self):
        from fly_wordbrain.unified_model import SENTIMENT, TOXICITY, pooled_targets
        unified = self.model.unified
        labels = torch.tensor([0, 1])
        sentiment_start = unified.ranges()[SENTIMENT][0]
        toxicity_start = unified.ranges()[TOXICITY][0]
        self.assertGreater(toxicity_start, sentiment_start)
        moved = pooled_targets(labels, unified, TOXICITY)
        self.assertTrue(bool((moved >= toxicity_start).all()))
        self.assertTrue(bool((moved < unified.head.out_features).all()))

    def one_record(self):
        rows = [{"ids": [1, 3, 4, 2, 5]}, {"ids": [1, 8, 2, 0, 0]}]
        inputs, targets, mask = trainer.batch_tensors(rows, "cpu", self.model.config.pad_token_id)
        _, _, _, _, _, record = train_unified.unified_losses(
            train_unified.STATE["unified"], (inputs, targets, mask, None),
            train_unified.STATE["batches"].next(), train_unified.STATE["weights"],
            self.model.config.pad_token_id,
            train_unified.STATE["sentiment"].next(), train_unified.STATE["toxicity"].next())
        return record

    def test_the_record_carries_the_toxicity_terms(self):
        record = self.one_record()
        for key in ("toxicity_loss", "toxicity_accuracy", "toxicity_leakage",
                    "sentiment_loss", "sentiment_accuracy"):
            self.assertIn(key, record, f"{key} missing from the training record")
        self.assertGreaterEqual(record["toxicity_accuracy"], 0.0)
        self.assertLessEqual(record["toxicity_accuracy"], 1.0)

    def test_the_router_is_supervised_on_all_four_tasks(self):
        record = self.one_record()
        self.assertIn("router_accuracy", record)
        self.assertEqual(self.model.unified.router.out_features, 4)
