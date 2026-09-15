"""Action training oracles: masks, causal proposals, and validation selection."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.action_model import CandidateActionBrain, FlyActionSelector
from fly_wordbrain.action_train import (
    ActionBatch, ActionMetrics, batch_logits, calibrate_features, collate_actions,
    evaluate, selection_improves, selection_loss, train, validate_training_validation,
)


def story(identifier, tokens):
    return {"id": identifier, "words": [str(t) for t in tokens],
            "word_ids": [2] + list(tokens) + [3], "target_mask": [False] + [True] * (len(tokens) + 1)}


def dataset():
    return {"vocabulary": ["<pad>", "<unk>", "<bos>", "<eos>"] + [str(i) for i in range(4, 16)],
            "splits": {"train": [story("a", [4, 5, 4, 6]), story("b", [4, 5, 7]),
                                  story("c", [8, 5, 4, 5, 8]), story("d", [9, 6, 7, 8])],
                       "val": [story("v", [4, 5, 10]), story("w", [8, 5, 7])],
                       "test": [{"deliberately_invalid": "must not be inspected"}]}}


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


@pytest.fixture(params=[5, 10])
def brain(tmp_path, request):
    torch.set_num_threads(1)
    make_graph(tmp_path)
    return CandidateActionBrain(tmp_path, 16, .5, strict_full_graph=False, input_gain=10., internal_steps=4,
                                top_k=request.param)


def test_whole_story_exclusion_matches_fresh_fit_and_validation_uses_all_training():
    rows = dataset()["splits"]["train"]
    proposals = ActionCandidates(rows, 16)
    batch = collate_actions(rows[:1], proposals, exclude_own_story=True)
    remaining = ActionCandidates(rows[1:], 16)
    for t in range(batch.previous.shape[1]):
        ids, probabilities = remaining.candidates(int(batch.previous[0, t]), int(batch.current[0, t]))
        np.testing.assert_array_equal(batch.candidates[0, t], ids)
        np.testing.assert_allclose(batch.probabilities[0, t], probabilities, rtol=1e-6)
    validation = collate_actions(dataset()["splits"]["val"], proposals)
    for t in range(validation.previous.shape[1]):
        ids, _ = proposals.candidates(int(validation.previous[0, t]), int(validation.current[0, t]))
        np.testing.assert_array_equal(validation.candidates[0, t], ids)


@pytest.mark.parametrize("top_k", [5, 10])
def test_missing_gold_is_excluded_from_loss_but_still_counts_as_an_accuracy_miss(top_k):
    prior = torch.arange(top_k, 0, -1, dtype=torch.float32)
    prior = prior / prior.sum() * .9
    batch = ActionBatch(torch.tensor([[2, 4, 5]]), torch.tensor([[4, 5, 6]]),
        torch.tensor([[list(range(4, 4 + top_k))] * 3]), prior.repeat(1, 3, 1),
        torch.tensor([[5, 15, 4]]), torch.tensor([[True, True, False]]), ("x",))
    logits = batch.probabilities.log().clone().requires_grad_()
    loss, count = selection_loss(logits, batch)
    assert count == 1
    loss.backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert torch.equal(logits.grad[0, 1:], torch.zeros_like(logits.grad[0, 1:]))
    # Promote action1, fixing first position. Missing gold stays absent.
    with torch.no_grad():
        logits[0, 0, 1] = 10
    metric = ActionMetrics()
    metric.update(logits, batch)
    summary = metric.summary()
    assert summary["target_count"] == 2
    assert summary["accuracy"] == .5
    assert summary["topk_coverage"] == .5
    assert summary["top_k"] == top_k
    assert summary["conditional_accuracy"] == 1.
    assert summary["fixes"] == 1 and summary["regressions"] == 0
    batch.targets.fill_(15)
    logits = batch.probabilities.log().clone().requires_grad_()
    zero, count = selection_loss(logits, batch)
    assert count == 0 and zero == 0
    zero.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_frozen_fly_and_count_baseline_are_exactly_equivalent_at_step_zero(brain):
    data = dataset()
    proposals = ActionCandidates(data["splits"]["train"], 16, top_k=brain.action_count)
    model = FlyActionSelector(brain)
    assert model.trainable_parameter_count() == 257 * brain.action_count
    actual = evaluate(model, data["splits"]["val"], proposals, batch_size=2)
    baseline = evaluate(None, data["splits"]["val"], proposals, batch_size=2, baseline_only=True)
    assert actual == baseline
    assert actual["fixes"] == actual["regressions"] == 0
    batch = collate_actions(data["splits"]["train"][:2], proposals, exclude_own_story=True)
    logits = batch_logits(model, batch)
    loss, _ = selection_loss(logits, batch)
    loss.backward()
    assert model.readout.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in brain.parameters())


def test_calibration_consumes_only_supplied_training_stories_and_finite_features(brain):
    rows = dataset()["splits"]["train"]
    proposals = ActionCandidates(rows, 16, top_k=brain.action_count)
    mean, std, receipt = calibrate_features(brain, rows[:2], proposals, batch_size=2)
    assert mean.shape == std.shape == (256,)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    assert receipt["story_ids"] == ["a", "b"]
    assert receipt["target_count"] == 9
    assert receipt["split"] == "train"
    with pytest.raises(ValueError, match="unknown training"):
        calibrate_features(brain, [story("val", [4, 5])], proposals)


def test_test_split_is_not_retrieved_and_only_full_validation_can_select():
    class GuardedSplits(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("test split accessed")
            return super().__getitem__(key)
    data = dataset()
    data["splits"] = GuardedSplits(data["splits"])
    validate_training_validation(data)
    baseline = {"scope": "full_validation", "accuracy": .3}
    assert not selection_improves({"scope": "monitor_subset", "accuracy": 1.}, baseline)
    assert not selection_improves({"scope": "full_validation", "accuracy": .3}, baseline)
    assert not selection_improves({"scope": "full_validation", "accuracy": .2}, baseline)
    assert selection_improves({"scope": "full_validation", "accuracy": .31}, baseline)


@pytest.mark.parametrize("top_k", [5, 10])
def test_cpu_training_produces_compact_checkpoints_and_separate_validation(tmp_path, top_k):
    torch.set_num_threads(1)
    graph = tmp_path / "graph"
    graph.mkdir()
    make_graph(graph)
    data_path = tmp_path / "dataset.json"
    data_path.write_text(json.dumps(dataset()))
    output = tmp_path / "run"
    result = train(data_path, graph, output, device="cpu", epochs=1, batch_size=2,
        monitor_stories=1, monitor_every=2, progress_every=1, calibration_stories=2,
        strict_full_graph=False, global_scale=.5, internal_steps=4, input_high=.02, top_k=top_k)
    assert result["global_step"] == 2
    assert result["test_evaluated"] is False
    assert result["top_k"] == top_k
    history = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert [(row["scope"], row["global_step"]) for row in history] == [
        ("full_validation", 0), ("monitor_subset", 0), ("monitor_subset", 1),
        ("monitor_subset", 2), ("full_validation", 2)]
    assert history[0]["accuracy"] == history[0]["baseline_accuracy"]
    assert all(row["kind"] == "action_selection" and "perplexity" not in row for row in history)
    assert all(row["top_k"] == top_k and "top5_coverage" not in row for row in history)
    for scope in ("full_validation", "monitor_subset"):
        rows = [row for row in history if row["scope"] == scope]
        assert len({row["baseline_accuracy"] for row in rows}) == 1
        assert len({row["topk_coverage"] for row in rows}) == 1
        assert len({row["subset_sha256"] for row in rows}) == 1
    payload = torch.load(output / "best.pt", weights_only=True)
    assert set(payload["parameters"]) == {"readout.weight", "readout.bias"}
    assert sum(t.numel() for t in payload["parameters"].values()) == 257 * top_k
    assert payload["top_k"] == top_k
    assert payload["protocol"]["brain"]["input_encoding"]["seed"] == payload["protocol"]["seed"]
    assert payload["selection_eligible"] is True
    assert payload["validation"]["scope"] == "full_validation"
    assert (output / "best.pt").stat().st_size < 200000
    with pytest.raises(ValueError, match="nonempty output"):
        train(data_path, graph, output, device="cpu")
