"""Small CPU-only inference and statistical checks for the quality audit."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "ngxson_quality_audit", Path(__file__).parents[1] / "scripts/evaluate_ngxson_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def record(name, tokens, nll, correct):
    return {"id": name, "tokens": tokens, "nll_sum": nll, "correct": correct}


def test_aggregate_uses_tokens_instead_of_mean_story_loss():
    result = quality.aggregate([record("a", 1, 2., 0), record("b", 9, 9., 9)])
    assert result["cross_entropy"] == pytest.approx(1.1)
    assert result["top1_accuracy"] == pytest.approx(.9)
    assert result["perplexity"] == pytest.approx(torch.exp(torch.tensor(1.1)).item())
    with pytest.raises(ValueError, match="Empty"):
        quality.aggregate([])


def test_paired_metrics_are_json_serializable_and_identical_is_equivalent():
    records = [record("a", 4, 8., 2), record("b", 40, 42., 32)]
    result = quality.paired_comparison(records, records, resamples=100)
    json.dumps(result, allow_nan=False)
    assert result["delta_cross_entropy"] == 0
    assert result["delta_accuracy"] == 0
    assert result["ce_95_ci"] == [0., 0.]
    assert result["accuracy_95_ci"] == [0., 0.]
    assert result["two_sided_equivalence_supported"] is True
    assert result["noninferiority_supported"] is True


def test_story_bootstrap_keeps_token_weighting_and_pairing():
    reference = [record("a", 10, 10., 5), record("b", 100, 200., 50)]
    ours = [record("a", 10, 12., 5), record("b", 100, 200., 50)]
    result = quality.paired_comparison(ours, reference, resamples=10000, seed=42)
    assert result["delta_cross_entropy"] == pytest.approx(2 / 110)
    assert result["ce_95_ci"] == pytest.approx([0., .2])
    assert result["point_estimates_within_margins"]
    assert not result["two_sided_equivalence_supported"]
    assert not result["noninferiority_supported"]
    assert result == quality.paired_comparison(ours, reference, resamples=10000, seed=42)


def test_better_is_noninferior_but_not_necessarily_two_sided_equivalent():
    reference = [record("a", 10, 20., 4), record("b", 100, 200., 40)]
    ours = [record("a", 10, 10., 4), record("b", 100, 100., 40)]
    result = quality.paired_comparison(ours, reference, resamples=100)
    assert result["ce_95_ci"] == [-1., -1.]
    assert result["noninferiority_supported"] is True
    assert result["two_sided_equivalence_supported"] is False


@pytest.mark.parametrize("other", [record("b", 10, 1., 0), record("a", 11, 1., 0)])
def test_unpaired_ids_or_counts_are_rejected(other):
    with pytest.raises(ValueError, match="Paired"):
        quality.paired_comparison([record("a", 10, 1., 0)], [other], resamples=10)


class TinyInferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.weight = nn.Parameter(torch.arange(44, dtype=torch.float32).reshape(4, 11) / 30)
        self.register_buffer("embedding", torch.arange(44, dtype=torch.float32).reshape(11, 4) / 40)
        self.cache_inputs = []

    def forward(self, input_ids, attention_mask, cache_params=None, **kwargs):
        assert not torch.is_grad_enabled()
        self.cache_inputs.append(cache_params)
        state = torch.zeros(len(input_ids), 4) if cache_params is None else cache_params.state
        logits = []
        for t in range(input_ids.shape[1]):
            proposed = torch.tanh(.6 * state + self.embedding[input_ids[:, t]])
            mask = attention_mask[:, t:t + 1]
            state = mask * proposed + (1 - mask) * state
            logits.append(state @ self.weight)
        return SimpleNamespace(logits=torch.stack(logits, dim=1), cache_params=SimpleNamespace(state=state))


ROWS = [{"id": "a", "ids": [1, 3, 4, 5, 6, 2]},
        {"id": "b", "ids": [1, 7, 2]},
        {"id": "c", "ids": [1, 8, 9, 2]}]


def test_scoring_matches_independent_whole_stories_with_padding_and_chunks(tmp_path):
    model = TinyInferenceModel()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    result = quality.score(model, ROWS, "fixture", tmp_path, batch_size=2, chunk_size=2)
    for scored, row in zip(result["per_story"], ROWS):
        ids = torch.tensor([row["ids"]])
        with torch.inference_mode():
            logits = model(ids[:, :-1], torch.ones_like(ids[:, :-1])).logits
            expected = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="sum")
        assert scored["tokens"] == len(row["ids"]) - 1
        assert scored["nll_sum"] == pytest.approx(float(expected), abs=2e-6)
        assert scored["correct"] == int((logits.argmax(-1) == ids[:, 1:]).sum())
    assert result["summary"]["tokens"] == 10
    assert model.cache_inputs[0] is None
    assert model.cache_inputs[1] is not None
    assert model.cache_inputs[2] is not None
    assert model.cache_inputs[3] is None
    assert not model.training
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])


def test_scoring_rejects_nonfinite_logits(tmp_path):
    model = TinyInferenceModel()
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        quality.score(model, ROWS, "fixture", tmp_path, batch_size=2, chunk_size=2)


def test_generation_uses_once_only_bos_fixed_greedy_config_and_all_prompts():
    class Tokenizer:
        bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

        def encode(self, prompt, add_special_tokens):
            assert add_special_tokens is False
            return [3, 4]

        def decode(self, tokens, skip_special_tokens):
            assert skip_special_tokens is True
            return " ".join(str(t) for t in tokens if t not in (0, 1, 2))

    class Model:
        def generate(self, ids, **kwargs):
            assert not torch.is_grad_enabled()
            assert ids.tolist() == [[1, 3, 4]]
            assert kwargs == {"max_new_tokens": 80, "do_sample": False,
                              "pad_token_id": 0, "eos_token_id": 2}
            return torch.tensor([[1, 3, 4, 5, 2]])

    samples = quality.generate(Model(), Tokenizer())
    assert [s["prompt"] for s in samples] == quality.PROMPTS
    assert all(s["new_tokens"] == 2 and s["continuation"] == "5" for s in samples)


def test_accuracy_checkpoint_uses_its_own_validation_not_embedded_ce_winner():
    checkpoint = {"cursor": {"updates": 200, "epoch": 2},
                  "best": {"updates": 100, "cross_entropy": 3.0}}
    metric = {"cross_entropy": 4.0, "top1_accuracy": .4, "correct": 4,
              "tokens": 10, "stories": 2}
    receipt = {"checkpoint_sha256": "abc", "validation_record": {
        "event": "validation", "updates": 200, "epoch": 2, "validation": metric}}
    assert quality.selected_validation(checkpoint, "max-accuracy", receipt, "abc") == metric
    with pytest.raises(ValueError, match="minimum-CE"):
        quality.selected_validation(checkpoint, "min-ce", None, "abc")
    with pytest.raises(ValueError, match="SHA"):
        quality.selected_validation(checkpoint, "max-accuracy", receipt, "different")
    receipt["validation_record"]["updates"] = 201
    with pytest.raises(ValueError, match="cursor"):
        quality.selected_validation(checkpoint, "max-accuracy", receipt, "abc")


@pytest.mark.parametrize("accuracy", [.5, float("nan")])
def test_accuracy_receipt_rejects_inconsistent_or_nonfinite_metrics(accuracy):
    checkpoint = {"cursor": {"updates": 200, "epoch": 2}}
    receipt = {"checkpoint_sha256": "abc", "validation_record": {
        "event": "validation", "updates": 200, "epoch": 2,
        "validation": {"cross_entropy": 4.0, "top1_accuracy": accuracy,
                       "correct": 4, "tokens": 10, "stories": 2}}}
    with pytest.raises(ValueError, match="Accuracy"):
        quality.selected_validation(checkpoint, "max-accuracy", receipt, "abc")
