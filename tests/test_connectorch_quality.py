"""Inference-helper contracts on tiny CPU/fake fixtures; no optimizer or training."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("connectorch_quality", ROOT / "scripts/evaluate_connectorch_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class FakeTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [3, 4]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(value) for value in ids if value not in (0, 1, 2))


class FakeModel(nn.Module):
    """Deterministic logits expose token/cache handling and full target counts."""
    def __init__(self, force_eos=True):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.calls = []
        self.force_eos = force_eos

    def forward(self, input_ids, attention_mask, cache_params, use_cache, return_dict):
        self.calls.append({"ids": input_ids.tolist(), "cache": cache_params, "grad_enabled": torch.is_grad_enabled()})
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8)
        for index in range(length):
            targets = (input_ids[:, index] + 1) % 8
            if self.force_eos and cache_params is not None:
                targets[:] = 2
            logits[torch.arange(batch), index, targets] = 4
        return SimpleNamespace(logits=logits, cache_params=SimpleNamespace(seq_len=length + (cache_params.seq_len if cache_params else 0)))


def metric(ce, correct=2):
    return {"cross_entropy": ce, "top1_accuracy": correct / 5, "correct": correct, "tokens": 5, "stories": 2}


def test_generation_includes_bos_once_fresh_cache_eos_and_greedy_cap():
    model = FakeModel()
    samples = quality.generate(model, FakeTokenizer(), device="cpu", prompts=["one", "two"])
    assert [sample["new_ids"] for sample in samples] == [[5, 2], [5, 2]]
    assert samples[0]["prompt_ids"] == [1, 3, 4]
    assert samples[0]["all_ids"] == [1, 3, 4, 5, 2]
    assert model.calls[0]["cache"] is None and model.calls[2]["cache"] is None
    assert model.calls[1]["ids"] == [[5]]
    assert not any(call["grad_enabled"] for call in model.calls)
    capped = quality.generate(FakeModel(False), FakeTokenizer(), "cpu", ["one"], max_new_tokens=1)
    assert capped[0]["new_tokens"] == 1 and not capped[0]["stopped_on_eos"]


def test_validation_padding_chunk_boundaries_and_replay_counts():
    rows = [{"id": "a", "ids": [1, 3, 4, 5, 2]}, {"id": "b", "ids": [1, 2]}]
    model = FakeModel(False)
    result = quality.score_validation(model, rows, "cpu", batch_size=2, chunk_size=2)
    summary = result["summary"]
    assert summary["stories"] == 2 and summary["tokens"] == 5 and summary["correct"] == 3
    assert model.calls[0]["cache"] is None and model.calls[1]["cache"].seq_len == 2
    assert not any(call["grad_enabled"] for call in model.calls)
    assert quality.verify_replay(summary, summary)["passed"]
    with pytest.raises(ValueError, match="count/accuracy"):
        quality.verify_replay(summary, {**summary, "correct": 2})
    with pytest.raises(ValueError, match="CE replay"):
        quality.verify_replay(summary, {**summary, "cross_entropy": summary["cross_entropy"] + .01})


def test_training_overlap_excludes_prompt_and_reports_exact_prefix():
    samples = [{"prompt": "One day", "continuation": "a little cat slept", "text": "One day a little cat slept"}]
    train = [{"id": "first", "text": "One day a little cat slept beside the fire"},
             {"id": "tie", "text": "a little cat slept elsewhere"}]
    result = quality.training_overlap(samples, train)[0]
    assert result["longest_contiguous_match"]["words"] == 4
    assert result["longest_contiguous_match"]["training_story_id"] == "first"
    assert result["exact_prefix_training_story_ids"] == ["first"]


def test_process_guard_allows_zombies_and_paused_dispatcher_but_not_live_trainers():
    records = quality.parse_processes("12 1 Z <defunct>\n13 1 Ts python scripts/continue_connectorch_campaign.py --worker\n14 1 S python scripts/evaluate_connectorch_quality.py")
    audit = quality.process_guard(records, worker_pids={12}, own_pid=14)
    assert len(audit["exited_zombies"]) == len(audit["paused_dispatchers"]) == 1
    for state in ("S", "Ts"):
        with pytest.raises(ValueError, match="exited training"):
            quality.process_guard(quality.parse_processes(f"12 1 {state} python scripts/train_connectorch.py"), {12})
    with pytest.raises(ValueError, match="held dispatchers"):
        quality.process_guard(quality.parse_processes("13 1 S python scripts/continue_connectorch_campaign.py --worker"))
    quality.process_guard(quality.parse_processes("15 1 S sh -c python scripts/evaluate_connectorch_quality.py"))


def test_host_guard_rejects_non_m3_or_wrong_host(monkeypatch):
    monkeypatch.setattr(quality.socket, "gethostname", lambda: "some-other-host.local")
    monkeypatch.setattr(quality.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="registered macm3"):
        quality.idle_host_guard(set())
    monkeypatch.setattr(quality.socket, "gethostname", lambda: "Fernandos-MacBook-Pro-2.local")
    monkeypatch.setattr(quality.subprocess, "check_output", lambda *args, **kwargs: "Apple M4 Max")
    with pytest.raises(ValueError, match="Apple M3"):
        quality.idle_host_guard(set())


def test_selector_verification_uses_independent_extrema_and_rejects_mutation(tmp_path):
    rows = [{"updates": 100, "epoch": 0, "validation": metric(4)},
            {"updates": 200, "epoch": 0, "validation": metric(5, 3)}]
    graph = {"brain.w_values": "graph"}
    selectors = {}
    for (selector, filename), row in zip(quality.SELECTORS.items(), rows):
        path = tmp_path / filename
        path.write_bytes(filename.encode())
        selectors[selector] = {"path": str(path), "sha256": quality.trainer.file_hash(path),
                               "cursor": {"updates": row["updates"], "epoch": 0}, "validation": row["validation"],
                               "frozen_buffers_preserved": True, "frozen_buffers_sha256": graph}
    receipt = tmp_path / "selected-checkpoints.json"
    receipt.write_text(json.dumps({"format_version": 1, "selectors": selectors}))
    assert quality.verify_selectors(tmp_path, {"frozen_buffers_sha256": graph}, rows, {}) == selectors
    (tmp_path / "best-accuracy.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Bound file"):
        quality.verify_selectors(tmp_path, {"frozen_buffers_sha256": graph}, rows, {})


def test_checkpoint_parameter_hashes_and_manifest_binding():
    graph = {"brain.w_values": "graph"}
    values = {"weight": torch.tensor([1., 2.])}
    cursor = {"updates": 100, "epoch": 0}
    validation = metric(4)
    manifest = {"frozen_buffers_sha256": graph}
    entry = {"cursor": cursor, "validation": validation, "frozen_buffers_sha256": graph}
    arm = {"manifest": manifest, "selectors": {"minimum_validation_ce": entry}}
    saved = {"format_version": 1, "manifest": manifest, "cursor": cursor, "parameters": values,
             "parameter_audit": quality.trainer.parameter_audit(values), "frozen_buffers_sha256": graph,
             "model_audit": {"frozen_buffers_preserved": True, "frozen_buffers_sha256": graph},
             "best": {**cursor, "cross_entropy": 4, "top1_accuracy": .4}}
    quality.verify_checkpoint(saved, arm, "minimum_validation_ce")
    damaged = copy.deepcopy(saved)
    damaged["parameters"]["weight"][0] = 9
    with pytest.raises(ValueError, match="parameter contents"):
        quality.verify_checkpoint(damaged, arm, "minimum_validation_ce")
    damaged = copy.deepcopy(saved)
    damaged["manifest"]["different"] = True
    with pytest.raises(ValueError, match="manifest differs"):
        quality.verify_checkpoint(damaged, arm, "minimum_validation_ce")


def test_architecture_restoration_checks_names_shapes_graph_and_width():
    # Reuse the tiny graph supplier only; no fixture training or real inference.
    from test_connectorch_training import TinyReference
    reference = TinyReference()
    groups = torch.arange(16) % 4
    config = {"d_embed": 32, "plasticity": "fixed", "history_length": 8, "readout_rank": 0, "seed": 42}
    model = quality.trainer.build_model(reference, SimpleNamespace(**config), groups)
    values = quality.trainer.parameters_cpu(model)
    manifest = {"config": config, "parameter_counts": quality.trainer.parameter_counts(model),
                "frozen_buffers_sha256": quality.trainer.frozen_hashes(model),
                "groups_provenance": {"frozen_buffers_sha256": quality.trainer.frozen_hashes(reference)}}
    arm = {"manifest": manifest, "groups": groups}
    restored = quality.restore_model(reference, arm, {"parameters": values})
    assert restored.config.d_embed == 32 and restored.config.lag_map == list(range(8))
    assert quality.trainer.parameter_audit(quality.trainer.parameters_cpu(restored)) == quality.trainer.parameter_audit(values)
    bad = dict(values, extra=torch.ones(1))
    with pytest.raises(ValueError, match="parameter names"):
        quality.restore_model(reference, arm, {"parameters": bad})
    bad = dict(values, **{"brain.in_proj": torch.ones(1)})
    with pytest.raises(ValueError, match="shape mismatch"):
        quality.restore_model(reference, arm, {"parameters": bad})
    damaged = copy.deepcopy(arm)
    damaged["manifest"]["config"]["d_embed"] = 128
    with pytest.raises(ValueError, match="parameter count"):
        quality.restore_model(reference, damaged, {"parameters": values})


def test_practical_flags_use_direction_and_both_metrics():
    assert not quality.compare_scores(metric(4), metric(4.05))["review_flag"]
    assert quality.compare_scores(metric(4), metric(4.2))["ce_loss_over_0_10"]
    assert quality.compare_scores(metric(4, 3), metric(3, 2))["accuracy_loss_over_1pp"]
