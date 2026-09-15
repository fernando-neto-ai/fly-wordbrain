"""Causal alignment and truncation tests independent of the large reference graph."""
import importlib.util
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F


spec = importlib.util.spec_from_file_location("ngxson_trainer", Path(__file__).parents[1] / "scripts/train_ngxson.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.embedding = nn.Embedding(12, 5, padding_idx=0)
        self.recurrent = nn.Linear(5, 5, bias=False)
        self.head = nn.Linear(5, 12)
        self.received_states = []

    def forward(self, input_ids, attention_mask, cache_params=None, **kwargs):
        batch = input_ids.shape[0]
        state = torch.zeros(batch, 5) if cache_params is None else cache_params.state
        self.received_states.append(state)
        logits = []
        for i in range(input_ids.shape[1]):
            new = torch.tanh(self.embedding(input_ids[:, i]) + self.recurrent(state))
            mask = attention_mask[:, i:i+1]
            state = mask * new + (1 - mask) * state
            logits.append(self.head(state))
        return SimpleNamespace(logits=torch.stack(logits, dim=1),
            cache_params=SimpleNamespace(state=state, last_tokens=input_ids[:, -2:],
                                          seq_len=input_ids.shape[1] + (0 if cache_params is None else cache_params.seq_len)))


ROWS = [{"id": "a", "ids": [1, 3, 4, 5, 6, 2]}, {"id": "b", "ids": [1, 7, 2]}]


def test_global_shift_preserves_every_boundary_target_and_ignores_padding():
    model = TinyCausalModel()
    inputs, targets, mask = trainer.batch_tensors(ROWS, "cpu")
    assert targets.tolist() == [[3, 4, 5, 6, 2], [7, 2, 0, 0, 0]]
    full = model(inputs, mask).logits
    reference = F.cross_entropy(full.reshape(-1, 12), targets.reshape(-1), ignore_index=0)
    sums, tokens, cache = 0, 0, None
    for offset in range(0, inputs.shape[1], 2):
        sl = slice(offset, offset + 2)
        loss, count, _, cache = trainer.chunk_loss(model, inputs[:, sl], targets[:, sl], mask[:, sl], cache)
        sums += loss * count
        tokens += count
    assert tokens == sum(len(row["ids"]) - 1 for row in ROWS)
    torch.testing.assert_close(sums / tokens, reference)


def test_tbptt_detach_cuts_history_but_preserves_state_and_supports_next_backward():
    model = TinyCausalModel()
    inputs, targets, mask = trainer.batch_tensors(ROWS, "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    first_loss, _, _, cache = trainer.chunk_loss(model, inputs[:, :2], targets[:, :2], mask[:, :2])
    old_state = cache.state
    old_state.retain_grad()
    first_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    cache = trainer.detach_cache(cache)
    assert cache.state.grad_fn is None and not cache.state.requires_grad
    torch.testing.assert_close(cache.state, old_state)
    previous_gradient = old_state.grad.clone()
    loss, _, _, _ = trainer.chunk_loss(model, inputs[:, 2:], targets[:, 2:], mask[:, 2:], cache)
    loss.backward()
    torch.testing.assert_close(old_state.grad, previous_gradient)
    assert model.recurrent.weight.grad.abs().sum() > 0


def test_evaluation_token_weighted_and_resets_each_story_batch():
    model = TinyCausalModel()
    result = trainer.evaluate(model, ROWS, "cpu", batch_size=1, chunk_size=2)
    assert result["tokens"] == 7 and result["stories"] == 2
    assert model.training
    manual = []
    with torch.no_grad():
        for row in ROWS:
            inputs, targets, mask = trainer.batch_tensors([row], "cpu")
            loss, n, _, _ = trainer.chunk_loss(model, inputs, targets, mask)
            manual.append((loss.item(), n))
    assert result["cross_entropy"] == pytest.approx(sum(loss*n for loss,n in manual) / 7, abs=1e-6)


def test_cache_roundtrip_preserves_next_predictions():
    model = TinyCausalModel()
    inputs, targets, mask = trainer.batch_tensors(ROWS, "cpu")
    _, _, _, cache = trainer.chunk_loss(model, inputs[:, :2], targets[:, :2], mask[:, :2])
    restored = trainer.restore_cache(trainer.cache_state(cache), "cpu")
    a = model(inputs[:, 2:], mask[:, 2:], cache_params=cache)
    b = model(inputs[:, 2:], mask[:, 2:], cache_params=restored)
    torch.testing.assert_close(a.logits, b.logits)
    assert restored.state.grad_fn is None


def test_planned_chunk_updates_match_length_aware_batches():
    batches = trainer.epoch_batches(ROWS * 3, 4, 2, 42)
    assert trainer.epoch_batches(ROWS * 3, 4, 2, 42) == batches
    observed = sum(len(range(0, trainer.batch_tensors([ROWS[i % 2] for i in batch], "cpu")[0].shape[1], 2)) for batch in batches)
    assert trainer.epoch_updates(ROWS * 3, 4, 2, 2, 42) == observed


def test_cosine_reset_does_not_reset_optimizer_moments():
    parameter = nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    schedule = trainer.cosine_scheduler(optimizer, 2)
    for _ in range(2):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
        schedule.step()
    assert optimizer.param_groups[0]["lr"] == 0
    moment = optimizer.state[parameter]["exp_avg"].clone()
    optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["initial_lr"] = .003
    trainer.cosine_scheduler(optimizer, 3)
    assert optimizer.param_groups[0]["lr"] == .003
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], moment)


class TinyReference(nn.Module):
    """Reference-shaped fixture exercises trainer control flow, not fly physics."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_neurons=5, n_out=5, d_embed=3, rec_target=5.,
                                      pad_token_id=0, bos_token_id=1, eos_token_id=2, vocab_size=12)
        self.brain = nn.Module()
        self.brain.wte = nn.Embedding(12, 3, padding_idx=0)
        self.brain.in_proj = nn.Parameter(torch.zeros(3, 5))
        for name in ("gain", "rec_gain", "bias"):
            self.brain.register_parameter(name, nn.Parameter(torch.zeros(5)))
        self.brain.register_buffer("w_offsets", torch.arange(6, dtype=torch.int32))
        self.brain.register_buffer("w_indices", torch.arange(5, dtype=torch.int32))
        self.brain.register_buffer("w_values", torch.full((5,), .2))
        self.brain.register_buffer("in_index", torch.arange(5))
        self.brain.register_buffer("out_index", torch.arange(5))
        self.ln = nn.LayerNorm(5)
        self.lm_head = nn.Linear(5, 12, bias=False)

    def forward(self, input_ids, attention_mask, cache_params=None, **kwargs):
        state = torch.zeros(input_ids.shape[0], 5) if cache_params is None else cache_params.state
        outputs = []
        b = self.brain
        for i in range(input_ids.shape[1]):
            drive = b.wte(input_ids[:, i]) @ b.in_proj
            new = .1 * state + .9 * torch.tanh(b.gain * (b.rec_gain * b.w_values * state + drive) + b.bias)
            state = torch.where(attention_mask[:, i:i+1].bool(), new, state)
            outputs.append(self.lm_head(self.ln(state)))
        return SimpleNamespace(logits=torch.stack(outputs, dim=1),
            cache_params=SimpleNamespace(state=state, last_tokens=input_ids[:, -2:],
                seq_len=input_ids.shape[1] + (0 if cache_params is None else cache_params.seq_len)))


def test_fresh_initialization_preserves_buffers_and_is_seeded():
    model = TinyReference()
    before = trainer.frozen_hashes(model)
    trainer.initialize_from_scratch(model, 42)
    values = trainer.parameters_cpu(model)
    trainer.initialize_from_scratch(model, 42)
    for name, actual in model.named_parameters():
        torch.testing.assert_close(actual, values[name])
    assert trainer.frozen_hashes(model) == before
    assert model.brain.wte.weight[0].count_nonzero() == 0
    torch.testing.assert_close(model.brain.rec_gain, torch.full((5,), 25.))


def test_mid_chunk_checkpoint_resume_matches_uninterrupted_two_phase_training(tmp_path, monkeypatch):
    template = TinyReference()
    loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: copy.deepcopy(template))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=loader))
    monkeypatch.setattr(trainer, "EXPECTED_PARAMETERS", sum(p.numel() for p in template.parameters()))
    monkeypatch.setattr(trainer, "verify_reference", lambda _: {"fixture": True})
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("config.json", "modeling_fly.py", "configuration_fly.py", "tokenizer.json"):
        (model_dir / name).write_text("fixture")
    dataset = {"train": [{"id": "t0", "ids": [1, 3, 4, 5, 6, 2]},
                         {"id": "t1", "ids": [1, 8, 9, 7, 6, 2]}],
               "validation": [{"id": "v0", "ids": [1, 5, 7, 2]}],
               "test": [{"id": "e0", "ids": [1, 6, 9, 2]}], "provenance": {"fixture": True}}
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(dataset))
    def arguments(output):
        return trainer.parser().parse_args(["--model", str(model_dir), "--data", str(data_path),
            "--output", str(output), "--device", "cpu", "--epochs", "1", "--second-epochs", "1",
            "--batch-size", "1", "--chunk-size", "2", "--eval-interval-updates", "2"])
    full = arguments(tmp_path / "full")
    trainer.run(full)
    resumed = arguments(tmp_path / "resumed")
    real_save = trainer.save_checkpoint
    class Interrupted(Exception):
        pass
    def interrupt_save(path, *args, **kwargs):
        real_save(path, *args, **kwargs)
        checkpoint = torch.load(path, weights_only=False)
        if Path(path).name == "latest.pt" and checkpoint["cursor"]["updates"] == 2:
            assert checkpoint["cache"] is not None
            raise Interrupted()
    monkeypatch.setattr(trainer, "save_checkpoint", interrupt_save)
    with pytest.raises(Interrupted):
        trainer.run(resumed)
    monkeypatch.setattr(trainer, "save_checkpoint", real_save)
    resumed.resume = resumed.output / "latest.pt"
    trainer.run(resumed)
    a = torch.load(full.output / "latest.pt", weights_only=False)
    b = torch.load(resumed.output / "latest.pt", weights_only=False)
    assert a["cursor"] == b["cursor"]
    assert a["cursor"]["updates"] == 12
    assert a["frozen_buffers_sha256"] == b["frozen_buffers_sha256"]
    for name in a["parameters"]:
        torch.testing.assert_close(a["parameters"][name], b["parameters"][name], rtol=0, atol=0)
    for path in (full.output, resumed.output):
        result = json.loads((path / "results.json").read_text())
        assert result["status"] == "completed" and result["test_evaluated"]
