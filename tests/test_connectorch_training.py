"""Exercise the new trainer with real sparse dynamics on a tiny CPU graph."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("connectorch_trainer", ROOT / "scripts/train_connectorch.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TinyReference(nn.Module):
    """Only supplies graph/configuration; the production model runs every forward."""
    def __init__(self):
        super().__init__()
        n, k, vocab = 16, 8, 12
        self.config = SimpleNamespace(n_neurons=n, n_out=n, n_in=n, n_edges=2*n,
            d_embed=128, delay_k=k, rec_target=5., leak=.9, use_cache=True,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, vocab_size=vocab)
        self.brain = nn.Module()
        self.brain.wte = nn.Embedding(vocab, 128, padding_idx=0)
        self.brain.in_proj = nn.Parameter(torch.zeros(k, 128, n // k))
        for name in ("gain", "rec_gain", "bias"):
            self.brain.register_parameter(name, nn.Parameter(torch.zeros(n)))
        self.brain.register_buffer("w_offsets", torch.arange(n+1, dtype=torch.int32)*2)
        edges = [[(row-1) % n, (row+2) % n] for row in range(n)]
        self.brain.register_buffer("w_indices", torch.tensor(edges, dtype=torch.int32).flatten())
        self.brain.register_buffer("w_values", torch.tensor([.2, -.1] * n))
        self.brain.register_buffer("in_index", torch.arange(n))
        self.brain.register_buffer("out_index", torch.arange(n))
        self.ln = nn.LayerNorm(n)
        self.lm_head = nn.Linear(n, vocab, bias=False)


ROWS = [{"id": "a", "ids": [1, 3, 4, 5, 6, 2]}, {"id": "b", "ids": [1, 7, 2]}]


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def adaptation(plasticity="fixed", d_embed=32, history_length=8, readout_rank=0):
    args = SimpleNamespace(d_embed=d_embed, plasticity=plasticity, history_length=history_length,
                           readout_rank=readout_rank, seed=42)
    return trainer.build_model(TinyReference(), args, torch.arange(16) % 4)


def test_global_shift_and_chunk_cache_preserve_all_targets_with_real_delays():
    model = adaptation()
    inputs, targets, mask = trainer.batch_tensors(ROWS, "cpu")
    assert targets.tolist() == [[3, 4, 5, 6, 2], [7, 2, 0, 0, 0]]
    full = model(inputs, mask).logits
    expected = F.cross_entropy(full.reshape(-1, 12), targets.reshape(-1), ignore_index=0)
    summed, tokens, cache = 0, 0, None
    for offset in range(0, inputs.shape[1], 2):
        sl = slice(offset, offset + 2)
        loss, count, _, cache = trainer.chunk_loss(model, inputs[:, sl], targets[:, sl], mask[:, sl], cache)
        summed += loss * count
        tokens += count
    assert tokens == sum(len(row["ids"])-1 for row in ROWS)
    torch.testing.assert_close(summed / tokens, expected)
    assert cache.last_tokens.shape == (2, 8)


def test_detach_and_restore_keep_history_but_cut_the_old_autograd_graph():
    model = adaptation("bounded10")
    inputs, targets, mask = trainer.batch_tensors(ROWS, "cpu")
    optimizer = trainer.optimizer_for(model, 0)
    loss, _, _, cache = trainer.chunk_loss(model, inputs[:, :2], targets[:, :2], mask[:, :2])
    state = cache.state
    assert state.grad_fn is not None
    old_state_backward = []
    state.register_hook(lambda grad: old_state_backward.append(grad.clone()))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    cache = trainer.detach_cache(cache)
    restored = trainer.restore_cache(trainer.cache_state(cache), "cpu")
    assert restored.state.grad_fn is None and not cache.state.requires_grad
    a = model(inputs[:, 2:], mask[:, 2:], cache_params=cache).logits
    b = model(inputs[:, 2:], mask[:, 2:], cache_params=restored).logits
    torch.testing.assert_close(a, b)
    loss, _, _, _ = trainer.chunk_loss(model, inputs[:, 2:], targets[:, 2:], mask[:, 2:], restored)
    loss.backward()
    assert not old_state_backward
    assert model.brain.edge_theta_source.grad.abs().sum() > 0


def test_fixed_and_bounded_start_equal_but_only_bounded_updates_edges():
    fixed, plastic = adaptation(), adaptation("bounded10")
    inputs, targets, mask = trainer.batch_tensors(ROWS, "cpu")
    torch.testing.assert_close(fixed(inputs, mask).logits, plastic(inputs, mask).logits, rtol=0, atol=0)
    original = trainer.frozen_hashes(plastic)
    optimizer = trainer.optimizer_for(plastic, 0)
    brain_params = {id(value) for value in optimizer.param_groups[0]["params"]}
    for name in ("edge_theta_source", "edge_theta_destination"):
        assert id(getattr(plastic.brain, name)) in brain_params
    loss, _, _, _ = trainer.chunk_loss(plastic, inputs, targets, mask)
    loss.backward()
    for name in ("edge_theta_source", "edge_theta_destination"):
        grad = getattr(plastic.brain, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    optimizer.step()
    assert torch.count_nonzero(plastic.brain.effective_values() - plastic.brain.w_values) > 0
    assert torch.equal(fixed.brain.effective_values(), fixed.brain.w_values)
    audit = trainer.model_audit(plastic, original)
    assert audit["adaptation"]["zero_edges_preserved"]
    assert audit["adaptation"]["nonzero_signs_preserved"]
    assert .9 <= audit["adaptation"]["edge_multiplier"]["min"] <= 1.1
    assert .9 <= audit["adaptation"]["edge_multiplier"]["max"] <= 1.1


@pytest.mark.parametrize("width,rank", [(128, 0), (32, 0), (32, 4)])
def test_parameter_count_formula_and_history_controls(width, rank):
    args = SimpleNamespace(d_embed=width, plasticity="bounded10", readout_rank=rank)
    a = adaptation("bounded10", width, 8, rank)
    b = adaptation("bounded10", width, 1, rank)
    assert trainer.parameter_counts(a) == trainer.expected_parameter_counts(a, args, torch.arange(16) % 4)
    assert trainer.parameter_counts(a) == trainer.parameter_counts(b)
    assert a.brain.lag_map == tuple(range(8)) and b.brain.lag_map == (0,) * 8
    assert trainer.frozen_hashes(a) == trainer.frozen_hashes(b)


def make_training_fixture(tmp_path, monkeypatch):
    template = TinyReference()
    loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: copy.deepcopy(template))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=loader))
    monkeypatch.setattr(trainer, "verify_reference", lambda _: {"fixture": True})
    # Source files are written concurrently by other implementation agents;
    # control the receipt here and test changed-source rejection separately.
    monkeypatch.setattr(trainer, "source_receipt", lambda: {"fixture_trainer": "pinned-source"})
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
    groups_path = tmp_path / "groups.npz"
    metadata = {"frozen_buffers_sha256": trainer.frozen_hashes(template)}
    np.savez(groups_path, node_type_index=np.arange(16, dtype=np.int32) % 4,
             metadata_json=json.dumps(metadata))
    def arguments(output, *extra):
        return trainer.parser().parse_args(["--model", str(model_dir), "--data", str(data_path),
            "--groups", str(groups_path), "--output", str(output), "--device", "cpu",
            "--epochs", "1", "--second-epochs", "1", "--batch-size", "1", "--chunk-size", "2",
            "--eval-interval-updates", "2", "--d-embed", "32", "--plasticity", "bounded10", *extra])
    return arguments, dataset, groups_path


def test_mid_chunk_resume_matches_all_parameters_optimizer_rng_and_two_phases(tmp_path, monkeypatch):
    arguments, _, _ = make_training_fixture(tmp_path, monkeypatch)
    full, resumed = arguments(tmp_path / "full", "--skip-final-test"), arguments(tmp_path / "resumed", "--skip-final-test")
    trainer.run(full)
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
    a, b = [torch.load(args.output / "latest.pt", weights_only=False) for args in (full, resumed)]
    assert a["cursor"] == b["cursor"] and a["cursor"]["updates"] == 12
    assert a["frozen_buffers_sha256"] == b["frozen_buffers_sha256"]
    assert a["scheduler"] == b["scheduler"]
    assert a["rng"]["python"] == b["rng"]["python"]
    torch.testing.assert_close(a["rng"]["torch_cpu"], b["rng"]["torch_cpu"], rtol=0, atol=0)
    for name in a["parameters"]:
        torch.testing.assert_close(a["parameters"][name], b["parameters"][name], rtol=0, atol=0)
    for parameter, state in a["optimizer"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][parameter][field], rtol=0, atol=0)
    for args in (full, resumed):
        result = json.loads((args.output / "results.json").read_text())
        assert result["status"] == "completed" and not result["test_evaluated"] and result["test_deferred"]
        assert result["checks"]["selected_checkpoint_audit"]["frozen_buffers_preserved"]
        assert result["checks"]["parameter_counts"]["edge_gains"] == 8


@pytest.mark.parametrize("debug,skip", [(False, False), (True, False), (False, True)])
def test_final_test_is_once_only_or_skipped_for_debug_or_campaign(tmp_path, monkeypatch, debug, skip):
    arguments, dataset, _ = make_training_fixture(tmp_path, monkeypatch)
    options = (["--eval-limit", "1"] if debug else []) + (["--skip-final-test"] if skip else [])
    args = arguments(tmp_path / "run", *options)
    seen = []
    real_evaluate = trainer.evaluate
    def tracked(model, rows, *args, **kwargs):
        if rows[0]["id"] == dataset["test"][0]["id"]:
            seen.append("test")
        return real_evaluate(model, rows, *args, **kwargs)
    monkeypatch.setattr(trainer, "evaluate", tracked)
    trainer.run(args)
    tested = not debug and not skip
    assert seen == (["test"] if tested else [])
    result = json.loads((args.output / "results.json").read_text())
    assert result["test_evaluated"] == tested
    if tested:
        args.resume = args.output / "latest.pt"
        with pytest.raises(ValueError, match="already evaluated its selected final test"):
            trainer.run(args)
        assert seen == ["test"]


def test_source_change_rejects_resume(tmp_path, monkeypatch):
    arguments, _, _ = make_training_fixture(tmp_path, monkeypatch)
    args = arguments(tmp_path / "run", "--max-updates", "2")
    trainer.run(args)
    args.resume = args.output / "latest.pt"
    monkeypatch.setattr(trainer, "source_receipt", lambda: {"fixture_trainer": "different-source"})
    with pytest.raises(ValueError, match="trainer_sources_sha256"):
        trainer.run(args)


def test_wrong_graph_grouping_is_rejected_before_model_construction(tmp_path, monkeypatch):
    arguments, _, groups_path = make_training_fixture(tmp_path, monkeypatch)
    np.savez(groups_path, node_type_index=np.arange(16, dtype=np.int32) % 4,
             metadata_json=json.dumps({"frozen_buffers_sha256": {"wrong": "graph"}}))
    with pytest.raises(ValueError, match="different reference graph"):
        trainer.run(arguments(tmp_path / "run"))


@pytest.mark.parametrize("indices", [np.array([-1, 0]), np.array([0, 2]), np.array([0., 1.]), np.empty(0, dtype=np.int32)])
def test_invalid_node_type_indices_are_rejected(tmp_path, indices):
    path = tmp_path / "bad.npz"
    np.savez(path, node_type_index=indices, metadata_json=json.dumps({"frozen_buffers_sha256": {}}))
    with pytest.raises(ValueError):
        trainer.load_groups(path)
