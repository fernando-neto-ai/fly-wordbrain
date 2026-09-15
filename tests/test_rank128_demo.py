"""CPU fake-fixture inference contracts; no training or real-model execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rank128_demo", ROOT / "scripts/export_rank128_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class Tokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [3, 4]

    def decode(self, ids, skip_special_tokens):
        # Deliberately differs from concatenating single-token decoded strings.
        return ":".join(str(value) for value in ids if value not in (0, 1, 2))


class Model(nn.Module):
    def __init__(self, eos=True):
        super().__init__()
        self.config = SimpleNamespace(n_neurons=3)
        self.calls = []
        self.eos = eos

    def forward(self, input_ids, attention_mask, cache_params, use_cache, return_dict):
        token = input_ids.item()
        length = 1 if cache_params is None else cache_params.seq_len + 1
        assert attention_mask.tolist() == [[1]] and use_cache and return_dict
        self.calls.append((token, cache_params is None, torch.is_grad_enabled(), input_ids.device.type))
        logits = torch.zeros(1, 1, 8)
        predicted = 2 if self.eos and length >= 5 else 5 + (length % 2)
        logits[0, 0, predicted] = 3
        state = torch.tensor([[0.0, -length / 10, length / 20]])
        return SimpleNamespace(logits=logits, cache_params=SimpleNamespace(state=state, seq_len=length))


def test_actual_state_alignment_bos_once_cumulative_decode_and_eos():
    model = Model()
    trace = demo.trace_story(model, Tokenizer(), {"id": "x", "title": "X", "prompt": "seed"}, [1, 2])
    assert trace["prompt_ids"] == [1, 3, 4]
    assert [frame["phase"] for frame in trace["frames"]] == ["prompt", "prompt", "generation", "generation", "generation"]
    assert [call[0] for call in model.calls] == [1, 3, 4, 6, 5]
    assert trace["generated_ids"] == [6, 5, 2]
    assert trace["frames"][2]["state_u8"] == [76, 38]
    for index, frame in enumerate(trace["frames"], start=1):
        full_state = torch.tensor([[0., -index / 10, index / 20]])
        expected_u8, expected_stats = demo.state_snapshot(full_state, torch.tensor([1, 2]))
        assert frame["state_u8"] == expected_u8
        assert frame["whole_brain_state"] == expected_stats
        assert frame["top_k"][0]["id"] == frame["predicted_token_id"]
        assert frame["top_k"][0]["probability"] == frame["probability"]
        assert 0 < frame["probability"] < 1
        if frame["phase"] == "prompt":
            assert "continuation_so_far" not in frame
    assert trace["frames"][3]["continuation_so_far"] == "6:5"
    assert trace["continuation"] == "6:5" and trace["full_text"] == "3:4:6:5"
    assert trace["generation"]["stopped_on_eos"] and trace["generation"]["forward_calls"] == 5
    assert all(not call[2] and call[3] == "cpu" for call in model.calls)
    again = demo.trace_story(model, Tokenizer(), {"id": "y", "prompt": "other"}, [1], max_new_tokens=1)
    assert model.calls[5][1] and again["generated_ids"] == [6]
    assert not again["generation"]["stopped_on_eos"]


def test_fixed_activity_scale_rejects_bad_data_and_preserves_unsigned_semantics():
    values, stats = demo.state_snapshot(torch.tensor([[-1., 0., .5, 1.]]), torch.tensor([0, 1, 2, 3]))
    assert values == [255, 0, 128, 255] and stats["minimum"] == -1
    with pytest.raises(ValueError, match="Nonfinite"):
        demo.state_snapshot(torch.tensor([[float("nan")]]), torch.tensor([0]))
    with pytest.raises(ValueError, match="exceeds"):
        demo.state_snapshot(torch.tensor([[1.1]]), torch.tensor([0]))


def test_layout_sample_is_stable_exact_id_subset_and_has_no_invented_positions():
    ids = [101, 102, 103, 104]
    neurons = [{"node_index": i, "body_id": str(ids[i]), "position": [i, i + 1, i + 2]} for i in [3, 0, 2]]
    layout = {"neurons": neurons, "coordinate_system": {"type": "anatomical", "units": "8nm"}}
    result = demo.select_layout(layout, ids, 2, 42)
    assert result == demo.select_layout({**layout, "neurons": list(reversed(neurons))}, ids, 2, 42)
    assert result["missing_positions"] == 1 and len(result["neurons"]) == 2
    assert all(value in neurons for value in result["neurons"])
    bad = {**layout, "neurons": [{**neurons[0], "body_id": "999"}]}
    with pytest.raises(ValueError, match="body ID"):
        demo.select_layout(bad, ids, 2, 42)
    with pytest.raises(ValueError, match="repeated"):
        demo.select_layout({**layout, "neurons": [neurons[0], neurons[0]]}, ids, 2, 42)


def test_checkpoint_hash_checked_before_deserialization(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"x": torch.tensor([3.])}, path)
    with pytest.raises(ValueError, match="SHA256"):
        demo.load_verified_checkpoint(path, "0" * 64)
    loaded = demo.load_verified_checkpoint(path, demo.trainer.file_hash(path))
    assert loaded["x"].device.type == "cpu" and loaded["x"].item() == 3


def test_package_relocation_ignores_only_path_and_generation_requires_nonempty():
    assert demo.relocated_receipt({"package_path": "/one", "sha256": "a"}) == {"sha256": "a"}
    assert demo.relocated_receipt({"package_path": "/two", "sha256": "b"}) != {"sha256": "a"}
    with pytest.raises(ValueError, match="positive"):
        demo.trace_story(Model(), Tokenizer(), {"prompt": "x"}, [1], max_new_tokens=0)
