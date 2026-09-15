"""Offline small-graph model checks; optional pinned upstream/Metal parity."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fly_wordbrain.connectorch_model import build_connectorch_model, GRAPH_NAMES
from fly_wordbrain.metal_sparse_trainable import TrainableCSR


MPS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
needs_mps = pytest.mark.skipif(not MPS, reason="Native MPS shader runtime unavailable")
PTR = torch.tensor([0, 2, 2, 4, 5, 7], dtype=torch.int32)
COL = torch.tensor([4, 1, 3, 0, 2, 4, 1], dtype=torch.int32)
VAL = torch.tensor([.25, -.5, .75, -.25, 0., .125, -.75])
TYPES = torch.tensor([0, 1, 2, 0, 1])
GROUPS = torch.tensor([0, 1, 2, 0, 1, 2, 0])


def reference_fixture():
    cfg = SimpleNamespace(n_neurons=5, n_edges=7, n_in=4, n_out=3,
                          d_embed=4, delay_k=2, vocab_size=11, pad_token_id=0,
                          leak=.7, rec_target=5., use_cache=True, return_dict=True)
    model = nn.Module()
    model.config = cfg
    model.brain = nn.Module()
    b = model.brain
    b.wte = nn.Embedding(11, 4, padding_idx=0)
    b.in_proj = nn.Parameter(torch.randn(2, 4, 2) * .2)
    for name, value in (("gain", torch.linspace(.8, 1.2, 5)),
                        ("rec_gain", torch.linspace(.7, 1.1, 5)),
                        ("bias", torch.linspace(-.1, .1, 5))):
        b.register_parameter(name, nn.Parameter(value))
    for name, value in (("w_offsets", PTR), ("w_indices", COL), ("w_values", VAL),
                        ("in_index", torch.tensor([1, 4, 0, 3])), ("out_index", torch.tensor([0, 2, 4]))):
        b.register_buffer(name, value.clone())
    model.ln = nn.LayerNorm(3)
    model.lm_head = nn.Linear(3, 11, bias=False)
    return model


def make_model(**kwargs):
    return build_connectorch_model(reference_fixture(), csr_factory=TrainableCSR, **kwargs)


def inputs():
    return torch.tensor([[1, 3, 4, 2], [1, 8, 2, 0]]), torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])


def test_seeded_initialization_and_compressed_interface_preserve_reference():
    reference = reference_fixture()
    before = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    a = build_connectorch_model(reference, d_embed=4, csr_factory=TrainableCSR)
    b = build_connectorch_model(reference, d_embed=4, csr_factory=TrainableCSR)
    small = build_connectorch_model(reference, d_embed=2, csr_factory=TrainableCSR)
    for name, value in a.named_parameters():
        assert torch.equal(value, dict(b.named_parameters())[name])
    for name in GRAPH_NAMES[:-2]:
        assert torch.equal(getattr(a.brain, name), getattr(small.brain, name))
    assert a.parameter_counts()["embedding"] == 2 * small.parameter_counts()["embedding"]
    assert a.parameter_counts()["input_projection"] == 2 * small.parameter_counts()["input_projection"]
    assert a.parameter_counts()["readout"] == small.parameter_counts()["readout"]
    assert a.brain.wte.weight[0].count_nonzero() == 0
    assert a.brain.rec_gain[1] == 1 # empty incoming row
    assert a.brain.rec_gain[3] == 1 # only stored zero edge
    for name, value in reference.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("grouping", ["pair", "factorized"])
def test_identity_adaptation_and_gradient_flow_preserve_canonical_graph(grouping):
    args = {"edge_group_index": GROUPS} if grouping == "pair" else {"node_type_index": TYPES}
    fixed, plastic = make_model(**args), make_model(plasticity="bounded10", **args)
    ids, mask = inputs()
    a, b = fixed(ids, mask), plastic(ids, mask)
    torch.testing.assert_close(a.logits, b.logits)
    a.logits.square().sum().backward()
    b.logits.square().sum().backward()
    for name, p in fixed.named_parameters():
        torch.testing.assert_close(p.grad, dict(plastic.named_parameters())[name].grad)
    gains = [p for name, p in plastic.named_parameters() if name.startswith("brain.edge_theta")]
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in gains)
    expected_gains = 3 if grouping == "pair" else 6
    assert plastic.parameter_counts()["edge_gains"] == expected_gains
    with torch.no_grad():
        for p in gains:
            p.copy_(torch.linspace(-100, 100, p.numel()))
    report = plastic.adaptation_report()
    assert report["nonzero_signs_preserved"] and report["zero_edges_preserved"]
    assert .9 - 1e-6 <= report["edge_multiplier"]["min"]
    assert report["edge_multiplier"]["max"] <= 1.1 + 1e-6
    assert report["neuron_gain_policy"] == "reference_unconstrained"
    json.dumps(report, allow_nan=False)
    assert plastic.verify_frozen()


def test_factorized_parameterization_matches_independent_edge_formula():
    model = make_model(plasticity="bounded10", node_type_index=TYPES)
    with torch.no_grad():
        model.brain.edge_theta_source.copy_(torch.tensor([.2, -.4, .8]))
        model.brain.edge_theta_destination.copy_(torch.tensor([-.1, .3, -.6]))
    expected = []
    for dst in range(5):
        for e in range(PTR[dst], PTR[dst + 1]):
            theta = model.brain.edge_theta_source[TYPES[COL[e]]] + model.brain.edge_theta_destination[TYPES[dst]]
            expected.append(VAL[e] * (1 + .1 * theta.tanh()))
    torch.testing.assert_close(model.brain.effective_values(), torch.stack(expected))


@pytest.mark.parametrize("lags", [[0, 1], [0, 0]])
def test_mask_cache_chunking_and_inputs_embeds_match(lags):
    model = make_model(lag_map=lags)
    ids, mask = inputs()
    full = model(ids, mask, output_hidden_states=True)
    first = model(ids[:, :2], mask[:, :2])
    last = model(ids[:, 2:], mask[:, 2:], cache_params=first.cache_params)
    torch.testing.assert_close(torch.cat([first.logits, last.logits], 1), full.logits)
    torch.testing.assert_close(last.cache_params.state, full.cache_params.state)
    assert last.cache_params.seq_len == 4
    assert torch.equal(last.cache_params.last_tokens, ids[:, -2:])
    torch.testing.assert_close(full.logits[1, -1], full.logits[1, -2])
    by_embeds = model(inputs_embeds=model.brain.wte(ids), attention_mask=mask)
    torch.testing.assert_close(by_embeds.logits, full.logits)
    assert torch.equal(by_embeds.cache_params.last_tokens, torch.zeros(2, 2, dtype=torch.long))
    assert model(ids, mask, use_cache=False).cache_params is None
    assert len(model(ids, mask, return_dict=False)) == 2


def test_lag_reassignment_changes_temporal_input_without_parameter_or_neuron_changes():
    delayed = make_model(lag_map=[0, 1])
    current = make_model(lag_map=[0, 0])
    assert delayed.parameter_counts() == current.parameter_counts()
    for name, p in delayed.named_parameters():
        assert torch.equal(p, dict(current.named_parameters())[name])
    ids, mask = inputs()
    assert not torch.allclose(delayed(ids, mask).logits, current(ids, mask).logits)
    assert torch.equal(delayed.brain.in_index, current.brain.in_index)


def test_rank_readout_accesses_all_neurons_and_has_expected_count():
    model = make_model(d_embed=2, readout_rank=2)
    assert model.lm_head[0].in_features == 3
    assert model.parameter_counts()["readout"] == 2 * (3 + 11)
    result = model(*inputs())
    assert result.logits.shape == (2, 4, 11)
    result.logits.square().sum().backward()
    assert all(p.grad.abs().sum() > 0 for p in model.lm_head.parameters())


def test_canonical_mutation_rejected_but_state_dict_roundtrip_is_allowed():
    model = make_model(plasticity="bounded10", node_type_index=TYPES)
    result = model(*inputs())
    result.logits.square().sum().backward()
    torch.optim.AdamW(model.parameters(), lr=.01).step()
    saved = copy.deepcopy(model.state_dict())
    assert not any("_csr" in name or "_edge_targets" in name for name in saved)
    restored = make_model(plasticity="bounded10", node_type_index=TYPES)
    restored.load_state_dict(saved)
    torch.testing.assert_close(model(*inputs()).logits, restored(*inputs()).logits)
    assert restored.verify_frozen()
    saved["brain.w_values"][0] += 1
    with pytest.raises(ValueError, match="canonical"):
        restored.load_state_dict(saved)
    assert restored.verify_frozen()
    restored.brain.w_indices[0] = 0
    with pytest.raises(RuntimeError, match="changed"):
        restored(*inputs())


@pytest.mark.parametrize("args", [{"plasticity": "unknown"}, {"plasticity": "bounded10"},
    {"node_type_index": [0, 1]}, {"node_type_index": [0, 2, 2, 0, 2]},
    {"edge_group_index": [0.] * 7}, {"lag_map": [0]}, {"lag_map": [0, 2]},
    {"readout_rank": 4}, {"d_embed": 0}, {"node_type_index": TYPES, "edge_group_index": GROUPS}])
def test_invalid_experiment_config_rejected(args):
    with pytest.raises(ValueError):
        make_model(**args)


@pytest.fixture(scope="module")
def upstream():
    # Optional strict parity uses only already-downloaded pinned source.
    path = Path(__file__).resolve().parents[1] / "data/ngxson-fly-llm-hf/65c677b3d566a2e9793d5f72999cdb441c6c0a9f"
    if not (path / "modeling_fly.py").is_file():
        pytest.skip("Pinned upstream source not prepared")
    pytest.importorskip("transformers")
    assert hashlib.sha256((path / "modeling_fly.py").read_bytes()).hexdigest() == "43efb042f2809fe87eb172718511aa884b7b9f7c356181bede9f70148bb87319"
    package = types.ModuleType("_connectorch_model_upstream")
    package.__path__ = [str(path)]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + ".modeling_fly")


def pinned_tiny(upstream):
    fixture = reference_fixture()
    cfg = upstream.FlyConfig(**vars(fixture.config))
    model = upstream.FlyForCausalLM(cfg)
    model.load_state_dict(fixture.state_dict())
    # Verify the hook never silently changes PAD's forward value.
    with torch.no_grad():
        model.brain.wte.weight[0].fill_(.12)
    return model


def test_pinned_original_forward_gradients_cache_and_mask_parity(upstream):
    reference = pinned_tiny(upstream)
    model = build_connectorch_model(reference, copy_reference_parameters=True, csr_factory=TrainableCSR)
    ids, mask = inputs()
    a, b = reference(ids, mask, labels=ids), model(ids, mask, labels=ids)
    torch.testing.assert_close(a.logits, b.logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(a.loss, b.loss)
    torch.testing.assert_close(a.cache_params.state, b.cache_params.state)
    a.loss.backward()
    b.loss.backward()
    for name, p in reference.named_parameters():
        torch.testing.assert_close(p.grad, dict(model.named_parameters())[name].grad, rtol=2e-5, atol=2e-6)
    ref_next = reference(ids[:, :1], cache_params=a.cache_params)
    model_next = model(ids[:, :1], cache_params=b.cache_params)
    torch.testing.assert_close(ref_next.logits, model_next.logits)
    assert ref_next.cache_params.seq_len == model_next.cache_params.seq_len


@needs_mps
def test_mps_factorized_gradients_padding_and_two_optimizer_updates():
    reference = reference_fixture()
    with torch.no_grad():
        reference.brain.wte.weight[0].fill_(.12)
    cpu = build_connectorch_model(reference, plasticity="bounded10", node_type_index=TYPES,
                                 copy_reference_parameters=True, csr_factory=TrainableCSR)
    mps = build_connectorch_model(reference, plasticity="bounded10", node_type_index=TYPES,
                                 copy_reference_parameters=True, device="mps", csr_factory=TrainableCSR)
    ids, mask = inputs()
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001) for model in (cpu, mps)]
    for _ in range(2):
        for model, optimizer, device in zip((cpu, mps), optimizers, ("cpu", "mps")):
            optimizer.zero_grad()
            model(ids.to(device), mask.to(device)).logits.square().sum().backward()
        for name, p in cpu.named_parameters():
            torch.testing.assert_close(p.grad, dict(mps.named_parameters())[name].grad.cpu(), atol=2e-5, rtol=2e-4)
        assert mps.brain.wte.weight.grad[0].count_nonzero() == 0
        for optimizer in optimizers:
            optimizer.step()
    for name, p in cpu.named_parameters():
        torch.testing.assert_close(p, dict(mps.named_parameters())[name].cpu(), atol=2e-5, rtol=2e-4)
    assert mps.verify_frozen()
    mps.cpu()
    torch.testing.assert_close(cpu(ids, mask).logits, mps(ids, mask).logits, atol=2e-5, rtol=2e-4)


def test_default_factory_uses_registered_connectorch_propagator():
    pytest.importorskip("connectorch")
    from connectorch.backends.base import BACKENDS
    from fly_wordbrain.connectorch_backend import MetalCSRPropagator
    reference = reference_fixture()
    registered = build_connectorch_model(reference, plasticity="bounded10", node_type_index=TYPES)
    independent = build_connectorch_model(reference, plasticity="bounded10", node_type_index=TYPES, csr_factory=TrainableCSR)
    a, b = registered(*inputs()), independent(*inputs())
    torch.testing.assert_close(a.logits, b.logits)
    a.logits.square().sum().backward()
    b.logits.square().sum().backward()
    for name, parameter in registered.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(independent.named_parameters())[name].grad)
    assert BACKENDS["metal_csr"] is MetalCSRPropagator
    assert registered.backend_metadata()["sparse_backend"]["connectorch_backend"] == "metal_csr"
    assert registered.verify_frozen()


def test_derived_edge_target_cache_is_guarded():
    model = make_model(plasticity="bounded10", node_type_index=TYPES)
    model(*inputs())
    model.brain._edge_targets[0] = 3
    with pytest.raises(RuntimeError, match="edge-target"):
        model(*inputs())
