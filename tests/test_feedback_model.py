"""Causal ordering, feedback effects and full-rule gradients on a tiny graph."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fly_wordbrain.action_model import CandidateActionBrain
from fly_wordbrain.feedback_model import FeedbackActionBrain, FeedbackSelector, fixed_identity_codes
from fly_wordbrain.fast_structure import (ARRAY_KEYS, ORIGINAL_GROUPS, array_digest,
    file_sha256, original_selection_sha256, save_structure)


def toy_graph(path, n=64):
    r = 56
    pre = np.arange(n, dtype=np.int32)
    post = 56 + pre % 8
    weights = (.2 + (pre % 7) * .1).astype(np.float32)
    weights[pre % 3 == 0] *= -1
    incoming_ids = np.lexsort((pre, post))
    selected = np.array([0, 1, 2, 3], np.int64)
    np.savez(path / "graph.npz", ids=np.arange(n), ptr=np.arange(n + 1), post=post, weight=weights,
        incoming_ptr=np.r_[0, np.cumsum(np.bincount(post, minlength=n))], incoming_pre=pre[incoming_ids],
        incoming_weight=weights[incoming_ids], incoming_edge_ids=incoming_ids, retina=np.arange(r),
        descending=np.arange(r, n), candidate_edge_ids=selected, candidate_pre=pre[selected],
        candidate_post=post[selected], candidate_group=np.arange(4))
    (path / "metadata.json").write_text(json.dumps({"toy_graph": True}))


def expanded_sidecar(path):
    """Keep the toy's four original edges, then select remaining real edges."""
    with np.load(path / "graph.npz", allow_pickle=False) as source:
        graph = {key: source[key].copy() for key in source.files}
    ids = np.arange(len(graph["weight"]), dtype=np.int64)
    groups = np.r_[np.arange(4), 4 + np.arange(len(ids)-4) % 2].astype(np.int64)
    arrays = dict(candidate_edge_ids=ids, candidate_group=groups,
        candidate_pre=(np.searchsorted(graph["ptr"], ids, side="right")-1).astype(np.int64),
        candidate_post=graph["post"][ids].astype(np.int64), candidate_weight=graph["weight"][ids].copy())
    metadata = dict(schema=1, kind="fast_structure", graph_sha256=file_sha256(path / "graph.npz"),
        original_selection_sha256=original_selection_sha256(graph),
        selection_array_sha256=array_digest([arrays[key] for key in ARRAY_KEYS]),
        group_names=list(ORIGINAL_GROUPS)+["toy_extra_even", "toy_extra_odd"],
        neurons=len(graph["ids"]), edges=len(ids), candidate_edges=len(ids), original_candidate_edges=4)
    sidecar = path / "expanded.npz"
    save_structure(sidecar, arrays, metadata)
    return sidecar, arrays


@pytest.fixture
def expanded_brain(tmp_path):
    torch.set_num_threads(1)
    toy_graph(tmp_path)
    sidecar, _ = expanded_sidecar(tmp_path)
    result = FeedbackActionBrain(tmp_path, 20, .5, strict_full_graph=False, seed=0,
        input_gain=10., internal_steps=4, structure_path=sidecar, trainable_susceptibility=True)
    result.set_activity_scales(.1, .1)
    return result


def batch():
    observed = torch.tensor([[4, 5, 6, 7, 8, 9, 10], [11, 10, 9, 8, 7, 6, 5]])
    previous = torch.cat((torch.full((2, 2), 2), observed[:, :6]), dim=1)
    current = torch.cat((torch.full((2, 1), 2), observed), dim=1)
    candidates = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9, 10, 11]).repeat(2, 8, 1)
    probabilities = torch.arange(10, 0, -1, dtype=torch.float32)
    probabilities = (.85 * probabilities / probabilities.sum()).repeat(2, 8, 1)
    return SimpleNamespace(previous=previous, current=current, candidates=candidates,
                           probabilities=probabilities, observed_ids=observed)


@pytest.fixture
def brain(tmp_path):
    torch.set_num_threads(1)
    toy_graph(tmp_path)
    model = FeedbackActionBrain(tmp_path, 20, .5, top_k=10, strict_full_graph=False,
                                seed=0, input_gain=10., internal_steps=4)
    model.set_activity_scales(.1, .1)
    return model


def predict_first(brain):
    values = batch()
    features, state = brain.predict(values.previous[:, 0], values.current[:, 0],
                                   values.candidates[:, 0], values.probabilities[:, 0], brain.initial_state(2))
    return values, features, state


def test_full_vocabulary_identity_codes_have_no_collisions():
    codes = fixed_identity_codes(1024, 7320)
    assert codes.shape == (1024, 16) and codes.dtype == np.float32
    assert np.unique(codes, axis=0).shape[0] == 1024
    np.testing.assert_allclose(np.linalg.norm(codes, axis=1), np.ones(1024), rtol=0, atol=0)
    np.testing.assert_array_equal(codes, fixed_identity_codes(1024, 7320))
    assert not np.array_equal(codes, fixed_identity_codes(1024, 7321))
    with pytest.raises(ValueError, match="65536"):
        fixed_identity_codes(65537, 7320)


def test_prediction_never_writes_and_observation_never_advances_activity(brain):
    values, features, predicted = predict_first(brain)
    assert torch.all(predicted.fast == 0)
    assert predicted.eligibility.abs().sum() > 0
    h_before, eligibility_before = predicted.h.clone(), predicted.eligibility.clone()
    observed = brain.observe(predicted, values.candidates[:, 0], values.probabilities[:, 0],
                             values.observed_ids[:, 0], 1)
    assert torch.equal(observed.h, h_before)
    assert torch.equal(observed.eligibility, eligibility_before)
    assert observed.fast.abs().sum() > 0
    assert torch.all(predicted.fast == 0)  # Input state remains immutable.
    assert observed.observations == 1 and not observed.pending


def test_rank_error_other_identity_and_position_are_explicit(brain):
    values = batch()
    candidates, probabilities = values.candidates[:, 0], values.probabilities[:, 0]
    encoded = brain.feedback_features(candidates, probabilities, torch.tensor([4, 15]), 1)
    assert encoded.shape == (2, 28)
    expected = -torch.cat((probabilities, 1-probabilities.sum(-1, keepdim=True)), dim=-1)
    expected[0, 2] += 1
    expected[1, 10] += 1
    torch.testing.assert_close(encoded[:, :11], expected)
    torch.testing.assert_close(encoded[:, :11].sum(-1), torch.zeros(2), atol=1e-6, rtol=0)
    assert torch.equal(encoded[:, 11:27], brain.observed_word_codes[torch.tensor([4, 15])])
    other_word = brain.feedback_features(candidates, probabilities, torch.tensor([4, 16]), 1)
    assert torch.equal(encoded[1, :11], other_word[1, :11])
    assert not torch.equal(encoded[1, 11:27], other_word[1, 11:27])
    later = brain.feedback_features(candidates, probabilities, torch.tensor([4, 15]), 7)
    assert torch.equal(encoded[:, :27], later[:, :27])
    assert torch.all(encoded[:, -1] == 0) and torch.all(later[:, -1] == 1)


def test_each_feedback_component_changes_fast_updates(brain):
    values, _, predicted = predict_first(brain)
    args = (values.candidates[:, 0], values.probabilities[:, 0])
    base = brain.observe(predicted, *args, torch.tensor([15, 15]), 1)
    identity = brain.observe(predicted, *args, torch.tensor([16, 16]), 1)
    rank = brain.observe(predicted, *args, torch.tensor([4, 4]), 1)
    # Identical prediction state at another legal observation count isolates
    # the numerical position signal without changing activity/eligibility.
    later = brain.observe(predicted._replace(observations=6), *args, torch.tensor([15, 15]), 7)
    for changed in (identity, rank, later):
        assert torch.isfinite(changed.fast).all()
        assert (changed.fast - base.fast).abs().max() > 1e-9


def test_pending_proposal_and_seven_observation_contract_are_enforced(brain):
    values, _, state = predict_first(brain)
    with pytest.raises(ValueError, match="precede"):
        brain.observe(brain.initial_state(2), values.candidates[:, 0], values.probabilities[:, 0], values.observed_ids[:, 0], 1)
    with pytest.raises(ValueError, match="pending"):
        brain.predict(values.previous[:, 1], values.current[:, 1], values.candidates[:, 1], values.probabilities[:, 1], state)
    with pytest.raises(ValueError, match="unchanged pending"):
        brain.observe(state, values.candidates[:, 0].flip(1), values.probabilities[:, 0], values.observed_ids[:, 0], 1)
    with pytest.raises(ValueError, match="causally"):
        brain.observe(state, values.candidates[:, 0], values.probabilities[:, 0], values.observed_ids[:, 0], 2)
    model = FeedbackSelector(brain)
    _, final = model.features(values)
    assert final.observations == 7 and final.pending
    with pytest.raises(ValueError, match="eighth word is hidden"):
        brain.observe(final, values.candidates[:, 7], values.probabilities[:, 7], torch.tensor([4, 5]), 8)


def test_model_never_reads_final_target_and_initial_prediction_is_prior(brain):
    class Guard:
        def __init__(self, fields):
            self.__dict__.update(vars(fields))
        @property
        def targets(self):
            raise AssertionError("Final target was read by the model")
    inputs = Guard(batch())
    model = FeedbackSelector(brain)
    logits = model(inputs)
    assert torch.equal(logits, inputs.probabilities[:, 7].log())
    assert model.trainable_parameter_count() == 2706
    assert brain.trainable_parameter_count() == 136
    assert set(name for name, p in brain.named_parameters() if p.requires_grad) == set(brain.rule_parameter_names)
    malformed = batch()
    malformed.current[:, 7] = 19
    with pytest.raises(ValueError, match="already observed prefix"):
        model(malformed)
    malformed = batch()
    malformed.observed_ids = torch.cat((malformed.observed_ids, torch.tensor([[4], [5]])), dim=1)
    with pytest.raises(ValueError, match="observed_ids"):
        model(malformed)


def test_resets_batch_independence_and_fast_feedback_affect_final_features(brain):
    model = FeedbackSelector(brain)
    inputs = batch()
    features, state = model.features(inputs)
    off, disabled = model.features(inputs, plasticity=False)
    assert (features - off).abs().max() > 1e-9
    assert state.fast.abs().sum() > 0 and torch.all(disabled.fast == 0)
    replay, replay_state = model.features(inputs)
    assert torch.equal(features, replay)
    assert torch.equal(state.fast, replay_state.fast)
    fresh = brain.initial_state(2)
    assert torch.all(fresh.h == 0) and torch.all(fresh.fast == 0) and torch.all(fresh.eligibility == 0)
    assert fresh.observations == 0 and not fresh.pending
    for row in range(2):
        single = SimpleNamespace(**{name: value[row:row+1] for name, value in vars(inputs).items()})
        separate, separate_state = model.features(single)
        torch.testing.assert_close(separate[0], features[row])
        torch.testing.assert_close(separate_state.fast[0], state.fast[row])


def test_zero_head_warmup_then_final_loss_reaches_every_shared_rule_parameter(brain):
    model = FeedbackSelector(brain)
    targets = torch.tensor([5, 8])  # Action ranks, external to the model inputs.
    first = F.cross_entropy(model(batch()), targets)
    first.backward()
    assert model.readout.weight.grad.abs().sum() > 0
    for name, parameter in brain.named_parameters():
        assert parameter.grad is None or torch.all(parameter.grad == 0), name
    torch.optim.SGD(model.readout.parameters(), lr=.1).step()
    model.zero_grad(set_to_none=True)
    second = F.cross_entropy(model(batch()), targets)
    second.backward()
    for name, parameter in brain.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert not brain.incoming.requires_grad and not brain.outgoing.requires_grad


def test_original_graph_signs_and_fixed_interfaces_are_preserved(brain):
    original = CandidateActionBrain(brain.graph_path, 20, .5, top_k=10, strict_full_graph=False,
                                    seed=0, input_gain=10., internal_steps=4)
    assert brain.frozen_fingerprint() == original.frozen_fingerprint()
    fingerprint = brain.verify_frozen()
    _, state = FeedbackSelector(brain).features(batch())
    effective = brain.effective_candidate_weights(state.fast)
    assert torch.equal(torch.sign(effective), torch.sign(brain.candidate_weight).expand_as(effective))
    ratios = effective / brain.candidate_weight
    assert bool(torch.all(ratios >= .5)) and bool(torch.all(ratios <= 2.))
    assert brain.verify_frozen() == fingerprint
    with torch.no_grad():
        brain.observed_word_codes[0, 0] += .1
    with pytest.raises(RuntimeError, match="interface"):
        brain.verify_frozen()


def test_ordered_pooling_matches_original_map_float64_oracle_and_gradient(tmp_path):
    # Hundreds of descending neurons force collisions in the old 256-bucket
    # scatter and require unequal padded rows in the new gather map.
    toy_graph(tmp_path, n=512)
    brain = FeedbackActionBrain(tmp_path, 20, .5, strict_full_graph=False, seed=0, internal_steps=4)
    generator = torch.Generator().manual_seed(812)
    h = torch.randn(3, brain.neurons, generator=generator, requires_grad=True)
    actual = brain.readout(h)
    source = h.detach().numpy().astype(np.float64)
    reference = np.zeros((3, 256), np.float64)
    bucket, nodes = brain.readout_bucket.numpy(), brain.descending.numpy()
    signs, scale = brain.readout_sign.numpy(), brain.readout_scale.numpy().astype(np.float64)
    for node, destination, sign in zip(nodes, bucket, signs):
        reference[:, destination] += source[:, node] * sign
    reference /= scale
    np.testing.assert_allclose(actual.detach().numpy(), reference, rtol=1e-6, atol=3e-7)
    old = CandidateActionBrain.readout(brain, h)
    torch.testing.assert_close(actual, old, rtol=1e-6, atol=3e-7)
    assert brain.ordered_readout_indices.shape[1] > 1
    occupied = brain.ordered_readout_signs != 0
    assert int(occupied.sum()) == len(brain.descending)
    assert torch.equal(brain.ordered_readout_indices[occupied].sort().values, brain.descending.sort().values)
    probe = torch.randn(actual.shape, generator=generator)
    gradient = torch.autograd.grad((actual * probe).sum(), h)[0]
    expected = torch.zeros_like(h)
    expected[:, brain.descending] = probe[:, brain.readout_bucket] / brain.readout_scale[brain.readout_bucket] * brain.readout_sign
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    for _ in range(3):
        assert torch.equal(brain.readout(h), actual)
    assert brain.trainable_parameter_count() == 136
    assert not brain.ordered_readout_signs.requires_grad
    before = brain.verify_frozen()
    with torch.no_grad():
        brain.ordered_readout_indices[0, 0] = (brain.ordered_readout_indices[0, 0] + 1) % brain.neurons
    with pytest.raises(RuntimeError, match="interface"):
        brain.verify_frozen()


@pytest.mark.parametrize("position", [0, 8, 1.5, True, [1]])
def test_invalid_feedback_positions_rejected(brain, position):
    inputs = batch()
    with pytest.raises(ValueError, match="position"):
        brain.feedback_features(inputs.candidates[:, 0], inputs.probabilities[:, 0], inputs.observed_ids[:, 0], position)


def test_expanded_selection_preserves_actual_base_graph_and_original_rule_initialization(expanded_brain):
    brain = expanded_brain
    original = FeedbackActionBrain(brain.graph_path, 20, .5, strict_full_graph=False,
        seed=0, input_gain=10., internal_steps=4)
    for name in ("neuron_ids", "raw_outgoing_ptr", "raw_outgoing_post", "raw_outgoing_weight", "retina", "descending"):
        assert torch.equal(getattr(brain, name), getattr(original, name)), name
    for name in ("incoming", "outgoing"):
        for component in ("crow_indices", "col_indices", "values"):
            assert torch.equal(getattr(getattr(brain, name), component)(),
                               getattr(getattr(original, name), component)())
    for name in ("candidate_edge_ids", "candidate_group", "candidate_pre", "candidate_post", "candidate_weight"):
        assert torch.equal(getattr(brain, name)[:4], getattr(original, name)), name
    for name in original.rule_parameter_names:
        assert torch.equal(getattr(brain, name)[:4], getattr(original, name)), name
    assert brain.base_graph_sha256_before_selection == brain.base_graph_sha256_after_selection == original.base_graph_fingerprint()
    assert brain.original_selection_inclusive_fingerprint == original.frozen_fingerprint()
    assert brain.frozen_fingerprint() != original.frozen_fingerprint()  # Selection is included here.
    assert brain.group_count == 6 and len(brain.group_names) == 6
    assert brain.candidates == 64
    assert brain.trainable_parameter_count() == 34 * 6 + 64
    model = FeedbackSelector(brain)
    assert model.trainable_parameter_count() == 34 * 6 + 64 + 2570
    assert torch.equal(model(batch()), batch().probabilities[:, 7].log())
    metadata = brain.metadata()
    assert metadata["group_count"] == 6 and metadata["group_names"] == brain.group_names
    assert metadata["trainable_susceptibility_parameters"] == 64
    assert metadata["base_graph_sha256_before_selection"] == metadata["base_graph_sha256_after_selection"]
    assert metadata["structure"]["selection_array_sha256"]
    assert brain.verify_frozen() == brain.initial_graph_fingerprint


def test_expanded_incidence_matches_dense_sum_and_transpose_gradient_without_padding(expanded_brain):
    brain = expanded_brain
    generator = torch.Generator().manual_seed(24)
    correction = torch.randn(3, brain.candidates, generator=generator, requires_grad=True)
    actual = brain.aggregate_candidate_corrections(correction)
    membership = (brain.candidate_unique_posts[:, None] == brain.candidate_post[None, :]).to(torch.float32)
    reference = correction @ membership.T
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=5e-7)
    probe = torch.randn(actual.shape, generator=generator)
    gradient = torch.autograd.grad((actual * probe).sum(), correction)[0]
    assert torch.equal(gradient, probe @ membership)
    assert brain.fast_incidence.values().numel() == brain.candidates
    assert brain.fast_incidence_transpose.values().numel() == brain.candidates
    assert torch.all(membership.sum(dim=0) == 1)
    for _ in range(3):
        assert torch.equal(brain.aggregate_candidate_corrections(correction), actual)
    assert not brain.fast_incidence.requires_grad and not brain.fast_incidence_transpose.requires_grad
    before = brain.verify_frozen()
    with torch.no_grad():
        brain.fast_incidence.values()[0] += 1
    with pytest.raises(RuntimeError, match="interface"):
        brain.verify_frozen()


def test_expanded_final_loss_trains_shared_groups_and_edge_susceptibility(expanded_brain):
    brain = expanded_brain
    model = FeedbackSelector(brain)
    assert torch.equal(2 * torch.sigmoid(brain.slow_susceptibility), torch.ones(brain.candidates))
    action_targets = torch.tensor([5, 8])
    F.cross_entropy(model(batch()), action_targets).backward()
    torch.optim.SGD(model.readout.parameters(), lr=.1).step()
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(batch()), action_targets).backward()
    for name, parameter in brain.named_parameters():
        assert parameter.requires_grad and parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()) and parameter.grad.abs().sum() > 0, name
    # Both added groups must learn. Two original singleton toy edges have
    # exactly zero eligibility for this sparse sensory batch, so requiring
    # every original row to change would assert activity the inputs lack.
    assert torch.all(brain.feedback_modulation.grad[4:].abs().sum(dim=1) > 0)
    assert brain.slow_susceptibility.grad[4:].abs().sum() > 0
    assert (brain.slow_susceptibility.grad != 0).sum() > 4
    assert not brain.candidate_weight.requires_grad
    before = brain.slow_susceptibility.detach().clone()
    torch.optim.Adam([brain.slow_susceptibility], lr=.01, eps=1e-12).step()
    assert not torch.equal(before, brain.slow_susceptibility)
    _, state = model.features(batch())
    ratios = brain.effective_candidate_weights(state.fast) / brain.candidate_weight
    assert bool(torch.all(ratios >= .5)) and bool(torch.all(ratios <= 2.))
    assert brain.verify_frozen() == brain.initial_graph_fingerprint


def test_expanded_correction_keeps_tiny_signal_and_episode_reset(expanded_brain):
    brain = expanded_brain
    # The original subtract-after-exp formula rounds this nonzero gain away.
    tiny = torch.full((2, brain.candidates), 1e-9)
    old = brain.effective_candidate_weights(tiny) - brain.candidate_weight
    accurate = brain.candidate_weight * torch.expm1(np.log(2.) * torch.tanh(tiny))
    assert torch.all(old == 0) and torch.all(accurate != 0)
    model = FeedbackSelector(brain)
    on, state = model.features(batch())
    off, disabled = model.features(batch(), plasticity=False)
    assert (on - off).abs().max() > 1e-8
    assert torch.all(disabled.fast == 0) and state.fast.abs().sum() > 0
    replay, replay_state = model.features(batch())
    assert torch.equal(on, replay) and torch.equal(state.fast, replay_state.fast)
    fresh = brain.initial_state(2)
    assert fresh.fast.shape == (2, brain.candidates)
    assert torch.all(fresh.h == 0) and torch.all(fresh.fast == 0) and torch.all(fresh.eligibility == 0)


def test_expanded_group_activity_scales_and_optional_susceptibility(expanded_brain):
    brain = expanded_brain
    values = torch.arange(1, brain.group_count + 1, dtype=torch.float32)
    brain.set_activity_scales(values, values + 2)
    assert torch.equal(brain.pre_scale, values[brain.candidate_group])
    assert torch.equal(brain.post_scale, (values + 2)[brain.candidate_group])
    with pytest.raises(ValueError, match="per-group"):
        brain.set_activity_scales(torch.ones(4), .1)
    without = FeedbackActionBrain(brain.graph_path, 20, .5, strict_full_graph=False,
        seed=0, structure_path=brain.structure_path)
    assert without.slow_susceptibility is None
    assert without.trainable_parameter_count() == 34 * brain.group_count
    assert "slow_susceptibility" not in without.rule_parameter_names


@pytest.mark.parametrize("field", ["candidate_pre", "candidate_post", "candidate_weight", "candidate_edge_ids"])
def test_model_rejects_sidecar_that_invents_or_changes_edges(tmp_path, field):
    toy_graph(tmp_path)
    sidecar, arrays = expanded_sidecar(tmp_path)
    metadata = json.loads(sidecar.with_suffix(".json").read_text())
    arrays[field][7] += 1
    metadata["selection_array_sha256"] = array_digest([arrays[key] for key in ARRAY_KEYS])
    tampered = tmp_path / ("tampered_" + field + ".npz")
    save_structure(tampered, arrays, metadata)
    with pytest.raises(ValueError):
        FeedbackActionBrain(tmp_path, 20, .5, strict_full_graph=False, structure_path=tampered)


def test_default_rule_keeps_original_formula_and_has_no_expanded_cache(brain):
    assert not brain.expanded_structure and not brain.trainable_susceptibility
    assert not hasattr(brain, "fast_incidence") and brain.slow_susceptibility is None
    values, _, state = predict_first(brain)
    features = brain.feedback_features(values.candidates[:, 0], values.probabilities[:, 0], values.observed_ids[:, 0], 1)
    group = brain.candidate_group
    modulation = torch.tanh(features @ brain.feedback_modulation.T)[:, group]
    pre = torch.tanh(state.h[:, brain.candidate_pre] / brain.pre_scale)
    post = torch.tanh(state.h[:, brain.candidate_post] / brain.post_scale)
    gate = torch.sigmoid(brain.gate_bias[group] + brain.gate_pre[group] * pre + brain.gate_post[group] * post)
    expected = (torch.sigmoid(brain.retention_logit[group]) * state.fast
        + brain.max_write_strength * torch.tanh(brain.write_strength[group]) * gate * modulation * state.eligibility)
    actual = brain.observe(state, values.candidates[:, 0], values.probabilities[:, 0], values.observed_ids[:, 0], 1)
    assert torch.equal(actual.fast, expected)
    assert brain.trainable_parameter_count() == 136
