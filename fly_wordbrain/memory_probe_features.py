"""Restore an immutable feedback checkpoint and expose its existing activity.

These utilities neither train the fly nor change its recurrent computation.
Each temporal sample is taken after a normal prediction and before observing
that prediction's word. The hidden eighth word is never read by extraction.
"""
import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from .brain import array_digest
from .feedback_model import FeedbackActionBrain, FeedbackSelector


INFERENCE_SOURCES = ("feedback_model.py", "action_model.py", "plastic_brain.py",
                     "metal_sparse.py", "brain.py", "pair_brain.py", "fast_structure.py")


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tensor(value, shape, name, positive=False):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.dtype != torch.float32 or tuple(value.shape) != tuple(shape)
            or not bool(torch.isfinite(value).all())):
        raise ValueError("Checkpoint requires finite float32 " + name + " with shape " + str(tuple(shape)))
    if positive and not bool(torch.all(value > 0)):
        raise ValueError("Checkpoint requires positive " + name)
    return value.detach()


def _parameter_digest(parameters):
    arrays = []
    for name in sorted(parameters):
        arrays.extend((np.frombuffer(name.encode(), dtype=np.uint8),
                       parameters[name].detach().cpu().numpy()))
    return array_digest(arrays)


def load_frozen_model(checkpoint, graph, structure, device):
    """Return ``(FeedbackSelector, receipt)`` from save_checkpoint's payload.

    Paths may be relocated, but canonical graph, sidecar and inference-source
    hashes must match the checkpoint protocol. Every named parameter and all
    four calibration arrays are restored exactly; missing or extra parameters
    are errors. No optimizer or runtime neural/fast state is restored.
    """
    path = Path(checkpoint).resolve()
    serialized = path.read_bytes()  # Hash and deserialize the same atomic snapshot.
    checkpoint_sha = hashlib.sha256(serialized).hexdigest()
    payload = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)
    if (not isinstance(payload, dict) or payload.get("schema") != 1
            or payload.get("kind") != "feedback_action" or payload.get("arm") != "feedback_fast_weights"):
        raise ValueError("Expected a schema-1 feedback_action checkpoint")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("top_k") != 10 or protocol.get("observed_words") != 7 or protocol.get("window_words") != 8:
        raise ValueError("Checkpoint protocol must declare seven observations and an eighth-word top-10 forecast")
    try:
        model_metadata = protocol["model"]["metadata"]
        brain_metadata = model_metadata["brain"]
        vocabulary_size = len(protocol["vocabulary"])
        expanded = bool(brain_metadata["expanded_structure"])
        susceptibility = bool(brain_metadata["trainable_susceptibility"])
        expected_sources = protocol["source_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint is missing model architecture or source provenance") from error
    if model_metadata.get("features_calibrated") is not True:
        raise ValueError("Checkpoint does not declare completed feature calibration")
    if expanded != (structure is not None):
        raise ValueError("Provide the checkpoint's verified structure sidecar exactly when expanded_structure is true")
    source_hashes = {}
    for name in INFERENCE_SOURCES:
        actual = _file_sha256(Path(__file__).with_name(name))
        if expected_sources.get(name) != actual:
            raise ValueError("Inference source differs from checkpoint provenance: " + name)
        source_hashes[name] = actual
    graph_path = Path(graph)
    graph_path = graph_path / "graph.npz" if graph_path.is_dir() else graph_path
    if _file_sha256(graph_path) != protocol.get("graph_sha256"):
        raise ValueError("Canonical graph archive differs from checkpoint provenance")
    structure_sha = _file_sha256(structure) if expanded else None
    if structure_sha != protocol.get("structure_sha256"):
        raise ValueError("Fast structure archive differs from checkpoint provenance")
    max_write = brain_metadata["max_absolute_write_strength"]
    brain = FeedbackActionBrain(graph_path, vocabulary_size,
        global_scale=brain_metadata["global_scale"], top_k=10,
        internal_steps=brain_metadata["internal_steps"], leak=brain_metadata["leak"],
        input_gain=brain_metadata["input_gain"], input_center=brain_metadata["input_center_on_retina"],
        input_high=brain_metadata["input_high"], seed=protocol["seed"],
        device=device, strict_full_graph=brain_metadata["strict_full_graph"],
        max_write_strength=max_write, initial_write_strength=.2 * max_write,
        structure_path=structure, trainable_susceptibility=susceptibility)
    if (brain.neurons != brain_metadata["neurons"] or brain.edges != brain_metadata["edges"]
            or brain.candidates != brain_metadata["candidate_edges"]
            or brain.group_names != brain_metadata["group_names"]):
        raise ValueError("Rebuilt graph or selected groups differ from checkpoint architecture")
    for key, actual in (("selection_inclusive_fingerprint", brain.frozen_fingerprint()),
                        ("base_graph_sha256_after_selection", brain.base_graph_fingerprint()),
                        ("fixed_interface_sha256", brain.interface_fingerprint())):
        if brain_metadata.get(key) != actual:
            raise ValueError("Rebuilt immutable model identity differs from checkpoint: " + key)
    mean = _finite_tensor(payload.get("feature_mean"), (256,), "feature_mean")
    std = _finite_tensor(payload.get("feature_std"), (256,), "feature_std", positive=True)
    pre = _finite_tensor(payload.get("pre_scale"), (brain.candidates,), "pre_scale", positive=True)
    post = _finite_tensor(payload.get("post_scale"), (brain.candidates,), "post_scale", positive=True)
    model = FeedbackSelector(brain, feature_mean=mean, feature_std=std)
    parameters = payload.get("parameters")
    expected = dict(model.named_parameters())
    if not isinstance(parameters, dict) or set(parameters) != set(expected):
        supplied = set(parameters) if isinstance(parameters, dict) else set()
        raise ValueError("Checkpoint parameter names differ; missing=" + str(sorted(set(expected) - supplied))
                         + ", extra=" + str(sorted(supplied - set(expected))))
    with torch.no_grad():
        for name, destination in expected.items():
            source = _finite_tensor(parameters[name], destination.shape, "parameter " + name)
            destination.copy_(source.to(destination.device))
        brain.set_activity_scales(pre, post)
    parameter_count = sum(value.numel() for value in expected.values())
    if parameter_count != protocol.get("trainable_parameters"):
        raise ValueError("Restored parameter count differs from checkpoint protocol")
    restored_digest = _parameter_digest(expected)
    if restored_digest != _parameter_digest(parameters):
        raise RuntimeError("Restored parameters differ numerically from checkpoint")
    for name, actual, source in (("feature_mean", model.feature_mean, mean),
                                 ("feature_std", model.feature_std, std),
                                 ("pre_scale", brain.pre_scale, pre), ("post_scale", brain.post_scale, post)):
        if not torch.equal(actual.detach().cpu(), source):
            raise RuntimeError("Restored calibration differs numerically from checkpoint: " + name)
    model.requires_grad_(False)
    model.eval()
    model.features_calibrated = True
    frozen_identity = brain.verify_frozen()
    receipt = {"schema": 1, "kind": "frozen_memory_probe_checkpoint", "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_step": payload["global_step"],
        "checkpoint_role": payload["checkpoint_role"], "dataset_sha256": protocol.get("dataset_sha256"),
        "graph_sha256": brain.graph_file_sha256, "structure_sha256": structure_sha,
        "base_graph_sha256": brain.base_graph_fingerprint(), "selection_inclusive_fingerprint": frozen_identity,
        "fixed_interface_sha256": brain.interface_fingerprint(), "parameter_sha256": restored_digest,
        "parameter_count_restored": parameter_count, "trainable_parameters": model.trainable_parameter_count(),
        "parameter_shapes": {name: list(value.shape) for name, value in expected.items()},
        "calibration_sha256": array_digest([value.numpy() for value in (mean, std, pre, post)]),
        "features_calibrated": True, "eval_mode": True, "device": str(brain.device),
        "inference_source_sha256": source_hashes,
        "probe_source_sha256": _file_sha256(__file__),
        "runtime_state": "Fresh zeros per extraction batch; no state restored from checkpoint"}
    return model, receipt


def candidate_neuron_indices(model, mode="edge_endpoints"):
    """Return sorted canonical row indices, excluding directly driven retina.

    ``edge_endpoints`` is the compact union of selected pre/post cells;
    ``all_nonretinal`` permits subsequent selection by training-only variance.
    Neither option examines labels, activity, validation data, or annotations.
    """
    brain = model.brain
    if mode == "edge_endpoints":
        candidates = np.unique(np.concatenate((brain.candidate_pre.detach().cpu().numpy(),
                                               brain.candidate_post.detach().cpu().numpy())))
    elif mode == "all_nonretinal":
        candidates = np.arange(brain.neurons, dtype=np.int64)
    else:
        raise ValueError("Unknown candidate neuron mode: " + str(mode))
    retina = brain.retina.detach().cpu().numpy()
    return np.setdiff1d(candidates, retina, assume_unique=True).astype(np.int64, copy=False)


def extract_batch(model, batch, selected_indices=None, plasticity=True):
    """Capture raw activity using precisely the selector's causal outer loop.

    Returns a dict of float32 numpy arrays: final_pooled[B,256],
    final_activity[B,N], final_selected[B,C], temporal_pooled[B,8,256], and
    temporal_selected[B,8,C], plus selected_indices[C] and metadata. Pooled
    features are raw; checkpoint standardization is not applied here.
    The final full neural vector is copied once; intermediate copies contain
    only C selected cells and the existing 256 pooled features.
    """
    if not isinstance(model, FeedbackSelector):
        raise TypeError("Extraction requires a FeedbackSelector")
    if any(p.requires_grad for p in model.parameters()) or any(module.training for module in model.modules()):
        raise ValueError("Extraction requires frozen parameters and eval mode")
    if not isinstance(plasticity, bool):
        raise ValueError("plasticity must be an explicit Boolean")
    brain = model.brain
    indices = candidate_neuron_indices(model) if selected_indices is None else np.asarray(selected_indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
            or np.any(indices < 0) or np.any(indices >= brain.neurons)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("Selected cells must be nonempty unique canonical integer row indices")
    indices = indices.astype(np.int64, copy=True)
    if np.intersect1d(indices, brain.retina.detach().cpu().numpy()).size:
        raise ValueError("Memory probes must exclude directly driven retinal cells")
    selected = torch.as_tensor(indices, dtype=torch.long, device=brain.device)
    pooled, cells = [], []
    with torch.no_grad():
        previous, current, candidates, probabilities, observed = model._inputs(batch)
        state = brain.initial_state(len(previous))
        for position in range(8):
            features, state = brain.predict(previous[:, position], current[:, position],
                candidates[:, position], probabilities[:, position], state, plasticity=plasticity)
            pooled.append(features.detach().cpu().numpy().copy())
            cells.append(state.h[:, selected].detach().cpu().numpy().copy())
            if position < 7:
                state = brain.observe(state, candidates[:, position], probabilities[:, position],
                                      observed[:, position], position + 1)
        final_activity = state.h.detach().cpu().numpy().copy()
        final_fast = state.fast.detach().cpu().numpy().astype(np.float64)
    temporal_pooled, temporal_selected = np.stack(pooled, axis=1), np.stack(cells, axis=1)
    if not all(np.isfinite(values).all() for values in (temporal_pooled, temporal_selected, final_activity, final_fast)):
        raise RuntimeError("Nonfinite frozen neural activity during memory extraction")
    return {"final_pooled": temporal_pooled[:, -1].copy(), "final_activity": final_activity,
        "final_selected": temporal_selected[:, -1].copy(), "temporal_pooled": temporal_pooled,
        "temporal_selected": temporal_selected, "selected_indices": indices,
        "metadata": {"schema": 1, "kind": "frozen_memory_probe_features", "batch_size": len(previous),
            "neurons": brain.neurons, "selected_cells": len(indices), "positions": 8,
            "selected_indices_sha256": array_digest([indices]), "plasticity": plasticity,
            "pooled_features": "Raw original 256-dimensional signed descending pool; no standardization",
            "temporal_timing": "Index t is prediction t+1 after t prior observations, before observation t+1",
            "retinal_cells_excluded": True, "hidden_target_accessed": False,
            "parameters_frozen": True, "gradients_enabled": False,
            "final_observations": state.observations, "fresh_runtime_state": True,
            "final_fast_abs_max": float(np.abs(final_fast).max()),
            "final_fast_abs_mean": float(np.abs(final_fast).mean()),
            "final_fast_rms": float(np.sqrt(np.mean(final_fast * final_fast)))}}
