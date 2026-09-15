"""Label-free diagnostics for temporary weights and their action-level effects.

These read-only measurements do not change model state or define acceptance
thresholds. In particular, a common shift of all action logits is not an effect
on the model's action distribution.
"""
from typing import Optional

import torch


def _matrix(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not all(value.shape):
        raise ValueError(name + " must be a nonempty [batch, columns] tensor")
    if not value.is_floating_point():
        raise ValueError(name + " must contain floating-point values")
    result = value.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(name + " must contain finite values")
    return result


def measure_action_effect(on_logits: torch.Tensor, off_logits: torch.Tensor) -> dict:
    """Compare paired [B,K] logits without accessing targets or retaining graphs.

    Centering is per example across K actions. RMS covers all B*K entries;
    the 95th percentile summarizes each example's largest centered difference.
    Total variation is half the L1 distance between the two softmax vectors,
    in [0,1]. Argmax ties use the same first-index convention as the model.
    CPU float64 arithmetic avoids introducing extra float32 reduction noise;
    precision already lost in the supplied logits cannot be recovered.
    """
    on = _matrix(on_logits, "on_logits")
    off = _matrix(off_logits, "off_logits")
    if on.shape != off.shape or on.shape[1] < 2:
        raise ValueError("Paired logits must have identical [batch, actions>=2] shapes")
    difference = on - off
    centered = difference - difference.mean(dim=1, keepdim=True)
    per_example_max = centered.abs().amax(dim=1)
    probabilities_on = torch.softmax(on, dim=1)
    probabilities_off = torch.softmax(off, dim=1)
    tv = .5 * (probabilities_on - probabilities_off).abs().sum(dim=1)
    changed = int((on.argmax(dim=1) != off.argmax(dim=1)).sum())
    return {"examples": int(on.shape[0]), "actions": int(on.shape[1]),
            "centered_logit_rms": float(centered.square().mean().sqrt()),
            "centered_logit_max": float(per_example_max.max()),
            "p95_centered_logit_max": float(torch.quantile(per_example_max, .95)),
            "mean_total_variation": float(tv.mean()),
            "max_total_variation": float(tv.max()),
            "changed_predictions": changed, "changed_prediction_rate": changed / len(on)}


def measure_group_state(fast: torch.Tensor, candidate_group: torch.Tensor,
                        susceptibility_grad: Optional[torch.Tensor] = None) -> dict:
    """Summarize actual [B,E] fast states and optional per-edge gradients [E].

    Fast moments cover every example and edge in a group. Susceptibility
    gradient moments cover its edges once (gradients already aggregate the
    training batch). Missing gradients are null, distinct from measured zero.
    Group IDs need not be contiguous; only groups with actual edges are listed.
    """
    values = _matrix(fast, "fast")
    if (not isinstance(candidate_group, torch.Tensor) or candidate_group.ndim != 1
            or candidate_group.numel() != values.shape[1]
            or candidate_group.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError("candidate_group must contain one integer group ID per edge")
    groups = candidate_group.detach().cpu().to(torch.int64)
    if bool((groups < 0).any()):
        raise ValueError("candidate_group IDs must be nonnegative")
    gradients = None
    if susceptibility_grad is not None:
        if (not isinstance(susceptibility_grad, torch.Tensor) or susceptibility_grad.ndim != 1
                or susceptibility_grad.numel() != values.shape[1] or not susceptibility_grad.is_floating_point()):
            raise ValueError("susceptibility_grad must contain one floating-point gradient per edge")
        gradients = susceptibility_grad.detach().cpu().to(torch.float64)
        if not bool(torch.isfinite(gradients).all()):
            raise ValueError("susceptibility_grad must contain finite values")
    rows = []
    for group in torch.unique(groups, sorted=True).tolist():
        mask = groups == group
        selected = values[:, mask]
        row = {"group": int(group), "edge_count": int(mask.sum()),
               "fast_rms": float(selected.square().mean().sqrt()),
               "fast_mean_abs": float(selected.abs().mean()),
               "fast_max_abs": float(selected.abs().max()),
               "susceptibility_gradient_rms": None,
               "susceptibility_gradient_mean_abs": None,
               "susceptibility_gradient_max_abs": None}
        if gradients is not None:
            selected_grad = gradients[mask]
            row.update(susceptibility_gradient_rms=float(selected_grad.square().mean().sqrt()),
                       susceptibility_gradient_mean_abs=float(selected_grad.abs().mean()),
                       susceptibility_gradient_max_abs=float(selected_grad.abs().max()))
        rows.append(row)
    return {"batch_size": int(values.shape[0]), "edge_count": int(values.shape[1]),
            "group_count": len(rows), "groups": rows}
