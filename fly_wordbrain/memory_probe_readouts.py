"""Small CPU readouts for cached, aligned memory-probe feature rows.

Only training rows determine normalization and gradient updates. Validation
cross entropy selects the checkpoint (earlier epochs win ties); test rows are
scored once after that selection. Numerical-zero feature dimensions are masked
using a training-only standard-deviation threshold, not amplified to unit scale.
"""
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _metrics(logits, labels, num_classes, majority_class):
    predictions = logits.argmax(1)
    confusion = torch.bincount(labels * num_classes + predictions,
                              minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return {"count": len(labels), "accuracy": float((predictions == labels).float().mean()),
            "cross_entropy": float(F.cross_entropy(logits, labels)),
            "chance_accuracy": 1. / num_classes,
            "majority_accuracy": float((labels == majority_class).float().mean()),
            "majority_class": majority_class,
            "label_counts": torch.bincount(labels, minlength=num_classes).tolist(),
            "confusion": confusion.tolist(), "confusion_axes": "rows=true label, columns=predicted label"}


def fit_probe(features, labels, splits, num_classes, *, architecture="linear", seed=0,
              shuffle_train_labels=False, max_epochs=150, patience=20, min_delta=1e-6,
              lr=.003, weight_decay=.01, batch_size=64, absolute_std_floor=1e-6,
              relative_std_floor=1e-4, threads=1):
    """Fit one linear or 16-unit ReLU MLP probe and return a JSON-ready receipt.

    ``features`` is a finite [N,256] array. Integer labels and string split IDs
    (``train``, ``val``, ``test``) each have N rows in exactly that order. Every
    split must be nonempty. Caller owns task/source-group separation upstream;
    row indices and hashes in the receipt allow alignment to be audited.

    ``shuffle_train_labels=True`` permutes only the fitting targets with a
    seeded NumPy permutation. Main train/val/test metrics retain true labels;
    ``training_fit_metrics`` separately reports the permuted training objective.
    This control preserves training class counts. It never shuffles validation
    or test labels and never selects a permutation using downstream scores.

    Training std <= max(absolute floor, relative floor * max training std)
    masks a dimension for every split; remaining dimensions are standardized.
    AdamW decays all readout parameters, including biases. No feature clipping,
    dropout, class reweighting, or test-dependent hyperparameter selection occurs.
    """
    if architecture not in ("linear", "mlp"):
        raise ValueError("architecture must be 'linear' or 'mlp'")
    for name, value, lower, upper in (("num_classes", num_classes, 2, None),
            ("seed", seed, 0, None), ("max_epochs", max_epochs, 1, 150),
            ("patience", patience, 1, None), ("batch_size", batch_size, 1, None),
            ("threads", threads, 1, None)):
        if type(value) is not int or value < lower or (upper is not None and value > upper):
            raise ValueError(name + " is outside its supported integer range")
    for name, value, strictly_positive in (("lr", lr, True), ("weight_decay", weight_decay, False),
            ("min_delta", min_delta, False), ("absolute_std_floor", absolute_std_floor, True),
            ("relative_std_floor", relative_std_floor, False)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or (
                value <= 0 if strictly_positive else value < 0):
            raise ValueError(name + " must be finite and " + ("positive" if strictly_positive else "nonnegative"))
    raw = np.asarray(features)
    targets, split_ids = np.asarray(labels), np.asarray(splits)
    if (raw.ndim != 2 or raw.shape[1] != 256 or not len(raw)
            or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all()):
        raise ValueError("Require finite numeric features with shape [N,256]")
    if (targets.shape != (len(raw),) or targets.dtype.kind not in "iu"
            or np.any(targets < 0) or np.any(targets >= num_classes)):
        raise ValueError("Require one integer class label per feature row")
    if split_ids.shape != (len(raw),) or any(value not in ("train", "val", "test") for value in split_ids):
        raise ValueError("Require one train/val/test split string per feature row")
    indices = {name: np.flatnonzero(split_ids == name) for name in ("train", "val", "test")}
    if any(not len(rows) for rows in indices.values()):
        raise ValueError("Train, validation and test splits must all be nonempty")
    raw, targets = raw.astype(np.float64, copy=True), targets.astype(np.int64, copy=True)
    train = raw[indices["train"]]
    mean, std = train.mean(axis=0), train.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Training feature moments overflow float64")
    floor = max(float(absolute_std_floor), float(relative_std_floor * std.max()))
    active = std > floor
    scales = np.maximum(std, floor)
    normalized = np.zeros(raw.shape, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        normalized[:, active] = ((raw[:, active] - mean[active]) / scales[active]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized features overflow float32")
    x = {name: torch.from_numpy(normalized[rows].copy()) for name, rows in indices.items()}
    y = {name: torch.from_numpy(targets[rows].copy()) for name, rows in indices.items()}
    permutation = np.arange(len(indices["train"]), dtype=np.int64)
    if shuffle_train_labels:
        permutation = np.random.default_rng(seed).permutation(len(permutation))
    fit_targets = y["train"][torch.from_numpy(permutation)].clone()
    majority_class = int(np.bincount(targets[indices["train"]], minlength=num_classes).argmax())
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        # Initialization restores the caller's RNG state. Mini-batch permutations
        # use their own generator; this evaluator never samples from global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            model = (nn.Linear(256, num_classes, device="cpu", dtype=torch.float32) if architecture == "linear" else
                     nn.Sequential(nn.Linear(256, 16, device="cpu", dtype=torch.float32), nn.ReLU(),
                                   nn.Linear(16, num_classes, device="cpu", dtype=torch.float32)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            best_ce = float(F.cross_entropy(model(x["val"]), y["val"]))
        best_epoch, stale = 0, 0
        best = {name: value.detach().clone() for name, value in model.state_dict().items()}
        history = [{"epoch": 0, "validation_cross_entropy": best_ce, "selected": True,
                    "training_fit_cross_entropy": None}]
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = torch.randperm(len(fit_targets), generator=generator)
            for offset in range(0, len(order), batch_size):
                rows = order[offset:offset + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x["train"][rows]), fit_targets[rows])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training probe loss")
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_ce = float(F.cross_entropy(model(x["val"]), y["val"]))
                fit_ce = float(F.cross_entropy(model(x["train"]), fit_targets))
            if not math.isfinite(val_ce) or not math.isfinite(fit_ce):
                raise RuntimeError("Nonfinite probe cross entropy")
            improved = val_ce < best_ce - min_delta
            if improved:
                best_ce, best_epoch, stale = val_ce, epoch, 0
                best = {name: value.detach().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
            history.append({"epoch": epoch, "validation_cross_entropy": val_ce,
                            "training_fit_cross_entropy": fit_ce, "selected": improved})
            if stale >= patience:
                break
        model.load_state_dict(best)
        model.eval()
        # Test forward and all reported label metrics occur only after selection.
        with torch.no_grad():
            logits = {name: model(values) for name, values in x.items()}
            metrics = {name: _metrics(logits[name], y[name], num_classes, majority_class) for name in x}
            fit_metrics = _metrics(logits["train"], fit_targets, num_classes, majority_class)
            predictions = {name: {"row_indices": indices[name].tolist(), "labels": y[name].tolist(),
                "predicted_labels": values.argmax(1).tolist(), "probabilities": values.softmax(1).tolist()}
                for name, values in logits.items()}
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        parameter_sha = _digest(*(best[name].numpy() for name in sorted(best)))
    finally:
        torch.set_num_threads(old_threads)
    return {"schema": 1, "kind": "cached_memory_readout", "architecture": architecture,
        "num_classes": num_classes, "input_dimensions": 256, "trainable_parameters": parameter_count,
        "selected_epoch": best_epoch, "epochs_run": len(history) - 1,
        "stopped_early": len(history) - 1 < max_epochs,
        "metrics": metrics, "predictions": predictions, "training_fit_metrics": fit_metrics,
        "history": history,
        "receipt": {"device": "cpu", "torch_version": str(torch.__version__), "seed": seed,
            "selection": "Lowest validation cross entropy improving by more than min_delta; earlier epoch wins ties",
            "test_used_for_selection": False, "feature_fit_split": "train", "gradient_split": "train",
            "source_group_validation": "Caller supplies source-disjoint aligned rows; no source/group IDs are inferred here",
            "optimizer": "AdamW", "lr": lr, "weight_decay": weight_decay, "weight_decay_includes_bias": True,
            "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
            "min_delta": min_delta, "threads": threads, "activation": "ReLU" if architecture == "mlp" else None,
            "hidden_dimensions": 16 if architecture == "mlp" else None,
            "normalization": {"mean": mean.tolist(), "raw_std": std.tolist(), "scale": scales.tolist(),
                "absolute_std_floor": absolute_std_floor, "relative_std_floor": relative_std_floor,
                "effective_std_floor": floor, "active_dimensions": int(active.sum()),
                "active_mask": active.tolist(), "inactive_dimensions_set_to_zero": True,
                "training_row_indices": indices["train"].tolist()},
            "shuffle_train_labels": bool(shuffle_train_labels),
            "train_label_permutation": permutation.tolist(),
            "changed_train_labels": int((fit_targets != y["train"]).sum()),
            "fitting_train_labels_sha256": _digest(fit_targets.numpy()),
            "selected_parameters_sha256": parameter_sha,
            "input_features_sha256": _digest(raw), "input_labels_sha256": _digest(targets),
            "split_row_indices": {name: rows.tolist() for name, rows in indices.items()}}}
