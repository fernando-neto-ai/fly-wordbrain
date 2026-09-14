"""Full-graph wiring, repeatability, and history-sensitivity checks."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .brain import FrozenBrain
from .extract import atomic_json


def sequence(brain, ids, disconnected=False):
    brain.reset()
    features, diagnostics = [], []
    for token in ids:
        features.append(brain.step(token, disconnected=disconnected))
        diagnostics.append(brain.last_diagnostics.copy())
    return np.asarray(features), diagnostics


def run(output, vocab_size=1024, word_ms=20., high=0.02):
    started = time.perf_counter()
    b = FrozenBrain(vocab_size, word_ms=word_ms, high=high)
    # Probe codes are fixed arbitrary IDs; no training/validation/test targets.
    a = [2, 4, 5, 6, 7, 8, 9, 10]
    c = [2, 11, 12, 13, 14, 15, 16, 10]
    xa, diag = sequence(b, a)
    repeat, _ = sequence(b, a)
    xc, _ = sequence(b, c)
    disconnected_a, _ = sequence(b, a, True)
    disconnected_c, _ = sequence(b, c, True)
    checks = {"repeat_after_reset_exact": bool(np.array_equal(xa, repeat)),
        "disconnected_ignores_word_ids": bool(np.array_equal(disconnected_a, disconnected_c)),
        "connected_histories_change_output": bool(not np.array_equal(xa[-1], xc[-1])),
        "finite_outputs": bool(np.isfinite(xa).all() and np.isfinite(xc).all()),
        "graph_unchanged": b.verify_frozen() == b.initial_frozen_hash}
    lags = []
    for context in [2, 8, 16, 32, 64, 128]:
        # First word differs; total lexical context includes that cue word.
        suffix = ([6, 7, 8, 9] * 33)[:context - 1]
        left, _ = sequence(b, [4] + suffix)
        right, _ = sequence(b, [5] + suffix)
        delta = left[-1] - right[-1]
        lags.append({"context_words": context, "same_suffix_words": context - 1,
            "max_abs_feature_difference": float(np.abs(delta).max()),
            "feature_l2_difference": float(np.linalg.norm(delta)),
            "relative_l2_difference": float(np.linalg.norm(delta) / max(np.linalg.norm(left[-1]), 1e-12)),
            "interpretation": "causal sensitivity only; not evidence of decodable or useful memory"})
    b.verify_frozen()
    report = {"brain": b.metadata(), "checks": checks, "passed": all(checks.values()),
        "diagnostics": diag, "history_sensitivity": lags,
        "wall_seconds": time.perf_counter() - started,
        "mean_kernel_ms_per_word": float(np.mean([d["kernel_seconds"] for d in diag]) * 1000),
        "scope": "software wiring test; neural physiology and language benefit remain unvalidated"}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Full-graph wiring probe failed; inspect report before training")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--word-ms", type=float, default=20.)
    p.add_argument("--high", type=float, default=0.02)
    a = p.parse_args()
    run(a.output, a.vocab_size, a.word_ms, a.high)


if __name__ == "__main__":
    main()
