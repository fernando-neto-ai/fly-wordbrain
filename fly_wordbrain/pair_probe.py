"""Full-graph checks of the ordered-pair interface before fitting decoders."""
import argparse
import time
from pathlib import Path
import numpy as np
from .extract import atomic_json
from .pair_brain import PairedFrozenBrain


def sequence(brain, words, disconnected=False):
    brain.reset()
    observations, direct, diagnostics = [], [], []
    previous = 2
    for current in words:
        pair = (previous, current)
        observations.append(brain.step(pair, disconnected=disconnected))
        direct.append(brain.direct_features(pair))
        diagnostics.append(brain.last_diagnostics.copy())
        previous = current
    return np.asarray(observations), np.asarray(direct), diagnostics


def run(output):
    started = time.perf_counter()
    brain = PairedFrozenBrain(1024)
    a = [2, 4, 5, 6, 7, 8, 9, 10]
    b = [2, 5, 4, 6, 7, 8, 9, 10]
    xa, da, diagnostics = sequence(brain, a)
    repeated, repeated_direct, _ = sequence(brain, a)
    xb, db, _ = sequence(brain, b)
    zero_a, _, _ = sequence(brain, a, True)
    zero_b, _, _ = sequence(brain, b, True)
    checks = {'repeat_after_reset_exact': bool(np.array_equal(xa, repeated)),
              'direct_repeat_exact': bool(np.array_equal(da, repeated_direct)),
              'connected_word_order_changes_observations': bool(np.any(xa != xb)),
              'disconnected_ignores_word_order': bool(np.array_equal(zero_a, zero_b)),
              'ordered_codes_distinct': bool(np.any(brain.encoder((4, 5)) != brain.encoder((5, 4)))),
              'equal_illumination': int(np.count_nonzero(brain.encoder((4, 5)))) == 834,
              'direct_has_no_older_history': bool(np.array_equal(da[-1], db[-1])),
              'graph_unchanged': brain.verify_frozen() == brain.initial_frozen_hash}
    report = {'brain': brain.metadata(), 'checks': checks, 'passed': all(checks.values()),
              'wall_seconds': time.perf_counter() - started, 'diagnostics': diagnostics,
              'mean_kernel_ms_per_interval': float(np.mean([d['kernel_seconds'] for d in diagnostics]) * 1000),
              'history_sensitivity_not_memory_score': float(np.linalg.norm(xa[-1] - xb[-1]))}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    if not report['passed']:
        raise RuntimeError('Pair wiring check failed; inspect before training')
    print(checks, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
