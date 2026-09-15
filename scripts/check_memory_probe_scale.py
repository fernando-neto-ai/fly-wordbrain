"""Post hoc numerical-scale ablation on an existing immutable feature cache."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fly_wordbrain.memory_probe_readouts import fit_probe
from fly_wordbrain.pair_decoder import file_sha256
from fly_wordbrain.action_train import write_json


def run(output):
    output = Path(output)
    rows = json.loads((output/'rows.json').read_text())
    cache = np.load(output/'features.npz', allow_pickle=False)
    report = {'kind': 'posthoc_numerical_scale_sensitivity', 'status': 'running',
        'rationale': 'Primary floor masked all early pooled identity features; exact MPS replay noise was zero. '
            'Check finite signal with floor1e-12 and no relative mask. Heldout results already inspected; '
            'exploratory diagnostic, not model selection.',
        'config': {'absolute_std_floor': 1e-12, 'relative_std_floor': 0},
        'features_sha256': file_sha256(output/'features.npz'),
        'rows_sha256': file_sha256(output/'rows.json'), 'results': {}}
    started = time.monotonic()
    for task in ('identity', 'order'):
        indices = [i for i, row in enumerate(rows) if row['task'] == task]
        labels = np.array([rows[i]['label'] for i in indices])
        splits = np.array([rows[i]['split'] for i in indices])
        classes = int(labels.max())+1
        for name in cache.files:
            for architecture in ('linear', 'mlp'):
                key = task+'/'+name+'/'+architecture
                result = fit_probe(cache[name][indices], labels, splits, classes,
                    architecture=architecture, absolute_std_floor=1e-12, relative_std_floor=0)
                report['results'][key] = result
                print('FIT', key, flush=True)
        for architecture in ('linear', 'mlp'):
            key = task+'/on_pooled_early/'+architecture+'_shuffled'
            report['results'][key] = fit_probe(cache['on_pooled_early'][indices], labels, splits, classes,
                architecture=architecture, absolute_std_floor=1e-12, relative_std_floor=0,
                shuffle_train_labels=True)
    report.update(status='completed', elapsed_seconds=time.monotonic()-started)
    write_json(output/'scale-sensitivity.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
