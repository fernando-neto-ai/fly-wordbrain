"""Causal sliding-pair observations with two genuinely future word targets."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from .extract import atomic_json, save_npz, sha256_file
from .pair_brain import PairedFrozenBrain

WORKER = None
KEYS = ('X', 'X_direct', 'y', 'target_mask', 'positions', 'previous_ids', 'current_ids', 'story_ids')


def story_rows(story):
    ids = story['word_ids']
    masks = story['target_mask']
    if not ids or ids[0] != 2 or len(ids) != len(masks):
        raise ValueError('Require aligned source IDs/masks beginning with BOS')
    if len(story['words']) > 128:
        raise ValueError('Source story exceeds 128 lexical words')
    for p in range(len(ids) - 1):
        if p > 128 or ids[p] == 3:
            raise ValueError('Observation after context limit or EOS')
        first_valid = bool(masks[p + 1])
        second_valid = p + 2 < len(ids) and bool(masks[p + 2]) and ids[p + 1] != 3
        # Source stories have no interior padding. Fail rather than skip a
        # recurrent update and silently change the state used by later rows.
        if not first_valid:
            raise ValueError('Interior masked targets unsupported by this bounded corpus')
        yield {'position': p, 'previous_id': ids[p - 1] if p else 2,
               'current_id': ids[p],
               'targets': (ids[p + 1], ids[p + 2] if second_valid else 0),
               'target_mask': (first_valid, second_valid)}


def init_worker(config):
    global WORKER
    WORKER = PairedFrozenBrain(**config)


def story_features(story):
    brain = WORKER
    brain.reset()
    arrays = {key: [] for key in KEYS}
    kernel_seconds, spikes, descending_spikes = 0., 0, 0
    started = time.perf_counter()
    for row in story_rows(story):
        pair = (row['previous_id'], row['current_id'])
        arrays['X'].append(brain.step(pair))
        arrays['X_direct'].append(brain.direct_features(pair))
        arrays['y'].append(row['targets'])
        arrays['target_mask'].append(row['target_mask'])
        arrays['positions'].append(row['position'])
        arrays['previous_ids'].append(row['previous_id'])
        arrays['current_ids'].append(row['current_id'])
        arrays['story_ids'].append(story['id'])
        diagnostics = brain.last_diagnostics
        kernel_seconds += diagnostics['kernel_seconds']
        spikes += diagnostics['all_spikes']
        descending_spikes += diagnostics['readout_spikes']
    arrays = {key: np.asarray(value, dtype=(np.float32 if key in ('X', 'X_direct')
                else bool if key == 'target_mask' else str if key == 'story_ids' else np.int64))
              for key, value in arrays.items()}
    arrays['diagnostics'] = {'story_id': story['id'], 'observations': len(arrays['X']),
        'scored_targets_by_horizon': arrays['target_mask'].sum(axis=0).tolist(),
        'wall_seconds': time.perf_counter() - started, 'kernel_seconds': kernel_seconds,
        'all_spikes': spikes, 'readout_spikes': descending_spikes,
        'frozen_graph_sha256': brain.verify_frozen()}
    return arrays


def extract(dataset_path, output, workers=4):
    if not 1 <= workers <= 4:
        raise ValueError('Require 1..4 independent story workers')
    dataset_path, output = Path(dataset_path), Path(output)
    dataset = json.loads(dataset_path.read_text())
    if dataset['vocabulary'][:4] != ['<pad>', '<unk>', '<bos>', '<eos>']:
        raise ValueError('Unexpected reserved IDs')
    for stories in dataset['splits'].values():
        for story in stories:
            list(story_rows(story))
    config = {'vocab_size': len(dataset['vocabulary']), 'word_ms': 20., 'warmup_ms': 100.,
              'bins': 128, 'seed': 1729, 'high': .02}
    brain = PairedFrozenBrain(**config)
    manifest = {'schema': 1, 'task': 'ordered bigram input, independent two-future-word readout',
                'dataset_sha256': sha256_file(dataset_path), 'brain': brain.metadata(),
                'workers': workers, 'code_sha256': {name: sha256_file(Path(__file__).with_name(name))
                    for name in ('brain.py', 'pair_brain.py', 'extract.py', 'pair_extract.py')},
                'target_contract': 'row p sees ids[p-1],ids[p]; targets ids[p+1],ids[p+2]; missing tail masked',
                'context_contract': 'Reset each story; <=128 lexical context words; BOS excluded',
                'scoring_contract': 'Two forecast horizons overlap across rows; head1 is ordinary next-word CE'}
    del brain
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['identity'] = identity
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()).get('identity') != identity:
        raise RuntimeError('Cache source/config differs; use a new output directory')
    atomic_json(path, manifest)
    progress = {'state': 'extracting', 'identity': identity, 'completed': 0,
                'total': sum(len(s) for s in dataset['splits'].values())}
    atomic_json(output / 'progress.json', progress)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=init_worker, initargs=(config,)) as pool:
        for split in ('train', 'val', 'test'):
            chunks = output / 'chunks' / split
            chunks.mkdir(parents=True, exist_ok=True)
            pending = {}
            for i, story in enumerate(dataset['splits'][split]):
                chunk = chunks / ('%05d.npz' % i)
                receipt = chunk.with_suffix('.json')
                if chunk.exists() and receipt.exists():
                    prior = json.loads(receipt.read_text())
                    if (prior['story_id'] != story['id'] or prior['npz_sha256'] != sha256_file(chunk)
                            or prior['frozen_graph_sha256'] != manifest['brain']['graph_sha256']):
                        raise RuntimeError('Cached story identity/hash mismatch')
                    progress['completed'] += 1
                else:
                    pending[pool.submit(story_features, story)] = chunk
            for future in as_completed(pending):
                chunk = pending[future]
                result = future.result()
                receipt = result.pop('diagnostics')
                if receipt['frozen_graph_sha256'] != manifest['brain']['graph_sha256']:
                    raise RuntimeError('Worker changed graph')
                save_npz(chunk, **result)
                receipt['npz_sha256'] = sha256_file(chunk)
                atomic_json(chunk.with_suffix('.json'), receipt)
                progress.update(completed=progress['completed'] + 1, split=split,
                                elapsed_seconds=time.perf_counter() - started, latest=receipt)
                atomic_json(output / 'progress.json', progress)
                if progress['completed'] % 8 == 0:
                    print(json.dumps(progress), flush=True)
            arrays = {key: [] for key in KEYS}
            for i in range(len(dataset['splits'][split])):
                with np.load(chunks / ('%05d.npz' % i), allow_pickle=False) as chunk:
                    for key in KEYS:
                        arrays[key].append(chunk[key])
            save_npz(output / (split + '.npz'), **{key: np.concatenate(value) for key, value in arrays.items()})
    verification = PairedFrozenBrain(**config)
    manifest['final_graph_sha256'] = verification.verify_frozen()
    assert manifest['final_graph_sha256'] == manifest['brain']['graph_sha256']
    manifest.update(completed=True, elapsed_seconds=time.perf_counter() - started,
                    split_hashes={s: sha256_file(output / (s + '.npz')) for s in ('train', 'val', 'test')})
    atomic_json(path, manifest)
    progress.update(state='complete', elapsed_seconds=manifest['elapsed_seconds'])
    atomic_json(output / 'progress.json', progress)
    print(json.dumps(progress), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    extract(args.dataset, args.output, args.workers)


if __name__ == '__main__':
    main()
