"""Bounded, frozen-circuit identity/order diagnostic; never updates the live run.

Run extraction on the M3 GPU, then fit cached tiny heads on CPU. Probe test
contexts are held out from probe fitting, not necessarily from the original
language-model training. This measures accessible context, not language quality.
"""
import argparse
from collections import Counter, defaultdict
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.action_model import CandidateInput
from fly_wordbrain.feedback_data import collate_feedback
from fly_wordbrain.memory_probe_data import build_memory_probes
from fly_wordbrain.memory_probe_features import load_frozen_model, extract_batch, candidate_neuron_indices
from fly_wordbrain.memory_probe_readouts import fit_probe
from fly_wordbrain.pair_decoder import file_sha256


TEMPORAL_POSITIONS = [3, 4, 5, 7]  # Zero-based prediction times, not observed-word indices.


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def parameter_digest(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        result.update(name.encode())
        result.update(parameter.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def groups(rows):
    result = defaultdict(list)
    for i, row in enumerate(rows):
        result[row['group_id']].append(i)
    return result


def check_final_inputs(batch, rows):
    for indices in groups(rows).values():
        for name in ('previous', 'current', 'candidates', 'probabilities'):
            value = getattr(batch, name)[indices, -1].cpu().numpy()
            if not np.array_equal(value, np.broadcast_to(value[:1], value.shape)):
                raise AssertionError('Final input varies with probe label: ' + name)
    return {'groups': len(groups(rows)), 'identical_final_inputs_within_every_group': True}


def within_group_rms(values, rows):
    by_task = defaultdict(list)
    for indices in groups(rows).values():
        x = values[indices].astype(np.float64)
        by_task[rows[indices[0]]['task']].append(float(np.sqrt(np.mean((x-x.mean(axis=0))**2))))
    return {task: {'mean': float(np.mean(v)), 'max': float(np.max(v))} for task, v in by_task.items()}


def extract(args):
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    started = time.monotonic()
    def progress(stage, **values):
        value = {'status': 'running', 'stage': stage, 'pid': os.getpid(),
                 'elapsed_seconds': time.monotonic()-started, **values}
        write_json(output/'progress.json', value)
        print(json.dumps(value), flush=True)
    torch.set_num_threads(4)
    progress('loading_checkpoint')
    model, receipt = load_frozen_model(args.checkpoint, args.graph, args.structure, args.device)
    before = parameter_digest(model)
    data = json.loads(Path(args.dataset).read_text())
    dataset_sha = file_sha256(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if dataset_sha != checkpoint['protocol']['dataset_sha256']:
        raise AssertionError('Dataset differs from frozen checkpoint')
    source_checks = {}
    root = Path(__file__).resolve().parents[1]
    for rel, expected in checkpoint['protocol']['source_sha256'].items():
        path = root/rel if '/' in rel else root/'fly_wordbrain'/rel
        source_checks[str(path)] = file_sha256(path) == expected
    if not all(source_checks.values()):
        raise AssertionError('Runtime source differs from checkpoint: ' + str(source_checks))
    del checkpoint
    probes = build_memory_probes(data)
    rows, windows = probes.rows, probes.windows
    proposals = ActionCandidates(data['splits']['train'], len(data['vocabulary']), top_k=10)
    cpu_batch = collate_feedback(windows, proposals, device='cpu', exclude_own_story=True)
    checks = check_final_inputs(cpu_batch, rows)
    train_groups = {task: [] for task in sorted({r['task'] for r in rows})}
    for row in rows:
        bucket = train_groups[row['task']]
        if row['split'] == 'train' and row['group_id'] not in bucket and len(bucket) < 8:
            bucket.append(row['group_id'])
    chosen = {g for values in train_groups.values() for g in values}
    calibration = [i for i, row in enumerate(rows) if row['group_id'] in chosen]
    assert all(rows[i]['split'] == 'train' for i in calibration)
    protocol = {'kind': 'frozen_context_accessibility_probe', 'schema': 1,
        'snapshot': receipt, 'dataset_sha256': dataset_sha, 'probe_data': probes.metadata,
        'windows': len(rows), 'batch_size': args.batch_size, 'primary_plasticity': True,
        'slow_parameters_frozen': True, 'counts': 'Full original TRAIN minus entire source story in all probe splits',
        'selection': {'method': 'sum of within-time activity variance across eight prediction times',
            'candidate_pool': 'all neurons except directly driven retina', 'labels_used': False,
            'calibration_row_indices': calibration, 'final_cells': 256, 'temporal_cells': 64},
        'temporal_prediction_positions_one_based': [x+1 for x in TEMPORAL_POSITIONS],
        'temporal_readout_caveat': 'External storage of earlier responses; does not demonstrate final-state memory',
        'heads': ['linear 256->classes', 'ReLU 256->16->classes'], 'head_seeds': [0],
        'positive_control': '128 actual current-role retinal currents for word2 concatenated with 128 for word3',
        'negative_control': '128 previous-role and128 current-role retinal currents at final prediction',
        'shuffled_label_control': 'Train-label permutation for fast-on temporal cells; true heldout labels',
        'limits': ['Synthetic intervention, not next-word accuracy', 'Probe groups are held out only from probe fitting',
                   'Small heldout context sample; no connectome geometry advantage control'],
        'source_checks': source_checks,
        'probe_source_sha256': {str(p.relative_to(root)): file_sha256(p) for p in
            [Path(__file__), root/'fly_wordbrain/memory_probe_data.py', root/'fly_wordbrain/memory_probe_features.py',
             root/'fly_wordbrain/memory_probe_readouts.py']}}
    write_json(output/'protocol.json', protocol)
    write_json(output/'rows.json', rows)
    candidates = candidate_neuron_indices(model, mode='all_nonretinal')
    candidates = np.asarray(candidates, dtype=np.int64)
    sums = np.zeros((8, len(candidates)), np.float64)
    squares = np.zeros_like(sums)
    count = 0
    for offset in range(0, len(calibration), args.batch_size):
        indices = calibration[offset:offset+args.batch_size]
        batch = collate_feedback([windows[i] for i in indices], proposals, device=args.device, exclude_own_story=True)
        features = extract_batch(model, batch, selected_indices=candidates, plasticity=True)
        values = features['temporal_selected'].astype(np.float64)
        sums += values.sum(axis=0)
        squares += (values*values).sum(axis=0)
        count += len(indices)
        progress('selecting_responsive_neurons', completed=count, total=len(calibration))
    variance = np.maximum(squares/count-(sums/count)**2, 0).sum(axis=0)
    ranking = np.argsort(-variance, kind='stable')
    selected = candidates[ranking[:256]]
    assert not np.intersect1d(selected, model.brain.retina.cpu().numpy()).size
    selection = {'indices': selected.tolist(), 'neuron_ids': model.brain.neuron_ids[selected].cpu().tolist(),
                 'summed_time_variance': variance[ranking[:256]].tolist(),
                 'candidate_count': len(candidates), 'calibration_rows': count}
    write_json(output/'selection.json', selection)
    del sums, squares, values, features
    cache = {}
    codes = model.brain.action_codes.cpu().numpy()
    previous_start, current_start = model.brain.bank_offsets[:2]
    gain, center = model.brain.input_gain, model.brain.input_center
    ids = cpu_batch.observed_ids.numpy()
    cache['input_history'] = gain*(np.concatenate([codes[ids[:,1],current_start:current_start+128],
                              codes[ids[:,2],current_start:current_start+128]], axis=1)-center)
    cache['input_final'] = gain*(np.concatenate([codes[ids[:,5],previous_start:previous_start+128],
                            codes[ids[:,6],current_start:current_start+128]], axis=1)-center)
    checks['input_final_within_group_rms'] = within_group_rms(cache['input_final'], rows)
    assert all(v['max'] == 0 for v in checks['input_final_within_group_rms'].values())
    activity_effects = []
    for enabled in (True, False):
        name = 'on' if enabled else 'off'
        parts = defaultdict(list)
        for offset in range(0, len(windows), args.batch_size):
            batch = collate_feedback(windows[offset:offset+args.batch_size], proposals,
                                     device=args.device, exclude_own_story=True)
            values = extract_batch(model, batch, selected_indices=selected, plasticity=enabled)
            parts['pooled_final'].append(values['final_pooled'])
            parts['cells_final'].append(values['final_selected'])
            parts['cells_temporal'].append(values['temporal_selected'][:,TEMPORAL_POSITIONS,:64].reshape(-1,256))
            parts['pooled_early'].append(values['temporal_pooled'][:,3])
            if offset == 0:
                with torch.no_grad():
                    replay, state = model.features(batch, plasticity=enabled)
                    delta = np.max(np.abs(replay.cpu().numpy()-values['final_pooled']))
                    checks[name+'_replay_and_extraction_max_abs'] = float(delta)
                    if delta > 1e-7:
                        raise AssertionError('Extraction differs from unmodified model.features')
                    changed = dataclasses.replace(batch, targets=(batch.targets+1) % model.brain.vocab_size)
                    target_changed, _ = model.features(changed, plasticity=enabled)
                    target_delta = (replay-target_changed).abs().max().item()
                    checks[name+'_target_mutation_max_abs'] = target_delta
                    if target_delta > 1e-7:
                        raise AssertionError('Eighth word label changed features')
                    checks[name+'_fast_state_rms'] = float(state.fast.cpu().double().square().mean().sqrt())
                    sensory = model.brain.retinal_current(CandidateInput(batch.previous[:,-1],
                        batch.candidates[:,-1],batch.probabilities[:,-1]),batch.current[:,-1])
                    read = torch.cat([sensory[:,model.brain.action_neurons[:128]],
                        sensory[:,model.brain.action_neurons[current_start:current_start+128]]], dim=1).cpu().numpy()
                    np.testing.assert_array_equal(read, cache['input_final'][:len(read)])
            progress('caching_'+name, completed=min(offset+args.batch_size,len(windows)), total=len(windows))
        for field, values in parts.items():
            cache[name+'_'+field] = np.concatenate(values)
    checks['parameter_digest_before'] = before
    checks['parameter_digest_after'] = parameter_digest(model)
    checks['parameters_unchanged'] = checks['parameter_digest_before'] == checks['parameter_digest_after']
    checks['all_parameters_frozen'] = all(not p.requires_grad for p in model.parameters())
    if not checks['parameters_unchanged'] or not checks['all_parameters_frozen']:
        raise AssertionError('Slow parameter mutation')
    checks['feature_within_group_rms'] = {name: within_group_rms(v,rows) for name,v in cache.items()}
    checks['on_off_feature_rms'] = {name: float(np.sqrt(np.mean((cache['on_'+name].astype(np.float64)-
        cache['off_'+name])**2))) for name in ('pooled_final','cells_final','cells_temporal','pooled_early')}
    checks['base_graph_preserved'] = model.brain.base_graph_fingerprint() == model.brain.base_graph_sha256_before_selection
    if not checks['base_graph_preserved']:
        raise AssertionError('Base graph mutation')
    np.savez_compressed(output/'features.npz', **cache)
    checks['cache_sha256'] = file_sha256(output/'features.npz')
    write_json(output/'checks.json', checks)
    write_json(output/'progress.json', {'status':'extracted','pid':os.getpid(),'elapsed_seconds':time.monotonic()-started,
        'windows':len(windows),'feature_sets':list(cache)})
    print('EXTRACTION_COMPLETED', flush=True)


def fit(args):
    output = Path(args.output)
    rows = json.loads((output/'rows.json').read_text())
    cache = np.load(output/'features.npz', allow_pickle=False)
    started = time.monotonic()
    report = {'status':'running','kind':'controlled_memory_probe_readouts','results':{},
              'features_sha256':file_sha256(output/'features.npz'), 'rows_sha256':file_sha256(output/'rows.json')}
    for task in sorted({row['task'] for row in rows}):
        indices = [i for i,row in enumerate(rows) if row['task'] == task]
        labels = np.array([rows[i]['label'] for i in indices])
        splits = np.array([rows[i]['split'] for i in indices])
        classes = int(labels.max())+1
        for name in cache.files:
            for architecture in ('linear','mlp'):
                key = task+'/'+name+'/'+architecture
                report['results'][key] = fit_probe(cache[name][indices],labels,splits,classes,architecture=architecture)
                write_json(output/'readouts.json',report)
                print('FIT',key,flush=True)
        for architecture in ('linear','mlp'):
            key = task+'/on_cells_temporal/'+architecture+'_shuffled'
            report['results'][key] = fit_probe(cache['on_cells_temporal'][indices],labels,splits,classes,
                                              architecture=architecture,shuffle_train_labels=True)
    report.update(status='completed',elapsed_seconds=time.monotonic()-started)
    write_json(output/'readouts.json',report)
    print('READOUTS_COMPLETED',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('extract','fit'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--graph')
    parser.add_argument('--structure')
    parser.add_argument('--dataset')
    parser.add_argument('--device',default='mps')
    parser.add_argument('--batch-size',type=int,default=16)
    args = parser.parse_args()
    try:
        (extract if args.mode == 'extract' else fit)(args)
    except Exception:
        write_json(Path(args.output)/'failure.json',{'traceback':traceback.format_exc()})
        raise
