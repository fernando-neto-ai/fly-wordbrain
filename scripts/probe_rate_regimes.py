#!/usr/bin/env python3
"""Train-only signal diagnostics under explicitly reported global graph scales.

Reuses one complete PlasticBrain. No optimizer, validation/test evaluation,
edge changes, or rule-parameter changes. A static repeat measures backend
variation before a small apparent fast-weight effect is accepted as visible.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly_wordbrain.brain import array_digest
from fly_wordbrain.plastic_brain import PlasticBrain
from fly_wordbrain.plastic_probe import digest, rows_for, stream_difference, sync, write_json


def parameters_digest(model):
    names, values = zip(*model.named_parameters())
    return {"names": list(names), "sha256": array_digest([v.detach().cpu().numpy() for v in values])}


def rms(values, axis=None):
    return np.sqrt(np.mean(np.asarray(values, np.float64) ** 2, axis=axis))


def summarize_array(values):
    values = np.asarray(values)
    return {"rms": float(rms(values)), "min": float(values.min()), "max": float(values.max()),
            "max_abs": float(np.max(np.abs(values)))}


def rollout(model, rows, *, with_constant=False, plastic=False):
    """Fresh state, exact model.step path, only observed training pairs."""
    batch = 2 if with_constant else 1
    state = model.initial_state(batch)
    collected = {key: [] for key in ("features", "pre", "post", "fast", "gain", "saturation_fraction", "activity_rms", "activity_max_abs")}
    for row in rows:
        previous = [row['previous_id']] + ([2] if with_constant else [])
        current = [row['current_id']] + ([2] if with_constant else [])
        features, state = model.step(previous, current, state, plasticity_override=plastic)
        gains = torch.exp(math.log(2.) * torch.tanh(state.fast))
        for key, value in (("features", features), ("pre", state.h[:, model.candidate_pre]),
                           ("post", state.h[:, model.candidate_post]), ("fast", state.fast), ("gain", gains)):
            collected[key].append(value.detach().cpu().numpy())
        collected['saturation_fraction'].append((state.h.abs() > .99).float().mean(1).cpu().numpy())
        collected['activity_rms'].append(state.h.square().mean(1).sqrt().cpu().numpy())
        collected['activity_max_abs'].append(state.h.abs().amax(1).cpu().numpy())
    return {key: np.asarray(value, np.float32) for key, value in collected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--scales', type=float, nargs='+', default=[.0008, .0012, .002, .004, .008])
    parser.add_argument('--positions', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if (not 1 <= args.positions <= 32 or args.threads < 1 or not 1 <= len(args.scales) <= 10
            or any(not math.isfinite(value) or value <= 0 for value in args.scales)):
        raise ValueError('Require1..32positions, positive threads and1..10finite positive explicit scales')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory is nonempty; choose a fresh diagnostic receipt')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    graph_path = args.graph / 'graph.npz' if args.graph.is_dir() else args.graph
    data = json.loads(args.dataset.read_text())
    if data['vocabulary'][:4] != ['<pad>', '<unk>', '<bos>', '<eos>']:
        raise ValueError('Unexpected special word IDs')
    story = data['splits']['train'][0]
    rows = rows_for(story, args.positions)
    if not rows:
        raise ValueError('First training story contains no causal observation rows')
    # All subsequent operations use only the vocabulary and this training prefix.
    vocabulary_size = len(data['vocabulary'])
    del data
    sources = {name: digest(ROOT / 'fly_wordbrain' / name) for name in
               ('plastic_brain.py', 'plastic_probe.py', 'brain.py', 'pair_brain.py', 'pair_extract.py')}
    if torch.device(args.device).type == 'mps':
        sources['metal_sparse.py'] = digest(ROOT / 'fly_wordbrain/metal_sparse.py')
    sources['scripts/probe_rate_regimes.py'] = digest(__file__)
    fixed_config = dict(internal_steps=8, leak=.5, input_gain=1., input_center=0.)
    started = time.perf_counter()
    model = PlasticBrain(graph_path, vocab_size=vocabulary_size, device=args.device,
                         global_scale=args.scales[0], **fixed_config)
    device = next(model.parameters()).device
    initial_frozen = model.verify_frozen()
    initial_parameters = parameters_digest(model)
    candidates_pre = model.candidate_pre.detach().cpu().numpy()
    candidates_post = model.candidate_post.detach().cpu().numpy()
    candidate_group = model.candidate_group.detach().cpu().numpy()
    unique_pre = np.unique(candidates_pre, return_index=True)[1]
    unique_post = np.unique(candidates_post, return_index=True)[1]
    report = {
        'completed': False, 'purpose': 'Train-only numerical signal calibration before any optimization; not language performance or memory evaluation',
        'host': platform.node(), 'python': platform.python_version(), 'torch': str(torch.__version__),
        'device': str(device), 'threads': args.threads, 'train_only': True,
        'story_id': story['id'], 'positions': len(rows),
        'observed_previous_ids': [row['previous_id'] for row in rows],
        'observed_current_ids': [row['current_id'] for row in rows],
        'observed_positions': [row['position'] for row in rows],
        'lexical_positions_after_initial_BOS': sum(row['position'] > 0 for row in rows),
        'graph_sha256': digest(graph_path), 'graph_metadata_sha256': digest(graph_path.with_name('metadata.json')),
        'dataset_sha256': digest(args.dataset), 'source_sha256': sources,
        'initial_frozen_graph_sha256': initial_frozen, 'rule_parameters': initial_parameters,
        'fixed_config': fixed_config, 'scales_requested': args.scales,
        'model_metadata_at_initial_scale': model.metadata(),
        'unique_candidate_sources': len(unique_pre), 'unique_candidate_targets': len(unique_post),
        'protocol': {
            'word_input_control': 'B2 simultaneous actual sliding pairs versus constant BOS/BOS, both static',
            'calibration': 'Actual B2 static lane only: per-edge RMS floor1e-9; feature standard-deviation floor max(max_std*1e-4,1e-9), matching plastic_probe.py',
            'repeat_control': 'Two independent B1 static replays of the identical actual word sequence and zero initial state',
            'fast_effect': 'B1 fast-enabled actual replay versus first B1 static actual replay; identical batch shape and prefix',
            'visible_effect_threshold': 'strictly greater than max(10*B1_static_repeat_max_normalized_feature_difference,1e-5)',
            'resets': 'All neural and fast state reset before each stream; original rule parameters unchanged across every regime',
            'changes': 'Only explicitly reported model.global_scale and prescribed diagnostic activity-scale buffers change',
            'interpretation': 'Visible word response and fast-weight effect establish an observable regime only; no token decodability, ordered memory, or task advantage is established'},
        'regimes': [],
    }
    write_json(args.output / 'report.json', report)
    with torch.no_grad():
        for index, scale in enumerate(args.scales):
            sync(device)
            regime_started = time.perf_counter()
            model.global_scale = float(scale)
            paired = rollout(model, rows, with_constant=True, plastic=False)
            actual_features = paired['features'][:, 0]
            actual_pre, actual_post = paired['pre'][:, 0], paired['post'][:, 0]
            pre_rms = rms(actual_pre, axis=0).astype(np.float32)
            post_rms = rms(actual_post, axis=0).astype(np.float32)
            pre_scale, post_scale = np.maximum(pre_rms, 1e-9), np.maximum(post_rms, 1e-9)
            feature_mean = actual_features.mean(0)
            raw_std = actual_features.std(0)
            std_floor = max(float(raw_std.max()) * 1e-4, 1e-9)
            feature_std = np.maximum(raw_std, std_floor)
            model.set_activity_scales(pre_scale, post_scale)
            reference = rollout(model, rows, plastic=False)
            repeat = rollout(model, rows, plastic=False)
            plastic = rollout(model, rows, plastic=True)
            repeat_effect = stream_difference(reference['features'][:, 0], repeat['features'][:, 0], feature_std)
            fast_effect = stream_difference(reference['features'][:, 0], plastic['features'][:, 0], feature_std)
            word_effect = stream_difference(paired['features'][:, 0], paired['features'][:, 1], feature_std)
            noise = repeat_effect['normalized_max_abs_difference']
            threshold = max(10. * noise, 1e-5)
            arrays = {'feature_mean': feature_mean, 'feature_std': feature_std,
                      'pre_scale': pre_scale, 'post_scale': post_scale,
                      **{'static_pair_' + key: value for key, value in paired.items()},
                      **{'static_reference_' + key: value for key, value in reference.items()},
                      **{'static_repeat_' + key: value for key, value in repeat.items()},
                      **{'plastic_' + key: value for key, value in plastic.items()}}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise RuntimeError('Nonfinite diagnostic arrays at global scale ' + str(scale))
            path = args.output / ('regime-%02d.npz' % index)
            with path.with_suffix('.npz.partial').open('wb') as stream:
                np.savez(stream, **arrays)
            path.with_suffix('.npz.partial').replace(path)
            groups = []
            for group, name in enumerate(('hDeltaH', 'hDeltaA', 'hDeltaI', 'hDeltaG')):
                edge_ix = np.flatnonzero(candidate_group == group)
                pre_ix = edge_ix[np.unique(candidates_pre[edge_ix], return_index=True)[1]]
                post_ix = edge_ix[np.unique(candidates_post[edge_ix], return_index=True)[1]]
                groups.append({'group': group, 'target_type': name, 'edges': len(edge_ix),
                    'unique_source_nodes': len(pre_ix), 'unique_target_nodes': len(post_ix),
                    'static_pre': summarize_array(actual_pre[:, pre_ix]),
                    'static_post': summarize_array(actual_post[:, post_ix]),
                    'pre_word_effect': stream_difference(actual_pre[:, pre_ix], paired['pre'][:, 1, pre_ix], pre_scale[pre_ix]),
                    'post_word_effect': stream_difference(actual_post[:, post_ix], paired['post'][:, 1, post_ix], post_scale[post_ix]),
                    'plastic_fast': summarize_array(plastic['fast'][:, 0, edge_ix]),
                    'plastic_gain': summarize_array(plastic['gain'][:, 0, edge_ix])})
            sync(device)
            entry = {'index': index, 'global_scale': scale, 'positions': len(rows), 'std_floor': std_floor,
                'feature_std_max': float(raw_std.max()),
                'actual_static_pre_rms_max': float(pre_rms.max()), 'actual_static_post_rms_max': float(post_rms.max()),
                'static_pair_activity_rms': paired['activity_rms'].tolist(),
                'static_pair_saturation_fraction': paired['saturation_fraction'].tolist(),
                'max_saturation_fraction': float(max(paired['saturation_fraction'].max(), plastic['saturation_fraction'].max())),
                'word_feature_effect': word_effect, 'static_repeat_feature_noise': repeat_effect,
                'fast_feature_effect': fast_effect,
                'batch_shape_feature_difference': stream_difference(actual_features, reference['features'][:, 0], feature_std),
                'visible_effect_threshold': threshold,
                'word_effect_above_threshold': word_effect['normalized_max_abs_difference'] > threshold,
                'fast_effect_above_threshold': fast_effect['normalized_max_abs_difference'] > threshold,
                'max_abs_fast_state': float(np.max(np.abs(plastic['fast']))),
                'candidate_gain_min': float(plastic['gain'].min()), 'candidate_gain_max': float(plastic['gain'].max()),
                'candidate_max_abs_gain_change': float(np.max(np.abs(plastic['gain'] - 1.))),
                'groups': groups, 'arrays_file': path.name, 'arrays_sha256': digest(path),
                'seconds': time.perf_counter() - regime_started,
                'rule_parameters_unchanged': parameters_digest(model) == initial_parameters}
            if not entry['rule_parameters_unchanged']:
                raise RuntimeError('Rule parameters changed during the no-optimizer diagnostic')
            entry['visible_word_and_fast_effect'] = entry['word_effect_above_threshold'] and entry['fast_effect_above_threshold']
            report['regimes'].append(entry)
            write_json(args.output / 'report.json', report)
            print(json.dumps({'event': 'rate_regime', **{key: entry[key] for key in
                ('global_scale', 'visible_effect_threshold', 'word_effect_above_threshold', 'fast_effect_above_threshold',
                 'max_abs_fast_state', 'candidate_max_abs_gain_change', 'max_saturation_fraction', 'seconds')},
                'word_effect': word_effect['normalized_max_abs_difference'],
                'repeat_noise': noise, 'fast_effect': fast_effect['normalized_max_abs_difference']}, allow_nan=False), flush=True)
    report['final_frozen_graph_sha256'] = model.verify_frozen()
    report['frozen_graph_unchanged'] = report['final_frozen_graph_sha256'] == initial_frozen
    report['final_rule_parameters'] = parameters_digest(model)
    report['rule_parameters_unchanged'] = report['final_rule_parameters'] == initial_parameters
    if not report['frozen_graph_unchanged'] or not report['rule_parameters_unchanged']:
        raise RuntimeError('Immutable graph/interface or rule parameter check failed')
    report['completed'] = True
    report['elapsed_seconds'] = time.perf_counter() - started
    write_json(args.output / 'report.json', report)
    print(json.dumps({'event': 'rate_regimes_complete', 'report': str(args.output / 'report.json'),
                      'elapsed_seconds': report['elapsed_seconds']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
