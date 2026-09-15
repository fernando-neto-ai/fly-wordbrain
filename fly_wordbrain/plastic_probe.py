"""Train-only observability, calibration, and measured sparse BPTT timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .pair_extract import story_rows
from .plastic_brain import PlasticBrain


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    elif str(device).startswith('mps'):
        torch.mps.synchronize()


def memory_sample(device):
    """MPS exposes current allocation, not a true resettable peak statistic."""
    if torch.device(device).type == 'mps':
        return dict(mps_current_allocated_bytes=int(torch.mps.current_allocated_memory()),
                    mps_driver_allocated_bytes=int(torch.mps.driver_allocated_memory()))
    if torch.device(device).type == 'cuda':
        return dict(current_allocated_bytes=int(torch.cuda.memory_allocated()),
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
    return {}


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), allow_nan=False), flush=True)


def rows_for(story, limit):
    return list(story_rows(story))[:limit]


def stream_difference(actual, constant, scale):
    """Descriptive effect sizes; a word perturbation is not a memory test."""
    actual, constant = np.asarray(actual, np.float64), np.asarray(constant, np.float64)
    scale = np.asarray(scale, np.float64)
    if actual.shape != constant.shape or actual.ndim != 2 or not np.all(scale > 0):
        raise ValueError('Mismatched stream controls or nonpositive diagnostic scales')
    difference = actual - constant
    normalized = difference / scale
    return dict(raw_rms_difference=float(np.sqrt(np.mean(difference ** 2))),
                raw_max_abs_difference=float(np.max(np.abs(difference))),
                normalized_rms_difference=float(np.sqrt(np.mean(normalized ** 2))),
                normalized_max_abs_difference=float(np.max(np.abs(normalized))),
                actual_raw_rms=float(np.sqrt(np.mean(actual ** 2))),
                constant_raw_rms=float(np.sqrt(np.mean(constant ** 2))),
                normalized_rms_by_position=np.sqrt(np.mean(normalized ** 2, axis=1)).tolist())


def numerical_effect_gate(fast_difference, repeat_difference, *, repeat_multiplier=10., practical_floor=1e-5):
    """Require a strict margin above a matched replay and an explicit floor."""
    values = np.asarray([fast_difference, repeat_difference, repeat_multiplier, practical_floor], np.float64)
    if not np.isfinite(values).all() or np.any(values[:2] < 0) or np.any(values[2:] <= 0):
        raise ValueError('Effect differences must be finite nonnegative values and thresholds positive')
    threshold = max(float(repeat_multiplier * repeat_difference), float(practical_floor))
    return dict(passed=bool(fast_difference > threshold),
                fast_max_normalized_difference=float(fast_difference),
                frozen_repeat_max_normalized_difference=float(repeat_difference),
                repeat_multiplier=float(repeat_multiplier), practical_normalized_floor=float(practical_floor),
                required_strict_lower_bound=threshold,
                criterion='fast_difference > max(10 * frozen_repeat_difference, 1e-5)',
                interpretation='Numerical effect screening only; one matched replay does not estimate a noise distribution or prove useful language memory.')


@torch.no_grad()
def measure_fast_effect(model, story, word_limit, feature_std):
    """Three independent causal states: frozen, frozen replay, and fast enabled."""
    rows = rows_for(story, word_limit)
    if not rows:
        raise ValueError('Fast-effect probe requires a nonempty causal word prefix')
    if feature_std.ndim != 1 or not bool(torch.isfinite(feature_std).all()) or not bool((feature_std > 0).all()):
        raise ValueError('Fast-effect scales must be a finite positive feature vector')
    static, repeated, plastic = (model.initial_state(1) for _ in range(3))
    fast_by_position, repeat_by_position = [], []
    fast_max = 0.
    for row in rows:
        prev = torch.tensor([row['previous_id']], device=feature_std.device)
        curr = torch.tensor([row['current_id']], device=feature_std.device)
        a, static = model.step(prev, curr, static, plasticity_override=False)
        replay, repeated = model.step(prev, curr, repeated, plasticity_override=False)
        b, plastic = model.step(prev, curr, plastic, plasticity_override=True)
        fast_by_position.append(float(((a - b) / feature_std).abs().max().item()))
        repeat_by_position.append(float(((a - replay) / feature_std).abs().max().item()))
        fast_max = max(fast_max, float(plastic.fast.abs().max().item()))
    delta, repeat_delta = max(fast_by_position), max(repeat_by_position)
    return dict(story_id=story['id'], positions=len(rows), max_normalized_feature_difference=delta,
        max_normalized_frozen_repeat_difference=repeat_delta, max_fast_state=fast_max,
        fast_difference_by_position=fast_by_position, frozen_repeat_difference_by_position=repeat_by_position,
        numerical_gate=numerical_effect_gate(delta, repeat_delta),
        control='Independent zero-initialized frozen and frozen-replay states receive the same actual ordered word stream; the third independent state enables fast weights. Only previous/current words enter each step.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--global-scale', type=float, required=True)
    p.add_argument('--internal-steps', type=int, default=8)
    p.add_argument('--leak', type=float, default=.5)
    p.add_argument('--input-gain', type=float, default=1.)
    p.add_argument('--input-center', type=float, default=0.)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--calibration-stories', type=int, default=4)
    p.add_argument('--calibration-words', type=int, default=32)
    p.add_argument('--benchmark-words', type=int, default=32)
    p.add_argument('--benchmark-batch', type=int, default=2)
    args = p.parse_args()
    if not 1 <= args.benchmark_words <= 128:
        raise ValueError('Benchmark must use 1..128 observed positions')
    if (not 1 <= args.calibration_words <= 128 or args.calibration_stories < 1
            or args.benchmark_batch < 1):
        raise ValueError('Require 1..128 calibration positions and positive story/batch counts')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    graph = args.graph / 'graph.npz' if args.graph.is_dir() else args.graph
    metadata = graph.with_name('metadata.json')
    data = json.loads(args.dataset.read_text())
    train = data['splits']['train']
    if not train or data['vocabulary'][:4] != ['<pad>', '<unk>', '<bos>', '<eos>']:
        raise ValueError('Require training stories and the pinned special-token IDs')
    config = {k: getattr(args, k) for k in
              ('global_scale', 'internal_steps', 'leak', 'input_gain', 'input_center')}
    provenance = dict(graph_sha256=digest(graph), graph_metadata_sha256=digest(metadata),
                      dataset_sha256=digest(args.dataset), train_only=True,
                      story_ids=[s['id'] for s in train[:args.calibration_stories]],
                      source_sha256={f: digest(Path(__file__).with_name(f)) for f in
                          ('plastic_probe.py', 'plastic_brain.py', 'pair_brain.py', 'brain.py', 'pair_extract.py')})
    if torch.device(args.device).type == 'mps':
        provenance['source_sha256']['metal_sparse.py'] = digest(Path(__file__).with_name('metal_sparse.py'))
    started = time.perf_counter()
    model = PlasticBrain(graph, vocab_size=len(data['vocabulary']), device=args.device,
                         **config)
    device = next(model.parameters()).device
    runtime = dict(host=platform.node(), python=platform.python_version(),
                   torch=str(torch.__version__), device=str(device), threads=args.threads,
                   sparse_backend=model.sparse_backend)
    if device.type == 'cuda':
        assert torch.cuda.device_count() == 1
        runtime.update(gpu=torch.cuda.get_device_name(), hip=torch.version.hip,
                       gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    elif device.type == 'mps':
        runtime.update(gpu='Apple Metal (MPS)',
                       mps_fallback_enabled=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '1',
                       mps_recommended_max_bytes=int(torch.mps.recommended_max_memory()),
                       mps_memory_reporting='Allocation samples at stage boundaries; no exact peak counter is exposed.')
    emit('model_loaded', seconds=time.perf_counter() - started, **runtime)
    before = model.verify_frozen()
    features, pre_values, post_values = [], [], []
    first_activity = []
    saturated, states_count = 0, 0
    with torch.no_grad():
        for index, story in enumerate(train[:args.calibration_stories]):
            state = model.initial_state(1)
            for position, row in enumerate(rows_for(story, args.calibration_words)):
                previous = torch.tensor([row['previous_id']], device=device)
                current = torch.tensor([row['current_id']], device=device)
                x, state = model.step(previous, current, state, plasticity_override=False)
                features.append(x.cpu().numpy()[0])
                pre_values.append(state.h[:, model.candidate_pre].cpu().numpy()[0])
                post_values.append(state.h[:, model.candidate_post].cpu().numpy()[0])
                if index == 0 and position < 16:
                    first_activity.append(state.h.cpu().numpy()[0])
                saturated += int((state.h.abs() > .99).sum().item())
                states_count += state.h.numel()
            emit('calibration_story', index=index, story_id=story['id'])
    X = np.asarray(features, np.float32)
    pre, post = np.asarray(pre_values, np.float32), np.asarray(post_values, np.float32)
    pre_rms = np.sqrt(np.mean(pre.astype(np.float64) ** 2, axis=0)).astype(np.float32)
    post_rms = np.sqrt(np.mean(post.astype(np.float64) ** 2, axis=0)).astype(np.float32)
    # Fixed calibration scales, never estimated from validation/test or updated online.
    pre_scale = np.maximum(pre_rms, 1e-9)
    post_scale = np.maximum(post_rms, 1e-9)
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    std_floor = max(float(feature_std.max()) * 1e-4, 1e-9)
    feature_std = np.maximum(feature_std, std_floor)
    model.set_activity_scales(pre_scale, post_scale)
    calibration_path = args.output / 'calibration.npz'
    np.savez(calibration_path, feature_mean=feature_mean, feature_std=feature_std,
             pre_scale=pre_scale, post_scale=post_scale)
    calibration = dict(**config, **provenance,
                       calibration_npz_sha256=digest(calibration_path),
                       calibration_positions=len(X), std_floor=std_floor,
                       calibration_position_limit_per_story=args.calibration_words,
                       probe_passed=False, probe_file='probe.json', probe_sha256=None)
    write_json(args.output / 'calibration.json', calibration)
    groups = model.candidate_group.detach().cpu().numpy()
    candidate_pre = model.candidate_pre.detach().cpu().numpy()
    candidate_post = model.candidate_post.detach().cpu().numpy()
    observability = dict(saturation_fraction=saturated / max(1, states_count),
        feature_varying_dimensions=int((X.std(axis=0) > 1e-10).sum()),
        feature_std_max=float(X.std(axis=0).max()),
        candidate_edges=len(groups), unique_candidate_sources=int(len(np.unique(candidate_pre))),
        unique_candidate_targets=int(len(np.unique(candidate_post))), groups={},
        interpretation='RMS and temporal variation include common startup transients. Edge counts repeat shared neurons; these values alone do not establish word information or memory.')
    for g in sorted(set(groups.tolist())):
        ix = groups == g
        observability['groups'][str(g)] = dict(edges=int(ix.sum()),
            unique_source_nodes=int(len(np.unique(candidate_pre[ix]))),
            unique_target_nodes=int(len(np.unique(candidate_post[ix]))),
            pre_rms_max=float(pre_rms[ix].max()), post_rms_max=float(post_rms[ix].max()),
            pre_varying=int((pre[:, ix].std(axis=0) > 1e-10).sum()),
            post_varying=int((post[:, ix].std(axis=0) > 1e-10).sum()))
    emit('observability', **observability)
    # Reuse the actual first-story calibration states; run just one additional
    # constant-input control, at most 16 positions, using this same full graph.
    # Both streams have writing/reading disabled so this isolates input response.
    control_started = time.perf_counter()
    control_length = len(first_activity)
    if control_length == 0:
        raise ValueError('First training story has no causal observation rows')
    constant_features, constant_pre, constant_post, constant_activity = [], [], [], []
    with torch.no_grad():
        control_state = model.initial_state(1)
        bos = torch.tensor([2], device=device)
        for _ in range(control_length):
            value, control_state = model.step(bos, bos, control_state, plasticity_override=False)
            constant_features.append(value.cpu().numpy()[0])
            constant_pre.append(control_state.h[:, model.candidate_pre].cpu().numpy()[0])
            constant_post.append(control_state.h[:, model.candidate_post].cpu().numpy()[0])
            constant_activity.append(control_state.h.cpu().numpy()[0])
    source_ix = np.unique(candidate_pre, return_index=True)[1]
    target_ix = np.unique(candidate_post, return_index=True)[1]
    constant_pre, constant_post = np.asarray(constant_pre), np.asarray(constant_post)
    actual_activity = np.asarray(first_activity, np.float32)
    activity_scale = max(float(np.sqrt(np.mean(actual_activity.astype(np.float64) ** 2))), 1e-9)
    word_control = dict(story_id=train[0]['id'], positions=control_length,
        lexical_positions_after_shared_BOS=max(0, control_length - 1),
        constant_input='BOS/BOS at every position; same zero initial state and length',
        actual_input='Actual ordered sliding word pairs from the first training story',
        plasticity_enabled=False, actual_stream_reused_from_calibration=True,
        unique_candidate_sources=len(source_ix), unique_candidate_targets=len(target_ix),
        source_activity=stream_difference(pre[:control_length, source_ix], constant_pre[:, source_ix], pre_scale[source_ix]),
        target_activity=stream_difference(post[:control_length, target_ix], constant_post[:, target_ix], post_scale[target_ix]),
        features=stream_difference(X[:control_length], constant_features, feature_std),
        all_neuron_activity=stream_difference(actual_activity, constant_activity, activity_scale),
        all_neuron_activity_rms_scale=activity_scale,
        shared_BOS_max_normalized_feature_difference=float(np.max(np.abs(
            (X[0].astype(np.float64) - np.asarray(constant_features[0], np.float64)) / feature_std))),
        scaling='Candidate activities use fixed calibration RMS scales; features use fixed calibration standard deviations; all-neuron activity uses actual-prefix global RMS with 1e-9 floor.',
        interpretation='Input-sensitive response diagnostic only. Actual-versus-constant differences do not demonstrate token decodability, sequence sensitivity, learned plasticity, or 128-word memory. The common first BOS position is an implementation consistency reference, not a measured repeat-run noise distribution.',
        seconds_excluded_from_optimization_benchmark=time.perf_counter() - control_started)
    emit('word_input_control', **word_control)
    del first_activity, actual_activity, constant_activity
    # Compare against an independent frozen replay of the same actual word
    # stream. A one-ULP backend variation is not evidence of a fast-weight effect.
    mean = torch.as_tensor(feature_mean, device=device)
    std = torch.as_tensor(feature_std, device=device)
    fast_effect = measure_fast_effect(model, train[0], args.benchmark_words, std)
    delta, fast_max = fast_effect['max_normalized_feature_difference'], fast_effect['max_fast_state']
    emit('fast_effect', **fast_effect)
    # One actual full-backward optimization step, on training stories only.
    rows = [rows_for(s, args.benchmark_words) for s in train[:args.benchmark_batch]]
    length = min(map(len, rows))
    batch = len(rows)
    head = nn.Linear(256, 2 * len(data['vocabulary'])).to(device)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=.001)
    old = {n: v.detach().clone() for n, v in model.named_parameters()}
    opt.zero_grad(set_to_none=True)
    state = model.initial_state(batch)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    memory_samples = {'before_forward': memory_sample(device)}
    clock = time.perf_counter()
    losses, counts = [], 0
    for t in range(length):
        prev = torch.tensor([r[t]['previous_id'] for r in rows], device=device)
        curr = torch.tensor([r[t]['current_id'] for r in rows], device=device)
        target = torch.tensor([r[t]['targets'] for r in rows], device=device)
        mask = torch.tensor([r[t]['target_mask'] for r in rows], device=device)
        x, state = model.step(prev, curr, state)
        logits = head((x - mean) / std).reshape(batch, 2, -1)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               target.reshape(-1), reduction='none').reshape(batch, 2)
        losses.append((loss * mask).sum())
        counts += int(mask.sum().item())
    loss = torch.stack(losses).sum() / counts
    sync(device)
    forward_seconds = time.perf_counter() - clock
    memory_samples['after_forward'] = memory_sample(device)
    backward_clock = time.perf_counter()
    loss.backward()
    sync(device)
    backward_seconds = time.perf_counter() - backward_clock
    memory_samples['after_backward'] = memory_sample(device)
    gradients = {n: None if v.grad is None else float(v.grad.detach().norm().item())
                 for n, v in model.named_parameters()}
    torch.nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1.)
    opt.step()
    sync(device)
    total_seconds = time.perf_counter() - clock
    memory_samples['after_optimizer'] = memory_sample(device)
    changes = {n: float((v.detach() - old[n]).abs().max().item())
               for n, v in model.named_parameters()}
    after = model.verify_frozen()
    checks = dict(frozen_graph_unchanged=before == after,
                  finite_loss=bool(torch.isfinite(loss).item()),
                  nonzero_fast_state=fast_max > 0,
                  fast_state_affects_features=fast_effect['numerical_gate']['passed'],
                  finite_rule_gradients=all(v is not None and np.isfinite(v) for v in gradients.values()),
                  nonzero_rule_gradient=any(v is not None and v > 0 for v in gradients.values()),
                  rule_parameters_changed=any(v > 0 for v in changes.values()))
    metal_execution = None
    if device.type == 'mps':
        from .metal_sparse import backend_metadata
        metal_execution = dict(**backend_metadata(),
            incoming_kernel_launches=model.incoming.kernel_launch_count,
            transpose_backward_kernel_launches=model.outgoing.kernel_launch_count)
        checks.update(mps_incoming_kernel_executed=model.incoming.kernel_launch_count > 0,
                      mps_transpose_backward_kernel_executed=model.outgoing.kernel_launch_count > 0,
                      mps_training_tensors_resident=all(value.device.type == 'mps' for value in
                          [*model.parameters(), *head.parameters(), *model.buffers(), state.h, state.fast, logits, loss]))
    report = dict(runtime=runtime, config=config, provenance=provenance,
        calibration_npz_sha256=digest(calibration_path),
        observability=observability, word_input_control=word_control, fast_effect=fast_effect, checks=checks,
        checks_interpretation='Hardware timing, finite backward execution and optimizer mechanics are reported even if the fast-effect gate fails. The fast feature effect must strictly exceed both ten times a same-stream frozen-repeat difference and 1e-5 normalized units; passing does not establish useful word information or memory.',
        rule_gradients=gradients,
        metal_execution=metal_execution,
        rule_parameter_changes=changes, learned_rule_parameters=sum(v.numel() for v in model.parameters()),
        decoder_parameters=sum(v.numel() for v in head.parameters()),
        benchmark=dict(batch=batch, observed_positions=length, total_positions=batch * length,
            internal_updates=length * args.internal_steps, scored_forecasts=counts,
            forward_seconds=forward_seconds, backward_seconds=backward_seconds,
            step_seconds=total_seconds, positions_per_second=batch * length / total_seconds,
            loss=float(loss.item()), max_normalized_feature_difference=delta,
            max_normalized_frozen_repeat_difference=fast_effect['max_normalized_frozen_repeat_difference'],
            max_fast_state=fast_max,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
            memory_samples=memory_samples,
            sampled_max_mps_allocated_bytes=max(s['mps_current_allocated_bytes'] for s in memory_samples.values()) if device.type == 'mps' else None,
            peak_process_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        elapsed_seconds=time.perf_counter() - started)
    write_json(args.output / 'probe.json', report)
    emit('global_step', step=1, loss=float(loss.item()), seconds=total_seconds, checks=checks)
    calibration.update(probe_passed=bool(all(checks.values())), probe_sha256=digest(args.output / 'probe.json'))
    write_json(args.output / 'calibration.json', calibration)
    if not all(checks.values()):
        raise RuntimeError('Fast-plasticity wiring probe failed; inspect probe.json before training')
    emit('probe_complete', report=str(args.output / 'probe.json'))


if __name__ == '__main__':
    main()
