#!/usr/bin/env python3
"""Fetch a verified rank-128 decoder campaign snapshot, separately from A-D."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('E32rank128fixed', 'F32rank128bounded')
DEFAULT_REMOTE_RUN = '/Users/fernando/fly_wordbrain_connectorch/results/connectorch-decoder-v1'
ARTIFACT_NAMES = ('manifest.json', 'launch.json', 'status.json', 'process-status.json', 'metrics.jsonl',
                  'selected-checkpoints.json', 'results.json', 'failure.json', 'accepted-early-stop.json', 'stop-receipt.json')
FILES = ['manifest.json', 'launch.json', 'campaign-status.json', 'readiness.json', 'results.json',
         'partial-results.json', 'failure.json', 'smoke-results.json', 'preflight/parity.json', 'preflight/quality.json']
FILES += [f'{phase}/{arm}/{name}' for phase in ('arms', 'smokes') for arm in ARMS for name in ARTIFACT_NAMES]
BASELINE_NAMES = ('manifest.json', 'launch.json', 'status.json', 'metrics.jsonl', 'selected-checkpoints.json',
                  'accepted-early-stop.json', 'stop-receipt.json')
SNAPSHOT_FILES = set(FILES) | {'baseline/' + name for name in BASELINE_NAMES} | {'baseline/quality.json'}
EXPECTED_COUNTS = {
    'B32fixed': {'encoder': 482816, 'readout': 50578432, 'neurons': 148179, 'layernorm': 98786, 'edge_gains': 0, 'total': 51308213},
    'E32rank128fixed': {'encoder': 482816, 'readout': 6453376, 'neurons': 148179, 'layernorm': 98786, 'edge_gains': 0, 'total': 7183157},
    'F32rank128bounded': {'encoder': 482816, 'readout': 6453376, 'neurons': 148179, 'layernorm': 98786, 'edge_gains': 18322, 'total': 7201479},
}
SELECTORS = ('minimum_validation_ce', 'maximum_validation_accuracy')

REMOTE = '''from pathlib import Path
import hashlib,json,sys
request=json.loads(sys.argv[1]);root=Path(request['root']).resolve();files={};sources={}
def fetch(path,name,expected=None):
 path=Path(path)
 if not path.exists():
  if expected:raise ValueError('Bound artifact is missing: '+str(path))
  return
 if path.is_symlink():raise ValueError('Unexpected artifact symlink: '+str(path))
 with path.open('rb') as handle:body=handle.read(16777217)
 if len(body)>16777216:raise ValueError('Artifact exceeds 16 MiB bound: '+str(path))
 digest=hashlib.sha256(body).hexdigest()
 if expected and digest!=expected:raise ValueError('Bound artifact changed: '+str(path))
 files[name]={'text':body.decode(),'sha256':digest};sources[name]=str(path)
for name in request['files']:fetch(root/name,name)
manifest=json.loads(files.get('manifest.json',{}).get('text','{}'))
if manifest:
 entry=manifest['baseline_acceptance'];baseline=Path(manifest['baseline_run']).resolve()
 acceptance=Path(entry['path']).resolve()
 if baseline.name!='B32fixed' or acceptance!=baseline/'accepted-early-stop.json':
  raise ValueError('Unexpected decoder baseline acceptance path')
 fetch(acceptance,'baseline/accepted-early-stop.json',entry['sha256'])
 receipt=json.loads(files['baseline/accepted-early-stop.json']['text'])
 if receipt.get('arm')!='B32fixed' or receipt.get('status')!='accepted_early_stop' or receipt.get('process_cessation',{}).get('confirmed') is not True or receipt.get('accepted_by') not in ('user','agent_under_standing_user_authorization'):
  raise ValueError('Decoder baseline is not a confirmed B accepted stop')
 bindings={**manifest.get('input_files_sha256',{}),**manifest.get('baseline_files_sha256',{}),**receipt.get('artifact_sha256',{})}
 for name in request['baseline_names']:
  if name=='accepted-early-stop.json':continue
  path=baseline/name
  fetch(path,'baseline/'+name,bindings.get(str(path)))
 stop=receipt['stop_receipt']
 if Path(stop['path']).resolve()!=baseline/'stop-receipt.json':raise ValueError('Unexpected baseline stop receipt path')
 fetch(Path(stop['path']),'baseline/stop-receipt.json',stop['sha256'])
 quality=manifest.get('baseline_quality')
 if quality:fetch(Path(quality['path']),'baseline/quality.json',quality['sha256'])
for arm in request['arms']:
 prefix='arms/'+arm+'/'
 if prefix+'accepted-early-stop.json' not in files:continue
 receipt=json.loads(files[prefix+'accepted-early-stop.json']['text'])
 if receipt.get('status')!='accepted_early_stop':continue
 if receipt.get('arm')!=arm or receipt.get('process_cessation',{}).get('confirmed') is not True or receipt.get('accepted_by') not in ('user','agent_under_standing_user_authorization'):
  raise ValueError('Invalid accepted arm stop: '+arm)
 stop=receipt['stop_receipt']
 if Path(stop['path']).resolve()!=root/prefix/'stop-receipt.json':raise ValueError('Unexpected arm stop receipt path')
 fetch(Path(stop['path']),prefix+'stop-receipt.json',stop['sha256'])
print(json.dumps({'remote_run':str(root),'files':files,'source_paths':sources}))
'''


def read(files, name):
    return json.loads(files[name]['text']) if name in files else {}


def validation_rows(files, prefix):
    rows = []
    metrics_text = files.get(prefix + 'metrics.jsonl', {}).get('text', '')
    # JSON Lines uses physical LF; Unicode separators may be string data.
    lines = metrics_text.split('\n')
    for i, line in enumerate(lines):
        if i == len(lines)-1 and not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines)-1 and not metrics_text.endswith('\n'):
                continue
            raise
        if row.get('event') == 'validation':
            if row.get('validation', {}).get('tokens') != 21874 or row['validation'].get('stories') != 100:
                raise ValueError('Unexpected full-validation population: ' + prefix)
            rows.append({key: row[key] for key in ('updates', 'epoch', 'validation', 'elapsed_seconds_this_invocation') if key in row})
    return rows


def arm_summary(files, arm, mapping, prefix=None):
    prefix = prefix or 'arms/' + arm + '/'
    manifest, launch = read(files, prefix+'manifest.json'), read(files, prefix+'launch.json')
    config, raw_counts = manifest.get('config', {}), manifest.get('parameter_counts')
    rank = 0 if arm == 'B32fixed' else 128
    expected_config = {'d_embed':32, 'readout_rank':rank, 'history_length':8,
                       'plasticity':'bounded10' if arm == 'F32rank128bounded' else 'fixed'}
    if config and any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError('Unexpected architecture in decoder comparison: ' + arm)
    counts = EXPECTED_COUNTS[arm].copy()
    if raw_counts:
        if sum(value for key, value in raw_counts.items() if key != 'total') != raw_counts.get('total'):
            raise ValueError('Unexpected parameter counts in decoder comparison: ' + arm)
        actual = {key: raw_counts.get(key) for key in counts if key != 'encoder'}
        actual['encoder'] = raw_counts['embedding'] + raw_counts['input_projection']
        if actual != counts:
            raise ValueError('Unexpected parameter counts in decoder comparison: ' + arm)
        counts = actual
    status = read(files, prefix+'status.json') or read(files, prefix+'process-status.json')
    rows = validation_rows(files, prefix)
    selectors = read(files, prefix+'selected-checkpoints.json').get('selectors', {})
    acceptance = read(files, prefix+'accepted-early-stop.json')
    accepted = acceptance.get('status') == 'accepted_early_stop'
    if accepted:
        if (acceptance.get('arm') != arm or acceptance.get('process_cessation', {}).get('confirmed') is not True
                or acceptance.get('accepted_by') not in ('user', 'agent_under_standing_user_authorization')):
            raise ValueError('Unconfirmed accepted stop: ' + arm)
        selected = acceptance.get('selected_checkpoints', acceptance.get('preserved_checkpoints', {}))
        selectors = selected.get('selectors', selected)
        selectors = {k:v for k,v in selectors.items() if k in SELECTORS}
    latest = acceptance.get('latest_checkpoint', acceptance.get('preserved_checkpoints', {}).get('latest', {}))
    updates = (acceptance.get('durable_updates', latest.get('cursor', {}).get('updates')) if accepted else None)
    if updates is None:
        updates = status.get('updates', rows[-1]['updates'] if rows else 0)
    state = 'accepted early stop' if accepted else status.get('status', 'planned')
    results, failure = read(files, prefix+'results.json'), read(files, prefix+'failure.json')
    if not accepted and (failure or results.get('status') == 'failed'):
        state = 'failed'
    elif not accepted and results.get('status') == 'completed':
        state = 'completed'
    git = launch.get('experiment_git') or {'repository':mapping.get('repository'), **mapping.get('arms', {}).get(arm, {})}
    return {'name':arm, 'status':state, 'activity':status.get('activity'), 'updates':updates,
            'observed_updates':acceptance.get('observed_updates', status.get('updates')), 'readout_rank':rank,
            'parameter_counts':counts, 'parameter_count_source':'verified arm manifest' if raw_counts else 'planned architecture',
            'selectors':selectors, 'validation_history':rows, 'experiment_git':git,
            'trainer_sources_sha256':manifest.get('trainer_sources_sha256', {}),
            'launch_sources_sha256':launch.get('sources_sha256', {}),
            'frozen_buffers_sha256':manifest.get('frozen_buffers_sha256', {}),
            'accepted_stop':acceptance if accepted else None, 'failure':failure or None}


def summarize_snapshot(files, host, now):
    manifest = read(files, 'manifest.json')
    status = read(files, 'campaign-status.json') or read(files, 'readiness.json')
    mapping = manifest.get('experiment_branches') or {}
    baseline = arm_summary(files, 'B32fixed', mapping, 'baseline/')
    arms = {arm:arm_summary(files, arm, mapping) for arm in ARMS}
    return {'format_version':1, 'snapshot_at_utc':now, 'host':host, 'status':status,
            'baseline':baseline, 'arms':arms, 'campaign_sources_sha256':manifest.get('sources_sha256', {}),
            'baseline_acceptance':manifest.get('baseline_acceptance'), 'baseline_quality':manifest.get('baseline_quality'),
            'preflight':read(files, 'preflight/parity.json'), 'smokes':read(files, 'smoke-results.json'),
            'failure':read(files, 'failure.json'), 'results':read(files, 'results.json'),
            'validation_split':{'stories':100, 'targets':21874, 'unit':'next-BPE-token'},
            'checkpoint_verification':'Hashes are retained checkpoint receipts; binaries are not fetched or reloaded.',
            'source_verification':'Source hashes are copied from bound manifests/launch receipts; this refresh does not rerun parity.'}


def selected_cell(arm, selector):
    value = arm['selectors'].get(selector)
    maximize = selector == 'maximum_validation_accuracy'
    if value:
        metric, update, suffix = value['validation'], value['cursor']['updates'], ''
    elif arm['validation_history']:
        key = 'top1_accuracy' if maximize else 'cross_entropy'
        row = (max if maximize else min)(arm['validation_history'], key=lambda row:row['validation'][key])
        metric, update, suffix = row['validation'], row['updates'], '; retention pending'
    else:
        return '—'
    ce, acc = metric['cross_entropy'], 100*metric['top1_accuracy']
    return (f'{acc:.2f}% ({ce:.4f}; {update:,}{suffix})' if maximize
            else f'{ce:.4f} ({acc:.2f}%; {update:,}{suffix})')


def render_report(summary):
    status = summary['status']
    baseline_description = ('B32fixed is the accepted full-head baseline.' if summary['baseline']['accepted_stop']
                            else 'B32fixed is the planned full-head comparison baseline; its acceptance receipt is pending.')
    report = ['# Rank-128 decoder experiment — partial validation', '',
              f"Snapshot: {summary['snapshot_at_utc']}. Host: {summary['host']}.", '',
              f"Campaign status: **{status.get('status', 'pending')}**. Phase: **{status.get('phase', 'pending')}**. "
              f"Current arm: **{status.get('arm') or '—'}**.", '',
              baseline_description + ' E/F retain its 482,816-parameter encoder and eight input delays, '
              'and reduce the readout from 50,578,432 to 6,453,376 parameters using rank 128.',
              'E retains fixed base edges; F adds 18,322 source/destination type gains with base-edge multipliers 0.9–1.1. '
              'Both retain 49,393 neurons, 9,050,172 base edges and the original unconstrained neuron gains.', '',
              '| Arm | Readout rank | Encoder | Readout | Edge gains | Total trainable | Count source | Status | Updates | Minimum CE (accuracy; update) | Maximum accuracy (CE; update) |',
              '|---|---:|---:|---:|---:|---:|---|---|---:|---|---|']
    for arm in (summary['baseline'], *summary['arms'].values()):
        c = arm['parameter_counts']
        report.append(f"| {arm['name']} | {arm['readout_rank'] or 'full'} | {c['encoder']:,} | {c['readout']:,} | "
                      f"{c['edge_gains']:,} | {c['total']:,} | {arm['parameter_count_source']} | {arm['status']} | "
                      f"{arm['updates']:,} | {selected_cell(arm, SELECTORS[0])} | {selected_cell(arm, SELECTORS[1])} |")
    if not summary['baseline']['accepted_stop']:
        report.extend(['', 'Baseline acceptance receipt has not yet been fetched; baseline results are pending.'])
    report.extend(['', 'B stopped early. Compare E/F with B at common recorded updates, and report retained-best results separately for unequal budgets. '
                   'Partial scores are next-BPE-token validation on 100 stories / 21,874 targets. The reserved test remains untouched.',
                   'The neuron parameters number 148,179 and output LayerNorm adds 98,786 in every arm. '
                   'Smoke scores are excluded from the full-training table.'])
    if summary['smokes']:
        report.extend(['', f"Smoke receipt passed: **{summary['smokes'].get('passed', 'unavailable')}**."])
    if summary['failure']:
        report.extend(['', 'Campaign failure: `' + json.dumps(summary['failure']) + '`.'])
    for arm in (summary['baseline'], *summary['arms'].values()):
        report.extend(['', '## ' + arm['name'], ''])
        git = arm['experiment_git']
        if git.get('branch'):
            report.extend([f"Branch: `{git['branch']}`; commit `{git.get('commit', 'unavailable')}`; "
                           f"repository {git.get('repository') or 'unavailable'}.", ''])
        for selector, value in arm['selectors'].items():
            report.extend([f"Saved {selector}: `{value['path']}`; SHA256 `{value['sha256']}`.", ''])
        if arm['trainer_sources_sha256']:
            report.extend(['Trainer-source and frozen-graph hashes are preserved in the structured `progress.json` snapshot.', ''])
        if not arm['validation_history']:
            report.append('No full-arm validation fetched yet.')
            continue
        report.extend(['| Updates | Epoch | Validation CE | Accuracy |', '|---:|---:|---:|---:|'])
        for row in arm['validation_history']:
            metric = row['validation']
            report.append(f"| {row['updates']:,} | {row['epoch']} | {metric['cross_entropy']:.4f} | {100*metric['top1_accuracy']:.2f}% |")
    report.extend(['', summary['checkpoint_verification'], summary['source_verification'],
                   'Only files present in this verified fetch are rendered; stale local artifacts are not merged.', ''])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='macm3')
    parser.add_argument('--remote-run', default=DEFAULT_REMOTE_RUN)
    parser.add_argument('--output', type=Path, default=ROOT/'results/connectorch-decoder-v1')
    args = parser.parse_args()
    import cluster_runner as cr
    request = {'root':args.remote_run, 'files':FILES, 'baseline_names':BASELINE_NAMES, 'arms':ARMS}
    command = 'python3 -c ' + shlex.quote(REMOTE) + ' ' + shlex.quote(json.dumps(request))
    fetched = cr.run(command, host=args.host, timeout=30)
    if fetched.exit_code:
        raise RuntimeError(fetched.stderr or fetched.stdout)
    snapshot = json.loads(fetched.stdout)
    if not snapshot['files']:
        raise RuntimeError('No decoder campaign artifacts found')
    hashes = {}
    for name, value in snapshot['files'].items():
        if name not in SNAPSHOT_FILES:
            raise ValueError('Unexpected artifact path')
        digest = hashlib.sha256(value['text'].encode()).hexdigest()
        if digest != value['sha256']:
            raise ValueError('Artifact transfer checksum mismatch')
        hashes[name] = digest
    now = datetime.now(timezone.utc).isoformat()
    summary = summarize_snapshot(snapshot['files'], args.host, now)
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in snapshot['files'].items():
        path = args.output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(value['text'])
        temp.replace(path)
    (args.output/'progress.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output/'progress.md').write_text('\n'.join(report))
    (args.output/'snapshot-receipt.json').write_text(json.dumps({'fetched_at_utc':now, 'host':args.host,
            'remote_run':snapshot['remote_run'], 'source_paths':snapshot['source_paths'], 'sha256':hashes}, indent=2)+'\n')
    print('\n'.join(report[:16]))


if __name__ == '__main__':
    main()
