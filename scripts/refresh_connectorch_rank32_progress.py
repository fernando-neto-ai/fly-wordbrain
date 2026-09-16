#!/usr/bin/env python3
"""Fetch rank-32 H progress beside the preserved rank-64 G baseline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'G32rank64fixed'
ARM = 'H32rank32fixed'
DEFAULT_REMOTE_RUN = '/Users/fernando/fly_wordbrain_connectorch/results/connectorch-rank32-v1'
NAMES = ('manifest.json', 'launch.json', 'status.json', 'process-status.json', 'metrics.jsonl',
         'selected-checkpoints.json', 'results.json', 'failure.json', 'accepted-early-stop.json',
         'stop-receipt.json', 'early-stop-decision.json')
FILES = ['manifest.json', 'launch.json', 'campaign-status.json', 'readiness.json', 'partial-results.json',
         'results.json', 'failure.json', 'smoke-results.json', 'preflight/parity.json']
FILES += [f'{phase}/{ARM}/{name}' for phase in ('arms', 'smokes') for name in NAMES]
BASELINE_NAMES = ('manifest.json', 'launch.json', 'status.json', 'process-status.json', 'metrics.jsonl',
                  'selected-checkpoints.json', 'accepted-early-stop.json', 'stop-receipt.json', 'early-stop-decision.json')
SNAPSHOT_FILES = set(FILES) | {'baseline/'+name for name in BASELINE_NAMES} | {'parent/manifest.json'}
COUNTS = {
    BASELINE: {'encoder':482816, 'readout':3226688, 'neurons':148179, 'layernorm':98786, 'edge_gains':0, 'total':3956469},
    ARM: {'encoder':482816, 'readout':1613344, 'neurons':148179, 'layernorm':98786, 'edge_gains':0, 'total':2343125},
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
 if path.is_symlink():raise ValueError('Unexpected artifact symlink')
 with path.open('rb') as handle:body=handle.read(16777217)
 if len(body)>16777216:raise ValueError('Artifact exceeds 16 MiB bound')
 digest=hashlib.sha256(body).hexdigest()
 if expected and digest!=expected:raise ValueError('Bound artifact changed: '+str(path))
 files[name]={'text':body.decode(),'sha256':digest};sources[name]=str(path)
for name in request['files']:fetch(root/name,name)
manifest=json.loads(files.get('manifest.json',{}).get('text','{}'))
if manifest:
 baseline=Path(manifest['baseline_run']).resolve();entry=manifest['baseline_acceptance']
 if baseline.name!=request['baseline'] or Path(entry['path']).resolve()!=baseline/'accepted-early-stop.json':
  raise ValueError('Expected the preserved G baseline')
 parent=manifest.get('parent_campaign')
 if parent:
  parent_manifest=Path(parent['path']).resolve()
  if parent_manifest.name!='manifest.json' or baseline!=parent_manifest.parent/'arms'/request['baseline']:
   raise ValueError('Rank32 parent does not contain the G baseline')
  fetch(parent_manifest,'parent/manifest.json',parent['sha256'])
  if json.loads(files['parent/manifest.json']['text']).get('launch_id')!=parent.get('launch_id'):
   raise ValueError('Rank32 parent launch identity differs')
 fetch(Path(entry['path']),'baseline/accepted-early-stop.json',entry['sha256'])
 receipt=json.loads(files['baseline/accepted-early-stop.json']['text'])
 if receipt.get('arm')!=request['baseline'] or receipt.get('status')!='accepted_early_stop' or receipt.get('process_cessation',{}).get('confirmed') is not True or receipt.get('accepted_by') not in ('user','agent_under_standing_user_authorization'):
  raise ValueError('G baseline is not a confirmed accepted stop')
 bindings={**manifest.get('input_files_sha256',{}),**manifest.get('baseline_files_sha256',{}),**receipt.get('artifact_sha256',{})}
 for name in request['baseline_names']:
  if name!='accepted-early-stop.json':fetch(baseline/name,'baseline/'+name,bindings.get(str(baseline/name)))
 stop=receipt['stop_receipt']
 if Path(stop['path']).resolve()!=baseline/'stop-receipt.json':raise ValueError('Unexpected G stop receipt path')
 fetch(Path(stop['path']),'baseline/stop-receipt.json',stop['sha256'])
 decision=receipt.get('decision_receipt')
 if decision:
  if Path(decision['path']).resolve()!=baseline/'early-stop-decision.json':raise ValueError('Unexpected G decision receipt path')
  fetch(Path(decision['path']),'baseline/early-stop-decision.json',decision['sha256'])
 prefix='arms/'+request['arm']+'/'
 if prefix+'accepted-early-stop.json' in files:
  receipt=json.loads(files[prefix+'accepted-early-stop.json']['text'])
  if receipt.get('status')=='accepted_early_stop':
   if receipt.get('arm')!=request['arm'] or receipt.get('process_cessation',{}).get('confirmed') is not True:
    raise ValueError('Unconfirmed H stop receipt')
   stop=receipt['stop_receipt']
   if Path(stop['path']).resolve()!=root/prefix/'stop-receipt.json':raise ValueError('Unexpected H stop receipt path')
   fetch(Path(stop['path']),prefix+'stop-receipt.json',stop['sha256'])
   for path,digest in receipt.get('artifact_sha256',{}).items():
    path=Path(path)
    if path.parent.resolve()!=root/prefix:raise ValueError('Accepted H artifact outside its arm')
    name=prefix+path.name
    if name not in request['files']:raise ValueError('Unexpected accepted H artifact')
    fetch(path,name,digest)
   decision=receipt.get('decision_receipt')
   if decision:
    if Path(decision['path']).resolve()!=root/prefix/'early-stop-decision.json':raise ValueError('Unexpected H decision receipt path')
    fetch(Path(decision['path']),prefix+'early-stop-decision.json',decision['sha256'])
print(json.dumps({'remote_run':str(root),'files':files,'source_paths':sources}))
'''


def read(files, name):
    return json.loads(files[name]['text']) if name in files else {}


def validation_rows(files, prefix):
    text = files.get(prefix+'metrics.jsonl', {}).get('text', '')
    lines, rows = text.split('\n'), []
    for index, line in enumerate(lines):
        if index == len(lines)-1 and not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines)-1 and not text.endswith('\n'):
                continue
            raise
        if row.get('event') == 'validation':
            if row['validation'].get('stories') != 100 or row['validation'].get('tokens') != 21874:
                raise ValueError('Unexpected full-validation population: '+prefix)
            rows.append({key:row[key] for key in ('updates','epoch','validation','elapsed_seconds_this_invocation') if key in row})
    return rows


def arm_summary(files, name, prefix, branch_mapping):
    manifest, launch = read(files,prefix+'manifest.json'), read(files,prefix+'launch.json')
    rank = 64 if name == BASELINE else 32
    config, raw = manifest.get('config',{}), manifest.get('parameter_counts')
    expected_config = {'d_embed':32, 'history_length':8, 'readout_rank':rank, 'plasticity':'fixed'}
    if config and any(config.get(key) != value for key,value in expected_config.items()):
        raise ValueError('Unexpected G/H architecture: '+name)
    counts = COUNTS[name].copy()
    if raw:
        actual = {key:raw.get(key) for key in counts if key != 'encoder'}
        actual['encoder'] = raw['embedding']+raw['input_projection']
        if actual != counts or sum(value for key,value in raw.items() if key != 'total') != raw.get('total'):
            raise ValueError('Unexpected G/H parameter counts: '+name)
    rows = validation_rows(files,prefix)
    process = read(files,prefix+'process-status.json')
    status = read(files,prefix+'status.json') or process
    accepted = read(files,prefix+'accepted-early-stop.json')
    selectors = read(files,prefix+'selected-checkpoints.json').get('selectors',{})
    stopped = accepted.get('status') == 'accepted_early_stop'
    if stopped:
        if (accepted.get('arm') != name or accepted.get('process_cessation',{}).get('confirmed') is not True
                or accepted.get('accepted_by') not in ('user','agent_under_standing_user_authorization')):
            raise ValueError('Invalid accepted stop: '+name)
        preserved = accepted.get('selected_checkpoints',accepted.get('preserved_checkpoints',{}))
        selectors = {key:value for key,value in preserved.get('selectors',preserved).items() if key in SELECTORS}
    latest = accepted.get('latest_checkpoint',accepted.get('preserved_checkpoints',{}).get('latest',{}))
    updates = accepted.get('durable_updates',latest.get('cursor',{}).get('updates')) if stopped else None
    if updates is None:
        updates = status.get('updates',rows[-1]['updates'] if rows else 0)
    state = 'accepted early stop' if stopped else status.get('status','planned')
    failure, result = read(files,prefix+'failure.json'), read(files,prefix+'results.json')
    if not stopped and (failure or result.get('status') == 'failed' or process.get('status') == 'failed'):
        state = 'failed'
    elif not stopped and result.get('status') == 'completed':
        state = 'completed'
    git = launch.get('experiment_git') or {'repository':branch_mapping.get('repository'), **branch_mapping.get('arms',{}).get(name,{})}
    return {'name':name, 'status':state, 'activity':status.get('activity'), 'updates':updates, 'readout_rank':rank,
            'observed_updates':accepted.get('observed_updates',status.get('updates')), 'parameter_counts':counts,
            'parameter_count_source':'verified arm manifest' if raw else 'planned architecture',
            'selectors':selectors, 'validation_history':rows, 'accepted_stop':accepted if stopped else None,
            'campaign_launch_id':launch.get('campaign_launch_id'),
            'trainer_pid':process.get('worker_pid',status.get('pid')), 'runner_pid':process.get('runner_pid'),
            'native_process_status':process,
            'experiment_git':git, 'trainer_sources_sha256':manifest.get('trainer_sources_sha256',{}),
            'launch_sources_sha256':launch.get('sources_sha256',{}), 'frozen_buffers_sha256':manifest.get('frozen_buffers_sha256',{}),
            'failure':failure or None}


def summarize_snapshot(files, host, now):
    manifest = read(files,'manifest.json')
    mapping = manifest.get('experiment_branches') or {}
    active = arm_summary(files,ARM,'arms/'+ARM+'/',mapping)
    native_status = read(files,'campaign-status.json') or read(files,'readiness.json')
    status = dict(native_status)
    if active['accepted_stop']:
        status.update(status='accepted early stop', phase='stopped', arm=ARM)
    return {'format_version':1, 'snapshot_at_utc':now, 'host':host,
            'status':status, 'native_campaign_status':native_status,
            'launch_id':manifest.get('launch_id'), 'campaign_launch':read(files,'launch.json'),
            'baseline':arm_summary(files,BASELINE,'baseline/',mapping),
            'arms':{ARM:active},
            'baseline_acceptance':manifest.get('baseline_acceptance'), 'parent_campaign':manifest.get('parent_campaign'),
            'campaign_sources_sha256':manifest.get('sources_sha256',{}),
            'preflight':read(files,'preflight/parity.json'), 'smokes':read(files,'smoke-results.json'),
            'results':read(files,'results.json'), 'failure':read(files,'failure.json'),
            'validation_split':{'stories':100,'targets':21874,'unit':'next-BPE-token'},
            'verification_scope':'Artifact transfer and bound baseline/stop hashes verified. Source and checkpoint hashes are copied from receipts; checkpoint binaries are not fetched or reloaded.'}


def selected_cell(arm, selector):
    selected = arm['selectors'].get(selector)
    maximize = selector == 'maximum_validation_accuracy'
    if selected:
        metric, update, suffix = selected['validation'],selected['cursor']['updates'],''
    elif arm['validation_history']:
        key = 'top1_accuracy' if maximize else 'cross_entropy'
        row = (max if maximize else min)(arm['validation_history'],key=lambda row:row['validation'][key])
        metric, update, suffix = row['validation'],row['updates'],'; retention pending'
    else:
        return '—'
    ce, accuracy = metric['cross_entropy'],100*metric['top1_accuracy']
    return (f'{accuracy:.2f}% ({ce:.4f}; {update:,}{suffix})' if maximize
            else f'{ce:.4f} ({accuracy:.2f}%; {update:,}{suffix})')


def render_report(summary):
    state = summary['status']
    report = ['# Rank-32 decoder experiment — partial validation','',
              f"Snapshot: {summary['snapshot_at_utc']}. Host: {summary['host']}.",'',
              f"Campaign status: **{state.get('status','pending')}**. Phase: **{state.get('phase','pending')}**. Current arm: **{state.get('arm') or '—'}**.",'',
              'H compares a rank-32 readout with the preserved rank-64 G baseline. Encoder capacity stays at 482,816 parameters; '
              'the readout falls from 3,226,688 to 1,613,344 parameters. Both use fixed canonical edges, 148,179 trained neuron parameters and 98,786 output LayerNorm parameters.',
              'Both retain 49,393 neurons, 9,050,172 canonical edges and eight explicit input delays. No additional edge gains are trained.','',
              '| Arm | Rank | Encoder | Readout | Total trainable | Count source | Status | Updates | Minimum CE (accuracy; update) | Maximum accuracy (CE; update) |',
              '|---|---:|---:|---:|---:|---|---|---:|---|---|']
    for arm in (summary['baseline'],summary['arms'][ARM]):
        counts=arm['parameter_counts']
        report.append(f"| {arm['name']} | {arm['readout_rank']} | {counts['encoder']:,} | {counts['readout']:,} | {counts['total']:,} | "
                      f"{arm['parameter_count_source']} | {arm['status']} | {arm['updates']:,} | {selected_cell(arm,SELECTORS[0])} | {selected_cell(arm,SELECTORS[1])} |")
    if not summary['baseline']['accepted_stop']:
        report.extend(['','G acceptance has not yet been fetched; baseline results remain pending.'])
    report.extend(['','G stopped early. Compare common recorded updates as well as retained-best results across actual budgets. '
                   'Scores cover 100 validation stories / 21,874 next-BPE targets. Smoke scores are excluded; the reserved test remains untouched.'])
    if summary['smokes']:
        report.extend(['',f"H smoke receipt passed: **{summary['smokes'].get('passed','unavailable')}**."])
    if summary['failure']:
        label = 'Preserved native signal/failure record; accepted stop governs status' if summary['arms'][ARM]['accepted_stop'] else 'Campaign failure'
        report.extend(['',label+': `'+json.dumps(summary['failure'])+'`.'])
    for arm in (summary['baseline'],summary['arms'][ARM]):
        report.extend(['','## '+arm['name'],''])
        git=arm['experiment_git']
        if git.get('branch'):
            report.extend([f"Branch: `{git['branch']}`; commit `{git.get('commit','unavailable')}`; repository {git.get('repository') or 'unavailable'}.",''])
        for selector,entry in arm['selectors'].items():
            report.extend([f"Saved {selector}: `{entry['path']}`; SHA256 `{entry['sha256']}`.",''])
        if not arm['validation_history']:
            report.append('No full-arm validation fetched yet.')
            continue
        report.extend(['| Updates | Epoch | Validation CE | Accuracy |','|---:|---:|---:|---:|'])
        for row in arm['validation_history']:
            metric=row['validation']
            report.append(f"| {row['updates']:,} | {row['epoch']} | {metric['cross_entropy']:.4f} | {100*metric['top1_accuracy']:.2f}% |")
    report.extend(['',summary['verification_scope'],'Source and frozen-graph receipts are included in structured `progress.json`. '
                   'Only this fetched snapshot is rendered; stale local files are not merged.',''])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='macm3')
    parser.add_argument('--remote-run',default=DEFAULT_REMOTE_RUN)
    parser.add_argument('--output',type=Path,default=ROOT/'results/connectorch-rank32-v1')
    args=parser.parse_args()
    import cluster_runner as cr
    request={'root':args.remote_run,'files':FILES,'baseline_names':BASELINE_NAMES,'baseline':BASELINE,'arm':ARM}
    fetched=cr.run('python3 -c '+shlex.quote(REMOTE)+' '+shlex.quote(json.dumps(request)),host=args.host,timeout=30)
    if fetched.exit_code:
        raise RuntimeError(fetched.stderr or fetched.stdout)
    snapshot=json.loads(fetched.stdout)
    if not snapshot['files']:
        raise RuntimeError('No rank32 campaign artifacts found')
    hashes={}
    for name,item in snapshot['files'].items():
        if name not in SNAPSHOT_FILES:
            raise ValueError('Unexpected artifact path')
        digest=hashlib.sha256(item['text'].encode()).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Artifact transfer checksum mismatch')
        hashes[name]=digest
    now=datetime.now(timezone.utc).isoformat()
    summary=summarize_snapshot(snapshot['files'],args.host,now)
    report=render_report(summary)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,item in snapshot['files'].items():
        path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(item['text']);temp.replace(path)
    (args.output/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'progress.md').write_text('\n'.join(report))
    (args.output/'snapshot-receipt.json').write_text(json.dumps({'fetched_at_utc':now,'host':args.host,
        'remote_run':snapshot['remote_run'],'source_paths':snapshot['source_paths'],'sha256':hashes},indent=2)+'\n')
    print('\n'.join(report[:15]))


if __name__ == '__main__':
    main()
