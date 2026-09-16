"""Rank32 progress parser and read-only fetch fixtures; no remote calls or GPU work."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rank32_progress', ROOT/'scripts/refresh_connectorch_rank32_progress.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def item(value):
    return {'text': json.dumps(value)}


def manifest(rank):
    counts = {'embedding': 32768, 'input_projection': 450048, 'neurons': 148179, 'layernorm': 98786,
              'readout': rank*(49393+1024), 'edge_gains': 0, 'other': 0}
    counts['total'] = sum(counts.values())
    return {'config': {'d_embed': 32, 'history_length': 8, 'readout_rank': rank, 'plasticity': 'fixed'},
            'parameter_counts': counts, 'trainer_sources_sha256': {'trainer': 'a'*64}}


def selectors():
    return {name: {'path': '/preserved/'+name+'.pt', 'sha256': 'c'*64, 'cursor': {'updates': update},
                  'validation': {'stories': 100, 'tokens': 21874, 'cross_entropy': ce, 'top1_accuracy': accuracy}}
            for name, update, ce, accuracy in [('minimum_validation_ce', 16600, 3.093705, .36367),
                                              ('maximum_validation_accuracy', 15552, 3.103450, .368885)]}


def acceptance(name=m.BASELINE):
    return {'format_version': 1, 'status': 'accepted_early_stop', 'arm': name,
            'accepted_by': 'agent_under_standing_user_authorization', 'process_cessation': {'confirmed': True},
            'durable_updates': 16800, 'observed_updates': 16808, 'completed_epochs': 14,
            'selected_checkpoints': {'format_version': 1, 'selectors': selectors()}}


def snapshot():
    return {'manifest.json': item({'launch_id': 'h-launch', 'experiment_branches': {'arms': {
                m.ARM: {'branch': 'exp/encoder32-readout32-fixed', 'commit': 'b'*40}}}}),
            'baseline/manifest.json': item(manifest(64)),
            'baseline/accepted-early-stop.json': item(acceptance()),
            'baseline/process-status.json': item({'status': 'failed', 'exit_code': -15, 'worker_pid': 20985})}


def test_planned_h_and_preserved_g_have_correct_counts_and_both_selectors():
    summary = m.summarize_snapshot(snapshot(), 'macm3', 'now')
    report = '\n'.join(m.render_report(summary))
    assert '| G32rank64fixed | 64 | 482,816 | 3,226,688 | 3,956,469 | verified arm manifest | accepted early stop | 16,800 |' in report
    assert '| H32rank32fixed | 32 | 482,816 | 1,613,344 | 2,343,125 | planned architecture | planned | 0 |' in report
    assert '36.89% (3.1035; 15,552)' in report
    assert summary['baseline']['trainer_pid'] == 20985 and summary['launch_id'] == 'h-launch'
    assert 'E32rank128fixed' not in report


@pytest.mark.parametrize('problem', ['rank', 'counts', 'population'])
def test_wrong_h_architecture_or_smoke_population_is_rejected(problem):
    files = snapshot(); native = manifest(32); prefix = 'arms/'+m.ARM+'/'
    if problem == 'rank':
        native['config']['readout_rank'] = 64
    elif problem == 'counts':
        native['parameter_counts']['readout'] = 3226688
    else:
        files[prefix+'metrics.jsonl'] = {'text': json.dumps({'event': 'validation', 'updates': 8,
            'validation': {'stories': 8, 'tokens': 1926}})+'\n'}
    files[prefix+'manifest.json'] = item(native)
    with pytest.raises(ValueError, match='Unexpected'):
        m.summarize_snapshot(files, 'macm3', 'now')


def test_accepted_h_overrides_native_signal_failure_without_rewriting_it():
    files = snapshot(); prefix = 'arms/'+m.ARM+'/'
    failure = {'status': 'failed', 'arm': m.ARM, 'error': 'interrupted by signal 15'}
    files.update({'campaign-status.json': item(failure), 'failure.json': item(failure),
                  prefix+'manifest.json': item(manifest(32)), prefix+'accepted-early-stop.json': item(acceptance(m.ARM)),
                  prefix+'process-status.json': item({'status': 'failed', 'exit_code': -15, 'worker_pid': 555, 'runner_pid': 444}),
                  prefix+'launch.json': item({'campaign_launch_id': 'h-launch'})})
    original = json.loads(json.dumps(files))
    summary = m.summarize_snapshot(files, 'macm3', 'now')
    assert summary['status']['status'] == summary['arms'][m.ARM]['status'] == 'accepted early stop'
    assert summary['native_campaign_status'] == failure and summary['failure'] == failure
    assert summary['arms'][m.ARM]['trainer_pid'] == 555 and summary['arms'][m.ARM]['runner_pid'] == 444
    assert summary['arms'][m.ARM]['campaign_launch_id'] == 'h-launch'
    assert 'accepted stop governs status' in '\n'.join(m.render_report(summary))
    assert files == original


def fixture(tmp_path):
    parent = tmp_path/'rank64'; baseline = parent/'arms'/m.BASELINE; baseline.mkdir(parents=True)
    root = tmp_path/'rank32'; root.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    parent_receipt = write(parent/'manifest.json', {'launch_id': 'g-launch'})
    stop = write(baseline/'stop-receipt.json', {'process_cessation': {'confirmed': True}})
    decision = write(baseline/'early-stop-decision.json', {'decision': 'plateau'})
    accepted = acceptance(); accepted.update(stop_receipt=stop, decision_receipt=decision)
    accepted_receipt = write(baseline/'accepted-early-stop.json', accepted)
    native = write(baseline/'manifest.json', manifest(64))
    write(root/'manifest.json', {'parent_campaign': {**parent_receipt, 'launch_id': 'g-launch'},
                                'baseline_run': str(baseline), 'baseline_acceptance': accepted_receipt,
                                'baseline_files_sha256': {native['path']: native['sha256']}})
    return root, baseline


def fetch(root):
    request = {'root': str(root), 'files': m.FILES, 'baseline_names': m.BASELINE_NAMES, 'baseline': m.BASELINE, 'arm': m.ARM}
    return subprocess.run([sys.executable, '-c', m.REMOTE, json.dumps(request)], capture_output=True, text=True)


def test_readonly_fetch_binds_g_parent_decision_and_baseline(tmp_path):
    root, baseline = fixture(tmp_path)
    output = fetch(root)
    assert output.returncode == 0, output.stderr
    snapshot = json.loads(output.stdout)
    assert set(snapshot['files']) <= m.SNAPSHOT_FILES
    assert snapshot['source_paths']['baseline/early-stop-decision.json'] == str(baseline/'early-stop-decision.json')
    assert 'parent/manifest.json' in snapshot['files']


@pytest.mark.parametrize('changed', ['manifest.json', 'accepted-early-stop.json', 'stop-receipt.json', 'early-stop-decision.json'])
def test_changed_g_bound_artifacts_fail_closed(tmp_path, changed):
    root, baseline = fixture(tmp_path)
    (baseline/changed).write_text('{}')
    output = fetch(root)
    assert output.returncode != 0 and 'Bound artifact changed' in output.stderr
