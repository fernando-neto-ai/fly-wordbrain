"""Rank64 monitor tracks G against E without old B/F campaign requirements."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rank64_progress',ROOT/'scripts/refresh_connectorch_rank64_progress.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def item(value):
    return {'text':json.dumps(value)}


def manifest(rank):
    counts={'embedding':32768,'input_projection':450048,'neurons':148179,'layernorm':98786,
            'readout':rank*(49393+1024),'edge_gains':0,'other':0}
    counts['total']=sum(counts.values())
    return {'config':{'d_embed':32,'history_length':8,'readout_rank':rank,'plasticity':'fixed'},
            'parameter_counts':counts,'trainer_sources_sha256':{'scripts/train_connectorch.py':'c'*64},
            'frozen_buffers_sha256':{'brain.w_values':'b'*64}}


def selector(updates=13700,ce=3.15,accuracy=.36):
    return {'path':'/E/stopped-checkpoints/best.pt','sha256':'a'*64,'cursor':{'updates':updates,'epoch':11},
            'validation':{'cross_entropy':ce,'top1_accuracy':accuracy,'stories':100,'tokens':21874},
            'frozen_buffers_preserved':True,'frozen_buffers_sha256':{'brain.w_values':'b'*64}}


def acceptance():
    return {'format_version':1,'status':'accepted_early_stop','arm':m.BASELINE,
            'accepted_by':'agent_under_standing_user_authorization','process_cessation':{'confirmed':True},
            'durable_updates':14800,'observed_updates':14823,'completed_epochs':12,'test_evaluated':False,
            'selected_checkpoints':{'selectors':{'minimum_validation_ce':selector(),
                                                'maximum_validation_accuracy':selector(14300,3.16,.366)}}}


def snapshot():
    return {'manifest.json':item({'experiment_branches':{'repository':'https://github.com/fernando-neto-ai/fly-wordbrain',
                        'arms':{m.ARM:{'branch':'exp/decoder-rank64-fixed','commit':'d'*40}}}}),
            'baseline/manifest.json':item(manifest(128)),
            'baseline/accepted-early-stop.json':item(acceptance()),
            'baseline/status.json':item({'status':'running','updates':14823})}


def report(files):
    return '\n'.join(m.render_report(m.summarize_snapshot(files,'macm3','now')))


def test_initial_view_uses_preserved_e_and_planned_g_counts_only():
    text=report(snapshot())
    assert '| E32rank128fixed | 128 | 482,816 | 6,453,376 | 7,183,157 | verified arm manifest | accepted early stop | 14,800 |' in text
    assert '| G32rank64fixed | 64 | 482,816 | 3,226,688 | 3,956,469 | planned architecture | planned | 0 |' in text
    assert '36.60% (3.1600; 14,300)' in text
    assert 'B32fixed' not in text and 'F32rank128bounded' not in text


def test_first_g_validation_updates_dual_selector_view_and_source_receipts():
    files=snapshot();prefix='arms/'+m.ARM+'/'
    files['campaign-status.json']=item({'status':'running','phase':'training','arm':m.ARM})
    files[prefix+'manifest.json']=item(manifest(64))
    files[prefix+'status.json']=item({'status':'running','updates':117})
    files[prefix+'selected-checkpoints.json']=item({'selectors':{'minimum_validation_ce':selector(100,5.4,.10)}})
    files[prefix+'metrics.jsonl']={'text':json.dumps({'event':'validation','updates':100,'epoch':0,
                'validation':selector(100,5.4,.10)['validation']})+'\n'}
    summary=m.summarize_snapshot(files,'macm3','now');text='\n'.join(m.render_report(summary))
    assert 'Current arm: **G32rank64fixed**' in text
    assert '| verified arm manifest | running | 117 | 5.4000 (10.00%; 100)' in text
    assert '10.00% (5.4000; 100; retention pending)' in text
    assert summary['arms'][m.ARM]['trainer_sources_sha256']['scripts/train_connectorch.py']=='c'*64
    assert '`exp/decoder-rank64-fixed`' in text


@pytest.mark.parametrize('problem',['rank','counts','population'])
def test_wrong_architecture_or_smoke_population_cannot_look_like_full_g(problem):
    files=snapshot();native=manifest(64);prefix='arms/'+m.ARM+'/'
    if problem=='rank':native['config']['readout_rank']=128
    elif problem=='counts':native['parameter_counts']['readout']=6453376
    else:
        metric=selector(8,6.,.1)['validation'];metric.update(stories=8,tokens=2000)
        files[prefix+'metrics.jsonl']={'text':json.dumps({'event':'validation','updates':8,'epoch':0,'validation':metric})+'\n'}
    files[prefix+'manifest.json']=item(native)
    with pytest.raises(ValueError,match='Unexpected'):
        m.summarize_snapshot(files,'macm3','now')


@pytest.mark.parametrize('terminal_lf',[False,True])
def test_physical_jsonl_lf_preserves_unicode_separators(terminal_lf):
    files=snapshot();row={'event':'validation','updates':100,'epoch':0,'validation':selector(100,5.4,.1)['validation'],
                        'metadata':{'note':'one\u2028two\u2029three\u0085four'}}
    files['arms/'+m.ARM+'/metrics.jsonl']={'text':json.dumps(row,ensure_ascii=False)+('\n' if terminal_lf else '')}
    assert len(m.summarize_snapshot(files,'macm3','now')['arms'][m.ARM]['validation_history'])==1


def fixture(tmp_path):
    parent=tmp_path/'decoder';baseline=parent/'arms'/m.BASELINE;baseline.mkdir(parents=True)
    root=tmp_path/'rank64';root.mkdir()
    def write(path,value):
        path.write_text(json.dumps(value));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    par=write(parent/'manifest.json',{'launch_id':'parent-id'})
    stop=write(baseline/'stop-receipt.json',{'process_cessation':{'confirmed':True}})
    accepted=acceptance();accepted['stop_receipt']=stop
    accepted=write(baseline/'accepted-early-stop.json',accepted)
    native=write(baseline/'manifest.json',manifest(128))
    write(root/'manifest.json',{'parent_campaign':{**par,'launch_id':'parent-id'},'baseline_run':str(baseline),
                              'baseline_acceptance':accepted,'baseline_files_sha256':{native['path']:native['sha256']}})
    return root,baseline,parent


def fetch(root):
    request={'root':str(root),'files':m.FILES,'baseline_names':m.BASELINE_NAMES,'baseline':m.BASELINE,'arm':m.ARM}
    return subprocess.run([sys.executable,'-c',m.REMOTE,json.dumps(request)],capture_output=True,text=True)


def test_rank64_fetch_requires_no_b_or_f_quality_receipt(tmp_path):
    root,baseline,parent=fixture(tmp_path)
    output=fetch(root)
    assert output.returncode==0,output.stderr
    snap=json.loads(output.stdout)
    assert set(snap['files']) <= m.SNAPSHOT_FILES
    assert snap['source_paths']['baseline/accepted-early-stop.json']==str(baseline/'accepted-early-stop.json')
    assert snap['source_paths']['parent/manifest.json']==str(parent/'manifest.json')
    assert not any('quality' in path for path in snap['source_paths'])
    assert 'G32rank64fixed' in report(snap['files'])


@pytest.mark.parametrize('changed',['manifest.json','accepted-early-stop.json','stop-receipt.json'])
def test_changed_e_baseline_artifact_is_rejected(tmp_path,changed):
    root,baseline,_=fixture(tmp_path);(baseline/changed).write_text('{}')
    output=fetch(root)
    assert output.returncode!=0
    assert 'Bound artifact changed' in output.stderr


def test_parent_identity_is_bound(tmp_path):
    root,_,parent=fixture(tmp_path);(parent/'manifest.json').write_text('{}')
    output=fetch(root)
    assert output.returncode!=0
    assert 'Bound artifact changed' in output.stderr
