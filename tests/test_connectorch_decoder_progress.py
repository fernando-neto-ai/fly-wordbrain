"""Decoder monitoring keeps B baseline and rank-128 E/F identities separate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('decoder_progress', ROOT/'scripts/refresh_connectorch_decoder_progress.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def item(value):
    return {'text':json.dumps(value)}


def selected(update=9700, ce=4.51, accuracy=.315):
    return {'path':'/model/stopped-checkpoints/best.pt', 'sha256':'a'*64,
            'cursor':{'updates':update, 'epoch':8},
            'validation':{'cross_entropy':ce, 'top1_accuracy':accuracy, 'tokens':21874, 'stories':100},
            'frozen_buffers_preserved':True, 'frozen_buffers_sha256':{'brain.w_values':'b'*64}}


def manifest(arm):
    counts = m.EXPECTED_COUNTS[arm].copy()
    counts.pop('encoder')
    counts.update(embedding=32768, input_projection=450048, other=0)
    return {'parameter_counts':counts,
            'config':{'d_embed':32, 'history_length':8, 'readout_rank':0 if arm=='B32fixed' else 128,
                      'plasticity':'bounded10' if arm=='F32rank128bounded' else 'fixed'},
            'trainer_sources_sha256':{'scripts/train_connectorch.py':'c'*64},
            'frozen_buffers_sha256':{'brain.w_values':'b'*64}}


def receipt():
    return {'format_version':1, 'status':'accepted_early_stop', 'arm':'B32fixed',
            'accepted_by':'agent_under_standing_user_authorization', 'durable_updates':16200,
            'observed_updates':16262, 'completed_epochs':13, 'process_cessation':{'confirmed':True},
            'selected_checkpoints':{'format_version':1, 'selectors':{
                'minimum_validation_ce':selected(), 'maximum_validation_accuracy':selected(15200,4.62,.333)}},
            'test_evaluated':False}


def base_snapshot():
    return {'manifest.json':item({'baseline_run':'/old/arms/B32fixed',
                                 'baseline_acceptance':{'path':'/old/arms/B32fixed/accepted-early-stop.json','sha256':'a'*64},
                                 'experiment_branches':{'repository':'https://github.com/fernando-neto-ai/fly-wordbrain',
                                     'arms':{'E32rank128fixed':{'branch':'exp/decoder128-fixed','commit':'d'*40}}}}),
            'baseline/manifest.json':item(manifest('B32fixed')),
            'baseline/accepted-early-stop.json':item(receipt()),
            'baseline/status.json':item({'status':'running','updates':16262})}


def report(files):
    return '\n'.join(m.render_report(m.summarize_snapshot(files,'macm3','now')))


def test_planned_decoder_arms_keep_accepted_full_head_baseline():
    text = report(base_snapshot())
    assert '| B32fixed | full | 482,816 | 50,578,432 | 0 | 51,308,213 | verified arm manifest | accepted early stop | 16,200 |' in text
    assert '| E32rank128fixed | 128 | 482,816 | 6,453,376 | 0 | 7,183,157 | planned architecture | planned | 0 |' in text
    assert '| F32rank128bounded | 128 | 482,816 | 6,453,376 | 18,322 | 7,201,479 | planned architecture | planned | 0 |' in text
    assert 'C128bounded' not in text and 'D32bounded' not in text
    assert '33.30% (4.6200; 15,200)' in text


def test_running_rank128_arm_tracks_dual_retention_sources_and_branch():
    files = base_snapshot()
    prefix='arms/E32rank128fixed/'
    files['campaign-status.json']=item({'status':'running','phase':'training','arm':'E32rank128fixed'})
    files[prefix+'manifest.json']=item(manifest('E32rank128fixed'))
    files[prefix+'status.json']=item({'status':'running','activity':'training','updates':120})
    files[prefix+'selected-checkpoints.json']=item({'selectors':{'minimum_validation_ce':selected(100,5.3,.27)}})
    files[prefix+'metrics.jsonl']={'text':json.dumps({'event':'validation','updates':100,'epoch':0,'validation':selected(100,5.3,.27)['validation']})+'\n'}
    summary=m.summarize_snapshot(files,'macm3','now')
    text='\n'.join(m.render_report(summary))
    assert 'Current arm: **E32rank128fixed**' in text
    assert '| verified arm manifest | running | 120 | 5.3000 (27.00%; 100)' in text
    assert '27.00% (5.3000; 100; retention pending)' in text
    assert summary['arms']['E32rank128fixed']['trainer_sources_sha256']['scripts/train_connectorch.py']=='c'*64
    assert '`exp/decoder128-fixed`' in text
    assert '`'+'a'*64+'`' in text


def test_completed_adaptive_arm_uses_manifest_count_and_result_status():
    files=base_snapshot();prefix='arms/F32rank128bounded/'
    files[prefix+'manifest.json']=item(manifest('F32rank128bounded'))
    files[prefix+'status.json']=item({'status':'running','updates':52609})
    files[prefix+'results.json']=item({'status':'completed'})
    arm=m.summarize_snapshot(files,'macm3','now')['arms']['F32rank128bounded']
    assert arm['status']=='completed'
    assert arm['parameter_counts']['edge_gains']==18322
    assert arm['parameter_count_source']=='verified arm manifest'


def test_decoder_dispatcher_configuration_and_counts_match_progress_contract():
    spec=importlib.util.spec_from_file_location('decoder_campaign_progress_contract', ROOT/'scripts/run_connectorch_decoder_campaign.py')
    controller=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    files=base_snapshot()
    for arm in controller.ARMS:
        native=manifest(arm[0])
        native['config']=controller.arm_config(arm)
        native['parameter_counts']=controller.expected_counts(arm)
        files['arms/'+arm[0]+'/manifest.json']=item(native)
    summary=m.summarize_snapshot(files,'macm3','now')
    assert set(summary['arms'])=={arm[0] for arm in controller.ARMS}
    assert all(arm['parameter_count_source']=='verified arm manifest' and arm['readout_rank']==128
               for arm in summary['arms'].values())


@pytest.mark.parametrize('change',['rank','counts','plasticity'])
def test_mismatched_architecture_does_not_get_silently_labeled_rank128(change):
    files=base_snapshot();native=manifest('F32rank128bounded')
    if change=='rank':native['config']['readout_rank']=0
    elif change=='counts':native['parameter_counts']['readout']=50578432
    else:native['config']['plasticity']='fixed'
    files['arms/F32rank128bounded/manifest.json']=item(native)
    with pytest.raises(ValueError,match='Unexpected'):
        m.summarize_snapshot(files,'macm3','now')


def test_smoke_scores_are_not_full_training_results_and_partial_line_is_ignored():
    files=base_snapshot()
    files['smokes/E32rank128fixed/metrics.jsonl']={'text':json.dumps({'event':'validation','updates':8,'epoch':0,
        'validation':{'cross_entropy':.12345,'top1_accuracy':.99}})+'\n'}
    files['arms/E32rank128fixed/metrics.jsonl']={'text':'{"event":"validation","up'}
    text=report(files)
    assert '.12345' not in text
    assert 'No full-arm validation fetched yet.' in text
    assert m.summarize_snapshot(files,'macm3','now')['arms']['E32rank128fixed']['validation_history']==[]


def test_unicode_separators_in_a_json_string_preserve_one_physical_record():
    files=base_snapshot()
    row={'event':'validation', 'updates':100, 'epoch':0, 'validation':selected(100,5.3,.27)['validation'],
         'note':'one\u2028two\u2029three\u0085four'}
    files['arms/E32rank128fixed/metrics.jsonl']={'text':json.dumps(row,ensure_ascii=False)+'\n'}
    rows=m.summarize_snapshot(files,'macm3','now')['arms']['E32rank128fixed']['validation_history']
    assert len(rows)==1
    assert rows[0]['updates']==100


def fetch_fixture(tmp_path):
    root=tmp_path/'decoder';root.mkdir()
    baseline=tmp_path/'encoder/arms/B32fixed';baseline.mkdir(parents=True)
    def write(path,value):
        path.write_text(json.dumps(value));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    stop=write(baseline/'stop-receipt.json',{'process_cessation':{'confirmed':True}})
    acceptance=receipt();acceptance['stop_receipt']=stop
    accepted=write(baseline/'accepted-early-stop.json',acceptance)
    native=write(baseline/'manifest.json',manifest('B32fixed'))
    write(root/'manifest.json',{'baseline_run':str(baseline),'baseline_acceptance':accepted,
                              'baseline_files_sha256':{native['path']:native['sha256']}})
    return root,baseline


def fetch(root):
    request={'root':str(root),'files':m.FILES,'baseline_names':m.BASELINE_NAMES,'arms':m.ARMS}
    return subprocess.run([sys.executable,'-c',m.REMOTE,json.dumps(request)],capture_output=True,text=True)


def test_fetch_binds_baseline_receipts_and_transfer_hashes(tmp_path):
    root,baseline=fetch_fixture(tmp_path)
    output=fetch(root)
    assert output.returncode==0,output.stderr
    fetched=json.loads(output.stdout)
    assert fetched['source_paths']['baseline/accepted-early-stop.json']==str(baseline/'accepted-early-stop.json')
    for value in fetched['files'].values():
        assert hashlib.sha256(value['text'].encode()).hexdigest()==value['sha256']
    assert report(fetched['files']).count('| B32fixed |')==1


@pytest.mark.parametrize('name',['accepted-early-stop.json','stop-receipt.json','manifest.json'])
def test_fetch_rejects_changed_baseline_artifacts(tmp_path,name):
    root,baseline=fetch_fixture(tmp_path)
    (baseline/name).write_text('{}')
    output=fetch(root)
    assert output.returncode!=0
    assert 'Bound artifact changed' in output.stderr


def test_missing_decoder_manifest_does_not_accidentally_fetch_encoder_campaign(tmp_path):
    (tmp_path/'readiness.json').write_text(json.dumps({'status':'waiting_for_preflight'}))
    output=fetch(tmp_path)
    assert output.returncode==0
    fetched=json.loads(output.stdout)
    assert list(fetched['files'])==['readiness.json']
    text=report(fetched['files'])
    assert 'Baseline acceptance receipt has not yet been fetched' in text
    assert '**waiting_for_preflight**' in text
