"""Matched delayed-memory training of existing fast rules versus cached heads.

The original language run is untouched. Both arms always train with fast writes
enabled. Only final prediction-8 identity/order CE supplies gradients.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from fly_wordbrain.action_candidates import ActionCandidates
from fly_wordbrain.action_train import utc_now, write_json
from fly_wordbrain.feedback_data import collate_feedback
from fly_wordbrain.memory_probe_features import load_frozen_model
from fly_wordbrain.pair_decoder import file_sha256
from fly_wordbrain.retention_data import build_retention_probes, balanced_training_batches
from fly_wordbrain.retention_model import RetentionModel, RetentionReadout


TASKS = ('identity', 'order')
FINAL_ONLY = [0., 0., 0., 0., 1.]


def parameter_hash(parameters):
    digest = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def restore(model, values):
    parameters = dict(model.named_parameters())
    if set(parameters) != set(values):
        raise AssertionError('Snapshot parameter names differ')
    with torch.no_grad():
        for name, target in parameters.items():
            target.copy_(values[name].to(target.device))


def gradient_receipt(model):
    result = {}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        grad = value.grad
        if grad is None:
            raise AssertionError('Missing gradient: '+name)
        g = grad.detach().cpu().double()
        if not torch.isfinite(g).all():
            raise AssertionError('Nonfinite gradient: '+name)
        result[name] = {'l2': float(g.square().sum().sqrt()), 'max_abs': float(g.abs().max()),
                        'nonzero': int(torch.count_nonzero(g)), 'values': value.numel()}
    return result


def save_parameters(path, model, protocol, step, role):
    path = Path(path)
    temporary = path.with_suffix('.pt.partial')
    torch.save({'schema': 1, 'kind': 'retention_inference', 'parameters': snapshot(model),
        'feature_means': model.feature_means.detach().cpu(), 'feature_stds': model.feature_stds.detach().cpu(),
        'step': step, 'role': role, 'protocol': protocol, 'training_resume_supported': False}, temporary)
    temporary.replace(path)


def main(args):
    if args.updates < 1 or args.validate_every < 1:
        raise ValueError('Require positive update and validation counts')
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        raise ValueError('Refusing to overwrite an existing retention experiment')
    def progress(phase, **values):
        record = {'status': 'running', 'phase': phase, 'pid': os.getpid(),
            'timestamp_utc': utc_now(), 'elapsed_seconds': time.monotonic()-started, **values}
        write_json(output/'progress.json', record)
        print(json.dumps(record), flush=True)
    torch.set_num_threads(4)
    progress('loading')
    selector, receipt = load_frozen_model(args.checkpoint, args.graph, args.structure, args.device)
    dataset = json.loads(Path(args.dataset).read_text())
    dataset_sha = file_sha256(args.dataset)
    if dataset_sha != receipt['dataset_sha256']:
        raise AssertionError('Checkpoint dataset mismatch')
    previous = json.loads(Path(args.previous_rows).read_text())
    probes = build_retention_probes(dataset, previous, dataset_sha256=dataset_sha)
    rows, windows = probes.rows, probes.windows
    write_json(output/'rows.json', rows)
    proposals = ActionCandidates(dataset['splits']['train'], len(dataset['vocabulary']), top_k=10)
    task_ids = torch.tensor([TASKS.index(r['task']) for r in rows], device=args.device)
    labels = torch.tensor([r['label'] for r in rows], device=args.device)
    split = {name: [i for i,r in enumerate(rows) if r['split']==name] for name in ('train','val','test')}
    if any(not indices for indices in split.values()):
        raise AssertionError('Empty experiment split')
    # Check final proposals before training; labels are never fed into the fly.
    grouped = {}
    for i,row in enumerate(rows):
        grouped.setdefault(row['group_id'], []).append(i)
    for indices in grouped.values():
        batch = collate_feedback([windows[i] for i in indices], proposals, exclude_own_story=True)
        for name in ('previous','current','candidates','probabilities'):
            values = getattr(batch,name)[:,-1]
            if not torch.equal(values, values[:1].expand_as(values)):
                raise AssertionError('Final inputs vary within group')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/'fly_wordbrain/retention_model.py', root/'fly_wordbrain/retention_data.py',
               root/'fly_wordbrain/memory_probe_features.py', root/'fly_wordbrain/memory_probe_data.py',
               root/'fly_wordbrain/feedback_data.py', root/'fly_wordbrain/action_candidates.py']
    protocol = {'schema': 1, 'kind': 'delayed_retention_training', 'snapshot': receipt,
        'timestamp_utc': utc_now(), 'device': args.device, 'torch_version': str(torch.__version__), 'dataset_sha256': dataset_sha,
        'previous_rows_sha256': file_sha256(args.previous_rows), 'data': probes.metadata,
        'updates': args.updates, 'batch_size': 16, 'per_task_per_batch': 8, 'seed': 0, 'batch_seed': 2718,
        'heads': {'identity': 2570, 'order': 514, 'total': 3084}, 'brain_parameters': 8838,
        'total_treatment_trainable': 11922, 'decoder_control_trainable': 3084,
        'objective': 'Mean of identity and order cross entropies at final prediction8 only',
        'loss_time_weights': FINAL_ONLY, 'auxiliary_loss': False,
        'head_optimizer': {'name': 'Adam', 'lr': .003, 'eps': 1e-8, 'clip_norm': 1.},
        'rule_optimizer': {'name': 'Adam', 'lr': .003, 'eps': 1e-12, 'clip_norm': 1.},
        'training_plasticity_enabled': True, 'calibration': 'All new TRAIN rows; fixed per-time mean and max(std,1e-12), no mask',
        'activity_scales': 'Original frozen checkpoint values, unchanged',
        'selection': 'Lowest validation mean final CE, checked at0, every validation interval, and final update; ties retain earlier step',
        'validation_every': args.validate_every, 'test_access': 'Neural test extraction/scoring only after both arms finish and select by validation',
        'test_conditions': ['fast_on','fast_off','clear_fast_before_final'],
        'primary_endpoint': 'Validation-selected treatment vs independently validation-selected decoder-only control on new test groups',
        'limits': ['Synthetic fixed-word identity/order, not language accuracy', 'One training seed, one starting checkpoint',
                   'New probe contexts all belong to original language TRAIN', 'No geometry-randomized control',
                   'H clearing tests final fast read; earlier fast influence may remain in neural state'],
        'source_sha256': {str(p.relative_to(root)): file_sha256(p) for p in sources}}
    write_json(output/'protocol.json', protocol)
    temporary = RetentionModel(selector, np.zeros((5,256),np.float32), np.ones((5,256),np.float32), train_rules=False)
    def neural_features(model, indices, condition='fast_on', announce=None):
        parts = []
        with torch.no_grad():
            for offset in range(0,len(indices),16):
                chosen = indices[offset:offset+16]
                batch = collate_feedback([windows[i] for i in chosen], proposals,
                                         device=args.device, exclude_own_story=True)
                feature = model.forward_features(batch, plasticity=condition!='fast_off',
                    clear_fast_before_final=condition=='clear_fast_before_final')
                parts.append(feature.detach().cpu().numpy())
                if announce:
                    progress(announce, completed=min(offset+16,len(indices)),total=len(indices))
        result = np.concatenate(parts)
        if not np.isfinite(result).all():
            raise AssertionError('Nonfinite raw features')
        return result
    trainval = split['train']+split['val']
    frozen_cache = np.zeros((len(rows),5,256),np.float32)
    frozen_cache[trainval] = neural_features(temporary,trainval,announce='caching_frozen_train_validation')
    values = frozen_cache[split['train']].astype(np.float64)
    means, stds = values.mean(axis=0).astype(np.float32), np.maximum(values.std(axis=0),1e-12).astype(np.float32)
    np.savez_compressed(output/'initial-cache.npz', features=frozen_cache, cached_indices=trainval, means=means,stds=stds)
    write_json(output/'calibration.json', {'train_rows':len(split['train']),'std_min_per_time':stds.min(1).tolist(),
        'std_max_per_time':stds.max(1).tolist(),'floor':1e-12,'masked_dimensions':0,
        'cache_sha256':file_sha256(output/'initial-cache.npz'),'test_features_extracted':False})
    del temporary,values
    model = RetentionModel(selector,means,stds,train_rules=True,seed=0)
    control = RetentionReadout(means,stds,device=args.device,seed=0)
    treatment_initial, control_initial = snapshot(model), snapshot(control)
    original_rule_values = {name:p.detach().cpu().clone() for name,p in selector.brain.named_parameters()}
    original_language_head = snapshot(selector.readout)
    interface_before = model.brain.interface_fingerprint()
    def head_params(head):
        return list(head.identity_head.parameters())+list(head.order_head.parameters())
    rules = [p for p in model.brain.parameters() if p.requires_grad]
    if sum(p.numel() for p in rules)!=8838 or sum(p.numel() for p in model.parameters() if p.requires_grad)!=11922:
        raise AssertionError('Wrong trainable parameter set')
    for name in ('identity_head','order_head'):
        for a,b in zip(getattr(model,name).parameters(),getattr(control,name).parameters()):
            if not torch.equal(a,b): raise AssertionError('Unmatched initial heads')
    def make_optimizers():
        return (torch.optim.Adam([{'params':rules,'lr':.003,'eps':1e-12},
                                  {'params':head_params(model),'lr':.003,'eps':1e-8}]),
                torch.optim.Adam(head_params(control),lr=.003,eps=1e-8))
    opt, control_opt = make_optimizers()
    def train_step(indices, audit=False):
        batch = collate_feedback([windows[i] for i in indices],proposals,device=args.device,exclude_own_story=True)
        opt.zero_grad(set_to_none=True)
        features = model.forward_features(batch,plasticity=True)
        loss = model.memory_loss(features,task_ids[indices],labels[indices],FINAL_ONLY)
        loss.backward()
        gradients = gradient_receipt(model)
        torch.nn.utils.clip_grad_norm_(rules,1.,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(head_params(model),1.,error_if_nonfinite=True)
        opt.step()
        control_opt.zero_grad(set_to_none=True)
        cached = torch.as_tensor(frozen_cache[indices],device=args.device)
        control_loss = control.memory_loss(cached,task_ids[indices],labels[indices],FINAL_ONLY)
        control_loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params(control),1.,error_if_nonfinite=True)
        control_opt.step()
        return {'loss':float(loss.detach().cpu()), 'control_loss':float(control_loss.detach().cpu()),
                'gradients':gradients if audit else {name: g['l2'] for name,g in gradients.items()}}
    first_epoch = balanced_training_batches(rows,epoch=0)
    first = first_epoch[0]
    batch = collate_feedback([windows[i] for i in first],proposals,device=args.device,exclude_own_story=True)
    checks = {'final_inputs_identical': True}
    with torch.no_grad():
        original,state = model.forward_features(batch,return_state=True)
        replay = model.forward_features(batch)
        mutated = model.forward_features(replace(batch,targets=(batch.targets+1)%selector.brain.vocab_size))
        checks.update(replay_max_abs=float((original-replay).abs().max().cpu()),
            hidden_target_max_abs=float((original-mutated).abs().max().cpu()),
            cache_max_abs=float(np.abs(original.cpu().numpy()-frozen_cache[first]).max()),
            fast_state_rms=float(state.fast.cpu().double().square().mean().sqrt()))
        if checks['replay_max_abs']!=0 or checks['hidden_target_max_abs']!=0 or checks['cache_max_abs']>1e-9:
            raise AssertionError('Replay/label/cache parity failed: '+str(checks))
    progress('preflight')
    checks['preflight'] = [train_step(first,audit=True) for _ in range(2)]
    deltas = {name:float((p.detach().cpu()-treatment_initial[name]).abs().max()) for name,p in model.named_parameters()}
    if not all(deltas['selector.brain.'+name]>0 for name in model.brain.rule_parameter_names):
        raise AssertionError('Preflight did not update every rule tensor')
    checks['preflight_max_parameter_deltas'] = deltas
    restore(model,treatment_initial);restore(control,control_initial)
    opt,control_opt = make_optimizers()
    checks['preflight_restored'] = parameter_hash(snapshot(model))==parameter_hash(treatment_initial)
    checks['control_preflight_restored'] = parameter_hash(snapshot(control))==parameter_hash(control_initial)
    write_json(output/'checks.json',checks)
    if not checks['preflight_restored'] or not checks['control_preflight_restored']:
        raise AssertionError('Preflight not restored')
    def score(head, features, indices):
        result = {}
        with torch.no_grad():
            x = torch.as_tensor(features,device=args.device)
            for task_id,task in enumerate(TASKS):
                keep = [j for j,i in enumerate(indices) if rows[i]['task']==task]
                chosen = [indices[j] for j in keep]
                logits = head.logits(x[keep],task).detach().cpu().double()
                target = labels[chosen].cpu()
                probabilities = logits.softmax(-1)
                predicted = logits.argmax(-1)
                correct = predicted==target[:,None]
                result[task] = {'count':len(chosen),'row_indices':chosen,'labels':target.tolist(),
                    'prediction_times':[4,5,6,7,8], 'accuracy':correct.double().mean(0).tolist(),
                    'correct':correct.sum(0).tolist(),
                    'cross_entropy':[float(F.cross_entropy(logits[:,j],target)) for j in range(5)],
                    'predictions':predicted.tolist(),'final_probabilities':probabilities[:,-1].tolist()}
        result['mean_final_ce'] = sum(result[task]['cross_entropy'][-1] for task in TASKS)/2
        return result
    monitor_groups=[]
    for task in TASKS:
        names=list(dict.fromkeys(rows[i]['group_id'] for i in split['train'] if rows[i]['task']==task))[:4]
        monitor_groups.extend(names)
    monitor=[i for i in split['train'] if rows[i]['group_id'] in monitor_groups]
    history=[]
    best={'treatment':{'ce':float('inf'),'step':None,'parameters':None},
          'decoder_only':{'ce':float('inf'),'step':None,'parameters':None}}
    def validate(step):
        progress('validation',step=step,total_steps=args.updates)
        raw=neural_features(model,split['val']+monitor)
        record={'step':step,'elapsed_seconds':time.monotonic()-started,
            'treatment':{'validation':score(model,raw[:len(split['val'])],split['val']),
                         'train_monitor':score(model,raw[len(split['val']):],monitor)},
            'decoder_only':{'validation':score(control,frozen_cache[split['val']],split['val']),
                            'train_monitor':score(control,frozen_cache[monitor],monitor)}}
        if step==0 and abs(record['treatment']['validation']['mean_final_ce']-
                          record['decoder_only']['validation']['mean_final_ce'])>1e-7:
            raise AssertionError('Unmatched initial validation scores')
        for name,head in (('treatment',model),('decoder_only',control)):
            ce=record[name]['validation']['mean_final_ce']
            if ce < best[name]['ce']:
                best[name]={'ce':ce,'step':step,'parameters':snapshot(head)}
                save_parameters(output/(name+'-best.pt'),head,protocol,step,'best_validation')
        history.append(record)
        write_json(output/'validation.json',history)
        print('VALIDATION '+json.dumps({'step':step,**{name:{task:record[name]['validation'][task]['accuracy'][-1]
            for task in TASKS} for name in best}}),flush=True)
    validate(0)
    for step in range(1,args.updates+1):
        epoch=(step-1)//len(first_epoch)
        batches=balanced_training_batches(rows,epoch=epoch)
        indices=batches[(step-1)%len(batches)]
        metrics=train_step(indices,audit=step==1)
        with (output/'training.jsonl').open('a') as stream:
            stream.write(json.dumps({'step':step,**metrics})+'\n')
        if step%8==0 or step==1:
            progress('training',step=step,total_steps=args.updates,loss=metrics['loss'],control_loss=metrics['control_loss'])
        if step%args.validate_every==0 or step==args.updates:
            validate(step)
    save_parameters(output/'treatment-last.pt',model,protocol,args.updates,'last')
    save_parameters(output/'decoder_only-last.pt',control,protocol,args.updates,'last')
    checks['final_parameter_max_deltas']={name:float((p.detach().cpu()-treatment_initial[name]).abs().max()) for name,p in model.named_parameters()}
    checks['final_rules_changed_values']={name:int(torch.count_nonzero(p.detach().cpu()-original_rule_values[name]))
        for name,p in model.brain.named_parameters()}
    checks['base_graph_preserved']=model.brain.base_graph_fingerprint()==receipt['base_graph_sha256']
    checks['interface_preserved']=model.brain.interface_fingerprint()==interface_before
    checks['original_language_head_preserved']=parameter_hash(snapshot(selector.readout))==parameter_hash(original_language_head)
    if not all(checks[k] for k in ('base_graph_preserved','interface_preserved','original_language_head_preserved')):
        raise AssertionError('Frozen graph/interface/original head changed')
    write_json(output/'checks.json',checks)
    result={'status':'running','selected_steps':{name:value['step'] for name,value in best.items()},
        'test_rows':len(split['test']),'arms':{}}
    # No test neural feature or metric has been computed before this point.
    restore(model,best['treatment']['parameters'])
    for name,head in (('treatment',model),('decoder_only',control)):
        if name=='decoder_only':
            restore(model,treatment_initial)
            restore(control,best[name]['parameters'])
        result['arms'][name]={}
        for condition in ('fast_on','fast_off','clear_fast_before_final'):
            raw=neural_features(model,split['test'],condition,announce='test_'+name+'_'+condition)
            result['arms'][name][condition]=score(head,raw,split['test'])
            np.savez_compressed(output/(name+'-'+condition+'-test-features.npz'),features=raw,row_indices=split['test'])
            write_json(output/'results.json',result)
    result.update(status='completed',elapsed_seconds=time.monotonic()-started)
    write_json(output/'results.json',result)
    checks['source_unchanged_after_run']={str(p.relative_to(root)):file_sha256(p)==protocol['source_sha256'][str(p.relative_to(root))] for p in sources}
    if not all(checks['source_unchanged_after_run'].values()):raise AssertionError('Sources changed during run')
    write_json(output/'checks.json',checks)
    write_json(output/'progress.json',{'status':'completed','timestamp_utc':utc_now(),'elapsed_seconds':time.monotonic()-started,
        'steps':args.updates,'selected_steps':result['selected_steps']})
    print('RETENTION_COMPLETED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','checkpoint','graph','structure','dataset','previous-rows'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--updates',type=int,default=256)
    parser.add_argument('--validate-every',type=int,default=64)
    args=parser.parse_args()
    try:
        main(args)
    except Exception:
        write_json(Path(args.output)/'failure.json',{'status':'failed','timestamp_utc':utc_now(),'traceback':traceback.format_exc()})
        raise
