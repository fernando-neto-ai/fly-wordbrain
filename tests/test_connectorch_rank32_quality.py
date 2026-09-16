"""Rank-aware evaluator contracts; fake fixtures and small CPU tensors only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("rank32_quality", ROOT / "scripts/evaluate_connectorch_rank32_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_physical_jsonl_preserves_unicode_separators(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"text": "first\u2028second\u2029third"}, ensure_ascii=False) + "\n" + json.dumps({"end": True}) + "\n")
    assert quality.jsonl(path) == [{"text": "first\u2028second\u2029third"}, {"end": True}]


@pytest.mark.parametrize("script", sorted(quality.DISPATCHERS))
def test_all_dispatcher_generations_must_be_held(script):
    row = {"pid": 41, "ppid": 1, "state": "S", "command": f"python -u scripts/{script} --worker"}
    with pytest.raises(ValueError, match="Idle GPU"):
        quality.process_guard([row])
    assert quality.process_guard([{**row, "state": "Ts"}])["held_dispatchers"][0]["pid"] == 41


@pytest.mark.parametrize("script", ["train_connectorch.py", "evaluate_connectorch_quality.py", "evaluate_connectorch_rank32_quality.py", "audit_connectorch_rank32.py"])
@pytest.mark.parametrize("state", ["S", "Ts"])
def test_paused_or_running_gpu_workers_are_rejected(script, state):
    with pytest.raises(ValueError, match="Idle GPU"):
        quality.process_guard([{"pid": 42, "ppid": 1, "state": state, "command": f"python scripts/{script}"}])


def test_zombie_exited_own_pid_excluded_shell_parent_not_gpu_worker():
    rows = quality.parse_processes("42 1 Z <defunct>\n43 1 S python scripts/evaluate_connectorch_rank32_quality.py\n44 1 S sh -c python scripts/train_connectorch.py\n")
    result = quality.process_guard(rows, {42}, own_pid=43)
    assert [r["pid"] for r in result["exited_zombies"]] == [42]


def test_declared_configs_accepted_rank0_and_recipe_drift_rejected():
    arms = []
    for label, rank in quality.ARMS.items():
        manifest = {"config": {**quality.COMMON_CONFIG, "readout_rank": rank},
                    "debug": False, "parameter_counts": quality.expected_counts(rank),
                    "frozen_buffers_sha256": {name: "graph" for name in quality.GRAPH_BUFFERS},
                    "reference_repository": "ngxson/fly-llm-hf", "reference_revision": quality.trainer.REVISION}
        for key in ("data_sha256", "groups_sha256", "reference_files", "trainer_sources_sha256",
                    "connectorch_source", "assumptions", "planned_phase_updates", "epoch_updates", "split_sizes", "data_provenance"):
            manifest[key] = {"same": "bound"}
        quality.verify_config(manifest, label)
        arms.append({"manifest": manifest})
        bad = copy.deepcopy(manifest); bad["config"]["readout_rank"] = 0
        with pytest.raises(ValueError, match="architecture"):
            quality.verify_config(bad, label)
    quality.verify_pair(*arms)
    bad = copy.deepcopy(arms[1]); bad["manifest"]["config"]["seed"] = 43
    with pytest.raises(ValueError, match="Recipe differs"):
        quality.verify_pair(arms[0], bad)
    assert quality.expected_counts(32)["total"] == 2343125
    assert quality.expected_counts(64)["total"] == 3956469


def test_cessation_covers_workers_and_is_not_a_paused_worker():
    receipt = {"process_cessation": {"confirmed": True, "processes": [{"pid": 7, "alive": False, "exited": True, "birth": "fixture"}]}}
    assert quality.verify_cessation(receipt, {7})[0]["pid"] == 7
    with pytest.raises(ValueError, match="omits"):
        quality.verify_cessation(receipt, {7, 8})
    receipt["process_cessation"]["processes"][0]["alive"] = True
    with pytest.raises(ValueError, match="exited"):
        quality.verify_cessation(receipt, {7})


def test_exact_chunk_exposure_matches_log_and_detects_missing_or_corrupt_updates():
    rows = [{"ids": [1, 4, 5, 6, 2]}, {"ids": [1, 8, 2]}]
    cfg = {"epochs": 2, "second_epochs": 0, "batch_size": 2, "chunk_size": 3, "seed": 42}
    table = quality.budget_table(rows, cfg, 4)
    assert [table[i]["tokens"] for i in range(5)] == [0, 5, 6, 11, 12]
    training = [{"updates": i, "tokens": table[i]["chunk_tokens"], **{k: table[i][k] for k in ("epoch", "batch", "offset")}} for i in range(1, 5)]
    arm = {"accepted": {"durable_updates": 4, "observed_updates": 4}, "training_records": training}
    assert quality.verify_training_budget(arm, table)["durable_training_tokens"] == 12
    training[2]["tokens"] += 1
    with pytest.raises(ValueError, match="data exposure"):
        quality.verify_training_budget(arm, table)
    assert quality.budget_table(rows, cfg, 0) == {0: {"tokens": 0, "epoch": 0}}


def test_common_budget_retains_earlier_ties_and_excludes_later_winners():
    def row(update, ce, acc):
        return {"updates": update, "validation": {"cross_entropy": ce, "top1_accuracy": acc}}
    e = {"records": [row(0, 7, .0), row(100, 4, .3), row(200, 4, .3), row(300, 3, .4)]}
    g = {"records": [row(0, 7, .0), row(100, 5, .2), row(200, 4.1, .29)]}
    table = {step: {"tokens": step * 10} for step in (0, 100, 200, 300)}
    result = quality.compare_history(e, g, table)
    assert result["common_budget_updates"] == 200
    assert result["common_budget_training_tokens"] == 2000
    assert result["best_through_common_budget"]["G"]["minimum_validation_ce"]["updates"] == 100
    assert result["matched_updates"][-1]["comparison"]["delta_accuracy"] == pytest.approx(-.01)


def test_per_neuron_differences_direction_and_zero_initial_bias():
    original = {"brain.gain": torch.tensor([1., 2.]), "brain.rec_gain": torch.tensor([2., 3.]), "brain.bias": torch.zeros(2)}
    changed = {"brain.gain": torch.tensor([2., 4.]), "brain.rec_gain": torch.tensor([2., 3.]), "brain.bias": torch.tensor([0., 1.])}
    result = quality.neuron_differences(original, changed)
    assert result["brain.gain"]["mean_delta"] == 1.5
    assert result["brain.gain"]["rms_delta"] == pytest.approx((2.5) ** .5)
    assert result["brain.rec_gain"]["different_values"] == 0
    assert result["brain.bias"]["relative_l2_delta"] is None
    assert result["gain_times_rec_gain"]["maximum_absolute_delta"] == 6


def test_wrong_host_fails_before_mps_is_queried(monkeypatch):
    monkeypatch.setattr(quality.socket, "gethostname", lambda: "other-host.local")
    monkeypatch.setattr(quality.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(quality.torch.backends.mps, "is_available", lambda: pytest.fail("No local MPS call allowed"))
    with pytest.raises(ValueError, match="registered macm3"):
        quality.idle_host_guard()


def test_checkpoint_contents_are_checked_against_parameter_and_selector_hashes():
    initial = {"brain.gain": torch.tensor([1.]), "brain.rec_gain": torch.tensor([2.]), "brain.bias": torch.tensor([0.])}
    manifest = {"initial_parameter_audit": quality.trainer.parameter_audit(initial), "frozen_buffers_sha256": {"edge": "hash"}}
    parameters = {key: value.clone() for key, value in initial.items()}; parameters["brain.gain"] += .2
    cursor = {"updates": 100, "epoch": 0}
    metric = {"cross_entropy": 3., "top1_accuracy": .4}
    entry = {"cursor": cursor, "validation": metric, "frozen_buffers_sha256": {"edge": "hash"}}
    arm = {"manifest": manifest, "selectors": {"minimum_validation_ce": entry}, "latest": entry}
    saved = {"format_version": 1, "manifest": manifest, "cursor": cursor, "frozen_buffers_sha256": {"edge": "hash"},
             "model_audit": {"frozen_buffers_preserved": True, "frozen_buffers_sha256": {"edge": "hash"}},
             "parameters": parameters, "parameter_audit": quality.trainer.parameter_audit(parameters, manifest["initial_parameter_audit"]),
             "best": {**cursor, **metric}}
    quality.verify_saved(saved, arm, "minimum_validation_ce")
    quality.verify_saved(saved, arm)
    saved["parameters"]["brain.gain"] += 1
    with pytest.raises(ValueError, match="hashes"):
        quality.verify_saved(saved, arm, "minimum_validation_ce")


def test_actual_published_configs_allow_only_omitted_null_debug_flags():
    for label, rank in quality.ARMS.items():
        declared = json.loads((ROOT / 'experiments/configs' / (label + '.json')).read_text())['training']
        native = {**quality.COMMON_CONFIG, 'readout_rank': rank}
        assert quality.declared_training_matches(declared, native)
        assert quality.declared_training_matches({**declared, 'max_updates': None, 'eval_limit': None}, native)
        assert not quality.declared_training_matches({**declared, 'max_updates': 8}, native)
        assert not quality.declared_training_matches({**declared, 'unknown': None}, native)
        assert not quality.declared_training_matches({k: v for k, v in declared.items() if k != 'seed'}, native)


def test_identical_selected_updates_still_produce_two_distinct_receipt_checks():
    selectors = {key: {'cursor': {'updates': 16600}, 'sha256': str(i)} for i, key in enumerate(quality.SELECTORS)}
    rows = quality.selector_entries({'selectors': selectors})
    assert len(rows) == 2 and [key for key, entry in rows] == list(quality.SELECTORS)
    assert rows[0][1]['sha256'] != rows[1][1]['sha256']
    selectors.pop('maximum_validation_accuracy')
    with pytest.raises(ValueError, match='Both selector'):
        quality.selector_entries({'selectors': selectors})


@pytest.mark.parametrize('rank', [32, 64])
def test_checkpoint_rank_shapes_and_full_parameter_count_without_tensor_allocation(rank):
    # Meta tensors exercise shape/count checks without model construction or data allocation.
    shapes = {'lm_head.0.weight': torch.empty((rank, 49393), device='meta'),
              'lm_head.1.weight': torch.empty((1024, rank), device='meta'),
              'fixture.other': torch.empty(quality.expected_counts(rank)['total'] - quality.expected_counts(rank)['readout'], device='meta')}
    arm = {'manifest': {'config': {'readout_rank': rank}}}
    assert quality.checkpoint_shapes({'parameters': shapes}, arm)['lm_head.0.weight'] == [rank, 49393]
    shapes['lm_head.1.weight'] = torch.empty((1024, rank + 1), device='meta')
    with pytest.raises(ValueError, match='architecture'):
        quality.checkpoint_shapes({'parameters': shapes}, arm)


def test_binding_merge_rejects_conflicting_hash_instead_of_overwriting():
    assert quality.merged_bindings({'/path': 'sha'}, {'/path': 'sha', '/other': 'two'}) == {'/path': 'sha', '/other': 'two'}
    with pytest.raises(ValueError, match='Conflicting'):
        quality.merged_bindings({'/path': 'sha'}, {'/path': 'different'})


def campaign_pair_fixture(tmp_path, monkeypatch):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    monkeypatch.setattr(quality, 'ROOT', tmp_path)
    calls = {'preflight': [], 'smoke': [], 'accepted_g': []}
    arms, campaigns, paths = [], [], []
    for index, (label, rank) in enumerate(quality.ARMS.items()):
        controller = quality.hcontroller.g if rank == 64 else quality.hcontroller
        run = tmp_path / ('rank' + str(rank))
        directory = run / 'arms' / label
        directory.mkdir(parents=True)
        launch_id, pid, birth = 'fixture-' + label, 40 + index, 'fixture birth ' + str(index)
        native = {**quality.COMMON_CONFIG, 'readout_rank': rank}
        config_path = tmp_path / 'experiments/configs' / (label + '.json')
        branch = {'branch': 'exp/encoder32-readout' + str(rank) + '-fixed', 'commit': str(index) * 40,
                  'configuration': str(config_path.relative_to(tmp_path))}
        write(config_path, {'experiment_id': label, 'branch': branch['branch'], 'training_host': 'macm3',
              'trainer': 'scripts/train_connectorch.py', 'training': {k: v for k, v in native.items() if k not in ('max_updates', 'eval_limit')},
              'trainable_parameters': quality.expected_counts(rank)['total'], 'dataset_sha256': 'data', 'groups_sha256': 'groups'})
        branch['configuration_sha256'] = quality.trainer.file_hash(config_path)
        experiment_git = {'repository': 'https://github.com/fernando-neto-ai/fly-wordbrain', **branch}
        parity_path = run / 'parity.json'; write(parity_path, {'passed': True})
        manifest = {'root': str(tmp_path), 'output': str(run), 'launch_id': launch_id,
              'arms': [{'name': label, **controller.arm_config((label, 32, 'fixed')), 'parameter_counts': quality.expected_counts(rank)}],
              'planned_phase_updates': quality.hcontroller.PLANNED_UPDATES,
              'sources_sha256': {}, 'input_files_sha256': {}, 'baseline_files_sha256': {},
              'experiment_branches': {'repository': experiment_git['repository'], 'arms': {label: branch}},
              'experiment_configs': {label: {'path': str(config_path), 'sha256': branch['configuration_sha256']}},
              'data_sha256': 'data', 'groups_sha256': 'groups', 'baseline_frozen_buffers_sha256': {'fixed': 'graph'},
              'preflight_connectorch_source': {'revision': 'package'},
              'preflight_receipt': {'path': str(parity_path), 'sha256': quality.trainer.file_hash(parity_path)}}
        smoke = {'name': label, 'passed': True, 'updates': 8}
        write(run / 'smoke-results.json', {'passed': True, 'arms': [smoke]})
        smoke_dir = run / 'smokes' / label; smoke_dir.mkdir(parents=True)
        for filename in ('manifest.json', 'launch.json', 'metrics.jsonl', 'status.json', 'process-status.json',
                         'selected-checkpoints.json', 'initial-gradients.json', 'best.pt', 'best-accuracy.pt', 'latest.pt'):
            (smoke_dir / filename).write_text('fixture')
        def parity_check(parity, manifest):
            calls['preflight'].append(manifest['launch_id'])
        def smoke_check(manifest, arm, directory, smoke=False):
            assert smoke is True
            calls['smoke'].append(arm[0])
            return {'name': arm[0], 'passed': True, 'updates': 8}
        monkeypatch.setattr(controller, 'verify_preflight', parity_check)
        monkeypatch.setattr(controller, 'verify_arm', smoke_check)
        acceptance = {'training_host': 'macm3', 'planned_updates': sum(quality.hcontroller.PLANNED_UPDATES), 'launch_id': launch_id,
              'process_cessation': {'confirmed': True, 'processes': [{'pid': pid, 'birth': birth, 'alive': False}]}}
        arm = {'directory': directory, 'manifest': {'config': native, 'data_sha256': 'data', 'groups_sha256': 'groups',
              'frozen_buffers_sha256': {'fixed': 'graph'}, 'connectorch_source': {'revision': 'package'}},
              'accepted': acceptance, 'stop': {'launch_id': launch_id, 'sources_sha256': {}},
              'worker_refs': [{'pid': pid, 'birth': birth, 'alive': False}],
              'launch': {'campaign_launch_id': launch_id, 'sources_sha256': {}, 'experiment_git': experiment_git},
              'selectors': {key: {'path': str(directory / filename), 'sha256': key} for key, filename in quality.SELECTORS.items()}}
        arms.append(arm); campaigns.append(manifest); paths.append(run)
    baseline_summary = {'fixture': 'accepted G'}
    write(paths[0] / 'manifest.json', campaigns[0])
    arms[0]['accepted']['campaign_manifest_sha256'] = quality.trainer.file_hash(paths[0] / 'manifest.json')
    acceptance_path = arms[0]['directory'] / 'accepted-early-stop.json'
    write(acceptance_path, arms[0]['accepted'])
    campaigns[1].update(parent_campaign={'path': str(paths[0] / 'manifest.json'),
              'sha256': quality.trainer.file_hash(paths[0] / 'manifest.json'), 'launch_id': campaigns[0]['launch_id']},
              baseline_run=str(arms[0]['directory']), baseline_acceptance={'path': str(acceptance_path), 'sha256': quality.trainer.file_hash(acceptance_path)},
              baseline_selected_checkpoints=arms[0]['selectors'], baseline_summary=baseline_summary)
    for arm, manifest, path in zip(arms, campaigns, paths):
        write(path / 'manifest.json', manifest)
        digest = quality.trainer.file_hash(path / 'manifest.json')
        arm['accepted']['campaign_manifest_sha256'] = digest
        write(path / 'launch.json', {'launch_id': manifest['launch_id'], 'manifest_sha256': digest,
              'worker_pid': arm['worker_refs'][0]['pid'], 'process_birth': arm['worker_refs'][0]['birth']})
    def accepted_check(parent, parent_dir, receipt_path):
        calls['accepted_g'].append(str(receipt_path))
        assert parent == campaigns[0] and parent_dir == paths[0] and receipt_path == acceptance_path
        return baseline_summary, {str(receipt_path): quality.trainer.file_hash(receipt_path)}
    monkeypatch.setattr(quality.hcontroller, 'accepted_g', accepted_check)
    return arms, campaigns, paths, calls


def test_campaign_ancestry_delegates_native_g_verifier_and_rechecks_both_smokes(tmp_path, monkeypatch):
    arms, _, _, calls = campaign_pair_fixture(tmp_path, monkeypatch)
    result, bindings = quality.inspect_campaign_pair(*arms)
    assert result['accepted_g_verified_by_rank32_controller'] is True
    assert len(calls['accepted_g']) == 1 and len(calls['preflight']) == len(calls['smoke']) == 2
    assert len(bindings) >= 28


@pytest.mark.parametrize('damage', ['baseline_hash', 'baseline_selectors', 'parent_launch', 'branch', 'acceptance_launch', 'dispatcher_birth'])
def test_campaign_ancestry_rejects_wrong_native_bindings(tmp_path, monkeypatch, damage):
    arms, campaigns, paths, _ = campaign_pair_fixture(tmp_path, monkeypatch)
    if damage == 'baseline_hash': campaigns[1]['baseline_acceptance']['sha256'] = 'wrong'
    elif damage == 'baseline_selectors': campaigns[1]['baseline_selected_checkpoints'] = {}
    elif damage == 'parent_launch': campaigns[1]['parent_campaign']['launch_id'] = 'wrong'
    elif damage == 'branch': arms[1]['launch']['experiment_git']['branch'] = 'wrong'
    elif damage == 'acceptance_launch': arms[1]['accepted']['launch_id'] = 'wrong'
    else: arms[1]['worker_refs'][0]['birth'] = 'wrong'
    (paths[1] / 'manifest.json').write_text(json.dumps(campaigns[1]))
    digest = quality.trainer.file_hash(paths[1] / 'manifest.json')
    arms[1]['accepted']['campaign_manifest_sha256'] = digest
    launch = json.loads((paths[1] / 'launch.json').read_text()); launch['manifest_sha256'] = digest
    (paths[1] / 'launch.json').write_text(json.dumps(launch))
    with pytest.raises(ValueError):
        quality.inspect_campaign_pair(*arms)
