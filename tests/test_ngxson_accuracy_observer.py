"""CPU fixtures exercise retention races and provenance, without any training."""
import copy
from contextlib import closing
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest
import torch

spec = importlib.util.spec_from_file_location("accuracy_observer", Path(__file__).parents[1] / "scripts/watch_ngxson_accuracy.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


@pytest.fixture
def fixture(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"validation": [{"id": "one", "ids": [1, 3, 4, 5, 2]}]}))
    source = tmp_path / "trainer.py"
    source.write_text("fixture, not executed")
    parameter = torch.tensor([1., 2., 3.])
    digest = observer.hashlib.sha256(parameter.numpy().tobytes()).hexdigest()
    audit = {"value": {"sha256": digest, "finite": True, "values": 3}}
    manifest = {"reference_repository": "fixture", "reference_revision": "abc", "data_path": str(data),
                "data_sha256": observer.file_hash(data), "config": {"eval_limit": None},
                "source_sha256": {"trainer": observer.file_hash(source)},
                "trainer_sources_sha256": {"trainer.py": observer.file_hash(source)},
                "frozen_buffers_sha256": {"graph": "frozen"}, "initial_parameter_audit": audit,
                "initialization": {"trainable_parameters": 3}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    checkpoint = {"format_version": 1, "manifest": manifest, "frozen_buffers_sha256": {"graph": "frozen"},
                  "parameter_audit": audit, "parameters": {"value": parameter}, "cursor": {"updates": 1, "epoch": 0}}

    def add(updates, correct, name="latest.pt", epoch=0, save=True, ce=2.):
        row = {"event": "validation", "updates": updates, "epoch": epoch,
               "validation": {"correct": correct, "tokens": 4, "stories": 1,
                              "top1_accuracy": correct / 4, "cross_entropy": ce}}
        with (run / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        saved = copy.deepcopy(checkpoint)
        saved["cursor"] = {"updates": updates, "epoch": epoch}
        if save:
            temp = run / (name + ".tmp")
            torch.save(saved, temp)
            temp.replace(run / name)
        return row, saved

    return run, tmp_path / "retained", tmp_path, add, checkpoint


def make(fixture, **kwargs):
    run, output, root, _, _ = fixture
    return observer.AccuracyObserver(run, output, source_root=root, **kwargs)


def test_winner_survives_atomic_latest_replacement_and_ce_best_is_untouched(fixture):
    run, output, _, add, _ = fixture
    add(1, 1, name="best.pt", ce=1.)
    add(2, 2)
    ce_hash = observer.file_hash(run / "best.pt")
    monitor = make(fixture)
    first = monitor.scan()
    retained = first["best_available_checkpoint_sha256"]
    add(3, 1)
    value = monitor.scan()
    assert value["best_available_validation_record"]["updates"] == 2
    assert observer.file_hash(output / "best_accuracy.pt") == retained
    assert observer.file_hash(run / "best.pt") == ce_hash
    assert value["historical_best_weights_available"]
    assert not list(output.glob(".capture*"))


def test_missing_historical_maximum_is_reported_and_seed_recovers_it(fixture):
    run, _, _, add, _ = fixture
    highest, _ = add(1, 3, name="seed.pt")
    add(2, 2)
    monitor = make(fixture)
    value = monitor.scan()
    assert value["historical_best_validation_record"] == highest
    assert not value["historical_best_weights_available"]
    assert value["unavailable_higher_accuracy_records"] == [highest]
    monitor.seeds.append(run / "seed.pt")
    value = monitor.scan()
    assert value["historical_best_weights_available"]
    assert value["best_available_validation_record"] == highest


def test_equal_accuracy_keeps_earliest_available_checkpoint_across_epoch_duplicate(fixture):
    _, _, _, add, _ = fixture
    add(5, 2)
    monitor = make(fixture)
    first = monitor.scan()
    add(5, 2, epoch=1)
    value = monitor.scan()
    assert value["best_available_checkpoint_sha256"] == first["best_available_checkpoint_sha256"]
    assert value["best_available_validation_record"]["epoch"] == 0


def test_checkpoint_append_racing_initial_metric_read_is_matched(fixture, monkeypatch):
    run, _, _, add, _ = fixture
    add(1, 1)
    monitor = make(fixture)
    original = observer.os.link
    changed = False

    def racing_link(source, dest):
        nonlocal changed
        if source == run / "latest.pt" and not changed:
            changed = True
            add(2, 2)
        return original(source, dest)

    monkeypatch.setattr(observer.os, "link", racing_link)
    value = monitor.scan()
    assert value["best_available_validation_record"]["updates"] == 2


@pytest.mark.parametrize("mutation", ["hash", "population", "parameter", "nonfinite", "no_record"])
def test_invalid_candidate_is_rejected_without_retained_file(fixture, mutation):
    run, output, _, add, checkpoint = fixture
    row, saved = add(1, 1)
    monitor = make(fixture)
    if mutation == "hash":
        saved["manifest"]["data_sha256"] = "wrong"
    elif mutation == "population":
        row["validation"]["tokens"] = 5
        (run / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    elif mutation == "parameter":
        saved["parameters"]["value"][0] = 7.
    elif mutation == "nonfinite":
        saved["parameters"]["value"][0] = float("nan")
    else:
        saved["cursor"]["updates"] = 99
    torch.save(saved, run / "latest.pt")
    with pytest.raises(ValueError):
        monitor.scan()
    assert not (output / "best_accuracy.pt").exists()
    assert not list(output.glob(".capture*"))


def test_partial_append_ignored_but_malformed_complete_row_fails(fixture):
    run, _, _, add, _ = fixture
    add(1, 1)
    with (run / "metrics.jsonl").open("a") as handle:
        handle.write('{"event":')
    assert make(fixture).scan()["completed_validation_records"] == 1
    with (run / "metrics.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(json.JSONDecodeError):
        make(fixture).scan()


def test_restart_recovers_retained_inode_when_selection_json_is_stale(fixture):
    _, output, _, add, _ = fixture
    add(1, 3)
    original = make(fixture).scan()
    (output / "selection.json").write_text("{}")
    add(2, 1)
    value = make(fixture).scan()
    assert value["best_available_checkpoint_sha256"] == original["best_available_checkpoint_sha256"]
    assert value["best_available_validation_record"]["updates"] == 1


@pytest.mark.skipif(not hasattr(select, "kqueue"), reason="Native macOS/BSD kqueue required")
def test_native_kqueue_startup_new_checkpoint_and_terminal_exit(fixture):
    """Run the real CLI observer; all checkpoints are tiny, static CPU fixtures."""
    run, output, _, add, checkpoint = fixture
    # Fixture source is deliberately not executable and need not be installed at
    # the real CLI's source root. The other provenance bindings remain checked.
    checkpoint["manifest"]["trainer_sources_sha256"] = {}
    (run / "manifest.json").write_text(json.dumps(checkpoint["manifest"]))
    add(1, 1)
    output.mkdir()
    fd = os.open(output, os.O_RDONLY)
    process = None
    try:
        with closing(select.kqueue()) as queue:
            event = select.kevent(fd, filter=select.KQ_FILTER_VNODE,
                                 flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                 fflags=select.KQ_NOTE_WRITE)
            queue.control([event], 0, 0)
            process = subprocess.Popen([
                sys.executable, str(Path(observer.__file__).resolve()),
                "--run", str(run), "--output", str(output), "--threads", "1",
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            def await_json(name, predicate, timeout=20):
                deadline = time.monotonic() + timeout
                path = output / name
                while True:
                    if path.exists():
                        value = json.loads(path.read_text())
                        if predicate(value):
                            return value
                    if process.poll() is not None:
                        stdout, stderr = process.communicate(timeout=5)
                        pytest.fail(f"Observer exited early ({process.returncode}): {stdout}\n{stderr}")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        pytest.fail(f"Timed out awaiting {name}")
                    # Genuine directory notifications, no sleep/poll loop. A
                    # finite deadline also bounds a missed-event regression.
                    queue.control(None, 8, remaining)

            ready = await_json("status.json", lambda x: x.get("status") == "running")
            assert ready["training_run"] is False and ready["device"] == "cpu"
            initial = json.loads((output / "selection.json").read_text())
            assert initial["best_available_validation_record"]["updates"] == 1
            add(2, 3)
            selected = await_json("selection.json", lambda x:
                                  x["best_available_validation_record"]["updates"] == 2)
            assert selected["historical_best_weights_available"]
            observer.atomic_json(run / "status.json", {"status": "completed"})
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, f"{stdout}\n{stderr}"
            status = json.loads((output / "status.json").read_text())
            assert status["status"] == "completed"
            assert status["trainer_terminal_status"]["status"] == "completed"
            retained = torch.load(output / "best_accuracy.pt", weights_only=False, map_location="cpu")
            assert retained["cursor"]["updates"] == 2
    finally:
        os.close(fd)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)
            else:
                process.communicate(timeout=5)
