import hashlib
import json
from types import SimpleNamespace
import pytest

from fly_wordbrain.dashboard_sync import RemoteRunSync


def response(files):
    entries = {name: {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
               for name, text in files.items()}
    return SimpleNamespace(exit_code=0, stdout=json.dumps({"run": "/remote/run", "files": entries}), stderr="")


def test_verified_snapshot_refreshes_only_on_demand_and_is_throttled(tmp_path):
    calls = []
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return response({"validation.jsonl": '{"event":"validation_snapshot"}\n'})
    sync = RemoteRunSync("macm3", "/remote/run", tmp_path, runner=runner)
    assert not calls
    assert sync.refresh()
    assert not sync.refresh()
    assert len(calls) == 1
    assert calls[0][1] == {"host": "macm3", "timeout": 20}
    assert json.loads((tmp_path / "sync-status.json").read_text())["state"] == "ok"
    assert (tmp_path / "validation.jsonl").read_text().endswith("\n")


def test_failure_retains_last_good_results_and_marks_them_stale(tmp_path):
    sync = RemoteRunSync("macm3", "/remote/run", tmp_path,
                         runner=lambda *a, **k: response({"protocol.json": '{"epoch":1}'}))
    assert sync.refresh()
    before = json.loads((tmp_path / "sync-status.json").read_text())["last_success_utc"]
    sync.runner = lambda *a, **k: SimpleNamespace(exit_code=1, stdout="", stderr="Host unavailable")
    assert not sync.refresh(force=True)
    assert json.loads((tmp_path / "protocol.json").read_text()) == {"epoch": 1}
    status = json.loads((tmp_path / "sync-status.json").read_text())
    assert status["state"] == "error" and status["last_success_utc"] == before
    assert "unavailable" in status["error"]


def test_unexpected_filename_cannot_escape_cache_and_snapshot_is_validated_first(tmp_path):
    sync = RemoteRunSync("macm3", "/remote/run", tmp_path, runner=lambda *a, **k:
        response({"protocol.json": '{}', "../escape.json": '{}'}))
    assert not sync.refresh()
    assert not (tmp_path / "protocol.json").exists()
    assert not (tmp_path.parent / "escape.json").exists()


def test_corrupt_checksum_does_not_replace_existing_data(tmp_path):
    sync = RemoteRunSync("macm3", "/remote/run", tmp_path,
                         runner=lambda *a, **k: response({"events.jsonl": "old\n"}))
    assert sync.refresh()
    result = response({"events.jsonl": "new\n"})
    payload = json.loads(result.stdout)
    payload["files"]["events.jsonl"]["sha256"] = "0" * 64
    result.stdout = json.dumps(payload)
    sync.runner = lambda *a, **k: result
    assert not sync.refresh(force=True)
    assert (tmp_path / "events.jsonl").read_text() == "old\n"


def test_cache_cannot_be_rebound_to_another_run(tmp_path):
    RemoteRunSync("macm3", "/remote/one", tmp_path)
    with pytest.raises(ValueError, match="different remote run"):
        RemoteRunSync("macm3", "/remote/two", tmp_path)


def test_existing_unbound_training_artifacts_are_not_adopted(tmp_path):
    (tmp_path / "protocol.json").write_text('{}')
    with pytest.raises(ValueError, match="empty dedicated cache"):
        RemoteRunSync("macm3", "/remote/run", tmp_path)


def test_remote_removal_does_not_leave_obsolete_cached_artifacts(tmp_path):
    sync = RemoteRunSync("macm3", "/remote/run", tmp_path,
                         runner=lambda *a, **k: response({"events.jsonl": "old\n"}))
    assert sync.refresh()
    sync.runner = lambda *a, **k: response({"protocol.json": '{}'})
    assert sync.refresh(force=True)
    assert not (tmp_path / "events.jsonl").exists()
