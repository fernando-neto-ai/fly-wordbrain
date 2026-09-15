"""Action-monitor population, baseline, freshness, and exact-route contracts."""
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from fly_wordbrain.action_dashboard import DashboardServer, read_state, validation_csv


NOW = datetime(2026, 9, 15, 3, 0, tzinfo=timezone.utc)


def snapshot(step=0, scope="monitor_subset", **extra):
    return {"schema": 1, "kind": "action_selection", "event": "validation_snapshot",
            "arm": "frozen_action", "scope": scope, "global_step": step, "epoch": int(step > 0), "top_k": 5,
            "story_count": 128 if scope == "monitor_subset" else 1024, "target_count": 1000,
            "accuracy": .31, "baseline_accuracy": .31, "topk_coverage": .50,
            "conditional_accuracy": .62, "fixes": 0, "regressions": 0,
            "conditional_cross_entropy": 1.1, "known_word_accuracy": .29,
            "timestamp_utc": NOW.isoformat(), "dataset_sha256": "dataset-sha",
            "subset_sha256": scope + "-sha", "evaluation_method": "count_model_exact_zero_correction",
            **extra}


def write(root, name, value):
    (root / name).write_text(json.dumps(value))


def lines(root, name, values):
    (root / name).write_text("".join(json.dumps(row) + "\n" for row in values))


def test_baseline_then_learned_measurements_and_csv_use_same_population(tmp_path):
    zero = snapshot()
    learned = snapshot(64, accuracy=.34, conditional_accuracy=.68, fixes=45, regressions=15,
                       conditional_cross_entropy=1.0, evaluation_method="model_inference")
    full = snapshot(64, scope="full_validation")
    lines(tmp_path, "validation.jsonl", [zero, learned, full])
    lines(tmp_path, "events.jsonl", [zero, learned, full])
    state = read_state(tmp_path, now=NOW)
    assert state["errors"] == [] and state["status"] == "ready"
    assert len(state["points"]) == 3
    assert state["points"][0]["evaluation_method"] == "count_model_exact_zero_correction"
    assert state["points"][1]["accuracy"] == .34
    assert state["points"][1]["baseline_accuracy"] == .31
    assert state["points"][1]["topk_coverage"] == .50
    csv_rows = list(csv.DictReader(io.StringIO(validation_csv(state))))
    assert len(csv_rows) == len(state["points"])
    for csv_row, point in zip(csv_rows, state["points"]):
        for field, value in point.items():
            assert csv_row[field] == ("" if value is None else str(value))


def test_empty_calibration_is_waiting_without_fabricated_zero(tmp_path):
    write(tmp_path, "protocol.json", {"schema": 1, "kind": "action_selection", "dataset_sha256": "dataset-sha"})
    write(tmp_path, "process-status.json", {"status": "running"})
    lines(tmp_path, "events.jsonl", [{"schema": 1, "kind": "action_selection", "event": "phase",
          "phase": "calibration", "arm": "frozen_action", "timestamp_utc": NOW.isoformat()}])
    state = read_state(tmp_path, now=NOW)
    assert state["status"] == "waiting" and not state["errors"]
    assert state["points"] == [] and state["latest_event"]["phase"] == "calibration"
    assert len(list(csv.DictReader(io.StringIO(validation_csv(state))))) == 0


@pytest.mark.parametrize("change", [
    {"subset_sha256": "wrong-subset"}, {"dataset_sha256": "wrong-dataset"},
    {"story_count": 127}, {"target_count": 999},
    {"baseline_accuracy": .30, "accuracy": .30, "conditional_accuracy": .60},
    {"topk_coverage": .60, "conditional_accuracy": .31 / .60},
])
def test_mixed_subset_or_changing_baseline_is_explicit_error(tmp_path, change):
    lines(tmp_path, "validation.jsonl", [snapshot(), snapshot(64, **change)])
    state = read_state(tmp_path)
    assert state["status"] == "error" and len(state["points"]) == 1
    assert any("identity" in error or "baseline changed" in error for error in state["errors"])


def test_full_scope_can_have_different_coverage_but_not_dataset(tmp_path):
    lines(tmp_path, "validation.jsonl", [snapshot(), snapshot(scope="full_validation", topk_coverage=.6,
          conditional_accuracy=.31 / .6)])
    assert not read_state(tmp_path)["errors"]
    write(tmp_path, "protocol.json", {"dataset_sha256": "other"})
    assert "dataset identity" in " ".join(read_state(tmp_path)["errors"])


def test_torn_last_line_warns_but_corrupt_interior_line_errors(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(snapshot()) + '\n{"schema":')
    state = read_state(tmp_path)
    assert len(state["points"]) == 1 and not state["errors"]
    assert "incomplete final" in " ".join(state["warnings"])
    path.write_text('{"schema":\n' + json.dumps(snapshot()) + '\n')
    assert "line 1" in " ".join(read_state(tmp_path)["errors"])


@pytest.mark.parametrize("change", [
    {"scope": "test"}, {"arm": "frozen"}, {"accuracy": float("nan")},
    {"accuracy": .6}, {"conditional_accuracy": .7}, {"fixes": 1},
    {"conditional_cross_entropy": -1}, {"target_count": 1.5},
    {"dataset_sha256": None}, {"timestamp_utc": "invalid"},
])
def test_invalid_metrics_or_comparisons_cannot_be_plotted(tmp_path, change):
    lines(tmp_path, "validation.jsonl", [snapshot(**change)])
    state = read_state(tmp_path)
    assert state["points"] == [] and state["status"] == "error"


def test_zero_covered_and_no_known_targets_keep_null_metrics(tmp_path):
    lines(tmp_path, "validation.jsonl", [snapshot(accuracy=0., baseline_accuracy=0., topk_coverage=0.,
          conditional_accuracy=None, conditional_cross_entropy=None, known_word_accuracy=None)])
    state = read_state(tmp_path)
    assert not state["errors"] and state["points"][0]["conditional_accuracy"] is None
    assert state["points"][0]["conditional_cross_entropy"] is None
    assert state["points"][0]["known_word_accuracy"] is None


def test_progress_phase_and_failure_are_visible(tmp_path):
    lines(tmp_path, "events.jsonl", [
        {"event": "progress", "kind": "action_selection", "phase": "training", "arm": "frozen_action",
         "global_step": 64, "total_steps": 1024, "stories_completed": 512, "total_stories": 8192},
        {"event": "phase", "kind": "action_selection", "phase": "validation", "arm": "frozen_action",
         "global_step": 64, "scope": "monitor_subset"}])
    state = read_state(tmp_path)
    assert state["progress"]["stories_completed"] == 512
    assert state["latest_event"]["phase"] == "validation"
    write(tmp_path, "process-status.json", {"status": "failed", "exit_code": 1})
    state = read_state(tmp_path)
    assert state["latest_event"]["phase"] == "failed"
    assert "Training process failed (exit 1)" in state["errors"]


def test_freshness_uses_last_success_and_retains_real_points_on_failure(tmp_path):
    lines(tmp_path, "validation.jsonl", [snapshot()])
    write(tmp_path, "sync-status.json", {"state": "error", "error": "offline",
          "timestamp_utc": NOW.isoformat(), "last_success_utc": (NOW - timedelta(seconds=200)).isoformat()})
    state = read_state(tmp_path, now=NOW)
    assert state["freshness"]["stale"] and state["freshness"]["age_seconds"] == 200
    assert len(state["points"]) == 1 and "Remote sync: offline" in state["errors"]


def test_same_step_conflicts_not_silently_overwritten(tmp_path):
    lines(tmp_path, "events.jsonl", [snapshot()])
    lines(tmp_path, "validation.jsonl", [snapshot(conditional_cross_entropy=1.2)])
    state = read_state(tmp_path)
    assert len(state["points"]) == 1
    assert "conflicting repeated" in " ".join(state["errors"])


def test_validation_file_cannot_silently_hide_wrong_record_kind(tmp_path):
    lines(tmp_path, "validation.jsonl", [{"schema": 1, "kind": "wrong"}])
    state = read_state(tmp_path)
    assert state["points"] == [] and state["status"] == "error"
    assert "schema or kind" in " ".join(state["errors"])


def test_exact_http_routes_csv_and_refresh_failures(tmp_path):
    lines(tmp_path, "validation.jsonl", [snapshot()])
    calls = []
    server = DashboardServer(("127.0.0.1", 0), tmp_path, refresh_callback=lambda: calls.append(1))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        with urlopen(base + "/") as response:
            page = response.read().decode()
            assert "All-position accuracy" in page and "Top-K coverage" in page
            assert "setTimeout(refresh,15000)" in page
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        assert not calls
        with urlopen(base + "/api/state") as response:
            assert len(json.load(response)["points"]) == 1
        with urlopen(base + "/api/validation.csv") as response:
            assert response.headers.get_content_type() == "text/csv"
            assert list(csv.DictReader(io.StringIO(response.read().decode())))[0]["accuracy"] == "0.31"
        assert len(calls) == 2
        for route in ("/protocol.json", "/../protocol.json", "/api/state/../../protocol.json"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + route)
            assert error.value.code == 404
        def fail():
            raise RuntimeError("transport broke")
        server.refresh_callback = fail
        with urlopen(base + "/api/state") as response:
            state = json.load(response)
            assert len(state["points"]) == 1 and "transport broke" in " ".join(state["errors"])
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/validation.csv")
        assert error.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_top_ten_protocol_is_available_during_calibration_before_points(tmp_path):
    write(tmp_path, "protocol.json", {"schema": 1, "kind": "action_selection", "top_k": 10})
    lines(tmp_path, "events.jsonl", [{"event": "phase", "phase": "calibration", "arm": "frozen_action"}])
    state = read_state(tmp_path)
    assert not state["errors"] and state["points"] == []
    assert state["top_k"] == 10 and state["protocol"]["top_k"] == 10


def test_top_ten_results_use_generic_coverage_and_export_k(tmp_path):
    write(tmp_path, "protocol.json", {"top_k": 10})
    lines(tmp_path, "validation.jsonl", [snapshot(top_k=10, topk_coverage=.70,
                                               conditional_accuracy=.31 / .70)])
    state = read_state(tmp_path)
    assert not state["errors"] and state["top_k"] == 10
    assert state["points"][0]["topk_coverage"] == .70
    assert "top5_coverage" not in state["points"][0]
    row = list(csv.DictReader(io.StringIO(validation_csv(state))))[0]
    assert row["top_k"] == "10" and row["topk_coverage"] == "0.7"


@pytest.mark.parametrize("protocol_k", [None, 10])
def test_mixed_candidate_counts_within_run_are_rejected_across_scopes(tmp_path, protocol_k):
    if protocol_k is not None:
        write(tmp_path, "protocol.json", {"top_k": protocol_k})
    lines(tmp_path, "validation.jsonl", [snapshot(top_k=10), snapshot(64, scope="full_validation", top_k=5)])
    state = read_state(tmp_path)
    assert len(state["points"]) == 1 and state["top_k"] == 10
    assert "mixed candidate counts" in " ".join(state["errors"])


@pytest.mark.parametrize("top_k", [None, 0, -1, True, 10.0])
def test_snapshot_requires_explicit_positive_integer_k(tmp_path, top_k):
    lines(tmp_path, "validation.jsonl", [snapshot(top_k=top_k)])
    state = read_state(tmp_path)
    assert state["points"] == [] and "top_k" in " ".join(state["errors"])
