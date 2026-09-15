"""Eight-word population and paired fast-state-ablation dashboard contracts."""
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from fly_wordbrain.feedback_dashboard import DashboardServer, read_state, validation_csv

NOW = datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc)


def snapshot(step=0, scope="monitor_subset", **extra):
    return {"schema": 1, "kind": "feedback_action", "event": "validation_snapshot",
            "arm": "feedback_fast_weights", "scope": scope, "global_step": step,
            "epoch": int(step > 0), "top_k": 10, "window_words": 8, "observed_words": 7,
            "story_count": 128 if scope == "monitor_subset" else 256,
            "window_count": 1000, "target_count": 1000,
            "accuracy": .30, "baseline_accuracy": .30, "topk_coverage": .5,
            "conditional_accuracy": .6, "conditional_cross_entropy": 1.1,
            "known_word_accuracy": .3, "fixes": 0, "regressions": 0,
            "overrides": 0, "override_rate": 0.,
            "disabled_fast_accuracy": .30, "disabled_fast_conditional_cross_entropy": 1.1,
            "timestamp_utc": NOW.isoformat(), "dataset_sha256": "dataset-sha",
            "subset_sha256": scope + "-windows-sha", "evaluation_method": "model_inference",
            **extra}


def write(root, name, value):
    (root / name).write_text(json.dumps(value))


def lines(root, name, rows):
    (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))


def protocol(**extra):
    return {"schema": 1, "kind": "feedback_action", "top_k": 10, "window_words": 8,
            "observed_words": 7, "dataset_sha256": "dataset-sha", "monitor_windows": 1000,
            "validation_windows": 1000, "train_windows": 8192, **extra}


def test_window_population_ablation_metrics_and_csv(tmp_path):
    write(tmp_path, "protocol.json", protocol())
    baseline = snapshot(evaluation_method="count_model_exact_zero_correction")
    learned = snapshot(32, accuracy=.32, conditional_accuracy=.64, fixes=30, regressions=10,
                       overrides=100, override_rate=.1, disabled_fast_accuracy=.29,
                       disabled_fast_conditional_cross_entropy=1.2)
    full = snapshot(32, scope="full_validation")
    lines(tmp_path, "validation.jsonl", [baseline, learned, full])
    lines(tmp_path, "events.jsonl", [baseline, learned, full])
    state = read_state(tmp_path, now=NOW)
    assert state["errors"] == [] and state["status"] == "ready"
    assert len(state["points"]) == 3
    assert state["points"][1]["accuracy"] == .32
    assert state["points"][1]["disabled_fast_accuracy"] == .29
    assert state["points"][1]["window_count"] == state["points"][1]["target_count"] == 1000
    rows = list(csv.DictReader(io.StringIO(validation_csv(state))))
    for row, point in zip(rows, state["points"]):
        for key, value in point.items():
            assert row[key] == ("" if value is None else str(value))
    assert rows[0]["observed_words"] == "7" and rows[0]["window_words"] == "8"


@pytest.mark.parametrize("change,needle", [
    ({"kind": "action_selection"}, "schema, kind or arm"),
    ({"arm": "frozen_action"}, "schema, kind or arm"),
    ({"window_words": 128}, "window_words"),
    ({"observed_words": 8}, "observed_words"),
    ({"top_k": 5}, "top_k"),
    ({"window_count": 999}, "one prediction per window"),
    ({"story_count": 1001}, "story_count"),
    ({"scope": "test"}, "scope"),
    ({"overrides": 100, "override_rate": .01}, "override_rate"),
    ({"fixes": 20, "regressions": 20, "overrides": 10, "override_rate": .01}, "cannot exceed overrides"),
    ({"disabled_fast_accuracy": .6}, "coverage"),
    ({"disabled_fast_accuracy": .2, "disabled_fast_conditional_cross_entropy": None}, "loss is missing"),
    ({"disabled_fast_accuracy": None, "disabled_fast_conditional_cross_entropy": 1}, "no paired accuracy"),
    ({"disabled_fast_conditional_cross_entropy": -1}, "valid range"),
])
def test_invalid_population_or_ablation_errors(tmp_path, change, needle):
    lines(tmp_path, "validation.jsonl", [snapshot(**change)])
    state = read_state(tmp_path)
    assert state["points"] == [] and state["status"] == "error"
    assert needle in " ".join(state["errors"])


def test_optional_ablation_missing_is_not_a_fake_zero(tmp_path):
    row = snapshot()
    del row["disabled_fast_accuracy"], row["disabled_fast_conditional_cross_entropy"]
    lines(tmp_path, "validation.jsonl", [row])
    state = read_state(tmp_path)
    assert not state["errors"]
    assert state["points"][0]["disabled_fast_accuracy"] is None
    assert state["points"][0]["disabled_fast_conditional_cross_entropy"] is None


def test_zero_coverage_keeps_conditional_loss_null(tmp_path):
    lines(tmp_path, "validation.jsonl", [snapshot(accuracy=0., baseline_accuracy=0., topk_coverage=0.,
          conditional_accuracy=None, conditional_cross_entropy=None, disabled_fast_accuracy=0.,
          disabled_fast_conditional_cross_entropy=None)])
    state = read_state(tmp_path)
    assert not state["errors"]
    assert state["points"][0]["conditional_cross_entropy"] is None


@pytest.mark.parametrize("change", [{"window_count": 2000, "target_count": 2000},
    {"subset_sha256": "other"}, {"dataset_sha256": "other"},
    {"accuracy": .2, "baseline_accuracy": .2, "conditional_accuracy": .4}])
def test_changed_population_or_baseline_cannot_share_a_curve(tmp_path, change):
    lines(tmp_path, "validation.jsonl", [snapshot(), snapshot(32, **change)])
    state = read_state(tmp_path)
    assert len(state["points"]) == 1 and state["errors"]


def test_old_protocol_rejected_and_configured_window_count_checked(tmp_path):
    write(tmp_path, "protocol.json", protocol(kind="action_selection"))
    assert "different experiment" in " ".join(read_state(tmp_path)["errors"])
    write(tmp_path, "protocol.json", protocol(monitor_windows=512))
    lines(tmp_path, "validation.jsonl", [snapshot()])
    state = read_state(tmp_path)
    assert state["points"] == [] and "declared validation population" in " ".join(state["errors"])


def test_atomic_progress_and_newer_event_phase(tmp_path):
    write(tmp_path, "protocol.json", protocol(model={"neurons": 166700, "edges": 25582938,
          "candidate_edges": 347, "trainable_parameters": 2590}))
    progress = {"status": "running", "phase": "training", "global_step": 32, "total_steps": 1024,
                "windows_completed": 256, "total_windows": 8192, "rule_gradient_norm": .12,
                "fast_effect": .001, "timestamp_utc": NOW.isoformat()}
    write(tmp_path, "progress.json", progress)
    state = read_state(tmp_path)
    assert state["status"] == "waiting" and state["points"] == []
    assert state["latest_event"]["phase"] == "training"
    assert state["progress"]["windows_completed"] == 256
    assert state["protocol"]["model"]["candidate_edges"] == 347
    lines(tmp_path, "events.jsonl", [{"event": "phase", "phase": "validation", "arm": "feedback_fast_weights",
          "global_step": 32, "timestamp_utc": (NOW + timedelta(seconds=1)).isoformat()}])
    assert read_state(tmp_path)["latest_event"]["phase"] == "validation"
    write(tmp_path, "process-status.json", {"status": "failed", "exit_code": 1})
    state = read_state(tmp_path)
    assert state["latest_event"]["phase"] == "failed"
    assert "Training process failed (exit 1)" in state["errors"]


def test_incomplete_append_stale_sync_and_conflicting_duplicate(tmp_path):
    row = snapshot()
    (tmp_path / "validation.jsonl").write_text(json.dumps(row) + '\n{"schema":')
    state = read_state(tmp_path)
    assert len(state["points"]) == 1 and state["warnings"] and not state["errors"]
    write(tmp_path, "sync-status.json", {"state": "error", "error": "offline",
          "timestamp_utc": NOW.isoformat(), "last_success_utc": (NOW - timedelta(seconds=200)).isoformat()})
    state = read_state(tmp_path, now=NOW)
    assert state["freshness"]["stale"] and state["freshness"]["age_seconds"] == 200
    assert len(state["points"]) == 1 and "Remote sync: offline" in state["errors"]
    lines(tmp_path, "events.jsonl", [snapshot(disabled_fast_accuracy=.2)])
    assert "conflicting repeated" in " ".join(read_state(tmp_path)["errors"])


def test_phase_progress_retains_latest_measured_diagnostics(tmp_path):
    lines(tmp_path, "events.jsonl", [
        {"event": "progress", "arm": "feedback_fast_weights", "rule_gradient_norm": .1,
         "fast_effect": .002},
        {"event": "progress", "arm": "feedback_fast_weights", "rule_gradient_norm": 0.},
        {"event": "progress", "arm": "feedback_fast_weights", "fast_effect": None},
        {"event": "progress", "arm": "other_arm", "rule_gradient_norm": 99., "fast_effect": 99.},
    ])
    current = {"phase": "validation", "global_step": 32, "timestamp_utc": NOW.isoformat()}
    write(tmp_path, "progress.json", current)
    state = read_state(tmp_path)
    assert state["progress"]["rule_gradient_norm"] == 0.
    assert state["progress"]["fast_effect"] == .002
    assert state["latest_event"]["phase"] == "validation"
    assert state["progress"]["global_step"] == 32
    write(tmp_path, "progress.json", {**current, "rule_gradient_norm": None, "fast_effect": .003})
    state = read_state(tmp_path)
    assert state["progress"]["rule_gradient_norm"] is None
    assert state["progress"]["fast_effect"] == .003


def test_http_exact_routes_page_semantics_and_cache_failure(tmp_path):
    lines(tmp_path, "validation.jsonl", [snapshot()])
    calls = []
    server = DashboardServer(("127.0.0.1", 0), tmp_path, refresh_callback=lambda: calls.append(1))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        with urlopen(base + "/") as response:
            page = response.read().decode()
            assert "Seven words in. Predict the eighth." in page
            assert "one prediction" in page.lower()
            assert "not a separately trained control" in page
            assert "same trained head" in page
            assert "Show evaluation comparison (fast off)" in page
            assert "The checkbox changes chart visibility only" in page
            assert "Trained model — fast on" in page
            assert "Fast off (evaluation only)" in page
            assert "not directly comparable" in page
            assert "setTimeout(refresh,15000)" in page
            assert "cdn." not in page
        with urlopen(base + "/api/state") as response:
            assert json.load(response)["kind"] == "feedback_action"
        with urlopen(base + "/api/validation.csv") as response:
            assert "disabled_fast_accuracy" in response.read().decode()
        assert len(calls) == 2
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/../protocol.json")
        assert error.value.code == 404
        def fail():
            raise RuntimeError("offline")
        server.refresh_callback = fail
        with urlopen(base + "/api/state") as response:
            state = json.load(response)
            assert len(state["points"]) == 1 and state["errors"]
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/validation.csv")
        assert error.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
