"""Read-only dashboard for seven observed words followed by one eighth-word choice.

This window population is distinct from the all-story action experiment.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from .action_dashboard import _snapshot as _action_snapshot, _integer
from .dashboard import DashboardHandler as _BaseHandler, _number, _read, _timestamp

KIND = "feedback_action"
ARM = "feedback_fast_weights"
SCOPES = ("monitor_subset", "full_validation")
CSV_FIELDS = ("arm", "scope", "epoch", "global_step", "top_k", "window_words", "observed_words",
              "story_count", "window_count", "target_count", "accuracy", "baseline_accuracy",
              "topk_coverage", "conditional_accuracy", "conditional_cross_entropy", "known_word_accuracy",
              "fixes", "regressions", "overrides", "override_rate", "disabled_fast_accuracy",
              "disabled_fast_conditional_cross_entropy", "fast_centered_logit_rms", "fast_centered_logit_max",
              "fast_mean_total_variation", "fast_changed_predictions", "fast_changed_prediction_rate",
              "fast_max_logit_difference", "evaluation_method", "dataset_sha256",
              "subset_sha256", "timestamp_utc")


def _snapshot(row):
    if row.get("schema") != 1 or row.get("kind") != KIND or row.get("arm") != ARM:
        raise ValueError("unknown feedback validation schema, kind or arm")
    for key, expected in (("top_k", 10), ("window_words", 8), ("observed_words", 7)):
        if type(row.get(key)) is not int or row[key] != expected:
            raise ValueError("%s must be %d for this experiment" % (key, expected))
    # Reuse the existing count-action metric identities, without changing its contract.
    validated = _action_snapshot({**row, "kind": "action_selection", "arm": "frozen_action"})
    point = {key: validated.get(key) for key in CSV_FIELDS}
    point.update(arm=ARM, window_words=8, observed_words=7)
    point["window_count"] = _integer(row.get("window_count"), "window_count", lower=1)
    if point["window_count"] != point["target_count"]:
        raise ValueError("one prediction per window requires window_count == target_count")
    if point["story_count"] > point["window_count"]:
        raise ValueError("story_count cannot exceed the evaluated window count")
    overrides = _integer(row.get("overrides"), "overrides")
    override_rate = _number(row.get("override_rate"), "override_rate", lower=0, upper=1)
    if overrides > point["target_count"] or not math.isclose(
            overrides / point["target_count"], override_rate, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("override_rate does not match override count")
    if overrides < point["fixes"] + point["regressions"]:
        raise ValueError("fixes and regressions cannot exceed overrides")
    point.update(overrides=overrides, override_rate=override_rate)
    accuracy, ce = row.get("disabled_fast_accuracy"), row.get("disabled_fast_conditional_cross_entropy")
    if accuracy is not None:
        accuracy = _number(accuracy, "disabled_fast_accuracy", lower=0, upper=1)
        if accuracy > point["topk_coverage"] + 1e-10:
            raise ValueError("disabled-fast accuracy exceeds candidate coverage")
        if point["topk_coverage"] > 0 and ce is None:
            raise ValueError("disabled-fast conditional loss is missing despite covered targets")
    elif ce is not None:
        raise ValueError("disabled-fast conditional loss has no paired accuracy")
    if ce is not None:
        ce = _number(ce, "disabled_fast_conditional_cross_entropy", lower=0)
        if point["topk_coverage"] == 0:
            raise ValueError("disabled-fast conditional loss must be null for zero coverage")
    point.update(disabled_fast_accuracy=accuracy, disabled_fast_conditional_cross_entropy=ce)
    for key in ("fast_centered_logit_rms", "fast_centered_logit_max", "fast_max_logit_difference"):
        if row.get(key) is not None:
            point[key] = _number(row[key], key, lower=0)
    if row.get("fast_mean_total_variation") is not None:
        point["fast_mean_total_variation"] = _number(
            row["fast_mean_total_variation"], "fast_mean_total_variation", lower=0, upper=1)
    if (point["fast_centered_logit_rms"] is not None and point["fast_centered_logit_max"] is not None
            and point["fast_centered_logit_rms"] > point["fast_centered_logit_max"] + 1e-10):
        raise ValueError("fast centered RMS exceeds its maximum difference")
    changed = row.get("fast_changed_predictions")
    rate = row.get("fast_changed_prediction_rate")
    if changed is not None:
        changed = _integer(changed, "fast_changed_predictions")
        if changed > point["target_count"]:
            raise ValueError("fast_changed_predictions exceeds target_count")
        expected_rate = changed / point["target_count"]
        if rate is not None:
            rate = _number(rate, "fast_changed_prediction_rate", lower=0, upper=1)
            if not math.isclose(rate, expected_rate, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("fast changed-prediction rate does not match its count")
        if accuracy is not None and abs(point["accuracy"] - accuracy) > expected_rate + 1e-10:
            raise ValueError("fast accuracy change exceeds changed-prediction count")
        point.update(fast_changed_predictions=changed, fast_changed_prediction_rate=expected_rate)
    elif rate is not None:
        raise ValueError("fast_changed_prediction_rate requires its changed-prediction count")
    return point


def read_state(run, *, now=None, stale_after=90):
    root = Path(run).resolve()
    now = now or datetime.now(timezone.utc)
    errors, warnings, objects = [], [], {}
    for name in ("protocol", "progress", "process-status", "sync-status"):
        value = _read(root / (name + ".json"), root, errors, warnings)
        if value is not None and not isinstance(value, dict):
            errors.append(name + ".json must be an object")
            value = None
        objects[name] = value or {}
    protocol = objects["protocol"]
    if protocol:
        if protocol.get("kind") != KIND:
            errors.append("protocol kind must be feedback_action; all-story runs are a different experiment")
        for key, expected in (("top_k", 10), ("window_words", 8), ("observed_words", 7)):
            if type(protocol.get(key)) is not int or protocol[key] != expected:
                errors.append("protocol %s must be %d" % (key, expected))
    events = _read(root / "events.jsonl", root, errors, warnings, jsonl=True)
    validations = _read(root / "validation.jsonl", root, errors, warnings, jsonl=True)
    points, seen, identities = [], {}, {}
    dataset_identity = protocol.get("dataset_sha256")
    for row, required in [(row, False) for row in events] + [(row, True) for row in validations]:
        if not required and row.get("event") != "validation_snapshot" and not (
                "scope" in row and "accuracy" in row):
            continue
        try:
            point = _snapshot(row)
            key = (point["scope"], point["epoch"], point["global_step"])
            if key in seen:
                if seen[key] != point:
                    raise ValueError("conflicting repeated snapshot " + str(key))
                continue
            if dataset_identity is not None and dataset_identity != point["dataset_sha256"]:
                raise ValueError("dataset identity differs from the run")
            identity = tuple(point[key] for key in (
                "dataset_sha256", "subset_sha256", "story_count", "window_count", "target_count",
                "baseline_accuracy", "topk_coverage"))
            scope = point["scope"]
            if scope in identities and identities[scope] != identity:
                raise ValueError("window population or fixed count-model baseline changed: " + scope)
            expected_windows = protocol.get("monitor_windows" if scope == "monitor_subset" else "validation_windows")
            if expected_windows is not None and point["window_count"] != expected_windows:
                raise ValueError("window_count differs from the declared validation population")
            identities[scope], dataset_identity, seen[key] = identity, point["dataset_sha256"], point
            points.append(point)
        except (ValueError, TypeError, AttributeError, OverflowError) as error:
            errors.append("feedback validation: " + str(error))
    points.sort(key=lambda p: (p["global_step"], SCOPES.index(p["scope"]), p["epoch"]))
    progress = objects["progress"]
    if not progress:
        for row in events:
            if row.get("arm") == ARM and row.get("event") in ("progress", "training_progress"):
                progress = row
    progress = dict(progress)
    # Phase-only updates omit diagnostics; retain the latest actual measurement.
    # An explicit current value, including null, takes precedence over history.
    for key in ("rule_gradient_norm", "fast_effect"):
        if key not in progress:
            for row in reversed(events):
                value = row.get(key)
                if (row.get("arm") == ARM and not isinstance(value, bool)
                        and isinstance(value, (int, float)) and math.isfinite(value)):
                    progress[key] = value
                    break
    latest = events[-1] if events else {}
    # progress.json is atomic; compare source times so a newer phase event wins.
    event_time = _timestamp(latest.get("timestamp_utc"))
    progress_time = _timestamp(progress.get("timestamp_utc"))
    if progress and (not latest or progress_time and (not event_time or progress_time >= event_time)):
        latest = progress
    sync, process = objects["sync-status"], objects["process-status"]
    if sync.get("error"):
        errors.append("Remote sync: " + str(sync["error"]))
    status = str(process.get("status", process.get("state", progress.get("status", "")))).lower()
    exit_code = process.get("exit_code", process.get("returncode"))
    failed = status in ("failed", "error") or exit_code is not None and exit_code != 0
    if failed:
        errors.append("Training process failed" + (" (exit %s)" % exit_code if exit_code is not None else ""))
    if sync:
        fresh_at = _timestamp(sync.get("last_success_utc"))
    else:
        paths = [root / name for name in ("events.jsonl", "validation.jsonl", "progress.json", "process-status.json", "protocol.json")]
        mtimes = [datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                  for path in paths if path.is_file() and not path.is_symlink()]
        fresh_at = max(mtimes, default=None)
    age = max(0., (now - fresh_at).total_seconds()) if fresh_at else None
    last_snapshot = max((_timestamp(p["timestamp_utc"]) for p in points), default=None)
    phase = "failed" if failed else "completed" if status in ("completed", "finished") else latest.get("phase")
    return {"schema": 1, "kind": KIND, "arm": ARM, "run": root.name,
            "top_k": 10, "window_words": 8, "observed_words": 7,
            "generated_at": now.isoformat(), "status": "error" if errors else "ready" if points else "waiting",
            "points": points, "errors": list(dict.fromkeys(errors)), "warnings": warnings,
            "latest_event": {**{key: latest.get(key) for key in (
                "event", "scope", "arm", "epoch", "global_step", "total_steps", "timestamp_utc")}, "phase": phase},
            "progress": progress, "process": process, "protocol": protocol, "sync": sync,
            "freshness": {"last_success_utc": fresh_at.isoformat() if fresh_at else None,
                          "age_seconds": age, "stale": age is None or age > stale_after,
                          "last_snapshot_utc": last_snapshot.isoformat() if last_snapshot else None}}


def validation_csv(state):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(state["points"])
    return stream.getvalue()


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, run, *, refresh_callback=None, stale_after=90):
        self.run = Path(run).resolve()
        self.refresh_callback, self.stale_after = refresh_callback, stale_after
        self.page = Path(__file__).with_suffix(".html").read_bytes()
        super().__init__(address, DashboardHandler)

    def state(self):
        failure = None
        if self.refresh_callback is not None:
            try:
                self.refresh_callback()
            except Exception as error:
                failure = "Remote refresh failed: " + str(error)
        state = read_state(self.run, stale_after=self.stale_after)
        if failure:
            state["errors"].append(failure)
            state["status"] = "error"
        return state


class DashboardHandler(_BaseHandler):
    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/":
            self._send(self.server.page, "text/html; charset=utf-8")
        elif route in ("/api/state", "/api/validation.csv"):
            state = self.server.state()
            if route.endswith(".csv") and not state["errors"]:
                self._send(validation_csv(state).encode(), "text/csv; charset=utf-8", download=True)
            else:
                self._send(json.dumps(state, allow_nan=False).encode(), "application/json; charset=utf-8",
                           status=409 if route.endswith(".csv") else 200)
        else:
            self._send(b"Not found\n", "text/plain; charset=utf-8", status=404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--stale-after", type=float, default=90)
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-run")
    args = parser.parse_args()
    if bool(args.remote_host) != bool(args.remote_run):
        parser.error("--remote-host and --remote-run must be provided together")
    if args.stale_after <= 0:
        parser.error("--stale-after must be positive")
    callback = None
    if args.remote_host:
        from .dashboard_sync import RemoteRunSync
        refresher = RemoteRunSync(args.remote_host, args.remote_run, args.run, minimum_interval=15)
        callback = refresher.refresh
    server = DashboardServer((args.host, args.port), args.run, refresh_callback=callback,
                             stale_after=args.stale_after)
    print("Eight-word feedback dashboard: http://%s:%d/ (run %s)" % (args.host, server.server_port, args.run), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
