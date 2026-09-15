"""Read-only live validation dashboard for the configurable action word selector."""
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

from .dashboard import DashboardHandler as _BaseHandler, _number, _read, _timestamp


SCOPES = ("monitor_subset", "full_validation")
CSV_FIELDS = ("arm", "scope", "epoch", "global_step", "top_k", "story_count", "target_count",
              "accuracy", "baseline_accuracy", "topk_coverage", "conditional_accuracy",
              "fixes", "regressions", "conditional_cross_entropy", "known_word_accuracy",
              "evaluation_method", "dataset_sha256", "subset_sha256", "timestamp_utc")


def _integer(value, name, lower=0):
    value = _number(value, name, lower=lower)
    if int(value) != value:
        raise ValueError(name + " must be an integer")
    return int(value)


def _snapshot(row):
    if row.get("schema") != 1 or row.get("kind") != "action_selection":
        raise ValueError("unknown action validation schema or kind")
    if row.get("arm") != "frozen_action" or row.get("scope") not in SCOPES:
        raise ValueError("unknown action validation arm or scope")
    point = {key: row.get(key) for key in CSV_FIELDS}
    if type(row.get("top_k")) is not int or row["top_k"] < 1:
        raise ValueError("top_k must be a positive integer")
    for key in ("global_step", "epoch", "fixes", "regressions"):
        point[key] = _integer(row.get(key), key)
    for key in ("story_count", "target_count"):
        point[key] = _integer(row.get(key), key, lower=1)
    for key in ("accuracy", "baseline_accuracy", "topk_coverage"):
        point[key] = _number(row.get(key), key, lower=0, upper=1)
    for key in ("conditional_accuracy", "known_word_accuracy"):
        if row.get(key) is not None:
            point[key] = _number(row[key], key, lower=0, upper=1)
    if row.get("conditional_cross_entropy") is not None:
        point["conditional_cross_entropy"] = _number(row["conditional_cross_entropy"],
                                                     "conditional_cross_entropy", lower=0)
    for key in ("dataset_sha256", "subset_sha256"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(key + " must identify the evaluated data")
    if _timestamp(row.get("timestamp_utc")) is None:
        raise ValueError("timestamp_utc must identify the measurement time")
    if max(point["accuracy"], point["baseline_accuracy"]) > point["topk_coverage"] + 1e-10:
        raise ValueError("accuracy exceeds the top-K coverage ceiling")
    if point["topk_coverage"] == 0:
        if point["conditional_accuracy"] is not None or point["conditional_cross_entropy"] is not None:
            raise ValueError("conditional metrics must be null when no target is covered")
    elif (point["conditional_accuracy"] is None or point["conditional_cross_entropy"] is None
          or not math.isclose(point["conditional_accuracy"] * point["topk_coverage"],
                              point["accuracy"], rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError("conditional accuracy does not match all-position accuracy and coverage")
    if (point["fixes"] + point["regressions"] > point["target_count"]
            or not math.isclose(point["accuracy"] - point["baseline_accuracy"],
                                (point["fixes"] - point["regressions"]) / point["target_count"],
                                rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError("fixes minus regressions does not match the accuracy change")
    return point


def read_state(run, *, now=None, stale_after=90):
    """Read actual action measurements without inventing initial zero values."""
    root = Path(run).resolve()
    now = now or datetime.now(timezone.utc)
    errors, warnings, objects = [], [], {}
    for name in ("protocol", "process-status", "sync-status", "launch"):
        value = _read(root / (name + ".json"), root, errors, warnings)
        if value is not None and not isinstance(value, dict):
            errors.append(name + ".json must be an object")
            value = None
        objects[name] = value or {}
    events = _read(root / "events.jsonl", root, errors, warnings, jsonl=True)
    validations = _read(root / "validation.jsonl", root, errors, warnings, jsonl=True)
    points, seen, identities = [], {}, {}
    dataset_identity = objects["protocol"].get("dataset_sha256")
    top_k = objects["protocol"].get("top_k")
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        errors.append("protocol top_k must be a positive integer")
        top_k = None
    for row, validation_file in ([(row, False) for row in events]
                                 + [(row, True) for row in validations]):
        if not validation_file and row.get("event") != "validation_snapshot" and not (
                row.get("kind") == "action_selection" and "scope" in row and "accuracy" in row):
            continue
        try:
            point = _snapshot(row)
            key = (point["arm"], point["scope"], point["epoch"], point["global_step"])
            if key in seen:
                if seen[key] != point:
                    raise ValueError("conflicting repeated snapshot " + str(key))
                continue
            if dataset_identity is not None and point["dataset_sha256"] != dataset_identity:
                raise ValueError("dataset identity differs from the run")
            if top_k is not None and point["top_k"] != top_k:
                raise ValueError("top_k differs from the run; mixed candidate counts cannot be compared")
            scope_key = (point["arm"], point["scope"])
            identity = tuple(point[name] for name in (
                "dataset_sha256", "subset_sha256", "top_k", "story_count", "target_count",
                "baseline_accuracy", "topk_coverage"))
            if scope_key in identities and identities[scope_key] != identity:
                raise ValueError("validation subset identity or fixed baseline changed: " + str(scope_key))
            dataset_identity = point["dataset_sha256"]
            top_k = point["top_k"]
            identities[scope_key] = identity
            seen[key] = point
            points.append(point)
        except (ValueError, TypeError, AttributeError, OverflowError) as error:
            errors.append("action validation: " + str(error))
    points.sort(key=lambda point: (point["global_step"], SCOPES.index(point["scope"]), point["epoch"]))
    progress = {}
    for row in events:
        if row.get("arm") == "frozen_action" and (row.get("event") in ("progress", "training_progress")
                                                  or row.get("phase") == "training"):
            progress = {key: row.get(key) for key in (
                "global_step", "total_steps", "epoch", "stories_completed", "total_stories",
                "processed_stories", "updates_per_epoch", "timestamp_utc")}
    latest = events[-1] if events else {}
    sync, process = objects["sync-status"], objects["process-status"]
    if sync.get("error"):
        errors.append("Remote sync: " + str(sync["error"]))
    process_status = str(process.get("status", process.get("state", ""))).lower()
    exit_code = process.get("exit_code", process.get("returncode"))
    if process_status in ("failed", "error") or (exit_code is not None and exit_code != 0):
        errors.append("Training process failed" + (" (exit %s)" % exit_code if exit_code is not None else ""))
    if sync:
        fresh_at = _timestamp(sync.get("last_success_utc"))
    else:
        paths = [root / name for name in ("events.jsonl", "validation.jsonl", "process-status.json", "protocol.json")]
        mtimes = [datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                  for path in paths if path.is_file() and not path.is_symlink()]
        fresh_at = max(mtimes, default=None)
    age = max(0., (now - fresh_at).total_seconds()) if fresh_at else None
    last_snapshot = max((_timestamp(point["timestamp_utc"]) for point in points), default=None)
    phase = latest.get("phase")
    if process_status in ("failed", "error") or (exit_code is not None and exit_code != 0):
        phase = "failed"
    elif process_status in ("completed", "finished"):
        phase = "completed"
    return {"schema": 1, "kind": "action_selection", "run": root.name, "top_k": top_k,
            "generated_at": now.isoformat(), "status": "error" if errors else "ready" if points else "waiting",
            "points": points, "errors": list(dict.fromkeys(errors)), "warnings": warnings,
            "latest_event": {**{key: latest.get(key) for key in (
                "event", "scope", "arm", "epoch", "global_step", "total_steps", "timestamp_utc")},
                "phase": phase},
            "progress": progress, "process": process, "protocol": objects["protocol"], "sync": sync,
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
    parser.add_argument("--port", type=int, default=8769)
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
    print("Action dashboard: http://%s:%d/ (run %s)" % (args.host, server.server_port, args.run), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
