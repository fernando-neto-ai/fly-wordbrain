"""Read-only, dependency-free dashboard for actual validation snapshots.

Run: python -m fly_wordbrain.dashboard --run results/my-run
Only the bundled page, a fixed JSON endpoint, and validation CSV are served.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

ARMS = ("frozen", "fixed_fast", "learned_fast")
SCOPES = ("monitor_subset", "full_validation")
MAX_BYTES = 16 * 1024 * 1024
CSV_FIELDS = ("arm", "scope", "epoch", "global_step", "story_count", "target_count",
              "cross_entropy", "perplexity", "accuracy", "subset_sha256", "timestamp_utc")


def _timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except ValueError:
        return None


def _read(path, root, errors, warnings, *, jsonl=False):
    if not path.exists():
        return [] if jsonl else None
    name = str(path.relative_to(root))
    try:
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError("symlinks outside the run are not supported")
        with path.open("rb") as stream:
            body = stream.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("file exceeds the 16 MiB read limit")
        if not jsonl:
            return json.loads(body)
        rows = []
        lines = body.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("event must be a JSON object")
                rows.append(row)
            except (ValueError, UnicodeDecodeError) as error:
                if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                    warnings.append(name + ": incomplete final line; waiting for append to finish")
                else:
                    errors.append("%s line %d: %s" % (name, index + 1, error))
        return rows
    except (OSError, ValueError, UnicodeDecodeError) as error:
        errors.append(name + ": " + str(error))
        return [] if jsonl else None


def _number(value, name, *, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be finite numeric data")
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(name + " is outside its valid range")
    return value


def _snapshot(row):
    arm, scope = row.get("arm"), row.get("scope")
    if arm not in ARMS or scope not in SCOPES:
        raise ValueError("unknown validation arm or scope")
    step = _number(row.get("global_step"), "global_step", lower=0)
    if int(step) != step:
        raise ValueError("global_step must be an integer")
    head = row.get("metrics", {}).get("horizons", {}).get("next_1", {})
    ce = _number(head.get("cross_entropy", row.get("head1_cross_entropy")), "cross_entropy", lower=0)
    accuracy = _number(head.get("accuracy", row.get("head1_accuracy")), "accuracy", lower=0, upper=1)
    perplexity = head.get("perplexity", row.get("head1_perplexity"))
    if perplexity is None:
        perplexity = math.exp(ce)
    perplexity = _number(perplexity, "perplexity", lower=1)
    for field, measured in (("head1_cross_entropy", ce), ("head1_perplexity", perplexity),
                            ("head1_accuracy", accuracy)):
        if field in row and not math.isclose(_number(row[field], field), measured, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("conflicting nested and top-level " + field)
    count = _number(row.get("story_count"), "story_count", lower=1)
    targets = row.get("target_counts", {}).get("next_1", head.get("examples"))
    if targets is not None:
        targets = _number(targets, "next_1 target count", lower=1)
    return {"arm": arm, "scope": scope, "epoch": row.get("epoch"), "global_step": int(step),
            "story_count": count, "target_count": targets, "cross_entropy": ce,
            "perplexity": perplexity, "accuracy": accuracy,
            "subset_sha256": row.get("subset_sha256"), "timestamp_utc": row.get("timestamp_utc")}


def read_state(run, *, now=None, stale_after=90):
    """Return valid observed snapshots and explicit parsing/identity errors."""
    root = Path(run).resolve()
    now = now or datetime.now(timezone.utc)
    errors, warnings = [], []
    objects = {}
    for name in ("protocol", "process-status", "sync-status"):
        value = _read(root / (name + ".json"), root, errors, warnings)
        if value is not None and not isinstance(value, dict):
            errors.append(name + ".json must be an object")
            value = None
        objects[name] = value or {}
    events = _read(root / "events.jsonl", root, errors, warnings, jsonl=True)
    candidates = list(events) + _read(root / "validation.jsonl", root, errors, warnings, jsonl=True)
    for arm in ARMS:
        history = _read(root / arm / "validation-history.json", root, errors, warnings)
        if history is not None:
            if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
                errors.append(arm + "/validation-history.json must be an array of events")
            else:
                candidates.extend(history)
    points, seen, scope_identities = [], {}, {}
    for row in candidates:
        if row.get("event") != "validation_snapshot":
            continue
        try:
            point = _snapshot(row)
            key = (point["arm"], point["scope"], point["epoch"], point["global_step"])
            if key in seen:
                if seen[key] != point:
                    raise ValueError("conflicting repeated snapshot " + str(key))
                continue
            identity = (point["story_count"], point["subset_sha256"])
            scope_key = point["scope"]
            if scope_key in scope_identities and scope_identities[scope_key] != identity:
                raise ValueError("validation subset identity changed within/across arms: " + scope_key)
            scope_identities[scope_key] = identity
            seen[key] = point
            points.append(point)
        except (ValueError, TypeError, AttributeError, OverflowError) as error:
            errors.append("validation_snapshot: " + str(error))
    points.sort(key=lambda p: (ARMS.index(p["arm"]), p["global_step"], p["scope"]))
    latest = events[-1] if events else {}
    progress = {}
    for row in events:
        if row.get("event") == "training_progress" and row.get("arm") in ARMS:
            progress[row["arm"]] = {key: row.get(key) for key in (
                "arm", "epoch", "global_step", "processed_stories", "processed_targets",
                "updates_per_epoch", "elapsed_seconds", "timestamp_utc")}
    sync = objects["sync-status"]
    if sync.get("error"):
        errors.append("Remote sync: " + str(sync["error"]))
    observed = []
    for row in events + points:
        time = _timestamp(row.get("timestamp_utc"))
        if time:
            observed.append(time)
    if sync:
        fresh_at = _timestamp(sync.get("last_success_utc"))
    else:
        # Without a remote heartbeat, artifact timestamps are our freshness evidence.
        paths = [root / name for name in ("events.jsonl", "validation.jsonl", "process-status.json")]
        mtimes = [datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                  for path in paths if path.is_file()]
        fresh_at = max(mtimes) if mtimes else max(observed, default=None)
    age = max(0., (now - fresh_at).total_seconds()) if fresh_at else None
    last_snapshot = max((_timestamp(p["timestamp_utc"]) for p in points
                         if _timestamp(p["timestamp_utc"])), default=None)
    monitoring = objects["protocol"].get("monitoring", {})
    if not isinstance(monitoring, dict):
        errors.append("protocol monitoring must be an object")
        monitoring = {}
    return {"schema": 1, "run": root.name, "generated_at": now.isoformat(),
            "points": points, "errors": list(dict.fromkeys(errors)), "warnings": warnings,
            "status": "error" if errors else "ready" if points else "waiting",
            "freshness": {"last_success_utc": fresh_at.isoformat() if fresh_at else None,
                          "age_seconds": age, "stale": age is None or age > stale_after,
                          "stale_after_seconds": stale_after,
                          "last_snapshot_utc": last_snapshot.isoformat() if last_snapshot else None},
            "latest_event": {key: latest.get(key) for key in (
                "event", "phase", "scope", "arm", "epoch", "global_step", "timestamp_utc")},
            "progress": progress, "process": objects["process-status"], "sync": sync,
            "protocol": {"arms": objects["protocol"].get("arms", list(ARMS)),
                         "epochs": objects["protocol"].get("epochs"),
                         "batch_size": objects["protocol"].get("batch_size"),
                         "monitoring": monitoring}}


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
        self.refresh_callback = refresh_callback
        self.stale_after = stale_after
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


class DashboardHandler(BaseHTTPRequestHandler):
    def _send(self, body, content_type, status=200, download=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="validation.csv"')
        self.end_headers()
        self.wfile.write(body)

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

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
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
    print("Dashboard: http://%s:%d/ (run %s)" % (args.host, server.server_port, args.run), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
