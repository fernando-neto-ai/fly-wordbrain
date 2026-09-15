import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from fly_wordbrain.dashboard import DashboardServer, read_state, validation_csv


NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def snapshot(step=1, arm="frozen", scope="monitor_subset", ce=2., **extra):
    return {"event": "validation_snapshot", "arm": arm, "epoch": 1,
            "global_step": step, "scope": scope, "story_count": 128 if scope == "monitor_subset" else 1024,
            "target_count": 2048, "target_counts": {"next_1": 1025, "next_2": 1023},
            "subset_sha256": scope + "-digest", "timestamp_utc": NOW.isoformat(),
            "metrics": {"horizons": {"next_1": {"cross_entropy": ce, "perplexity": 7.38905609893065,
                                                  "accuracy": .25},
                                      "next_2": {"cross_entropy": 12., "accuracy": .01}}}, **extra}


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def lines(self, name, rows):
        (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_real_head_one_scope_counts_and_csv(self):
        self.lines("validation.jsonl", [snapshot(), snapshot(64, scope="full_validation")])
        state = read_state(self.root, now=NOW)
        self.assertEqual(state["errors"], [])
        self.assertEqual(state["status"], "ready")
        self.assertEqual(len(state["points"]), 2)
        self.assertEqual(state["points"][0]["cross_entropy"], 2.)
        self.assertEqual(state["points"][0]["target_count"], 1025)
        rows = list(csv.DictReader(io.StringIO(validation_csv(state))))
        self.assertEqual([row["story_count"] for row in rows], ["128", "1024"])
        self.assertEqual([row["scope"] for row in rows], ["monitor_subset", "full_validation"])
        self.assertEqual(rows[0]["accuracy"], "0.25")

    def test_duplicate_receipts_dedup_but_conflicts_reported(self):
        row = snapshot()
        self.lines("events.jsonl", [row])
        self.lines("validation.jsonl", [row])
        self.write("frozen/validation-history.json", [row])
        self.assertEqual(len(read_state(self.root)["points"]), 1)
        self.write("frozen/validation-history.json", [snapshot(ce=3.)])
        state = read_state(self.root)
        self.assertEqual(state["status"], "error")
        self.assertIn("conflicting repeated", " ".join(state["errors"]))
        self.assertEqual(len(state["points"]), 1)

    def test_incomplete_last_append_is_pending_not_fabricated(self):
        (self.root / "events.jsonl").write_text(json.dumps(snapshot()) + '\n{"event":')
        state = read_state(self.root)
        self.assertEqual(len(state["points"]), 1)
        self.assertFalse(state["errors"])
        self.assertIn("incomplete final", state["warnings"][0])
        (self.root / "events.jsonl").write_text('{"bad":\n' + json.dumps(snapshot()) + '\n')
        state = read_state(self.root)
        self.assertEqual(len(state["points"]), 1)
        self.assertIn("line 1", state["errors"][0])

    def test_missing_waits_malformed_errors_and_no_fake_zero(self):
        empty = read_state(self.root)
        self.assertEqual(empty["status"], "waiting")
        self.assertEqual(empty["points"], [])
        self.assertTrue(empty["freshness"]["stale"])
        (self.root / "validation.jsonl").write_text('not json\n')
        broken = read_state(self.root)
        self.assertEqual(broken["status"], "error")
        self.assertTrue(broken["errors"])
        self.assertEqual(broken["points"], [])

    def test_stale_uses_success_not_recent_failed_attempt(self):
        self.lines("events.jsonl", [snapshot()])
        self.write("sync-status.json", {"state": "error", "timestamp_utc": NOW.isoformat(),
                   "last_success_utc": (NOW - timedelta(seconds=200)).isoformat(), "error": "offline"})
        state = read_state(self.root, now=NOW)
        self.assertTrue(state["freshness"]["stale"])
        self.assertEqual(state["freshness"]["age_seconds"], 200)
        self.assertEqual(state["freshness"]["last_snapshot_utc"], NOW.isoformat())
        self.assertEqual(len(state["points"]), 1)
        self.assertIn("Remote sync: offline", state["errors"])
        self.write("sync-status.json", {"state": "ok", "last_success_utc": NOW.isoformat()})
        self.assertFalse(read_state(self.root, now=NOW)["freshness"]["stale"])

    def test_subset_identity_cannot_silently_change_across_arms(self):
        self.lines("events.jsonl", [snapshot(), snapshot(64, arm="fixed_fast", subset_sha256="different")])
        state = read_state(self.root)
        self.assertEqual(len(state["points"]), 1)
        self.assertIn("subset identity changed", " ".join(state["errors"]))

    def test_nonfinite_negative_metrics_and_unknown_scope_rejected(self):
        self.lines("events.jsonl", [snapshot(ce=-1), snapshot(2, scope="test"), snapshot(3, ce=float("nan"))])
        state = read_state(self.root)
        self.assertEqual(state["points"], [])
        self.assertEqual(len(state["errors"]), 3)

    def test_top_level_and_nested_metric_conflict_rejected(self):
        self.lines("events.jsonl", [snapshot(head1_cross_entropy=22)])
        self.assertIn("conflicting nested", " ".join(read_state(self.root)["errors"]))

    def test_current_phase_and_arm_progress_kept_separate(self):
        progress = {"event": "training_progress", "arm": "fixed_fast", "global_step": 64,
                    "processed_stories": 512, "updates_per_epoch": 1024, "epoch": 1}
        phase = {"event": "phase", "phase": "validation", "scope": "monitor_subset",
                 "arm": "fixed_fast", "global_step": 64}
        self.lines("events.jsonl", [snapshot(), progress, phase])
        state = read_state(self.root)
        self.assertEqual(state["latest_event"]["phase"], "validation")
        self.assertEqual(state["latest_event"]["arm"], "fixed_fast")
        self.assertEqual(state["progress"]["fixed_fast"]["processed_stories"], 512)
        self.assertEqual(state["points"][0]["arm"], "frozen")

    def test_http_routes_refresh_and_failure_cache(self):
        self.lines("events.jsonl", [snapshot()])
        self.write("secret.json", {"secret": "never served"})
        calls = []
        def refresh():
            calls.append(1)
        server = DashboardServer(("127.0.0.1", 0), self.root, refresh_callback=refresh)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:" + str(server.server_port)
        try:
            with urlopen(base + "/") as response:
                page = response.read().decode()
                self.assertIn("Monitor subset", page)
                self.assertIn("Full validation", page)
                self.assertIn("setTimeout(refresh,15000)", page)
                self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
                self.assertNotIn("cdn.", page)
            self.assertFalse(calls)
            with urlopen(base + "/api/state") as response:
                self.assertEqual(len(json.load(response)["points"]), 1)
            with urlopen(base + "/api/validation.csv") as response:
                self.assertIn("attachment", response.headers["Content-Disposition"])
                self.assertIn("monitor_subset", response.read().decode())
            self.assertEqual(len(calls), 2)
            for route in ("/secret.json", "/../secret.json", "/api/state/../../secret.json"):
                with self.assertRaises(HTTPError) as raised:
                    urlopen(base + route)
                self.assertEqual(raised.exception.code, 404)
            def fail():
                raise RuntimeError("transport broke")
            server.refresh_callback = fail
            with urlopen(base + "/api/state") as response:
                state = json.load(response)
                self.assertEqual(len(state["points"]), 1)
                self.assertIn("transport broke", " ".join(state["errors"]))
            with self.assertRaises(HTTPError) as raised:
                urlopen(base + "/api/validation.csv")
            self.assertEqual(raised.exception.code, 409)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
