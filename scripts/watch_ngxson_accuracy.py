#!/usr/bin/env python3
"""Retain validation-accuracy winners without changing the running trainer.

The trainer replaces checkpoints atomically. This CPU-only observer hardlinks
each completed inode before reading it, matches its cursor to validation logs,
and keeps the best available weights. It cannot reconstruct overwritten weights.
Directory kqueue notifications avoid polling and are registered before capture.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import sys
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ("reference_repository", "reference_revision", "data_sha256", "config",
            "source_sha256", "trainer_sources_sha256", "frozen_buffers_sha256",
            "initial_parameter_audit", "initialization")
TERMINAL = {"completed", "debug_completed", "debug_stopped", "failed"}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def cursor_key(record):
    return record["updates"], record["epoch"]


def read_rows(path):
    """Ignore only a trailing partial append; completed malformed rows are errors."""
    raw = path.read_bytes() if path.exists() else b""
    lines = raw.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b"\n")]


class AccuracyObserver:
    def __init__(self, run, output, seed_checkpoints=(), source_root=ROOT):
        self.run, self.output = Path(run).resolve(), Path(output).resolve()
        if self.output == self.run:
            raise ValueError("Retention output must be separate from the trainer output")
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((self.run / "manifest.json").read_text())
        for key in BINDINGS:
            if key not in self.manifest:
                raise ValueError("Missing manifest binding: " + key)
        self.binding_hash = hashlib.sha256(json.dumps(
            {k: self.manifest[k] for k in BINDINGS}, sort_keys=True).encode()).hexdigest()
        data_path = Path(self.manifest["data_path"])
        if file_hash(data_path) != self.manifest["data_sha256"]:
            raise ValueError("Training data hash mismatch")
        data = json.loads(data_path.read_text())
        rows = data["validation"]
        limit = self.manifest["config"].get("eval_limit")
        if limit is not None:
            rows = rows[:limit]
        self.expected_stories = len(rows)
        self.expected_tokens = sum(len(row["ids"]) - 1 for row in rows)
        if self.expected_stories <= 0 or self.expected_tokens <= 0:
            raise ValueError("Empty validation data")
        for name, digest in self.manifest["trainer_sources_sha256"].items():
            if file_hash(Path(source_root) / name) != digest:
                raise ValueError("Bound training source hash mismatch: " + name)
        self.seeds = [Path(p).resolve() for p in seed_checkpoints]
        self.seen = {}
        self.best_record = None
        self.best_hash = None
        self.best_inode = None
        self.records = []
        self.started_at = utcnow()
        self.captures = 0
        self.selection_path = self.output / "selection.json"
        self.best_path = self.output / "best_accuracy.pt"
        # Reconstruct state from actual retained bytes after an interrupted
        # checkpoint/JSON pair update; do not trust a possibly stale receipt.
        if self.best_path.exists():
            self.seeds.insert(0, self.best_path)

    def validate_record(self, record):
        for key in ("updates", "epoch"):
            if type(record.get(key)) is not int or record[key] < 0:
                raise ValueError("Invalid validation cursor")
        metric = record["validation"]
        if (type(metric.get("tokens")) is not int or metric["tokens"] != self.expected_tokens
                or type(metric.get("stories")) is not int or metric["stories"] != self.expected_stories):
            raise ValueError("Validation population mismatch")
        hits = metric["correct"]
        if type(hits) is not int or not 0 <= hits <= self.expected_tokens:
            raise ValueError("Invalid correct prediction count")
        for key in ("cross_entropy", "top1_accuracy"):
            value = metric[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Nonfinite validation metric: " + key)
        if metric["cross_entropy"] < 0 or not math.isclose(
                metric["top1_accuracy"], hits / self.expected_tokens, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Inconsistent accuracy or cross entropy")
        if "perplexity" in metric:
            perplexity = metric["perplexity"]
            if not math.isfinite(perplexity) or perplexity <= 0 or not math.isclose(
                    math.log(perplexity), metric["cross_entropy"], abs_tol=1e-10):
                raise ValueError("Inconsistent perplexity")

    def read_metrics(self):
        records = [r for r in read_rows(self.run / "metrics.jsonl") if r.get("event") == "validation"]
        for record in records:
            self.validate_record(record)
        self.records = records

    def verify_checkpoint(self, saved):
        if saved.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        for key in BINDINGS:
            if saved["manifest"].get(key) != self.manifest[key]:
                raise ValueError("Checkpoint binding mismatch: " + key)
        if saved["frozen_buffers_sha256"] != self.manifest["frozen_buffers_sha256"]:
            raise ValueError("Checkpoint frozen buffer receipt mismatch")
        key = cursor_key(saved["cursor"])
        matches = [row for row in self.records if cursor_key(row) == key]
        if not matches:
            # A validation append + checkpoint replace can race scan's initial
            # metrics read. Once this immutable inode exists its log row exists.
            self.read_metrics()
            matches = [row for row in self.records if cursor_key(row) == key]
        if not matches:
            raise ValueError("Checkpoint cursor has no completed validation record: " + str(key))
        if any(row["validation"] != matches[0]["validation"] for row in matches[1:]):
            raise ValueError("Conflicting validation metrics at the same cursor")
        return matches[0]

    def verify_parameters(self, saved):
        values, audit = saved["parameters"], saved["parameter_audit"]
        initial = self.manifest["initial_parameter_audit"]
        if not values or values.keys() != audit.keys() or values.keys() != initial.keys():
            raise ValueError("Checkpoint parameter names mismatch")
        total = 0
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ValueError("Expected CPU tensor: " + name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite checkpoint parameter: " + name)
            digest = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            if (digest != audit[name]["sha256"] or audit[name].get("finite") is not True
                    or value.numel() != audit[name]["values"] or value.numel() != initial[name]["values"]):
                raise ValueError("Checkpoint parameter fingerprint mismatch: " + name)
            total += value.numel()
        if total != self.manifest["initialization"]["trainable_parameters"]:
            raise ValueError("Checkpoint trainable parameter count mismatch")

    def capture(self, source):
        try:
            stat = source.stat()
        except FileNotFoundError:
            return False
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self.seen.get(str(source)) == signature:
            return False
        candidate = self.output / (".capture-" + uuid.uuid4().hex + ".pt")
        try:
            # link() refers to either completed inode around a concurrent rename.
            # Inspect the link itself, never re-open the mutable source path.
            os.link(source, candidate)
            captured_stat = candidate.stat()
            captured_signature = (captured_stat.st_dev, captured_stat.st_ino,
                                  captured_stat.st_size, captured_stat.st_mtime_ns)
            saved = torch.load(candidate, map_location="cpu", mmap=True, weights_only=False)
            record = self.verify_checkpoint(saved)
            self.captures += 1
            better = (self.best_record is None or
                      record["validation"]["correct"] > self.best_record["validation"]["correct"] or
                      (record["validation"]["correct"] == self.best_record["validation"]["correct"]
                       and cursor_key(record) < cursor_key(self.best_record)))
            if better:
                self.verify_parameters(saved)
                digest = file_hash(candidate)
                candidate.replace(self.best_path)
                self.best_record, self.best_hash = record, digest
                stat = self.best_path.stat()
                self.best_inode = {"device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size}
            del saved
            self.seen[str(source)] = captured_signature
            return better
        finally:
            candidate.unlink(missing_ok=True)

    def selection(self):
        historical = min(self.records, key=lambda r: (-r["validation"]["correct"], *cursor_key(r))) if self.records else None
        exact_available = (historical is not None and self.best_record is not None
                           and cursor_key(historical) == cursor_key(self.best_record))
        accuracy_attained = (historical is not None and self.best_record is not None
                             and historical["validation"]["correct"] == self.best_record["validation"]["correct"])
        return {"format_version": 1, "updated_at_utc": utcnow(), "started_at_utc": self.started_at,
                "run_path": str(self.run), "observer_pid": os.getpid(),
                "manifest_binding_sha256": self.binding_hash,
                "criterion": "maximum token-weighted validation top1 accuracy; earliest available update on ties",
                "selection_scope": "available checkpoints observed since watcher startup, including supplied seeds",
                "best_available_validation_record": self.best_record,
                "best_available_checkpoint_sha256": self.best_hash,
                "best_available_checkpoint_path": str(self.best_path) if self.best_record else None,
                "best_available_checkpoint_stat": self.best_inode,
                "historical_best_validation_record": historical,
                "historical_best_weights_available": exact_available,
                "historical_best_accuracy_attained": accuracy_attained,
                "unavailable_higher_accuracy_records": [row for row in self.records if self.best_record is None
                    or row["validation"]["correct"] > self.best_record["validation"]["correct"]],
                "limitation": "Log entries cannot restore overwritten weights. A logged checkpoint may still be saving.",
                "completed_validation_records": len(self.records), "checkpoint_captures": self.captures,
                "training_run": False, "device": "cpu"}

    def scan(self):
        self.read_metrics()
        sources = [*self.seeds, self.run / "best.pt", self.run / "latest.pt"]
        for source in dict.fromkeys(sources):
            self.capture(source)
        value = self.selection()
        # Directory notifications caused by trainer logs may be frequent. Avoid
        # unnecessary disk writes when neither metrics nor retained bytes changed.
        old = json.loads(self.selection_path.read_text()) if self.selection_path.exists() else {}
        if any(old.get(k) != value[k] for k in
               ("best_available_checkpoint_sha256", "completed_validation_records", "checkpoint_captures")):
            atomic_json(self.selection_path, value)
        return value

    def terminal_status(self):
        for filename in ("failure.json", "status.json", "process-status.json"):
            path = self.run / filename
            if path.exists():
                value = json.loads(path.read_text())
                if value.get("status") in TERMINAL:
                    return value
        return None


def watch(args):
    if not hasattr(select, "kqueue"):
        raise RuntimeError("This watcher requires macOS/BSD kqueue")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observer = AccuracyObserver(args.run, args.output, args.seed_checkpoint)
        torch.set_num_threads(args.threads)
        fd = os.open(observer.run, os.O_RDONLY)
        try:
            with closing(select.kqueue()) as queue:
                event = select.kevent(fd, filter=select.KQ_FILTER_VNODE,
                                     flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                     fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_RENAME | select.KQ_NOTE_DELETE)
                # Install before initial scan, so updates racing startup stay queued.
                queue.control([event], 0, 0)
                observer.scan()
                atomic_json(args.output / "status.json", {"status": "running", "pid": os.getpid(),
                    "started_at_utc": observer.started_at, "training_run": False, "device": "cpu"})
                (args.output / "failure.json").unlink(missing_ok=True)
                while True:
                    terminal = observer.terminal_status()
                    if terminal:
                        observer.scan()
                        atomic_json(args.output / "status.json", {"status": "completed", "pid": os.getpid(),
                            "finished_at_utc": utcnow(), "trainer_terminal_status": terminal})
                        return
                    queue.control(None, 8, None)
                    observer.scan()
        finally:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        watch(args)
    except BlockingIOError:
        print("An accuracy observer already holds this output lock", file=sys.stderr)
        raise SystemExit(2)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "pid": os.getpid(), "at_utc": utcnow(),
                   "type": type(error).__name__, "error": str(error)}
        atomic_json(args.output / "failure.json", failure)
        atomic_json(args.output / "status.json", failure)
        raise


if __name__ == "__main__":
    main()
