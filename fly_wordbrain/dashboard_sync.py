"""Refresh a dashboard cache from an explicitly selected remote training run.

Browser requests trigger a throttled, read-only snapshot. No background polling
or model work occurs, and failed refreshes leave the last verified cache intact.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import threading
import time


ARMS = ("frozen", "fixed_fast", "learned_fast")
FILES = ("protocol.json", "process-status.json", "launch.json", "superseded.json", "progress.json",
         "events.jsonl", "validation.jsonl", "metrics.json") + tuple(
    arm + "/" + filename for arm in ARMS
    for filename in ("validation-history.json", "history.json", "selection.json", "metrics.json"))
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024

REMOTE_SCRIPT = '''from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).expanduser().resolve()
names=json.loads(sys.argv[2])
if not root.is_dir():
    raise RuntimeError('Requested training run directory does not exist')
files={}
total=0
for name in names:
    path=root/name
    if not path.exists():
        continue
    if path.is_symlink() or root not in path.resolve().parents:
        raise RuntimeError('Unexpected symlink in run snapshot')
    with path.open('rb') as stream:
        body=stream.read(16777217)
    total+=len(body)
    if len(body)>16777216 or total>33554432:
        raise RuntimeError('Run snapshot exceeds the declared size bound')
    files[name]={'text':body.decode('utf-8'),'sha256':hashlib.sha256(body).hexdigest()}
print(json.dumps({'run':str(root),'files':files}))
'''


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _atomic_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".sync-tmp")
    temporary.write_bytes(body)
    temporary.replace(path)


class RemoteRunSync:
    def __init__(self, host, remote_run, local_run, minimum_interval=15, runner=None):
        self.host = str(host)
        self.remote_run = str(remote_run)
        self.local_run = Path(local_run).resolve()
        if minimum_interval < 1:
            raise ValueError("Remote refresh interval must be at least one second")
        self.minimum_interval = float(minimum_interval)
        self.runner = runner
        self._last_attempt = float("-inf")
        self._lock = threading.Lock()
        self.local_run.mkdir(parents=True, exist_ok=True)
        binding_path = self.local_run / "sync-binding.json"
        binding = {"host": self.host, "remote_run": self.remote_run}
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("Dashboard cache belongs to a different remote run")
        else:
            if any((self.local_run / name).exists() for name in FILES):
                raise ValueError("Use an empty dedicated cache for a remote dashboard")
            with binding_path.open("x") as stream:
                json.dump(binding, stream, indent=2)
                stream.write("\n")

    def _status(self, **fields):
        path = self.local_run / "sync-status.json"
        previous = {}
        if path.is_file():
            try:
                previous = json.loads(path.read_text())
            except (ValueError, OSError):
                pass
        status = {"host": self.host, "remote_run": self.remote_run,
                  "timestamp_utc": _utc(), "last_success_utc": previous.get("last_success_utc"),
                  **fields}
        _atomic_bytes(path, (json.dumps(status, indent=2) + "\n").encode())

    def refresh(self, force=False):
        if not self._lock.acquire(blocking=False):
            return False
        try:
            now = time.monotonic()
            if not force and now - self._last_attempt < self.minimum_interval:
                return False
            self._last_attempt = now
            self._status(state="syncing", error=None)
            try:
                runner = self.runner
                if runner is None:
                    import cluster_runner
                    runner = cluster_runner.run
                command = "python3 -c " + shlex.quote(REMOTE_SCRIPT) + " " + shlex.quote(
                    self.remote_run) + " " + shlex.quote(json.dumps(FILES))
                result = runner(command, host=self.host, timeout=20)
                if result.exit_code:
                    raise RuntimeError((result.stderr or "Remote snapshot failed")[-1000:])
                snapshot = json.loads(result.stdout)
                if not isinstance(snapshot, dict) or not isinstance(snapshot.get("files"), dict):
                    raise ValueError("Remote snapshot is not a file manifest")
                verified = {}
                total = 0
                for name, entry in snapshot["files"].items():
                    if name not in FILES or not isinstance(entry, dict):
                        raise ValueError("Unexpected remote snapshot entry")
                    body = entry["text"].encode("utf-8")
                    total += len(body)
                    if len(body) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                        raise ValueError("Remote snapshot exceeds the declared size bound")
                    if hashlib.sha256(body).hexdigest() != entry.get("sha256"):
                        raise ValueError("Remote snapshot checksum mismatch: " + name)
                    verified[name] = body
                for name, body in verified.items():
                    _atomic_bytes(self.local_run / name, body)
                # This is a dedicated, identity-bound cache. If an artifact is
                # absent remotely, retaining an old copy would misreport state.
                for name in set(FILES) - set(verified):
                    (self.local_run / name).unlink(missing_ok=True)
                self._status(state="ok", error=None, last_success_utc=_utc(),
                             files_refreshed=len(verified), bytes_refreshed=total)
                return True
            except Exception as error:
                self._status(state="error", error=str(error))
                return False
        finally:
            self._lock.release()
