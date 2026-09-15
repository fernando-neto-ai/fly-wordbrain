"""Run an exact launch.json command and retain its exit status after SSH closes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run = args.output.resolve()
    launch = json.loads((run / "launch.json").read_text())
    command = launch["command"]
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("Launch command must be a nonempty argument list")
    status = {"status": "starting", "runner_pid": os.getpid(),
              "started_at_utc": utc(), "command": command}
    def save():
        temporary = run / "process-status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(run / "process-status.json")
    save()
    environment = dict(os.environ)
    environment.update(launch.get("environment", {}))
    child, awake = None, None
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if Path("/usr/bin/caffeinate").exists():
            awake = subprocess.Popen(["/usr/bin/caffeinate", "-is"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child = subprocess.Popen(command, cwd=launch["cwd"], env=environment,
                                 stdin=subprocess.DEVNULL)
        status.update(status="running", worker_pid=child.pid)
        if awake is not None:
            status["caffeinate_pid"] = awake.pid
        save()
        code = child.wait()
        status.update(status="completed" if code == 0 else "failed", exit_code=code)
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        if child is not None and child.poll() is None:
            child.terminate()
            status["exit_code"] = child.wait()
        raise
    finally:
        status["finished_at_utc"] = utc()
        save()
        if awake is not None:
            awake.terminate()
            awake.wait()


if __name__ == "__main__":
    main()
