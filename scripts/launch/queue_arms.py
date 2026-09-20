#!/usr/bin/env python3
"""Run the two language-recovery arms in turn, once Z is finished and audited.

Serial, not parallel: both would share one MPS device and neither timing nor the arms
themselves would be comparable to W and Y, which had the machine to themselves.

Each arm is gated on the one before it reaching 16,800 updates. An arm that stopped short
is not a usable control -- the whole point of these two is that they differ from
W32threetasksbounded in exactly one field at a matched budget -- so a short run stops the
queue rather than letting the next one start against a broken reference.
"""
import json, os, subprocess, sys, time
from pathlib import Path

HOME = Path.home()
ROOT = HOME / "fly_wordbrain_connectorch"
ARMS = ROOT / "results/unified-v1/arms"
TARGET = 16800
PYTHON = str(ROOT / ".venv-connectorch/bin/python")
QUEUE = ["R32fourtasksrank256bounded", "Q32fourtasksheavylanguage"]


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def finished(arm):
    status = ARMS / arm / "status.json"
    if not status.exists():
        return False, f"{arm}: no status.json"
    body = json.loads(status.read_text())
    if body.get("updates") != TARGET:
        return False, f"{arm}: stopped at {body.get('updates')} of {TARGET}"
    return True, f"{arm}: {body['updates']:,} updates, best CE {body['best']['cross_entropy']:.4f}"


def main():
    for pid in (int(a) for a in sys.argv[1:]):
        print(f"waiting for pid {pid}", flush=True)
        while alive(pid):
            time.sleep(60)

    ok, message = finished("Z32fourtasksbounded")
    print(("OK " if ok else "ABORT ") + message, flush=True)
    if not ok:
        raise SystemExit(1)

    for arm in QUEUE:
        launch = subprocess.run(["zsh", str(HOME / "launch_task_n.sh"), arm],
                                capture_output=True, text=True)
        print((launch.stdout + launch.stderr).strip(), flush=True)
        if launch.returncode != 0:
            print(f"LAUNCH FAILED {arm}; stopping the queue"); raise SystemExit(1)
        pid = int(launch.stdout.split("PID ")[1].split()[0])
        subprocess.run([PYTHON, "-u", str(HOME / "chain_t.py"), str(pid), arm],
                       cwd=str(HOME), check=False)
        ok, message = finished(arm)
        print(("DONE " if ok else "ABORT ") + message, flush=True)
        if not ok:
            raise SystemExit(1)
    print("LANGUAGE QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
