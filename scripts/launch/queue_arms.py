#!/usr/bin/env python3
"""Run the remaining four-task arms in turn once the leak arm finishes.

Order is Z, R, Q and the reason is dependency, not preference. Every four-task contrast is
against Z32fourtasksbounded -- R differs from it on readout_rank alone, Q on weights alone,
L on leak alone -- and Z was stopped at 8,992 to bring L forward. Until Z reaches 16,800
there is no matched baseline for any of the three. Putting it first costs nothing and makes
L comparable ~23 h sooner than putting it last.

Rank is the arm to watch: 64 -> 128 bought -0.1517 nats [-0.1620, -0.1413] and +2.06pp
[+1.71, +2.40] on language, against plasticity's -0.0200 and the fly's own wiring at
0.0100. It is worth more than everything else measured, combined.

Each arm is gated on the previous reaching 16,800, and each gets a checkpoint snapshotter:
the trainer keeps only best.pt and latest.pt, latest.pt advances under you, and a matched
mid-run comparison against another arm is otherwise a one-shot.
"""
import json, os, subprocess, sys, time
from pathlib import Path

HOME = Path.home()
ROOT = HOME / "fly_wordbrain_connectorch"
ARMS = ROOT / "results/unified-v1/arms"
PYTHON = str(ROOT / ".venv-connectorch/bin/python")
TARGET = 16800
LEAK = "L32fourtaskstrainableleak"
QUEUE = ["Z32fourtasksbounded", "R32fourtasksrank256bounded", "Q32fourtasksheavylanguage"]


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
        return False, f"{arm}: stopped at {body.get('updates')} of {TARGET:,}"
    return True, f"{arm}: {body['updates']:,} updates, best CE {body['best']['cross_entropy']:.4f}"


def wait_for_chain(arm):
    while subprocess.run(["pgrep", "-f", f"chain_t.py.*{arm}"], capture_output=True).returncode == 0:
        time.sleep(60)


def main():
    pid = int(sys.argv[1])
    print(f"waiting for {LEAK} (pid {pid})", flush=True)
    while alive(pid):
        time.sleep(60)
    ok, message = finished(LEAK)
    print(("OK " if ok else "ABORT ") + message, flush=True)
    if not ok:
        raise SystemExit(1)
    wait_for_chain(LEAK)

    for arm in QUEUE:
        launch = subprocess.run(["zsh", str(HOME / "launch_task_n.sh"), arm],
                                capture_output=True, text=True)
        print((launch.stdout + launch.stderr).strip(), flush=True)
        if launch.returncode != 0:
            print(f"LAUNCH FAILED {arm}; stopping the queue"); raise SystemExit(1)
        new_pid = int(launch.stdout.split("PID ")[1].split()[0])
        subprocess.Popen(["zsh", str(HOME / "snapshot_checkpoints.sh"), arm, "2000"],
                         stdout=open(HOME / f"snapshots-{arm}.log", "w"), stderr=subprocess.STDOUT)
        print(f"{arm} running as {new_pid}, snapshots every 2,000 updates", flush=True)
        subprocess.run([PYTHON, "-u", str(HOME / "chain_t.py"), str(new_pid), arm],
                       cwd=str(HOME), check=False)
        ok, message = finished(arm)
        print(("DONE " if ok else "ABORT ") + message, flush=True)
        if not ok:
            raise SystemExit(1)
    print("FOUR-TASK QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
