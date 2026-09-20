#!/usr/bin/env python3
"""Wait for the three-task arm, gate it, then score it."""
import json, os, subprocess, sys, time
from pathlib import Path

HOME = Path.home()
ROOT = HOME / "fly_wordbrain_connectorch"
ARMS = ROOT / "results/unified-v1/arms"
TARGET = 16800
MODEL = "data/ngxson-fly-llm-hf/65c677b3d566a2e9793d5f72999cdb441c6c0a9f"
PYTHON = str(ROOT / ".venv-connectorch/bin/python")


def wait(pid):
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(60)


def finished(arm):
    status = ARMS / arm / "status.json"
    if not status.exists():
        return False, f"{arm}: no status.json"
    body = json.loads(status.read_text())
    if body.get("status") not in ("completed", "debug_stopped"):
        return False, f"{arm}: status {body.get('status')!r}"
    if body.get("updates") != TARGET:
        return False, f"{arm}: stopped at {body.get('updates')} of {TARGET}"
    report = ARMS / arm / "unified-results.json"
    if not report.exists():
        return False, f"{arm}: no unified-results.json"
    r = json.loads(report.read_text())
    return True, (f"{arm}: {body['updates']:,} updates | language val CE "
                  f"{body['best']['cross_entropy']:.4f} | chess "
                  f"{r['audit']['chess']['top1_legal_masked']*100:.2f}% | sentiment "
                  f"{r['audit']['sentiment']['accuracy']*100:.2f}%")


def main():
    pid, arm = int(sys.argv[1]), sys.argv[2]
    print(f"waiting for {arm} (pid {pid})", flush=True)
    wait(pid)
    ok, message = finished(arm)
    print(("OK " if ok else "ABORT ") + message, flush=True)
    if not ok:
        raise SystemExit(1)
    out = ARMS / arm
    environment = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0", OMP_NUM_THREADS="4",
                       VECLIB_MAXIMUM_THREADS="4")
    for label, command in (
        ("audit", [PYTHON, "scripts/evaluate_unified_quality.py", "--run", str(out),
                   "--config", arm, "--model", MODEL,
                   "--groups", "data/connectorch-groups-v1/groups.npz",
                   "--audit-data", str(HOME / "fly_wordbrain/data/ngxson-quality-v1/dataset.json"),
                   "--reference-records",
                   str(ROOT / "results/graph-control-audit-v1/I32rank64-10k.json"),
                   "--device", "mps", "--output", str(out / "audit-language.json")]),
        ("router control", [PYTHON, "scripts/probe_task_identity.py", "--run", str(out),
                            "--config", arm, "--model", MODEL,
                            "--groups", "data/connectorch-groups-v1/groups.npz",
                            "--data", "data/ngxson-tinystories-10k/dataset.json",
                            "--chess-corpus", "data/chess-positions-v1",
                            "--steps", "5", "32", "--device", "mps",
                            "--output", str(out / "task-identity-probe.json")]),
    ):
        print(f"[{arm}] {label}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        print((result.stdout or result.stderr)[-1600:], flush=True)
        if result.returncode != 0:
            print(f"EVAL FAILED {arm} {label}", flush=True)
    print("THREE TASK ARM COMPLETE", flush=True)


if __name__ == "__main__":
    main()
