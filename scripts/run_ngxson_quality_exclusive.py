#!/usr/bin/env python3
"""Temporarily suspend one verified trainer for a bounded GPU inference audit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--trainer-pid", type=int, required=True)
    p.add_argument("--name", default="ngxson-quality-v1-mps")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--selection", choices=("min-ce", "max-accuracy"), default="min-ce")
    p.add_argument("--selection-receipt", type=Path)
    p.add_argument("--text-only", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    expected_script = str(root / "scripts/train_ngxson.py")
    expected_output = str(root / "results/ngxson-reconstructed-v1")
    if Path(args.name).name != args.name or not args.name.startswith("ngxson-quality-"):
        raise ValueError("Use a single ngxson-quality- directory name")
    output = root / "results" / args.name
    receipt_path = root / "results/ngxson-quality-snapshot-v1" / (args.name + "-exclusive.json")
    if output.exists() or receipt_path.exists():
        raise ValueError("Existing audit or launch receipt; do not launch twice")
    command = subprocess.check_output(["ps", "-p", str(args.trainer_pid), "-o", "command="], text=True).strip()
    parts = shlex.split(command)
    assert expected_script in parts and "--output" in parts
    assert parts[parts.index("--output") + 1] == expected_output
    state = subprocess.check_output(["ps", "-p", str(args.trainer_pid), "-o", "state="], text=True).strip()
    assert "T" not in state, "Trainer was already suspended"
    receipt = {"training_run": False, "trainer_pid": args.trainer_pid,
               "verified_trainer_command": command, "audit_output": str(output),
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "timeout_seconds": 900}

    def save():
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(receipt_path)

    def interrupted(signum, frame):
        raise InterruptedError(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    suspended = False
    try:
        # Set this before signaling so an interruption cannot skip the resume.
        suspended = True
        os.kill(args.trainer_pid, signal.SIGSTOP)
        receipt["trainer_suspended"] = True
        save()
        script = "compare_ngxson_texts.py" if args.text_only else "evaluate_ngxson_quality.py"
        argv = [sys.executable, "-u", str(root / "scripts" / script),
            "--model", str(root / "data/ngxson-fly-llm-hf/65c677b3d566a2e9793d5f72999cdb441c6c0a9f"),
            "--checkpoint", str(args.checkpoint or root / "results/ngxson-quality-snapshot-v1/best.pt"),
            "--training-data", str(root / "data/ngxson-tinystories-v1/dataset.json"),
            "--output", str(output), "--threads", "4", "--device", "mps"]
        if not args.text_only:
            argv += ["--audit-data", str(root / "data/ngxson-quality-v1/dataset.json"), "--selection", args.selection]
        if args.selection_receipt:
            argv += ["--selection-receipt", str(args.selection_receipt)]
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0", OMP_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4")
        with (root / "results" / (args.name + ".log")).open("x") as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=900)
        receipt["audit_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Quality audit failed with exit {result.returncode}")
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        if suspended:
            os.kill(args.trainer_pid, signal.SIGCONT)
            receipt["trainer_resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
            receipt["trainer_process_state_after_resume"] = subprocess.check_output(
                ["ps", "-p", str(args.trainer_pid), "-o", "state="], text=True).strip()
        save()
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
