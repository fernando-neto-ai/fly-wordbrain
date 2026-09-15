#!/usr/bin/env python3
"""Pull bounded, checksum-verified macm3 run metrics and render a Markdown report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
FILES = ["manifest.json", "status.json", "metrics.jsonl", "training.jsonl", "initial-gradients.json",
         "results.json", "failure.json", "launch.json", "process-status.json"]
REMOTE = '''from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
files={}
for name in json.loads(sys.argv[2]):
    path=root/name
    if not path.exists(): continue
    if path.is_symlink(): raise ValueError('Unexpected symlink')
    with path.open('rb') as f: body=f.read(16777217)
    if len(body)>16777216: raise ValueError('File exceeds 16 MiB bound')
    files[name]={'text':body.decode(),'sha256':hashlib.sha256(body).hexdigest()}
print(json.dumps(files))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="macm3")
    parser.add_argument("--remote-run", default="/Users/fernando/fly_wordbrain/results/ngxson-reconstructed-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ngxson-reconstructed-v1")
    args = parser.parse_args()
    import cluster_runner as cr
    command = "python3 -c " + shlex.quote(REMOTE) + " " + shlex.quote(args.remote_run) + " " + shlex.quote(json.dumps(FILES))
    result = cr.run(command, host=args.host, timeout=30)
    if result.exit_code:
        raise RuntimeError(result.stderr or result.stdout)
    files = json.loads(result.stdout)
    if not files:
        raise RuntimeError("No run artifacts found")
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, entry in files.items():
        if name not in FILES:
            raise ValueError("Unexpected source filename")
        body = entry["text"].encode()
        if hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError("Transfer checksum mismatch")
        temporary = args.output / (name + ".tmp")
        temporary.write_bytes(body)
        temporary.replace(args.output / name)
        receipts[name] = entry["sha256"]
    now = datetime.now(timezone.utc).isoformat()
    (args.output / "snapshot-receipt.json").write_text(json.dumps({
        "host": args.host, "remote_run": args.remote_run, "fetched_at_utc": now, "sha256": receipts,
    }, indent=2) + "\n")
    status = json.loads(files.get("status.json", {"text": "{}"})["text"])
    process = json.loads(files.get("process-status.json", {"text": "{}"})["text"])
    rows = []
    lines = files.get("metrics.jsonl", {"text": ""})["text"].splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not line.endswith("\n"):
                continue
            raise
        if row.get("event") == "validation":
            rows.append(row)
    text = ["# ngxson Fly LLM — partial validation", "",
            f"Snapshot: {now}. Host: {args.host}.", "",
            f"Training status: **{status.get('status', 'starting')}**; process: **{process.get('status', 'unknown')}**.",
            f"Completed optimizer updates: **{status.get('updates', 0):,}**. Completed epochs: **{status.get('epoch', 0)} / 44**.", "",
            "52,756,661 trainable parameters; 49,393 neurons and 9,050,172 frozen directed edges. "
            "Training from random learned values on 1,000 stories, with 100 validation stories and 100 test stories.", "",
            "Validation uses all 100 stories at initialization, every 100 updates, and each epoch. "
            "Accuracy is next-BPE-token top-1, not next-word or ten-candidate accuracy. "
            "Training CE is the running average during the current epoch, not an independently evaluated training-set score.", "",
            "| Updates | Epoch cursor | Validation CE ↓ | Validation accuracy ↑ | Training CE so far |",
            "|---:|---:|---:|---:|---:|"]
    for row in rows:
        val = row["validation"]
        train_ce = row.get("training_epoch_so_far", {}).get("cross_entropy")
        train = "—" if train_ce is None else f"{train_ce:.4f}"
        text.append(f"| {row['updates']:,} | {row['epoch']} | {val['cross_entropy']:.4f} | {100*val['top1_accuracy']:.2f}% | {train} |")
    if not rows:
        text.append("| — | — | Pending | Pending | — |")
    if rows:
        best = rows[-1]["best"]
        text.extend(["", f"Best validation checkpoint so far: update {best['updates']:,}, CE {best['cross_entropy']:.4f}."])
    text.extend(["", "This is a timestamped snapshot. Refresh it with:", "", "```bash",
                 "/opt/anaconda3/bin/python scripts/refresh_ngxson_progress.py", "```", "",
                 "Raw measurements are in metrics.jsonl; immutable snapshot checksums are in snapshot-receipt.json.", ""])
    (args.output / "progress.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
