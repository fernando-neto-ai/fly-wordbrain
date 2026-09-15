#!/usr/bin/env python3
"""Fetch verified campaign snapshots and render per-arm partial validation curves."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A128fixed", "B32fixed", "C128bounded", "D32bounded")
FILES = ["manifest.json", "launch.json", "campaign-status.json", "readiness.json", "results.json", "partial-results.json",
         "failure.json", "smoke-results.json", "preflight/parity.json"]
FILES += [f"arms/{arm}/{name}" for arm in ARMS for name in
          ("manifest.json", "launch.json", "status.json", "process-status.json", "metrics.jsonl",
           "selected-checkpoints.json", "results.json", "failure.json")]
REMOTE = '''from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
files={}
for name in json.loads(sys.argv[2]):
 p=root/name
 if not p.exists():continue
 if p.is_symlink():raise ValueError('Unexpected artifact symlink')
 with p.open('rb') as f:body=f.read(16777217)
 if len(body)>16777216:raise ValueError('Artifact exceeds16MiB bound')
 files[name]={'text':body.decode(),'sha256':hashlib.sha256(body).hexdigest()}
print(json.dumps(files))
'''


def render_report(files, host, now):
    """Render only this fetched snapshot; never mix it with stale local files."""
    def read(name):
        return json.loads(files[name]["text"]) if name in files else {}

    manifest = read("manifest.json")
    branch_mapping = manifest.get("experiment_branches") or {}
    status = read("campaign-status.json") or read("readiness.json")
    mode = manifest.get("baseline_acceptance_mode")
    if mode == "user_accepted_early_stop":
        baseline = ("Baseline: **accepted by the user after early stopping**. Its minimum-CE and "
                    "maximum-accuracy checkpoints are preserved; the 44-epoch reference recipe was not completed.")
    elif mode == "completed_reference_recipe":
        baseline = "Baseline: the reference completed its 44-epoch recipe and passed the campaign launch gates."
    else:
        baseline = ("Baseline launch gate: a successfully completed reference or an explicitly accepted, "
                    "verified early-stop receipt is required before M3 preflight and training.")
    phase, active_arm = status.get("phase", "pending"), status.get("arm")
    report = ["# Connectorch encoder experiment — partial validation", "",
              f"Snapshot: {now}. Host: {host}.", "",
              f"Campaign status: **{status.get('status', 'waiting')}**. Phase: **{phase}**. "
              f"Current arm: **{active_arm or '—'}**.", "", baseline,
              "All arms retain the full output head, eight explicit input lags, 49,393 neurons and 9,050,172 base edges.",
              "Bounded arms add 18,322 source/destination cell-type gains; base-edge multipliers stay within 0.9–1.1. Original neuron gains remain unconstrained.", "",
              "| Arm | Width | Edge gains | Status | Updates | Minimum CE (accuracy; update) | Maximum accuracy (CE; update) |",
              "|---|---:|---|---|---:|---|---|"]
    histories, selections, provenance = {}, {}, {}
    for arm in ARMS:
        prefix = "arms/" + arm + "/"
        arm_status = read(prefix + "status.json") or read(prefix + "process-status.json")
        lines = files.get(prefix + "metrics.jsonl", {}).get("text", "").splitlines(keepends=True)
        rows = []
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1 and not line.endswith("\n"):
                    continue
                raise
            if row.get("event") == "validation":
                rows.append(row)
        histories[arm] = rows
        selections[arm] = read(prefix + "selected-checkpoints.json").get("selectors", {})
        provenance[arm] = (read(prefix + "launch.json").get("experiment_git")
                           or {"repository": branch_mapping.get("repository"),
                               **branch_mapping.get("arms", {}).get(arm, {})})

        def selected_cell(selector, metric, maximize=False):
            receipt = selections[arm].get(selector)
            if receipt:
                value, update, suffix = receipt["validation"], receipt["cursor"]["updates"], ""
            elif rows:
                best = (max if maximize else min)(rows, key=lambda row: row["validation"][metric])
                value, update, suffix = best["validation"], best["updates"], "; retention pending"
            else:
                return "—"
            ce, acc = value["cross_entropy"], 100 * value["top1_accuracy"]
            scores = f"{acc:.2f}% ({ce:.4f}" if maximize else f"{ce:.4f} ({acc:.2f}%"
            return f"{scores}; {update:,}{suffix})"

        ce = selected_cell("minimum_validation_ce", "cross_entropy")
        accuracy = selected_cell("maximum_validation_accuracy", "top1_accuracy", maximize=True)
        arm_state = arm_status.get("status", "pending")
        if arm_status.get("activity"):
            arm_state += " / " + arm_status["activity"]
        updates = arm_status.get("updates", rows[-1]["updates"] if rows else 0)
        report.append(f"| {arm} | {128 if '128' in arm else 32} | {'fixed' if 'fixed' in arm else 'bounded10'} | "
                      f"{arm_state} | {updates:,} | {ce} | {accuracy} |")
    report.extend(["", "Winner columns use the saved checkpoint receipts when available. A newer validation can "
                   "appear in the history while its checkpoint is still being saved; exact metric ties retain the earlier checkpoint."])
    if read("failure.json"):
        report.extend(["", "Campaign failure: `" + json.dumps(read("failure.json")) + "`"])
    for arm, rows in histories.items():
        git = provenance[arm]
        if not rows and not git.get("branch"):
            continue
        report.extend(["", "## " + arm, ""])
        if git.get("branch"):
            report.extend([f"Branch: `{git['branch']}`. Commit: `{git.get('commit', 'unavailable')}`. "
                           f"Repository: {git.get('repository') or 'unavailable'}.", ""])
        for selector, receipt in selections[arm].items():
            report.extend([f"Saved {selector}: `{receipt['path']}`; SHA256 `{receipt['sha256']}`.", ""])
        if not rows:
            report.append("No full-arm validation yet.")
            continue
        report.extend(["| Updates | Epoch | Validation CE | Accuracy |", "|---:|---:|---:|---:|"])
        for row in rows:
            metric = row["validation"]
            report.append(f"| {row['updates']:,} | {row['epoch']} | {metric['cross_entropy']:.4f} | {100*metric['top1_accuracy']:.2f}% |")
    report.extend(["", "All partial scores are next-BPE-token results on the 100-story validation split. Test evaluation is deferred.",
                   "This is a timestamped snapshot; each refresh verifies artifact SHA256 checksums. "
                   "Checkpoint hashes are copied from trainer receipts; checkpoint binaries are not fetched by this command.", ""])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="macm3")
    p.add_argument("--remote-run", default="/Users/fernando/fly_wordbrain_connectorch/results/connectorch-encoder-v1")
    p.add_argument("--output", type=Path, default=ROOT / "results/connectorch-encoder-v1")
    args = p.parse_args()
    import cluster_runner as cr
    command = "python3 -c " + shlex.quote(REMOTE) + " " + shlex.quote(args.remote_run) + " " + shlex.quote(json.dumps(FILES))
    result = cr.run(command, host=args.host, timeout=30)
    if result.exit_code:
        raise RuntimeError(result.stderr or result.stdout)
    files = json.loads(result.stdout)
    if not files:
        raise RuntimeError("No campaign artifacts found")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, item in files.items():
        if name not in FILES:
            raise ValueError("Unexpected artifact path")
        body = item["text"].encode()
        if hashlib.sha256(body).hexdigest() != item["sha256"]:
            raise ValueError("Artifact transfer checksum mismatch")
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(body)
        temp.replace(path)
        hashes[name] = item["sha256"]
    now = datetime.now(timezone.utc).isoformat()
    report = render_report(files, args.host, now)
    (args.output / "progress.md").write_text("\n".join(report))
    (args.output / "snapshot-receipt.json").write_text(json.dumps({"fetched_at_utc": now, "host": args.host,
                     "remote_run": args.remote_run, "sha256": hashes}, indent=2) + "\n")
    print("\n".join(report[:17]))


if __name__ == "__main__":
    main()
