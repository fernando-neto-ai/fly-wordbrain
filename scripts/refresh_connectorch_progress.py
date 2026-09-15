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
DEFAULT_REMOTE_RUN = "/Users/fernando/fly_wordbrain_connectorch/results/connectorch-encoder-v1"
FILES = ["manifest.json", "launch.json", "campaign-status.json", "readiness.json", "results.json", "partial-results.json",
         "failure.json", "smoke-results.json", "preflight/parity.json", "continuation-disposition.json", "queue-review-gate.json"]
FILES += [f"arms/{arm}/{name}" for arm in ARMS for name in
          ("manifest.json", "launch.json", "status.json", "process-status.json", "metrics.jsonl",
           "selected-checkpoints.json", "results.json", "failure.json", "accepted-early-stop.json", "stop-receipt.json")]
PARENT_FILES = ("manifest.json", "partial-results.json", "continuation-disposition.json")
SNAPSHOT_FILES = FILES + ["parent/" + name for name in PARENT_FILES]
REMOTE = '''from pathlib import Path
import hashlib,json,sys
request=json.loads(sys.argv[1])
candidates=[Path(p).resolve() for p in request['candidates']]
root=next((p for p in candidates if (p/'manifest.json').exists()),candidates[-1])
files={};sources={}
def fetch(p,name,expected=None):
 if not p.exists():
  if expected:raise ValueError('Bound artifact is missing: '+str(p))
  return
 if p.is_symlink():raise ValueError('Unexpected artifact symlink')
 with p.open('rb') as f:body=f.read(16777217)
 if len(body)>16777216:raise ValueError('Artifact exceeds16MiB bound')
 digest=hashlib.sha256(body).hexdigest()
 if expected and digest!=expected:raise ValueError('Bound artifact changed: '+str(p))
 files[name]={'text':body.decode(),'sha256':digest};sources[name]=str(p)
for name in request['files']:fetch(root/name,name)
manifest=json.loads(files.get('manifest.json',{}).get('text','{}'))
parent_entry=manifest.get('parent_campaign')
if parent_entry:
 parent_manifest=Path(parent_entry['path']).resolve()
 parent=parent_manifest.parent
 if parent_manifest.name!='manifest.json' or parent==root:raise ValueError('Invalid parent campaign')
 fetch(parent_manifest,'parent/manifest.json',parent_entry['sha256'])
 for name in request['parent_files']:
  if name!='manifest.json':fetch(parent/name,'parent/'+name)
 accepted=manifest['accepted_early_stopped_arm']
 arm='A128fixed';prefix='arms/'+arm+'/'
 if accepted.get('name')!=arm or Path(accepted['output']).resolve()!=parent/prefix:
  raise ValueError('Unexpected accepted-arm output')
 for name in request['files']:
  if name.startswith(prefix):fetch(parent/name,name)
 acceptance=Path(manifest['acceptance_receipt']).resolve()
 if acceptance!=parent/prefix/'accepted-early-stop.json':raise ValueError('Unexpected acceptance receipt path')
 fetch(acceptance,prefix+'accepted-early-stop.json',manifest['input_files_sha256'][str(acceptance)])
 receipt=json.loads(files[prefix+'accepted-early-stop.json']['text'])
 stop=Path(receipt['stop_receipt']['path']).resolve()
 if stop!=parent/prefix/'stop-receipt.json':raise ValueError('Unexpected stop receipt path')
 fetch(stop,prefix+'stop-receipt.json',receipt['stop_receipt']['sha256'])
for arm in ('A128fixed','B32fixed','C128bounded','D32bounded'):
 prefix='arms/'+arm+'/'
 item=files.get(prefix+'accepted-early-stop.json')
 if not item:continue
 receipt=json.loads(item['text'])
 if receipt.get('status')!='accepted_early_stop':continue
 if receipt.get('accepted_by') not in ('user','agent_under_standing_user_authorization') or receipt.get('arm')!=arm:
  raise ValueError('Invalid accepted-stop identity: '+arm)
 if receipt.get('process_cessation',{}).get('confirmed') is not True:
  raise ValueError('Accepted stop lacks confirmed process cessation: '+arm)
 acceptance=Path(sources[prefix+'accepted-early-stop.json'])
 stop=Path(receipt['stop_receipt']['path']).resolve()
 if stop!=acceptance.parent/'stop-receipt.json':raise ValueError('Unexpected stop receipt path')
 fetch(stop,prefix+'stop-receipt.json',receipt['stop_receipt']['sha256'])
print(json.dumps({'remote_run':str(root),'files':files,'source_paths':sources}))
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
    accepted_stops = {}
    accepted_arm = manifest.get("accepted_early_stopped_arm") or {}
    if accepted_arm:
        accepted_stops[accepted_arm["name"]] = accepted_arm
    for arm in ARMS:
        acceptance = read(f"arms/{arm}/accepted-early-stop.json")
        if acceptance.get("status") != "accepted_early_stop":
            continue
        if (acceptance.get("accepted_by") not in ("user", "agent_under_standing_user_authorization")
                or acceptance.get("arm") != arm or acceptance.get("process_cessation", {}).get("confirmed") is not True):
            raise ValueError("Invalid accepted-stop identity or process cessation: " + arm)
        preserved = acceptance.get("preserved_checkpoints", {})
        selected = acceptance.get("selected_checkpoints", preserved)
        selectors = selected.get("selectors", selected)
        latest = acceptance.get("latest_checkpoint", preserved.get("latest", {}))
        accepted_stops[arm] = {
            "name": arm, "status": "accepted_early_stop",
            "updates": acceptance.get("durable_updates", latest.get("cursor", {}).get("updates")),
            "observed_updates": acceptance.get("observed_updates"), "epochs": acceptance.get("completed_epochs"),
            "accepted_by": acceptance["accepted_by"],
            "selectors": {key: value for key, value in selectors.items()
                          if key in ("minimum_validation_ce", "maximum_validation_accuracy")},
        }
    accepted_arm = accepted_stops.get("A128fixed", {})
    if accepted_arm and read("continuation-disposition.json") and not manifest.get("parent_campaign"):
        status = {**status, "status": "accepted_early_stop"}
        phase, active_arm = "awaiting_continuation", None
    queue_gate = read("queue-review-gate.json")
    queue_held = queue_gate.get("status") == "queue_paused_training_continues"
    held_arms = set(queue_gate.get("blocked_arms", [])) if queue_held else set()
    queue_message = None
    if queue_held:
        training_arm = queue_gate.get("active_arm", "B32fixed")
        training_status = read(f"arms/{training_arm}/status.json").get("status")
        if training_arm in accepted_stops:
            status = {**status, "status": "awaiting_assessment"}
            phase, active_arm = "post_B_assessment", None
            queue_message = ("B training **stopped early under user authorization; awaiting post-B assessment**. "
                             "Both checkpoint winners and the latest resumable checkpoint are preserved. "
                             "Raw trainer/process status is retained and may still show its pre-stop state.")
        elif training_status == "completed" or read(f"arms/{training_arm}/results.json").get("status") == "completed":
            status = {**status, "status": "awaiting_assessment"}
            phase, active_arm = "post_B_assessment", None
            queue_message = "B training completed; **awaiting post-B assessment**. The adaptive queue remains held."
        elif training_status == "running":
            queue_message = "**B training continues; adaptive queue held for post-B assessment.**"
        else:
            queue_message = (f"B trainer status: **{training_status or 'unavailable'}**. "
                             "The adaptive queue remains held for post-B assessment.")
        queue_message += " C/D will stay held while reduced-encoder quality and the necessary additional brain plasticity are assessed."
    def count(value):
        return f"{value:,}" if isinstance(value, int) else "—"
    report = ["# Connectorch encoder experiment — partial validation", "",
              f"Snapshot: {now}. Host: {host}.", "",
              f"Campaign status: **{status.get('status', 'waiting')}**. Phase: **{phase}**. "
              f"Current arm: **{active_arm or '—'}**.", "", baseline,
              "All arms retain the full output head, eight explicit input lags, 49,393 neurons and 9,050,172 base edges.",
              "Bounded arms add 18,322 source/destination cell-type gains; base-edge multipliers stay within 0.9–1.1. Original neuron gains remain unconstrained.", ""]
    if queue_message:
        report.extend([queue_message, ""])
    for arm, accepted in accepted_stops.items():
        origin = " from the parent campaign" if arm == accepted_arm.get("name") and manifest.get("parent_campaign") else ""
        report.extend([f"{arm}: **accepted early stop**, {accepted.get('epochs', '—')} completed epochs; "
                       f"{count(accepted.get('updates'))} durable updates and {count(accepted.get('observed_updates'))} observed updates. "
                       f"Its complete history and both checkpoint winners are retained{origin}. "
                       f"Retained-best comparisons use unequal training budgets; {arm} did not complete the 44-epoch schedule.", ""])
        if accepted.get("accepted_by") == "agent_under_standing_user_authorization":
            report.extend(["The agent made the operational stop under standing user authorization; "
                           "this receipt does not represent a new, arm-specific user approval.", ""])
    report.extend([
              "| Arm | Width | Edge gains | Status | Updates | Minimum CE (accuracy; update) | Maximum accuracy (CE; update) |",
              "|---|---:|---|---|---:|---|---|"])
    histories, selections, provenance = {}, {}, {}
    for arm in ARMS:
        prefix = "arms/" + arm + "/"
        arm_status = read(prefix + "status.json") or read(prefix + "process-status.json")
        metrics_text = files.get(prefix + "metrics.jsonl", {}).get("text", "")
        # JSON Lines uses physical LF; Unicode separators may be string data.
        lines = metrics_text.split("\n")
        rows = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1 and not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1 and not metrics_text.endswith("\n"):
                    continue
                raise
            if row.get("event") == "validation":
                rows.append(row)
        histories[arm] = rows
        selections[arm] = (accepted_stops.get(arm, {}).get("selectors")
                           or read(prefix + "selected-checkpoints.json").get("selectors", {}))
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
        if arm in accepted_stops:
            arm_state = "accepted early stop"
        elif arm in held_arms and arm_state in ("pending", "waiting"):
            arm_state = "held for post-B assessment"
        elif arm_status.get("activity"):
            arm_state += " / " + arm_status["activity"]
        updates = arm_status.get("updates", rows[-1]["updates"] if rows else 0)
        if arm in accepted_stops:
            updates = accepted_stops[arm].get("updates") or updates
        report.append(f"| {arm} | {128 if '128' in arm else 32} | {'fixed' if 'fixed' in arm else 'bounded10'} | "
                      f"{arm_state} | {updates:,} | {ce} | {accuracy} |")
    report.extend(["", "Winner columns use the saved checkpoint receipts when available. A newer validation can "
                   "appear in the history while its checkpoint is still being saved; exact metric ties retain the earlier checkpoint."])
    if read("failure.json"):
        if accepted_arm and read("continuation-disposition.json") and not manifest.get("parent_campaign"):
            report.extend(["", "The parent controller was interrupted for the accepted stop of A. "
                           "Its raw failure record is preserved; no continuation failure is reported in this parent snapshot."])
        else:
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
    p.add_argument("--remote-run", help="Exact remote run; default selects the continuation when its manifest exists, otherwise the original campaign")
    p.add_argument("--output", type=Path, default=ROOT / "results/connectorch-encoder-v1")
    args = p.parse_args()
    import cluster_runner as cr
    request = {"candidates": [args.remote_run] if args.remote_run else [DEFAULT_REMOTE_RUN + "-continuation", DEFAULT_REMOTE_RUN],
               "files": FILES, "parent_files": PARENT_FILES}
    command = "python3 -c " + shlex.quote(REMOTE) + " " + shlex.quote(json.dumps(request))
    result = cr.run(command, host=args.host, timeout=30)
    if result.exit_code:
        raise RuntimeError(result.stderr or result.stdout)
    snapshot = json.loads(result.stdout)
    files = snapshot["files"]
    if not files:
        raise RuntimeError("No campaign artifacts found")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, item in files.items():
        if name not in SNAPSHOT_FILES:
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
                     "remote_run": snapshot["remote_run"], "source_paths": snapshot["source_paths"],
                     "sha256": hashes}, indent=2) + "\n")
    print("\n".join(report[:17]))


if __name__ == "__main__":
    main()
