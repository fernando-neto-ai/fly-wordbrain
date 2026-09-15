"""Progress reports distinguish live training from a verified operational stop."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("connectorch_progress", ROOT / "scripts/refresh_connectorch_progress.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def item(value):
    return {"text": json.dumps(value)}


def selector(update, ce, acc, path="stopped-checkpoints/best.pt"):
    return {"cursor": {"updates": update}, "validation": {"cross_entropy": ce, "top1_accuracy": acc},
            "path": path, "sha256": "saved-checkpoint-sha"}


def snapshot():
    return {"manifest.json": item({"baseline_acceptance_mode": "user_accepted_early_stop"}),
            "campaign-status.json": item({"status": "running", "phase": "training", "arm": "B32fixed"}),
            "queue-review-gate.json": item({"status": "queue_paused_training_continues", "active_arm": "B32fixed",
                                            "blocked_arms": ["C128bounded", "D32bounded"]}),
            "arms/B32fixed/status.json": item({"status": "running", "activity": "training", "updates": 15555})}


def acceptance(arm="B32fixed", accepted_by="agent_under_standing_user_authorization"):
    return {"format_version": 1, "status": "accepted_early_stop", "accepted_by": accepted_by,
            "arm": arm, "observed_updates": 15555, "durable_updates": 15500, "completed_epochs": 12,
            "selected_checkpoints": {"format_version": 1, "selectors": {
                "minimum_validation_ce": selector(3000, 4.8, .30),
                "maximum_validation_accuracy": selector(15000, 5.4, .33, "stopped-checkpoints/best-accuracy.pt")}},
            "latest_checkpoint": {"cursor": {"updates": 15500}},
            "process_cessation": {"confirmed": True}, "test_evaluated": False}


def test_live_b_without_stop_receipt_keeps_training_and_queue_held():
    report = "\n".join(m.render_report(snapshot(), "macm3", "now"))
    assert "B training continues; adaptive queue held" in report
    assert "| B32fixed | 32 | fixed | running / training | 15,555 |" in report
    assert report.count("| held for post-B assessment |") == 2
    assert "B32fixed: **accepted early stop**" not in report


def test_b_stop_overrides_stale_running_status_and_uses_preserved_winners():
    files = snapshot()
    receipt = acceptance()
    files["arms/B32fixed/accepted-early-stop.json"] = item(receipt)
    files["arms/B32fixed/selected-checkpoints.json"] = item({"selectors": {
        "minimum_validation_ce": selector(100, 9., .10, "stale.pt")}})
    original_status = files["arms/B32fixed/status.json"]["text"]
    report = "\n".join(m.render_report(files, "macm3", "now"))
    assert "Campaign status: **awaiting_assessment**. Phase: **post_B_assessment**" in report
    assert "Current arm: **—**" in report
    assert "| B32fixed | 32 | fixed | accepted early stop | 15,500 | 4.8000 (30.00%; 3,000) | 33.00% (5.4000; 15,000) |" in report
    assert "not represent a new, arm-specific user approval" in report
    assert "B32fixed did not complete the 44-epoch schedule" in report
    assert "stale.pt" not in report
    assert files["arms/B32fixed/status.json"]["text"] == original_status
    assert report.count("| held for post-B assessment |") == 2


def test_a_legacy_receipt_and_parent_manifest_still_render():
    files = snapshot()
    receipt = acceptance("A128fixed", "user")
    receipt["preserved_checkpoints"] = {**receipt.pop("selected_checkpoints")["selectors"],
                                       "latest": receipt.pop("latest_checkpoint")}
    receipt.pop("durable_updates")
    files["arms/A128fixed/accepted-early-stop.json"] = item(receipt)
    files["manifest.json"] = item({"parent_campaign": {"path": "parent/manifest.json"},
                                   "accepted_early_stopped_arm": {"name": "A128fixed", "updates": 15500}})
    report = "\n".join(m.render_report(files, "macm3", "now"))
    assert "| A128fixed | 128 | fixed | accepted early stop | 15,500 |" in report
    assert "retained from the parent campaign" in report
    assert "B training continues" in report


def test_unicode_line_separators_inside_json_strings_do_not_split_records():
    files = snapshot()
    row = {"event": "validation", "updates": 100, "epoch": 0,
           "validation": {"cross_entropy": 5.4, "top1_accuracy": .3},
           "note": "first\u2028second\u2029third\u0085fourth"}
    files["arms/B32fixed/metrics.jsonl"] = {"text": json.dumps(row, ensure_ascii=False) + "\n"}
    report = "\n".join(m.render_report(files, "macm3", "now"))
    assert "| 100 | 0 | 5.4000 | 30.00% |" in report


@pytest.mark.parametrize("field,value", [("accepted_by", "unrecognized"), ("arm", "A128fixed"),
                                         ("process_cessation", {"confirmed": False})])
def test_invalid_stop_receipt_is_not_silently_treated_as_success(field, value):
    files = snapshot()
    receipt = acceptance()
    receipt[field] = value
    files["arms/B32fixed/accepted-early-stop.json"] = item(receipt)
    with pytest.raises(ValueError, match="Invalid accepted-stop"):
        m.render_report(files, "macm3", "now")


@pytest.mark.parametrize("tampered", [False, True])
def test_fetch_checks_stop_receipt_binding_in_continuation_arm(tmp_path, tampered):
    (tmp_path / "manifest.json").write_text("{}")
    directory = tmp_path / "arms/B32fixed"
    directory.mkdir(parents=True)
    stop = directory / "stop-receipt.json"
    stop.write_text(json.dumps({"process_cessation": {"confirmed": True}}))
    receipt = acceptance()
    receipt["stop_receipt"] = {"path": str(stop), "sha256": hashlib.sha256(stop.read_bytes()).hexdigest()}
    (directory / "accepted-early-stop.json").write_text(json.dumps(receipt))
    if tampered:
        stop.write_text("{}")
    request = {"candidates": [str(tmp_path)], "files": m.FILES, "parent_files": m.PARENT_FILES}
    result = subprocess.run([sys.executable, "-c", m.REMOTE, json.dumps(request)], capture_output=True, text=True)
    if tampered:
        assert result.returncode != 0
        assert "Bound artifact changed" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        fetched = json.loads(result.stdout)
        assert fetched["source_paths"]["arms/B32fixed/stop-receipt.json"] == str(stop)
