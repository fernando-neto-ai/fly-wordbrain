"""Pure stop-decision fixtures: no native host, checkpoint, process or signal calls."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rank64_plateau_stop", ROOT / "scripts/stop_connectorch_rank64_plateau.py")
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)


def history(completed_epochs=8):
    rows = [{"updates": update, "epoch": completed_epochs, "reason": "update_interval",
             "validation": {"cross_entropy": 3.25, "top1_accuracy": .34}}
            for update in range(9000, 13001, 100)]
    # The historical winner precedes both trailing windows. Neither the latest
    # score nor the window endpoint is the global winner.
    rows[0]["validation"] = {"cross_entropy": 3.20, "top1_accuracy": .35}
    rows[30]["validation"] = {"cross_entropy": 3.18, "top1_accuracy": .352}
    for index, update in enumerate((9000, 10000, 11000)):
        rows.append({"updates": update, "epoch": completed_epochs - 2 + index, "reason": "epoch_end",
                     "validation": {"cross_entropy": 3.26, "top1_accuracy": .33},
                     "training_epoch_so_far": {"cross_entropy": 2.8 - index * .1,
                                                "top1_accuracy": .4 + index * .01, "tokens": 234796}})
    return sorted(rows, key=lambda row: row["updates"])


def test_plateau_uses_global_bests_and_two_complete_periodic_windows():
    result = stop.plateau(history(), {"epoch": 8})
    assert result["decision_validation_update"] == 13000
    assert result["minimum_ce_improvement"] == pytest.approx(.02)
    assert result["maximum_accuracy_improvement_percentage_points"] == pytest.approx(.2)
    assert result["completed_epochs"] == 8
    assert [(window["start_exclusive"], window["end_inclusive"], window["count"])
            for window in result["validation_windows"]] == [(11000, 12000, 10), (12000, 13000, 10)]
    assert [row["epoch"] for row in result["last_completed_epoch_training"]] == [6, 7, 8]


@pytest.mark.parametrize("accuracy", [.355, .3558974, .36])
def test_renewed_accuracy_gain_at_or_above_half_percentage_point_rejects(accuracy):
    rows = history()
    rows[-1]["validation"]["top1_accuracy"] = accuracy
    with pytest.raises(ValueError, match="Plateau no longer met"):
        stop.plateau(rows, {"epoch": 8})


@pytest.mark.parametrize("cross_entropy", [3.10, 3.0])
def test_ce_gain_at_or_above_point_one_rejects(cross_entropy):
    rows = history()
    rows[-1]["validation"]["cross_entropy"] = cross_entropy
    with pytest.raises(ValueError, match="Plateau no longer met"):
        stop.plateau(rows, {"epoch": 8})


def test_incomplete_epochs_or_inconsistent_live_epoch_rejects():
    with pytest.raises(ValueError, match="eight completed epochs"):
        stop.plateau(history(7), {"epoch": 7})
    with pytest.raises(ValueError, match="differs from durable epoch history"):
        stop.plateau(history(8), {"epoch": 9})


def test_missing_periodic_point_is_not_replaced_by_epoch_end_validation():
    rows = history()
    missing = next(row for row in rows if row["updates"] == 12000)
    missing["reason"] = "epoch_end"
    missing["training_epoch_so_far"] = {"cross_entropy": 2.5}
    with pytest.raises(ValueError, match="ten periodic validations"):
        stop.plateau(rows, {"epoch": 8})


@pytest.mark.parametrize("updates,expected", [(13004, False), (13005, True), (13045, True),
                                             (13046, False), (13100, False)])
def test_safe_interval_update_boundaries(updates, expected):
    status = {"status": "running", "pid": stop.TRAINER, "activity": "training", "updates": updates, "batch": 114}
    assert stop.safe_interval(status, 13000) is expected


@pytest.mark.parametrize("change,last_update", [({"activity": "validation"}, 13000),
    ({"activity": "final_test"}, 13000), ({"status": "failed"}, 13000),
    ({"pid": stop.TRAINER + 1}, 13000), ({"batch": 115}, 13000),
    ({}, 12900), ({}, 13100)])
def test_safe_interval_requires_training_identity_and_current_durable_generation(change, last_update):
    status = {"status": "running", "pid": stop.TRAINER, "activity": "training", "updates": 13020, "batch": 114}
    assert not stop.safe_interval({**status, **change}, last_update)
