"""Selected step zero must remain distinct from trained retention mechanisms."""
import json

import numpy as np
import pytest

from scripts.report_retention import build_summary, report_text


def artifacts(path, selected_step=None):
    def write(name, value):
        (path / name).write_text(json.dumps(value))
    write("protocol.json", {"updates": 256, "data": {"groups_per_task_split": {"train": 64, "val": 16, "test": 32}}})
    write("progress.json", {"status": "running", "phase": "training", "step": 112})
    rows = []
    for task, classes in (("identity", 10), ("order", 2)):
        for group in range(2):
            for label in range(classes):
                rows.append({"task": task, "split": "test", "label": label,
                    "group_id": task + str(group), "source_story_id": task + str(group)})
    write("rows.json", rows)
    history = []
    for step in (0, 64):
        record = {"step": step, "elapsed_seconds": step + 1}
        for arm in ("treatment", "decoder_only"):
            ce = 1. if step == (selected_step or 0) else 2.
            scores = {task: {"accuracy": [1/classes]*5, "cross_entropy": [ce]*5, "count": classes*2}
                      for task, classes in (("identity", 10), ("order", 2))}
            scores["mean_final_ce"] = ce
            record[arm] = {"validation": scores, "train_monitor": scores}
        history.append(record)
    write("validation.json", history)
    if selected_step is not None:
        result = {"status": "completed", "selected_steps": {"treatment": selected_step, "decoder_only": selected_step}, "arms": {}}
        for arm in ("treatment", "decoder_only"):
            scores = {}
            for task in ("identity", "order"):
                indices = [i for i, row in enumerate(rows) if row["task"] == task]
                labels = [rows[i]["label"] for i in indices]
                predictions = np.zeros((len(labels), 5), dtype=np.int64)
                correct = predictions == np.asarray(labels)[:, None]
                scores[task] = {"count": len(labels), "row_indices": indices, "labels": labels,
                    "prediction_times": [4, 5, 6, 7, 8], "predictions": predictions.tolist(),
                    "accuracy": correct.mean(0).tolist(), "correct": correct.sum(0).tolist(), "cross_entropy": [2.]*5}
            result["arms"][arm] = {"fast_on": scores, "clear_fast_before_final": scores}
        write("results.json", result)


def test_partial_best_zero_is_provisional_and_preserves_absolute_image_link(tmp_path):
    artifacts(tmp_path)
    summary = build_summary(tmp_path)
    state = summary["selection_state"]["treatment"]
    assert state["before_retention_training"] and not state["selection_final"]
    report = report_text(summary, image_path=tmp_path / "plot.png")
    assert "Selection has not been finalized" in report
    assert str(tmp_path / "plot.png") in report
    assert "Equal argmax accuracy can coexist with different probabilities" in report
    assert "prediction 8" in report and not summary["test_results"]


@pytest.mark.parametrize("step", [0, 64])
def test_final_comparisons_identify_whether_retention_trained_rules_are_tested(tmp_path, step):
    artifacts(tmp_path, step)
    summary = build_summary(tmp_path, resamples=100)
    for value in summary["test_results"].values():
        assert value["selected_step"] == step
        assert value["before_retention_training"] is (step == 0)
    primary = summary["paired_comparisons"]["identity/treatment_minus_control"]
    assert primary["first_selected_step"] == primary["second_selected_step"] == step
    assert primary["includes_pre_retention_initialization"] is (step == 0)
    assert primary["tests_retention_trained_rules"] is (step > 0)
    report = report_text(summary)
    if step == 0:
        assert "selected step 0: no retention-training updates" in report
        assert "0 (initialization)" in report
        assert "cannot demonstrate that learned retention mechanisms failed" in report
    else:
        assert "selected step 0: no retention-training updates" not in report
    assert "low earlier-time accuracy does not establish failure to encode the input" in report
