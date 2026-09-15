"""Read-only partial/final reports for the bounded delayed-retention experiment.

Run: python -m scripts.report_retention --output results/retention-step3072-v1
The script reads atomic run artifacts and writes summary.json, report.md, and
learning-curves.png, and index.html. It never imports the trainer or changes execution sources.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.summarize_memory_probe import interval


ARMS = ("treatment", "decoder_only")
TASKS = ("identity", "order")
CONDITIONS = ("fast_on", "fast_off", "clear_fast_before_final")


def _read(directory, name, default):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else default


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite " + name)
    return value


def _test_record(metric, rows, task, seed, resamples):
    indices, labels = metric["row_indices"], metric["labels"]
    predicted = np.asarray(metric["predictions"])
    expected = {i for i, row in enumerate(rows) if row["split"] == "test" and row["task"] == task}
    if (not indices or len(indices) != len(labels) or len(set(indices)) != len(indices)
            or any(type(index) is not int for index in indices) or set(indices) != expected
            or predicted.shape != (len(indices), 5) or metric["prediction_times"] != [4, 5, 6, 7, 8]):
        raise ValueError("Test metrics do not cover the complete aligned task/time population")
    classes = 10 if task == "identity" else 2
    if predicted.dtype.kind not in "iu" or np.any(predicted < 0) or np.any(predicted >= classes):
        raise ValueError("Invalid predicted class")
    grouped = defaultdict(list)
    for index, label, prediction in zip(indices, labels, predicted[:, -1]):
        row = rows[index]
        if label != row["label"]:
            raise ValueError("Test metric label disagrees with source rows")
        grouped[row["group_id"]].append((row, int(prediction == label)))
    groups = []
    for group_id, values in sorted(grouped.items()):
        if sorted(row["label"] for row, _ in values) != list(range(classes)):
            raise ValueError("Test nuisance group is not complete and class balanced")
        sources = {row["source_story_id"] for row, _ in values}
        if len(sources) != 1:
            raise ValueError("Test nuisance group contains multiple source stories")
        groups.append({"group_id": group_id, "source_story_id": next(iter(sources)), "rows": len(values),
                       "correct": sum(correct for _, correct in values),
                       "accuracy": sum(correct for _, correct in values) / len(values)})
    correct = predicted == np.asarray(labels)[:, None]
    accuracy = correct.mean(0)
    if (metric["count"] != len(indices) or not np.allclose(accuracy, metric["accuracy"], atol=1e-6)
            or not np.array_equal(correct.sum(0), metric["correct"])):
        raise ValueError("Reported accuracy disagrees with test predictions")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, predicted[:, -1]), 1)
    return {"task": task, "classes": classes, "count": len(indices), "group_count": len(groups),
            "chance_accuracy": 1. / classes, "final_accuracy": float(accuracy[-1]),
            "final_cross_entropy": _finite(metric["cross_entropy"][-1], "test CE"),
            "prediction_times": [4, 5, 6, 7, 8], "accuracy_by_time": accuracy.tolist(),
            "cross_entropy_by_time": [_finite(value, "time CE") for value in metric["cross_entropy"]],
            "group_accuracy": groups, "final_confusion": confusion.tolist(),
            "group_accuracy_interval": interval([group["accuracy"] for group in groups], seed=seed, resamples=resamples)}


def _paired(left, right, seed, resamples):
    a = {group["group_id"]: group for group in left["group_accuracy"]}
    b = {group["group_id"]: group for group in right["group_accuracy"]}
    if set(a) != set(b) or any(a[key]["source_story_id"] != b[key]["source_story_id"] or a[key]["rows"] != b[key]["rows"] for key in a):
        raise ValueError("Paired results refer to different source groups")
    differences = [{"group_id": key, "accuracy_difference": a[key]["accuracy"] - b[key]["accuracy"]} for key in sorted(a)]
    return {"group_differences": differences,
            "accuracy_difference_interval": interval([row["accuracy_difference"] for row in differences], seed=seed, resamples=resamples),
            "cross_entropy_difference": left["final_cross_entropy"] - right["final_cross_entropy"]}


def build_summary(directory, *, seed=0, resamples=10000):
    directory = Path(directory)
    protocol = _read(directory, "protocol.json", {})
    progress = _read(directory, "progress.json", {})
    failure = _read(directory, "failure.json", None)
    validation = _read(directory, "validation.json", [])
    results = _read(directory, "results.json", {})
    rows = _read(directory, "rows.json", [])
    checks = _read(directory, "checks.json", {})
    status = "failed" if failure is not None else progress.get("status", "pending")
    if status == "completed" and results.get("status") != "completed":
        status = "incomplete"
    selected = results.get("selected_steps", {})
    curve, observed_steps = [], set()
    best_so_far = {arm: None for arm in ARMS}
    for record in validation:
        step = record["step"]
        if type(step) is not int or step < 0 or step in observed_steps or (curve and step <= curve[-1]["step"]):
            raise ValueError("Validation steps must be unique and increasing")
        observed_steps.add(step)
        item = {"step": step, "elapsed_seconds": _finite(record["elapsed_seconds"], "elapsed time"), "arms": {}}
        for arm in ARMS:
            item["arms"][arm] = {}
            for scope in ("validation", "train_monitor"):
                source = record[arm][scope]
                value = {task: {"accuracy": _finite(source[task]["accuracy"][-1], "accuracy"),
                                "cross_entropy": _finite(source[task]["cross_entropy"][-1], "CE"),
                                "count": source[task]["count"]} for task in TASKS}
                value["mean_final_ce"] = _finite(source["mean_final_ce"], "mean CE")
                if not math.isclose(value["mean_final_ce"], sum(value[task]["cross_entropy"] for task in TASKS)/2, abs_tol=1e-8):
                    raise ValueError("Validation task-average CE is inconsistent")
                item["arms"][arm][scope] = value
            ce = item["arms"][arm]["validation"]["mean_final_ce"]
            if best_so_far[arm] is None or ce < best_so_far[arm]["validation_mean_ce"]:
                best_so_far[arm] = {"step": step, "validation_mean_ce": ce}
        curve.append(item)
    for arm, step in selected.items():
        if arm not in best_so_far or best_so_far[arm] is None or best_so_far[arm]["step"] != step:
            raise ValueError("Selected checkpoint differs from strict validation-only selection")
    selection_state = {}
    for arm in ARMS:
        best = best_so_far[arm]
        if best is not None:
            step = selected.get(arm, best["step"])
            selection_state[arm] = {"step": step, "selection_final": arm in selected,
                "retention_training_updates": step, "before_retention_training": step == 0,
                "retention_trained_heads": step > 0,
                "retention_trained_rules": arm == "treatment" and step > 0,
                "description": "Initialization before retention training; circuit comes from the language-trained starting checkpoint"
                    if step == 0 else "Checkpoint after {} retention updates".format(step)}
    test = {}
    for arm, conditions in results.get("arms", {}).items():
        if arm not in ARMS:
            raise ValueError("Unknown test arm")
        if arm not in selected:
            raise ValueError("Test results require a finalized validation-selected checkpoint")
        for condition, metrics in conditions.items():
            if condition not in CONDITIONS:
                raise ValueError("Unknown test intervention")
            for task in TASKS:
                key = arm + "/" + condition + "/" + task
                test[key] = _test_record(metrics[task], rows, task, seed, resamples)
                test[key].update(selected_step=selected[arm],
                    before_retention_training=selected[arm] == 0,
                    retention_trained_rules=selection_state[arm]["retention_trained_rules"])
    comparisons = {}
    for task in TASKS:
        pairs = [("treatment/fast_on/"+task, "decoder_only/fast_on/"+task, "treatment_minus_control")]
        pairs.extend((arm+"/fast_on/"+task, arm+"/"+condition+"/"+task, arm+"_fast_on_minus_"+condition)
                     for arm in ARMS for condition in ("fast_off", "clear_fast_before_final"))
        for left, right, name in pairs:
            if left in test and right in test:
                comparisons[task+"/"+name] = {"task": task, "first": left, "second": right,
                    "primary": name == "treatment_minus_control",
                    "first_selected_step": test[left]["selected_step"], "second_selected_step": test[right]["selected_step"],
                    "includes_pre_retention_initialization": test[left]["before_retention_training"] or test[right]["before_retention_training"],
                    "tests_retention_trained_rules": test[left]["retention_trained_rules"] or test[right]["retention_trained_rules"],
                    **_paired(test[left], test[right], seed, resamples)}
    return {"schema": 1, "kind": "retention_training_report", "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(), "progress": progress, "failure": failure,
        "snapshot": protocol.get("snapshot"), "config": {key: protocol.get(key) for key in
            ("updates", "batch_size", "per_task_per_batch", "seed", "heads", "brain_parameters",
             "total_treatment_trainable", "decoder_control_trainable", "objective", "head_optimizer", "rule_optimizer")},
        "sample": protocol.get("data", {}).get("rows_per_task_split"),
        "groups": protocol.get("data", {}).get("groups_per_task_split"),
        "validation_curve": curve, "best_validation_so_far": best_so_far, "selected_steps": selected,
        "selection_state": selection_state, "primary_prediction_time": 8,
        "test_status": results.get("status", "not_started"), "test_results": test, "paired_comparisons": comparisons,
        "checks": checks, "bootstrap": {"unit": "whole nuisance/source-story groups", "resamples": resamples, "seed": seed,
            "confidence": .95, "pointwise": True, "independent_word_samples_assumed": False},
        "source_sha256": {name: _sha(directory/name) for name in
            ("protocol.json", "progress.json", "validation.json", "results.json", "rows.json", "checks.json", "failure.json")
            if (directory/name).exists()},
        "caveats": ["One starting checkpoint and one training seed; these are controlled synthetic identity/order tasks, not language accuracy.",
            "Fresh groups are held out from retention training and earlier probes, but their source stories belong to the original language-training corpus.",
            "The frozen control learns matched heads from cached features; its fast rules remain at the starting checkpoint values.",
            "H clearing tests the final fast-weight read. Earlier effects of fast weights can persist in neural activity even when the final H state is cleared.",
            "If validation selects retention step 0, its tests and ablations evaluate the initialization, not learned retention parameters. They cannot establish a mechanistic failure of retention training.",
            "Group-bootstrap intervals describe this context sample; they are unadjusted across comparisons and exclude training-seed uncertainty.",
            "Earlier-time scores reuse the same final-only heads (untrained if step 0 is selected); they are not independently optimized probes, and low earlier-time accuracy does not establish failure to encode the input.",
            "No randomized-geometry control is present."]}


def plot_curves(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"treatment": "#176ec0", "decoder_only": "#cc7130"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    curve = summary["validation_curve"]
    for column, task in enumerate(TASKS):
        for arm in ARMS:
            for scope, style, alpha in (("validation", "-", 1.), ("train_monitor", "--", .55)):
                for row, field in ((0, "accuracy"), (1, "cross_entropy")):
                    values = [record["arms"][arm][scope][task][field] for record in curve]
                    axes[row, column].plot([record["step"] for record in curve], values,
                        color=colors[arm], linestyle=style, alpha=alpha, linewidth=2,
                        marker="o" if scope == "validation" else None, markersize=4,
                        label=arm.replace("_", " ")+" · "+scope.replace("_", " "))
        axes[0, column].axhline(.1 if task == "identity" else .5, color="#888888", linewidth=1, linestyle=":", label="chance")
        axes[0, column].set_ylim(-.025, 1.025)
        axes[0, column].set_title(task.title()+" at prediction 8", fontsize=13)
        axes[0, column].set_ylabel("Accuracy")
        axes[1, column].set_ylabel("Cross entropy (nats)")
        axes[1, column].set_xlabel("Optimizer update")
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        if not curve:
            axes[1, column].text(.5, .5, "Awaiting first validation", transform=axes[1, column].transAxes, ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    latest = curve[-1]["step"] if curve else 0
    fig.suptitle("Delayed retention · {} · latest validation update {}".format(summary["status"], latest), fontsize=16)
    fig.tight_layout(rect=(0, .08, 1, .94))
    temporary = Path(path).with_suffix(".png.partial")
    fig.savefig(temporary, format="png", dpi=160, facecolor="white")
    plt.close(fig)
    temporary.replace(path)


def report_text(summary, image_path=None):
    progress = summary["progress"]
    lines = ["# Delayed retention experiment", "",
        "Status: **{}**. Phase: `{}`. Latest reported update: **{} / {}**.".format(summary["status"],
            progress.get("phase", "completed" if summary["status"] == "completed" else "pending"),
            progress.get("step", progress.get("steps", 0)), summary["config"].get("updates") or "pending"), "",
        "Treatment trains 8,838 existing fast-rule/susceptibility parameters and 3,084 memory-head parameters. The decoder-only control trains the same 3,084 head parameters. Both optimize final prediction-8 identity/order cross entropy; each selects its own checkpoint by validation mean CE.", "",
        "![Validation and fixed training-monitor curves]({})".format(
            str(Path(image_path).resolve()) if image_path is not None else "learning-curves.png"), ""]
    if summary["failure"] is not None:
        lines.extend(["**The runner reported a failure.** Any completed metrics below are partial evidence; inspect `failure.json` before interpreting completion.", ""])
    if summary["groups"]:
        group_counts = summary["groups"]
        lines.extend(["Each task has **{} training, {} validation, and {} test nuisance groups**. Every identity group has ten label variants; every order group has two. Groups, rather than individual variants, are the bootstrap unit.".format(group_counts["train"], group_counts["val"], group_counts["test"]), ""])
    lines.extend(["| Arm | Best validation update so far | Mean validation CE |", "|---|---:|---:|"])
    for arm, best in summary["best_validation_so_far"].items():
        if best:
            lines.append("| {} | {} | {:.4f} |".format(arm, best["step"], best["validation_mean_ce"]))
    for arm, state in summary.get("selection_state", {}).items():
        if state["before_retention_training"]:
            if state["selection_final"]:
                lines.extend(["", "**{} selected step 0: no retention-training updates.** Its circuit is the language-trained starting checkpoint and its memory heads are at initialization. All reported tests and ablations for this arm evaluate that state. Validation did not select a trained retention checkpoint; these ablations cannot demonstrate that learned retention mechanisms failed.".format(arm)])
            else:
                lines.extend(["", "**{}: best validation so far is step 0**, before retention training. Selection has not been finalized. Later validation curves describe trained states, but the currently best state contains no retention updates.".format(arm)])
    lines.extend(["", "The primary endpoint is accuracy/CE at **prediction 8**. Equal argmax accuracy can coexist with different probabilities and cross entropy; a flat accuracy curve does not mean the outputs or gradients are identical."])
    if not summary["test_results"]:
        lines.extend(["", "Test results are pending. The learning curves show validation and a fixed training subset only."])
    else:
        lines.extend(["", "Test extraction status: **{}**. Test conditions use the already selected checkpoints; no condition selects another checkpoint.".format(summary["test_status"]), "",
            "| Arm | Selected retention step | Condition | Task | Accuracy | Group bootstrap 95% interval | CE | Groups |",
            "|---|---|---|---|---:|---:|---:|---:|"])
        for key, value in summary["test_results"].items():
            arm, condition, task = key.split("/")
            ci = value["group_accuracy_interval"]
            step_label = "0 (initialization)" if value["before_retention_training"] else str(value["selected_step"])
            lines.append("| {} | {} | {} | {} | {:.1f}% | {:.1f}%–{:.1f}% | {:.4f} | {} |".format(
                arm, step_label, condition, task, 100*value["final_accuracy"], 100*ci["low"], 100*ci["high"], value["final_cross_entropy"], value["group_count"]))
        lines.extend(["", "Paired accuracy differences use identical held-out nuisance groups. Positive values favor the first result; values are percentage points.", "",
            "| Task | First minus second | Difference | Group bootstrap 95% interval |", "|---|---|---:|---:|"])
        for value in summary["paired_comparisons"].values():
            ci = value["accuracy_difference_interval"]
            lines.append("| {} | {} (step {}) − {} (step {}) | {:+.1f} | {:+.1f} to {:+.1f} |".format(value["task"],
                "/".join(value["first"].split("/")[:2]), value["first_selected_step"],
                "/".join(value["second"].split("/")[:2]), value["second_selected_step"],
                100*ci["estimate"], 100*ci["low"], 100*ci["high"]))
    lines.extend(["", "`fast_off` disables fast writes/reads throughout. `clear_fast_before_final` zeros only H after word 7 and before prediction 8, retaining neural activity and eligibility.", ""])
    lines.extend("- "+value for value in summary["caveats"])
    return "\n".join(lines)+"\n"


def dashboard_html():
    return r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Delayed recall · Fly circuit</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#182633;font:15px/1.5 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:34px 24px}h1{font-size:32px;letter-spacing:-.8px;margin:5px 0}h2{font-size:18px;margin:0 0 12px}p{margin:8px 0;color:#52616e}.eyebrow{font-size:12px;letter-spacing:1.4px;text-transform:uppercase;color:#687581}.cards,.plots{display:grid;gap:16px;margin-top:22px}.cards{grid-template-columns:repeat(3,1fr)}.plots{grid-template-columns:1fr 1fr}.card,.panel{padding:20px;background:white;border:1px solid #dce3e8;border-radius:14px}.card strong{display:block;font-size:23px;margin-top:4px}.label{font-size:12px;color:#687581;text-transform:uppercase;letter-spacing:.6px}.legend{display:flex;flex-wrap:wrap;gap:20px;margin:18px 0 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}.blue{background:#176ec0}.orange{background:#cc7130}svg{width:100%;display:block}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e7ebef;padding:10px 8px}th{color:#687581;font-size:12px}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.tablewrap{overflow:auto}.notice{border-radius:10px;background:#fff0da;color:#765017;padding:12px 16px;margin-top:16px;display:none}.footer{margin-top:24px;font-size:13px;color:#63717d}a{color:#176ec0;margin-right:16px}.wide{margin-top:18px}.muted{font-size:12px;color:#687581}@media(max-width:760px){main{padding:22px 14px}.cards,.plots{grid-template-columns:1fr}.card strong{font-size:20px}h1{font-size:27px}}
</style>
<main><div class="eyebrow">Controlled memory experiment</div><h1>Can the fly retain an earlier word?</h1>
<p>Identity and word order at prediction 8. These curves track the new retention experiment only.</p>
<div id="notice" class="notice"></div>
<div class="cards"><div class="card"><span class="label">Run phase</span><strong id="phase">Loading local snapshot…</strong><span id="phaseDetail" class="muted"></span></div><div class="card"><span class="label">Training updates</span><strong id="step">—</strong><span id="validated" class="muted">Waiting for validation</span></div><div class="card"><span class="label">Last run update</span><strong id="updated" style="font-size:18px">—</strong><span id="age" class="muted">This page checks local files every 15 seconds</span></div></div>
<div class="legend"><span><i class="dot blue"></i>Fast rules + heads · 11,922 trainable parameters</span><span><i class="dot orange"></i>Frozen circuit + heads · 3,084 trainable parameters</span></div>
<div class="plots"><section class="panel"><h2>Identity · validation accuracy</h2><svg id="identity" viewBox="0 0 520 260" role="img" aria-label="Identity validation accuracy over optimizer updates"></svg><p class="muted">10 classes · chance 10%</p></section><section class="panel"><h2>Order · validation accuracy</h2><svg id="order" viewBox="0 0 520 260" role="img" aria-label="Order validation accuracy over optimizer updates"></svg><p class="muted">2 classes · chance 50%</p></section></div>
<section class="panel wide"><h2 id="tableTitle">Latest validation</h2><p class="muted">Step 0 is the matched initialization before retention training, starting from the language-trained circuit. Later steps are learned retention states. Each arm selects its own checkpoint using mean validation cross entropy. Equal accuracy can coexist with different probabilities and CE.</p><p id="selectionNote"></p><div class="tablewrap"><table><thead><tr><th>Task</th><th>Arm</th><th class="num">Step 0 accuracy</th><th class="num">Latest accuracy</th><th class="num">Latest CE</th></tr></thead><tbody id="validationRows"><tr><td colspan="5">Validation has not started.</td></tr></tbody></table></div></section>
<section class="panel wide" id="tests" hidden><h2>Selected-checkpoint test results · prediction 8</h2><p id="testNote" class="muted"></p><div class="tablewrap"><table><thead><tr><th>Task</th><th>Arm / condition</th><th class="num">Selected step</th><th class="num">Accuracy</th><th class="num">Group 95% interval</th><th class="num">CE</th></tr></thead><tbody id="testRows"></tbody></table></div></section>
<div class="footer"><p id="context">Fresh nuisance contexts are held out from retention fitting. Their source stories belong to the original language-training corpus.</p><p>Both arms train only final recall. Fast-OFF changes the whole trajectory; clearing H before prediction 8 tests the final fast-weight read. If step 0 is selected, these ablations evaluate the initialization and cannot establish failure of learned retention mechanisms.</p><p>One starting checkpoint and one seed. Earlier-time scores reuse the same final-only head; low early accuracy is not evidence of an input-encoding failure. This is a synthetic memory test, not next-word language accuracy.</p><p><a href="report.md">Report</a><a href="summary.json">Summary JSON</a><a href="learning-curves.png">Accuracy and CE chart</a><a href="checks.json">Run checks</a></p></div>
</main><script>
'use strict';
const names={treatment:'Fast rules + heads',decoder_only:'Frozen circuit + heads'},colors={treatment:'#176ec0',decoder_only:'#cc7130'};
const phaseNames={loading:'Loading starting circuit',caching_frozen_train_validation:'Caching frozen features',preflight:'Checking gradients and restoration',training:'Training memory',validation:'Measuring validation',completed:'Completed'};
const el=id=>document.getElementById(id),pct=x=>(100*x).toFixed(1)+'%',num=x=>Number.isFinite(x)?x.toFixed(4):'—';
async function read(name){const response=await fetch(name+'?t='+Date.now(),{cache:'no-store'});if(response.status===404)return null;if(!response.ok)throw Error(name+': HTTP '+response.status);return response.json()}
function svgNode(svg,tag,attrs,text){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));if(text!==undefined)node.textContent=text;svg.append(node);return node}
function graph(id,data,total){const svg=el(id);svg.replaceChildren();const x=s=>44+454*s/Math.max(total,1),y=v=>218-190*v;
 for(const value of [0,.25,.5,.75,1]){svgNode(svg,'line',{x1:44,x2:498,y1:y(value),y2:y(value),stroke:'#e3e9ee'});svgNode(svg,'text',{x:36,y:y(value)+4,'text-anchor':'end',fill:'#6c7782','font-size':11},Math.round(value*100)+'%')}
 for(let k=0;k<=4;k++){const step=Math.round(total*k/4);svgNode(svg,'text',{x:x(step),y:239,'text-anchor':'middle',fill:'#6c7782','font-size':11},step)}
 svgNode(svg,'text',{x:270,y:257,'text-anchor':'middle',fill:'#6c7782','font-size':11},'Optimizer update');const chance=id==='identity'?.1:.5;svgNode(svg,'line',{x1:44,x2:498,y1:y(chance),y2:y(chance),stroke:'#87929b','stroke-dasharray':'3 4'});
 for(const arm of ['treatment','decoder_only']){const points=data.map(r=>({step:r.step,value:r[arm].validation[id].accuracy.at(-1)}));if(points.length)svgNode(svg,'polyline',{points:points.map(p=>x(p.step)+','+y(p.value)).join(' '),fill:'none',stroke:colors[arm],'stroke-width':2.5});for(const p of points){const circle=svgNode(svg,'circle',{cx:x(p.step),cy:y(p.value),r:4,fill:colors[arm]});svgNode(circle,'title',{},names[arm]+' · update '+p.step+' · '+pct(p.value))}}
 if(!data.length)svgNode(svg,'text',{x:270,y:120,'text-anchor':'middle',fill:'#77828c','font-size':13},'Awaiting first validation');}
function row(body,values){const tr=document.createElement('tr');values.forEach((value,i)=>{const td=document.createElement('td');td.textContent=value;if(i>1)td.className='num';tr.append(td)});body.append(tr)}
async function refresh(){try{const [protocol,progress,validation,summary,failure]=await Promise.all(['protocol.json','progress.json','validation.json','summary.json','failure.json'].map(read));const data=validation||[],p=progress||{},status=failure?'failed':p.status||'pending',latest=data.at(-1),initial=data.find(r=>r.step===0),total=protocol?.updates||256;
 const phase=status==='failed'?'Runner reported a failure':status==='completed'?'Completed':phaseNames[p.phase]||(p.phase?.startsWith('test_')?'Evaluating selected checkpoint':p.phase||'Waiting for run snapshot');el('phase').textContent=phase;el('phaseDetail').textContent=p.total?Math.min(p.completed||0,p.total)+' / '+p.total+' windows':'';el('step').textContent=(p.step??p.steps??latest?.step??0)+' / '+total;el('validated').textContent=latest?'Latest validation: update '+latest.step:'Validation has not started';
 const stamp=p.timestamp_utc?new Date(p.timestamp_utc):null;el('updated').textContent=stamp?stamp.toLocaleString():'No timestamp yet';const seconds=stamp?Math.max(0,Math.round((Date.now()-stamp)/1000)):0;el('age').textContent=stamp?'Snapshot age '+(seconds<60?seconds+' seconds':Math.floor(seconds/60)+' minutes')+' · refresh every 15 seconds':'Refresh every 15 seconds';el('notice').style.display=failure?'block':'none';if(failure)el('notice').textContent='The runner reported a failure. Existing metrics may be partial. See failure.json and the report.';
 graph('identity',data,total);graph('order',data,total);el('tableTitle').textContent=latest?'Validation · update '+latest.step:'Latest validation';const body=el('validationRows');body.replaceChildren();if(latest){for(const task of ['identity','order'])for(const arm of ['treatment','decoder_only'])row(body,[task,names[arm],initial?pct(initial[arm].validation[task].accuracy.at(-1)):'—',pct(latest[arm].validation[task].accuracy.at(-1)),num(latest[arm].validation[task].cross_entropy.at(-1))])}else{row(body,['Waiting for the first validation result.','','','',''])}
 const selections=[];for(const arm of ['treatment','decoder_only']){let best=null;for(const record of data)if(best===null||record[arm].validation.mean_final_ce<best[arm].validation.mean_final_ce)best=record;if(best){const finalized=summary?.selected_steps?.[arm],step=finalized??best.step;selections.push(names[arm]+': '+(finalized===undefined?'best so far':'selected')+' step '+step+(step===0?' (before retention training)':''))}}el('selectionNote').textContent=selections.join(' · ');
 const tests=summary?.test_results||{};el('tests').hidden=!Object.keys(tests).length;const testBody=el('testRows');testBody.replaceChildren();for(const [key,value] of Object.entries(tests)){const [arm,condition,task]=key.split('/'),ci=value.group_accuracy_interval,step=value.selected_step??summary?.selected_steps?.[arm];row(testBody,[task,names[arm]+' / '+condition,step===0?'0 · initialization':String(step??'unknown'),pct(value.final_accuracy),pct(ci.low)+'–'+pct(ci.high),num(value.final_cross_entropy)])}const zeroSelected=Object.values(summary?.selected_steps||{}).some(step=>step===0);el('testNote').textContent='Test extraction: '+(summary?.test_status||'pending')+'. Whole nuisance groups are resampled for uncertainty; checkpoints were selected by validation.'+(zeroSelected?' A selected step-0 state has no retention updates: its comparisons and ablations do not test learned retention parameters.':'');
 if(protocol?.data?.groups_per_task_split){const g=protocol.data.groups_per_task_split;el('context').textContent=g.train+' training, '+g.val+' validation, and '+g.test+' test nuisance groups per task. All are fresh relative to the earlier probe. Source stories belong to the original language-training corpus.'}
 }catch(error){el('notice').style.display='block';el('notice').textContent='Could not refresh the local snapshot: '+error.message+'. Existing results remain visible.'}}
refresh();setInterval(refresh,15000);document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh()});
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.output, seed=args.seed, resamples=args.resamples)
    summary["reporter_sha256"] = _sha(__file__)
    plot_curves(summary, args.output / "learning-curves.png")
    for name, content in (("summary.json", json.dumps(summary, indent=2, allow_nan=False)+"\n"),
                          ("report.md", report_text(summary,args.output/"learning-curves.png")), ("index.html", dashboard_html())):
        temporary = args.output/(name+".partial")
        temporary.write_text(content)
        temporary.replace(args.output/name)
    print(json.dumps({"status": summary["status"], "validation_updates": len(summary["validation_curve"]),
                      "test_conditions": len(summary["test_results"]), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
