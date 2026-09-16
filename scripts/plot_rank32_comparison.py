#!/usr/bin/env python3
"""Render stopped H/G validation histories from local JSON receipts only.

No model imports, inference, network access, or training. All validation rows
are plotted without smoothing; each selector is checked against full history.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
ARM_SPECS = (
    ("G32rank64fixed", "G · rank 64 · 3.23M readout", "#16718B", 64, 3226688),
    ("H32rank32fixed", "H · rank 32 · 1.61M readout", "#C35B35", 32, 1613344),
)


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_arm(directory, specification):
    directory = Path(directory)
    name, label, color, rank, readout = specification
    manifest = read_json(directory / "manifest.json")
    accepted = read_json(directory / "accepted-early-stop.json")
    require(accepted.get("arm") == name and accepted.get("status") == "accepted_early_stop"
            and accepted.get("process_cessation", {}).get("confirmed") is True,
            "Expected the preserved stopped arm: " + name)
    require(manifest["config"]["readout_rank"] == rank and manifest["config"]["d_embed"] == 32
            and manifest["config"]["history_length"] == 8 and manifest["config"]["plasticity"] == "fixed"
            and manifest["config"]["seed"] == 42 and manifest["parameter_counts"]["readout"] == readout,
            "Architecture/recipe differs from the plotted comparison: " + name)
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().split("\n") if line.strip()]
    rows = [row for row in rows if row.get("event") == "validation"]
    require(rows and rows[0]["updates"] == 0 and rows[-1]["updates"] == accepted["durable_updates"],
            "The complete initialization-to-stop validation history is required")
    require(all(a["updates"] <= b["updates"] for a, b in zip(rows, rows[1:])), "Validation history is out of order")
    for row in rows:
        metric = row["validation"]
        require(metric["stories"] == 100 and metric["tokens"] == 21874
                and math.isfinite(metric["cross_entropy"]) and metric["cross_entropy"] >= 0
                and 0 <= metric["correct"] <= metric["tokens"]
                and abs(metric["top1_accuracy"] - metric["correct"] / metric["tokens"]) < 1e-12,
                "Invalid or unequal validation population")
    selectors = accepted["selected_checkpoints"]["selectors"]
    for selector, key, choose in (("minimum_validation_ce", "cross_entropy", min),
                                 ("maximum_validation_accuracy", "top1_accuracy", max)):
        winner = choose(rows, key=lambda row: row["validation"][key])
        entry = selectors[selector]
        require(entry["cursor"]["updates"] == winner["updates"] and entry["validation"] == winner["validation"],
                "Retained selector differs from full history: " + name + "/" + selector)
    # These are small acceptance-bound artifacts, not checkpoint binaries.
    for path, expected in accepted["artifact_sha256"].items():
        filename = Path(path).name
        if filename in ("manifest.json", "metrics.jsonl"):
            require(hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected,
                    "Acceptance-bound plotted input changed: " + filename)
    return {"name": name, "label": label, "color": color, "rank": rank, "rows": rows,
            "manifest": manifest, "accepted": accepted, "selectors": selectors}


def render(arms, output):
    for key in ("data_sha256", "groups_sha256", "frozen_buffers_sha256", "trainer_sources_sha256"):
        require(arms[0]["manifest"][key] == arms[1]["manifest"][key], "G/H identity differs: " + key)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.titleweight": "semibold", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#84929A",
                         "axes.labelcolor": "#293B46", "text.color": "#263945",
                         "xtick.color": "#52616B", "ytick.color": "#52616B",
                         "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 7.4), sharex=True)
    figure.subplots_adjust(left=.075, right=.965, top=.805, bottom=.36, wspace=.23)
    figure.patch.set_facecolor("white")
    figure.suptitle("Rank 32 versus rank 64: validation over the full training run", x=.075, y=.978,
                   ha="left", fontsize=17, fontweight="semibold")
    figure.text(.075, .929, "Same encoder width and full modeled connectome · independently retained checkpoints · accepted early stops",
                fontsize=10.8, color="#596B77")
    handles = []
    for arm in arms:
        updates = [row["updates"] for row in arm["rows"]]
        ce = [row["validation"]["cross_entropy"] for row in arm["rows"]]
        accuracy = [100 * row["validation"]["top1_accuracy"] for row in arm["rows"]]
        for axis, values, selector, key, multiplier in (
            (axes[0], ce, "minimum_validation_ce", "cross_entropy", 1),
            (axes[1], accuracy, "maximum_validation_accuracy", "top1_accuracy", 100),
        ):
            axis.plot(updates, values, color=arm["color"], linewidth=1.8, alpha=.96, zorder=3)
            best = arm["selectors"][selector]
            axis.scatter([best["cursor"]["updates"]], [multiplier * best["validation"][key]],
                         marker="*", s=145, color=arm["color"], edgecolor="white", linewidth=.7, zorder=6)
            stop = arm["accepted"]["durable_updates"]
            axis.axvline(stop, color=arm["color"], linestyle=(0, (3, 3)), linewidth=1, alpha=.65, zorder=2)
            axis.scatter([stop], [values[-1]], marker="o", s=23, facecolor="white",
                         edgecolor=arm["color"], linewidth=1.3, zorder=5)
        handles.append(Line2D([0], [0], color=arm["color"], linewidth=2.5, label=arm["label"]))
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(.069, .910), ncol=2,
                  frameon=False, handlelength=2.5, columnspacing=2.5, fontsize=11)
    axes[0].set_title("Cross-entropy ↓", loc="left", pad=13, fontsize=12)
    axes[1].set_title("Next-token accuracy ↑", loc="left", pad=13, fontsize=12)
    axes[0].set_ylabel("Cross-entropy (nats)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(2.8, 7.2)
    axes[0].yaxis.set_major_locator(MultipleLocator(1))
    axes[1].set_ylim(0, 40)
    axes[1].yaxis.set_major_locator(MultipleLocator(10))
    for axis in axes:
        axis.set_xlim(0, 18000)
        axis.set_xlabel("Optimizer updates", labelpad=9)
        axis.xaxis.set_major_locator(MultipleLocator(4000))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
        axis.grid(axis="y", color="#E2E8EC", linewidth=.8, zorder=0)
        axis.tick_params(axis="both", which="both", length=3)
    figure.text(.075, .276, "Retained winners and durable stopping points", fontsize=11, fontweight="semibold")
    figure.text(.965, .276, "★ = selector winner   ·   dashed line / open circle = stop", ha="right", fontsize=9.5, color="#596B77")
    table_axis = figure.add_axes([.075, .114, .89, .135])
    table_axis.axis("off")
    table_rows = []
    for arm in arms:
        ce, acc = (arm["selectors"][key] for key in ("minimum_validation_ce", "maximum_validation_accuracy"))
        table_rows.append([f"{arm['name'][0]} · rank {arm['rank']}",
                           f"{ce['validation']['cross_entropy']:.6f}  ({ce['cursor']['updates']:,})",
                           f"{100 * acc['validation']['top1_accuracy']:.4f}%  ({acc['cursor']['updates']:,})",
                           f"{arm['accepted']['durable_updates']:,}"])
    table = table_axis.table(cellText=table_rows,
        colLabels=["Model", "Lowest CE (update)", "Highest accuracy (update)", "Durable stop"],
        colWidths=[.18, .30, .32, .20], cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        cell.set_facecolor("#EAF0F3" if row == 0 else ("#F4F7F9" if row == 1 else "#FAF5F1"))
        if row == 0:
            cell.set_text_props(weight="semibold", color="#405460")
        elif column == 0:
            cell.set_text_props(weight="semibold", color=arms[row - 1]["color"])
    figure.text(.075, .061, "100 validation stories · 21,874 next-BPE-token targets · seed 42 · all evaluations shown without smoothing",
                fontsize=9, color="#596B77")
    figure.text(.075, .033, "Checkpoints were selected on this validation set. Stop budgets differ; rank also changes initialization scale. No reserved-test scores.",
                fontsize=9, color="#596B77")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white", metadata={"Description": "Complete local stopped G64 and H32 validation histories; no model inference."})
    figure.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g", type=Path, default=ROOT / "results/connectorch-rank32-v1/baseline")
    parser.add_argument("--h", type=Path, default=ROOT / "results/connectorch-rank32-v1/arms/H32rank32fixed")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/reports/figures/rank32-vs-rank64-validation.png")
    args = parser.parse_args()
    arms = [load_arm(path, specification) for path, specification in zip((args.g, args.h), ARM_SPECS)]
    render(arms, args.output)
    print(json.dumps({"png": str(args.output), "svg": str(args.output.with_suffix('.svg')),
                      "validation_rows": {arm["name"]: len(arm["rows"]) for arm in arms}}))


if __name__ == "__main__":
    main()
