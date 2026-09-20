#!/usr/bin/env python3
"""Paired comparison between two unified arms scored on the same stories.

`evaluate_unified_quality.py` compares each arm to the released reference, which answers
"is this arm good". It is the wrong baseline for "does adapting the connectome help":
that is a contrast between two arms that differ only in their plasticity setting, and
their marginal intervals against a third model overlap far more than a paired interval
would, because the two arms see identical stories.

That evaluator computes per-story records and then aggregates them away, so the paired
interval was never recoverable from its output. This rescales both arms on the same audit
population and runs the same paired whole-story bootstrap -- same estimator, same 10,000
resamples, same seed 1729 -- as every other interval in this repository.

Rescoring rather than reusing records raises one risk: that this script rebuilds an arm
differently from the evaluator that produced the published numbers, and silently reports a
contrast between two models nobody shipped. So the rebuild is *checked*. Each arm's
reproduced aggregate is compared against its own published `audit-language.json`, and a
mismatch beyond 1e-9 is a hard failure. The paired numbers are only printed once both arms
have been shown to be the ones already on record.
"""
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_connectorch as trainer
from evaluate_ngxson_quality import aggregate, paired_comparison
from evaluate_unified_quality import score
from fly_wordbrain.unified_model import UnifiedFly

TOLERANCE = 1e-9


def rebuild(config_name, model_path, groups_path, run, checkpoint, device):
    """Rebuild an arm in the form its config declares. Mirrors evaluate_unified_quality.main."""
    from transformers import AutoModelForCausalLM

    config = json.loads((ROOT / "experiments/configs" / (config_name + ".json")).read_text())
    training = config["training"]
    reference = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True,
                                                     local_files_only=True, torch_dtype=torch.float32)
    groups, _ = trainer.load_groups(groups_path)
    build = Namespace(d_embed=training["d_embed"], plasticity=training["plasticity"],
                      readout_rank=training["readout_rank"],
                      history_length=training["history_length"], seed=training["seed"],
                      leak=training.get("leak", "fixed"))
    model = trainer.build_model(reference, build, groups)
    form = config.get("unified", {})
    space = form.get("output_space", {})
    unified = UnifiedFly(model.brain, settle_steps=5, readout_rank=training["readout_rank"],
                         tokens=space.get("language_tokens", 1024),
                         moves=space.get("chess_moves", 1968),
                         classes=space.get("sentiment_classes", 0),
                         toxicity_classes=space.get("toxicity_classes", 0),
                         tasks=len(form.get("tasks", ["language", "chess"])),
                         task_cue=form.get("task_cue", False),
                         sentiment_pooling=form.get("sentiment_pooling", "mean"))
    model.unified = unified
    saved = torch.load(run / checkpoint, map_location="cpu", weights_only=False)
    trainer.restore_parameters(model, saved["parameters"])
    return model.to(device).eval(), unified, training["plasticity"]


def comparable_spaces(treatment_space, control_space, treatment, control):
    """Which of the two scored spaces can honestly be differenced between these arms.

    Arms trained on different task sets have different output-space sizes: a three-task arm
    carries a sentiment range a two-task arm never had. Their `language_range_only` scores
    renormalise over the same token logits and stay comparable, but `full_output_space`
    cross-entropies are sums over different class counts, so differencing them measures the
    head's width as much as the model's skill.
    """
    if treatment_space["language_tokens"] != control_space["language_tokens"]:
        raise SystemExit(
            f"{treatment} and {control} disagree on the language range itself "
            f"({treatment_space['language_tokens']} vs {control_space['language_tokens']} "
            f"tokens); nothing here is paired.")
    if treatment_space == control_space:
        return ["full_output_space", "language_range_only"], None
    return ["language_range_only"], (
        f"{treatment} predicts over {treatment_space['total']} classes and {control} over "
        f"{control_space['total']}; a difference of cross-entropies over different class "
        f"counts is not a difference in skill. Both renormalise over the same "
        f"{control_space['language_tokens']} token logits, so language_range_only is the "
        f"paired contrast.")


def verify(reproduced, published, arm, space):
    """Refuse to report a contrast between models that are not the published ones."""
    for key in ("cross_entropy", "top1_accuracy", "tokens"):
        got, want = reproduced[key], published[space][key]
        if abs(got - want) > TOLERANCE:
            raise SystemExit(f"{arm}: rebuilt model does not reproduce its published "
                             f"{space}.{key}: got {got!r}, on record {want!r}. "
                             f"This script's rebuild has drifted from the evaluator's.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", type=Path, required=True, help="Directory holding both arms")
    parser.add_argument("--treatment", required=True, help="Arm whose effect is being measured")
    parser.add_argument("--control", required=True, help="Arm it is measured against")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--audit-data", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.audit_data.read_text())["audit"]
    records, plasticity, spaces = {}, {}, {}
    for arm in (args.treatment, args.control):
        run = args.arms / arm
        published = json.loads((run / "audit-language.json").read_text())
        model, unified, how = rebuild(arm, args.model, args.groups, run, args.checkpoint, args.device)
        full, language, seconds = score(unified, model.brain, rows, args.device, unified.tokens)
        verify(aggregate(full), published, arm, "full_output_space")
        verify(aggregate(language), published, arm, "language_range_only")
        print(f"[{arm}] reproduced its published audit exactly "
              f"(CE {aggregate(language)['cross_entropy']:.6f}, {seconds:.1f}s)", flush=True)
        records[arm] = {"full_output_space": full, "language_range_only": language}
        plasticity[arm] = how
        # Measure the head, do not read it off the record. An evaluator once wrote
        # `tokens + moves` and omitted the sentiment classes, so a 2,994-wide head was
        # recorded as 2,992 -- and a comparability check that trusted that field passed
        # two arms as equal when they were not. A guard inherits the bugs of whatever it
        # reads, so this reads the model and treats a disagreeing record as stale.
        measured = {"total": unified.tokens + unified.moves + unified.classes,
                    "language_tokens": unified.tokens}
        recorded = {"total": published["outputs"]["total"],
                    "language_tokens": published["outputs"]["language"]}
        if measured != recorded:
            raise SystemExit(
                f"{arm}: the checkpoint emits {measured['total']} outputs but its "
                f"audit-language.json records {recorded['total']}. The record is stale -- "
                f"re-run evaluate_unified_quality.py for this arm before comparing.")
        spaces[arm] = measured
        del model, unified

    comparable, note = comparable_spaces(spaces[args.treatment], spaces[args.control],
                                         args.treatment, args.control)

    report = {
        "question": f"Does {args.treatment} differ from {args.control} on held-out language?",
        "treatment": args.treatment, "control": args.control,
        "plasticity": plasticity,
        "checkpoint": args.checkpoint,
        "direction": f"{args.treatment} minus {args.control}; negative cross-entropy means "
                     f"{args.treatment} is better",
        "rebuild_verified_against_published_audit": True,
        "stories": len(rows),
        "output_spaces": spaces,
        "spaces_compared": comparable,
    }
    if note:
        report["full_output_space_not_compared"] = note
    for space in comparable:
        report[space] = paired_comparison(records[args.treatment][space],
                                          records[args.control][space])
        report[space]["direction"] = report["direction"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
