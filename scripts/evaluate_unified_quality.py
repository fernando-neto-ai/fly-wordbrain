#!/usr/bin/env python3
"""Score a unified arm's language on the shared audit population, and separate two costs.

A unified arm predicts over 2,992 outputs -- 1,024 tokens and 1,968 chess moves -- while
every earlier arm predicted over 1,024. Comparing its cross-entropy to Stage 6's 2.9493
without saying so would be comparing a harder task to an easier one, so the two effects are
reported apart:

    full        cross-entropy over all 2,992 outputs. This is what the model actually
                does, and the honest headline for a shared output space.
    language    cross-entropy over the 1,024 token logits alone, renormalised. This is the
                same quantity every earlier arm reported, so it is what compares to 2.9493.
    leakage     the difference. Exactly what sharing the output space costs, in nats.

The arithmetic is not an approximation: renormalising a softmax over a subset and taking
the difference gives -log of the mass the model left on that subset, per token, which is
the leakage cost by definition.

Per-story records are written in the same shape the other evaluators use, so the paired
whole-story bootstrap can run against any arm's records without reshaping anything.
"""
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_connectorch as trainer
from evaluate_ngxson_quality import aggregate, paired_comparison
from fly_wordbrain.unified_model import UnifiedFly


@torch.no_grad()
def score(unified, brain, rows, device, tokens, batch_size=8, chunk_size=32, pad=0):
    """Whole stories, fresh cache per batch, chunked exactly as the other evaluators do."""
    full, language = [], []
    started = time.monotonic()
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        inputs, targets, mask = trainer.batch_tensors(group, device, pad)
        cache = None
        totals = torch.zeros(len(group), dtype=torch.float64)
        narrow = torch.zeros(len(group), dtype=torch.float64)
        counts = torch.zeros(len(group), dtype=torch.long)
        hits = torch.zeros_like(counts)
        for offset in range(0, inputs.shape[1], chunk_size):
            sl = slice(offset, offset + chunk_size)
            logits, _, cache = unified.language(inputs[:, sl], attention_mask=mask[:, sl],
                                                cache_params=cache)
            if not bool(torch.isfinite(logits).all()):
                raise SystemExit("Nonfinite logits while scoring")
            target = targets[:, sl]
            valid = target.ne(pad)
            wide = F.cross_entropy(logits.flatten(0, 1).float(), target.flatten(),
                                   ignore_index=pad, reduction="none").reshape_as(target)
            # The same targets against a softmax restricted to the token range.
            inside = F.cross_entropy(logits[..., :tokens].flatten(0, 1).float(), target.flatten(),
                                     ignore_index=pad, reduction="none").reshape_as(target)
            totals += wide.cpu().double().sum(dim=1)
            narrow += inside.cpu().double().sum(dim=1)
            counts += valid.sum(dim=1).cpu()
            hits += ((logits.argmax(-1) == target) & valid).sum(dim=1).cpu()
        for index, row in enumerate(group):
            if int(counts[index]) != len(row["ids"]) - 1:
                raise SystemExit(f"Lost next-token targets on story {row['id']}")
            common = {"id": row["id"], "tokens": int(counts[index]), "correct": int(hits[index])}
            full.append({**common, "nll_sum": float(totals[index])})
            language.append({**common, "nll_sum": float(narrow[index])})
    return full, language, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--audit-data", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path,
                        help="Per-story records of the arm to compare against, e.g. arm I's")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    training = json.loads((ROOT / "experiments/configs" / (args.config + ".json")).read_text())["training"]
    reference = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                                                     local_files_only=True, torch_dtype=torch.float32)
    groups, _ = trainer.load_groups(args.groups)
    build = Namespace(d_embed=training["d_embed"], plasticity=training["plasticity"],
                      readout_rank=training["readout_rank"],
                      history_length=training["history_length"], seed=training["seed"])
    model = trainer.build_model(reference, build, groups)
    # Rebuild the arm in the form it was trained in. Defaults describe the two-task arms,
    # which predate both the sentiment range and the task cue.
    form = json.loads((ROOT / "experiments/configs" / (args.config + ".json")).read_text()) \
        .get("unified", {})
    space = form.get("output_space", {})
    unified = UnifiedFly(model.brain, settle_steps=5, readout_rank=training["readout_rank"],
                         tokens=space.get("language_tokens", 1024),
                         moves=space.get("chess_moves", 1968),
                         classes=space.get("sentiment_classes", 0),
                         tasks=len(form.get("tasks", ["language", "chess"])),
                         task_cue=form.get("task_cue", False))
    model.unified = unified
    saved = torch.load(args.run / args.checkpoint, map_location="cpu", weights_only=False)
    trainer.restore_parameters(model, saved["parameters"])
    model = model.to(args.device).eval()

    rows = json.loads(args.audit_data.read_text())["audit"]
    full, language, seconds = score(unified, model.brain, rows, args.device, unified.tokens)

    wide, narrow = aggregate(full), aggregate(language)
    report = {
        "run": str(args.run), "checkpoint": args.checkpoint,
        "checkpoint_sha256": trainer.file_hash(args.run / args.checkpoint),
        "updates": saved["cursor"]["updates"], "stories": len(rows), "seconds": seconds,
        "outputs": {"total": unified.tokens + unified.moves, "language": unified.tokens},
        "full_output_space": wide,
        "language_range_only": narrow,
        "leakage_cost_nats": wide["cross_entropy"] - narrow["cross_entropy"],
        "note": "language_range_only renormalises over the 1,024 token logits and is what "
                "compares to arms that never had a chess head; full_output_space is what "
                "this model actually predicts over. Their difference is the leakage cost.",
    }
    if args.reference_records:
        records = json.loads(args.reference_records.read_text())
        reference_rows = records["audit"]["per_story"] if "audit" in records else records["per_story"]
        report["versus_reference"] = {
            "reference": str(args.reference_records),
            "full_output_space": paired_comparison(full, reference_rows),
            "language_range_only": paired_comparison(language, reference_rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("updates", "stories", "full_output_space", "language_range_only",
                       "leakage_cost_nats")}, indent=2))


if __name__ == "__main__":
    main()
