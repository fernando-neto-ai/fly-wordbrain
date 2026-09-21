#!/usr/bin/env python3
"""Generate text from unified arms on identical prompts, side by side, with proxies.

Coherence is a judgement, so the text is shown. But two things a reader can misjudge are
measured beside it: how often the generation falls into repetition (repeated 3-gram rate
and the longest exactly-repeated span) and how confident the model is in each token it
emits (mean top-1 probability over the generated span). Decoding is greedy inside the
language range only -- an argmax over the whole shared head could emit a chess move index,
which would decode to a token that the model never chose as a word.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_unified_arms import rebuild


@torch.no_grad()
def generate(unified, tokens, prompt_ids, steps, device):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    logits, _, cache = unified.language(ids, attention_mask=mask, cache_params=None)
    out, confidences = [], []
    for _ in range(steps):
        last = logits[0, -1, :tokens].float().softmax(-1)
        nxt = int(last.argmax())
        confidences.append(float(last[nxt]))
        out.append(nxt)
        if nxt == 2:  # eos
            break
        step = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, _, cache = unified.language(step, attention_mask=torch.ones_like(step), cache_params=cache)
    return out, confidences


def repetition(ids):
    grams = [tuple(ids[i:i+3]) for i in range(len(ids) - 2)]
    if not grams:
        return 0.0, 0
    repeated = sum(c - 1 for c in Counter(grams).values() if c > 1)
    # Longest span that occurs twice, in tokens.
    longest = 0
    for size in range(2, len(ids) // 2 + 1):
        spans = [tuple(ids[i:i+size]) for i in range(len(ids) - size + 1)]
        if len(spans) != len(set(spans)):
            longest = size
        else:
            break
    return repeated / len(grams), longest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--specs", nargs="+", required=True, help="name:config:checkpoint")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--language", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=12)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    rows = json.loads(args.language.read_text())["validation"][:args.prompts]
    prompts = [r["ids"][:args.prompt_tokens] for r in rows]

    report = {"prompts": [tok.decode(p[1:]) for p in prompts], "arms": {}}
    for spec in args.specs:
        name, config, ckpt = spec.split(":")
        model, unified, _ = rebuild(config, args.model, args.groups, args.arms / name, ckpt, args.device)
        status = json.loads((args.arms / name / "status.json").read_text())
        entries = []
        for p in prompts:
            out, conf = generate(unified, unified.tokens, p, args.steps, args.device)
            rep, longest = repetition(out)
            entries.append({"text": tok.decode(out), "tokens": len(out),
                            "repeated_3gram_rate": rep, "longest_repeated_span": longest,
                            "mean_top1_confidence": sum(conf) / max(len(conf), 1)})
        report["arms"][name] = {"checkpoint": ckpt, "updates": status.get("updates"),
                                "generations": entries,
                                "mean_repeated_3gram_rate": sum(e["repeated_3gram_rate"] for e in entries) / len(entries),
                                "mean_top1_confidence": sum(e["mean_top1_confidence"] for e in entries) / len(entries)}
        del model, unified
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    # Human-readable side by side.
    for i, prompt in enumerate(report["prompts"]):
        print(f"\n=== PROMPT {i+1}: {prompt!r}")
        for name, arm in report["arms"].items():
            e = arm["generations"][i]
            print(f"--- {name} @ {arm['updates']:,} updates  [rep3 {e['repeated_3gram_rate']:.2f}  longest-repeat {e['longest_repeated_span']}  conf {e['mean_top1_confidence']:.2f}]")
            print("    " + e["text"].replace("\n", " ")[:400])
    print("\n=== SUMMARY")
    for name, arm in report["arms"].items():
        print(f"  {name:<36} updates {arm['updates']:>6,}  repeated-3gram {arm['mean_repeated_3gram_rate']:.3f}  top1-conf {arm['mean_top1_confidence']:.3f}")


if __name__ == "__main__":
    main()
