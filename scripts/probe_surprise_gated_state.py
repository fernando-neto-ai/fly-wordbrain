#!/usr/bin/env python3
"""Does the brain already overwrite its state more when a token is surprising?

Active inference says perception should update state in proportion to prediction error.
This brain is driven by the raw token, not the error, and at leak 0.9 replaces 90% of its
state every step regardless. Whether surprise nonetheless modulates how much the state
moves is a measurement: per position, the surprise of the actual token under the model's
own prediction, and the relative change in the 49,393-dimensional state. If the two are
uncorrelated the brain updates uniformly and error-driven drive would be a real change; if
they correlate, it is already leaning that way and the change is a small step.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from compare_unified_arms import rebuild
import train_connectorch as trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True); p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--model", type=Path, required=True); p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--language", type=Path, required=True); p.add_argument("--stories", type=int, default=40)
    p.add_argument("--chunk", type=int, default=32); p.add_argument("--device", default="mps")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    model, unified, _ = rebuild(a.config, a.model, a.groups, a.run, a.checkpoint, a.device)
    rows = json.loads(a.language.read_text())["validation"][:a.stories]
    surprises, changes, drives = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            group = rows[start:start + 8]
            inputs, targets, mask = trainer.batch_tensors(group, a.device, 0)
            cache, prev = None, None
            for off in range(0, inputs.shape[1], a.chunk):
                sl = slice(off, off + a.chunk)
                ids, m = inputs[:, sl], mask[:, sl]
                out = model.brain(input_ids=ids, inputs_embeds=unified.embed(ids, 0), attention_mask=m,
                                  cache_params=cache, use_cache=True, return_dict=True)
                hidden, cache = out.last_hidden_state, out.cache_params
                logits = unified.read(hidden)[0][..., :unified.tokens].float()
                tgt = targets[:, sl]
                ce = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=0,
                                     reduction="none").view_as(tgt)
                states = torch.cat([prev.unsqueeze(1), hidden], 1) if prev is not None else hidden
                d = (states[:, 1:] - states[:, :-1]).norm(dim=-1) / states[:, :-1].norm(dim=-1).clamp_min(1e-6)
                if prev is None:
                    d = torch.cat([torch.full_like(d[:, :1], float("nan")), d], 1)
                valid = tgt.ne(0) & ~torch.isnan(d)
                surprises.append(ce[valid].cpu()); changes.append(d[valid].cpu())
                prev = hidden[:, -1]
    s, c = torch.cat(surprises).numpy(), torch.cat(changes).numpy()
    def pearson(x, y):
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
    def ranks(v):
        r = np.empty_like(v); r[np.argsort(v, kind="stable")] = np.arange(v.size); return r
    pr = (pearson(s, c), None)
    sr = (pearson(ranks(s), ranks(c)), None)   # Spearman = Pearson on ranks
    q = np.quantile(s, [0.25, 0.5, 0.75])
    bins = np.digitize(s, q)
    by_quartile = {f"q{i+1}": {"mean_surprise": float(s[bins == i].mean()),
                                "mean_relative_state_change": float(c[bins == i].mean()),
                                "n": int((bins == i).sum())} for i in range(4)}
    report = {"arm": a.config, "checkpoint": a.checkpoint, "positions": int(s.size),
              "pearson_r": pr[0], "spearman_rho": sr[0],
              "mean_relative_state_change": float(c.mean()),
              "by_surprise_quartile": by_quartile,
              "note": "relative change = ||s_t - s_{t-1}|| / ||s_{t-1}|| over all 49,393 neurons; "
                      "at leak 0.9 the floor is set by the update rule, so the question is the slope"}
    print(json.dumps(report, indent=2))
    if a.output: a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
