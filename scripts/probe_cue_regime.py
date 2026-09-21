#!/usr/bin/env python3
"""Does the task cue change which edges are transmitting?

A cue can condition a recurrent brain two ways. As a bias on the drive, it offsets the
same dynamics. As a state, it puts the brain in a different regime: tanh saturates
different neurons, a saturated neuron passes almost no perturbation downstream, and so the
*effective* subgraph -- the edges actually carrying signal -- depends on where the state
is. Whether the learned cue does the first or the second is a measurement, not a metaphor.

The design that makes it decisive is holding the input fixed. The same 32-token windows
go through the brain under the language cue and under the sentiment cue, and the active
sets are compared. A second baseline runs the same cue over different windows, so the cue's
effect can be read against how much content alone moves the regime. Chess is reported too,
but its input is a board rather than tokens, so that comparison conflates cue with content
and is labelled as such.

An edge transmits a perturbation in proportion to tanh' at its destination, so a
"transmitting" edge here is one whose destination neuron is not saturated. Rows of the CSR
graph are destinations.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_unified_arms import rebuild
from fly_wordbrain.chess_model import settle
from fly_wordbrain.unified_model import CHESS, LANGUAGE, SENTIMENT
from train_task_gate import windows

SATURATED = 0.9      # |state| above this: the neuron is in tanh's flat region
TRANSMITTING = 0.95  # destination below this: tanh' large enough to pass a perturbation


def read_regime(cue_saturated_jaccard, content_saturated_jaccard, margin=0.1):
    """Turn two Jaccards into a reading, with a margin so a hair's difference stays silent.

    The saturated set is the informative metric; the transmitting-edge Jaccard is dominated
    by the ~90% of edges that transmit under every condition. Lower Jaccard = more change.
    A cue that selected a distinct regime would move the set FAR more than content does.
    """
    gap = cue_saturated_jaccard - content_saturated_jaccard
    if gap < -margin:
        return "The cue moves the saturated set FAR MORE than content does: it selects a distinct regime"
    if gap > margin:
        return "Content moves the saturated set far more than the cue: the cue is a weak bias"
    return ("The cue moves the saturated set about as much as replacing the input does "
            f"(gap {gap:+.3f}): regime selection comparable to a strong input, not a switch")


def jaccard(a, b):
    inter = (a & b).sum(-1).float()
    union = (a | b).sum(-1).float().clamp_min(1)
    return (inter / union).mean().item()


@torch.no_grad()
def token_states(unified, brain, ids, mask, task, device):
    embeds = unified.embed(ids.to(device), task)
    out = brain(input_ids=ids.to(device), inputs_embeds=embeds, attention_mask=mask.to(device),
                use_cache=False, return_dict=True)
    return out.last_hidden_state[:, -1, :].cpu()   # settled regime at the last position


@torch.no_grad()
def board_states(unified, brain, features, device):
    drive = unified.board_drive(features.to(device)).t().contiguous()
    return settle(brain, drive, unified.settle_steps).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--language", type=Path, required=True)
    parser.add_argument("--chess-corpus", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model, unified, plasticity = rebuild(args.config, args.model, args.groups, args.run,
                                         "best.pt", args.device)
    brain = model.brain
    rows = json.loads(args.language.read_text())["validation"]
    ids, mask = windows(rows, args.width, limit=args.windows,
                        generator=np.random.default_rng(1729))

    lang = token_states(unified, brain, ids, mask, LANGUAGE, args.device)
    sent = token_states(unified, brain, ids, mask, SENTIMENT, args.device)

    import train_multitask
    batches = train_multitask.ChessBatches(args.chess_corpus, "cpu", args.windows, 1729)
    features = batches.next()[0].float()   # (windows, 780): one board per window
    chess = board_states(unified, brain, features, args.device)

    sat = {"language": lang.abs() > SATURATED, "sentiment": sent.abs() > SATURATED,
           "chess": chess.abs() > SATURATED}
    dest = torch.from_numpy(np.repeat(np.arange(brain.config.n_neurons),
                                      np.diff(brain.w_offsets.cpu().numpy())))
    trans = {k: (v.abs() < TRANSMITTING)[:, dest] for k, v in
             (("language", lang), ("sentiment", sent), ("chess", chess))}

    # Same input, different cue: the cue's effect.
    cue_sat = jaccard(sat["language"], sat["sentiment"])
    cue_edge = jaccard(trans["language"], trans["sentiment"])
    # Same cue, different input: content's effect, over all window pairs.
    def across_inputs(m):
        vals = []
        for i in range(m.shape[0]):
            for j in range(i + 1, m.shape[0]):
                vals.append(jaccard(m[i:i+1], m[j:j+1]))
        return float(np.mean(vals))
    content_sat = across_inputs(sat["language"])
    content_edge = across_inputs(trans["language"])
    # Chess vs language: different cue AND different input, labelled as such.
    chess_sat = jaccard(sat["language"], sat["chess"])
    chess_edge = jaccard(trans["language"], trans["chess"])

    report = {
        "arm": args.config, "plasticity": plasticity, "windows": int(ids.shape[0]),
        "width": args.width, "thresholds": {"saturated": SATURATED, "transmitting": TRANSMITTING},
        "saturated_fraction": {k: v.float().mean().item() for k, v in sat.items()},
        "transmitting_edge_fraction": {k: v.float().mean().item() for k, v in trans.items()},
        "same_input_different_cue": {
            "note": "language cue vs sentiment cue on identical token windows -- the cue's effect alone",
            "saturated_set_jaccard": cue_sat, "transmitting_edge_jaccard": cue_edge},
        "same_cue_different_input": {
            "note": "language cue on different windows -- content's effect, the calibration",
            "saturated_set_jaccard": content_sat, "transmitting_edge_jaccard": content_edge},
        "chess_vs_language": {
            "note": "different cue AND different input (a board is not tokens); conflated, reported for scale",
            "saturated_set_jaccard": chess_sat, "transmitting_edge_jaccard": chess_edge},
    }
    report["reading"] = read_regime(cue_sat, content_sat)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
