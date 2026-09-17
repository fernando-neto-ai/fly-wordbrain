#!/usr/bin/env python3
"""Train one fly brain on language and chess at the same time.

Every stage so far gave the brain one job. This asks whether a single brain can hold two,
and it is a sharper question than it looks, because of exactly *what* is shared. The graph
is frozen, so it cannot be fought over. The only shared trainable parameters are the
148,179 per-neuron gain, rec_gain and bias values -- the brain's dynamics. Both tasks must
agree on them, while each keeps its own encoder and readout.

    shared      the frozen 49,393-neuron graph, and the per-neuron dynamics on top of it
    private     the language encoder and readout, the board encoder and the move/value heads

So the measurement is: what does the language arm lose by having to share its dynamics with
chess, and what does chess lose by sharing with language? Stage 7 showed the *wiring*
contributes about 1%. If sharing the dynamics is also nearly free, the brain is close to a
generic reservoir and the tasks are being solved almost entirely in their heads. If sharing
is expensive, the tuned dynamics are task-specific, and the obvious next lever is to let
the connectome itself adapt -- which `--plasticity bounded10` does, giving every synapse a
bounded, sign-preserving multiplier.

This never edits the pinned trainer. It patches the one seam where a training step
computes its loss, so the optimizer, the two-phase cosine schedule, the update budget,
the checkpoint selectors and every audit stay byte-identical to the language arm this is
compared against. A chess batch rides along on each language update.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_connectorch as trainer
from fly_wordbrain.chess_encoding import FEATURES, features_from_packed
from fly_wordbrain.chess_model import ChessFly, top1_against_legal

STATE = {}


class ChessBatches:
    """Endless shuffled stream of training positions, unpacked to features on demand."""

    def __init__(self, corpus, device, batch_size, seed):
        arrays = np.load(corpus / "corpus.npz")
        span = arrays["span_train"]
        self.boards = arrays["boards"][span[0]:span[1]]
        self.extras = arrays["extras"][span[0]:span[1]]
        self.moves = arrays["moves"][span[0]:span[1]].astype(np.int64)
        self.values = arrays["values"][span[0]:span[1]]
        self.arrays, self.device, self.batch_size = arrays, device, batch_size
        self.generator = np.random.default_rng(seed)
        self.order, self.cursor, self.passes = self.generator.permutation(len(self.moves)), 0, 0

    def __len__(self):
        return len(self.moves)

    def next(self):
        if self.cursor + self.batch_size > len(self.order):
            self.order, self.cursor = self.generator.permutation(len(self.moves)), 0
            self.passes += 1
        rows = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        features = features_from_packed(self.boards[rows], self.extras[rows])
        return (torch.from_numpy(features).to(self.device),
                torch.from_numpy(self.moves[rows]).to(self.device),
                torch.from_numpy(self.values[rows]).to(self.device))

    def split(self, name, limit=None):
        span = self.arrays[f"span_{name}"]
        stop = span[1] if limit is None else min(span[1], span[0] + limit)
        features = features_from_packed(self.arrays["boards"][span[0]:stop],
                                        self.arrays["extras"][span[0]:stop])
        offsets = self.arrays[f"legal_offsets_{name}"][:stop - span[0] + 1]
        return (torch.from_numpy(features), self.arrays["moves"][span[0]:stop].astype(np.int64),
                self.arrays["values"][span[0]:stop], offsets,
                self.arrays[f"legal_indices_{name}"][:offsets[-1]].astype(np.int64))


def chess_step(fly, batch, weight):
    features, moves, values = batch
    logits, value_logits = fly(features)
    policy = F.cross_entropy(logits.float(), moves)
    value = -(fly.value_targets(values) * value_logits.float().log_softmax(-1)).sum(-1).mean()
    hits = float((logits.argmax(-1) == moves).float().mean())
    return weight * (policy + value), {"chess_policy_loss": float(policy.item()),
                                       "chess_value_loss": float(value.item()),
                                       "chess_unmasked_top1": hits}


@torch.no_grad()
def evaluate_chess(fly, batches, split, device, limit=None, chunk=256):
    features, moves, values, offsets, indices = batches.split(split, limit)
    fly.eval()
    logits, decoded = [], []
    for start in range(0, features.shape[0], chunk):
        piece = features[start:start + chunk].to(device)
        move_logits, value_logits = fly(piece)
        logits.append(move_logits.float().cpu())
        decoded.append(fly.value_from_logits(value_logits.float()).cpu())
    fly.train()
    logits, decoded = torch.cat(logits), torch.cat(decoded)
    targets = torch.from_numpy(moves)
    return {
        "positions": int(targets.numel()),
        "policy_cross_entropy": float(F.cross_entropy(logits, targets).item()),
        "top1_unmasked": float((logits.argmax(-1) == targets).float().mean()),
        "top1_legal_masked": top1_against_legal(logits, targets, offsets, indices),
        "value_mae": float((decoded - torch.from_numpy(values)).abs().mean()),
    }


def install(args):
    """Patch the seams where the second task has to be admitted, and no others."""
    original_chunk_loss = trainer.chunk_loss
    original_build = trainer.build_model
    original_expected = trainer.expected_parameter_counts
    original_optimizer_for = trainer.optimizer_for

    def chess_parameter_count():
        return sum(p.numel() for p in STATE["fly"].chess_parameters() if p.requires_grad)

    def patched_expected(model, training_args, groups):
        # The trainer refuses to start if the model carries a parameter it cannot account
        # for, which is the right behaviour. Extend the expectation by exactly the chess
        # head rather than relaxing the check: an unexplained parameter must still fail.
        expected = dict(original_expected(model, training_args, groups))
        chess = chess_parameter_count()
        expected["other"] = expected.get("other", 0) + chess
        expected["total"] = expected["total"] + chess
        return expected

    def patched_optimizer_for(model, phase):
        rates = (1e-3, 1e-4) if phase == 0 else (3e-4, 3e-5)
        inputs, readouts = [], []
        for name, value in model.named_parameters():
            if not value.requires_grad:
                continue
            # Mirror the language split: whatever writes current into the brain learns at
            # the higher rate, whatever reads state out of it at the lower one. The board
            # encoder is the counterpart of brain.in_proj, not of lm_head.
            encoder = name.startswith("brain.") or name.startswith("chess.encoder.")
            (inputs if encoder else readouts).append(value)
        return torch.optim.AdamW(
            [{"params": inputs, "lr": rates[0], "name": "input_and_neurons"},
             {"params": readouts, "lr": rates[1], "name": "readout_and_norm"}],
            betas=(.9, .999), weight_decay=.01)

    def patched_build(reference, training_args, groups):
        model = original_build(reference, training_args, groups)
        fly = ChessFly(model.brain, settle_steps=args.settle_steps,
                       encoder_rank=args.encoder_rank, readout_rank=training_args.readout_rank,
                       features=FEATURES)
        # Registering the chess head on the language model is what puts its parameters in
        # front of the pinned trainer's optimizer, clipping and checkpointing untouched.
        model.chess = fly
        STATE["fly"] = fly
        return model

    def patched_chunk_loss(model, inputs, targets, mask, cache=None, pad_id=0):
        loss, count, correct, next_cache = original_chunk_loss(model, inputs, targets, mask,
                                                               cache, pad_id)
        # The trainer evaluates through this same function, after model.eval(). Adding the
        # chess term there would fold it into the reported validation cross-entropy -- the
        # number that selects checkpoints and the number this arm is compared against. So
        # the second task rides along on training steps only, and validation stays a pure
        # language measurement, weighting included.
        if not model.training:
            return loss, count, correct, next_cache
        extra, record = chess_step(STATE["fly"], STATE["batches"].next(), args.chess_weight)
        record["language_loss"] = float(loss.item())
        record["updates"] = STATE.get("updates", 0) + 1
        STATE["updates"] = record["updates"]
        if STATE.get("log"):
            trainer.append_json(STATE["log"], record)
        return args.language_weight * loss + extra, count, correct, next_cache

    trainer.build_model = patched_build
    trainer.chunk_loss = patched_chunk_loss
    trainer.expected_parameter_counts = patched_expected
    trainer.optimizer_for = patched_optimizer_for
    return original_chunk_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chess-corpus", type=Path, required=True)
    parser.add_argument("--chess-batch", type=int, default=32)
    parser.add_argument("--chess-weight", type=float, default=1.0)
    parser.add_argument("--language-weight", type=float, default=1.0)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--encoder-rank", type=int, default=64)
    parser.add_argument("--chess-eval-limit", type=int, default=2000)
    known, rest = parser.parse_known_args()

    training_args = trainer.parser().parse_args(rest)
    install(known)

    original_run = trainer.run
    original_validation_hook = None

    def run_with_chess(args):
        STATE["batches"] = ChessBatches(known.chess_corpus, args.device, known.chess_batch, args.seed)
        args.output.mkdir(parents=True, exist_ok=True)
        # The trainer's own loss column is the joint objective it backpropagates. The
        # decomposition lives here so neither task's curve has to be inferred from it.
        STATE["log"] = args.output / "multitask.jsonl"
        print(json.dumps({"chess": {"corpus": str(known.chess_corpus),
                                    "train_positions": len(STATE["batches"]),
                                    "batch": known.chess_batch,
                                    "weights": {"language": known.language_weight,
                                                "chess": known.chess_weight},
                                    "settle_steps": known.settle_steps,
                                    "encoder_rank": known.encoder_rank}}), flush=True)
        return original_run(args)

    run_with_chess(training_args)

    fly = STATE["fly"]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "chess_batch": known.chess_batch, "chess_weight": known.chess_weight,
              "language_weight": known.language_weight,
              "chess_positions_seen": known.chess_batch * int(
                  json.loads((training_args.output / "status.json").read_text())["updates"]),
              "chess_passes_over_corpus": STATE["batches"].passes
                                          + STATE["batches"].cursor / max(len(STATE["batches"]), 1)}
    for split in ("validation", "audit"):
        report[split] = evaluate_chess(fly, STATE["batches"], split, training_args.device,
                                       known.chess_eval_limit)
    (training_args.output / "chess-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
