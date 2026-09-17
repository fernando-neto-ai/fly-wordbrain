#!/usr/bin/env python3
"""Train one fly brain on language and chess through one encoder and one decoder.

The two-head arm lets each task keep a private encoder and a private readout, so it can
report no interference for an uninteresting reason: the tasks never really meet. This arm
removes that escape route. Both tasks inject through the same `brain.in_proj` and read out
of the same head over a single output space -- tokens below 1,024, chess moves above it --
so the model has to decide, from the settled state alone, whether it is looking at a story
or at a board.

    shared     the frozen graph, the per-neuron dynamics, the injection, the trunk,
               the output head
    private    a token embedding and a board projection, only so the two inputs are
               commensurate before they reach the shared injection

Three numbers come out of that decision:

    leakage        probability mass landing in the other task's range
    router probe   a head reading the trunk and predicting which task this is; it never
                   gates the output, so its accuracy is a measurement of whether the
                   brain's state carries task identity
    language CE    over all 2,992 outputs, so it can only match the Stage 6 arm's 2.9493
                   once leakage is gone -- the excess is exactly what leakage costs

The pinned trainer is never edited. Four seams are patched: the loss, the model build, the
parameter-count expectation and the learning-rate grouping. The language recipe, schedule,
update budget and checkpoint selectors stay byte-identical to I32rank64fixed10k.

The language readout that the pinned model builds is unused here and is frozen rather than
left to drift, so it neither collects gradient nor inflates the reported trainable count.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_connectorch as trainer
from fly_wordbrain.chess_encoding import FEATURES
from fly_wordbrain.chess_model import top1_against_legal
from fly_wordbrain.unified_model import CHESS, LANGUAGE, UnifiedFly, move_targets
from train_multitask import ChessBatches

STATE = {}


def unified_losses(unified, language_batch, chess_batch, weights, pad_id):
    """One update's worth of both tasks, through one head over one output space."""
    inputs, targets, mask, cache = language_batch
    logits, language_router, next_cache = unified.language(inputs, attention_mask=mask,
                                                           cache_params=cache)
    flat, flat_targets = logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    summed = F.cross_entropy(flat.float(), flat_targets, ignore_index=pad_id, reduction="sum")
    count = int(targets.ne(pad_id).sum().item())
    if count == 0:
        raise ValueError("A chunk contains no supervised target tokens")
    language_loss = summed / count
    correct = int(((logits.argmax(-1) == targets) & targets.ne(pad_id)).sum().item())

    features, moves, values = chess_batch
    move_logits, value_logits, chess_router = unified.chess(features)
    chess_loss = F.cross_entropy(move_logits.float(), move_targets(moves, unified.tokens))
    value_loss = -(unified.value_targets(values)
                   * value_logits.float().log_softmax(-1)).sum(-1).mean()

    # The router is supervised on both, and never feeds the output path.
    supervised = targets.ne(pad_id).reshape(-1)
    router_logits = torch.cat([language_router.reshape(-1, 2)[supervised], chess_router])
    router_labels = torch.cat([
        torch.full((int(supervised.sum()),), LANGUAGE, device=router_logits.device, dtype=torch.long),
        torch.full((moves.shape[0],), CHESS, device=router_logits.device, dtype=torch.long)])
    router_loss = F.cross_entropy(router_logits.float(), router_labels)

    total = (weights["language"] * language_loss + weights["chess"] * (chess_loss + value_loss)
             + weights["router"] * router_loss)
    record = {
        "language_loss": float(language_loss.item()), "chess_policy_loss": float(chess_loss.item()),
        "chess_value_loss": float(value_loss.item()), "router_loss": float(router_loss.item()),
        "router_accuracy": float((router_logits.argmax(-1) == router_labels).float().mean()),
        "language_leakage": unified.leakage(logits.detach(), LANGUAGE),
        "chess_leakage": unified.leakage(move_logits.detach(), CHESS),
        "chess_unmasked_top1": float((move_logits.argmax(-1)
                                      == move_targets(moves, unified.tokens)).float().mean()),
    }
    return total, language_loss, count, correct, next_cache, record


@torch.no_grad()
def evaluate_chess(unified, batches, split, device, limit=None, chunk=256):
    features, moves, values, offsets, indices = batches.split(split, limit)
    was_training = unified.training
    unified.eval()
    logits, decoded, routed, leak = [], [], [], []
    for start in range(0, features.shape[0], chunk):
        piece = features[start:start + chunk].to(device)
        move_logits, value_logits, router = unified.chess(piece)
        logits.append(move_logits.float().cpu())
        decoded.append(unified.value_from_logits(value_logits.float()).cpu())
        routed.append(router.float().cpu())
        leak.append(unified.leakage(move_logits, CHESS) * piece.shape[0])
    unified.train(was_training)
    logits, decoded, routed = torch.cat(logits), torch.cat(decoded), torch.cat(routed)
    targets = torch.from_numpy(moves)
    shifted = targets + unified.tokens
    return {
        "positions": int(targets.numel()),
        "policy_cross_entropy": float(F.cross_entropy(logits, shifted).item()),
        "top1_unmasked": float((logits.argmax(-1) == shifted).float().mean()),
        # Legality is applied inside the move range only; a token is never a legal move.
        "top1_legal_masked": top1_against_legal(logits[:, unified.tokens:], targets, offsets, indices),
        "value_mae": float((decoded - torch.from_numpy(values)).abs().mean()),
        "router_accuracy": float((routed.argmax(-1) == CHESS).float().mean()),
        "leakage_into_language": sum(leak) / max(int(targets.numel()), 1),
    }


def install(args):
    original_build = trainer.build_model
    original_expected = trainer.expected_parameter_counts

    def patched_build(reference, training_args, groups):
        model = original_build(reference, training_args, groups)
        unified = UnifiedFly(model.brain, settle_steps=args.settle_steps,
                             readout_rank=training_args.readout_rank, features=FEATURES)
        # The pinned model's own readout is unused here. Freeze it rather than leave it to
        # collect weight decay and inflate the reported trainable count.
        model.lm_head.requires_grad_(False)
        model.ln.requires_grad_(False)
        model.unified = unified
        STATE["unified"] = unified
        return model

    def patched_expected(model, training_args, groups):
        expected = dict(original_expected(model, training_args, groups))
        expected["readout"] = 0            # frozen above, so parameter_counts skips it
        expected["layernorm"] = 0
        expected["other"] = sum(p.numel() for p in STATE["unified"].own_parameters()
                                if p.requires_grad)
        expected["total"] = sum(v for k, v in expected.items() if k != "total")
        return expected

    def patched_optimizer_for(model, phase):
        rates = (1e-3, 1e-4) if phase == 0 else (3e-4, 3e-5)
        inputs, readouts = [], []
        for name, value in model.named_parameters():
            if not value.requires_grad:
                continue
            encoder = name.startswith("brain.") or name.startswith("unified.board_adapter")
            (inputs if encoder else readouts).append(value)
        return torch.optim.AdamW(
            [{"params": inputs, "lr": rates[0], "name": "input_and_neurons"},
             {"params": readouts, "lr": rates[1], "name": "readout_and_norm"}],
            betas=(.9, .999), weight_decay=.01)

    def patched_chunk_loss(model, inputs, targets, mask, cache=None, pad_id=0):
        unified = STATE["unified"]
        if not model.training:
            # Evaluation is a pure language measurement -- over all 2,992 outputs, so
            # leakage shows up as cost rather than being hidden by a narrower softmax.
            logits, _, next_cache = unified.language(inputs, attention_mask=mask, cache_params=cache)
            flat = logits.reshape(-1, logits.shape[-1]).float()
            summed = F.cross_entropy(flat, targets.reshape(-1), ignore_index=pad_id, reduction="sum")
            count = int(targets.ne(pad_id).sum().item())
            correct = int(((logits.argmax(-1) == targets) & targets.ne(pad_id)).sum().item())
            return summed / count, count, correct, next_cache
        total, _, count, correct, next_cache, record = unified_losses(
            unified, (inputs, targets, mask, cache), STATE["batches"].next(),
            STATE["weights"], pad_id)
        record["updates"] = STATE.get("updates", 0) + 1
        STATE["updates"] = record["updates"]
        if STATE.get("log"):
            trainer.append_json(STATE["log"], record)
        return total, count, correct, next_cache

    trainer.build_model = patched_build
    trainer.expected_parameter_counts = patched_expected
    trainer.optimizer_for = patched_optimizer_for
    trainer.chunk_loss = patched_chunk_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chess-corpus", type=Path, required=True)
    parser.add_argument("--chess-batch", type=int, default=32)
    parser.add_argument("--chess-weight", type=float, default=1.0)
    parser.add_argument("--language-weight", type=float, default=1.0)
    parser.add_argument("--router-weight", type=float, default=0.1)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--chess-eval-limit", type=int, default=10000)
    known, rest = parser.parse_known_args()

    training_args = trainer.parser().parse_args(rest)
    install(known)
    STATE["weights"] = {"language": known.language_weight, "chess": known.chess_weight,
                        "router": known.router_weight}

    original_run = trainer.run
    STATE["batches"] = ChessBatches(known.chess_corpus, training_args.device,
                                    known.chess_batch, training_args.seed)
    training_args.output.mkdir(parents=True, exist_ok=True)
    STATE["log"] = training_args.output / "unified.jsonl"
    print(json.dumps({"unified": {"corpus": str(known.chess_corpus),
                                  "train_positions": len(STATE["batches"]),
                                  "chess_batch": known.chess_batch,
                                  "weights": STATE["weights"],
                                  "settle_steps": known.settle_steps}}), flush=True)
    original_run(training_args)

    unified = STATE["unified"]
    status = json.loads((training_args.output / "status.json").read_text())
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "weights": STATE["weights"], "chess_batch": known.chess_batch,
              "chess_positions_seen": known.chess_batch * int(status["updates"]),
              "chess_passes_over_corpus": STATE["batches"].passes
                                          + STATE["batches"].cursor / max(len(STATE["batches"]), 1),
              "language_best_validation": status["best"]}
    for split in ("validation", "audit"):
        report[split] = evaluate_chess(unified, STATE["batches"], split, training_args.device,
                                       known.chess_eval_limit)
    (training_args.output / "unified-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
