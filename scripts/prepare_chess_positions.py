#!/usr/bin/env python3
"""Build a chess position corpus from the Lichess Stockfish evaluations.

The source is de-normalised: one row per principal variation, and a single position may
carry several independent analyses at different depths. Picking a row naively gives a
label that is neither Stockfish's top move nor a consistent evaluation, so the reduction
is explicit:

    for each position   keep the deepest analysis, breaking ties on node count
    within it           keep the line whose score is best FOR THE SIDE TO MOVE

The second step selects by score rather than by row order. Rows *are* ordered best-first
within an analysis, but only 99.7% of the time, and nothing forces us to depend on it.

`cp` and `mate` are recorded from White's point of view; both are re-expressed from the
mover's. Positions with Black to move are mirrored so the network is only ever asked one
question, and the answer is always "what should I, to move, play here".

Output is packed bytes, not features: 32 bytes of board plus 2 of state per position. The
machine that trains never needs a chess library, and the corpus stays small enough to hold
in memory.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_wordbrain.chess_encoding import (PACKED_BOARD_BYTES, build_move_vocabulary,
                                          pack_board, win_probability)

DATASET = "Lichess/chess-position-evaluations"
SHARD = "data/data_0000.parquet"


def piece_codes_and_extras(board, chess):
    """Piece codes 1..12 (mover's P N B R Q K, then the opponent's) plus castling and ep."""
    codes = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        offset = 0 if piece.color == chess.WHITE else 6
        codes[square] = offset + piece.piece_type
    rights = (int(board.has_kingside_castling_rights(chess.WHITE))
              | int(board.has_queenside_castling_rights(chess.WHITE)) << 1
              | int(board.has_kingside_castling_rights(chess.BLACK)) << 2
              | int(board.has_queenside_castling_rights(chess.BLACK)) << 3)
    ep = 0 if board.ep_square is None else chess.square_file(board.ep_square) + 1
    return codes, np.array([rights, ep], dtype=np.uint8)


def verify_mate_convention(samples, chess):
    """Decide empirically whose point of view `mate` is written from.

    `cp` and `mate` never appear on the same row, so they cannot be cross-checked against
    each other. Instead the principal variation is played out: when it ends in checkmate,
    the side that delivered it is known, and the sign of `mate` can be compared against it.
    """
    white_delivers_positive = played = 0
    for fen, line, mate in samples:
        board = chess.Board(fen + " 0 1")
        try:
            for uci in line.split():
                board.push(chess.Move.from_uci(uci))
        except (ValueError, AssertionError):
            continue
        if not board.is_checkmate():
            continue
        played += 1
        white_won = board.turn == chess.BLACK      # the side to move is the one mated
        white_delivers_positive += (mate > 0) == white_won
    return played, white_delivers_positive


def reduce_rows(path_or_handle, max_rows, progress_every=2_000_000, mate_samples=4000):
    """Stream the shard and keep one analysis per position."""
    import pyarrow.parquet as pq
    best, scanned = {}, 0
    both_signs = Counter()
    mate_rows = []
    reader = pq.ParquetFile(path_or_handle)
    for batch in reader.iter_batches(batch_size=50_000,
                                     columns=["fen", "line", "depth", "knodes", "cp", "mate"]):
        for row in batch.to_pylist():
            scanned += 1
            line = row["line"]
            if not line:
                continue
            fen, cp, mate = row["fen"], row["cp"], row["mate"]
            if cp is None and mate is None:
                continue
            white = fen.split()[1] == "w"
            sign = 1 if white else -1
            # A mate is worth more than any centipawn score, and a faster mate more
            # than a slower one; being mated is the mirror image.
            if mate is not None:
                score = sign * mate
                ranked = (1, -abs(mate)) if score > 0 else (-1, abs(mate))
            else:
                ranked = (0, sign * cp)
            key = (row["depth"], row["knodes"])
            previous = best.get(fen)
            if previous is None or key > previous[0] or (key == previous[0] and ranked > previous[1]):
                best[fen] = (key, ranked, line.split()[0], cp, mate, white)
            if mate is not None and len(mate_rows) < mate_samples:
                mate_rows.append((fen, line, mate))
        if scanned % progress_every < 50_000:
            print(f"  scanned {scanned:,} rows, {len(best):,} positions", flush=True)
        if max_rows and scanned >= max_rows:
            break
    return best, scanned, mate_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", default=SHARD)
    parser.add_argument("--local-shard", type=Path, help="Use a downloaded parquet instead of streaming")
    parser.add_argument("--max-rows", type=int, default=8_000_000)
    parser.add_argument("--positions", type=int, default=1_000_000)
    parser.add_argument("--validation", type=int, default=10_000)
    parser.add_argument("--test", type=int, default=10_000)
    parser.add_argument("--audit", type=int, default=10_000)
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if (args.output / "corpus.npz").exists():
        parser.error("A corpus already exists here; use a new output directory")

    import chess

    vocabulary = build_move_vocabulary()
    index_of = {uci: i for i, uci in enumerate(vocabulary)}
    print(f"move vocabulary: {len(vocabulary):,}", flush=True)

    if args.local_shard:
        best, scanned, mate_rows = reduce_rows(str(args.local_shard), args.max_rows)
    else:
        from huggingface_hub import HfFileSystem
        with HfFileSystem().open(f"datasets/{DATASET}/{args.shard}", "rb") as handle:
            best, scanned, mate_rows = reduce_rows(handle, args.max_rows)
    print(f"scanned {scanned:,} rows -> {len(best):,} distinct positions", flush=True)

    resolved, consistent = verify_mate_convention(mate_rows, chess)
    agree = consistent / max(resolved, 1)
    print(f"mate convention: {resolved:,} principal variations played out to checkmate, "
          f"{agree:.1%} consistent with `mate` being written from White's point of view",
          flush=True)

    rng = np.random.default_rng(args.seed)
    fens = sorted(best)
    rng.shuffle(fens)

    boards, extras, moves, values, kept_fens = [], [], [], [], []
    rejected = Counter()
    wanted = args.positions + args.validation + args.test + args.audit
    for fen in fens:
        if len(kept_fens) >= wanted:
            break
        (depth, _), _, uci, cp, mate, white = best[fen]
        if depth < args.min_depth:
            rejected["shallow"] += 1
            continue
        board = chess.Board(fen + " 0 1")
        # The source writes castling both ways, e1g1 and king-takes-rook e1h1. Both are
        # accepted by `in board.legal_moves`, but iterating legal moves only ever yields
        # e1g1 -- so a target parsed the other way would never match its own legality mask.
        # parse_uci normalises both to the board's own form.
        try:
            move = board.parse_uci(uci)
        except (ValueError, chess.IllegalMoveError):
            rejected["illegal_line_move"] += 1
            continue
        if board.is_game_over():
            rejected["terminal"] += 1
            continue
        if not white:
            board = board.mirror()
            mirrored = chess.Move(chess.square_mirror(move.from_square),
                                  chess.square_mirror(move.to_square), promotion=move.promotion)
            try:
                move = board.parse_uci(mirrored.uci())
            except (ValueError, chess.IllegalMoveError):
                rejected["illegal_after_mirroring"] += 1
                continue
        if move.uci() not in index_of:
            rejected["move_outside_vocabulary"] += 1
            continue
        codes, extra = piece_codes_and_extras(board, chess)
        if mate is not None:
            value = 1.0 if (mate if white else -mate) > 0 else 0.0
        else:
            value = float(win_probability(cp if white else -cp))
        boards.append(codes); extras.append(extra)
        moves.append(index_of[move.uci()]); values.append(value); kept_fens.append(fen)
    if len(kept_fens) < wanted:
        raise SystemExit(f"Only {len(kept_fens):,} usable positions for a requested {wanted:,}; "
                         "raise --max-rows")

    packed = pack_board(np.array(boards, dtype=np.uint8))
    extras = np.array(extras, dtype=np.uint8)
    moves = np.array(moves, dtype=np.uint16)
    values = np.array(values, dtype=np.float32)

    bounds = np.cumsum([args.positions, args.validation, args.test, args.audit])
    slices = {"train": slice(0, bounds[0]), "validation": slice(bounds[0], bounds[1]),
              "test": slice(bounds[1], bounds[2]), "audit": slice(bounds[2], bounds[3])}

    # Legal-move masks, needed to score top-1 against Stockfish, for the evaluation splits
    # only: they are the expensive part and training never uses them.
    legal = {}
    target_in_mask = 0
    scored = 0
    for name in ("validation", "test", "audit"):
        offsets, indices = [0], []
        for position in range(slices[name].start, slices[name].stop):
            board = chess.Board(kept_fens[position] + " 0 1")
            if board.turn == chess.BLACK:
                board = board.mirror()
            start = len(indices)
            for candidate in board.legal_moves:
                uci = candidate.uci()
                if uci in index_of:
                    indices.append(index_of[uci])
            scored += 1
            target_in_mask += int(moves[position] in indices[start:])
            offsets.append(len(indices))
        legal[name] = (np.array(offsets, dtype=np.int64), np.array(indices, dtype=np.uint16))
        print(f"  {name}: {len(indices) / max(len(offsets) - 1, 1):.1f} legal moves per position",
              flush=True)

    split_fens = {name: [kept_fens[i] for i in range(s.start, s.stop)] for name, s in slices.items()}
    checks = {
        "splits_are_disjoint_by_position": len(set().union(*[set(v) for v in split_fens.values()]))
                                           == sum(len(v) for v in split_fens.values()),
        "every_target_move_appears_in_its_legality_mask": target_in_mask == scored,
        "every_move_index_within_vocabulary": int(moves.max()) < len(vocabulary),
        "values_within_unit_interval": bool((values >= 0).all() and (values <= 1).all()),

        "mate_is_written_from_whites_point_of_view": resolved >= 100 and agree > 0.98,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        raise SystemExit("Audit failed; refusing to write this corpus")

    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {"boards": packed, "extras": extras, "moves": moves, "values": values}
    for name, (offsets, indices) in legal.items():
        arrays[f"legal_offsets_{name}"] = offsets
        arrays[f"legal_indices_{name}"] = indices
    for name, s in slices.items():
        arrays[f"span_{name}"] = np.array([s.start, s.stop], dtype=np.int64)
    np.savez(args.output / "corpus.npz", **arrays)

    body = (args.output / "corpus.npz").read_bytes()
    provenance = {
        "dataset": DATASET, "shard": args.shard, "rows_scanned": scanned,
        "distinct_positions_reduced": len(best),
        "selection": "deepest analysis per position (ties on knodes), then the line scoring "
                     "best for the side to move; chosen by score, not by row order",
        "min_depth": args.min_depth, "seed": args.seed,
        "positions": {name: int(s.stop - s.start) for name, s in slices.items()},
        "rejected": dict(rejected),
        "mirroring": "positions with Black to move are mirrored; values are from the mover's view",
        "move_vocabulary": len(vocabulary),
        "features": "780 = 12 piece planes x 64 squares + 4 castling + 8 en-passant files",
        "value_target": "Lichess logistic win probability from centipawns; a mate is 1 or 0",
        "disjointness_audit": checks,
        "corpus_sha256": hashlib.sha256(body).hexdigest(),
        "megabytes": round(len(body) / 1e6, 1),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "moves.json").write_text(json.dumps(vocabulary) + "\n")
    print(json.dumps({k: provenance[k] for k in
                      ("positions", "rejected", "corpus_sha256", "megabytes")}, indent=2))


if __name__ == "__main__":
    main()
