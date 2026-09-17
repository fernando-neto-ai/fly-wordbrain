# Stage 8 — A second task for the same brain

**Objective.** Seven stages asked the connectome one question: predict the next token. That
makes every finding a claim about *this brain on this task*, and leaves open whether
anything generalises. [ChessFly](https://hf.co/spaces/mlabonne/chessfly) showed a fly
connectome can be trained to play chess, so chess is the natural second task — it is
spatial rather than sequential, its labels come from an engine rather than from text, and
it has published numbers to check a replication against.

This stage builds the task: the corpus, the encoding, and the baselines a trained model has
to beat. What it does *not* do is treat a good chess score as evidence about anatomy; that
is [Stage 7](../07-graph-control/README.md)'s business, and Stage 7's answer was 1%.

## Replicating the reference

Three things match the reference exactly, and two conventions were measured rather than
assumed.

| | ours | ChessFly |
|---|---|---|
| move vocabulary | **1,968** | 1,968 |
| board features | **780** | 780 |
| connectome | MaleCNS, 49,393 neurons | FlyWire FAFB, 138,639 neurons |

The move vocabulary is every from/to pair a queen or knight could traverse on an empty
board — 1,792 of them — plus the four promotion choices for pawn moves reaching the last
rank, 176 more. Features are 12 piece planes over 64 squares, then 4 castling flags and 8
en-passant file flags. Positions with Black to move are mirrored, so the model is only ever
asked one question and the value is always from the mover's point of view.

### What had to be measured

Labels come from [`Lichess/chess-position-evaluations`](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
(CC0): 957,860,115 rows of Stockfish analysis. It is **de-normalised** — one row per
principal variation, and a position may carry several independent analyses at different
depths — so a naive row gives neither the top move nor a consistent evaluation. The
reduction keeps the deepest analysis per position, then the line scoring best for the side
to move, **selected by score rather than by row order**: rows are ordered best-first only
99.7% of the time, and nothing forces a dependency on it.

Two sign conventions decide every label, and neither is documented:

- **`cp` is written from White's point of view.** Confirmed on 35,336 multi-PV analyses:
  within a single analysis, scores are monotone in the direction that favours the side to
  move 99.7% of the time. Grouping by position *without* also grouping by depth destroys
  this — it reads as 43%, which is chance.
- **`mate` likewise.** `cp` and `mate` never appear on the same row, so they cannot be
  cross-checked. Instead 1,580 principal variations were played out to checkmate and the
  delivering side compared against the sign: **100% consistent**.

### The trap

The source writes castling **both** as `e1g1` and as king-takes-rook `e1h1`. python-chess
accepts both through `move in board.legal_moves`, but only ever yields `e1g1` when
iterating — so a target parsed the other way could never appear in its own legality mask.
16 of 500 audit positions were silently wrong until everything was routed through
`board.parse_uci`, which normalises both. The audit that caught it had been written as a
hard-coded `True`; it now computes.

## The corpus

[`scripts/prepare_chess_positions.py`](../../scripts/prepare_chess_positions.py) reduces
6,000,000 rows to 1,177,573 distinct positions and keeps 630,000 of them: **600,000 for
training**, and 10,000 each for validation, test and audit, disjoint by position.

Output is packed bytes rather than features — 32 for the board, 2 for castling and en
passant — so the machine that trains never needs a chess library and the whole corpus is
27.2 MB. Five audits gate the write, including that every target move appears in its own
legality mask.

## The baselines

A chess number means nothing without the floor it clears. On the 10,000-position audit
split:

| | ours | ChessFly |
|---|---:|---:|
| most common legal move | **12.33%** | 13% |
| random legal move | **3.23%** | 3% |
| constant value predictor (MAE) | **0.189** | — |
| *reference's trained model* | — | *30.4%, MAE 0.081* |

The random figure is reported two ways because they answer different questions. The
familiar "one in about thirty legal moves" is 1/E[n] = 3.23%. The expected accuracy of
actually drawing uniformly is E[1/n], which is **8.48%** pooled — far higher, because
forced-mate positions are near-terminal: they are 19.1% of the split and a fifth of them
have five or fewer legal moves. E[1/n] is the figure a trained model has to beat.

## Does it fit on a Mac

Measured on macm3 rather than estimated, forward and backward at the shape that would
actually be trained:

| batch | ms/update | positions/s |
|---:|---:|---:|
| 8 | 27.1 | 294.7 |
| 128 | 410.9 | 311.5 |

**~312 positions/second, flat in batch size** — the sparse product over 9,050,172 edges
dominates, so batching buys nothing, but one million positions is 0.89 hours and the
reference's 4.4M would be 3.9. A chess arm and its control are an overnight run.

## Result

*The chess arms are [Stage 9](../09-two-tasks/README.md), because the interesting question
turned out not to be "can it play chess" but "can one brain hold both".*

Receipts: [`records/chess-baselines-v1.json`](../records/chess-baselines-v1.json), corpus
provenance in `data/chess-positions-v1/provenance.json`.
