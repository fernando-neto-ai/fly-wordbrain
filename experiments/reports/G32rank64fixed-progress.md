# Rank64 partial progress — 2026-09-15 23:17 UTC

G32rank64fixed remains running on macm3. The snapshot at23:17:46UTC observes
13,029updates and10completed epochs, with complete validation through13,000.
No stop signal was sent. F remains stopped; no quality-inference worker was
launched alongside training.

| Retained selector | G rank64 | E rank128 |
|---|---:|---:|
| Lowest validation CE | 3.124713 at12,800 | 3.149704 at13,700 |
| Highest validation accuracy | 36.2531% at11,800 | 36.5822% at14,300 |
| Decoder parameters | 3,226,688 | 6,453,376 |
| Total trainable parameters | 3,956,469 | 7,183,157 |

These independently selected metrics come from different checkpoint updates.
G currently has0.024991lower CE and0.3292percentagepoints lower accuracy than
E's retained best scores, using half the decoder parameters. G has not finished.

At exactly13,000updates, E has CE3.164297/accuracy36.2577%; G has
CE3.130633/accuracy36.0885%. G's CE is0.033664lower and its accuracy is
0.1692points lower. Best scores through the same13,000-update budget are
E3.151513/36.4954% versus G3.124713/36.2531%. All evaluations cover the same
100stories/21,874next-BPE targets. This snapshot compares logged update budgets;
exact token-exposure and regenerated-text audits remain pending the accepted stop.

## Fresh plateau decision: continue

The earlier9,400→11,400 window met both operational plateau thresholds.
A stop-helper attempt while validation12,900 was active returned
`not_ready_no_signal` before loading checkpoint binaries or sending a signal.
The fresh11,000→13,000 window supersedes that earlier plateau observation:

- Global-best CE improvement:0.035131, below0.10.
- Global-best accuracy improvement:0.589741percentagepoints, above0.5.
- Both stopping conditions must hold; therefore continue G.

The two adjacent1,000-update windows each contain ten periodic validations.
Mean CE improved3.158300→3.143144 and mean accuracy35.5792%→35.9125%.
Completed-epoch training CE fell2.963429→2.866572→2.767776 across epochs8–10.
This is diminishing validation improvement with renewed accuracy progress,
not a claim that later training cannot improve further.

The exact-launch stop helper is `scripts/stop_connectorch_rank64_plateau.py`.
Its default is read-only; `--execute` still requires fresh plateau, identity,
coherent-checkpoint and process-cessation checks. It never starts another run.
The separate rank-aware evaluator, `scripts/evaluate_connectorch_decoder_quality.py`,
is prepared and uploaded for after accepted cessation. Its GPU execution remains
deferred; the stop/evaluation guard tests pass (43 CPU/fake cases). Both selected checkpoints and latest state
remain owned by the active trainer until an accepted preservation receipt exists.

The [structured receipt](../records/G32rank64fixed-progress.json) records both
selectors, snapshot hashes, matched comparisons and the superseding decision.
Checkpoint binaries were not reloaded in this progress snapshot. One seed, rank-
dependent head initialization and the small selection-validation set limit
interpretation. The reserved test remains untouched.
