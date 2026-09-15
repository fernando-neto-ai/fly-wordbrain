# Connectorch encoder experiment — partial validation

Snapshot: 2026-09-15T14:55:14.043315+00:00. Host: macm3.

Campaign status: **running**. Phase: **training**. Current arm: **A128fixed**.

Baseline: **accepted by the user after early stopping**. Its minimum-CE and maximum-accuracy checkpoints are preserved; the 44-epoch reference recipe was not completed.
All arms retain the full output head, eight explicit input lags, 49,393 neurons and 9,050,172 base edges.
Bounded arms add 18,322 source/destination cell-type gains; base-edge multipliers stay within 0.9–1.1. Original neuron gains remain unconstrained.

| Arm | Width | Edge gains | Status | Updates | Minimum CE (accuracy; update) | Maximum accuracy (CE; update) |
|---|---:|---|---|---:|---|---|
| A128fixed | 128 | fixed | running / validation | 100 | 7.3587 (8.72%; 100) | 8.72% (7.3587; 100) |
| B32fixed | 32 | fixed | pending | 0 | — | — |
| C128bounded | 128 | bounded10 | pending | 0 | — | — |
| D32bounded | 32 | bounded10 | pending | 0 | — | — |

Winner columns use the saved checkpoint receipts when available. A newer validation can appear in the history while its checkpoint is still being saved; exact metric ties retain the earlier checkpoint.

## A128fixed

Branch: `exp/encoder128-fixed`. Commit: `3641b9af124d0eb905715265073b576e2324aa59`. Repository: https://github.com/fernando-neto-ai/fly-wordbrain.

Saved minimum_validation_ce: `/Users/fernando/fly_wordbrain_connectorch/results/connectorch-encoder-v1/arms/A128fixed/best.pt`; SHA256 `d5e211b6949c28cde8c333a928dcef7e02777291574f84f2befbf0602575dc1b`.

Saved maximum_validation_accuracy: `/Users/fernando/fly_wordbrain_connectorch/results/connectorch-encoder-v1/arms/A128fixed/best-accuracy.pt`; SHA256 `d4c4ac775e59c1580f147cef1d8c65b5c4bef7e7460db25d88a239f5c337fe4b`.

| Updates | Epoch | Validation CE | Accuracy |
|---:|---:|---:|---:|
| 0 | 0 | 7.4866 | 0.06% |
| 100 | 0 | 7.3587 | 8.72% |

## B32fixed

Branch: `exp/encoder32-fixed`. Commit: `68d721ef82de4401f2665c6fdf33579f2865014a`. Repository: https://github.com/fernando-neto-ai/fly-wordbrain.

No full-arm validation yet.

## C128bounded

Branch: `exp/encoder128-bounded`. Commit: `71d845460c7535f4e3b0ab83f3ae6c28dbe67af3`. Repository: https://github.com/fernando-neto-ai/fly-wordbrain.

No full-arm validation yet.

## D32bounded

Branch: `exp/encoder32-bounded`. Commit: `6b725e1c6727794b1c342d7a433c0d3f644cbc08`. Repository: https://github.com/fernando-neto-ai/fly-wordbrain.

No full-arm validation yet.

All partial scores are next-BPE-token results on the 100-story validation split. Test evaluation is deferred.
This is a timestamped snapshot; each refresh verifies artifact SHA256 checksums. Checkpoint hashes are copied from trainer receipts; checkpoint binaries are not fetched by this command.
