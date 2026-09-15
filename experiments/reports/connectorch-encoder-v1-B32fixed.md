# Connectorch encoder experiment — partial validation

Snapshot: 2026-09-15T16:12:37.846494+00:00. Host: macm3.

Campaign status: **running**. Phase: **training**. Current arm: **B32fixed**.

Baseline: **accepted by the user after early stopping**. Its minimum-CE and maximum-accuracy checkpoints are preserved; the 44-epoch reference recipe was not completed.
All arms retain the full output head, eight explicit input lags, 49,393 neurons and 9,050,172 base edges.
Bounded arms add 18,322 source/destination cell-type gains; base-edge multipliers stay within 0.9–1.1. Original neuron gains remain unconstrained.

A128fixed: **accepted early stop**, 10 completed epochs; 12,400 durable updates and 12,442 observed updates. Its complete history and both checkpoint winners are retained from the parent campaign. Retained-best comparisons use unequal training budgets; A did not complete the 44-epoch schedule.

| Arm | Width | Edge gains | Status | Updates | Minimum CE (accuracy; update) | Maximum accuracy (CE; update) |
|---|---:|---|---|---:|---|---|
| A128fixed | 128 | fixed | accepted early stop | 12,400 | 4.7283 (30.49%; 3,600) | 32.90% (5.5470; 12,200) |
| B32fixed | 32 | fixed | running / training | 332 | 7.0229 (11.09%; 200) | 11.09% (7.0229; 200) |
| C128bounded | 128 | bounded10 | pending | 0 | — | — |
| D32bounded | 32 | bounded10 | pending | 0 | — | — |

Winner columns use the saved checkpoint receipts when available. A newer validation can appear in the history while its checkpoint is still being saved; exact metric ties retain the earlier checkpoint.

## A128fixed

Branch: `exp/encoder128-fixed`. Commit: `3641b9af124d0eb905715265073b576e2324aa59`. Repository: https://github.com/fernando-neto-ai/fly-wordbrain.

Saved minimum_validation_ce: `/Users/fernando/fly_wordbrain_connectorch/results/connectorch-encoder-v1/arms/A128fixed/best.pt`; SHA256 `2f6fad5b9c1a17d0ff616159510a8c37dc0e03f56b8aee0ff345c41f8ea9db40`.

Saved maximum_validation_accuracy: `/Users/fernando/fly_wordbrain_connectorch/results/connectorch-encoder-v1/arms/A128fixed/best-accuracy.pt`; SHA256 `eddb423ef0ff5180d9e740f357c2206c1c544f93de2b1c35549ff4cfa4e3da5f`.

| Updates | Epoch | Validation CE | Accuracy |
|---:|---:|---:|---:|
| 0 | 0 | 7.4866 | 0.06% |
| 100 | 0 | 7.3587 | 8.72% |
| 200 | 0 | 6.7804 | 13.94% |
| 300 | 0 | 6.8102 | 12.40% |
| 400 | 0 | 6.6162 | 14.03% |
| 500 | 0 | 5.9686 | 17.66% |
| 600 | 0 | 6.1641 | 17.05% |
| 700 | 0 | 6.4587 | 17.14% |
| 800 | 0 | 6.2773 | 16.85% |
| 900 | 0 | 5.7290 | 22.08% |
| 1,000 | 0 | 5.4291 | 23.96% |
| 1,100 | 0 | 6.2302 | 17.75% |
| 1,196 | 1 | 5.2699 | 23.54% |
| 1,200 | 1 | 5.2525 | 24.43% |
| 1,300 | 1 | 5.6708 | 21.08% |
| 1,400 | 1 | 5.3085 | 23.61% |
| 1,500 | 1 | 5.2508 | 25.38% |
| 1,600 | 1 | 5.0721 | 26.36% |
| 1,700 | 1 | 5.1213 | 25.52% |
| 1,800 | 1 | 5.1047 | 26.14% |
| 1,900 | 1 | 5.2566 | 24.96% |
| 2,000 | 1 | 4.9693 | 26.84% |
| 2,100 | 1 | 4.9847 | 26.52% |
| 2,200 | 1 | 5.0210 | 27.21% |
| 2,300 | 1 | 4.9416 | 27.19% |
| 2,389 | 2 | 5.2238 | 25.64% |
| 2,400 | 2 | 5.0754 | 26.95% |
| 2,500 | 2 | 4.8367 | 27.93% |
| 2,600 | 2 | 4.8161 | 28.39% |
| 2,700 | 2 | 4.9058 | 27.94% |
| 2,800 | 2 | 4.9597 | 27.75% |
| 2,900 | 2 | 4.8625 | 29.22% |
| 3,000 | 2 | 4.9799 | 27.65% |
| 3,100 | 2 | 5.0002 | 27.46% |
| 3,200 | 2 | 4.8477 | 29.09% |
| 3,300 | 2 | 4.8625 | 27.70% |
| 3,400 | 2 | 4.9478 | 28.28% |
| 3,500 | 2 | 4.7585 | 29.32% |
| 3,587 | 3 | 4.8137 | 29.06% |
| 3,600 | 3 | 4.7283 | 30.49% |
| 3,700 | 3 | 4.8217 | 29.11% |
| 3,800 | 3 | 4.8950 | 29.02% |
| 3,900 | 3 | 4.7903 | 29.74% |
| 4,000 | 3 | 4.7928 | 30.04% |
| 4,100 | 3 | 4.9961 | 29.40% |
| 4,200 | 3 | 4.8744 | 30.06% |
| 4,300 | 3 | 4.8425 | 29.54% |
| 4,400 | 3 | 4.8973 | 29.72% |
| 4,500 | 3 | 4.8244 | 30.36% |
| 4,600 | 3 | 4.8079 | 30.42% |
| 4,700 | 3 | 4.8220 | 30.62% |
| 4,783 | 4 | 4.8492 | 29.94% |
| 4,800 | 4 | 4.7642 | 30.70% |
| 4,900 | 4 | 4.7970 | 30.62% |
| 5,000 | 4 | 4.9050 | 30.17% |
| 5,100 | 4 | 4.8493 | 30.83% |
| 5,200 | 4 | 4.9530 | 29.50% |
| 5,300 | 4 | 4.9203 | 30.10% |
| 5,400 | 4 | 4.8980 | 30.28% |
| 5,500 | 4 | 4.9705 | 30.73% |
| 5,600 | 4 | 4.9120 | 31.03% |
| 5,700 | 4 | 4.9116 | 30.89% |
| 5,800 | 4 | 4.8538 | 30.93% |
| 5,900 | 4 | 4.9511 | 30.67% |
| 5,978 | 5 | 4.9020 | 30.74% |
| 6,000 | 5 | 4.9482 | 31.28% |
| 6,100 | 5 | 4.9609 | 31.29% |
| 6,200 | 5 | 4.9836 | 31.02% |
| 6,300 | 5 | 5.0798 | 30.00% |
| 6,400 | 5 | 5.0358 | 30.92% |
| 6,500 | 5 | 5.0273 | 30.48% |
| 6,600 | 5 | 5.0190 | 30.90% |
| 6,700 | 5 | 5.0508 | 30.28% |
| 6,800 | 5 | 5.0574 | 30.73% |
| 6,900 | 5 | 5.0294 | 30.97% |
| 7,000 | 5 | 5.0125 | 31.41% |
| 7,100 | 5 | 5.0036 | 31.61% |
| 7,173 | 6 | 5.0195 | 30.70% |
| 7,200 | 6 | 4.9968 | 31.89% |
| 7,300 | 6 | 5.0329 | 31.56% |
| 7,400 | 6 | 5.0355 | 31.94% |
| 7,500 | 6 | 5.0447 | 32.34% |
| 7,600 | 6 | 5.1053 | 31.30% |
| 7,700 | 6 | 5.0898 | 31.45% |
| 7,800 | 6 | 5.0750 | 31.99% |
| 7,900 | 6 | 5.0830 | 31.75% |
| 8,000 | 6 | 5.0990 | 32.23% |
| 8,100 | 6 | 5.0830 | 31.84% |
| 8,200 | 6 | 5.0836 | 32.07% |
| 8,300 | 6 | 5.0650 | 32.26% |
| 8,371 | 7 | 5.1006 | 31.74% |
| 8,400 | 7 | 5.1552 | 32.42% |
| 8,500 | 7 | 5.1635 | 32.12% |
| 8,600 | 7 | 5.1764 | 32.56% |
| 8,700 | 7 | 5.2068 | 31.86% |
| 8,800 | 7 | 5.1919 | 31.31% |
| 8,900 | 7 | 5.2171 | 31.63% |
| 9,000 | 7 | 5.2166 | 31.69% |
| 9,100 | 7 | 5.2267 | 31.93% |
| 9,200 | 7 | 5.2270 | 31.70% |
| 9,300 | 7 | 5.2103 | 31.80% |
| 9,400 | 7 | 5.2031 | 32.28% |
| 9,500 | 7 | 5.1718 | 32.55% |
| 9,563 | 8 | 5.2185 | 31.49% |
| 9,600 | 8 | 5.1884 | 32.42% |
| 9,700 | 8 | 5.2442 | 32.75% |
| 9,800 | 8 | 5.2617 | 32.19% |
| 9,900 | 8 | 5.2897 | 32.51% |
| 10,000 | 8 | 5.3104 | 32.08% |
| 10,100 | 8 | 5.2744 | 31.69% |
| 10,200 | 8 | 5.3365 | 31.56% |
| 10,300 | 8 | 5.3544 | 32.00% |
| 10,400 | 8 | 5.3404 | 32.00% |
| 10,500 | 8 | 5.3195 | 31.93% |
| 10,600 | 8 | 5.2960 | 32.06% |
| 10,700 | 8 | 5.3461 | 32.11% |
| 10,766 | 9 | 5.3581 | 30.93% |
| 10,800 | 9 | 5.3773 | 32.75% |
| 10,900 | 9 | 5.4272 | 31.97% |
| 11,000 | 9 | 5.3959 | 32.28% |
| 11,100 | 9 | 5.3866 | 32.59% |
| 11,200 | 9 | 5.4118 | 32.28% |
| 11,300 | 9 | 5.4327 | 32.31% |
| 11,400 | 9 | 5.4684 | 31.43% |
| 11,500 | 9 | 5.4277 | 32.24% |
| 11,600 | 9 | 5.4372 | 32.25% |
| 11,700 | 9 | 5.4600 | 32.43% |
| 11,800 | 9 | 5.4199 | 32.72% |
| 11,900 | 9 | 5.4171 | 31.67% |
| 11,967 | 10 | 5.4096 | 31.85% |
| 12,000 | 10 | 5.4683 | 32.87% |
| 12,100 | 10 | 5.4715 | 32.61% |
| 12,200 | 10 | 5.5470 | 32.90% |
| 12,300 | 10 | 5.5603 | 32.02% |
| 12,400 | 10 | 5.4762 | 32.20% |

## B32fixed

Branch: `exp/encoder32-fixed`. Commit: `68d721ef82de4401f2665c6fdf33579f2865014a`. Repository: https://github.com/fernando-neto-ai/fly-wordbrain.

Saved minimum_validation_ce: `/Users/fernando/fly_wordbrain_connectorch/results/connectorch-encoder-v1-continuation/arms/B32fixed/best.pt`; SHA256 `d6cc4708b65789e036d38974b6eb8adc920354717fd3fa3186aa2424436e030c`.

Saved maximum_validation_accuracy: `/Users/fernando/fly_wordbrain_connectorch/results/connectorch-encoder-v1-continuation/arms/B32fixed/best-accuracy.pt`; SHA256 `def6a3cae3e861dc09e77682db01200b9cd1e3b1ab3f48e51418664e745e8757`.

| Updates | Epoch | Validation CE | Accuracy |
|---:|---:|---:|---:|
| 0 | 0 | 7.3269 | 0.09% |
| 100 | 0 | 7.2118 | 8.09% |
| 200 | 0 | 7.0229 | 11.09% |
| 300 | 0 | 7.3035 | 10.43% |

## C128bounded

Branch: `exp/encoder128-bounded`. Commit: `71d845460c7535f4e3b0ab83f3ae6c28dbe67af3`. Repository: https://github.com/fernando-neto-ai/fly-wordbrain.

No full-arm validation yet.

## D32bounded

Branch: `exp/encoder32-bounded`. Commit: `6b725e1c6727794b1c342d7a433c0d3f644cbc08`. Repository: https://github.com/fernando-neto-ai/fly-wordbrain.

No full-arm validation yet.

All partial scores are next-BPE-token results on the 100-story validation split. Test evaluation is deferred.
This is a timestamped snapshot; each refresh verifies artifact SHA256 checksums. Checkpoint hashes are copied from trainer receipts; checkpoint binaries are not fetched by this command.
