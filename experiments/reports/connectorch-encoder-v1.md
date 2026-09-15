# Encoder/connectome campaign launch

The campaign started on **Apple M3 Max (macm3)** on 2026-09-15. The original
reference was accepted and stopped by the user; both final validation winners
are preserved. This does not imply completion of its planned 44 epochs or full
training-quality equivalence.

Numerical preflight passed: the fixed ConnecTorch CPU case exactly matched the
upstream reference on the tested inputs. Maximum relative gradient errors on MPS
were 1.09e-5 for the fixed model and 2.20e-5 for the compressed adaptive model.
All four eight-update training smoke runs passed. Adaptive edge factors moved;
fixed factors remained unchanged. Canonical graph buffers, edge signs and zero
edges were preserved throughout.

Full training is now running serially in the order below. All arms initialize
learned parameters from scratch and use the same 30+14 epoch recipe, seed and
dataset. The final test remains reserved. Minimum-CE and maximum-accuracy weights
are retained separately. This is an execution checkpoint, not a quality verdict.

| Arm | Input width | Edge adaptation | Branch |
|---|---:|---|---|
| A128fixed | 128 | fixed | [encoder128-fixed](https://github.com/fernando-neto-ai/fly-wordbrain/tree/exp/encoder128-fixed) |
| B32fixed | 32 | fixed | [encoder32-fixed](https://github.com/fernando-neto-ai/fly-wordbrain/tree/exp/encoder32-fixed) |
| C128bounded | 128 | 18,322 gains, ±10% | [encoder128-bounded](https://github.com/fernando-neto-ai/fly-wordbrain/tree/exp/encoder128-bounded) |
| D32bounded | 32 | 18,322 gains, ±10% | [encoder32-bounded](https://github.com/fernando-neto-ai/fly-wordbrain/tree/exp/encoder32-bounded) |

All four retain 49,393 neurons, 9,050,172 directed base edges, eight explicit
input delays and the full readout. Encoder parameters fall 75% at width32;
total model parameters fall only about2.75% because the full readout remains.
The ±10% limit applies to base-edge multipliers; original neuron gains remain
unconstrained.

Refresh timestamped local partial results with
`python scripts/refresh_connectorch_progress.py` from a control host configured
with `cluster_runner`. Published per-arm run snapshots live on their branches.
The campaign runs under `/Users/fernando/fly_wordbrain_connectorch` on macm3;
its output is `results/connectorch-encoder-v1`. Checkpoint binaries and downloaded
data stay in verified artifact storage rather than Git.

Evidence: [source/branch mapping](../records/connectorch-encoder-v1-provenance.json),
[numerical checks](../records/connectorch-m3-preflight.json),
[training smoke checks](../records/connectorch-m3-smokes.json), and
[accepted reference](../records/ngxson-reference/accepted-baseline.json).
