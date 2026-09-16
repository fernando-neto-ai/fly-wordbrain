# Fly Wordbrain

Train and study a language model built around a measured fruit-fly connectome,
using native Apple GPU sparse kernels. The current pipeline reproduces the
[ngxson Fly LLM](https://huggingface.co/ngxson/fly-llm-hf) architecture and tests
smaller input encoders and output decoders with minimally adapted synaptic weights through
[ConnecTorch](https://github.com/us/connectorch).

**All real training runs on macm3**, including pilots, fine-tuning and training
smoke tests. Development, report inspection and small correctness checks may run
locally. GPU jobs on macm3 run serially.

## What this model preserves

The pinned language-model checkpoint contains **49,393 neurons and 9,050,172
directed edges**, a 1,024-token BPE, eight explicit token-delay slots and a
full-state linear readout. Its original architecture has **52,756,661 trainable
parameters**, including a 50,578,432-parameter output matrix. This is a central
brain subset with rate dynamics; the separate historical Doomfly experiments
used 166,700 neurons and spiking dynamics.

The Apple GPU implementation preserves the original recurrence and canonical
graph buffers. The first ConnecTorch experiment changes input width and optional
bounded edge gains while keeping neuron identities, every endpoint, tokenizer,
readout and eight input-history slots matched.

| Experiment | Input width | Edge adaptation | Trainable parameters |
|---|---:|---|---:|
| A128fixed | 128 | Fixed base weights | 52,756,661 |
| B32fixed | 32 | Fixed base weights | 51,308,213 |
| C128bounded | 128 | Shared gains, ±10% | 52,774,983 |
| D32bounded | 32 | Shared gains, ±10% | 51,326,535 |

The bounded arms add 18,322 source/destination type-gain parameters. Existing
per-neuron gains remain unconstrained, so a ±10% base-edge bound does not bound
the complete effective recurrence. Width 32 reduces input-interface parameters
by 75%; it reduces the full fixed model by only 2.75%.

The [rank64 decoder experiment](experiments/DECODER_REDUCTION.md) keeps width32
and uses 3,226,688 decoder parameters, 3,956,469 total, with fixed base edges.
G is now accepted as an early stop: **14 completed epochs, 16,808 observed /
16,800 durable updates**. Its separate retained winners are validation
**CE 3.093705 at 16,600** and **accuracy 36.8885% at 15,552**. E's preserved
rank128 baseline reached CE 3.149704 and accuracy 36.5822%; actual budgets differ.
The [post-G assessment](experiments/reports/rank64-post-G-assessment.md),
[matched generated texts](experiments/reports/rank64-generated-texts.md) and
[text review](experiments/reports/rank64-text-review.md) document the comparison.
F remains stopped; see the [experiment registry](experiments/INDEX.md) for receipts.

The subsequent **H32rank32fixed** experiment is also accepted as an early stop:
**14 completed epochs, 17,409 observed / 17,400 durable updates**. Both selectors
retain update **16,600: CE 3.137654, accuracy 35.4713%**. Its 1,613,344-parameter
decoder halves G's readout; **2,343,125 total parameters** is **40.7774% fewer**
than G, with the same encoder and canonical graph. All four G/H checkpoint
validation replays passed on macm3 without backward passes or parameter/graph
changes; the reserved test was not evaluated. See the
[H stop report](experiments/reports/H32rank32fixed-accepted-early-stop.md),
[post-H assessment](experiments/reports/rank32-post-H-assessment.md),
[matched texts](experiments/reports/rank32-generated-texts.md), and
[validation curves](experiments/reports/figures/rank32-vs-rank64-validation.png).
G remains the recommended quality baseline; H preserves a smaller option with
a 1.4172-point retained-best accuracy cost. No subsequent training experiment
is selected automatically.

## Start here

- [Fly Stories demo](docs/RANK128_DEMO.md): an articulated fly, recorded rank128
  stories and the corresponding measured neuron-state replay. Run locally with
  `cd demo && npm ci && npm run dev -- --port 8781 --strictPort`.
- [Complete pipeline](docs/PIPELINE.md): environments, pinned downloads, data,
  training, checkpoint selection, evaluation and partial results.
- [Experiment registry](experiments/INDEX.md): configurations, branch names,
  controlled comparisons and result-recording rules.
- [Reference implementation and provenance](NGXSON_REPLICATION.md).
- [Quality audit](NGXSON_QUALITY_AUDIT.md) and
  [next-stage research plan](CONNECTORCH_NEXT_STAGE.md).
- [Historical frozen-brain and fast-weight pilots](LEGACY_PILOTS.md).

Run scripts from the checkout root. The project uses pinned requirements and
repository imports; an editable package install is not needed. On macm3, with
Python 3.13 available:

```bash
python3.13 -m venv .venv-connectorch
.venv-connectorch/bin/python -m pip install -r requirements-connectorch.txt
.venv-connectorch/bin/python scripts/prepare_ngxson.py
.venv-connectorch/bin/python scripts/prepare_ngxson_data.py
```

ConnecTorch anatomical groups need an additional verified source-graph preparation
step described in the pipeline. Keep all generated data and checkpoints outside
Git; commit configurations, source provenance and compact reports on experiment
branches.

## What has been reproduced

Released-checkpoint inference and CPU/Apple GPU numerical agreement are verified.
Our reconstruction reaches similar top-1 accuracy under a declared practical
margin, but substantially worse cross-entropy than the released model. Familiar
text continuations can reproduce training passages. The author did not publish
an exact training split or full trainer, so **full training-quality reproduction
has not been established**.

On 2026-09-15 the user accepted the current reconstruction as a working reference,
requested that its training stop, and authorized the next experiments. This is
an experimental decision; it does not change the quality-audit findings. Run
receipts record the actual stopping cursor and retained checkpoints.

An anatomical language-learning advantage requires matched graph controls,
multiple seeds and independent evaluation. These first four arms test encoder
compression and bounded adaptation; they do not yet establish that advantage.

## Source and data attribution

The reference is pinned to ngxson revision
`65c677b3d566a2e9793d5f72999cdb441c6c0a9f`; ConnecTorch is pinned to
`4bbfb645099aeb85bdbf850e1a87cc094769af87`. Downloaded model/data assets retain
their upstream terms. The Fly LLM model card declares CC BY 4.0; ConnecTorch
library code uses MIT. MaleCNS/FlyEM data attribution includes HHMI Janelia,
University of Cambridge, MRC LMB and Google Research. TinyStories is by Eldan
and Li; its source receipts and dataset terms must accompany redistribution.

The vendored Doomfly subset retains its [MIT license](vendor/doomfly/LICENSE),
[third-party notices](vendor/doomfly/THIRD_PARTY_NOTICES.md) and
[source inventory](vendor/DOOMFLY.md). Those licenses apply to their identified
upstream material, not automatically to this repository's original code. No new
blanket license is assigned here.
