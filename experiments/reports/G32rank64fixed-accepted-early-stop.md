# G32rank64fixed — accepted early stop

G stopped on macm3 at **2026-09-15 23:37:28 UTC**, after 16,808 observed
updates, with a complete durable checkpoint at **16,800 updates / 14 completed
epochs**. The planned 52,609-update schedule was not completed. This was an
operational plateau stop under standing user authorization; no subsequent
training experiment was launched by the stop helper.

The authoritative disposition is the [native accepted-stop receipt](../records/G32rank64fixed-accepted-early-stop.json).
The native dispatcher failure and trainer exit code **−15** record intentional
SIGTERM cleanup, not an unexplained training failure. Both exact process
identities—dispatcher 20942 and trainer 20985—have recorded kernel exit events
and confirmed cessation. Native failure records were preserved.

## Retained validation checkpoints

| Selector | Update | Validation CE | Accuracy | Correct / targets |
|---|---:|---:|---:|---:|
| Minimum CE | 16,600 | 3.0937050783 | 36.3673768% | 7,955 / 21,874 |
| Maximum accuracy | 15,552 | 3.1034503651 | 36.8885435% | 8,069 / 21,874 |

Both winners match the full 183-row validation history, including epoch-end
evaluations. Every evaluation covers the same 100 validation stories and
21,874 next-BPE-token targets. The maximum-accuracy winner is an epoch-end
checkpoint; its non-round update count is expected. The reserved test remains
untouched.

The two winners and `latest.pt` were preserved under the native arm's
`stopped-checkpoints/` directory. Native verification checked each checkpoint's
bytes, manifest, parameters, cursor and frozen-graph hashes before and after
stopping, and checked the preserved files again. Optimizer, scheduler, CPU/MPS
RNG and recurrent cache state remain in the checkpoints.

| Preserved file | SHA256 |
|---|---|
| `best.pt` | `a552be53892d43d461e65e2c93488779ce0e7f12d93860d9968d90035df23b75` |
| `best-accuracy.pt` | `d25b495e1bb9f9096331af393eefcd55564d5394849a095f78b4daf6d5dbcbb8` |
| `latest.pt` | `ca70f8d802652db7f1ec6cf2d839ec71c4423e201ff07241cf70ac7c53029c3e` |

## Plateau evidence

Over the trailing 2,000 updates, comparing global bests through 14,800 with
global bests through 16,800, CE improved **0.01450427** and accuracy improved
**0.23315352 percentage points**. Both are below the established thresholds
of 0.10 CE and 0.5 percentage points, after the minimum eight completed epochs.

| Periodic validation window | Evaluations | Mean CE | Median CE | Mean accuracy |
|---|---:|---:|---:|---:|
| (14,800, 15,800] | 10 | 3.11446415 | 3.11374835 | 36.3120600% |
| (15,800, 16,800] | 10 | 3.11660817 | 3.11638425 | 36.2206272% |

Training CE continued falling across completed epochs 12–14
(2.596823 → 2.518018 → 2.443595), while these equal validation windows were
approximately flat. This supports an operational diminishing-returns stop;
it does not establish that later training or scheduled learning-rate changes
could never improve the model.

## Architecture and audit scope

G retains the full 49,393-neuron, 9,050,172-edge graph, encoder width 32 and
eight explicit input delays. Its readout factors have shapes **64 × 49,393**
and **1,024 × 64**, totaling 3,226,688 parameters. The complete trainable model
has **3,956,469 parameters**: 482,816 encoder, 148,179 neuron, 98,786 output
LayerNorm and 3,226,688 readout parameters, with zero additional edge-gain
parameters. Stored graph and grouping buffers remain unchanged; neuron gains,
biases and recurrent gains were trainable.

The independent local receipt audit verified all **28 fetched artifact
hashes**, all six acceptance-bound arm artifacts, the stop-receipt and campaign
manifest bindings, agreement between original and preserved selector receipts,
all nine current trainer/model source hashes, and frozen-graph receipts across
all validation rows. Plateau statistics were recomputed from the full history.
The copied accepted receipt is byte-identical to the native fetched receipt,
SHA256 `1def3071e7bdcd062c2aa3b8d81d018055466a7b1f1a2aeea9bf2e169582387d`.

This local audit verified the refreshed snapshot fetched at **2026-09-16
00:04:48 UTC**. It did not
reload checkpoint binaries or inspect remote processes; those checks are
attested by the native stop receipt. The separately fetched
[stop decision](../records/G32rank64fixed-early-stop-decision.json) was also
copied byte-for-byte and independently verified against the acceptance-bound
SHA256 `743d8f2fe8e37b0aee1b8e1a8d266ee39125610184d371af364210cb68791d0d`;
its plateau evidence and launch identity match the accepted receipt.
Compare E and G at common recorded budgets as well as their
retained winners: their total training budgets differ. Text-quality and
checkpoint-replay assessments are reported separately.
