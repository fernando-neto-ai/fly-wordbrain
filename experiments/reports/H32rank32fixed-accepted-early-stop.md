# H32rank32fixed — accepted early stop

H stopped on macm3 at **2026-09-16 02:26:35 UTC**, with **17,409 observed
updates, 17,400 durable updates and 14 completed epochs**. The planned
52,609-update schedule was not completed. The
[native accepted-stop receipt](../records/H32rank32fixed-accepted-early-stop.json)
records an operational plateau stop under standing user authorization.

The dispatcher (PID 85176, birth `Tue Sep 15 21:57:06 2026`) and trainer
(PID 85584, birth `Tue Sep 15 21:57:20 2026`) both have recorded kernel exit
events and confirmed cessation. Native `failed` status and exit code **−15**
describe intentional SIGTERM cleanup; the accepted receipt governs the logical
status. No successor training run was launched by the stop helper.

## Checkpoints and architecture

Both independent selectors choose **update 16,600: CE 3.1376539815,
accuracy 35.4713358% (7,759 / 21,874 targets)**. They remain separately saved
files with separate byte hashes. Their cursor is epoch 13, batch 108, offset
224, phase 0. The latest resumable checkpoint is epoch 14, batch 67, offset
32, phase 0, update 17,400.

| Preserved file | SHA256 |
|---|---|
| `best.pt` | `6b9badc7fb43d883b50ff3538601b333690317cab572a7c90619c83633cf0515` |
| `best-accuracy.pt` | `6ed28abd44ec3d7049573b6ebdc3b81723991da20def653b23b3e15af001b2c8` |
| `latest.pt` | `677fc95e152baefd49527e8a50a377e8c7a274ab381b6075ee46d6c584930834` |

All three files are preserved under the native arm's `stopped-checkpoints/`
directory. Native checks verified their bytes, parameters, manifests, cursors
and graph hashes before and after stopping and after preservation. Optimizer,
scheduler, CPU/MPS RNG and recurrent cache state remain available for resumption.

The model retains 49,393 neurons, 9,050,172 stored edges, encoder width 32 and
eight explicit delays. Readout factors are **32 × 49,393** and **1,024 × 32**:
1,613,344 readout parameters and **2,343,125 total trainable parameters**.
The remaining counts are 482,816 encoder, 148,179 neuron and 98,786 output
LayerNorm parameters, with no additional edge-gain parameters. Stored graph
and grouping buffers are preserved; existing neuron gains and biases were
trainable.

## Why training stopped

Global bests through **15,400 → 17,400** improved by **0.02412440 CE** and
**0.25144007 accuracy percentage points**. Both strict stopping thresholds
were met: less than 0.10 CE and less than 0.5 points over 2,000 updates, after
at least eight completed epochs.

| Periodic validation window | Evaluations | Mean CE | Mean accuracy |
|---|---:|---:|---:|
| (15,400, 16,400] | 10 | 3.16598244 | 35.1682363% |
| (16,400, 17,400] | 10 | 3.16079478 | 35.1033190% |

Training CE continued falling across epochs 12–14
(2.917381 → 2.854495 → 2.796941), while validation gains had diminished.
This is an operational stopping rule, not proof that later training or a
scheduled learning-rate change could never help.

## Comparison with rank64 G

The last shared validation budget is **16,800 updates**. At that exact update,
H has CE **3.150774 / accuracy 35.3342%**, versus G's **3.108550 / 36.5777%**:
H is worse by **0.042224 CE and 1.2435 accuracy points**.

The retained-winner comparison is separate. H's two winners are at 16,600;
G's minimum CE is **3.093705 at 16,600**, and maximum accuracy is
**36.8885% at 15,552**. H's retained-best gaps are **0.043949 CE and 1.4172
accuracy points**. These same winners had already been selected by the shared
16,800-update budget. Actual durable stopping budgets differ: H 17,400 versus
G 16,800; both completed 14 epochs. Halving the readout therefore carries a
measurable validation accuracy cost in this run. Text-quality evaluation is
pending; no storytelling-quality conclusion is made here.

## Receipt audit

The independent local audit verified **31 fetched artifact hashes**, six
acceptance-bound arm files, the raw stop and decision receipts, the campaign
manifest, all nine current trainer/model source hashes, and all **189** full
validation records. Every record covers 100 stories / 21,874 next-BPE targets;
both selectors match the complete history, and graph hashes match preserved G.
Plateau statistics were recomputed. The reserved test remains untouched.

| Raw receipt | SHA256 |
|---|---|
| Accepted stop, copied byte-for-byte | `03cbf54ad62a789b0c38c97e02479d987e24a78045a32fd707ba050db4dc8a20` |
| Stop receipt | `a7293553253fcc0426141a636001996f4a5485dd1066e56909a29cd548129521` |
| Stop decision | `f86e3b42f014b838c0ff197317637257abff69a34a54ba35fd4f06c4cdbec1a8` |

The snapshot was fetched at **2026-09-16 02:26:42 UTC**. This local audit did
not reload native checkpoint binaries or inspect remote processes; those
checks are attested by the verified native receipts. The campaign identity is
`2556b84a-d9f9-49c4-83ee-bc6cd7434258`, with manifest SHA256
`880b2e9cf957c7b16607a118f7cd058caebab15d7de509d131de5fb20eaadaac`.
