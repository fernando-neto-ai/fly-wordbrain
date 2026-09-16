# Rank64 post-G assessment

**Keep encoder width 32 and the rank 64 readout as the working baseline, with both checkpoint selectors retained.** G removes half the decoder parameters and performs at least as well on the recorded common-budget validation comparison. The generated examples show mixed changes and persistent coherence problems, so this is a useful compression result within this experiment, not evidence of better storytelling or an anatomical advantage. **No further unfreezing, rewiring, resumed training, or next run is selected by this assessment.**

G stopped under the established plateau rule at **16,808 observed / 16,800 durable updates, 14 completed epochs**; E stopped at **15,918 observed / 15,900 durable updates, 13 completed epochs**. Neither completed the planned 52,609-update schedule. See the [accepted-stop audit](G32rank64fixed-accepted-early-stop.md) and [native receipt](../records/G32rank64fixed-accepted-early-stop.json).

## Size and architecture

| Component | E: width 32 / rank 128 | G: width 32 / rank 64 |
|---|---:|---:|
| Encoder |482,816|482,816|
| Trainable neuron gain, recurrent gain and bias |148,179|148,179|
| Output LayerNorm |98,786|98,786|
| Readout |6,453,376|3,226,688|
| Additional edge-gain parameters |0|0|
| **Total trainable parameters** |**7,183,157**|**3,956,469**|

The readout shrank **50%** and the whole model **44.9202%**, removing 3,226,688 parameters. G retains the same 49,393 neurons, 9,050,172 stored directed edges, eight input delays and 1,024-token vocabulary. Canonical topology, stored edge values, input/output indices and grouping buffers remained identical. The neuron parameters were trainable in both arms; “fixed” describes the canonical connectome, not every learned value.

## Checkpoint replay and retained winners

Sequential inference ran on idle **macm3 / Apple M3 Max / MPS**, without training. All four retained checkpoints replayed **100 validation stories / 21,874 next-BPE-token targets**. Correct counts matched their selection receipts exactly; the largest absolute CE difference from the logged value was **2.78031×10⁻⁸**, below the 1×10⁻⁴ tolerance. Checkpoint hashes and cursors matched, all per-story totals summed to their summaries, parameter/graph hashes stayed unchanged before and after inference, and no backward kernel ran. The reserved test was not evaluated.

| Arm / selector | Update | Actual training targets | Replayed CE | Accuracy | Correct / 21,874 |
|---|---:|---:|---:|---:|---:|
| E minimum CE |13,700|2,689,179|3.14970417|36.4816677%|7,980|
| E maximum accuracy |14,300|2,806,777|3.15990839|36.5822438%|8,002|
| G minimum CE |16,600|3,257,286|3.09370508|36.3673768%|7,955|
| G maximum accuracy |15,552|3,052,348|3.10345037|36.8885435%|8,069|

Comparing like selectors, G's CE-selected checkpoint improves CE by **0.05599909** but loses **0.11429094 percentage points of accuracy** (25 fewer correct targets). Its accuracy-selected checkpoint improves accuracy by **0.30629972 points** (67 more correct targets) and CE by **0.05645802**. Both selectors matter: lower CE and higher top1 accuracy are different objectives, and G's two winners are different checkpoints. These retained-winner comparisons have unequal training budgets.

## Comparison at the same training budget

Both runs reached **15,900 updates / 3,120,358 actual training targets**. Chunk-level accounting, rather than updates multiplied by nominal batch size, establishes the shared target budget.

| Best attained through 15,900 updates | E | G |
|---|---:|---:|
| Minimum validation CE |3.14970420 at 13,700|3.10345037 at 15,552|
| Maximum validation accuracy |36.5822438% at 14,300|36.8885435% at 15,552|

Within this common search budget, G improves best CE by **0.04625383** and best accuracy by **0.30629972 percentage points**. At the exact 15,900-update evaluation, G also has lower CE (**3.13186203 vs 3.21512608**) and slightly higher accuracy (**35.8964981% vs 35.8050654%**).

The 22 matched evaluations in **(13,900,15,900]**, including epoch-end evaluations, average **CE 3.12064279 / accuracy 36.2387475% for G**, versus **CE 3.17948498 / accuracy 36.1286126% for E**. Thus the comparison is not supported solely by one best checkpoint. These are descriptive historical validation scores; text from unavailable historical checkpoints was not regenerated. They are not independent test results or a significance analysis.

## What changed in the neurons

Shared encoder, neuron and normalization initialization hashes match exactly. All 49,393 values in each learned neuron vector—gain, recurrent gain and bias—differ from initialization in both retained arms. All canonical graph hashes remain exact, with no edge-gain parameters or rewiring.

At the maximum-accuracy checkpoints, G differs from E by **24.4544% relative L2 in gain**, **0.78684% in recurrent gain**, and **0.101924 RMS in bias**. The elementwise product `gain × rec_gain` differs by **18.4466% relative L2**; for the minimum-CE pair that product differs by **19.3157%**. Relative to the shared initialization, the product displacement is **48.3510% for E's accuracy winner** and **50.6077% for G's accuracy winner**. These comparisons retain the actual unequal checkpoint budgets.

The update is `state_new = (1-leak)*state + leak*tanh(gain*(rec_gain*(W*state) + input) + bias)`. The product is therefore a per-neuron scaling of recurrent input before `tanh`. Its vector-norm difference is **not** a percentage of changed edges, increased brain activity, or proof that the connectome took over 18% more work. Saturation, bias, input scaling and leak also affect state dynamics. A smaller head changed the learned solution; this audit does not establish why it helped.

## Text review and limits

The [exact machine-generated report and all 24 continuations](rank64-generated-texts.md) are preserved byte-for-byte. The [independent prompt-by-prompt review](rank64-text-review.md) finds mixed changes: some G sentences improve locally, while the accuracy-selected Tom story is a clear regression in character continuity. Both models lose the needle, ball/box, bird/rain and cake-sharing scenarios. Neither an occasional fluent passage nor EOS establishes sustained prompt tracking.

All examples use the same six prompts, greedy decoding, one BOS, a fresh cache and an 80-new-BPE-token cap with EOS stopping. Longest contiguous training overlaps reach **13/10 words for E's CE/accuracy winners** and **12/20 for G's**. None of the 24 entire prompt-plus-continuation strings is a normalized training-story prefix. That does not establish absence of memorization, and common short phrases do not establish memorization either.

This is one seed, trained on 1,000 stories, with checkpoint selection and evaluation on the same validation split. No randomized-connectome or causal burden-transfer control was run; rank changes readout shape and initial logit scale. The early stops leave later scheduled learning-rate phases unexplored. Keep the compression result, preserve both winners and the coherence failures, and leave the reserved test untouched until the comparison protocol is fixed.

## Evidence

The [compact assessment record](../records/rank64-post-G-assessment.json) contains source SHA256 values, checkpoint identities, actual target budgets, parameter counts and neuron summaries. Native quality inference completed **2026-09-15 23:39:07 UTC**. Its source `results.json` SHA256 is `8a26d30b30e4a77fd29c98af5a4f504a2ba6b077249b30f8b94d7a0e34f48344`. The unedited generated-text report SHA256 is `cde19ce204c789a2f0a5a558309702cf7ba47f0da51c1de0f208aace5ca23e88`.
