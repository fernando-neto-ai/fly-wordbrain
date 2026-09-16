# H32rank32fixed: rank32 decoder launch

H full training launched on **macm3 at 00:57:20 UTC on 2026-09-16**, after its
native numerical preflight and separate eight-update optimizer smoke passed.
The first full validation was verified at 100 updates. Passing these gates establishes that the
rank32 model is training through the intended path; it does not establish quality.

The user selected this experiment after the completed E/G comparison. The primary
baseline is the accepted, stopped **G32rank64fixed** run. H starts from scratch,
retaining G's width32 encoder, eight input-delay slots, 1,024-token BPE, seed42,
data and planned training recipe. It does not initialize from a G checkpoint.

## Architecture and comparison

| Component | G rank64 | H rank32 |
|---|---:|---:|
| Token embedding and input projection | 482,816 | 482,816 |
| Bias-free decoder | 3,226,688 | 1,613,344 |
| Neuron gain, recurrent gain and bias | 148,179 | 148,179 |
| Layer normalization | 98,786 | 98,786 |
| **Total trainable** | **3,956,469** | **2,343,125** |

H's decoder is 49,393→32→1,024 with no intermediate activation. This halves the
decoder, removing 1,613,344 parameters, while retaining readout access to every
neuron. All 49,393 neuron identities and 9,050,172 canonical edge endpoints and
stored base weights remain exact. No additional edge gains, unfreezing or
rewiring are part of H. The existing neuron gains and biases remain trainable.

The same from-scratch initialization policy and seed preserve the encoder and
neuron initialization, but rank32 changes head shapes and initial logit variance.
Consequently, H/G is not a strictly initialization-matched rank intervention.
It is also not an SVD compression or fine-tune of G.

The unchanged planned schedule is 30+14 epochs, AdamW, batch 8 and TBPTT32,
with full validation every 100 updates. Training uses 1,000 stories; validation
uses the same 100 stories and 21,874 non-padding targets. The 100 reserved test
stories remain unused. See the [configuration](../configs/H32rank32fixed.json)
for pinned source, data, anatomical groups and optimizer settings.

## Verified launch gates

The [rank32 numerical preflight](../records/rank32-preflight.json) passed on
Apple M3 Max at 00:50:08 UTC on 2026-09-16. CPU/MPS CE difference was zero;
maximum gradient relative-L2 difference was 3.0516065×10⁻⁵. Maximum absolute
logit and state differences were 2.8908253×10⁻⁶ and 4.5426190×10⁻⁵ respectively.
It verified exact graph identity and unchanged source/input/ConnecTorch hashes.
This check performed backward passes but zero optimizer steps.

The subsequent eight-update macm3 optimizer smoke passed. Both decoder factors
changed weights and had nonzero gradient entries: 1,580,576 in the first factor
and 32,768 in the second. Full training followed with
`PYTORCH_ENABLE_MPS_FALLBACK=0`. Training and smoke workers run serially.

## Run identity

- Experiment branch: `exp/encoder32-readout32-fixed`.
- H campaign source commit: `3754f6578fa7a8827026e434c1287e2043ec99f0`.
- Inherited training-source commit: `3aaa2120` (unchanged trainer/model sources).
- Branch-mapping commit: `33ab371106baee7ce7f46c80b65fef07c7120f59`.
- Launch ID: `2556b84a-d9f9-49c4-83ee-bc6cd7434258`.
- Launch time: `2026-09-16T00:57:20.608982Z`.
- Manifest SHA256: `880b2e9cf957c7b16607a118f7cd058caebab15d7de509d131de5fb20eaadaac`.
- Controller/trainer PIDs at launch: `85176` / `85584`; these are historical
  process locators, not a current liveness check.
- Remote output: `/Users/fernando/fly_wordbrain_connectorch/results/connectorch-rank32-v1`.
- Full arm: `arms/H32rank32fixed`; smoke: `smokes/H32rank32fixed`.

The [launch receipt](../runs/rank32-v1-launch.json) and native manifest bind
execution to exact artifacts. Local partial snapshots are refreshed under
`results/connectorch-rank32-v1`; `scripts/refresh_connectorch_rank32_progress.py`
shows H alongside preserved G and both independent checkpoint selectors.
The existing follow-up monitor is active every 10 minutes and now targets H;
it reports meaningful changes and applies the declared plateau checks without
automatically starting a successor.

## Initial full validation

The snapshot fetched at **00:58:32 UTC on 2026-09-16** verified the first full
validation at **100 optimizer updates**:

| Validation CE | Top-1 accuracy | Correct / targets | Stories |
|---:|---:|---:|---:|
| 5.827075091491313 | 8.448386211941117% | 1,848 / 21,874 | 100 |

The trainer reported 40.0879 elapsed seconds at that measurement, including
validation/checkpoint work; this is not an isolated GPU-training throughput
benchmark. The worker had advanced to 200 observed updates when that snapshot
was fetched. These are launch measurements, not H/G quality conclusions. The
progress helper provides subsequent dated measurements.

## Assessment protocol

G stopped at 16,808 observed / 16,800 durable updates after 14 completed epochs.
Its independent retained winners are CE 3.093705 at 16,600 and accuracy 36.8885%
at 15,552; see the [accepted G stop](G32rank64fixed-accepted-early-stop.md) and
[post-G assessment](rank64-post-G-assessment.md).

Retain H's actual minimum-validation-CE and maximum-validation-accuracy weights
separately. Compare H/G at common update and non-padding training-target budgets,
as well as each arm's independent winners and actual runtime. A difference in
stopping budget must remain explicit. After stopping, compare identical text
prompts for both selectors and inspect training overlap before drawing quality
conclusions. Keep the reserved test untouched.

Apply the established plateau review only after at least eight completed epochs:
global-best CE gain below 0.10 **and** accuracy gain below 0.5 percentage points
over the last 2,000 updates, with training-loss and equal-window inspection and
a fresh pre-stop recheck. More than 0.10 CE or one accuracy point of degradation
at comparable budgets, or consistent text deterioration, triggers quality review.
These are practical review thresholds, not significance tests. Rank32 alone is
selected; no further rank reduction or brain adaptation is automatically queued.
