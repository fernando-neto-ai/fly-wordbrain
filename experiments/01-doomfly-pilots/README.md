# Stage 1 — Doomfly: a frozen 166,700-neuron brain as a language feature extractor

**Objective.** Before touching a connectome-shaped *language model*, find out whether the
raw measured wiring of a fruit fly, driven by spiking dynamics and read out by a small
linear head, carries any usable signal about the next word in a story.

**Answer: no measurable advantage.** These are negative results, and they are kept because
they bound what the later stages may claim.

Everything here uses the **full 166,700-neuron Doomfly subset with spiking dynamics and
word-level vocabulary**. Stage 2 onward uses a different model entirely — a 49,393-neuron
central-brain subset with rate dynamics and byte-level BPE. **Their scores are not
comparable.** Word accuracy is not BPE accuracy.

## What was run

| Experiment | What trained | Outcome |
|---|---|---|
| [Frozen-brain word decoder](REPORT.md) | 263,168-parameter linear decoder only | Test PPL 189.16 vs bigram 124.30 — worse than a bigram |
| [Ordered previous/current word pair](PAIR_REPORT.md) ([protocol](PAIR_PROTOCOL.md)) | 526,336-parameter two-head decoder | PPL 185.11; same-capacity head on the raw stimulus reached 89.51 |
| [Differentiable plasticity](PLASTIC_PROTOCOL.md) + [Metal backend](MPS_REPORT.md) | 347 hDeltaB edge gains via 20 shared rules | Frozen/fixed/learned validation CE differed in the 6th decimal |
| [Data expansion](DATA_EXPANSION.md) ([dashboard](VALIDATION_DASHBOARD.md)) | — | 8,192 stories / 1,021,293 targets; three-arm run superseded mid-flight |
| [Top-ten action selection](ACTION_SELECTION.md) | 2,570-parameter reranker over a trigram proposer | 30.36% vs matched 30.39% baseline — no gain |
| [Seven-word feedback](FEEDBACK_EXPERIMENT.md) | 11,408 parameters over 8,192 anatomical edges | Fast weights changed 49/1,968 choices; accuracy still below the count baseline |
| [Memory probe](MEMORY_PROBE.md) | diagnostic readouts at frozen checkpoint 3072 | Early pooled identity/order 62.5%/100%, final readouts at chance |
| [Retention](RETENTION_REPORT.md) ([protocol](RETENTION_PROTOCOL.md)) | 8,838 plasticity + 3,084 head values vs matched control | Both arms selected step 0; CE worsened 1.5101 → 1.7387/1.7356 |

Data contracts and the word-level split policy are in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## What these results do and do not establish

They **do** show that a frozen connectome read by a small head did not beat cheap count
baselines under these budgets, that plasticity parameters genuinely received gradients and
changed choices, and that the wiring, causality and independent-logit audits passed.

They do **not** establish a capacity limit of the fly connectome. The memory probe's
early-window success reflects the cue still being present in the sensory input — that is
recognition, not retention. Failure to decode is not proof that information is absent.
Each run used one seed, one objective and a bounded budget.

The FlyOCR review recorded here is a reading of upstream results, not our reproduction.
