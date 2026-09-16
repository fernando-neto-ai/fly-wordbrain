# Experiment registry

Repository: `fernando-neto-ai/fly-wordbrain`. All real training runs on macm3,
including smoke tests. This registry describes configurations and lineage;
run manifests and stop/completion receipts are authoritative for execution.

| Experiment | Branch | Configuration | Trainable parameters |
|---|---|---|---:|
| Original reconstruction | `exp/ngxson-reference` | [reference](configs/ngxson-reference.json) | 52,756,661 |
| A128fixed | `exp/encoder128-fixed` | [A](configs/A128fixed.json) | 52,756,661 |
| B32fixed | `exp/encoder32-fixed` | [B](configs/B32fixed.json) | 51,308,213 |
| C128bounded | `exp/encoder128-bounded` | [C](configs/C128bounded.json) | 52,774,983 |
| D32bounded | `exp/encoder32-bounded` | [D](configs/D32bounded.json) | 51,326,535 |
| E32rank128fixed | `exp/encoder32-readout128-fixed` | [E](configs/E32rank128fixed.json) | 7,183,157 |
| F32rank128bounded | `exp/encoder32-readout128-bounded` | [F](configs/F32rank128bounded.json) | 7,201,479 |
| G32rank64fixed (accepted early stop) | `exp/encoder32-readout64-fixed` | [G](configs/G32rank64fixed.json) | 3,956,469 |
| H32rank32fixed (accepted early stop) | `exp/encoder32-readout32-fixed` | [H](configs/H32rank32fixed.json) | 2,343,125 |

## Protocol and status

On 2026-09-15 the user accepted the current reference checkpoint quality as
sufficient to move forward and requested an early stop. This is the working
reference for subsequent experiments; it is not a claim that all 44 planned
epochs completed or that the original model's probability quality was reproduced.
The highest-accuracy and lowest-CE retained weights remain separate artifacts.
Observed stopping update: 24,736; latest durable checkpoint: 24,700. The accepted
minimum-CE winner is update 3,500 (CE 4.692816, accuracy 30.5797%); the accepted
maximum-accuracy winner is update 24,100 (CE 5.893177, accuracy 35.1605%). The
original macm3 run's `accepted-baseline.json` binds these facts to exact artifact
hashes and the user instruction.

A/B/C/D form a matched 2×2 comparison of input width (128/32) and base-edge
adaptation (fixed/bounded10). All four retain the 50,578,432-parameter readout,
eight explicit delays, 49,393 neurons, 9,050,172 directed edges, seed 42 and the
same data/order and originally planned training budget. Bounded arms add 18,322 source/destination
cell-type gain values with multipliers in [0.9,1.1]; no endpoint is rewired.
Original unconstrained neuron gains remain in every arm.

The JSON files are specifications, not directly executable configuration files.
See [pipeline commands](../docs/PIPELINE.md) for their CLI mapping. Input data
hashes and exact source/package revisions are explicit. Each run writes its
actual manifest; a changed field or seed gets its own run identity and must not
overwrite an existing result directory.

## Accepted encoder stops

A128fixed was stopped at the user's request after diminishing validation returns:
10 complete epochs, 12,442 observed updates and a latest durable checkpoint at
12,400. Its independent winners are CE4.728279 at3,600 and accuracy32.8975% at
12,200. See the [stop report](reports/A128fixed-accepted-early-stop.md).
B32fixed also stopped under the standing plateau instruction at16,262 observed /
16,200 durable updates after13 complete epochs. Its minimum-CE checkpoint is
4.512509 at9,700; its maximum-accuracy checkpoint is33.313523% at15,200.
The [completed post-B assessment](reports/encoder32-post-B-assessment.md) supports
retaining width32. Bound model and trainer sources remain unchanged.
Actual training budgets now differ. Report the shared validation-update comparison
alongside retained-best scores, and do not claim four completed44-epoch runs.

## Decoder reduction

The user selected the [rank128 readout pair](DECODER_REDUCTION.md) next. E keeps
fixed base edges; F adds18,322 bounded source/destination type gains. Both keep
width32 and eight explicit delays, with a6,453,376-parameter decoder. Rank128
preflight, two serial eight-update smokes, and full E then F training all run on
macm3. C/D full-head runs are deferred; their paused dispatcher was retired without
resuming its queue. Use the new campaign's receipts for actual running status.

Refresh partial validation with `scripts/refresh_connectorch_decoder_progress.py`.
It writes `results/connectorch-decoder-v1/progress.md` and structured snapshots,
showing preserved B alongside E/F and both independent checkpoint selectors.

## Branch and artifact workflow

Keep `main` as the common tested pipeline and protocol. Create each experiment
branch from the same recorded baseline commit, with the named configuration and
its own report. If an arm changes implementation, commit that change only to its
branch and state how it affects comparability. Do not switch branches or edit
source in a checkout used by a live worker. Use a separate checkout/worktree for
each version; execute GPU workers serially on macm3.

For each run, add a small `experiments/runs/<run-id>.json` and
`experiments/reports/<run-id>.md` to its experiment branch. Record:

- experiment ID, branch, source commit, configuration path and hash;
- macm3 host, environment, launch time, exact argv, run/artifact location;
- model, dataset and anatomical-group hashes;
- completed epochs/updates and actual stop/completion reason;
- both selected checkpoints: SHA256, cursor, validation CE/accuracy and criterion;
- independent evaluation population/protocol and metrics, if performed;
- graph/gradient checks, gain displacement, measured timing/memory limitations.

Do not commit model weights, optimizer/RNG states, downloaded datasets, virtual
environments, credentials or cache directories. A checkpoint path is a locator,
not an identity: retain its SHA256 and relevant source revision. Do not label
best historical logged accuracy as a retained checkpoint unless the corresponding
weights exist. Commit compact result summaries by selecting their exact paths;
`results/` remains ignored so a broad add cannot upload training artifacts.

## Interpretation

Assess `(B−A)` versus `(D−C)` for CE and accuracy, with their directions stated.
Retain full validation curves to see overfitting and both selection criteria.
Do not choose a checkpoint or architecture from the reserved test set. A reduced
compression penalty motivates paired-seed replication and matched graph controls;
it does not alone show a biological language prior.

Later readout-rank, explicit-history and randomized-graph experiments use new
`exp/<question>-<condition>` branches and new configurations. They should not be
folded silently into these four matched arms.

## Rank128 handoff status

E32rank128fixed stopped after 13 completed epochs under the plateau rule, at
15,918 observed / 15,900 durable updates. Both winners remain preserved: CE3.149704
at13,700 and accuracy36.5822% at14,300. F32rank128bounded launched separately
from scratch through the receipt-verified F-only handoff. See
[the E stop report](reports/E32rank128fixed-accepted-early-stop.md). The old
dispatcher's native signal failure is intentional history; the new continuation
manifest and process state govern that historical F execution.

## Rank64 order correction

The user clarified that rank64 should already be training. The automatic F
continuation was the wrong next experiment. It stopped at22:04:28UTC on2026-09-15
after2,630 observed /2,600 durable updates and2completedepochs. Both winners and
latest are preserved, with exact process cessation and source/checkpoint audits.
This is an explicit correction of ordering, not a plateau or full completion.
G32rank64fixed launched after its own parity and smoke gates;
F completion or generated-text evaluation was not a prerequisite. E is the
preserved rank128 comparator. No automatic F restart is selected.

Refresh G and E comparison with `scripts/refresh_connectorch_rank64_progress.py`;
its output is `results/connectorch-rank64-v1/progress.md`.

G full training launched at22:09:21UTC after its native M3 parity and eight-update
smoke passed; more than100full updates were verified. See the
[rank64 launch report](reports/G32rank64fixed-launch.md) and
[launch receipt](runs/rank64-v1-launch.json).

The [13,000-update comparison](reports/G32rank64fixed-progress.md) is historical:
renewed accuracy gains correctly deferred stopping at that point. G later met
both plateau thresholds and stopped at **23:37:28 UTC on 2026-09-15**, after
**14 completed epochs, 16,808 observed / 16,800 durable updates**. Its preserved
winners are **CE 3.093705 at 16,600** and **accuracy 36.8885% at 15,552**.
The [accepted-stop report](reports/G32rank64fixed-accepted-early-stop.md) and
[native receipt](records/G32rank64fixed-accepted-early-stop.json) govern its
terminal status; native failed/−15 records describe intentional SIGTERM cleanup.

The subsequent idle-macm3 evaluation completed. See the
[post-G assessment](reports/rank64-post-G-assessment.md),
[matched generated texts](reports/rank64-generated-texts.md), and
[text review](reports/rank64-text-review.md). E/G total training budgets differ;
both selectors and common-budget comparisons remain explicit. The user
subsequently selected H32rank32fixed as the next experiment.

## Rank32 decoder comparison

H32rank32fixed trained from scratch with the same G recipe and fixed canonical
edges. The encoder remains 482,816 parameters; the rank32 decoder has 1,613,344,
bringing the total to 2,343,125. The original 148,179 neuron gains and biases
remain trainable. G's accepted stop supplies the primary comparison baseline;
rank32 changes head shapes and initial logit variance, not just parameter count.
No additional brain plasticity or reserved-test evaluation is selected.

Native rank32 CPU/MPS parity and a separate eight-update optimizer smoke passed.
Full training launched on macm3 at **00:57:20 UTC on 2026-09-16**, with MPS fallback
disabled. Its first full validation at 100 updates measured CE 5.827075 and
accuracy 8.4484% (1,848/21,874 targets across 100 stories); no quality conclusion
is drawn from this initial measurement. See the
[rank32 launch report](reports/H32rank32fixed-launch.md),
[launch receipt](runs/rank32-v1-launch.json) and
[numerical preflight](records/rank32-preflight.json).

H stopped under the established plateau rule at **02:26:35 UTC on 2026-09-16**:
**14 completed epochs, 17,409 observed / 17,400 durable updates**. Both selectors
retain **16,600: CE 3.137654, accuracy 35.4713%**. See the
[accepted-stop report](reports/H32rank32fixed-accepted-early-stop.md) and
[native receipt](records/H32rank32fixed-accepted-early-stop.json). Native failed/−15
records preserve intentional SIGTERM history; accepted-stop status is authoritative.

The idle-macm3 numerical quality audit completed at **02:36:21 UTC**. All four
G/H full-validation checkpoint replays passed, with no backward passes and no
parameter or graph changes; the reserved test remains untouched. H halves G's
decoder and reduces total trainable parameters by **40.7774%**. Compare the
retained winners and common update/target budgets in the
[post-H assessment](reports/rank32-post-H-assessment.md),
[matched generated texts](reports/rank32-generated-texts.md),
[structured assessment](records/rank32-post-H-assessment.json), and
[full validation curves](reports/figures/rank32-vs-rank64-validation.png).
Actual durable budgets differ: G 16,800 versus H 17,400. No successor is selected
automatically; numerical replay alone does not establish text quality.

`scripts/refresh_connectorch_rank32_progress.py` renders the preserved H/G
receipts under `results/connectorch-rank32-v1/`.
