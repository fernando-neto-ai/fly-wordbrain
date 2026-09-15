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

## Accepted stop and active continuation

A128fixed was stopped at the user's request after diminishing validation returns:
10 complete epochs, 12,442 observed updates and a latest durable checkpoint at
12,400. Its independent winners are CE4.728279 at3,600 and accuracy32.8975% at
12,200. See the [stop report](reports/A128fixed-accepted-early-stop.md).
B32fixed is running through the separate continuation controller; C128bounded
and D32bounded follow. The bound model and trainer sources are unchanged.
Actual training budgets now differ. Report the shared validation-update comparison
alongside retained-best scores, and do not claim four completed44-epoch runs.

## Immediate decision after B

B32fixed continues training, while C/D are held for the user's requested
[post-B plasticity assessment](POST_B_REVIEW.md). The goal is to retain the
reduced encoder and determine the smallest additional brain adaptation needed
if quality degrades. The first existing candidate adds18,322 bounded shared
edge gains; finer adaptation remains an untested proposal.

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
