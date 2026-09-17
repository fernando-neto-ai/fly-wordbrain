# Experiment registry

Repository: `fernando-neto-ai/fly-wordbrain`. **All real training runs on macm3**, including
smoke tests, executed serially. This file is the map and the rules; the machine-readable
receipts under `records/` and `runs/` are authoritative for what actually executed.

## Layout

Narrative is organised by research stage. Provenance stays flat and stable, because these
paths are cited by hash in run receipts, tests and external notes.

| Path | Contents |
|---|---|
| [`01-doomfly-pilots/`](01-doomfly-pilots/README.md) | Frozen 166,700-neuron spiking brain, plasticity, action selection, feedback, retention — all negative |
| [`02-ngxson-reference/`](02-ngxson-reference/README.md) | Exact reproduction of the 49,393-neuron Fly LLM and its quality gap |
| [`03-encoder-compression/`](03-encoder-compression/README.md) | A/B/C/D — how wide the input interface needs to be |
| [`04-decoder-compression/`](04-decoder-compression/README.md) | E/F/G/H — replacing the 50.6M-parameter readout |
| [`05-shared-population/`](05-shared-population/README.md) | Every model scored on one population, with paired confidence intervals |
| [`06-corpus-size/`](06-corpus-size/README.md) | I/J — 2×2 over readout and corpus size; tests whether Stage 4 was a small-data artifact |
| [`07-graph-control/`](07-graph-control/README.md) | K — the same model on a randomly rewired graph; tests whether the fly's wiring matters at all |
| `configs/` | Declarative experiment specifications. **Specifications, not runnable config files** — see [pipeline commands](../docs/PIPELINE.md) |
| `records/` | Selector hashes, parity, source ancestry, stop receipts, quality results |
| `runs/` | Launch receipts |

## Arms

| Experiment | Branch | Configuration | Trainable | Status |
|---|---|---|---:|---|
| Original reconstruction | `exp/ngxson-reference` | [reference](configs/ngxson-reference.json) | 52,756,661 | accepted early stop |
| A128fixed | `exp/encoder128-fixed` | [A](configs/A128fixed.json) | 52,756,661 | accepted early stop |
| B32fixed | `exp/encoder32-fixed` | [B](configs/B32fixed.json) | 51,308,213 | accepted early stop |
| C128bounded | `exp/encoder128-bounded` | [C](configs/C128bounded.json) | 52,774,983 | deferred |
| D32bounded | `exp/encoder32-bounded` | [D](configs/D32bounded.json) | 51,326,535 | deferred |
| E32rank128fixed | `exp/encoder32-readout128-fixed` | [E](configs/E32rank128fixed.json) | 7,183,157 | accepted early stop |
| F32rank128bounded | `exp/encoder32-readout128-bounded` | [F](configs/F32rank128bounded.json) | 7,201,479 | stopped for ordering; **do not resume** |
| **G32rank64fixed** | `exp/encoder32-readout64-fixed` | [G](configs/G32rank64fixed.json) | **3,956,469** | accepted early stop — **recommended** |
| H32rank32fixed | `exp/encoder32-readout32-fixed` | [H](configs/H32rank32fixed.json) | 2,343,125 | accepted early stop |
| **I32rank64fixed10k** | `exp/corpus10k-rank64` | [I](configs/I32rank64fixed10k.json) | **3,956,469** | budget-capped at 16,800 updates — **best audit score** |
| J32fullfixed10k | `exp/corpus10k-fullreadout` | [J](configs/J32fullfixed10k.json) | 51,308,213 | budget-capped at 16,800 updates; Stage 6 control |
| K32rank64shuffled10k | `exp/graph-control-shuffle` | [K](configs/K32rank64shuffled10k.json) | 3,956,469 | randomised graph, degrees preserved; Stage 7 control for I |

The two 10k arms are **budget-capped**, not plateau-stopped: they were given exactly 16,800
updates to match G's compute, so the trainer records them as `debug_stopped` with
`debug: true`. That flags the cap, not a failure. They completed 1.83 passes over their
corpus and are **not converged**; read [Stage 6](06-corpus-size/README.md) before quoting
their numbers.

An accepted early stop is **not** a completed run. The planned schedule was 44 epochs
(52,609 updates); no arm reached it. Operational plateau stopping required at least eight
completed epochs with less than 0.10 best-CE and 0.5 accuracy-point improvement over the
trailing 2,000 updates. That is a practical rule, not proof that a later learning-rate
phase could not improve a model.

## Recording rules

For each run, add `runs/<run-id>.json` and a report in the owning stage directory,
recording: experiment ID, branch, source commit, configuration path and hash; macm3 host,
environment, launch time, exact argv, artifact location; model, dataset and anatomical-group
hashes; completed epochs/updates and the actual stop reason; **both** selected checkpoints
with SHA256, cursor, validation CE/accuracy and criterion; independent evaluation protocol
and metrics if performed; graph/gradient checks, gain displacement, measured limits.

- Keep `main` as the common tested pipeline. Branch each experiment from the same recorded
  baseline commit. If an arm changes implementation, commit that change only to its branch
  and state how it affects comparability.
- Never switch branches or edit source in a checkout bound to a live worker. Use a separate
  checkout per version.
- **Retain both the lowest-CE and the highest-accuracy weights.** Two independent selectors
  are never one checkpoint. A logged peak is not a retained checkpoint unless the weights
  exist.
- A checkpoint path is a locator, not an identity. Keep its SHA256 and source revision.
- Do not commit weights, optimizer/RNG state, datasets, environments or caches. `results/`
  is ignored so a broad `git add` cannot upload training artifacts; commit compact summaries
  by selecting exact paths.

## Interpretation rules

Do not choose a checkpoint or an architecture from the reserved test set; it remains
unopened. Report the shared-budget comparison alongside retained-best scores — actual
budgets differ between arms. Numerical replay alone does not establish text quality, and
compression alone does not establish an anatomical prior: that requires trained randomized-
graph and zero-edge controls, multiple seeds and a declared final-test protocol. The
randomised-graph control is [Stage 7](07-graph-control/README.md) and it came back near
zero — 0.0100 nats — so an anatomical claim now has to clear that bar, not merely exist.

New questions get new `exp/<question>-<condition>` branches and new configurations. They
must not be folded silently into an existing matched set.
