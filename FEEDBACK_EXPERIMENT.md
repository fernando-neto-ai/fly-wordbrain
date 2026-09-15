# Seven observations, one prediction

This experiment learns how delayed observations change temporary synaptic state
inside the complete Doomfly graph. It keeps 166,700 neurons, 25,582,938 directed
edges, the original endpoints, base weights, and transmitter signs. Smooth leaky
rate dynamics and the candidate sensory interface are the same as the action
selector, not Doomfly's original spiking physiology.

The first full-graph preflight exposed GPU roundoff variation in colliding
output-pooling sums: neuron states repeated exactly, but pooled features did
not. This model uses a deterministic padded gather and sum for the same fixed
1,314-neuron signed projection. It preserves the projection and adds no learned
parameters. The failed preflight and its numerical audit remain archived.

Each example is eight consecutive lexical words. Starting with zero activity,
zero temporary weights, zero eligibility, and BOS/BOS count context:

1. Propose ten candidates using only the previously observed words.
2. Run the brain with those candidates and the last two observed word IDs.
3. Reveal the actual word. Form an eleven-way prediction error (ten ranks plus
   OTHER), a fixed unique 16-dimensional word code, and its position in 1–7.
4. Use this feedback to modulate prediction-time eligibility on the 347 selected
   existing edges. Repeat for the seven known words.
5. Produce and score word-eight logits before revealing word eight. Reset the
   entire episode state for the next window.

No label is injected into a candidate set. IDs are categorical lookup keys;
their numerical magnitudes have no linguistic meaning. Rank refers to each
proposal's current ordering. OTHER preserves all probability outside top ten.

The shared update rule has **136 learned parameters**: the original twenty
four-group write/retention/gate parameters, four eligibility-retention values,
and a 4×28 feedback projection. The action head has **2,570 parameters**, for
**2,706 total**. Each window additionally holds 347 temporary edge values and
347 eligibility values, which are runtime state rather than learned independent
parameters. The fixed observed-word code has 16,384 values for vocabulary 1,024;
the existing retinal encoder remains fixed.

Effective selected weights are base_weight × exp(log(2) × tanh(fast_state)),
so gains remain between 0.5 and 2 without adding edges or reversing signs. The
head adds its ten corrections to the count model's log probabilities. Its zero
initialization exactly recovers the count model. The first optimizer step
therefore trains the head; gradients reach the rule on subsequent steps.

## Data and scoring

The existing 8,192 training stories yield **126,600 nonoverlapping windows**.
These contain 886,200 observed historical words and 126,600 final training
labels. Incomplete tails (6,614 words) are dropped. Windows never cross story
boundaries. The count model is fitted on all training stories and excludes the
complete current story when generating training/calibration proposals.
Validation proposals use only training counts. No test split is evaluated.

Partial validation uses all **1,968 windows** from the first 128 validation
stories. Full validation uses **15,824 windows** from all 1,024 validation
stories. Calibration uses the first 256 training windows and whole-story
exclusion. It estimates per-edge activity RMS and final-feature normalization
with fast writing/reading disabled.

The loss is conditional cross entropy over top-ten candidates only when word
eight is present. Overall accuracy includes every window, so missing targets
remain errors. We also report coverage, conditional accuracy/loss, known-word
accuracy, corrected count-model mistakes, introduced mistakes, and overrides.

At each validation snapshot a paired ablation disables temporary weight reads
and writes using the **same trained head**. Its accuracy and loss show whether
the current model uses temporary state. It is not a separately trained frozen
control and cannot by itself demonstrate a benefit from biological geometry.
A geometry claim will require matched independently trained controls and seeds.

Only full-validation accuracy selects checkpoints, with the initial count
baseline eligible and ties keeping the earlier checkpoint. Partial validation
does not select checkpoints. These window-level results are a different
population from the older all-position action-selector curves.

## Launch and inspect

Run with the existing PyTorch 2.8 environment and custom Metal sparse backend:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u -m fly_wordbrain.feedback_train \
  --dataset data/expanded-8k/dataset.json --graph data/plastic-graph \
  --output results/feedback-8word-top10-e1 --device mps \
  --batch-size 16 --epochs 1 --lr .001 --seed 0 \
  --monitor-stories 128 --monitor-every 128 --calibration-windows 256
```

Before training, a mandatory train-only probe checks real full-graph forward,
backward, and optimizer updates; hidden-target independence; graph hashes and
weight bounds; feedback effects above replay noise; and nonzero rule gradients
after a head warmup. Probe parameter updates are restored before step zero.
A failed probe prevents training. `--probe-only` stops after this check.

The dashboard at port 8770 provides partial validation curves and CSV export:

```bash
python -m fly_wordbrain.feedback_dashboard \
  --run results/feedback-8word-top10-e1 --port 8770 \
  --remote-host macm3 \
  --remote-run /Users/fernando/fly_wordbrain/results/feedback-8word-top10-e1
```

Use a dedicated empty local cache when attaching a remote run. Browser requests
refresh the cache at most every fifteen seconds. Protocol, source/data/graph
hashes, calibration, preflight, process status, progress and validation histories
are saved per run. Compact checkpoints save learned parameters and calibration,
not copies of the full graph. They support inference, not optimizer resumption.
