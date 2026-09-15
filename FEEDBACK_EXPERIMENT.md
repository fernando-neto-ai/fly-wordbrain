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
4. Use this feedback to modulate prediction-time eligibility on the selected
   existing edges (347 originally; 8,192 in the expanded experiment). Repeat for the seven known words.
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

## Expanded anatomical selection

The original 347-edge run produced tiny fast/on-off logit effects and identical
choices. Its partial checkpoint is preserved at step 640. The expanded
experiment selects **8,192 existing edges across 19 groups**, retaining the
original 347 and their four groups as an exact prefix. Added groups cover
mushroom-body, central-complex, visual, central-brain, and descending pathways.
Selection uses the graph's anatomy labels, signed weights, and distances to
sensory/output neurons; it never uses text labels or validation scores. These
are engineering candidates, not claims that all selected edges are biologically
plastic.

The selection is a separately hashed NPZ/JSON sidecar. Canonical neuron IDs,
endpoints, base weights, and signs are verified before and after loading it.
Nothing is rewired. The selection-inclusive fingerprint changes; the canonical
base-graph fingerprint does not. The small selected-edge correction uses a
fixed sparse incidence reduction and `expm1` to preserve tiny corrections
without colliding GPU sums.

With per-edge susceptibility enabled, the model learns **646 shared rule
parameters + 8,192 write multipliers + 2,570 head parameters = 11,408 total**.
Each learned write multiplier is `2*sigmoid(parameter)`, initialized at one.
It controls how historical feedback writes temporary state; it cannot change
a base weight directly. Each window starts with 8,192 zero fast values and
8,192 zero eligibility values, and discards them after word eight.

The rule optimizer uses learning rate .003 and epsilon 1e-12; the head uses
learning rate .0003 and epsilon 1e-8, with separate gradient clipping. This
prevents the head's gradients and Adam epsilon from suppressing small rule
updates. Training always enables fast weights. The fast-off dashboard series
is an evaluation-only comparison of the same trained model.

The expanded preflight additionally measures centered logit changes and total
variation between fast-on/off action distributions after a temporary head
warmup. Those warmup updates are restored before the training run. Probability
changes establish that the mechanism is connected, not that it improves
held-out language prediction. Dashboard snapshots report changed choices,
centered logit effects, and the mean fraction of probability mass moved.

To select and check the expanded configuration:

```bash
.venv/bin/python -m fly_wordbrain.fast_structure \
  --graph data/plastic-graph --target-edges 8192 \
  --output data/fast-structures/anatomical-8192.npz
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u -m fly_wordbrain.feedback_train \
  --dataset data/expanded-8k/dataset.json --graph data/plastic-graph \
  --output results/feedback-anatomical8192-e1 --device mps \
  --structure-path data/fast-structures/anatomical-8192.npz \
  --trainable-susceptibility --lr .0003 --rule-lr .003 --rule-eps 1e-12 \
  --probe-steps 8 --minimum-feature-effect .01 --minimum-action-tv .001
```

A bounded mechanism comparison runs three arms with the same train-only
calibration and 32-update budget: original selection, expanded shared rules,
and expanded rules plus per-edge susceptibility. The two probe sets use
separate training stories excluded from calibration and gradient updates.
Their count proposals still use whole-current-story exclusion. These probes
are not validation results and do not select a checkpoint.

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u scripts/scout_fast_structures.py \
  --dataset data/expanded-8k/dataset.json --graph data/plastic-graph \
  --structure data/fast-structures/anatomical-8192.npz \
  --output results/feedback-anatomical8192-scout-v1.json --shuffled-feedback
```

The explicit script path avoids an unrelated installed `scripts` package on
macm3 shadowing this repository's namespace directory.

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
with fast writing/reading disabled in the original experiment. The expanded
experiment first measures activity scales with writing disabled, then measures
final-feature normalization in a second pass with fast weights enabled. Both
passes use exactly the same train-only windows and whole-story exclusion.

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
