# Frozen context accessibility probe

This diagnostic asks whether the existing word interface and fly activity carry
an earlier word's identity or the order of two earlier words. It uses a frozen
copy of a feedback checkpoint. No original graph edge, base weight, trained
fast-weight rule, or live training process is changed.

The default dataset has 576 synthetic interventions from 96 original training
stories. Identity examples replace observed word 3 with one of ten frequent
training words. Order examples swap words 2 and 3 using one fixed pair. Every
complete nuisance group stays within one probe split, and its final words 6/7
and ten candidate proposals are identical. The dummy eighth word is constant;
the identity/order labels are separate from the model inputs.

The source stories are held out from probe fitting, not necessarily from the
original brain training. All proposal counts use original training data with
the entire source story subtracted. Original validation/test stories are unused.

The readout comparisons each use 256 features:

- The existing final signed pool of descending neurons.
- Final activity of 256 individual neurons, selected by activity variance on
  96 probe-training examples, excluding directly driven retinal cells.
- Four response snapshots from the first 64 selected neurons, at prediction
  times 4, 5, 6, and 8. This adds external temporal storage and cannot establish
  that the *final neural state* retains the earlier context.
- The existing pooled response at prediction time 4, as an earlier-path check.
- Actual fixed retinal-code subsets for earlier words (positive control), or
  the final previous/current words (negative control).

Slow parameters remain frozen during fast-on and fast-off extraction. Fast-on
still performs the seven causal temporary-weight updates within each episode.
All states reset between episodes. Linear and 16-unit ReLU heads are fitted on
CPU using cached features; train-only standardization and validation CE select
the parameters before test scoring. The ten-class heads have 2,570 and 4,282
parameters respectively. Two-class heads have 514 and 4,146 parameters.

Run from the repository root, using a separate output directory and a copied
inference checkpoint (for example, `partial.pt`, not a mutable live path):

```sh
PYTHONPATH=. .venv/bin/python scripts/probe_frozen_memory.py extract \
  --checkpoint results/memory-probe/frozen.pt \
  --output results/memory-probe \
  --dataset data/expanded-8k/dataset.json \
  --graph data/plastic-graph/graph.npz \
  --structure data/fast-structures/anatomical-8192.npz --device mps

PYTHONPATH=. .venv/bin/python scripts/probe_frozen_memory.py fit \
  --output results/memory-probe

PYTHONPATH=. .venv/bin/python scripts/summarize_memory_probe.py \
  --output results/memory-probe
```

If the primary run masks small responses despite exact replay, preserve it and
run the explicitly exploratory numerical-scale check on the same cache:

```sh
PYTHONPATH=. .venv/bin/python scripts/check_memory_probe_scale.py \
  --output results/memory-probe
PYTHONPATH=. .venv/bin/python scripts/summarize_memory_probe.py \
  --output results/memory-probe
PYTHONPATH=. .venv/bin/python scripts/plot_memory_probe.py \
  --output results/memory-probe
```

The summarizer keeps primary and exploratory results separate. Changing a
threshold after inspecting test results does not create a fresh test set.

The extraction receipt checks checkpoint/source/graph identity, exact restored
parameters and calibration, final-input equality within groups, original-model
forward parity, replay, eighth-target independence, and unchanged frozen
parameters/base graph. The saved protocol fixes neuron selection and feature
arms before extraction and test scoring. `features.npz` permits additional CPU
analyses without further neural simulation.

This is a small, single-checkpoint diagnostic of accessible information under
these readouts. It does not measure next-word accuracy, novel-word learning,
biological plausibility, or a benefit from the connectome's particular geometry.

The first completed probe, `results/memory-probe-20260915`, used checkpoint
step 3,072 (SHA256 `6527999333896856d0b8c0265ed02424385340f1c829ab12d20236d7bdb67797`).
After the exploratory scale check, early pooled identity/order scored 62.5%/100%
with linear heads, versus 10%/50% chance. Final pooled, selected-cell and selected
temporal readouts remained near chance. Exact raw responses were distinct, so
this is a failure to demonstrate reliable final decoding, not proof of absent
information. See that run's `interpretation.md`, `report.md`, and independent
cache audit for controls and limitations. No live training configuration changed.
