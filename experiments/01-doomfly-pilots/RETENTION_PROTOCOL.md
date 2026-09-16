# Train delayed memory through existing fast weights

This experiment starts from the immutable step-3,072 checkpoint used by the
frozen memory probes. It tests whether a final-state memory objective can make
earlier words accessible through the existing fast-weight path. The separate
language-training run continues unchanged.

The fly has the same 166,700 neurons, 25,582,938 directed base edges, input
interface, rate dynamics, and 8,192 selected existing fast edges in 19 groups.
No connection is added or rewired. The original language head and checkpoint
pre/post activity scales remain frozen. Trainable plasticity consists of 646
shared-rule values and 8,192 per-edge write susceptibilities: **8,838 values**.

Two new linear heads read only the existing 256-dimensional signed descending
pool at prediction eight: ten-way earlier-word identity and two-way earlier
word order. They contain **3,084 parameters** together. The treatment trains the
heads plus plasticity rules (**11,922 trainable parameters**). The control trains
only identical heads against cached responses of the original frozen checkpoint.
Both arms train with temporary writes enabled, on the same MPS device, with the
same head initialization, examples, ordering, normalization, and head optimizer.

The 1,344 synthetic eight-word episodes use 224 fresh original-training stories,
excluding all 96 sources and full nuisance contexts from the previous probe.
There are 64 training, 16 validation, and 32 test groups per task. Every group
contains all labels while holding final words six/seven and their proposals
fixed. Identity changes observed word three; order swaps words two/three. The
dummy eighth word is constant, and memory labels never enter the neural input.
All proposals use **all original training stories minus the entire source
story**, including for the new validation/test groups.

Training uses 256 updates with eight identity and eight order rows per batch.
Each 80-update data cycle covers all 640 identity training rows once and all
128 order rows five times. The loss is the mean of the two tasks' cross entropies
at prediction eight. There is no earlier-position auxiliary loss or external
store of intermediate neural responses. Episode activity, eligibility, and fast
weights reset before every window; the seven observations are causal.

Head Adam uses lr .003 and epsilon 1e-8; rule Adam uses lr .003 and epsilon 1e-12.
The head and rule gradient norms are clipped separately at one. Mean/std buffers
come only from initial frozen TRAIN responses, with std floored at 1e-12 and no
feature mask. Both arms keep those buffers fixed. The full MPS preflight takes
two optimizer steps, checks real rule gradients/changes, and restores parameters
and fresh optimizers before experimental update one.

Validation is reported at updates 0, 64, 128, 192, and 256. Each arm independently
selects its lowest validation mean final CE; ties keep the earlier step. Only
then are new test features extracted and scored. Test comparisons include normal
fast-on evaluation, fast-off evaluation, and clearing only H after observation
seven immediately before prediction eight. H clearing preserves activity and
eligibility, so it tests the final fast read; an earlier fast contribution can
remain in activity even if the final erasure does not hurt.

Run with a distinct output directory:

```sh
PYTHONPATH=. .venv/bin/python scripts/train_retention.py \
  --checkpoint results/memory-probe-20260915/frozen.pt \
  --previous-rows results/memory-probe-20260915/rows.json \
  --dataset data/expanded-8k/dataset.json \
  --graph data/plastic-graph/graph.npz \
  --structure data/fast-structures/anatomical-8192.npz \
  --output results/retention-step3072-v1 --device mps \
  --updates 256 --validate-every 64

PYTHONPATH=. .venv/bin/python scripts/report_retention.py \
  --output results/retention-step3072-v1
```

The result addresses a bounded synthetic memory task with one seed and starting
checkpoint. It does not measure next-word accuracy, broad language learning,
biological plausibility, or any benefit over a matched randomized graph. New
test contexts are held out from retention fitting and the prior probe, but they
belong to the original language-training corpus.
