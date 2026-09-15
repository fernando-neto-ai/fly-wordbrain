# Next stage: smaller language interfaces, minimally adapted connectome

Approved for implementation by the user on 2026-09-15. The native Metal backend,
reference-compatible model variants and independent trainer are implemented.
The four-arm campaign must wait for successful completion of
`ngxson-reconstructed-v1`, then pass its own M3 preflight before starting.
Checkpoint inference and Apple GPU numerical parity are reproduced;
the original full training run is still in progress. The author's exact training split
and several optimizer details are unpublished, so score comparisons must retain
the reconstruction caveat in `NGXSON_REPLICATION.md`.

## Question and parameter accounting

Can a small, bounded change to the measured connectome compensate for a smaller
input encoder, and eventually a smaller output decoder, on held-out language?
Can matched controls distinguish an anatomical advantage from generic sparse
recurrent capacity?

The current input interface is a token embedding followed by eight linear
projections. It explicitly supplies the current token and seven previous tokens.
Its parameter count is `d * (1024 + 8 * 1758) = 15088 * d`.

| Embedding width | Embedding + input projections | Total with original readout |
|---:|---:|---:|
| 128 | 1,931,264 | 52,756,661 |
| 64 | 965,632 | 51,791,029 |
| 32 | 482,816 | 51,308,213 |

The bias-free output matrix alone has 50,578,432 parameters (95.87% of the
original trainable total). Width 32 cuts the input encoder by 75%, but the whole
model by only 2.75%. Counts measure storage and learned capacity, not causal
contribution to accuracy; the large readout is linear, while the input interacts
with nonlinear recurrent dynamics. Both interfaces warrant separate experiments.

## 1. Port the exact computation before changing the model

Pin Connectorch to the reviewed revision
`4bbfb645099aeb85bdbf850e1a87cc094769af87` initially. Build a separate integration
that preserves the 49,393 neurons, 9,050,172 stored edges, exact signed values
including zeros, input/output indices, delayed input and recurrent update.
Connectorch's loader, normalization and transmitter defaults differ from ngxson;
using them unchanged would silently alter the baseline.

Reuse the tested Metal CSR forward and transpose state-gradient work. Add dynamic
edge values, a canonical-to-transpose edge mapping, and gradients for shared
gains. For `Y = W H`, an edge derivative is
`dW[e] = sum_b dY[target[e], b] * H[source[e], b]`.
Reduce these derivatives into gain groups without retaining an edges-by-batch
message tensor per time step. Shared parameters alone do not solve activation
memory use. Initially support native FP32 MPS and first-order training.

Acceptance: identity gains reproduce the fixed backend; independently check
forward values, input and gain derivatives, finite-difference CPU references on
small graphs, zero/sign preservation, transpose mapping, padding semantics,
optimizer updates and checkpoint resumption. Then benchmark full-model forward,
backward and optimizer time with MPS synchronization and actual peak memory.
Do not assume an acceleration before measuring it. Run this work separately
from the active reference environment and source files.

## 2. Isolate encoder compression and modest synaptic adaptation

Use a small factorial comparison, initially with width 32 as the compressed
condition. Width 64 is a fallback diagnostic if the larger reduction fails.

| Arm | Input width | Base-edge adaptation |
|---|---:|---|
| A | 128 | Fixed |
| B | 32 | Fixed |
| C | 128 | Bounded shared gains |
| D | 32 | Bounded shared gains |

Keep all eight input groups, all 14,064 injected neurons, full readout, tokenizer,
data ordering, optimizer budget, state updates and validation procedure matched.
Use the original input initialization scale proportional to `1/sqrt(d)`.
Train primary arms from scratch with paired seeds. A compressed checkpoint
fine-tuning study would be a separate transfer experiment with its own matched
controls, not a substitute for this comparison.

The measured graph has 8,156 named cell types and 1,005 untyped neurons. Giving
each untyped neuron its own group yields 9,161 node groups and 2,625,911 connected
type pairs. A free parameter for every pair would exceed the encoder savings.
The first implementation therefore factorizes pair gains into source-type and
destination-type vectors, for **18,322** new trainable values:
`W[e] = W0[e] * (1 + 0.1 * tanh(theta_source[type(src)] + theta_destination[type(dst)]))`.
Both vectors start at zero. This is less expressive than independent pair gains.
Retain
zero-valued edges as zero and preserve every endpoint and nonzero edge sign.
Begin with at most a 10% base-weight change; consider a wider bound only after
evaluating this range. This changes synaptic strengths without changing topology.

The original per-neuron `gain` and `rec_gain` are unconstrained and can change
effective recurrent strength or polarity. Bounding new edge multipliers alone
does not bound the whole recurrence. Use the same neuron-parameter policy in
every arm; if imposing positive/bounded neuron gains, first establish a matched
full-width baseline under that policy. Report differences from the exact
reference separately. Neither fixed topology nor a small parameter count proves
that the effective geometry stayed close to its starting point.

Measure validation CE/accuracy, paired per-story losses, train-validation gap,
training time, peak memory, group/edge gain displacement, fraction of edges
changed beyond specified tolerances, incoming-strength drift, and neuron gains.
For the input question, compare how much compression hurts with and without
plasticity, not just whether D beats B. Repeat any promising comparison across
paired seeds before drawing a geometry conclusion.

## 3. Reduce external memory and the output head as separate axes

After the input experiment, test explicit history lengths 8, 4 and 1 while
retaining all eight projection groups, the same input-neuron coverage and the
same parameter count. Change which token lag each group receives; do not simply
set `delay_k=1`, which would also change the interface. Train each condition.
Width compression alone cannot establish that the reservoir learned to remember
the previous seven tokens. Even a current-token-only model has individual leaky
states, so recurrent inter-neuron memory still needs a graph control.

Separately factorize the readout as two bias-free linear matrices, keeping the
LayerNorm and access to all neurons. A rank-64 head has
`64 * (49393 + 1024) = 3,226,688` parameters instead of 50,578,432. Rank 128 is
6,453,376; rank 32 is 1,613,344. This avoids confounding a smaller head with
discarding most output neurons. Combining width 32 and rank 64 would yield
3,956,469 trainable parameters before adding the 18,322 shared edge gains, assuming the
original 148,179 neuron and 98,786 LayerNorm parameters. These are proposed
counts, not validated quality results.

## 4. Test anatomy and generalization

For promising compressed configurations, train matched randomized graph controls
with identical interfaces and plasticity budgets. Document exactly what each
randomization preserves: degree-preserving swaps do not automatically preserve
each target's incoming strength or inhibitory/excitatory composition. A separate
trained zero-edge control retains the input delay, individual leaky states and
readout, and measures what those components can achieve without inter-neuron
connections. Inference-time removal alone measures disruption to a trained model.

Keep validation for model selection and use final testing according to a declared
protocol; do not turn the current reference's test set into a development target.
Confirm promising findings on more training stories and an independent evaluation
split, matched across arms. Success means retaining useful held-out performance
with smaller interfaces and measured small graph changes, followed by evidence
of benefit over matched non-anatomical controls. Fluency alone does not establish
that the biological wiring supplies a special language prior.

Sources: the [pinned reference](https://huggingface.co/ngxson/fly-llm-hf/tree/65c677b3d566a2e9793d5f72999cdb441c6c0a9f),
[Connectorch parameterization](https://github.com/us/connectorch/blob/4bbfb645099aeb85bdbf850e1a87cc094769af87/src/connectorch/nn/parameterisation.py),
and the local verified reports `NGXSON_REPLICATION.md` and
`results/connectorch-review/knowledge.md`.

## Implementation and verified scope

`fly_wordbrain/connectorch_backend.py` explicitly registers `metal_csr` with
ConnecTorch. Stock `ct.nn.ConnectomeRNN` can use the new propagator; the language
model uses the same registered propagator with the exact ngxson neuron update.
`metal_sparse_trainable.py` supplies CSR forward, transpose state backward and
edge-dot backward. It retains O(E) edge values/gradients, without an E-by-batch
message tensor per time step. Values and type gains are computed once per chunk.
MPS is FP32 and first-order only; CPU is an explicit independent execution mode.
ConnecTorch's stock memory-warning estimator still assumes its scatter backend;
it is not an actual memory measurement for this extension.

`prepare_connectorch_groups.py` verifies the pinned 688.8 MB Doomfly export and
the pinned HF checkpoint before mapping sorted biological IDs. Every one of the
9,050,172 checkpoint edges matches the corresponding source endpoints, signs and
globally scaled magnitudes (maximum absolute discrepancy 2.98e-8). The unfiltered
five-class induced source graph has 9,679,074 edges: 628,902 source edges are not
in the checkpoint. Their filtering reason is not established by this audit and
they are never restored. The exported archive binds all five original graph and
interface buffer hashes; its SHA256 is
`a07d311cd04126b5e54ea9cb3e8945470af9049bcb8499fb406e3c38d185dd7f`.

Independent local checks cover native sparse gradients, actual ConnecTorch
recurrence and biological weights, exact upstream language-model parity,
padding/cache behavior, cross-backend checkpoint loading, and two-phase trainer
resumption. A production full-graph MPS smoke at width32/bounded10 completed
eight updates: both type-gain vectors received nonzero finite gradients and
changed, all base buffers stayed fixed, and no test evaluation occurred.
This establishes working training, not a compression benefit.

The first campaign fixes the large readout and all eight explicit history slots.
Widths128/32 crossed with fixed/bounded10 edges give 52,756,661 / 51,308,213 /
52,774,983 / 51,326,535 trainable parameters respectively. All arms use the same
1,000 training and 100 validation stories, 30+14 epochs, B8/TBPTT32 and seed42.
The campaign defers final testing and reports compression penalties plus their
interaction. Rank/history changes and seed replication follow assessment of
these four arms; they do not begin automatically as an unbounded sweep.
