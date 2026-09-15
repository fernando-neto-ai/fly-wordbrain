# Full-connectome training on macm3's GPU

The custom Metal backend runs the complete rate-model training step on macm3's existing PyTorch 2.8.0 installation, with CPU fallback disabled. It keeps all 166,700 neurons, 25,582,938 directed edges, original signs, fixed input codes, and the 526,336-parameter decoder. The plasticity rule adds only 20 learned parameters.

## Measured full-model step

The initial matched measurement used two training stories, 32 observed positions each, eight internal recurrent updates per position, float32, the same seed, and the same calibration procedure. Compilation, graph loading, and calibration occurred before timing. Timings include forward, backward, gradient clipping, and Adam updates.

| Backend | Forward | Backward | Complete update | Observed positions/s |
|---|---:|---:|---:|---:|
| CPU, 4 threads | 31.134 s | 29.398 s | 60.534 s | 1.06 |
| Custom Metal | 0.796 s | 0.628 s | 1.720 s | 37.22 |

This is a **35.2× speedup for this measured training step**, not a claim about total time to convergence. Both runs reported loss 6.982086181640625. Whole-dataset throughput depends on batch size, story length, validation, and other workloads.

The GPU run sampled approximately 1.44 GB of live MPS tensor allocations after the forward pass and 2.34 GB of driver allocations. These are samples, not exact peaks. The full process peak RSS was approximately 2.50 GB. The CPU process peak RSS was approximately 2.57 GB.

A longer GPU run completed **128 observed positions per story, batch 2**, with an unbroken backward pass through 1,024 internal recurrent updates. It took **3.909 s** and sampled **2.49 GB** of live MPS tensor allocations after forward. Observed positions include the initial BOS step; this timing does not demonstrate retention of 128 lexical words.

Raw measurements and source snapshots are stored under `results/plastic-probe-macm3-s0008-t32-b2/` and `results/plastic-probe-mps-s0008-t32-b2/`. The first GPU probe's old nonzero-effect gate is superseded: tiny GPU rounding variation passed it despite a negligible synaptic influence. Its timing and numerical execution remain valid; its calibration must not be used for a learning claim.

## Backend and correctness

`fly_wordbrain/metal_sparse.py` compiles an FP32 Metal shader with PyTorch's `torch.mps.compile_shader`. A 32-lane SIMD group reduces each sparse row. The backward pass uses the explicitly transposed sparse graph. This avoids a dense adjacency matrix, gradients for frozen base weights, and edge-by-batch-by-time intermediates. Canonical graph arrays remain separately fingerprinted; sorted runtime indices preserve every edge and weight.

Select MPS when constructing or loading the model, as the CLI does. Moving an already constructed native sparse CPU model with `.to("mps")` is not the supported conversion path.

Independent tests compare the actual Metal kernel with a CPU float64 dense oracle on small asymmetric graphs, including empty rows, long rows, self-edges, signed/zero weights, batches 1/2/8, and noncontiguous states and gradients. Full small-model tests compare outputs, recurrent states, initial-state gradients, and all 20 plasticity-rule gradients. They prohibit native `torch.sparse.mm` in the Metal path and require custom-kernel launch counters to increase. The full-graph probe also verifies both forward and transpose kernels ran and that training tensors stayed on MPS.

The macm3 backend/model/trainer test selection passed **32 tests** on PyTorch 2.8.0; the complete local suite passed **94 tests**. Only pytest and its small test dependencies were installed in the project's existing virtual environment; PyTorch was not replaced.

## Scientific status

GPU support is operational. Language-learning benefit is still an experimental question. The conservative initial global scale, `.0008`, leaves activity at the chosen synapses around 1e-11 to 1e-12 and yields negligible rule gradients. The CPU intervention had no observable output effect. The corrected calibration now uses a same-input repeat control and requires a fast effect greater than both ten times the repeat difference and a fixed 1e-5 normalized floor.

A training-prefix-only diagnostic examined the declared global scales `.0008, .0012, .002, .004, .008`. The first passing setting was **`.002`**. It changes a single global multiplier, preserving all relative baseline weights, signs, and endpoints. This selection used signal visibility, not language loss. At 128 observed positions, the subsequent full probe measured a normalized fast effect of **0.001015** against **0.000003787** repeat variation. All five rule-parameter tensors received finite gradients and changed after the optimizer update. The complete update took **3.603 s**, with about **2.50 GB** live tensor allocations sampled after forward. Only roughly 0.013% of neuron observations exceeded absolute rate 0.99 during calibration.

This stronger scale is outside the initial estimated linear contraction regime. It must be assessed through the actual nonlinear dynamics; signal visibility alone is not evidence of useful word information or long memory. The observable effect can include common startup and tonic activity.

The passed calibration is available on macm3 at `results/plastic-probe-mps-s002-t128-b2/calibration.npz`. GPU training selects the backend with `--device mps`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m fly_wordbrain.plastic_train \
  --dataset data/pilot/dataset.json --graph data/plastic-graph \
  --calibration results/plastic-probe-mps-s002-t128-b2/calibration.npz \
  --output results/my-new-mps-run --device mps \
  --batch-size 8 --threads 4 --epochs 10
```

Run from `/Users/fernando/fly_wordbrain` on macm3 and choose a fresh output directory. The trainer rejects changed source, graph, dataset, or calibration receipts.

After reproducing the original graph and dataset using the README, regenerate the rate-model export and a fresh calibration with:

```bash
.venv/bin/python scripts/prepare_plastic_graph.py
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m fly_wordbrain.plastic_probe \
  --graph data/plastic-graph --dataset data/pilot/dataset.json \
  --output results/my-new-calibration --device mps --global-scale .002 \
  --internal-steps 8 --leak .5 --threads 4 --benchmark-words 128
```

Require the probe to complete successfully and use its `calibration.npz` in the training command. A fresh corpus or changed dynamics requires a fresh calibration assessment.

The neuron physiology remains the explicitly declared smooth rate model, not Doomfly's original spiking simulation. No evidence of 128-word memory or a special language advantage from connectome geometry follows from a successful hardware benchmark.

## Complete-corpus pipeline check

The short integration run uses the existing 128 training / 32 validation / 32 test stories, one epoch per arm, batch size 8, and scale `.002`. Each epoch makes 16 optimizer updates. All model selections are locked before test evaluation.

| Arm | Epoch including validation | Validation next-word CE |
|---|---:|---:|
| Frozen rate circuit | 139.74 s | 5.91148047 |
| Fixed fast-plasticity rule | 147.29 s | 5.91147284 |
| Learned fast-plasticity rule | 269.20 s | 5.91147157 |

All **20 internal rule scalars changed** in the learned arm; frozen/fixed rule scalars remained identical. A fresh CPU process loaded the GPU-trained checkpoint with strict source/graph identity checks and produced finite outputs through the full graph. The selected checkpoint is only **2,152,898 bytes** because the fixed graph is stored separately.

The near-identical validation losses triggered a wiring audit. All source/data/calibration identities and every actual causal row/mask matched, the arm switches executed correctly, the checkpoint readouts differed, and internal gradients and changes were verified. This establishes functioning training mechanics. One epoch here shows no substantive plasticity advantage and is not a completed geometry or memory-capacity experiment.

The entire run, including all three training passes, selected-checkpoint evaluations, and midpoint interventions, completed in **1,202.54 seconds (20.0 minutes)**. The graph fingerprint remained unchanged.

| Arm | Test next-word CE | Test next-word perplexity |
|---|---:|---:|
| Frozen rate circuit | 5.94176649 | 380.6067 |
| Fixed fast-plasticity rule | 5.94175946 | 380.6040 |
| Learned fast-plasticity rule | 5.94175825 | 380.6035 |

Resetting fast state changed learned-arm second-half next-word CE by only **+0.00000228**. Resetting neural activity increased it by about **+5.45** in both the frozen and learned arms. This shared reset penalty cannot be attributed to learned fast memory; the intervention also disrupts the normal neural dynamics and calibrated readout. These results provide no substantive forecasting benefit at this one-epoch setting.

Complete checkpoints, per-forecast losses, and audit receipts are available under `results/plastic-mps-smoke-3arms-e1/` on both hosts. Its selected learned checkpoint is `learned_fast/best.pt`; `checkpoint-reload.json` records the fresh CPU reload and verification that all 20 rule scalars changed.

## References

- [PyTorch 2.8 custom Metal shader API](https://docs.pytorch.org/docs/2.8/generated/torch.mps.compile_shader.html)
- [Experiment protocol](PLASTIC_PROTOCOL.md)
