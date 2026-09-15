# Replicating ngxson's Fly LLM

This reference reproduces [ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf),
the model behind the [browser demo](https://huggingface.co/spaces/ngxson/fly-llm-demo).
The user selected the exact reference architecture first, including its large
readout. The earlier full-graph, small-head experiments remain separate.

## Pinned sources

| Artifact | Revision |
|---|---|
| PyTorch checkpoint and modeling code | `65c677b3d566a2e9793d5f72999cdb441c6c0a9f` |
| Browser Space | `41a1483c05aacad99e6ac50febbddf2264f8497a` |
| ONNX export | `e293b8c2919a3cf41fddd530458c22199db631b8` |
| Xenova data packaging inspected | `776d115ee5aa934578a87fd6d260d138084f59c1` |

The model weights SHA256 is
`355f06c44d14e38af50e9c801f51839c37a0a56ac4ca016da2be9b3ae6f215ad`.
The preparation script verifies every small file against its pinned Git blob
hash and the 284,134,372-byte safetensors file against its LFS SHA256. The
original modeling source is unmodified and reviewed before execution.

Xenova's simulation packages the same MaleCNS source data as our original
Doomfly import: 166,700 classified entries and 25,582,938 directed connections.
Its body animation uses engineered controls. The language model takes an
induced central-brain subset from that data; it does not use Xenova's body or
spiking dynamics.

## Exact architecture

- 49,393 neurons; 9,050,172 stored directed edges.
- Frozen signed, globally scaled connectome matrix. Stored zero-valued edges
  remain in the checkpoint. No synaptic rewiring or fast weights.
- Released 1,024-token byte-level BPE and 128-dimensional trainable embeddings.
- Eight explicit token-delay slots, each projecting to 1,758 neurons. The code
  injects 14,064 neurons; five of the nominal 14,069 input neurons are unused
  because the slot size uses integer division.
- One recurrent step per token:
  `x = 0.1*x + 0.9*tanh(gain * (rec_gain * W@x + input) + bias)`.
- LayerNorm over all 49,393 states, followed by a bias-free 49,393-to-1,024 head.

| Trainable component | Parameters |
|---|---:|
| Token embeddings | 131,072 |
| Eight input projections | 1,800,192 |
| Neuron gain, recurrent gain, and bias | 148,179 |
| LayerNorm scale and bias | 98,786 |
| Output matrix | 50,578,432 |
| **Total** | **52,756,661** |

The HF page's larger tensor-element total includes frozen buffers and integer
indices. Learned neuron gains modify effective recurrent strengths, even though
the stored edge weights are fixed. Gains in the released checkpoint are
positive, but the original training parameterization does not constrain their
sign. The base matrix's spectral radius of 0.99 is not a guarantee about the
effective trained recurrence.

The delay line directly supplies recent token history. This is not evidence of
eight-token memory learned inside the reservoir, nor of 128-word retention.

## Reproduce the released checkpoint

Use a separate environment with `requirements-ngxson.txt`; the original macm3
Python 3.9 environment cannot import the reference's Python 3.10+ type syntax.

```bash
python scripts/prepare_ngxson.py
python scripts/run_ngxson_reference.py --output results/ngxson-reference/cpu.json
PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/run_ngxson_reference.py \
  --compare-mps --output results/ngxson-reference/cpu-mps.json
```

The inference script checks the downloaded source and model hashes before
loading local custom code, uses BOS, greedy decoding, and the author's three
sample prompts. It records exact token IDs, timing, parameter counts, and
before/after graph-buffer hashes. Existing result files cannot be overwritten.

The first CPU run on the local Mac used PyTorch 2.11.0, Transformers 5.9.0, and
four CPU threads. All three prompts generated 80 new tokens; continuations
match the displayed author examples, including their unusual wording. Measured
throughput was 19.0–22.1 new tokens/s, including prefill and excluding model load.
This reproduces checkpoint inference, not a training result.

## Apple GPU adapter

`fly_wordbrain/ngxson_mps.py` adapts one model instance. It preserves the exact
upstream forward function and replaces only the sparse multiplication primitive
inside a private function namespace. Sorted forward and transpose CSR caches
use our existing Metal kernels. Canonical checkpoint buffers retain their
original order, values, and state-dict keys. No dense adjacency is created.

Load the checkpoint on CPU, then call `enable_mps(model)`. The adapter validates
the pinned upstream source hash. Native MPS fallback is disabled for verification;
backward multiplies by the explicitly transposed fixed graph. Higher-order
derivatives are unsupported. Call `disable_mps(model)` before moving the model
back to CPU or replacing graph buffers.

The adapter also installs a gradient hook on this model's padding embedding
row. A standalone PyTorch 2.11 MPS test gave that row a nonzero gradient while
CPU gave zero, including with a dense model and no custom sparse kernels. The
hook restores CPU `padding_idx` semantics without changing forward values.
It is removed when the adapter is disabled. Gradient and two-step AdamW
parameter/momentum comparisons cover the fix.

Verified on macm3 (M3 Max), with native CPU fallback disabled:

- **29 tests passed**: 21 adapter checks and eight trainer checks, including
  interrupted-versus-uninterrupted resumption across both optimizer phases.
- Released checkpoint: all 80 generated token IDs match CPU. On 89 identical
  teacher-forced positions, maximum logit difference was 2.48e-5, maximum
  probability total variation was 2.32e-6, and all top choices agreed.
- One 80-token GPU generation took 0.474 seconds, **168.6 new tokens/s including
  prefill**, excluding model/kernel initialization. This is one measured sample,
  not sustained training throughput.
- Full randomly initialized model, two stories x 32 tokens: CPU/MPS losses
  both 7.5144267; maximum parameter-gradient relative L2 error 1.09e-5.
  Padding-row gradient was exactly zero; canonical buffers were unchanged;
  32 forward and 31 backward Metal kernel calls were recorded.
- An eight-update, eight-validation-story smoke run completed successfully,
  saved resumable checkpoints, and deliberately skipped final testing. Its
  short validation trajectory is not a learning result.

Full parity receipts are under `results/ngxson-reference/`. The unquantized
PyTorch checkpoint is the inference reference. The browser uses an int8 ONNX
readout and quantized edge representation, so browser/PyTorch bitwise parity
is not claimed.

## Reconstructed training protocol

The author publishes a description, but no training script, exact story IDs,
random seeds, batch size, weight decay, optimizer betas, or full training logs.
Training here can reproduce the architecture and documented recipe, not the
author's exact optimization trajectory.

Published settings: 1,000 stories and 100 held out, story limit 320 tokens,
next-token cross-entropy, 32-token truncated backpropagation, AdamW with cosine
schedule, 30 epochs at learning rates 1e-3 for input/neuron parameters and 1e-4
for readout, then 14 epochs at 3e-4 / 3e-5.

Our explicit reconstruction choices:

- Reuse the released tokenizer to preserve the input/output interface. Its
  original fitting corpus is unknown, so tokenizer exposure to our held-out
  stories cannot be ruled out.
- Select the first 1,000 qualifying official TRAIN stories. Select the first
  200 qualifying official VALIDATION stories and divide them with seed 42 into
  100 validation and 100 test stories. Reject full-text duplicates across splits
  after whitespace normalization and case folding. This is not semantic deduplication.
- The 320-token limit includes BOS and EOS; whole stories are filtered, never
  truncated. Source responses, row IDs, dataset revision, and hashes are saved.
- Initialize all trainable values from scratch, retaining only the frozen
  graph and input/output neuron indices from the checkpoint. No pretrained
  language weights initialize the training run.
- Batch size 8, seed 42, AdamW betas (0.9, 0.999), weight decay 0.01, gradient
  clipping at norm 1. LayerNorm parameters use the readout learning rate.
- Initialize recurrent gains to `5 / sum(abs(incoming W))`; rows with zero
  total input use gain 1. Retain unconstrained gains to match the architecture.
- Update after each 32-token chunk; detach recurrent state across chunks but
  preserve the last eight token IDs. Reset both states at every story boundary.
- Preserve every next-token target across chunk boundaries. Right padding is
  masked and excluded from both loss and accuracy denominators.
- Report token-weighted validation CE and top-1 accuracy during training. Select
  the best checkpoint on validation only; access the final test once afterward.

```bash
python scripts/prepare_ngxson_data.py
PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/train_ngxson.py \
  --model data/ngxson-fly-llm-hf/65c677b3d566a2e9793d5f72999cdb441c6c0a9f \
  --data data/ngxson-tinystories-v1/dataset.json \
  --device mps --output results/ngxson-reconstructed-v1
```

The model card reports final training/validation CE 0.86/3.99. At epoch 30 it
reports real wiring 1.02/3.84 versus permuted wiring 0.77/4.27. These are author
measurements, not our results, and exact scores are not directly comparable
across different splits. A geometry claim needs independently trained matched
controls; a successful story-generation reproduction alone does not establish it.

Model/data attribution: ngxson and MaleCNS/FlyEM, HHMI Janelia, University of
Cambridge, MRC LMB, Google Research, CC BY 4.0. Upstream modeling code is MIT.
TinyStories is by Eldan and Li; retain its dataset license and source receipts.
