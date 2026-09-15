# Frozen fly brain word-prediction pilot

A complete frozen Doomfly connectome receives fixed word codes. Only a linear
next-word decoder is trained. The first experiment runs on macm3's CPU; the
native sparse spiking kernel does not require MPS or a GPU.

The follow-up [pair protocol](PAIR_PROTOCOL.md) uses ordered bigram inputs and
two independent future-word heads, with 526,336 trainable parameters. It keeps
the original graph and adds direct-input and cached single-word controls.
After the original run, reproduce it in a new output directory with:

```bash
PYTHON=.venv/bin/python bash scripts/run_pair_pilot.sh results/pair-reproduction results/pilot-verified
```

The second argument is the original run directory containing its `features/`
and `decoder/` artifacts. Pair extraction uses the same corpus and word timing.
Results are described in [PAIR_REPORT.md](PAIR_REPORT.md); the original report
and checkpoints remain intact.

The [ten-action selector](ACTION_SELECTION.md) uses a count model to propose
ten words and a 2,570-parameter head to adjust their ranking from fly-brain
activity. It preserves the full graph and starts from the count baseline.
Its action dashboard displays accuracy, baseline and candidate coverage.

The [seven-word feedback experiment](FEEDBACK_EXPERIMENT.md) learns a 136-parameter
rule that updates temporary weights on 347 existing edges after observing each
historical word, then selects the eighth word from ten candidates. Including its
small action head, it trains 2,706 parameters and keeps the complete base graph.
Its separate dashboard reports partial validation over eight-word windows.

The new [fast-plasticity arm](PLASTIC_PROTOCOL.md) keeps the full graph and
526,336-parameter decoder, adds 20 shared synaptic-rule parameters, and uses
explicitly different smooth rate dynamics. Its [custom Metal backend](MPS_REPORT.md)
runs forward and backward on macm3's GPU with the existing PyTorch 2.8 runtime.
The [expanded corpus](DATA_EXPANSION.md) increases training to 8,192 stories
and validation/test to 1,024 each, with fresh vocabulary and calibration.
The [live validation dashboard](VALIDATION_DASHBOARD.md) shows partial held-out
results throughout GPU training, separately from full validation measurements.
The sections below describe the original frozen spiking experiments.

## What is preserved

The project vendors byte-identical source from
[nftechie/doomfly, commit 71ecf53](https://github.com/nftechie/doomfly/tree/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33).
It imports the complete original graph: **166,700 neurons, 25,582,938 directed
connections and 124,177,617 synaptic contacts**, including one-contact edges and
autapses. Neuron identities, connectivity, directed weights and transmitter
signs remain fixed. Hashes are checked after each story.

The simulator is the **original fixed NativeBrain baseline**, not the later v6
model. Its original 0.1 ms timestep, membrane/synaptic dynamics, conduction
delay, refractory period, R1–R6 retinal filter and lamina bias are retained.
Upstream source and raw-data hashes are verified before import. Raw anatomical
annotations remain available with the data. This preserves Doomfly's graph and
point-neuron dynamics; it does not add spatial cable morphology or reconstruct
a biological language faculty.

## Experimental interface

Each word has an independent, fixed random pattern over 3,335 retinal inputs.
Every code illuminates the same number of receptors at the same intensity.
There are no pretrained or learned word embeddings. One word lasts 20 ms of
simulated time; each story starts from a fresh state with 100 ms warmup.

All 1,314 neurons annotated `descending_neuron` supply the readout. Fixed signed
pooling produces 128 spike-rate features and 128 end-of-word voltage features.
Pooling changes the observation interface only; every original neuron and
connection still participates in simulation.

The decoder is a single **256 → 1,024 linear softmax layer**: **263,168 trainable
parameters**, about 1 MiB in float32. It receives only those neural features,
with train-fitted standardization. It has no hidden layer, recurrent state,
attention, word-ID input or direct route from the encoder. PAD/BOS are reserved
symbols; the vocabulary also contains UNK and EOS.

TinyStories supplies 128 training, 32 validation and 32 test stories, each
limited to 128 lexical words. Vocabulary and normalization statistics are fitted
on training data only. Natural EOS is scored; truncation does not invent EOS.
See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for source receipts and split rules.

Three decoder seeds are reported independently. Each selects its L2 penalty
from `0, 0.0001, 0.01` and its epoch using validation cross-entropy, with at most
30 epochs. Seed 0 is the predetermined primary checkpoint. Test data does not
choose hyperparameters or the seed. Train-fitted unigram and bigram models are
reference points. Known-word metrics accompany scores that include UNK.

A separate four-class linear diagnostic asks whether an early name can be
decoded from the final neural observation after 8, 16, 32, 64 or 128 words.
It has its own training, validation and test prompts; it does not train or
evaluate the natural-text decoder. There are only eight test prompts per
context length, so it is an exploratory diagnostic.

## Reproduce on macm3

Run from the repository root. The measured environment is macOS on an M3 Max,
Python 3.9.6, PyTorch 2.8.0 and the versions in `requirements-graph.txt`.
PyTorch was already installed on macm3 and is exposed to the isolated virtual
environment; another machine should install the matching PyTorch separately.
A C++ compiler is required to build the unchanged native kernel.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-graph.txt
.venv/bin/python scripts/prepare_graph.py
.venv/bin/python -m fly_wordbrain.data data/pilot
PYTHON=.venv/bin/python bash scripts/run_pilot.sh results/reproduction
```

Graph preparation downloads about 1.1 GB of pinned source data. Dataset creation
downloads a bounded set of source rows and saves response receipts. Existing
downloaded graph files must match their expected hashes. Feature extraction is
resumable only under matching data/code/configuration identity. Decoder output
directories must be new. Large source data, caches and checkpoints are ignored
by Git.

The runner verifies wiring, extracts all story features, trains the decoder,
runs the separate recall diagnostic, and saves deterministic-seed sample
continuations. Results include manifests, per-story freeze checks, feature
caches, checkpoints, learning curves and held-out metrics. A complete second
extraction is also compared with the first in this initial run to check exact
reproducibility across worker scheduling.

Tests run independently of the large graph:

```bash
python -m pytest -q tests
```

## Interpretation

This tests whether a particular fixed sensory code, original Doomfly dynamics,
and small observation interface expose useful next-word information. Failure
would not show that the connectome has no useful language-related inductive
bias: the drive, timescale, encoding and readout may be poor interfaces.
Success would not establish a biological fly's ability to speak.

Attributing an effect to biological topology requires matched controls,
especially degree/sign/weight-matched rewired graphs with identical interfaces
and training budgets. Those controls are not part of this first frozen pilot.
Sensitivity to a distant cue is also not equivalent to useful recall; the
inherited retinal filter contributes to the complete model's temporal state.
