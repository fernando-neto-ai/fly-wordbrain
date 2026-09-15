# Reproducing and extending the pipeline

Run real training only on **macm3**. This includes eight-update training smoke
runs. Use one GPU worker at a time. A fresh checkout and output directory give
each experiment its own source identity; never edit bound source files or change
branches inside a running worker's checkout.

## 1. Environment and paths

Clone the private repository on macm3. The variables below are resolved on that
host; substitute the checkout location once and reuse the same values.

```bash
git clone git@github.com:fernando-neto-ai/fly-wordbrain.git
cd fly-wordbrain
export FLY_ROOT="$PWD"
export FLY_REVISION=65c677b3d566a2e9793d5f72999cdb441c6c0a9f
export FLY_MODEL="$FLY_ROOT/data/ngxson-fly-llm-hf/$FLY_REVISION"
export FLY_DATA="$FLY_ROOT/data/ngxson-tinystories-v1/dataset.json"
export FLY_GROUPS="$FLY_ROOT/data/connectorch-groups-v1/groups.npz"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export OMP_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
python3.13 -m venv .venv-connectorch
export FLY_PYTHON="$FLY_ROOT/.venv-connectorch/bin/python"
"$FLY_PYTHON" -m pip install -r requirements-connectorch.txt
```

Python 3.13, PyTorch 2.11.0 and the pinned dependency versions are the measured
reference environment. ConnecTorch is installed at the commit recorded in
`connectorch-source.json`; the backend verifies installed source hashes.
`requirements-ngxson.txt` alone is sufficient for the original reference.
Scripts resolve repository imports from their own location. No `pip install -e`
is required. `cluster_runner` is used only by the optional remote snapshot
helpers; it is infrastructure on the control host, not a model dependency.

## 2. Download and verify model and text data

```bash
"$FLY_PYTHON" scripts/prepare_ngxson.py --output "$FLY_MODEL"
"$FLY_PYTHON" scripts/prepare_ngxson_data.py \
  --model "$FLY_MODEL" --output "$FLY_ROOT/data/ngxson-tinystories-v1" --seed 42
```

The model download checks pinned file hashes, including the 284 MB weights.
Data preparation records source rows and revision receipts, filters whole stories
to at most 320 BPE tokens including BOS/EOS, and creates 1,000 training, 100
validation and 100 test stories. It excludes normalized full-text duplicates
across these splits. This is not semantic decontamination. The released BPE is
reused; its original fitting population is unknown.

The original dataset SHA256 is
`c1f553263545744786a8811ee6af7ace8c452820bbb99d15140b81625e40b3a8`.
Changed source data is a new dataset version, not a silent replacement.
Preparation scripts refuse incompatible cached files or nonempty new outputs.

## 3. Rebuild or reuse verified anatomical groups

The exact HF checkpoint supplies the language-model graph. Anatomical gain groups
also require the original MaleCNS annotations and the preserved Doomfly export.
This preparation is CPU data processing, not model training.

To rebuild the historical export, use the separate legacy Python 3.9 environment
and Xcode command-line tools for the native build:

```bash
python3.9 -m venv .venv-graph
export FLY_GRAPH_PYTHON="$FLY_ROOT/.venv-graph/bin/python"
"$FLY_GRAPH_PYTHON" -m pip install -r requirements-graph.txt
"$FLY_GRAPH_PYTHON" scripts/prepare_graph.py
"$FLY_GRAPH_PYTHON" scripts/prepare_plastic_graph.py \
  --output "$FLY_ROOT/data/plastic-graph"
"$FLY_PYTHON" scripts/prepare_connectorch_groups.py \
  --graph "$FLY_ROOT/data/plastic-graph/graph.npz" \
  --model "$FLY_MODEL/model.safetensors" \
  --output "$FLY_ROOT/data/connectorch-groups-v1"
```

The first preparation downloads about 1.1 GB of pinned source data. The exported
archive is checked against SHA256
`08bd21db0883698ac98501f4d70bb01eb72ea2736874fdb9092f9243b889db7b`.
The current group archive is 466 KB, with SHA256
`a07d311cd04126b5e54ea9cb3e8945470af9049bcb8499fb406e3c38d185dd7f`.
Existing verified graph/group artifacts may instead be copied with checksum
verification to the paths above. Keep their manifests and source attribution.
If a rebuild fails a hash check, investigate the dependency/export difference;
do not rewrite expected hashes merely to pass it.

Every checkpoint edge is matched to source endpoint IDs, sign and globally
scaled magnitude. The source's induced subgraph has 628,902 additional edges
whose exclusion reason is not established; they are not added to the HF graph.
The 8,156 named cell types plus 1,005 individually grouped untyped neurons give
9,161 groups and 18,322 factorized gain parameters.

## 4. Numerical preflight on macm3

With no other GPU worker running:

```bash
"$FLY_PYTHON" scripts/run_ngxson_reference.py \
  --model "$FLY_MODEL" --compare-mps \
  --output "$FLY_ROOT/results/reference-parity/cpu-mps.json"
"$FLY_PYTHON" scripts/audit_connectorch.py \
  --model "$FLY_MODEL" --data "$FLY_DATA" --groups "$FLY_GROUPS" \
  --output "$FLY_ROOT/results/connectorch-preflight/parity.json"
```

These check the pinned implementation, CPU/MPS outputs and gradients, PAD row,
canonical graph preservation and parameter identity. A successful numerical
preflight establishes correct computation, not successful language learning.
Native MPS sparse kernels use FP32 and support first-order gradients.

## 5. Original reference training and two selectors

The reference configuration is
[`ngxson-reference.json`](../experiments/configs/ngxson-reference.json).
For a **new** full reconstruction, launch on macm3:

```bash
export FLY_REFERENCE_RUN="$FLY_ROOT/results/ngxson-reconstructed-v1"
"$FLY_PYTHON" scripts/train_ngxson.py \
  --model "$FLY_MODEL" --data "$FLY_DATA" --device mps \
  --epochs 30 --second-epochs 14 --batch-size 8 --chunk-size 32 \
  --seed 42 --eval-interval-updates 100 --output "$FLY_REFERENCE_RUN"
```

The current historical run was accepted early by the user; this command is a
new reproduction recipe, not an instruction to resume that stopped run. All
learned values are initialized from scratch. Only the frozen graph/interface
comes from the released checkpoint. `best.pt` retains the minimum validation CE.
`latest.pt` includes optimizer, scheduler, RNG and within-story cache for exact
continuation under the same manifest. `--resume` requires matching bound sources,
data and configuration; changing the recipe creates a new experiment.

Once the run's `manifest.json` exists, a second CPU-only process on macm3 can
retain maximum-accuracy weights without changing the original trainer:

```bash
"$FLY_PYTHON" scripts/watch_ngxson_accuracy.py \
  --run "$FLY_REFERENCE_RUN" \
  --output "$FLY_ROOT/results/ngxson-dual-selection-v1"
```

Its `best_accuracy.pt` and `selection.json` distinguish the highest logged metric
from the best actually retained checkpoint. Start the observer at the beginning;
logs cannot reconstruct overwritten weights. Use the same filesystem for the
observer's output because retention uses immutable hardlinks.

For detached operation, `scripts/detached_run.py` accepts a run directory with a
`launch.json` containing an exact argument-list `command`, `cwd` and `environment`.
It records worker identity and exit status and keeps macm3 awake. Preserve that
receipt and verify the specific PID before stopping a job; never stop by a shared
process name. An intentional early stop must retain its stop receipt and must
not be rewritten as completion of the published epoch schedule.

## 6. Four controlled encoder experiments

The JSON files in [`experiments/configs`](../experiments/configs) are declarative
specifications. The trainers do not accept `--config`; map their `training` fields
to CLI flags as below. The campaign controller defines the same four arms and
checks preflight and source/input bindings. Its launcher is idempotent; all
smokes and full arms run serially on macm3. A user-accepted early-stop reference
must be represented by a recorded acceptance path, never fake completion files.
The ordinary gate requires a fully completed reference. For the historical
user-accepted early stop, set `FLY_BASELINE_RUN` to its preserved run directory
and `FLY_ACCEPTANCE` to its checksum-bound acceptance receipt. The original
run stores that receipt at `$FLY_BASELINE_RUN/accepted-baseline.json`:

```bash
export FLY_ACCEPTANCE="$FLY_BASELINE_RUN/accepted-baseline.json"
```

Then launch:

```bash
"$FLY_PYTHON" scripts/run_connectorch_campaign.py \
  --root "$FLY_ROOT" --baseline-run "$FLY_BASELINE_RUN" \
  --accepted-baseline-receipt "$FLY_ACCEPTANCE" \
  --groups "$FLY_GROUPS" --python "$FLY_PYTHON" \
  --output "$FLY_ROOT/results/connectorch-encoder-v1" --launch
```

The receipt records the user instruction, actual stopping cursor, both retained
checkpoints and stopped-process evidence. The launcher verifies the receipt;
an accepted early stop remains distinct from recipe completion. Inspect
`readiness.json`, `launch.json` and `campaign-status.json` to confirm dispatch.

If A128fixed is subsequently accepted and stopped early, preserve both winners,
latest checkpoint, metrics, native exit status and a checksum-bound acceptance
receipt with process cessation evidence. Do not edit the original controller's
completion gates or relabel the stopped run as successful completion. The
companion verifies the stopped arm plus original parity and smoke receipts, and
continues only B/C/D in a fresh output:

```bash
export FLY_PARENT="$FLY_ROOT/results/connectorch-encoder-v1"
"$FLY_PYTHON" scripts/continue_connectorch_campaign.py \
  --parent "$FLY_PARENT" \
  --accepted-arm-receipt "$FLY_PARENT/arms/A128fixed/accepted-early-stop.json" \
  --output "$FLY_ROOT/results/connectorch-encoder-v1-continuation" --launch
```

This is the historical encoder continuation. A stopped after10 complete epochs;
B later stopped after13 complete epochs. C/D were held and their paused dispatcher
was retired when the user selected the decoder pair below. The continuation records unequal budgets
and a comparison at the latest validation update present in every arm. The
refresh helper automatically combines stopped A with continuation results;
`--remote-run` still selects an explicit campaign.

A single full arm can be run in its own immutable checkout after preflight:

```bash
export FLY_ARM=A128fixed
export FLY_WIDTH=128
export FLY_PLASTICITY=fixed
"$FLY_PYTHON" scripts/train_connectorch.py \
  --model "$FLY_MODEL" --data "$FLY_DATA" --groups "$FLY_GROUPS" \
  --d-embed "$FLY_WIDTH" --plasticity "$FLY_PLASTICITY" \
  --readout-rank 0 --history-length 8 --device mps \
  --epochs 30 --second-epochs 14 --batch-size 8 --chunk-size 32 \
  --seed 42 --eval-interval-updates 100 --skip-final-test \
  --output "$FLY_ROOT/results/connectorch-encoder-v1/arms/$FLY_ARM"
```

Use B32fixed=(32,fixed), C128bounded=(128,bounded10),
D32bounded=(32,bounded10). An eight-update smoke adds `--max-updates 8
--eval-limit 8` and uses a separate `smokes/$FLY_ARM` directory. These are real
training updates and therefore run on macm3. Debug runs never stand in for a
full arm. New ConnecTorch training retains separate minimum-CE `best.pt` and
maximum-accuracy `best-accuracy.pt` checkpoints. This hyphenated filename is
native to the new trainer; the original external observer uses `best_accuracy.pt`
with an underscore.

Keep source, full readout, eight explicit delays, dataset ordering, optimizer
budget and seed matched. The primary comparison is the compression penalty:
B−A with fixed weights versus D−C with bounded adaptation. For CE, a negative
interaction `(D−C)−(B−A)` suggests adaptation reduces that penalty. One seed
cannot establish an anatomical advantage. Final test evaluation is deferred;
additional seeds, randomized topology controls, history changes and readout
compression are distinct subsequent experiments.

## 6b. Rank128 decoder pair

After the completed B quality assessment, the user selected E32rank128fixed and
F32rank128bounded. See [the protocol](../experiments/DECODER_REDUCTION.md).
Use fresh outputs; both scripts reject overwriting an existing execution.
With the registered macm3 environment and pinned paths:

```bash
"$FLY_PYTHON" scripts/audit_connectorch_decoder.py \
  --model "$FLY_MODEL" --data "$FLY_DATA" --groups "$FLY_GROUPS" \
  --output "$FLY_ROOT/results/connectorch-decoder-preparation-v1/parity.json"
"$FLY_PYTHON" scripts/run_connectorch_decoder_campaign.py \
  --root "$FLY_ROOT" --python "$FLY_PYTHON" \
  --baseline-b "$FLY_ROOT/results/connectorch-encoder-v1-continuation/arms/B32fixed" \
  --accepted-b-receipt "$FLY_ROOT/results/connectorch-encoder-v1-continuation/arms/B32fixed/accepted-early-stop.json" \
  --quality-results "$FLY_ROOT/results/connectorch-post-b-quality-v1/results.json" \
  --preflight-receipt "$FLY_ROOT/results/connectorch-decoder-preparation-v1/parity.json" \
  --experiment-branches "$FLY_ROOT/experiments/records/decoder-branch-mapping.json" \
  --output "$FLY_ROOT/results/connectorch-decoder-v1" --launch
```

The first command performs initialization and CPU/MPS gradient checks on macm3.
The detached controller verifies B's accepted stop and completed quality audit,
then runs eight-update smokes for both arms before full E followed by F. It binds
configs and experiment-branch commits to actual training arguments. A failure
halts the queue and requires review. A deliberate plateau stop also requires a
receipt-verified handoff before F can continue; it is not schedule completion.

From the control host:

```bash
python scripts/refresh_connectorch_decoder_progress.py
```

Read `results/connectorch-decoder-v1/progress.md` for B/E/F comparison and both
checkpoint selectors. The older A/B quality evaluator accepts only full heads;
use a rank-aware evaluation extension before scoring E/F retained checkpoints.

## 7. Quality scoring and generated-text comparison

For reference-compatible checkpoints, build the separate audit population once:

```bash
"$FLY_PYTHON" scripts/prepare_ngxson_quality_data.py \
  --existing "$FLY_DATA" --model "$FLY_MODEL" \
  --output "$FLY_ROOT/data/ngxson-quality-v1"
"$FLY_PYTHON" scripts/evaluate_ngxson_quality.py \
  --model "$FLY_MODEL" --checkpoint "$FLY_REFERENCE_RUN/best.pt" \
  --training-data "$FLY_DATA" \
  --audit-data "$FLY_ROOT/data/ngxson-quality-v1/dataset.json" \
  --selection min-ce --device mps \
  --output "$FLY_ROOT/results/quality-min-ce"
```

For the accuracy winner, use its immutable `best_accuracy.pt`,
`--selection max-accuracy` and `--selection-receipt` pointing to the observer's
`selection.json`. `scripts/compare_ngxson_texts.py` accepts that checkpoint,
receipt, `--model`, `--training-data`, `--device mps` and a fresh `--output` to
generate matched fixed-prompt continuations. Stop or suspend the exact training
worker first; the project does not run evaluation and training concurrently on
macm3's GPU.

The evaluator reports paired, token-weighted CE/accuracy and whole-story
bootstrap intervals. The first 200-story audit is already revealed; it is no
longer a fresh confirmation set. The reserved test split stays separate. Compare
text against training passages before interpreting fluency as generalization.

These two evaluation scripts load the **original ngxson architecture**. Do not
use them directly for compressed or plastic ConnecTorch checkpoints; those need
a restoration path that validates architecture, group mapping and gain vectors.
Their trainer's validation metrics and saved structural audits remain available.

## 8. Partial results and versioned records

On macm3, read `metrics.jsonl` (complete validation), `training.jsonl` (updates),
`status.json`, `manifest.json` and `process-status.json`. The reference evaluates
all 100 validation stories every 100 updates and at epoch boundaries. Accuracy
is next-BPE-token accuracy; it is not word accuracy or free-generation coherence.

From a control host with `cluster_runner` installed, refresh small verified
snapshots using explicit remote paths:

```bash
python scripts/refresh_ngxson_progress.py --host macm3 \
  --remote-run /path/on/macm3/results/ngxson-reconstructed-v1 \
  --accuracy-run /path/on/macm3/results/ngxson-dual-selection-v1 \
  --output results/ngxson-reconstructed-v1
python scripts/refresh_connectorch_progress.py --host macm3 \
  --remote-run /path/on/macm3/results/connectorch-encoder-v1 \
  --output results/connectorch-encoder-v1
```

Each snapshot writes `progress.md` and a receipt with retrieval time/checksums.
These are snapshots, not live browser dashboards. Campaign partial results show
completed arms while the remaining arms are pending or running.

Follow the [experiment registry](../experiments/INDEX.md) to commit a small
summary, exact source commit, configuration hash, source-data/checkpoint hashes,
actual training cursor, selection criterion and metrics on each branch. Keep
large raw data, optimizer states and weights in artifact storage or the macm3
run directory; Git branches store how to recreate and locate them.
