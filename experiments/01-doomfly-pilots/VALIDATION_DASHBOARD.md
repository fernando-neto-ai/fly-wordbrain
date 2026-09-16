# Live partial validation results

Open [the validation dashboard](http://127.0.0.1:8768/) on the Mac running this
Codex workspace. It refreshes from macm3 about every 15 seconds while the page
is open. The three variants appear as their measurements become available;
training runs them sequentially: frozen, fixed plasticity, learned plasticity.

The active run is `results/plastic-expanded-8k-3arms-e1-monitored` on macm3.
It uses the same 8,192 training stories, full connectome, small decoder, seed,
optimizer, and calibration as the previous launch. The earlier epoch-only run
was stopped before its first checkpoint so partial validation could be added.
Its logs and original sources remain under `results/plastic-expanded-8k-3arms-e1`.

## Reading the curves

Every point measures the current model on held-out validation examples. It is
not a running average of training batches or an average across checkpoints.

- **Monitoring subset:** the same 128 validation stories, at step 0, step 1,
  and every 64 optimizer updates. These contain 15,915 next-word targets and
  15,787 second-future-word targets. They have no source-ID overlap with train
  or test. All variants and measurement times share the same subset identity.
- **Full validation:** all 1,024 validation stories at the end of each epoch,
  displayed distinctly. These measurements select checkpoints; subset curves
  do not influence selection or patience.
- **Loss:** next-word cross-entropy in nats; lower is better.
- **Perplexity:** exponential of next-word cross-entropy; lower is better.
- **Accuracy:** fraction of correct next-word predictions; higher is better.

The horizontal axis is optimizer updates within each variant. Training
progress is recorded every eight updates. A validation phase pauses optimizer
updates while collecting a snapshot, so its duration adds to wall-clock time.
Unknown words and genuine EOS are included in the main metrics, consistently
with the training protocol. The raw results also retain known-word and second
forecast metrics. No test-set results are used for the live monitoring curves.

Subset SHA256:
`db20424be574dec6168bafa80353893d308386639b753f406ae8bf767c1ca2a7`.

## Data and operation

Use the page's CSV download to export observed points. On macm3, each run keeps:

- `validation.jsonl`: all validation snapshots with story/target counts,
  dataset/subset identity, timestamps, and complete metric summaries.
- `<variant>/validation-history.json`: the same history for one variant.
- `events.jsonl`: training progress, phases, checkpoint and validation events.
- `<variant>/partial.pt`: a rolling inference checkpoint saved before each
  nonzero monitoring measurement. It is not an optimizer/RNG resume checkpoint;
  older snapshots remain in the metric history after this file is replaced.
- `process-status.json`: detached runner status and eventual exit code.

The dashboard serves only its page and metric endpoints on localhost. Its
read-only SSH refresh verifies checksums and writes an identity-bound local
cache. If the connection fails, the page retains the last results and reports
the error and their age. Browser requests drive synchronization; no separate
training or GPU work is launched by viewing the page.

To restart only the dashboard server from this workspace:

```bash
/opt/anaconda3/bin/python -u -m fly_wordbrain.dashboard \
  --run results/plastic-expanded-8k-3arms-e1-monitored \
  --host 127.0.0.1 --port 8768 --remote-host macm3 \
  --remote-run /Users/fernando/fly_wordbrain/results/plastic-expanded-8k-3arms-e1-monitored
```

The monitored training command adds `--monitor-stories 128 --monitor-every 64
--progress-every 8` to the expanded run. Monitoring preserves model modes and
training RNG; independent toy-graph tests verify unchanged learned parameters
with monitoring enabled and disabled, checkpoint-to-metric parity, and that
only full validation selects checkpoints. Core dynamics and calibrated math
sources remain unchanged.
