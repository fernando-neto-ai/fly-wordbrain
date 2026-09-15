# Expanded language corpus

The expanded corpus contains 8,192 training stories, 1,024 validation stories,
and 1,024 test stories from the same pinned TinyStories revision as the pilot.
Training has 64 times as many stories; each evaluation split has 32 times as
many. Stories remain limited to 128 lexical words.

| Split | Stories | Lexical words | Next-word targets | Second-future-word targets |
|---|---:|---:|---:|---:|
| Training | 8,192 | 1,019,414 | 1,021,293 | 1,013,101 |
| Validation | 1,024 | 127,421 | 127,651 | 126,627 |
| Test | 1,024 | 127,134 | 127,387 | 126,363 |

There are **2,034,394 overlapping training forecast targets per epoch**.
Targets include genuine end-of-story markers. The two forecasts at each
position are not independent training examples. About 90.9–91.0% of lexical
words in each split are covered by the vocabulary; the rest map to UNK.

The original 128/32/32 pilot stories retain their split assignments. New
stories are deduplicated against the original stories and each other using
both normalized complete text and the retained lexical prefix. Completed
splits are deterministically shuffled. The 1,024-entry vocabulary is fitted
only on expanded training data. This keeps the decoder at 526,336 parameters,
but changes word IDs, so previous checkpoints and calibration are not reused.

The corpus remains a bounded prefix sample with local story splits, not the
official TinyStories validation benchmark. Each enlarged evaluation split
includes its 32 previously used pilot stories plus 992 additional stories.
Exact text/prefix deduplication does not establish semantic deduplication.

## Build and audit

Run from the project root, using the appropriate Python executable:

```bash
python -m fly_wordbrain.expanded_data --pilot data/pilot/dataset.json \
  --output data/expanded-8k --train-stories 8192 \
  --val-stories 1024 --test-stories 1024 --max-words 128 \
  --vocab-size 1024 --seed 1729
python scripts/audit_expanded_data.py --dataset data/expanded-8k/dataset.json \
  --pilot data/pilot/dataset.json --output data/expanded-8k/audit.json
```

Source JSON responses and hash/revision receipts are retained under `source/`.
The builder refuses to overwrite a completed dataset. Verified source caches
can be reused after an interrupted download.

The build read 10,300 source rows from 103 verified pages. An independent audit
reconstructed all 10,240 selected stories from source bytes and checked
tokenization, full/prefix hashes, vocabulary ranking, IDs, natural EOS, target
masks, both causal forecasts, and padded trainer batches. All 192 original
split assignments were positively matched; no selected source IDs or exact
full/prefix hashes overlap across splits. The 16 focused data tests passed both
locally and on macm3's Python 3.9.6.

Dataset SHA256: `ac5e68cd17ab49e02c5183763d71a1340b8dafae87e59513bc01fb211272650e`.
The 66,827,088-byte dataset, source receipts, manifest, and audit are stored at
`data/expanded-8k/` in both project copies. Source revision:
[`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64).

## Fresh GPU calibration

On macm3, run from `/Users/fernando/fly_wordbrain`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u -m fly_wordbrain.plastic_probe \
  --graph data/plastic-graph --dataset data/expanded-8k/dataset.json \
  --output results/plastic-probe-expanded-8k-s002 \
  --device mps --global-scale .002 --internal-steps 8 --leak .5 \
  --input-gain 1 --input-center 0 --threads 4 \
  --calibration-stories 32 --calibration-words 128 \
  --benchmark-words 128 --benchmark-batch 2
```

The first 32 stories of the deterministically shuffled training split provide
calibration. Validation and test data do not set the activity or feature scales.
The full-graph gradient benchmark still uses only two stories at a time. The
probe must pass its repeat-noise, graph-integrity, and gradient checks before
its calibration can be used by the trainer.

This calibration **completed successfully on macm3** using 32 newly added
training stories and 3,944 observed positions. All 256 features varied; the
fast-state effect was 0.001718 normalized units, versus 0.000005604 for a
repeated identical frozen trajectory. It passed the required ten-times-repeat
threshold. The full-graph two-story benchmark took 3.63 seconds and verified
Metal forward/backward execution, finite gradients, parameter updates, and
unchanged graph fingerprints. These are numerical readiness checks, not
language-learning results.

The trainer's strict `load_calibration` check passed in a fresh process.
`results/plastic-probe-expanded-8k-s002/readiness.json` records the bound data,
calibration and probe hashes. Calibration artifacts are verified on both
hosts. The full expanded training run has **not** been launched.

## Expanded training command

The following command is prepared for a first full pass over the expanded
corpus. It trains all three arms from fresh, matched initializations:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u -m fly_wordbrain.plastic_train \
  --dataset data/expanded-8k/dataset.json --graph data/plastic-graph \
  --calibration results/plastic-probe-expanded-8k-s002/calibration.npz \
  --output results/plastic-expanded-8k-3arms-e1 --device mps \
  --epochs 1 --patience 1 --batch-size 8 --threads 4 --gradient-clip 1 --seed 0
```

One epoch now makes **1,024 optimizer updates per arm**, versus 16 in the
pilot. The pilot timings extrapolate to roughly **18–20 hours for the entire
three-arm run**, including its substantial final evaluation work; about
9–10 hours of that estimate is training plus per-epoch validation. These are
rough scaling estimates, not measurements on the expanded corpus. Ten epochs
could take multiple days with the current evaluation schedule.

The current trainer saves checkpoints after each epoch and its validation;
those checkpoints do not include optimizer/RNG state for training resumption.
Longer runs would benefit from resumable checkpoints within an epoch.

More data permits a meaningful learning curve, but a single run still cannot
attribute an advantage to connectome geometry. That requires matched topology
controls and adequate replication. The baseline graph, small decoder, and
20-parameter plasticity rule are unchanged by this corpus expansion.
