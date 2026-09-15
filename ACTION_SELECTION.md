# Fly as a ten-action selector

The count language model proposes ten next-word candidates. The fly sees the
observed context, candidate identities and their probabilities, then chooses
one. This tests whether connectome activity can improve a useful local
predictor through contextual action selection.

The original random input encoder did not contain the corpus transition
probabilities behind the count baseline. This model incorporates that prior
explicitly. The experiment now uses K=10; K=5 remains supported as a control.

## Measured opportunity

The two-word-context count model (conventionally a trigram model) is fitted to
all 8,192 training stories. Its first-choice accuracy is **31.0182%** on the
1,024 held-out validation stories, containing 127,651 next-word targets.

| Candidate count | Correct answer present | Action-head parameters |
|---|---:|---:|
| 5 | 57.3337% | 1,285 |
| 10 | **68.3590%** | **2,570** |

Top-10 coverage is 87,261 / 127,651 targets. On the 128-story monitoring subset,
first-choice accuracy is 30.3864% and top-10 coverage is 67.9925%. Coverage is
an accuracy ceiling, not achieved selector accuracy. These word-ID metrics
include UNK and genuine EOS; known lexical targets alone have full-validation
top-10 coverage of 65.5338%. No test targets were evaluated.

Measurements are under `results/action-picker-preflight/coverage-top10.json`.
The earlier top-five measurements and evaluator remain in the same directory.

## Architecture

A fixed encoder divides the existing 3,335 retinal neurons into twelve banks:
previous word, current word, and ten candidate roles. Candidate probabilities
modulate their fixed sensory patterns. The encoder still stores 3,415,040
float32 values (13.66 MB). No learned embedding or connectome edge is added.

All 166,700 neurons and 25,582,938 directed connections remain in the circuit.
The first action experiment freezes its weights and trains one affine layer
from 256 pooled neural features to ten action corrections:

```text
action_logits = log(candidate_probabilities) + linear(brain_features)
chosen_word = candidate_ids[argmax(action_logits)]
```

The affine weights and biases start at zero, reproducing the count model's
first choice exactly. There are 256 × 10 + 10 = **2,570 learned parameters**.
Training can still degrade accuracy; the unchanged count baseline remains
eligible for checkpoint selection. The original spiking dynamics are replaced
by the previously declared smooth leaky rate model.

## Training and evaluation contract

- Training and calibration candidates use counts with the entire current
  story removed. Validation candidates use all training stories and no
  validation/test counts. Candidate generation never sees the target label.
- Candidates are distinct, sorted by probability, with smaller vocabulary IDs
  breaking ties. PAD/BOS cannot be selected; UNK/EOS can. Returned probabilities
  retain their full-vocabulary mass.
- Recurrent activity carries across each story, resets between stories and
  freezes on padding. Context remains bounded to 128 lexical words.
- Cross-entropy trains only positions whose target appears among the candidates.
  Missing targets remain errors in overall accuracy. No correct word is inserted
  into the candidates, and there is no extra fallback action.
- Report overall accuracy, count first-choice accuracy, candidate coverage,
  conditional accuracy, known-word accuracy, fixes and regressions. Conditional
  action cross-entropy is not language-model perplexity.
- Monitor the same 128 validation stories at steps 0, 1, and every 64 updates.
  Full 1,024-story validation selects checkpoints; monitoring never selects.
  The zero-correction full baseline is evaluated exactly through the count
  model on CPU. Subsequent validation runs the complete neural model.
- Fresh feature calibration uses 32 training stories. Its inputs and statistics
  are independent of validation and test. Compact partial/best checkpoints
  contain the head and calibration; they do not support training resumption.

## GPU launch and dashboard

The top-ten full-graph preflight passed on macm3, PyTorch 2.8 MPS, with CPU
fallback disabled. All 256 features varied, candidate changes exceeded replay
noise at every sampled late-story position, the head received finite nonzero
gradients and updated, and graph fingerprints stayed unchanged. This establishes
computational readiness, not a language-learning gain. Receipts are under
`results/action-selector-preflight-top10-s002/`.

The launch uses one epoch, batch size 8, learning rate .001, seed 0, scale .002,
eight internal updates, leak .5, and the expanded dataset:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u -m fly_wordbrain.action_train \
  --dataset data/expanded-8k/dataset.json --graph data/plastic-graph \
  --output results/action-expanded-8k-top10-e1 --device mps --top-k 10 \
  --epochs 1 --batch-size 8 --lr .001 --seed 0 --threads 4 \
  --monitor-stories 128 --monitor-every 64 --progress-every 8 \
  --calibration-stories 32 --global-scale .002
```

The new [action dashboard](http://127.0.0.1:8769/) uses action metrics and
separates monitoring from full validation. It refreshes every 15 seconds and
exports CSV. Run receipts and logs persist after the launching connection
closes. The earlier language-model run's step-384 checkpoint and source files
are preserved; its GPU worker was stopped for this replacement. The earlier
694-edge conditional retry is canceled.

A successful selector would show added value beyond local counts. A claim
about the original geometry requires matched connectome controls.
