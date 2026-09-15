# Fly as a five-action selector

The count language model proposes five next-word candidates. The fly sees the
observed context and those candidates, then chooses one. This asks whether the
connectome can improve a useful local predictor through contextual action
selection.

The original input encoder only mapped word IDs to fixed random sensory
patterns. It did not contain the corpus transition probabilities that produced
the count baseline's 31% accuracy. The candidate selector incorporates those
probabilities explicitly.

## Measured opportunity

Counts are fitted to the 8,192 training stories. The following measurements
use the 1,024 held-out validation stories (127,651 next-word targets), with the
same 1,024-entry vocabulary as the current experiment:

| Candidate generator | First-choice accuracy | Correct answer in top five |
|---|---:|---:|
| Last word only (standard bigram) | 24.2810% | 49.4121% |
| Previous + current word (two-word context / trigram) | 31.0182% | 57.3337% |

The prototype uses the two-word-context generator. Its top-five coverage is a
hard accuracy ceiling for any selector restricted to those five. The ceiling
is not a prediction of attainable performance. On the 128-story monitoring
subset, first-choice accuracy is 30.3864% and coverage is 56.7515%.

These are word-ID metrics including UNK and genuine EOS. Restricting targets
to known lexical words, full-validation first-choice accuracy is 29.2232% and
top-five coverage is 54.2509%. No test targets were evaluated.

The measurements and their standalone evaluator are retained under
`results/action-picker-preflight/coverage.json` and `coverage.py`.

## Five choices and a small correction

At each observed position, the generator returns five distinct candidate IDs
in descending probability order. Ties use the smaller word ID. PAD and BOS
cannot be chosen; UNK and EOS can. Candidate generation never consults the
current target to fill or reorder the list.

The fly's sensory input includes the previous word, current word, five
candidate identities, and their probabilities. A fixed encoder assigns seven
disjoint banks within the existing retinal neurons to those word roles.
Candidate probabilities modulate their sensory patterns. This adds no neurons
or connectome connections and no learned input embedding.

The brain still produces 256 pooled neural features. A single affine layer
produces five corrections, with **256 × 5 + 5 = 1,285 parameters**:

```text
action_logits = log(candidate_probabilities) + linear(brain_features)
chosen_word = candidate_ids[argmax(action_logits)]
```

Initialize the affine weights and biases to zero. The initial selection then
matches the count model's first choice exactly, including its tie ordering.
The single affine correction can learn from this initialization. There is no
additional zero-initialized multiplicative gate. Training can still make the
model worse; compare every checkpoint with the unchanged first-choice policy.

The first prototype freezes the brain and trains only this action head.
Later plasticity comparisons should retain a matched frozen selector. The
original 347-edge plastic subset remains available; the earlier conditional
694-edge retry has been canceled in favor of the action-selection direction.

## Training and evaluation contract

- For training examples, generate candidates using counts with the entire
  current story removed. This prevents the story's own target occurrences
  from making its candidate generator artificially accurate. Validation
  candidates use all training stories, and no validation or test counts.
- Keep whole-story recurrent context, reset at story boundaries, and preserve
  the maximum 128 lexical words. Inputs at a prediction position contain only
  observed words and candidates derived from the allowed count table.
- If the correct word is in the five candidates, its candidate index is the
  supervised action label. If absent, exclude that position from the action
  cross-entropy but retain it as an error in overall accuracy. Do not insert
  the target or invent a sixth action.
- Report overall next-word accuracy, prior first-choice accuracy, top-five
  coverage, accuracy conditional on the answer being present, and both fixes
  and regressions relative to the prior. Report known-word metrics separately.
- Five-way conditional cross-entropy is not full-vocabulary language-model
  perplexity. Existing language-model curves must remain separately labeled.
- Changing sensory encoding requires fresh feature/activity checks and
  calibration on training stories before a full-graph GPU run. The previous
  ordered-pair input calibration cannot be reused unchanged.

The current language-model run remains a reference experiment. The new source
modules are a CPU-tested prototype; full-graph calibration and training of the
action selector have not yet been performed. A successful selector would
demonstrate added value beyond local counts; a geometry-specific claim still
requires matched connectome controls.
