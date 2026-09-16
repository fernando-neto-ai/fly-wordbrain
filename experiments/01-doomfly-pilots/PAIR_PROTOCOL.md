# Ordered bigram input and two future words

Protocol fixed before this run's evaluation. The original word experiment and
its artifacts remain intact. This follow-up uses the same pilot story splits
and vocabulary; the test set is reused for a paired developmental comparison,
not an untouched final benchmark for the entire research program.

The full original fixed Doomfly NativeBrain graph and dynamics stay frozen.
At source position p, the stimulus represents `(word[p-1], word[p])` and the
two linear heads forecast `word[p+1]` and `word[p+2]`. At the start, the pair
is `(BOS, BOS)`. The second head never sees the actual or predicted first
future word. Both output distributions are conditionally independent given
the observed features. Target overlap across successive windows is recorded;
there is no claim that two-head training updates the frozen representation.

We advance one word per 20 ms neural interval, with the original 100 ms story
warmup, reset policy, 0.1 ms integration step and lamina bias. Each story has
at most 128 lexical words; BOS does not count toward this limit. Missing final
second targets are masked. Truncated stories never acquire artificial EOS.

Input roles use a fixed random partition of the 3,335 retinal inputs into
1,667/1,668 receptors. Each word illuminates exactly 417 receptors in its role,
at intensity 0.02. The 834 active receptors and total brightness per interval
match the original word code. Partition seed is 1729+1907; role code seeds
are 1729 and 1730. Swapping words changes their stimulus, while sharing a word
in one role shares that role's pattern. This adds one explicit previous word
at the input interface; no original neuron or edge is removed or merged.

The main observation remains 256 fixed pooled spike-rate/voltage features
from all 1,314 descending neurons. Two independent affine softmax heads over
the 1,024-entry vocabulary have 526,336 trainable parameters in total.

Three representations are compared with exactly this decoder capacity:

- **Pair through brain:** the new ordered-pair stimulus and recurrent brain.
- **Direct pair:** a fixed 256-dimensional signed CountSketch of the exact
  retinal stimulus (seed 3187), with no neural simulation or temporal state.
- **Single word through brain:** reuse the original word-encoding neural
  observations, now with a second future-word head. Head 1 should reproduce
  the original result exactly under its unchanged optimizer and seed.

Train-only count references predict each horizon from `(previous,current)`,
with smoothed current-word and unigram backoff. Horizon 2 conditions on the
same observed pair, never the unavailable future word. These are separate
trigram-style and skip-position forecasts, not teacher-forced chain scores.

Each head independently fits train-only feature normalization and selects
epoch/L2 using validation CE: seeds 0/1/2, L2 0/0.0001/0.01, at most 30 epochs,
Adam learning rate 0.01, batch 256, patience 8. All selections precede test
array loading. Seed 0 is primary; no selection between seeds. Expected target
counts by horizon are train 16,017/15,889, validation 3,991/3,959, and test
3,992/3,960. First-horizon CE/perplexity compares directly with the original
next-word score. Combined CE averages scored forecasts; its exponential is
labeled forecast perplexity because windows overlap. Also report each
horizon separately, exact two-word accuracy, and known-lexical scores.

Generation samples both heads from the same brain state before feedback.
Then it feeds both successive sliding input windows through the brain before
the next prediction. PAD/BOS are forbidden outputs, UNK remains visible, and
EOS stops generation. Use the original three prompts and sampling seed 1729,
temperature 0.8, at most 24 new words. Over 128 lexical words, reset and replay
the latest 128 words with the proper initial BOS pair.

A gain over the direct linear control could reflect nonlinear pair processing
or older temporal state; it would not isolate biological topology. Matched
rewired graphs remain a future experiment. This run does not refit the separate
name-memory diagnostic.
