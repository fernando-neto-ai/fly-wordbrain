# Pilot data and evaluation contracts

The natural-text corpus is a small subset of the authors' public
[TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories),
licensed CDLA-Sharing-1.0. This repository's data code requests at most 100 rows
per page from Hugging Face's dataset viewer, not the multi-gigabyte full corpus.
Each source response is cached with its URL, retrieval date, SHA-256 and HTTP
`x-revision`. The viewer revision must match the repository metadata revision.
`manifest.json` hashes the final dataset and the separate recall probes.

Run `python -m fly_wordbrain.data data/pilot` from the repository root. The
default split contains 128/32/32 train/validation/test stories selected from a
bounded prefix of the official training split. Selection and shuffling use
seed 1729. These are our own held-out story splits, not the dataset's official
validation split. This is a pilot sample, not a representative benchmark.

Full stories are normalized into lexical words and deduplicated before any
split. Identical retained prefixes are also deduplicated. Whole stories belong
to a single split. A maximum of 128 lexical words is retained from each story.
Words are lowercase letter sequences with internal apostrophes; punctuation and
numbers do not become tokens, and hyphenated forms become separate words.
Consequently "128 words" really refers to these lexical units, not subwords.

The vocabulary is fitted to training words only: at most 1,024 entries including
PAD, UNK, BOS and EOS. Frequency ties are resolved alphabetically. Each split's
unknown-word fraction, unknown types and frequent unknown words are reported.
All comparisons use the same vocabulary. Cross-entropy with an UNK class should
be interpreted alongside known-word coverage and lexical-only metrics.

`words` contains lexical words. `word_ids` contains BOS, then those words, then
EOS only if the original story ended within the retained window. Truncation
does not manufacture a false end-of-story target. `target_mask` is aligned with
`word_ids`, with PAD and BOS excluded. A decoder consumes `word_ids[:-1]` and
predicts `word_ids[1:]` under `target_mask[1:]`. Reset the brain before every
story; no neuronal state may cross story or split boundaries. EOS predictions
may use at most 128 preceding lexical words; BOS/EOS are not counted as words.

`recall_probes.json` is a separate, deterministic synthetic diagnostic. It is
never mixed into natural-text training. Each paired group varies only one early
name across Lily, Tim, Tom and Sam. Four labels are balanced separately at every
split and context length, and all cues occur at the same position. Exactly
8, 16, 32, 64 or 128 lexical words precede the required name prediction. The cue
is at zero-based word position 1, so there are C-2 words between cue and answer.
The final five-word query and filler never repeat any label.

Recall train/validation/test splits use distinct cue/query templates and filler
combinations. A negative result on this split cannot separate inability to
interpret a held-out query from loss of the name memory. Any diagnostic trained
on these examples must be reported separately from the natural text decoder.
The probes declare themselves invalid for the text word encoder if one of the
four label names is missing from the training vocabulary; mapping two names to
UNK cannot support a meaningful memory comparison. These are memory probes,
not evidence of natural-language communication.
