# Recorded rank128 storyteller

`scripts/export_rank128_demo.py` restores the preserved `E32rank128fixed`
checkpoint and records **CPU inference** through all 49,393 modeled neurons and
9,050,172 stored edges. There is no optimizer, backward pass, GPU selection, or
training. Existing model and training sources remain unchanged.

The exporter verifies checkpoint bytes against the accepted native selector,
loads and hashes the same open file, strictly restores all named parameters,
and checks model, data, grouping, package-content, and reference hashes. It
checks sources, checkpoint, parameters, and graph again after inference.
Package installation paths may differ; pinned package contents must match.

## Files

The default destination is `demo/public/data/`. Publish `manifest.json` only
after every story has completed and all final checks passed. Reusing a
destination containing a published manifest is rejected.

- `manifest.json`: `format_version: 1`, `mode: "recorded_replay"`, model selector,
  checkpoint SHA256/update, architecture counts, generation provenance, and
  `stories: [{id, title, prompt, trace_url, sha256, new_tokens, frames}]`.
- `neuron-layout.json`: `neurons: [{node_index, body_id, position: [x,y,z]}]`,
  `coordinate_system`, full/displayed counts, missing-position count, and
  deterministic sample metadata. All trace rows follow this neuron order.
- `<story-id>.json`: prompt IDs, frames, generated IDs, final `continuation`,
  `full_text`, and generation counters. Manifest URLs are relative to the
  manifest's directory.

## Frame alignment

Each frame represents one real forward pass with a fresh story cache on its
first frame. `state_u8` records the recurrent state **after consuming**
`input_token_id` and **before predicting** `predicted_token_id`. The state is
the modeled continuous recurrent activation used for the next-token readout,
not recorded biological spikes or a measurement of a living fly.

Frames before the final prompt input have `phase: "prompt"`. Their next-token
predictions are diagnostic and are not emitted. The frame consuming the last
prompt token has `phase: "generation"` and predicts the first generated token.
Subsequent generation frames consume the previously generated token. EOS is
emitted once and stops generation; it is omitted from decoded display text.
BOS is prepended once to the prompt. Generation is greedy and independently
starts with an empty recurrent cache for each story.

Common frame fields are `position`, `phase`, `input_token_id`, `input_text`,
`predicted_token_id`, `predicted_text`, `probability`,
`top_k: [{id, text, probability}]`, `state_u8`, and `whole_brain_state` statistics.
Generation frames add `generation_index` and `continuation_so_far`. Display the
latter as the canonical decoded prefix: concatenating individually decoded BPE
tokens can produce incorrect spacing or incomplete byte sequences.

## Activity and anatomy

Each `state_u8` value is `round(255 * abs(h))`, with a fixed range from zero to
one across every frame and story. The model's tanh recurrence with leak 0.9 and
zero initial state keeps these magnitudes within that range; the exporter
rejects nonfinite or out-of-range states. Brightness has no excitatory or
inhibitory meaning. It is not normalized independently for each frame.

The layout input must declare `coordinate_system.type` as `anatomical` or
`schematic`. Each body ID must match the canonical graph's grouping archive.
For measured anatomy, neurons missing coordinates are omitted; no coordinates
are invented. The exporter deterministically samples 2,048 known positions
by default, sorts them by canonical index, and preserves their coordinate
metadata. Sampling affects the display only; inference still uses the full
graph. Source-coordinate units and provenance belong in `coordinate_system`.

The frontend may animate the recorded values and illustrate body movement, but
must identify this as a recorded inference replay. Illustrated movement is
not inferred motor behavior. These selected generations are a demonstration
from a model trained on 1,000 TinyStories examples, not a quality benchmark.

## CPU smoke

Use the local `--arm`, `--checkpoint`, `--model`, `--data`, `--groups`, and
`--layout` paths with a fresh `--output` directory. Set `--story-limit 1
--max-new-tokens 2` for a short latency smoke. The default final recording is
three prompts with at most 120 new tokens each, stopping earlier on EOS.
`--prompts-json` accepts a list of `{id,title,prompt}` records; the prompt file
is included in the provenance bindings. A recording failure never falls back
to synthetic neural activity.
