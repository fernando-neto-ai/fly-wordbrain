# Rank128 fruit-fly storyteller

An interactive local demo pairs an articulated fruit fly with recorded text
generation and the corresponding neuron states from **E32rank128fixed**. The
first version uses the retained maximum-validation-accuracy checkpoint at
14,300 updates. It does not start training or require a model service.

## Open the demo

```bash
cd demo
npm ci
npm run dev -- --port 8781 --strictPort
```

Open `http://127.0.0.1:8781`. `npm run build` produces a static `demo/dist`;
`npm run preview -- --port 8781 --strictPort` previews that build. The checked-in
recordings and licensed geometry make playback self-contained. Narration uses
the browser's installed speech voices and requires user interaction.

The viewer is a recorded replay, not live prompt completion. Play/pause, scrubbing,
story selection and speed controls advance the same recorded text and neural
frames. Body movement is a crafted illustration. The language model did not
learn to walk or fly, and the speech voice is not produced by its decoder.

## What is real model output

The exporter restores all 7,183,157 learned parameters and evaluates the complete
49,393-neuron / 9,050,172-edge model on CPU, with no optimizer or backward pass.
Checkpoint SHA256:
`bd0dca1dd68239cea1e07769a5d7939066d315af4177512053bb332ae8352943`.
The encoder has 482,816 parameters and the rank128 decoder has 6,453,376.
Stored base edges are fixed; neuron dynamics were trained.

Each frame contains the actual recurrent state **after consuming the current
token and before predicting the next token**. Text is decoded cumulatively to
preserve BPE boundaries. Brightness uses `round(255 * abs(h))`, with a fixed
zero-to-one range; it is a modeled rate-state magnitude, not a biological spike
or a label for excitatory versus inhibitory neurons. See the
[trace schema](RANK128_DEMO_SCHEMA.md).

All 49,393 published subset IDs match the checkpoint's biological node IDs.
44,279 have measured soma coordinates; 5,114 lack them. The animation displays
a deterministic sample of 2,048 neurons with known positions. Missing locations
are never invented. The soma cloud is scaled into the head for illustration;
the female NeuroMechFly body and male connectome are different specimens.

The three shipped recordings are unchanged greedy generations. Two ended on
EOS; the bird recording reached its 240-token cap. These selected examples can
be repetitive or ungrammatical and are not an independent quality benchmark.
The model was trained on 1,000 TinyStories examples. The reserved test set is not
used by this demo. A lossless packaging step reduces JSON size without changing
decoded text, tokens, probabilities, positions or activity values.

## Rebuild the recordings

Use the verified local checkpoint plus its original `manifest.json` and
`accepted-early-stop.json`. Paths may be relocated; content hashes must match.

```bash
python scripts/prepare_demo_geometry.py --download
python scripts/export_rank128_demo.py \
  --arm results/story-demo-v1/checkpoint \
  --checkpoint results/story-demo-v1/checkpoint/best-accuracy.pt \
  --model data/ngxson-fly-llm-hf/65c677b3d566a2e9793d5f72999cdb441c6c0a9f \
  --data data/ngxson-tinystories-v1/dataset.json \
  --groups data/connectorch-groups-v1/groups.npz \
  --layout results/story-demo-v1/neuron-layout-source.json \
  --output results/story-demo-new/raw-replays --max-new-tokens 240
python scripts/package_rank128_demo.py \
  --source results/story-demo-new/raw-replays \
  --output results/story-demo-new/site-data \
  --story-id story-3 --story-id story-1 --story-id story-2
```

Review the new manifest and recordings before replacing `demo/public/data`.
Both tools require fresh output manifests. The exporter verifies selected
checkpoint identity, source and package contents, graph buffers and parameter
hashes before and after inference. The package verifies original file hashes
and exact JSON-value preservation. All actual model training stays on macm3;
these recordings were made with CPU inference while rank64 training continued.

## Verification

The exporter tests cover full-frame state capture, BOS/cache/EOS handling,
cumulative BPE decoding, probability alignment and checkpoint hash rejection.
All five pass. The shipped trace and layout hashes were independently checked.
Browser checks cover play/pause, exact-frame scrubbing, story switching, EOS
versus token-limit endings, movement, the provenance drawer and narration
controls. Desktop and 390-pixel mobile layouts were rendered and inspected;
playback works without failed asset requests or JavaScript errors. The narrated
audio itself is supplied by the user's installed browser voices.

## Visual references and attribution

The motion reference is
[Xenova's fruit-fly simulation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation):
an articulated fly, walking/turning and flight/landing alongside a brain view.
The story-and-neuron pairing is inspired by
[ngxson's Fly LLM demo](https://huggingface.co/spaces/ngxson/fly-llm-demo).
The referenced X clips could not be opened during this implementation; the HF
demo was run and inspected directly. No social-video media is bundled.

The anatomical fly derives from NeuroMechFly v2 assets distributed with the
Xenova demo. Body assets are Apache-2.0; kinematic/application code retains MIT
notices; MaleCNS data is CC BY 4.0. Exact pinned source URLs, hashes, notices,
coordinate coverage and reconstruction metadata are under
`demo/public/assets/`. `scripts/prepare_demo_geometry.py` reproduces geometry
from its ignored download cache. Original model weights and optimizer states
are not committed to this repository.
