---
title: Fly Recital
emoji: 🪰
colorFrom: indigo
colorTo: pink
sdk: static
app_file: dist/index.html
pinned: false
license: mit
models:
  - fernandofernandes/fly-wordbrain-rank64
datasets:
  - fernandofernandes/fly-connectome-49k
short_description: A fruit-fly connectome that recites the stories it writes
---

# Fly Recital

A fruit fly steps up to a microphone and reads you a story it is making up as it goes.

The brain doing the writing is real. **49,393 neurons and 9,050,172 measured synapses**
from the MaleCNS connectome, frozen exactly as they were mapped, wired up as the recurrent
layer of a language model. Around them sit **3,956,469 learned parameters** — and that is
the whole model. It runs in your browser tab.

## Not a recording

Every token is computed on your machine, one recurrent step per token over all 49,393
neurons and every one of the 9 million synapses. Nothing is sent to a server, and nothing
was generated ahead of time. The point cloud, the raster, the confidence waveform and the
counters are all measurements from the pass that just ran.

The forward pass is ~120 lines of plain JavaScript — no inference runtime, no WebGPU.
It reproduces the PyTorch model **token for token**: a NumPy reference was checked against
PyTorch, and the JavaScript is held to that reference's golden trace in CI-style tests
(`space/test/parity.mjs`). The weights are exact float32, not quantised.

## What the console shows

- **NEURAL ACTIVITY** — 2,048 sampled neurons at their real measured soma positions,
  brightness = `|h|`. These are **continuous rates, not biological spikes**.
- **STATE RASTER** — 256 neurons across the last 96 tokens.
- **DECK A** — the model, its per-token top-1 probability as a waveform, and three knobs
  that genuinely drive the sampler: temperature, top-p, and recital pace.
- **DECK B** — the story. Text already spoken is dimmed; text the voice has not reached
  yet is bright.

The body is a female *NeuroMechFly* rig; the connectome is from a male specimen. Its
movement is stagecraft — the language model does not control it.

## About the model

It is 13.3× smaller than the [ngxson Fly LLM](https://huggingface.co/ngxson/fly-llm-hf)
it derives from, and on a held-out population that selected neither model it scores
**1.039 nats better** in cross-entropy. That result is about **readouts**, not anatomy:
the reference spends 95.9% of its parameters on one output matrix that overfits 1,000 short
stories, and constraining its rank regularizes it. The fly's wiring is worth surprisingly
little of it: rewire the connectome at random, keeping every degree and weight, and the
model loses **0.0100 nats** — about 1% of its margin over the reference.

It was trained on 10,000 TinyStories and only knows how to ramble about Lily and Tom. It
loses story premises and sometimes recites training phrases back at you. Enjoy it for what
it is.

We also tested whether the low-rank advantage was real or just an artifact of a small
corpus: at 10× the data the gap between a full and a low-rank readout collapses from 1.632
nats to 0.313, so about four fifths of it was scarcity. This demo runs the 10,000-story
weights, which score **2.9493** against the reference's 3.9882 — though at 1.83 passes they
are not converged.

Model: [fly-wordbrain-rank64](https://huggingface.co/fernandofernandes/fly-wordbrain-rank64) ·
Connectome: [fly-connectome-49k](https://huggingface.co/datasets/fernandofernandes/fly-connectome-49k) ·
Research, receipts and negative results: [fly-wordbrain](https://github.com/fernando-neto-ai/fly-wordbrain)

## Credits

Connectome: MaleCNS v1.0 (CC BY 4.0) — FlyEM / HHMI Janelia Research Campus, University of
Cambridge, MRC Laboratory of Molecular Biology, Google Research. Architecture and tokenizer
after [ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf). Body rig and animation
conventions after [Xenova/fruit-fly-simulation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation)
and NeuroMechFly, with notices retained under `public/assets/`. App code MIT.

The source lives alongside this Space; `dist/` is the built site that HF serves.

```bash
npm install
npm run dev      # http://127.0.0.1:5173
npm run build    # dist/
```
