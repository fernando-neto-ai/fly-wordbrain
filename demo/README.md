# Fly Stories

A local, static browser demo of the preserved rank-128 language model. The page replays actual recorded CPU inference; it does not train a model or run inference in the browser.

```sh
cd demo
npm ci
npm run dev -- --port 8781 --strictPort
```

Open `http://127.0.0.1:8781`. `npm run build` creates a portable `dist/` directory; `npm run preview` serves the production build locally. Fonts, the Three.js runtime, body assets, coordinates and recordings are bundled locally.

## Interaction

- Play or pause with the main button or Space. Replay restarts the current recording.
- Drag the timeline to inspect an exact recorded frame; change the pace or choose one of the preserved stories.
- Drag the specimen to orbit it. “Closer look” moves the camera toward the brain and makes the body transparent.
- Listen, Wander and Take flight control an **illustrated actor**, independently of language inference. Its leg and wing articulation uses NeuroMechFly kinematics; these movements are not motor outputs from the language model.
- The speaker button enables optional browser speech synthesis. Recorded token playback remains the timing reference; browser speech can have different timing.

## Recorded data

`public/data/manifest.json` selects the preserved checkpoint, trace files and measured neuron layout. The frontend accepts the exporter’s `recorded_replay` format and validates every frame’s activation count, byte values, and exact final decoded text. Story text is taken from `continuation_so_far`, avoiding concatenation errors with subword tokens. Prompt frames are displayed as prefill; generation frames reveal their next-token prediction.

The shared layout contains 2,048 measured MaleCNS soma positions, sampled from 44,279 mapped neurons in the 49,393-neuron model. The display preserves relative distances with a uniform scale and converts coordinate axes for viewing. The body is a different, female specimen; the brain overlay is illustrative registration. See the in-page provenance drawer and the notices under `public/assets/`.

Neuron brightness is based on the exported `round(255 × abs(h))` states with a **fixed square-root display curve**. The brief transition interpolates two recorded states; there are no synthetic neural flashes. The small line plot displays the sample mean absolute state. The displayed probability and current token come from the same frame as the neural state. Colors do not indicate excitation or inhibition, and the states are not biological spikes.

The data can be inspected through the read-only portions of `window.__flyDemo.state`; the debug helper also exposes `seek(index)`, `play()`, `pause()` and `setView('fly' | 'brain')` for browser QA.
