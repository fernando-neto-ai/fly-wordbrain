import './styles.css';
import { Stage } from './stage.js';
import { FlyTokenizer } from './tokenizer.js';
import { NeuronCloud, StateRaster, ProbabilityWave, Knob } from './console.js';

// Both the connectome and the weights are streamed from their own repositories, so this
// Space carries no large files of its own and the artifacts have a single home each.
const GRAPH_BASE = 'https://huggingface.co/datasets/fernandofernandes/fly-connectome-49k/resolve/main';
const MODEL_BASE = 'https://huggingface.co/fernandofernandes/fly-wordbrain-rank64/resolve/main/corpus-10k/web';
const EXPECTED_BYTES = 71_000_000;
const CLOUD_POINTS = 2048;
const RASTER_ROWS = 256;
const MAX_TOKENS = 220;
const CHUNK_TOKENS = 26;

const PROMPTS = [
  'Once upon a time, there was a',
  'One day, a little girl named Lily found a needle',
  'Tom and his dog',
  'The little bird was afraid of the rain. Its mother',
  'Sara put the red ball inside the box. Later,',
];

const $ = (id) => document.getElementById(id);
const state = {
  reciting: false,
  cancelled: false,
  voice: true,
  temperature: 0.7,
  topP: 0.9,
  pace: 1,
  tokens: 0,
  started: 0,
  spokenWords: 0,
};

let worker;
let tokenizer;
let stage;
let cloud;
let raster;
let wave;
let info;

/* ---------------- worker plumbing ---------------- */

const pending = new Map();
let sequence = 0;

function callWorker(message, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = (sequence += 1);
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...message, id }, transfer);
  });
}

function setupWorker(onProgress) {
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (event) => {
    const data = event.data;
    if (data.type === 'progress') return onProgress(data);
    if (data.type === 'error') {
      for (const [, entry] of pending) entry.reject(new Error(data.message));
      pending.clear();
      showError(data.message);
      return;
    }
    // Replies arrive in request order for the single-threaded worker loop.
    const first = pending.keys().next();
    if (!first.done) {
      const entry = pending.get(first.value);
      pending.delete(first.value);
      entry.resolve(data);
    }
  };
}

function showError(message) {
  const box = $('boot-error');
  box.hidden = false;
  box.textContent = `${message} — if this is a network or storage error, reload to retry.`;
  $('boot-progress').hidden = true;
  $('boot-start').hidden = false;
}

/* ---------------- boot ---------------- */

async function boot() {
  $('boot-start').hidden = true;
  $('boot-progress').hidden = false;
  const fill = document.querySelector('#boot-progress .bar i');
  const status = document.querySelector('.boot-status');

  setupWorker(({ bytes, label }) => {
    const share = Math.min(0.99, bytes / EXPECTED_BYTES);
    fill.style.width = `${(share * 100).toFixed(1)}%`;
    status.textContent = `${label} · ${(bytes / 1e6).toFixed(1)} MB`;
  });

  try {
    const [ready, loadedTokenizer] = await Promise.all([
      callWorker({
        type: 'load',
        modelBase: MODEL_BASE,
        graphBase: GRAPH_BASE,
        cloudCount: CLOUD_POINTS,
        rasterCount: RASTER_ROWS,
      }),
      FlyTokenizer.load('tokenizer.json'),
    ]);
    tokenizer = loadedTokenizer;
    info = ready.manifest;
    fill.style.width = '100%';
    status.textContent = 'assembling the stage';
    await start(ready);
    $('boot').remove();
  } catch (error) {
    showError(String(error.message || error));
  }
}

async function start(ready) {
  $('app').hidden = false;

  stage = new Stage($('stage-canvas'));
  await stage.load();
  const frame = () => {
    stage.render();
    cloud.draw();
    raster.draw();
    wave.draw();
    requestAnimationFrame(frame);
  };

  cloud = new NeuronCloud($('cloud'));
  cloud.setPositions(ready.cloudPositions);
  raster = new StateRaster($('raster'));
  raster.setRows(ready.rasterCount);
  wave = new ProbabilityWave($('wave-prob'));

  $('cloud-meta').textContent = `${CLOUD_POINTS.toLocaleString()} of ${info.neurons.toLocaleString()} sampled`;
  $('model-updates').textContent = `${info.updates.toLocaleString()} UPDATES`;
  document.querySelector('#deck-model .deck-title').textContent = info.experiment;

  buildKnobs();
  buildSuggestions();

  $('prompt-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const text = $('prompt-input').value.trim() || $('prompt-input').placeholder;
    recite(text);
  });
  $('camera-orbit').addEventListener('click', (event) => {
    const on = event.currentTarget.classList.toggle('is-on');
    stage.setOrbiting(on);
  });
  $('voice-toggle').addEventListener('click', (event) => {
    state.voice = event.currentTarget.classList.toggle('is-on');
    if (!state.voice) speechSynthesis.cancel();
  });

  $('stage-caption').textContent =
    'A 3,956,469-parameter model on a frozen fruit-fly connectome. Give it a beginning.';
  requestAnimationFrame(frame);
}

function buildKnobs() {
  const specs = {
    temperature: { min: 0, max: 1.4, value: 0.7, format: (v) => v.toFixed(2), apply: (v) => { state.temperature = v; } },
    topP: { min: 0.1, max: 1, value: 0.9, format: (v) => v.toFixed(2), apply: (v) => { state.topP = v; } },
    pace: { min: 0.55, max: 1.6, value: 1, format: (v) => `${v.toFixed(2)}×`, apply: (v) => { state.pace = v; } },
  };
  for (const element of document.querySelectorAll('.knob')) {
    const spec = specs[element.dataset.knob];
    const knob = new Knob(element, { ...spec, onChange: spec.apply });
    knob.defaultValue = spec.value;
  }
}

function buildSuggestions() {
  const host = $('suggestions');
  for (const prompt of PROMPTS) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = prompt;
    button.addEventListener('click', () => {
      $('prompt-input').value = prompt;
      recite(prompt);
    });
    host.appendChild(button);
  }
}

/* ---------------- speech ---------------- */

function pickVoice() {
  const voices = speechSynthesis.getVoices();
  if (!voices.length) return null;
  const english = voices.filter((v) => v.lang && v.lang.toLowerCase().startsWith('en'));
  const preferred = english.find((v) => /samantha|serena|daniel|karen|google uk|google us/i.test(v.name));
  return preferred || english[0] || voices[0];
}

function speak(text) {
  return new Promise((resolve) => {
    if (!state.voice || !('speechSynthesis' in window) || !text.trim()) return resolve();
    const utterance = new SpeechSynthesisUtterance(text);
    const voice = pickVoice();
    if (voice) utterance.voice = voice;
    utterance.rate = 0.92 * state.pace;
    utterance.pitch = 1.32;
    utterance.onstart = () => {
      stage.setSpeaking(true);
      $('voice-state').textContent = 'RECITING';
    };
    utterance.onboundary = (event) => {
      if (event.name && event.name !== 'word') return;
      state.spokenWords += 1;
      stage.pulse(0.7);
    };
    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();
    speechSynthesis.speak(utterance);
  });
}

/* ---------------- the recital ---------------- */

function endsClause(text) {
  return /[.!?]["')\]]?\s*$/.test(text);
}

async function recite(promptText) {
  if (state.reciting) {
    state.cancelled = true;
    speechSynthesis.cancel();
    await new Promise((resolve) => setTimeout(resolve, 60));
  }
  state.reciting = true;
  state.cancelled = false;
  state.tokens = 0;
  state.spokenWords = 0;
  state.started = performance.now();

  const promptIds = [info.bos, ...tokenizer.encode(promptText)];
  $('recite').disabled = true;
  $('prompt-echo').textContent = promptText;
  $('continuation').textContent = '';
  $('transcript').classList.add('is-live');
  $('voice-title').textContent = promptText;
  $('model-state').textContent = 'RUNNING';
  $('model-state').classList.add('is-live');
  $('voice-state').textContent = 'THINKING';
  $('voice-state').classList.add('is-live');
  $('stage-state').textContent = 'RECITING';
  document.querySelector('.stage-label').classList.add('is-live');
  $('stage-caption').textContent = 'Every token below is being computed in this tab, right now.';
  wave.clear();
  raster.clear();

  const primed = await callWorker({ type: 'prime', promptIds });
  applyTelemetry(primed);

  const produced = [];
  let spokenUpTo = 0;
  let finished = false;

  // The voice trails the model, so the transcript shows what has been said in one
  // colour and what is still queued in another.
  function renderStory(text, saidUpTo) {
    const said = document.createElement('span');
    said.className = 'said';
    said.textContent = text.slice(0, saidUpTo);
    const rest = document.createElement('span');
    rest.textContent = text.slice(saidUpTo);
    const host = $('continuation');
    host.replaceChildren(said, rest);
  }

  while (!finished && !state.cancelled && produced.length < MAX_TOKENS) {
    // Generate one clause, showing it as it arrives, then hand it to the voice.
    const chunkStart = produced.length;
    while (!state.cancelled && produced.length < MAX_TOKENS) {
      const result = await callWorker({
        type: 'next',
        temperature: state.temperature,
        topP: state.topP,
      });
      if (result.eos) { finished = true; break; }
      produced.push(result.token);
      state.tokens += 1;
      applyTelemetry(result);
      wave.push(result.probability);
      $('fader-value').textContent = `${(result.probability * 100).toFixed(0)}%`;
      $('fader-fill').style.width = `${(result.probability * 100).toFixed(1)}%`;
      renderStory(tokenizer.decode(produced), spokenUpTo);
      $('voice-count').textContent = `${state.tokens} TOKENS`;
      const seconds = (performance.now() - state.started) / 1000;
      $('voice-rate').textContent = `${(state.tokens / Math.max(seconds, 0.001)).toFixed(1)} TOK/S`;
      $('transcript').scrollTop = $('transcript').scrollHeight;
      stage.pulse(0.22);
      const soFar = tokenizer.decode(produced.slice(chunkStart));
      if (endsClause(soFar) || produced.length - chunkStart >= CHUNK_TOKENS) break;
      // Let the browser paint between tokens; the model is far faster than the voice.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    if (state.cancelled) break;
    const full = tokenizer.decode(produced);
    const phrase = full.slice(spokenUpTo);
    await speak(phrase);
    spokenUpTo = full.length;
    renderStory(full, spokenUpTo);
    stage.setSpeaking(false);
  }

  $('transcript').classList.remove('is-live');
  $('model-state').textContent = 'CUED';
  $('model-state').classList.remove('is-live');
  $('voice-state').textContent = state.cancelled ? 'STOPPED' : 'DONE';
  $('voice-state').classList.remove('is-live');
  $('stage-state').textContent = 'STANDING BY';
  document.querySelector('.stage-label').classList.remove('is-live');
  $('stage-caption').textContent = finished
    ? 'The fly reached its end-of-story token.'
    : 'Recital complete. Try another beginning, or turn the temperature knob.';
  $('recite').disabled = false;
  stage.setSpeaking(false);
  state.reciting = false;
}

function applyTelemetry(result) {
  if (result.cloud) cloud.update(result.cloud);
  if (result.raster) raster.push(result.raster);
  if (typeof result.meanAbs === 'number') $('stat-mean').textContent = result.meanAbs.toFixed(4);
  if (typeof result.active === 'number') $('stat-active').textContent = result.active.toLocaleString();
}

if ('speechSynthesis' in window) speechSynthesis.getVoices();
$('boot-start').addEventListener('click', boot);
