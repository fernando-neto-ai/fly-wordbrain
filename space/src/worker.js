// Inference worker: downloads the brain once, then runs one recurrent step per request.
//
// The main thread drives the pace so the text appears as the fly says it, rather than
// racing ahead. Generation is cheap enough (tens of tokens per second) that the
// bottleneck is speech, not the model.

import { FlyEngine } from './engine.js';

let engine = null;
let manifest = null;
let cloudIndex = null;
let rasterIndex = null;
let logits = null;
let generated = 0;

const TENSORS = ['wte', 'in_proj', 'gain', 'rec_gain', 'bias', 'ln_weight', 'ln_bias', 'head_a', 'head_b'];

async function fetchTracked(url, label, onBytes) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${label}: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body) return { buffer: await response.arrayBuffer(), total };
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onBytes(value.length, label);
  }
  const merged = new Uint8Array(received);
  let at = 0;
  for (const chunk of chunks) {
    merged.set(chunk, at);
    at += chunk.length;
  }
  return { buffer: merged.buffer, total: received };
}

function pickIndices(valid, count, neurons) {
  // Even stride over neurons that have a measured soma position, so the sample is
  // spread across the brain instead of clustered at low indices.
  const usable = [];
  for (let i = 0; i < neurons; i += 1) if (valid[i]) usable.push(i);
  const step = usable.length / count;
  const out = new Int32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = usable[Math.min(usable.length - 1, Math.floor(i * step))];
  return out;
}

async function load({ modelBase, graphBase, cloudCount, rasterCount }) {
  let done = 0;
  const report = (bytes, label) => {
    done += bytes;
    self.postMessage({ type: 'progress', bytes: done, label });
  };

  manifest = await (await fetch(`${modelBase}/manifest.json`)).json();
  const config = manifest.config;

  const params = {};
  for (const name of TENSORS) {
    const { buffer } = await fetchTracked(`${modelBase}/${name}.f32`, name, report);
    params[name] = new Float32Array(buffer);
  }

  const [offsets, source, weight, inIndex, soma, valid] = await Promise.all([
    fetchTracked(`${graphBase}/graph/edges_offsets.i32`, 'edges', report),
    fetchTracked(`${graphBase}/graph/edges_source.u16`, 'edges', report),
    fetchTracked(`${graphBase}/graph/edges_weight.f32`, 'synapses', report),
    fetchTracked(`${graphBase}/interface/in_index.i32`, 'interface', report),
    fetchTracked(`${graphBase}/neurons/soma_position.i32`, 'anatomy', report),
    fetchTracked(`${graphBase}/neurons/soma_valid.u8`, 'anatomy', report),
  ]);

  const graph = {
    offsets: new Int32Array(offsets.buffer),
    source: new Uint16Array(source.buffer),
    weight: new Float32Array(weight.buffer),
    inIndex: new Int32Array(inIndex.buffer),
  };
  if (graph.offsets.length !== config.neurons + 1) throw new Error('CSR offsets do not match the model');
  if (graph.source.length !== graph.weight.length) throw new Error('edge arrays disagree');

  const positions = new Int32Array(soma.buffer);
  const somaValid = new Uint8Array(valid.buffer);
  cloudIndex = pickIndices(somaValid, cloudCount, config.neurons);
  rasterIndex = cloudIndex.filter((_value, i) => i % Math.floor(cloudCount / rasterCount) === 0)
    .slice(0, rasterCount);

  const cloudPositions = new Float32Array(cloudIndex.length * 3);
  for (let i = 0; i < cloudIndex.length; i += 1) {
    const at = cloudIndex[i] * 3;
    cloudPositions[i * 3] = positions[at];
    cloudPositions[i * 3 + 1] = positions[at + 1];
    cloudPositions[i * 3 + 2] = positions[at + 2];
  }

  engine = new FlyEngine({ params, graph, config });
  self.postMessage({
    type: 'ready',
    manifest: {
      experiment: manifest.experiment,
      selector: manifest.selector,
      updates: manifest.updates,
      parameters: manifest.trainable_parameters,
      edges: graph.weight.length,
      neurons: config.neurons,
      eos: config.eos_token_id,
      bos: config.bos_token_id,
    },
    cloudPositions,
    rasterCount: rasterIndex.length,
  }, [cloudPositions.buffer]);
}

function sample() {
  const state = engine.state;
  const cloud = new Float32Array(cloudIndex.length);
  for (let i = 0; i < cloudIndex.length; i += 1) cloud[i] = Math.abs(state[cloudIndex[i]]);
  const raster = new Float32Array(rasterIndex.length);
  for (let i = 0; i < rasterIndex.length; i += 1) raster[i] = Math.abs(state[rasterIndex[i]]);
  // tanh saturates, so a peak magnitude is pinned near 1 and says nothing. Count how
  // many neurons are strongly driven instead; that number actually moves.
  let active = 0;
  for (let i = 0; i < state.length; i += 1) {
    const value = state[i] < 0 ? -state[i] : state[i];
    if (value > 0.5) active += 1;
  }
  return { cloud, raster, active };
}

function prime(promptIds) {
  engine.reset();
  generated = 0;
  for (const id of promptIds) logits = engine.step(id);
  const { cloud, raster, active } = sample();
  self.postMessage({ type: 'primed', meanAbs: engine.meanAbsState, active, cloud, raster },
    [cloud.buffer, raster.buffer]);
}

function next({ temperature, topP }) {
  const started = performance.now();
  const probabilities = engine.probabilities(logits);
  const token = engine.pick(logits, { temperature, topP });
  const probability = probabilities[token];
  let best = 0;
  for (let v = 1; v < probabilities.length; v += 1) if (probabilities[v] > probabilities[best]) best = v;
  generated += 1;
  const eos = token === manifest.config.eos_token_id;
  if (!eos) logits = engine.step(token);
  const { cloud, raster, active } = sample();
  self.postMessage({
    type: 'token',
    token,
    eos,
    index: generated,
    probability,
    topProbability: probabilities[best],
    meanAbs: engine.meanAbsState,
    active,
    milliseconds: performance.now() - started,
    cloud,
    raster,
  }, [cloud.buffer, raster.buffer]);
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === 'load') await load(message);
    else if (message.type === 'prime') prime(message.promptIds);
    else if (message.type === 'next') next(message);
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error && error.message ? error.message : error) });
  }
};
