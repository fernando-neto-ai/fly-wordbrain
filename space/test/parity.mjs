// Hold the browser engine to the exported golden trace.
//
// The golden trace comes from a NumPy implementation that was itself checked against
// PyTorch token-for-token, and it uses the same exact float32 edges the browser
// downloads. So any mismatch here is a defect in engine.js, not a precision artifact.
//
//   node space/test/parity.mjs <model-dir> <connectome-dir> [tokenizer.json]

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { FlyEngine } from '../src/engine.js';
import { FlyTokenizer } from '../src/tokenizer.js';

const [modelDir, graphDir, tokenizerPath] = process.argv.slice(2);
if (!modelDir || !graphDir) {
  console.error('usage: node space/test/parity.mjs <model-dir> <connectome-dir> [tokenizer.json]');
  process.exit(2);
}

const f32 = (path) => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};
const i32 = (path) => {
  const buffer = readFileSync(path);
  return new Int32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};
const u16 = (path) => {
  const buffer = readFileSync(path);
  return new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
};

const manifest = JSON.parse(readFileSync(join(modelDir, 'manifest.json'), 'utf8'));
const config = manifest.config;
const params = {};
for (const key of Object.keys(manifest.files)) params[key] = f32(join(modelDir, `${key}.f32`));

const graph = {
  offsets: i32(join(graphDir, 'graph/edges_offsets.i32')),
  source: u16(join(graphDir, 'graph/edges_source.u16')),
  weight: f32(join(graphDir, 'graph/edges_weight.f32')),
  inIndex: i32(join(graphDir, 'interface/in_index.i32')),
};

console.log(`model      ${manifest.experiment} / ${manifest.selector} @ ${manifest.updates} updates`);
console.log(`parameters ${manifest.trainable_parameters.toLocaleString()}`);
console.log(`graph      ${graph.weight.length.toLocaleString()} edges, ${config.neurons.toLocaleString()} neurons`);

if (graph.offsets.length !== config.neurons + 1) throw new Error('CSR offsets do not match neuron count');
if (graph.source.length !== graph.weight.length) throw new Error('edge arrays disagree');

const engine = new FlyEngine({ params, graph, config });
let failures = 0;
let checkedSteps = 0;
let worstProbability = 0;
let worstState = 0;
const started = Date.now();
let totalTokens = 0;

for (const entry of manifest.golden) {
  engine.reset();
  let logits = null;
  for (const id of entry.prompt_ids) logits = engine.step(id);
  const produced = [];
  const problems = [];
  for (const expected of entry.steps) {
    const probabilities = engine.probabilities(logits);
    const token = engine.pick(logits);
    produced.push(token);
    totalTokens += 1;
    if (token !== expected.token) {
      problems.push(`step ${produced.length}: got ${token}, expected ${expected.token}`);
      break;
    }
    const dp = Math.abs(probabilities[token] - expected.top1_probability);
    const ds = Math.abs(engine.meanAbsState - expected.mean_abs_state);
    worstProbability = Math.max(worstProbability, dp);
    worstState = Math.max(worstState, ds);
    checkedSteps += 1;
    if (token === config.eos_token_id) break;
    logits = engine.step(token);
  }
  const identical = produced.length === entry.tokens.length
    && produced.every((value, index) => value === entry.tokens[index]);
  console.log(`\n  ${identical && !problems.length ? 'PASS' : 'FAIL'}  ${entry.prompt}`);
  if (!identical || problems.length) {
    failures += 1;
    problems.forEach((line) => console.log(`        ${line}`));
    console.log(`        expected ${entry.tokens.length} tokens, produced ${produced.length}`);
  }
}

// Tolerances are for float32 summation order, not for a different computation:
// any changed token would already have failed above.
const PROBABILITY_TOLERANCE = 2e-4;
const STATE_TOLERANCE = 2e-5;
console.log(`\n  checked ${checkedSteps} steps across ${manifest.golden.length} prompts`);
console.log(`  worst top-1 probability difference  ${worstProbability.toExponential(3)} (tolerance ${PROBABILITY_TOLERANCE})`);
console.log(`  worst mean |h| difference           ${worstState.toExponential(3)} (tolerance ${STATE_TOLERANCE})`);
console.log(`  throughput ${(totalTokens / ((Date.now() - started) / 1000)).toFixed(1)} tokens/s single-threaded`);

if (worstProbability > PROBABILITY_TOLERANCE) {
  console.log('  FAIL  top-1 probability drifted beyond float32 summation order');
  failures += 1;
}
if (worstState > STATE_TOLERANCE) {
  console.log('  FAIL  mean |h| drifted beyond float32 summation order');
  failures += 1;
}
console.log(failures ? `\n${failures} failure(s).` : '\nAll prompts reproduce the golden trace exactly.');
process.exit(failures ? 1 : 0);
