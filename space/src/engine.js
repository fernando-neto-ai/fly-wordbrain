// The Fly LLM forward pass, in plain JavaScript.
//
// This is the same computation as the PyTorch model, not a recording and not an
// approximation: one recurrent step per token over all 49,393 neurons and every one
// of the 9,050,172 measured synapses, then a factorized readout.
//
//   drive    = concat over delay slots j of  wte[token(t - j)] @ in_proj[j]
//   x        = (1 - leak) * x + leak * tanh(gain * (rec_gain * (W @ x) + drive) + bias)
//   logits   = layer_norm(x) @ head_a.T @ head_b.T
//
// scripts/export_web_model.py writes a golden trace from a NumPy implementation that
// was itself checked against PyTorch, and space/test/parity.mjs holds this file to it.

export class FlyEngine {
  constructor({ params, graph, config }) {
    this.p = params;
    this.g = graph;
    this.cfg = config;
    const n = config.neurons;
    this.state = new Float32Array(n);
    this.recurrent = new Float32Array(n);
    this.pre = new Float32Array(n);
    this.normed = new Float32Array(n);
    this.drive = new Float32Array(config.delay_k * config.neurons_per_slot);
    this.projected = new Float32Array(config.readout_rank);
    this.logits = new Float32Array(config.vocabulary);
    this.recent = new Int32Array(config.delay_k).fill(config.pad_token_id);
    this.tokens = 0;
  }

  reset() {
    this.state.fill(0);
    this.recent.fill(this.cfg.pad_token_id);
    this.tokens = 0;
  }

  // One token in, logits out.
  step(token) {
    const { cfg, p, g } = this;
    const n = cfg.neurons;
    const k = cfg.delay_k;
    const slot = cfg.neurons_per_slot;
    const width = cfg.d_embed;

    // Delay slot j reads the token j positions back; slot 0 is the incoming token.
    this.drive.fill(0);
    for (let j = 0; j < k; j += 1) {
      const id = j === 0 ? token : this.recent[k - j];
      const embedAt = id * width;
      const projAt = j * width * slot;
      const driveAt = j * slot;
      for (let d = 0; d < width; d += 1) {
        const value = p.wte[embedAt + d];
        if (value === 0) continue;
        const row = projAt + d * slot;
        for (let m = 0; m < slot; m += 1) this.drive[driveAt + m] += value * p.in_proj[row + m];
      }
    }

    // Sparse recurrent mix. Rows are destination neurons; edgeSource holds the
    // presynaptic index. The state vector is 198 KB, so the gather stays in cache.
    const { offsets, source, weight } = g;
    const state = this.state;
    const recurrent = this.recurrent;
    for (let dst = 0; dst < n; dst += 1) {
      let total = 0;
      const end = offsets[dst + 1];
      for (let e = offsets[dst]; e < end; e += 1) total += weight[e] * state[source[e]];
      recurrent[dst] = total;
    }

    const pre = this.pre;
    for (let i = 0; i < n; i += 1) pre[i] = p.rec_gain[i] * recurrent[i];
    // in_index is verified unique at export time, so a plain add is a scatter-add.
    const inIndex = g.inIndex;
    for (let m = 0; m < inIndex.length; m += 1) pre[inIndex[m]] += this.drive[m];

    const leak = cfg.leak;
    const keep = 1 - leak;
    let absolute = 0;
    for (let i = 0; i < n; i += 1) {
      const next = keep * state[i] + leak * Math.tanh(p.gain[i] * pre[i] + p.bias[i]);
      state[i] = next;
      absolute += next < 0 ? -next : next;
    }

    // LayerNorm over every neuron, then the two readout factors.
    let mean = 0;
    for (let i = 0; i < n; i += 1) mean += state[i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i += 1) {
      const d = state[i] - mean;
      variance += d * d;
    }
    variance /= n;
    const inverse = 1 / Math.sqrt(variance + cfg.layer_norm_eps);
    const normed = this.normed;
    for (let i = 0; i < n; i += 1) normed[i] = (state[i] - mean) * inverse * p.ln_weight[i] + p.ln_bias[i];

    const rank = cfg.readout_rank;
    const projected = this.projected;
    for (let r = 0; r < rank; r += 1) {
      let total = 0;
      const row = r * n;
      for (let i = 0; i < n; i += 1) total += normed[i] * p.head_a[row + i];
      projected[r] = total;
    }
    const logits = this.logits;
    for (let v = 0; v < cfg.vocabulary; v += 1) {
      let total = 0;
      const row = v * rank;
      for (let r = 0; r < rank; r += 1) total += projected[r] * p.head_b[row + r];
      logits[v] = total;
    }

    this.recent.copyWithin(0, 1);
    this.recent[k - 1] = token;
    this.tokens += 1;
    this.meanAbsState = absolute / n;
    return logits;
  }

  // Greedy by default; temperature and nucleus sampling are exposed as demo controls.
  pick(logits, { temperature = 0, topP = 1, random = Math.random } = {}) {
    if (temperature <= 0) {
      let best = 0;
      for (let v = 1; v < logits.length; v += 1) if (logits[v] > logits[best]) best = v;
      return best;
    }
    const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
    const top = logits[order[0]];
    const scaled = order.map((v) => Math.exp((logits[v] - top) / temperature));
    const total = scaled.reduce((a, b) => a + b, 0);
    let cumulative = 0;
    const kept = [];
    for (let i = 0; i < order.length; i += 1) {
      kept.push(order[i]);
      cumulative += scaled[i] / total;
      if (cumulative >= topP) break;
    }
    const mass = kept.reduce((a, _v, i) => a + scaled[i], 0);
    let draw = random() * mass;
    for (let i = 0; i < kept.length; i += 1) {
      draw -= scaled[i];
      if (draw <= 0) return kept[i];
    }
    return kept[kept.length - 1];
  }

  probabilities(logits) {
    let top = logits[0];
    for (let v = 1; v < logits.length; v += 1) if (logits[v] > top) top = logits[v];
    let total = 0;
    const out = new Float32Array(logits.length);
    for (let v = 0; v < logits.length; v += 1) {
      out[v] = Math.exp(logits[v] - top);
      total += out[v];
    }
    for (let v = 0; v < logits.length; v += 1) out[v] /= total;
    return out;
  }
}
