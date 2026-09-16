// The instrument console: everything drawn here is a measurement from the running
// model. Nothing is decorative noise and nothing is pre-recorded.

const dpr = () => Math.min(window.devicePixelRatio || 1, 2);

function fit(canvas) {
  const ratio = dpr();
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = Math.max(1, Math.round(width * ratio));
    canvas.height = Math.max(1, Math.round(height * ratio));
  }
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

/** Measured soma positions, lit by |h|. Rates, not spikes. */
export class NeuronCloud {
  constructor(canvas) {
    this.canvas = canvas;
    this.points = null;
    this.values = null;
  }

  setPositions(flat) {
    // Project the two widest anatomical axes so the familiar brain outline shows.
    const count = flat.length / 3;
    let minX = Infinity; let maxX = -Infinity;
    let minY = Infinity; let maxY = -Infinity;
    for (let i = 0; i < count; i += 1) {
      const x = flat[i * 3];
      const y = flat[i * 3 + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const spanX = Math.max(1, maxX - minX);
    const spanY = Math.max(1, maxY - minY);
    this.aspect = spanX / spanY;
    this.points = new Float32Array(count * 2);
    for (let i = 0; i < count; i += 1) {
      this.points[i * 2] = (flat[i * 3] - minX) / spanX;
      // MaleCNS y grows downward relative to the usual viewing orientation.
      this.points[i * 2 + 1] = 1 - (flat[i * 3 + 1] - minY) / spanY;
    }
    this.values = new Float32Array(count);
  }

  update(values) { this.values = values; }

  draw() {
    const { context, width, height } = fit(this.canvas);
    context.clearRect(0, 0, width, height);
    if (!this.points) return;
    const count = this.points.length / 2;
    const scale = Math.min(width / this.aspect, height) * 0.94;
    const boxWidth = scale * this.aspect;
    const offsetX = (width - boxWidth) / 2;
    const offsetY = (height - scale) / 2;
    let peak = 1e-6;
    for (let i = 0; i < count; i += 1) if (this.values[i] > peak) peak = this.values[i];
    for (let i = 0; i < count; i += 1) {
      const x = offsetX + this.points[i * 2] * boxWidth;
      const y = offsetY + this.points[i * 2 + 1] * scale;
      // A fixed square-root display curve, the same convention the recorded demo used.
      const level = Math.sqrt(Math.min(1, this.values[i] / peak));
      if (level < 0.06) {
        context.fillStyle = 'rgba(70,104,116,0.34)';
        context.fillRect(x, y, 1.1, 1.1);
        continue;
      }
      const alpha = 0.22 + level * 0.78;
      context.fillStyle = level > 0.72
        ? `rgba(233,250,255,${alpha})`
        : `rgba(${Math.round(90 + level * 150)},${Math.round(200 + level * 45)},${Math.round(215 + level * 35)},${alpha})`;
      const size = 1.1 + level * 2.2;
      context.fillRect(x - size / 2, y - size / 2, size, size);
    }
  }
}

/** Sampled neuron magnitudes over recent tokens, newest column on the right. */
export class StateRaster {
  constructor(canvas, columns = 96) {
    this.canvas = canvas;
    this.columns = columns;
    this.rows = 0;
    this.buffer = [];
  }

  setRows(rows) {
    this.rows = rows;
    this.buffer = [];
  }

  push(values) {
    this.buffer.push(values);
    if (this.buffer.length > this.columns) this.buffer.shift();
  }

  clear() { this.buffer = []; }

  draw() {
    const { context, width, height } = fit(this.canvas);
    context.clearRect(0, 0, width, height);
    if (!this.rows || !this.buffer.length) return;
    const columnWidth = width / this.columns;
    const rowHeight = height / this.rows;
    let peak = 1e-6;
    for (const column of this.buffer) {
      for (let r = 0; r < column.length; r += 1) if (column[r] > peak) peak = column[r];
    }
    const start = this.columns - this.buffer.length;
    for (let c = 0; c < this.buffer.length; c += 1) {
      const column = this.buffer[c];
      const x = (start + c) * columnWidth;
      for (let r = 0; r < column.length; r += 1) {
        const level = Math.sqrt(Math.min(1, column[r] / peak));
        if (level < 0.05) continue;
        context.fillStyle = `rgba(232,183,106,${0.1 + level * 0.9})`;
        context.fillRect(x, r * rowHeight, Math.max(1, columnWidth - 0.35), Math.max(0.7, rowHeight - 0.25));
      }
    }
  }
}

/** Per-token top-1 probability, drawn like a deck's loaded waveform. */
export class ProbabilityWave {
  constructor(canvas, capacity = 120) {
    this.canvas = canvas;
    this.capacity = capacity;
    this.values = [];
  }

  push(value) {
    this.values.push(value);
    if (this.values.length > this.capacity) this.values.shift();
  }

  clear() { this.values = []; }

  draw() {
    const { context, width, height } = fit(this.canvas);
    context.clearRect(0, 0, width, height);
    const middle = height / 2;
    context.fillStyle = 'rgba(255,255,255,0.05)';
    context.fillRect(0, middle - 0.5, width, 1);
    if (!this.values.length) return;
    const barWidth = width / this.capacity;
    const start = this.capacity - this.values.length;
    for (let i = 0; i < this.values.length; i += 1) {
      const value = this.values[i];
      const half = Math.max(1, value * (height / 2 - 2));
      const x = (start + i) * barWidth;
      // Confident tokens read violet; uncertain ones flare magenta.
      context.fillStyle = value > 0.66
        ? `rgba(143,123,240,${0.55 + value * 0.45})`
        : `rgba(226,88,156,${0.5 + value * 0.5})`;
      context.fillRect(x, middle - half, Math.max(1, barWidth - 0.6), half * 2);
    }
  }
}

/** A draggable EQ-style knob. These control the sampler for real. */
export class Knob {
  constructor(element, { min, max, value, format, onChange }) {
    this.element = element;
    this.canvas = element.querySelector('canvas');
    this.readout = element.querySelector('b');
    this.min = min;
    this.max = max;
    this.value = value;
    this.format = format;
    this.onChange = onChange;
    this.accent = element.dataset.knob === 'pace' ? '#4fd1b5' : '#e8b76a';
    this.#bind();
    this.draw();
  }

  #bind() {
    let startY = 0;
    let startValue = 0;
    let active = false;
    const down = (event) => {
      active = true;
      startY = (event.touches ? event.touches[0] : event).clientY;
      startValue = this.value;
      this.element.setPointerCapture?.(event.pointerId);
      event.preventDefault();
    };
    const move = (event) => {
      if (!active) return;
      const y = (event.touches ? event.touches[0] : event).clientY;
      const span = this.max - this.min;
      this.set(startValue + ((startY - y) / 120) * span);
      event.preventDefault();
    };
    const up = () => { active = false; };
    this.element.addEventListener('pointerdown', down);
    window.addEventListener('pointermove', move, { passive: false });
    window.addEventListener('pointerup', up);
    this.element.addEventListener('dblclick', () => this.set(this.defaultValue ?? this.value));
  }

  set(value) {
    this.value = Math.min(this.max, Math.max(this.min, value));
    this.draw();
    this.onChange?.(this.value);
  }

  draw() {
    const { context, width, height } = fit(this.canvas);
    context.clearRect(0, 0, width, height);
    const cx = width / 2;
    const cy = height / 2 + 2;
    const radius = Math.min(width, height) / 2 - 4;
    const from = Math.PI * 0.75;
    const to = Math.PI * 2.25;
    const fraction = (this.value - this.min) / (this.max - this.min);
    context.lineWidth = 2.5;
    context.lineCap = 'round';
    context.strokeStyle = 'rgba(255,255,255,0.09)';
    context.beginPath();
    context.arc(cx, cy, radius, from, to);
    context.stroke();
    context.strokeStyle = this.accent;
    context.beginPath();
    context.arc(cx, cy, radius, from, from + (to - from) * fraction);
    context.stroke();
    const angle = from + (to - from) * fraction;
    context.beginPath();
    context.moveTo(cx + Math.cos(angle) * (radius - 7), cy + Math.sin(angle) * (radius - 7));
    context.lineTo(cx + Math.cos(angle) * (radius - 1), cy + Math.sin(angle) * (radius - 1));
    context.stroke();
    if (this.readout) this.readout.textContent = this.format(this.value);
  }
}
