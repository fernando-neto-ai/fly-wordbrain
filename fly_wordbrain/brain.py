"""An input/output adapter; the upstream graph and neural kernel are untouched."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "doomfly"
sys.path.insert(0, str(UPSTREAM))


def array_digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


class WordEncoder:
    """Fixed word-specific patterns, same illuminated receptor count per word.

    This is an artificial sensory code, not rendered text or a learned language
    embedding. Word identity influences only mapped photoreceptor inputs.
    """
    def __init__(self, vocab_size, receptors, seed=1729, high=0.02, fraction=0.25):
        self.vocab_size = int(vocab_size)
        self.receptors = int(receptors)
        self.high = float(high)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.codes = np.zeros((vocab_size, receptors), np.float32)
        count = max(1, int(round(receptors * fraction)))
        for word in range(vocab_size):
            self.codes[word, rng.choice(receptors, count, replace=False)] = high
        self.sha256 = array_digest([self.codes])

    def __call__(self, word_id):
        if not 0 <= int(word_id) < self.vocab_size:
            raise ValueError("Word ID outside the fixed vocabulary")
        return self.codes[int(word_id)]


class OutputProjection:
    """Fixed observation pooling only: no neurons or edges are removed."""
    def __init__(self, nodes, bins=128, seed=741):
        self.nodes = np.asarray(nodes, dtype=np.int32)
        if not len(self.nodes):
            raise ValueError("No descending-neuron readout population")
        self.bins = int(bins)
        rng = np.random.default_rng(seed)
        self.bucket = rng.integers(0, self.bins, size=len(nodes))
        self.sign = rng.choice(np.array([-1., 1.], np.float32), size=len(nodes))
        self.scale = np.sqrt(np.maximum(1, np.bincount(self.bucket, minlength=self.bins)))
        self.sha256 = array_digest([self.nodes, self.bucket, self.sign])

    def pool(self, values):
        return (np.bincount(self.bucket, weights=values * self.sign,
                            minlength=self.bins) / self.scale).astype(np.float32)

    def __call__(self, counts, voltage, word_ms):
        return np.concatenate((self.pool(counts[self.nodes] * (1000. / word_ms)),
                               self.pool(voltage[self.nodes] + 52.))).astype(np.float32)


class FrozenBrain:
    def __init__(self, vocab_size, word_ms=20., warmup_ms=100., bins=128, seed=1729,
                 high=0.02, graph=None):
        from doom.native import NativeBrain
        self.graph_path = Path(graph or UPSTREAM / "outputs/doom/malecns_v1/graph.npz")
        self.native = NativeBrain(self.graph_path)
        b = self.native
        if b.n != 166700 or len(b.post) != 25582938:
            raise ValueError("Expected the complete pinned DOOMFLY graph")
        self.manifest = json.loads(self.graph_path.with_name("manifest.json").read_text())
        if self.manifest["synaptic_contacts"] != 124177617:
            raise ValueError("Synaptic-contact count differs from the pinned release")
        self.word_ms = float(word_ms)
        self.warmup_ms = float(warmup_ms)
        for duration in [self.word_ms, self.warmup_ms]:
            if duration <= 0 or abs(duration / b.dt - round(duration / b.dt)) > 1e-6:
                raise ValueError("Duration must be a positive multiple of original dt=0.1ms")
        self.encoder = WordEncoder(vocab_size, len(b.retina), seed, high)
        nodes = np.flatnonzero(b.superclass == "descending_neuron")
        self.projection = OutputProjection(nodes, bins)
        self.initial_frozen_hash = self.frozen_hash()
        # These const arrays are never updated by the upstream native routine.
        for a in (b.ptr, b.post, b.weight, b.ids):
            a.flags.writeable = False
        self.last_diagnostics = {}
        self.reset()

    def frozen_hash(self):
        b = self.native
        return array_digest([b.ids, b.ptr, b.post, b.weight])

    def verify_frozen(self):
        digest = self.frozen_hash()
        if digest != self.initial_frozen_hash:
            raise RuntimeError("Frozen connectome changed during inference")
        return digest

    def reset(self):
        """Exactly reproduce upstream fresh state, retaining allocated graph."""
        b = self.native
        b.cursor = 0
        b.v.fill(-52)
        for name in ["g", "drive", "refractory", "queue", "queue_count", "counts",
                     "luminance", "active", "active_flag", "previous_drive"]:
            getattr(b, name).fill(0)
        b.last.fill(-1)
        initial = np.unique(np.r_[b.retina, b.lamina, b.sugar])
        b.active[:len(initial)] = initial
        b.active_flag[initial] = 1
        b.nactive[0] = len(initial)
        b.total_spikes = 0
        b.sim_ms = 0
        b.step(np.zeros(len(b.retina), np.float32), self.warmup_ms, sugar=False,
               lamina_bias=12.)

    def step(self, word_id, disconnected=False):
        stimulus = self.encoder(word_id)
        if disconnected:
            stimulus = np.zeros_like(stimulus)
        counts, kernel_seconds = self.native.step(stimulus, self.word_ms,
                                                  sugar=False, lamina_bias=12.)
        x = self.projection(counts, self.native.v, self.word_ms)
        if not np.all(np.isfinite(x)):
            raise RuntimeError("Non-finite neural output")
        idx = self.projection.nodes
        self.last_diagnostics = {"kernel_seconds": kernel_seconds,
            "all_spikes": int(counts.sum()), "readout_spikes": int(counts[idx].sum()),
            "readout_voltage_std": float(self.native.v[idx].std()),
            "feature_norm": float(np.linalg.norm(x))}
        return x

    def metadata(self):
        from doom.native import BUILD
        return {"upstream_commit": "71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33",
            "upstream_python_sha256": {f: hashlib.sha256((UPSTREAM / "doom" / f).read_bytes()).hexdigest()
                                       for f in ["native.py", "engine.py", "transmitters.py"]},
            "sensory_arrays_sha256": array_digest([self.native.retina, self.native.uv,
                                                     self.native.lamina, self.native.sugar]),
            "superclasses_sha256": array_digest([self.native.superclass]),
            "neurons": self.native.n, "edges": len(self.native.post),
            "synaptic_contacts": self.manifest["synaptic_contacts"],
            "graph_sha256": self.initial_frozen_hash, "dt_ms": self.native.dt,
            "word_ms": self.word_ms, "warmup_ms": self.warmup_ms,
            "retinal_inputs": len(self.native.retina),
            "readout_population": "all descending_neuron annotations",
            "readout_neurons": len(self.projection.nodes),
            "feature_dimensions": self.projection.bins * 2,
            "features": "fixed pooled spike rates and end-of-word voltages",
            "encoder_sha256": self.encoder.sha256,
            "projection_sha256": self.projection.sha256,
            "encoder_seed": self.encoder.seed, "encoder_high": self.encoder.high,
            "kernel": BUILD, "plasticity": False, "lamina_bias": 12.,
            "decoder_temporal_memory": False}
