"""Ordered two-word sensory codes around the unchanged original frozen brain."""
import numpy as np
from .brain import FrozenBrain, OutputProjection, array_digest


class OrderedPairEncoder:
    """Two fixed receptor roles; no learned embeddings or atomic-pair vocabulary."""
    def __init__(self, vocab_size, receptors, seed=1729, high=.02):
        self.vocab_size, self.receptors = int(vocab_size), int(receptors)
        self.seed, self.high = int(seed), float(high)
        if self.vocab_size < 4 or self.receptors < 8 or self.high <= 0:
            raise ValueError('Require reserved vocabulary, at least 8 receptors, positive drive')
        order = np.random.default_rng(seed + 1907).permutation(receptors).astype(np.int32)
        self.previous_bank = order[:receptors // 2]
        self.current_bank = order[receptors // 2:]
        total_lit = max(2, int(round(receptors * .25)))
        self.lit_counts = (total_lit // 2, total_lit - total_lit // 2)
        codes = []
        for bank, count, code_seed in zip((self.previous_bank, self.current_bank),
                                          self.lit_counts, (seed, seed + 1)):
            rng = np.random.default_rng(code_seed)
            code = np.zeros((vocab_size, len(bank)), np.float32)
            for word in range(vocab_size):
                code[word, rng.choice(len(bank), count, replace=False)] = high
            code.flags.writeable = False
            codes.append(code)
        self.previous_codes, self.current_codes = codes
        self.sha256 = array_digest([self.previous_bank, self.current_bank, *codes])
        self.previous_bank.flags.writeable = self.current_bank.flags.writeable = False

    def __call__(self, pair):
        if len(pair) != 2:
            raise ValueError('Require exactly (previous_word_id, current_word_id)')
        previous, current = map(int, pair)
        if not (0 <= previous < self.vocab_size and 0 <= current < self.vocab_size):
            raise ValueError('Word ID outside fixed vocabulary')
        stimulus = np.zeros(self.receptors, np.float32)
        stimulus[self.previous_bank] = self.previous_codes[previous]
        stimulus[self.current_bank] = self.current_codes[current]
        return stimulus

    def metadata(self):
        return {'type': 'fixed_ordered_disjoint_receptor_roles', 'version': 1,
                'seed': self.seed, 'partition_seed': self.seed + 1907,
                'role_code_seeds': [self.seed, self.seed + 1],
                'bank_sizes': [len(self.previous_bank), len(self.current_bank)],
                'illuminated_per_role': list(self.lit_counts), 'high': self.high,
                'sha256': self.sha256, 'learned': False}


class DirectProjection:
    """Memoryless fixed CountSketch of the exact retinal stimulus, for control."""
    def __init__(self, receptors, bins=256, seed=3187):
        self.projector = OutputProjection(np.arange(receptors, dtype=np.int32), bins, seed)
        self.receptors, self.bins, self.seed = int(receptors), int(bins), int(seed)
        self.sha256 = self.projector.sha256

    def __call__(self, stimulus):
        stimulus = np.asarray(stimulus, np.float32)
        if stimulus.shape != (self.receptors,) or not np.isfinite(stimulus).all():
            raise ValueError('Direct control requires one finite complete retinal stimulus')
        return self.projector.pool(stimulus)

    def metadata(self):
        return {'type': 'signed_countsketch_of_current_retinal_stimulus',
                'dimensions': self.bins, 'seed': self.seed, 'sha256': self.sha256,
                'temporal_memory': False, 'learned': False}


class PairedFrozenBrain(FrozenBrain):
    def __init__(self, vocab_size, word_ms=20., warmup_ms=100., bins=128,
                 seed=1729, high=.02, graph=None):
        super().__init__(vocab_size, word_ms, warmup_ms, bins, seed, high, graph)
        self.encoder = OrderedPairEncoder(vocab_size, len(self.native.retina), seed, high)
        self.direct_projection = DirectProjection(len(self.native.retina), 2 * bins)

    def direct_features(self, pair):
        return self.direct_projection(self.encoder(pair))

    def metadata(self):
        result = super().metadata()
        result.update(input_encoding=self.encoder.metadata(),
                      external_input_context_words=2,
                      advancement='one new word per neural interval; ordered previous/current code',
                      direct_control=self.direct_projection.metadata())
        return result
