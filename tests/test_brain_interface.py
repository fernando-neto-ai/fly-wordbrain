import numpy as np
from fly_wordbrain.brain import WordEncoder, OutputProjection


def test_equal_energy_distinct_fixed_codes():
    a = WordEncoder(20, 200)
    b = WordEncoder(20, 200)
    assert np.array_equal(a.codes, b.codes)
    assert len({row.tobytes() for row in a.codes}) == 20
    assert np.allclose(a.codes.sum(axis=1), a.codes[0].sum())


def test_readout_does_not_observe_unselected_neurons():
    p = OutputProjection([2, 5], bins=8)
    c = np.zeros(10, np.int32)
    v = np.full(10, -52, np.float32)
    initial = p(c, v, 20)
    c[1] = 1000
    v[9] = 1000
    assert np.array_equal(initial, p(c, v, 20))
    c[2] = 1
    assert not np.array_equal(initial, p(c, v, 20))
