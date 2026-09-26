"""The columnar codec's contract: a seeded, sign-preserving interval code whose
decoder inverts it well enough to identify the item, and which the spiking
BCPNN arm can be run behind through the ordinary vector interface."""
import numpy as np
import pytest

from memval.encoders.columnar_codec import (
    ColumnarSpikeDecoder, ColumnarSpikeEncoder, make_columnar_codec,
)
from memval.models.baselines.bcpnn_spiking import CodecBCPNNNetwork
from memval.models.capabilities import supports_online


def _items(n, D=40, seed=0):
    V = np.random.default_rng(seed).normal(size=(n, D))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def test_layout_is_the_arms_hypercolumn_minicolumn_cell_order():
    enc = ColumnarSpikeEncoder(4, n_mc=3, n_per_mc=2, seed=0)
    assert enc.n_neurons == 4 * 3 * 2
    cells = enc.cells(np.array([0, 2, 1, 0]))
    # (hc, mc, k) -> (hc * n_mc + mc) * n_per_mc + k
    assert cells.tolist() == [0, 1, 10, 11, 14, 15, 18, 19]


def test_exactly_one_minicolumn_per_hypercolumn_fires():
    enc = ColumnarSpikeEncoder(40, seed=0)
    S = enc.encode(_items(1)[0])
    counts = S.sum(axis=1).reshape(enc.n_hc, enc.n_mc, enc.n_per_mc).sum(axis=2)
    assert np.all((counts > 0).sum(axis=1) == 1)
    active = counts[counts > 0]
    assert 0.85 * enc.spikes_per_unit < active.mean() < 1.15 * enc.spikes_per_unit


def test_bins_are_monotone_in_the_value_and_clip_at_the_range():
    enc = ColumnarSpikeEncoder(1, n_mc=10, z_range=2.5)
    v = np.linspace(-3.0, 3.0, 61)          # D = 1: scale 1, so value == z; beyond +-2.5 clips
    b = np.array([enc.bins(np.array([x]))[0] for x in v])
    assert np.all(np.diff(b) >= 0) and b[0] == 0 and b[-1] == 9
    assert enc.bins(np.array([-10.0]))[0] == 0 and enc.bins(np.array([10.0]))[0] == 9


def test_seeded_and_private_rng():
    np.random.seed(5)
    before = np.random.get_state()[1].copy()
    a = ColumnarSpikeEncoder(40, seed=3).encode(_items(1)[0])
    b = ColumnarSpikeEncoder(40, seed=3).encode(_items(1)[0])
    assert np.array_equal(a, b)
    assert np.array_equal(np.random.get_state()[1], before)


def test_round_trip_identifies_the_item_and_preserves_sign():
    enc, dec = make_columnar_codec(40, seed=0)
    V = _items(12)
    for i, v in enumerate(V):
        y = dec.decode(enc.encode(v))
        assert abs(np.linalg.norm(y) - 1.0) < 1e-9
        assert (V @ y).argmax() == i
        assert np.mean(np.sign(y) == np.sign(v)) > 0.8


def test_decoder_returns_zeros_when_nothing_fired():
    enc, dec = make_columnar_codec(8, seed=0)
    assert not dec.decode(np.zeros((enc.n_neurons, 10), dtype=bool)).any()


def test_arm_runs_behind_the_columnar_codec_and_stays_pure():
    D = 6
    m = CodecBCPNNNetwork(n_features=D, codec="columnar", n_mc=4, n_per_mc=2,
                          window_steps=20, n_epochs=2, recall_steps=20, seed=0)
    assert m.inner.n_hc == D and m.inner.n_mc == 4 and m.inner.cue_repeats == 1
    assert isinstance(m.decoder, ColumnarSpikeDecoder) and supports_online(m)
    V = _items(3, D=D)
    m.fit_sequence(V, epochs=2)
    y = m.predict_next(V[0])
    assert y.shape == (D,) and np.array_equal(y, m.predict_next(V[0]))


def test_bad_codec_name_is_rejected():
    with pytest.raises(ValueError):
        CodecBCPNNNetwork(n_features=4, codec="grid")
