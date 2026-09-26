# -*- coding: utf8 -*-
"""Phase 1 gate for the CA3 arm's spike encoder/decoder.

The adapter is pure NumPy by design (no Brian2), so this suite runs without the
optional `brian2` extra installed. See docs/ca3net_port_plan.md Phase 1.
"""

import numpy as np
import pytest

from memval.models.baselines._ca3net_io import (
    Ca3netIO, block_assignment, cosine, EVENT_MS_DEFAULT,
)

ROUND_TRIP_GATE = 0.95


# ------------------------------------------------------------------ blocks
def test_block_assignment_covers_every_cell_evenly():
    blocks = block_assignment(8000, 400)
    assert blocks.shape == (8000,)
    sizes = np.bincount(blocks)
    assert sizes.size == 400
    assert sizes.sum() == 8000
    assert sizes.max() - sizes.min() == 0  # 8000 / 400 divides exactly


def test_block_assignment_handles_indivisible_dim():
    blocks = block_assignment(8000, 300)   # 26.67 cells per block
    sizes = np.bincount(blocks)
    assert sizes.sum() == 8000
    assert sizes.max() - sizes.min() <= 1


def test_block_assignment_rejects_degenerate_dims():
    with pytest.raises(ValueError):
        block_assignment(8000, 0)
    with pytest.raises(ValueError):
        block_assignment(10, 20)          # would leave empty blocks


# ------------------------------------------------------------- round trip
def _spatial_events(n=8, n_cells_per_dim=20):
    from memval.encoders.spatial import PlaceCellEncoder
    enc = PlaceCellEncoder(n_cells_per_dim=n_cells_per_dim, seed=0)
    traj = np.stack([np.linspace(-0.8, 0.8, n), 0.3 * np.sin(np.linspace(0, 3, n))], axis=1)
    return enc.encode(traj)


def test_round_trip_meets_gate_on_spatial_items():
    """The Phase 1 gate: encode -> decode with no network in between."""
    events = _spatial_events()
    io = Ca3netIO(dim=events.shape[1], seed=1)
    recovered = io.round_trip(events)
    cosines = [cosine(events[i], recovered[i]) for i in range(len(events))]
    assert min(cosines) > ROUND_TRIP_GATE, "worst-case cosine %.4f" % min(cosines)


def test_round_trip_meets_gate_on_sparse_items():
    """Sparse, near-orthogonal vectors -- the regime symbolic encoders produce."""
    rng = np.random.default_rng(0)
    dim, n_active = 400, 40
    events = np.zeros((6, dim))
    for i in range(6):
        events[i, rng.choice(dim, n_active, replace=False)] = 1.0
    io = Ca3netIO(dim=dim, seed=2)
    recovered = io.round_trip(events)
    cosines = [cosine(events[i], recovered[i]) for i in range(len(events))]
    assert min(cosines) > ROUND_TRIP_GATE, "worst-case cosine %.4f" % min(cosines)


def test_fidelity_increases_with_window_width():
    """Guards the documented event_ms table: more spikes -> better decode."""
    events = _spatial_events(n=4)
    scores = []
    for ms in (25.0, 100.0, 400.0):
        io = Ca3netIO(dim=events.shape[1], event_ms=ms, seed=3)
        rec = io.round_trip(events)
        scores.append(np.mean([cosine(events[i], rec[i]) for i in range(len(events))]))
    assert scores[0] < scores[1] < scores[2]
    assert scores[0] < ROUND_TRIP_GATE < scores[1]   # 25 ms is why the default moved


# ---------------------------------------------------------------- encoding
def test_encode_is_deterministic_under_seed():
    events = _spatial_events(n=3)
    a = Ca3netIO(dim=events.shape[1], seed=7).encode_events(events)
    b = Ca3netIO(dim=events.shape[1], seed=7).encode_events(events)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_encode_rejects_wrong_dimensionality():
    io = Ca3netIO(dim=400, seed=0)
    with pytest.raises(ValueError):
        io.encode_events(np.zeros((2, 399)))


def test_spikes_stay_inside_the_presentation_window():
    events = _spatial_events(n=3)
    io = Ca3netIO(dim=events.shape[1], seed=0)
    st, _ = io.encode_events(events)
    assert st.min() >= 0.0
    assert st.max() < 3 * EVENT_MS_DEFAULT
    assert np.all(np.diff(st) >= 0), "spike times must be returned sorted"


def test_intervals_lengthen_the_train_and_are_quiet():
    """A blank gap is endured at baseline rate -- the metric-time hook."""
    events = _spatial_events(n=2)
    io = Ca3netIO(dim=events.shape[1], seed=0)
    gap = 500.0
    st, ni = io.encode_events(events, intervals_ms=[gap, 0.0])
    assert st.max() < 2 * EVENT_MS_DEFAULT + gap

    # the gap window carries only out-of-field firing
    in_gap = (st >= EVENT_MS_DEFAULT) & (st < EVENT_MS_DEFAULT + gap)
    expected = io.n_cells * io.outfield_rate * (gap * 1e-3)
    assert in_gap.sum() < 3 * expected


# ---------------------------------------------------------------- decoding
def test_decode_of_empty_window_is_zero():
    events = _spatial_events(n=2)
    io = Ca3netIO(dim=events.shape[1], seed=0)
    st, ni = io.encode_events(events)
    out = io.decode_window(st, ni, 10_000.0, 10_100.0)
    assert out.shape == (events.shape[1],)
    assert np.allclose(out, 0.0)


def test_decode_separates_two_different_events():
    """Adjacent windows must decode to their own event, not to each other."""
    rng = np.random.default_rng(1)
    dim = 400
    a = np.zeros(dim); a[rng.choice(dim, 40, replace=False)] = 1.0
    b = np.zeros(dim); b[rng.choice(dim, 40, replace=False)] = 1.0
    io = Ca3netIO(dim=dim, seed=4)
    st, ni = io.encode_events(np.stack([a, b]))
    dec_a = io.decode_window(st, ni, 0.0, EVENT_MS_DEFAULT)
    dec_b = io.decode_window(st, ni, EVENT_MS_DEFAULT, 2 * EVENT_MS_DEFAULT)
    assert cosine(a, dec_a) > cosine(b, dec_a)
    assert cosine(b, dec_b) > cosine(a, dec_b)


def test_baseline_subtraction_suppresses_silent_blocks():
    dim = 400
    v = np.zeros(dim); v[:10] = 1.0
    io = Ca3netIO(dim=dim, seed=5)
    st, ni = io.encode_events(v[None, :])
    with_sub = io.decode_window(st, ni, 0.0, EVENT_MS_DEFAULT, subtract_baseline=True)
    without = io.decode_window(st, ni, 0.0, EVENT_MS_DEFAULT, subtract_baseline=False)
    assert with_sub[10:].mean() < without[10:].mean()
    assert cosine(v, with_sub) >= cosine(v, without)


# ------------------------------------------------------------- refractoriness
def test_no_neuron_spikes_twice_within_the_refractory_period():
    """Brian2's SpikeGeneratorGroup raises outright on sub-timestep doublets, so
    this is a hard requirement, not a nicety. Regression for the failure that
    surfaced when Phase 2 first fed a train to the vendored learning stage."""
    io = Ca3netIO(dim=400, seed=0)
    st, ni = io.encode_events(np.ones((5, 400)))
    order = np.lexsort((st, ni))
    t, i = st[order], ni[order]
    same = np.zeros(t.size, dtype=bool)
    same[1:] = i[1:] == i[:-1]
    gap = np.full(t.size, np.inf)
    gap[1:] = t[1:] - t[:-1]
    assert gap[same].min() >= io.refractory_ms - 1e-9


def test_refractory_can_be_disabled():
    io = Ca3netIO(dim=50, refractory_ms=0.0, seed=0)
    st, _ = io.encode_events(np.ones((2, 50)))
    assert st.size > 0   # simply must not raise or empty the train
