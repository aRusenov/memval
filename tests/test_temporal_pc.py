"""Temporal predictive coding arms: interface conformance, learning, and the
purity contract that `measure_recall_associative` depends on."""

import numpy as np
import pytest

from memval.models.baselines.temporal_pc import (
    TemporalPCNetwork,
    MultilayerTemporalPCNetwork,
)

DIM = 16
HID = 24


def _seq(seed=1, n=6):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n, DIM))
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def make(kind, **over):
    if kind == "tpc1":
        kw = dict(n_features=DIM, learning_rate=0.05, n_epochs=100, seed=0)
        kw.update(over)
        return TemporalPCNetwork(**kw)
    kw = dict(n_features=DIM, n_hidden=HID, learning_rate=0.05, n_epochs=300,
              inf_iters=50, seed=0)
    kw.update(over)
    return MultilayerTemporalPCNetwork(**kw)


ALL = ["tpc1", "tpc2"]


# ----------------------------------------------------------------- interface
@pytest.mark.parametrize("kind", ALL)
def test_predict_next_shape_and_recall_shape(kind):
    m = make(kind)
    seq = _seq()
    m.fit_sequence(seq)
    assert m.predict_next(seq[0]).shape == (DIM,)
    assert m.recall(seq[0], length=4).shape == (4, DIM)
    # 2D prompt: seeds from the last row
    assert m.recall(seq[:3], length=2).shape == (2, DIM)


@pytest.mark.parametrize("kind", ALL)
def test_predict_next_accepts_and_ignores_context(kind):
    m = make(kind)
    m.fit_sequence(_seq())
    x = _seq()[0]
    assert np.allclose(m.predict_next(x), m.predict_next(x, current_context=np.array([1.0])))


@pytest.mark.parametrize("kind", ALL)
def test_short_sequence_is_a_noop_not_a_crash(kind):
    m = make(kind)
    before = {k: v.copy() for k, v in m.named_parameters().items()}
    m.fit_sequence(np.zeros((1, DIM)))
    if kind == "tpc1":
        for k, v in before.items():
            assert np.array_equal(v, m.named_parameters()[k])


# ------------------------------------------------------------------ learning
@pytest.mark.parametrize("kind", ALL)
def test_learns_a_sequence(kind):
    """After training, predict_next(x_t) should be closer to x_{t+1} than to a
    random other item of the sequence."""
    m = make(kind)
    seq = _seq()
    m.fit_sequence(seq)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    hits = 0
    for t in range(len(seq) - 1):
        pred = m.predict_next(seq[t])
        sims = [cos(pred, seq[k]) for k in range(len(seq))]
        if int(np.argmax(sims)) == t + 1:
            hits += 1
    assert hits >= len(seq) - 2, f"only {hits}/{len(seq) - 1} transitions recalled"


@pytest.mark.parametrize("kind", ALL)
def test_online_and_batch_both_learn(kind):
    seq = _seq()
    m = make(kind, n_epochs=1)
    for _ in range(120):
        for v in seq:
            m.fit_event(v)
        m.on_event_boundary()

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    pred = m.predict_next(seq[0])
    assert cos(pred, seq[1]) > cos(pred, seq[3])


def test_tpc1_first_event_only_primes():
    """A lone streamed event forms no transition, so weights must not move."""
    m = make("tpc1")
    W0 = m.W_r.copy()
    m.fit_event(_seq()[0])
    assert np.array_equal(W0, m.W_r)


# -------------------------------------------------------------------- purity
@pytest.mark.parametrize("kind", ALL)
def test_predict_next_is_pure(kind):
    """`measure_recall_associative` probes each position independently with a
    noisy cue. predict_next must therefore not leak state between calls, or
    those probes stop being independent."""
    m = make(kind)
    seq = _seq()
    m.fit_sequence(seq)

    m.reset_context()
    a = m.predict_next(seq[0])
    m.predict_next(seq[3])
    m.predict_next(seq[4])
    b = m.predict_next(seq[0])
    assert np.allclose(a, b), "predict_next leaked state across probes"


@pytest.mark.parametrize("kind", ALL)
def test_predict_next_does_not_change_weights(kind):
    m = make(kind)
    m.fit_sequence(_seq())
    before = {k: v.copy() for k, v in m.named_parameters().items()}
    for _ in range(3):
        m.predict_next(_seq()[2])
    for k, v in before.items():
        assert np.array_equal(v, m.named_parameters()[k])


def test_tpc2_reset_clears_latent():
    m = make("tpc2")
    seq = _seq()
    m.fit_sequence(seq)
    for v in seq:
        m.fit_event(v)
    assert np.linalg.norm(m._prev_z) > 0
    m.reset_context()
    assert np.array_equal(m._prev_z, np.zeros(HID))


# --------------------------------------------------------------- diagnostics
@pytest.mark.parametrize("kind", ALL)
def test_diagnostics_hooks(kind):
    m = make(kind)
    seq = _seq()
    m.fit_sequence(seq)

    reps = m.named_representations(seq[0])
    assert reps["input"].shape == (DIM,)
    assert reps["output"].shape == (DIM,)
    if kind == "tpc2":
        assert reps["hidden"].shape == (HID,)

    params = m.named_parameters()
    grads = m.transition_grads(seq[0], seq[1])
    assert set(grads) == set(params)
    for k in params:
        assert grads[k].shape == params[k].shape


@pytest.mark.parametrize("kind", ALL)
def test_transition_grads_does_not_mutate_weights(kind):
    m = make(kind)
    m.fit_sequence(_seq())
    before = {k: v.copy() for k, v in m.named_parameters().items()}
    m.transition_grads(_seq()[0], _seq()[1])
    for k, v in before.items():
        assert np.array_equal(v, m.named_parameters()[k])


def test_tpc1_linear_matches_whitened_ahn():
    """The paper's theorem: linear single-layer tPC converges to
    W = (Σ x_{t+1} x_tᵀ)(Σ x_t x_tᵀ)⁻¹ -- AHN with the input covariance whitened
    away. Check the trained W reproduces the sequence better than raw AHN does."""
    seq = _seq(seed=3, n=8)
    m = TemporalPCNetwork(n_features=DIM, learning_rate=0.05, n_epochs=3000,
                          nonlinearity="linear", seed=0)
    m.fit_sequence(seq)

    X_prev, X_next = seq[:-1], seq[1:]
    C_cross = X_next.T @ X_prev
    C_auto = X_prev.T @ X_prev
    W_whitened = C_cross @ np.linalg.pinv(C_auto)

    err_tpc = np.linalg.norm(X_next - X_prev @ m.W_r.T)
    err_whitened = np.linalg.norm(X_next - X_prev @ W_whitened.T)
    err_ahn = np.linalg.norm(X_next - X_prev @ C_cross.T)

    assert err_tpc < err_ahn
    assert err_tpc == pytest.approx(err_whitened, abs=0.15)
