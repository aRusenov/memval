"""Predictive recirculation arm (Chen et al. 2024): interface conformance,
learning, the purity contract, and the Equation-2 identities."""

import numpy as np
import pytest

from memval.models.baselines.predictive_recirculation import PredictiveRecirculationNetwork
from memval.models.capabilities import supports_online, supports_priming, is_online_equivalent

DIM = 16
HID = 24


def _seq(seed=1, n=6):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n, DIM))
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def make(**over):
    kw = dict(n_features=DIM, n_hidden=HID, learning_rate=0.05, n_epochs=300, seed=0)
    kw.update(over)
    return PredictiveRecirculationNetwork(**kw)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ----------------------------------------------------------------- interface
def test_declarations():
    assert supports_online(PredictiveRecirculationNetwork)
    assert supports_priming(PredictiveRecirculationNetwork)
    assert is_online_equivalent(PredictiveRecirculationNetwork)
    assert PredictiveRecirculationNetwork.prompt_conditioned


def test_shapes():
    m = make()
    seq = _seq()
    m.fit_sequence(seq)
    assert m.predict_next(seq[0]).shape == (DIM,)
    assert m.recall(seq[0], length=4).shape == (4, DIM)
    assert m.recall(seq[:3], length=2).shape == (2, DIM)


def test_short_sequence_is_a_noop():
    m = make()
    before = {k: v.copy() for k, v in m.named_parameters().items()}
    m.fit_sequence(np.zeros((1, DIM)))
    for k, v in before.items():
        assert np.array_equal(v, m.named_parameters()[k])


def test_epochs_kwarg_is_honoured():
    a, b = make(n_epochs=5), make(n_epochs=5)
    seq = _seq()
    a.fit_sequence(seq)
    b.fit_sequence(seq, epochs=1)
    assert not np.allclose(a.V, b.V)
    c = make(n_epochs=1)
    c.fit_sequence(seq)
    assert np.allclose(b.V, c.V)


# ------------------------------------------------------------------ learning
def test_learns_a_sequence():
    m = make()
    seq = _seq()
    m.fit_sequence(seq)
    hits = 0
    for t in range(len(seq) - 1):
        pred = m.predict_next(seq[t])
        if int(np.argmax([cos(pred, seq[k]) for k in range(len(seq))])) == t + 1:
            hits += 1
    assert hits >= len(seq) - 2, f"only {hits}/{len(seq) - 1} transitions recalled"


def test_online_and_batch_are_the_same_rule():
    """online_equivalent: one epoch of fit_sequence == streaming the events."""
    seq = _seq()
    a, b = make(n_epochs=1), make(n_epochs=1)
    a.fit_sequence(seq)
    for v in seq:
        b.fit_event(v)
    b.on_event_boundary()
    for k in a.named_parameters():
        assert np.allclose(a.named_parameters()[k], b.named_parameters()[k])


def test_first_event_only_primes():
    m = make()
    before = {k: v.copy() for k, v in m.named_parameters().items()}
    m.fit_event(_seq()[0])
    for k, v in before.items():
        assert np.array_equal(v, m.named_parameters()[k])


def test_recall_completes_the_sequence():
    """The paper's completion protocol: a prefix drives CA3, the rest unrolls."""
    m = make(n_epochs=600)
    seq = _seq(n=5)
    m.fit_sequence(seq)
    out = m.recall(seq[:2], length=2)
    assert cos(out[0], seq[2]) > cos(out[0], seq[4])


# -------------------------------------------------------------------- purity
def test_predict_next_is_pure():
    m = make()
    seq = _seq()
    m.fit_sequence(seq)
    m.reset_context()
    a = m.predict_next(seq[0])
    m.predict_next(seq[3]); m.predict_next(seq[4])
    assert np.allclose(a, m.predict_next(seq[0]))


def test_predict_next_and_observe_do_not_change_weights():
    m = make()
    m.fit_sequence(_seq())
    before = {k: v.copy() for k, v in m.named_parameters().items()}
    for v in _seq():
        m.predict_next(v); m.observe(v)
    for k, v in before.items():
        assert np.array_equal(v, m.named_parameters()[k])


def test_observe_moves_state_and_reset_clears_it():
    m = make()
    m.fit_sequence(_seq())
    m.reset_context()
    assert not np.any(m._h)
    m.observe(_seq()[0])
    assert np.linalg.norm(m._h) > 0
    m.reset_context()
    assert not np.any(m._h) and m._prev_x is None


# ------------------------------------------------------------ Equation 2
def test_dV_is_the_exact_gradient():
    """dV = dx h^T is the exact gradient of 0.5*||x - V h||^2 (STAR Eq. 5)."""
    m = make()
    seq = _seq()
    h = m._advance(np.zeros(HID), seq[0])
    d = m._deltas(np.zeros(HID), seq[0], h, seq[1])
    eps = 1e-6
    num = np.zeros_like(m.V)
    for i in range(3):
        for j in range(3):
            Vp = m.V.copy(); Vp[i, j] += eps
            Vm = m.V.copy(); Vm[i, j] -= eps
            lp = 0.5 * np.sum((seq[1] - Vp @ h) ** 2)
            lm = 0.5 * np.sum((seq[1] - Vm @ h) ** 2)
            num[i, j] = -(lp - lm) / (2 * eps)
    assert np.allclose(d["V"][:3, :3], num[:3, :3], atol=1e-5)


def test_error_is_recirculated_through_U_not_V_transpose():
    """The one place this rule departs from the gradient: dh = U dx, so the
    hidden-layer updates must change with U even when V is held fixed."""
    m = make()
    seq = _seq()
    h_prev = m._advance(np.zeros(HID), seq[0])      # non-zero so dW is live
    h = m._advance(h_prev, seq[1])
    d1 = m._deltas(h_prev, seq[1], h, seq[2])
    m.U = -m.U
    d2 = m._deltas(h_prev, seq[1], h, seq[2])
    assert np.allclose(d1["V"], d2["V"])            # exact gradient, U-free
    assert np.allclose(d1["W"], -d2["W"])           # dh = U dx flips with U
    assert np.allclose(d1["U"], -d2["U"])


# --------------------------------------------------------------- diagnostics
def test_diagnostics_hooks():
    m = make()
    seq = _seq()
    m.fit_sequence(seq)
    reps = m.named_representations(seq[0])
    assert reps["input"].shape == (DIM,) and reps["hidden"].shape == (HID,)
    params = m.named_parameters()
    before = {k: v.copy() for k, v in params.items()}
    grads = m.transition_grads(seq[0], seq[1])
    assert set(grads) == set(params)
    for k in params:
        assert grads[k].shape == params[k].shape
        assert np.array_equal(before[k], m.named_parameters()[k])
