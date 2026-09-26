"""A criterion probe must give the arm the kind of input it trained on."""
import numpy as np
import pytest

from memval.benchmarks.spatial_pipeline import probe_next_from_history
from memval.models.baselines import (AsymmetricHopfieldNetwork, DTSESNSequenceNetwork,
                                     MultilayerTemporalPCNetwork)
from memval.models.capabilities import supports_priming


def _route(n=12, d=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def test_dts_esn_probe_is_warm_after_observe_and_cold_otherwise():
    """The StatePrimeable contract, restored: observe moves state and a following
    predict_next reads it; with no observe since reset, predict_next is from rest."""
    X = _route()
    m = DTSESNSequenceNetwork(n_features=X.shape[1], n_units=200, seed=1)
    m.fit_sequence(X, epochs=3)
    t = 6
    m.reset_context(); cold = m.predict_next(X[t])
    m.reset_context(); cold_again = m.predict_next(X[t])
    assert np.allclose(cold, cold_again), "un-primed probes must stay pure"
    m.reset_context(); m.observe_sequence(X[:t]); warm = m.predict_next(X[t])
    assert not np.allclose(cold, warm), "observe must change what predict_next reads"
    m.reset_context(); after_reset = m.predict_next(X[t])
    assert np.allclose(cold, after_reset), "reset_context must clear the prime"


def test_dts_esn_warm_probe_is_closer_to_the_true_next_item():
    X = _route()
    m = DTSESNSequenceNetwork(n_features=X.shape[1], n_units=300, seed=1)
    m.fit_sequence(X, epochs=5)
    def err(pred, t): return np.linalg.norm(pred - X[t + 1])
    ts = range(3, 11)
    m_cold = np.mean([err((m.reset_context(), m.predict_next(X[t]))[1], t) for t in ts])
    m_warm = np.mean([err(probe_next_from_history(m, X, t), t) for t in ts])
    assert m_warm < m_cold, f"warm {m_warm:.3f} should beat cold {m_cold:.3f}"


def test_temporal_pc_takes_the_same_declared_path():
    """TemporalPC already read observed latent; the helper must go through it, not
    hand it a trajectory (its predict_next ravels 2-D input)."""
    X = _route(d=16)
    m = MultilayerTemporalPCNetwork(n_features=16, n_hidden=24, learning_rate=0.05,
                                    n_epochs=1, inf_iters=10, seed=1)
    m.fit_sequence(X, epochs=1)
    assert supports_priming(m)
    out = probe_next_from_history(m, X, 5)
    assert out.shape == (16,), "helper must hand the arm a 1-D cue"


def test_ordinal_arm_gets_a_bare_cue_and_is_unchanged():
    """AHN's predict_next reshapes its input to a column; a trajectory would break
    it. The helper must never route a non-primeable arm through observe."""
    X = _route(d=16)
    m = AsymmetricHopfieldNetwork(n_features=16, learning_rate=0.1)
    m.fit_sequence(X, epochs=1)
    assert not supports_priming(m)
    direct = m.predict_next(X[5])
    via = probe_next_from_history(m, X, 5)
    assert np.allclose(direct, via)
