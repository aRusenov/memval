"""Timing capabilities are declared, never inferred from a signature."""
import numpy as np
import pytest

from memval.models.baselines import AsymmetricHopfieldNetwork, DTSESNSequenceNetwork
from memval.models.capabilities import (
    TemporallyClocked, TimingPredictive, supports_intervals, predicts_timing)


def test_dts_esn_declares_both_timing_capabilities():
    assert supports_intervals(DTSESNSequenceNetwork)
    assert predicts_timing(DTSESNSequenceNetwork)


def test_ordinal_clocked_arm_declares_neither():
    assert not supports_intervals(AsymmetricHopfieldNetwork)
    assert not predicts_timing(AsymmetricHopfieldNetwork)


def test_swallowing_intervals_is_not_a_capability():
    """The failure mode the nominal check exists to prevent.

    AsymmetricHopfieldNetwork.fit_sequence takes **kwargs, so passing intervals
    raises nothing and changes nothing. A duck-typed check would report the
    capability; the declaration does not.
    """
    m = AsymmetricHopfieldNetwork(n_features=8, learning_rate=0.1)
    X = np.random.default_rng(0).normal(size=(4, 8))
    m.fit_sequence(X, epochs=1, intervals=[0.0, 1.0, 8.0, 1.0])   # accepted...
    before = m.W.copy()
    m2 = AsymmetricHopfieldNetwork(n_features=8, learning_rate=0.1)
    m2.fit_sequence(X, epochs=1)                                   # ...and ignored
    assert np.allclose(before, m2.W), "intervals were silently dropped"
    assert not supports_intervals(m), "so the arm must not claim the capability"


def test_declaring_without_implementing_fails_at_construction():
    class Bad(TemporallyClocked):
        pass

    with pytest.raises(TypeError):
        Bad()


def test_timing_prediction_is_strictly_stronger_than_being_clocked():
    """An arm may absorb elapsed time and still have no read-out that emits a gap."""
    class ClockedOnly(TemporallyClocked):
        def predict_next(self, current_event, current_context=None, elapsed=None, **kw):
            return current_event

    m = ClockedOnly()
    assert supports_intervals(m)
    assert not predicts_timing(m)
