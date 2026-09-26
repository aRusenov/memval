"""Tests for the efficiency index (McClelland 2013)."""
import numpy as np
import pytest

from memval.metrics.plasticity import PlasticityTracker, discover_params


class Toy:
    """Minimal stand-in: one weight matrix, one bias, one transient state."""

    def __init__(self, n=4):
        self.W = np.zeros((n, n))
        self.b = np.zeros(n)
        self.current_state = np.zeros(n)      # transient, must not be tracked


def test_discovers_parameters_and_skips_transient_state():
    assert discover_params(Toy()) == ["W", "b"]


def test_declared_params_win_over_discovery():
    m = Toy()
    m.plasticity_params = ("W",)
    assert discover_params(m) == ["W"]


def test_coherent_updates_give_ei_one():
    """Every update points the same way, so net change == summed change."""
    m = Toy()
    t = PlasticityTracker(m).begin()
    for _ in range(10):
        m.W += 0.1
        t.checkpoint()
    r = t.result()
    assert r["efficiency_index"] == pytest.approx(1.0)
    assert r["n_checkpoints"] == 10
    assert r["total_change"] == pytest.approx(r["summed_change"])


def test_cancelling_updates_give_ei_zero():
    """Updates that alternate sign cancel: weights end where they started."""
    m = Toy()
    t = PlasticityTracker(m).begin()
    for i in range(10):
        m.W += 0.1 * (-1) ** i
        t.checkpoint()
    r = t.result()
    assert r["efficiency_index"] == pytest.approx(0.0)
    assert r["total_change"] == pytest.approx(0.0)
    assert r["summed_change"] > 0


def test_ei_is_bounded_by_the_triangle_inequality():
    rng = np.random.default_rng(0)
    m = Toy(8)
    t = PlasticityTracker(m).begin()
    for _ in range(20):
        m.W += rng.normal(0, 0.1, m.W.shape)
        m.b += rng.normal(0, 0.1, m.b.shape)
        t.checkpoint()
    ei = t.result()["efficiency_index"]
    assert 0.0 <= ei <= 1.0


def test_no_weight_change_is_nan_not_zero():
    """Nothing learned is not the same as incoherent learning."""
    t = PlasticityTracker(Toy()).begin()
    for _ in range(5):
        t.checkpoint()
    r = t.result()
    assert np.isnan(r["efficiency_index"])
    assert r["summed_change"] == 0.0


def test_begin_restarts_the_window():
    """Schema acquisition must be excludable from the measurement."""
    m = Toy()
    t = PlasticityTracker(m)
    t.begin()
    for i in range(6):                      # noisy "pre-training", cancels out
        m.W += 0.5 * (-1) ** i
        t.checkpoint()
    t.begin()                               # window reopens here
    for _ in range(4):
        m.W += 0.1
        t.checkpoint()
    r = t.result()
    assert r["n_checkpoints"] == 4
    assert r["efficiency_index"] == pytest.approx(1.0)


def test_per_parameter_breakdown_separates_layers():
    """A layer learning coherently and one thrashing must not be averaged away."""
    m = Toy()
    t = PlasticityTracker(m).begin()
    for i in range(10):
        m.W += 0.1                          # coherent
        m.b += 0.1 * (-1) ** i              # cancels
        t.checkpoint()
    bp = t.result()["by_param"]
    assert bp["W"]["ei"] == pytest.approx(1.0)
    assert bp["b"]["ei"] == pytest.approx(0.0)


def test_checkpoint_before_begin_raises():
    with pytest.raises(RuntimeError):
        PlasticityTracker(Toy()).checkpoint()


def test_tracks_a_real_arm():
    from memval.models.baselines.asymmetric_hopfield import AsymmetricHopfieldNetwork
    rng = np.random.default_rng(0)
    m = AsymmetricHopfieldNetwork(n_features=12, learning_rate=0.1)
    assert "W" in discover_params(m)
    assert "current_state" not in discover_params(m)
    t = PlasticityTracker(m).begin()
    seq = rng.normal(0, 1, (5, 12))
    for _ in range(6):
        m.reset_context()
        m.fit_sequence(seq, epochs=1)
        t.checkpoint()
    r = t.result()
    assert r["summed_change"] > 0
    assert 0.0 <= r["efficiency_index"] <= 1.0
