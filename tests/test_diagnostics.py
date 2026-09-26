"""Smoke tests for memval.diagnostics: full-capability path + graceful degradation."""

import numpy as np
import pytest

from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork
from memval.diagnostics import (
    ab_forgetting_report, representation_overlap, update_interference,
    fisher_diagonal, fisher_attribution, snapshot_parameters,
    supports_representations, supports_gradients,
)


class ToyEncoder:
    def __init__(self, symbols, dim=16, seed=0):
        rng = np.random.default_rng(seed)
        raw = np.abs(rng.standard_normal((len(symbols), dim)))
        self.codes = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self.symbols = list(symbols)
        self.idx = {s: i for i, s in enumerate(self.symbols)}

    def encode(self, seq):
        return np.stack([self.codes[self.idx[s]] for s in seq])

    def decode(self, vec):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        return self.symbols[int(np.argmax(self.codes @ v))]


class CoreOnlyModel:
    """A model exposing only the core interface — no representation/gradient probes."""
    def __init__(self, dim):
        self.dim = dim

    def fit_sequence(self, seq, **kw):
        pass

    def predict_next(self, x, **kw):
        return np.asarray(x, dtype=float).ravel()


@pytest.fixture
def setup():
    list_a, list_b = list("ABC"), list("DEF")
    enc = ToyEncoder(list_a + list_b, dim=16, seed=1)
    model = OriginalEqPropSequenceNetwork(
        n_features=16, n_hidden=24, learning_rate=0.1, beta=0.5,
        n_settle_steps=20, n_epochs=20, seed=1)
    return model, enc, list_a, list_b


def test_capabilities_detected(setup):
    model, *_ = setup
    assert supports_representations(model)
    assert supports_gradients(model)


def test_full_report_structure(setup):
    model, enc, list_a, list_b = setup
    rep = ab_forgetting_report(model, enc, list_a, list_b)

    # behavioural present and in [0, 1]
    for k in ("base_a", "retain_a", "b_learn"):
        assert 0.0 <= rep["behavioral"][k] <= 1.0

    # geometry: one entry per named layer, with a floor
    assert set(rep["geometry"]) == {"input", "hidden", "output"}
    for layer in rep["geometry"].values():
        assert 0.0 <= layer["jaccard_ab"] <= 1.0
        assert 0.0 <= layer["floor"] <= 1.0

    # interference: one block per parameter group
    assert set(rep["interference_init"]) == {"W_ih", "W_ho", "b_h", "b_o"}

    # fisher attribution: shares sum to 1
    shares = rep["fisher_attribution"]["contrib_share"]
    assert set(shares) == {"W_ih", "W_ho", "b_h", "b_o"}
    assert np.isclose(sum(shares.values()), 1.0)


def test_fisher_attribution_matches_manual(setup):
    """The report's attribution equals a hand-computed F_A * displacement^2."""
    model, enc, list_a, list_b = setup
    ta = [(enc.encode([x])[0], enc.encode([y])[0]) for x, y in zip(list_a, list_a[1:])]

    model.fit_sequence(enc.encode(list_a))
    theta_A = snapshot_parameters(model)
    F_A = fisher_diagonal(model, ta)
    model.fit_sequence(enc.encode(list_b))
    theta_B = snapshot_parameters(model)

    attr = fisher_attribution(F_A, theta_A, theta_B)
    manual = float((F_A["b_o"] * (theta_B["b_o"] - theta_A["b_o"]) ** 2).sum())
    assert np.isclose(attr["contrib"]["b_o"], manual)


def test_graceful_degradation():
    """A core-only model yields the behavioural tier and empty geometric/weight tiers."""
    enc = ToyEncoder(list("ABCDEF"), dim=8, seed=2)
    model = CoreOnlyModel(dim=8)
    assert not supports_representations(model)
    assert not supports_gradients(model)

    rep = ab_forgetting_report(model, enc, list("ABC"), list("DEF"))
    assert rep["geometry"] == {}
    assert rep["interference_init"] == {}
    assert rep["fisher_attribution"] == {}
    assert "retain_a" in rep["behavioral"]
