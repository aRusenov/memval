"""use_output_bias flag: b_o stays exactly 0 when off (batch + online), updates when on.
Covers Original / DG / XdG (DG & XdG inherit the gated update sites)."""

import numpy as np
import pytest

from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork
from memval.models.baselines.dg_original_eqprop import DGOriginalEqPropSequenceNetwork
from memval.models.baselines.dg_xdg_eqprop import DGXdGEqPropSequenceNetwork

DIM = 8


def make(model_type, use_output_bias):
    common = dict(n_features=DIM, n_hidden=16, learning_rate=0.1, beta=0.5,
                  n_settle_steps=15, n_epochs=20, seed=0, use_output_bias=use_output_bias)
    if model_type == "original":
        return OriginalEqPropSequenceNetwork(**common)
    if model_type == "dg":
        return DGOriginalEqPropSequenceNetwork(n_dg=32, dg_target_sparsity=0.1, dg_seed=0, **common)
    return DGXdGEqPropSequenceNetwork(n_dg=32, dg_target_sparsity=0.1, gate_sparsity=0.25,
                                      gate_seed=0, dg_seed=0, **common)


def _seq(seed=1):
    rng = np.random.default_rng(seed)
    s = np.abs(rng.standard_normal((6, DIM)))
    return s / np.linalg.norm(s, axis=1, keepdims=True)


@pytest.mark.parametrize("mt", ["original", "dg", "xdg"])
def test_bias_off_keeps_bo_zero_batch(mt):
    model = make(mt, use_output_bias=False)
    model.fit_sequence(_seq())
    assert np.array_equal(model.b_o, np.zeros(DIM))


@pytest.mark.parametrize("mt", ["original", "dg", "xdg"])
def test_bias_on_updates_bo(mt):
    model = make(mt, use_output_bias=True)
    model.fit_sequence(_seq())
    assert np.linalg.norm(model.b_o) > 0.0


def test_bias_off_keeps_bo_zero_online():
    """Online (fit_event) path also respects the flag."""
    model = make("original", use_output_bias=False)
    for evt in _seq():
        model.fit_event(evt)
    assert np.array_equal(model.b_o, np.zeros(DIM))
