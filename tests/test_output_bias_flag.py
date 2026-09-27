"""use_output_bias flag: b_o stays exactly 0 when off (batch + online), updates when on."""

import numpy as np

from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork

DIM = 8


def make(use_output_bias):
    return OriginalEqPropSequenceNetwork(
        n_features=DIM, n_hidden=16, learning_rate=0.1, beta=0.5, n_settle_steps=15,
        n_epochs=20, seed=0, use_output_bias=use_output_bias)


def _seq(seed=1):
    rng = np.random.default_rng(seed)
    s = np.abs(rng.standard_normal((6, DIM)))
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def test_bias_off_keeps_bo_zero_batch():
    model = make(use_output_bias=False)
    model.fit_sequence(_seq())
    assert np.array_equal(model.b_o, np.zeros(DIM))


def test_bias_on_updates_bo():
    model = make(use_output_bias=True)
    model.fit_sequence(_seq())
    assert np.linalg.norm(model.b_o) > 0.0


def test_bias_off_keeps_bo_zero_online():
    """Online (fit_event) path also respects the flag."""
    model = make(use_output_bias=False)
    for evt in _seq():
        model.fit_event(evt)
    assert np.array_equal(model.b_o, np.zeros(DIM))
