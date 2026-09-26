"""Every registered arm must train for exactly the `epochs` it is passed.

`epochs_to_criterion` states the contract in its own signature docs --
"fit(model, epochs) -- trains for exactly `epochs` passes" -- and 13 of the 26
`fit_sequence` call sites in the suite rely on it *without* also pinning
`n_epochs` on the instance.

Nine arms broke it. The EP family declared `fit_sequence(self, sequence_data,
**kwargs)` and looped over `self.n_epochs`, so a call-level `epochs=` landed in
`**kwargs` and was discarded; `SpikingEqPropSequenceNetwork` looked the key up
under the wrong name (`n_epochs` rather than `epochs`), which always missed and
so always fell back to the same place.

The damage was not subtle. `continual_chain` steps **one epoch at a time**
(`epochs_per_step=1`, `max_epochs=512`) and probes after each step, so every
step silently ran the registry default of 100 epochs instead of 1 -- a 100x
over-training that made the section take 40 minutes rather than ~3, and made
`chain_epochs_to_criterion` a count of *steps* worth 100 epochs each for the EP
family while remaining a count of single epochs for everyone else. That is
precisely the cross-arm confound the criterion-referenced exposure policy exists
to remove.

These tests are behavioural on purpose. An earlier audit inspected `fit_sequence`
signatures for the string "epochs" and mis-classified `DGOriginalEqPropNetwork`
-- which takes `**kwargs` but reads `kwargs.get("epochs", ...)` correctly -- as
broken. Only running the thing settles it.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin"))

from run_benchmark import MODEL_REGISTRY  # noqa: E402

N_FEATURES = 40


def _sequence():
    return np.random.default_rng(0).normal(size=(5, N_FEATURES)) * 0.2


def _build(name, n_epochs):
    """The arm at its registry defaults, with `n_epochs` set explicitly."""
    entry = MODEL_REGISTRY[name]
    kwargs = {k: v for k, v in entry["default_kwargs"].items() if k != "n_epochs"}
    model = entry["class"](n_features=N_FEATURES, **kwargs)
    model.n_epochs = n_epochs
    return model


def _flat_state(model):
    """All learned parameters as one vector, for comparing training outcomes."""
    return np.concatenate([np.asarray(v).ravel()
                           for v in model.get_latent_state().values()])


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_epochs_kwarg_changes_training(name):
    """`epochs=1` and `epochs=3` must not produce the same trained model.

    The failure mode this catches is the kwarg being swallowed entirely: an arm
    that ignores it trains `self.n_epochs` times either way and lands in exactly
    the same place.
    """
    x = _sequence()
    one, three = _build(name, 100), _build(name, 100)
    one.fit_sequence(x, epochs=1)
    three.fit_sequence(x, epochs=3)
    assert not np.allclose(_flat_state(one), _flat_state(three)), (
        f"{name}.fit_sequence ignored epochs=: training for 1 and for 3 passes "
        f"gave identical parameters, so the call-level budget is being dropped "
        f"and self.n_epochs ({one.n_epochs}) used instead."
    )


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_epochs_kwarg_equals_pinning_n_epochs(name):
    """Passing `epochs=n` must equal constructing with `n_epochs=n`.

    Stronger than the previous test, and the property the suite actually relies
    on: the two ways of asking for n passes have to be the same n. An arm could
    honour the kwarg but apply it on top of its own default and still fail here.
    """
    x = _sequence()
    via_kwarg = _build(name, 100)
    via_kwarg.fit_sequence(x, epochs=3)
    via_attribute = _build(name, 3)
    via_attribute.fit_sequence(x, epochs=3)
    assert np.allclose(_flat_state(via_kwarg), _flat_state(via_attribute)), (
        f"{name}: fit_sequence(x, epochs=3) differs from the same call on an arm "
        f"built with n_epochs=3, so the call-level budget is not the number of "
        f"passes actually run."
    )
