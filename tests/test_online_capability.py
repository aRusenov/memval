"""The OnlineTrainable capability contract, and the `online_equivalent` claims.

These tests exist because the previous design -- a concrete `fit_event` stub on
`HippocampalModel` that raised -- made every capability check unsound: any
structural probe (`hasattr`, a runtime_checkable Protocol) reported True for
every arm, and the benchmark discovered the gap only by catching an exception
per sweep point and writing 0.0.
"""

import numpy as np
import pytest

from memval.benchmarks.ingest import ingest
from memval.models.base import HippocampalModel
from memval.models.capabilities import (
    OnlineTrainable,
    UnsupportedRegime,
    is_online_equivalent,
    supports_online,
)
from memval.models.baselines import (
    AsymmetricHopfieldNetwork,
    DTSESNSequenceNetwork,
    MultilayerTemporalPCNetwork,
    OriginalEqPropSequenceNetwork,
    TemporalPCNetwork,
    ThetaPhaseSequenceNetwork,
    PredictiveRecirculationNetwork,
)

N_FEATURES = 10


def _seq(t=6, seed=0):
    return np.random.default_rng(seed).normal(size=(t, N_FEATURES)) * 0.5


def _weights(model):
    return {k: v.copy() for k, v in vars(model).items()
            if k.startswith(("W", "b_")) and isinstance(v, np.ndarray)}


def _max_delta(a, b):
    wa, wb = _weights(a), _weights(b)
    assert wa and wa.keys() == wb.keys()
    return max(float(np.max(np.abs(wa[k] - wb[k]))) for k in wa)


# ---------------------------------------------------------------------------
# The base class must not satisfy the capability
# ---------------------------------------------------------------------------

def test_base_class_does_not_define_the_online_methods():
    """A concrete stub anywhere in the MRO would satisfy the @abstractmethod and
    make `isinstance(arm, OnlineTrainable)` true for every arm. This is the
    invariant the whole design rests on."""
    assert "fit_event" not in vars(HippocampalModel)
    assert "on_event_boundary" not in vars(HippocampalModel)
    assert not issubclass(HippocampalModel, OnlineTrainable)


def test_batch_only_arm_is_not_online():
    assert not supports_online(AsymmetricHopfieldNetwork)
    assert not supports_online(AsymmetricHopfieldNetwork(n_features=N_FEATURES))
    assert not hasattr(AsymmetricHopfieldNetwork(n_features=N_FEATURES), "fit_event")


def test_declaring_without_implementing_fails_at_construction():
    """The Java `implements`-without-a-body analogue: caught when the object is
    built, not at the first sweep point."""

    class Broken(HippocampalModel, OnlineTrainable):
        def fit_sequence(self, sequence_data, context_data=None, **kwargs): ...
        def predict_next(self, current_event, current_context=None, **kwargs): ...
        def recall(self, prompt_event, length, prompt_context=None, **kwargs): ...
        def get_latent_state(self): return {}
        def reset_context(self): ...

    with pytest.raises(TypeError, match="fit_event"):
        Broken()


def test_capability_is_inherited_by_subclassed_arms():
    """A subclass of an online arm inherits `fit_event` and with it the
    declaration, without restating it."""

    class Subclassed(OriginalEqPropSequenceNetwork):
        pass

    assert supports_online(Subclassed)
    assert supports_online(Subclassed(n_features=N_FEATURES, n_hidden=8, seed=0))


# ---------------------------------------------------------------------------
# The ingest seam
# ---------------------------------------------------------------------------

def test_ingest_refuses_streaming_on_a_batch_only_arm():
    model = AsymmetricHopfieldNetwork(n_features=N_FEATURES)
    with pytest.raises(UnsupportedRegime, match="OnlineTrainable"):
        ingest(model, _seq(), regime="streamed")


def test_ingest_rejects_an_unknown_regime():
    with pytest.raises(ValueError, match="unknown ingestion regime"):
        ingest(ThetaPhaseSequenceNetwork(n_features=N_FEATURES), _seq(), regime="nope")


def test_batch_ingest_works_on_arms_whose_fit_sequence_omits_context_data():
    """`original_eqprop` declares `fit_sequence(self, sequence_data, **kwargs)`. Passing context positionally
    raises TypeError, so `ingest` must pass it by keyword."""
    model = OriginalEqPropSequenceNetwork(n_features=N_FEATURES, n_hidden=8, seed=0)
    ingest(model, _seq(), regime="batch")
    ingest(model, _seq(), regime="batch", context_data=np.zeros((6, 3)))


def test_ingest_fences_both_seams():
    """Two streamed presentations must not form a transition across the seam:
    the result is the same as fencing by hand."""
    seq = _seq()
    a = ThetaPhaseSequenceNetwork(n_features=N_FEATURES)
    ingest(a, seq, regime="streamed")
    ingest(a, seq, regime="streamed")

    b = ThetaPhaseSequenceNetwork(n_features=N_FEATURES)
    for _ in range(2):
        b.on_event_boundary()
        for x in seq:
            b.fit_event(x)
        b.on_event_boundary()
    assert _max_delta(a, b) == 0.0


# ---------------------------------------------------------------------------
# online_equivalent is a claim about rule identity -- verify it
# ---------------------------------------------------------------------------

EQUIVALENT_ARMS = [
    (PredictiveRecirculationNetwork, dict(n_hidden=12, n_epochs=1, seed=0)),
    (ThetaPhaseSequenceNetwork, {}),
    (TemporalPCNetwork, dict(n_epochs=1)),
    (MultilayerTemporalPCNetwork, dict(n_hidden=12, n_epochs=1, seed=0)),
    (DTSESNSequenceNetwork, dict(n_units=20, seed=0)),
]


@pytest.mark.parametrize("cls,kw", EQUIVALENT_ARMS, ids=lambda v: getattr(v, "__name__", ""))
def test_online_equivalent_arms_take_identical_paths(cls, kw):
    """At one epoch, `fit_sequence` must be exactly the streamed rule. This is
    what makes a batch-vs-streamed contrast on these arms pure ingestion."""
    assert is_online_equivalent(cls)
    seq = _seq()
    a, b = cls(n_features=N_FEATURES, **kw), cls(n_features=N_FEATURES, **kw)
    ingest(a, seq, regime="batch")
    ingest(b, seq, regime="streamed")
    assert _max_delta(a, b) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("cls,kw", [
    (OriginalEqPropSequenceNetwork, dict(n_hidden=12, seed=0, n_epochs=1)),
])
def test_non_equivalent_arms_really_do_differ(cls, kw):
    """EP averages the batch delta by 1/n_transitions and steps once per epoch,
    while `fit_event` takes a full step per transition. The flag is False, and
    that is not pessimism -- the paths genuinely diverge."""
    assert not is_online_equivalent(cls)
    seq = _seq()
    a, b = cls(n_features=N_FEATURES, **kw), cls(n_features=N_FEATURES, **kw)
    ingest(a, seq, regime="batch")
    ingest(b, seq, regime="streamed")
    assert _max_delta(a, b) > 1e-6
