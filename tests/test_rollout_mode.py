"""Every arm must declare how its own ``recall()`` rolls out, and the
declaration must match what the code does.

``measure_recall_autoregressive`` drives ``predict_next`` itself and takes an
explicit ``feedback_mode``, so that rollout path is a declared protocol
condition. ``model.recall()`` is the other path -- scored by
``benchmarks/pattern_completion.py`` -- and
each arm implements it privately. These tests stop those private choices from
drifting apart unnoticed.
"""

import inspect

import numpy as np
import pytest

import memval.models.baselines as baselines
from memval.models.base import HippocampalModel
from memval.models.capabilities import (
    RolloutMode,
    rollout_mode_of,
    rollout_modes_comparable,
)
from memval.models.baselines import (
    AsymmetricHopfieldNetwork,
    MultilayerTemporalPCNetwork,
    OriginalEqPropSequenceNetwork,
    TemporalPCNetwork,
    ThetaPhaseSequenceNetwork,
)

N_FEATURES = 10


def _arms():
    for name in sorted(dir(baselines)):
        cls = getattr(baselines, name)
        if (isinstance(cls, type) and issubclass(cls, HippocampalModel)
                and cls is not HippocampalModel and not inspect.isabstract(cls)):
            yield name, cls


def test_every_arm_declares_a_rollout_mode():
    """The base leaves `rollout_mode` None on purpose -- a silent default is the
    failure this declaration exists to prevent. So every concrete arm must set
    it, and this test is the enforcement."""
    undeclared = [n for n, c in _arms() if rollout_mode_of(c) is None]
    assert not undeclared, f"arms with no declared rollout_mode: {undeclared}"


def test_declared_modes_are_enum_members():
    for name, cls in _arms():
        assert isinstance(rollout_mode_of(cls), RolloutMode), name


def test_the_declaration_survives_instantiation():
    m = ThetaPhaseSequenceNetwork(n_features=N_FEATURES)
    assert rollout_mode_of(m) is RolloutMode.OBSERVATION


# ---------------------------------------------------------------------------
# The declarations are verified, not trusted
# ---------------------------------------------------------------------------

OBSERVATION_ARMS = [
    (ThetaPhaseSequenceNetwork, {}),
    (TemporalPCNetwork, dict(n_epochs=1)),
    (OriginalEqPropSequenceNetwork, dict(n_hidden=8, seed=0, n_epochs=1)),
    (AsymmetricHopfieldNetwork, {}),
]


def _trained(cls, kw, seq):
    model = cls(n_features=N_FEATURES, **kw)
    model.fit_sequence(seq)
    return model


@pytest.mark.parametrize("cls,kw", OBSERVATION_ARMS, ids=lambda v: getattr(v, "__name__", ""))
def test_observation_arms_really_feed_the_raw_prediction_back(cls, kw):
    """OBSERVATION means `recall` is exactly a raw-feedback loop over
    `predict_next` -- no renormalisation, no codebook snap. If an arm quietly
    started cleaning up its feedback, this catches it."""
    assert rollout_mode_of(cls) is RolloutMode.OBSERVATION
    seq = np.random.default_rng(0).normal(size=(6, N_FEATURES)) * 0.5

    model = _trained(cls, kw, seq)
    got = model.recall(seq[0], length=4)

    ref_model = _trained(cls, kw, seq)
    ref_model.reset_context()
    cur, expected = seq[0].copy(), []
    for _ in range(4):
        cur = ref_model.predict_next(cur)
        expected.append(cur.copy())

    assert np.allclose(got, np.array(expected), atol=1e-12)


def test_latent_arm_does_not_feed_observations_back():
    """MultilayerTPC declares LATENT: its rollout runs in latent space, so it
    must NOT coincide with the observation-space loop. A match here would mean
    the declaration is wrong."""
    cls, kw = MultilayerTemporalPCNetwork, dict(n_hidden=12, n_epochs=1, seed=0)
    assert rollout_mode_of(cls) is RolloutMode.LATENT
    seq = np.random.default_rng(0).normal(size=(6, N_FEATURES)) * 0.5

    model = _trained(cls, kw, seq)
    got = model.recall(seq[0], length=4)

    ref_model = _trained(cls, kw, seq)
    ref_model.reset_context()
    cur, obs_loop = seq[0].copy(), []
    for _ in range(4):
        cur = ref_model.predict_next(cur)
        obs_loop.append(cur.copy())

    assert not np.allclose(got, np.array(obs_loop), atol=1e-8)


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------

def test_encoder_and_observation_arms_are_not_comparable():
    """The point of the declaration. An ENCODER arm hardcodes what the symbolic
    suite calls a "drift-free upper bound ... [that] hides sub-threshold
    representational degradation"; the EP family hardcodes the maximal-drift
    protocol. Both are scored by pattern_completion and disambiguation."""

    class EncoderArm(HippocampalModel):
        rollout_mode = RolloutMode.ENCODER

        def fit_sequence(self, sequence_data, context_data=None, **kwargs): ...
        def predict_next(self, current_event, current_context=None, **kwargs): ...
        def recall(self, prompt_event, length, prompt_context=None, **kwargs): ...
        def get_latent_state(self): return {}
        def reset_context(self): ...

    assert rollout_mode_of(EncoderArm) is RolloutMode.ENCODER
    assert rollout_mode_of(OriginalEqPropSequenceNetwork) is RolloutMode.OBSERVATION
    assert not rollout_modes_comparable(EncoderArm, OriginalEqPropSequenceNetwork)
    assert rollout_modes_comparable(ThetaPhaseSequenceNetwork, OriginalEqPropSequenceNetwork)


def test_undeclared_is_never_comparable():
    class Undeclared(HippocampalModel):
        def fit_sequence(self, sequence_data, context_data=None, **kwargs): ...
        def predict_next(self, current_event, current_context=None, **kwargs): ...
        def recall(self, prompt_event, length, prompt_context=None, **kwargs): ...
        def get_latent_state(self): return {}
        def reset_context(self): ...

    assert not rollout_modes_comparable(Undeclared, Undeclared)
