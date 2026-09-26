"""Prompt conditioning and non-learning state priming.

Two declarations, both about *retrieval-time state*:

- ``prompt_conditioned`` -- does ``recall`` use the whole 2D prompt, or reduce
  it to ``prompt_event[-1]``? The base contract permits a trajectory and
  ``pattern_completion.py`` already passes one.
- ``StatePrimeable`` -- can the arm advance carried state *without learning*?
  ``fit_event`` moves state but writes weights; ``predict_next`` is pure. This
  is the missing middle, and it is what lets a delayed-cue task deliver history.
"""

import numpy as np
import pytest

import memval.models.baselines as baselines
from memval.models.base import HippocampalModel
from memval.models.capabilities import (
    StatePrimeable,
    consumes_prompt_trajectory,
    supports_priming,
)
from memval.models.baselines import (
    DTSESNSequenceNetwork,
    MultilayerTemporalPCNetwork,
    OriginalEqPropSequenceNetwork,
    ThetaPhaseSequenceNetwork,
    PredictiveRecirculationNetwork,
)

N = 10


def _arms():
    import inspect
    for name in sorted(dir(baselines)):
        cls = getattr(baselines, name)
        if (isinstance(cls, type) and issubclass(cls, HippocampalModel)
                and cls is not HippocampalModel and not inspect.isabstract(cls)):
            yield name, cls


# ---------------------------------------------------------------------------
# prompt_conditioned
# ---------------------------------------------------------------------------

def test_prompt_conditioned_is_declared_as_a_bool_everywhere():
    for name, cls in _arms():
        assert isinstance(getattr(cls, "prompt_conditioned", None), bool), name


@pytest.mark.parametrize("cls,kw", [
    (ThetaPhaseSequenceNetwork, {}),
    (OriginalEqPropSequenceNetwork, dict(n_hidden=8, seed=0, n_epochs=1)),
])
def test_memoryless_arms_declare_false_and_ignore_the_prefix(cls, kw):
    """False is the honest declaration for an arm with no carried state: a
    longer prompt cannot establish anything, so `recall` must be indifferent to
    everything but the last row."""
    assert not consumes_prompt_trajectory(cls)
    rng = np.random.default_rng(0)
    tail = rng.normal(size=(1, N))
    A = np.vstack([rng.normal(size=(3, N)), tail])
    B = np.vstack([rng.normal(size=(3, N)), tail])

    m = cls(n_features=N, **kw)
    m.fit_sequence(np.vstack([A, rng.normal(size=(3, N))]))
    m.reset_context(); ra = m.recall(A, length=3)
    m.reset_context(); rb = m.recall(B, length=3)
    assert np.allclose(ra, rb)


def test_latent_arm_declares_true_and_the_prefix_changes_the_rollout():
    """The regression this fixes: MultilayerTPC is the only latent-state arm in
    the suite and its `recall` used to take `prompt_event[-1]`, so two routes
    sharing a final observation produced bit-identical recalls."""
    assert consumes_prompt_trajectory(MultilayerTemporalPCNetwork)
    rng = np.random.default_rng(0)
    tail = rng.normal(size=(1, N))
    A = np.vstack([rng.normal(size=(3, N)), tail])
    B = np.vstack([rng.normal(size=(3, N)), tail])

    m = MultilayerTemporalPCNetwork(n_features=N, n_hidden=16, n_epochs=40, seed=0)
    m.fit_sequence(np.vstack([A, rng.normal(size=(3, N))]))
    m.fit_sequence(np.vstack([B, rng.normal(size=(3, N))]))

    m.reset_context(); ra = m.recall(A, length=3)
    m.reset_context(); rb = m.recall(B, length=3)
    assert not np.allclose(ra, rb), "prompt prefix must reach the carried latent"


# ---------------------------------------------------------------------------
# StatePrimeable
# ---------------------------------------------------------------------------

PRIMEABLE = [
    (PredictiveRecirculationNetwork, dict(n_hidden=16, seed=0), "_h"),
    (MultilayerTemporalPCNetwork, dict(n_hidden=16, seed=0), "_prev_z"),
    (DTSESNSequenceNetwork, dict(n_units=30, seed=0), "_x"),
]


@pytest.mark.parametrize("cls,kw,state_attr", PRIMEABLE,
                         ids=lambda v: getattr(v, "__name__", ""))
def test_observe_moves_state_and_never_touches_weights(cls, kw, state_attr):
    """The whole point of `observe`: state moves, plasticity does not. If it
    wrote weights it would be `fit_event` and unusable at probe time."""
    assert supports_priming(cls)
    seq = np.random.default_rng(0).normal(size=(6, N))

    m = cls(n_features=N, **kw)
    m.fit_sequence(seq)
    before = {k: v.copy() for k, v in vars(m).items()
              if isinstance(v, np.ndarray) and k.startswith(("W", "P", "b_"))}
    assert before, "expected some weight arrays to guard"

    m.reset_context()
    m.observe_sequence(seq[:4])

    assert not np.allclose(getattr(m, state_attr), 0), "carried state did not move"
    for k, v in before.items():
        assert np.allclose(getattr(m, k), v), f"observe modified {k}"


def test_memoryless_arms_do_not_declare_primeable():
    """Declaring it on an arm with nothing to carry would claim a capability
    that does nothing -- the exact ambiguity these declarations remove."""
    assert not supports_priming(ThetaPhaseSequenceNetwork)
    assert not supports_priming(OriginalEqPropSequenceNetwork)


def test_declaring_primeable_without_implementing_is_caught_by_the_override_check():
    """The construction-time guard is GONE, deliberately, and this pins what
    replaced it.

    `HippocampalModel` grew a concrete no-op `observe` on 2026-09-08 so the
    disambiguation probe could deliver a stem prefix to every arm through one
    call. That concrete method sits in the MRO ahead of `StatePrimeable`'s
    `@abstractmethod`, so declaring the capability without implementing it no
    longer raises -- the exact trap documented on `base.py` for `fit_event`.

    What still holds, and what does not:
      * `supports_priming()` remains correct for the roster, because it is a
        NOMINAL check: arms that cannot be primed simply do not declare
        `StatePrimeable`. The base no-op does not make them subclasses.
      * A future arm that declares `StatePrimeable` and forgets `observe` would
        silently inherit the no-op and report `supports_priming() == True` while
        priming did nothing. Construction will NOT catch that;
        `tests/test_observe_default.py::test_state_primeable_arms_override_observe`
        is the check that does, by comparing `cls.observe` against the base.
    """
    class Declared(HippocampalModel, StatePrimeable):
        def fit_sequence(self, sequence_data, context_data=None, **kwargs): ...
        def predict_next(self, current_event, current_context=None, **kwargs): ...
        def recall(self, prompt_event, length, prompt_context=None, **kwargs): ...
        def get_latent_state(self): return {}
        def reset_context(self): ...

    m = Declared()                                    # no longer raises
    assert supports_priming(m)                        # and still claims the capability
    assert Declared.observe is HippocampalModel.observe   # while inheriting the no-op
    # ...which is precisely what the override check in test_observe_default.py rejects.


def test_observe_sequence_is_order_sensitive():
    m1 = MultilayerTemporalPCNetwork(n_features=N, n_hidden=16, seed=0)
    m2 = MultilayerTemporalPCNetwork(n_features=N, n_hidden=16, seed=0)
    seq = np.random.default_rng(1).normal(size=(4, N))
    m1.observe_sequence(seq)
    m2.observe_sequence(seq[::-1])
    assert not np.allclose(m1._prev_z, m2._prev_z)
