"""Optional model capabilities, declared nominally.

MemVal's suites vary the **ingestion regime** (batch ``fit_sequence`` vs.
streamed ``fit_event``) as a *condition inside a suite*, so the regime is a
property of the protocol, not of the arm. That rules out splitting
``HippocampalModel`` into separate online/offline base classes: an arm must be
able to run under both regimes wherever it supports both.

What is a property of the arm is *which optional interfaces it implements*, and
that is what this module declares.

Why nominal (inheritance) rather than a ``typing.Protocol``
----------------------------------------------------------
A ``@runtime_checkable`` Protocol only tests for the *presence of method names*
-- it cannot check signatures, and it cannot see the difference between an arm
that implements ``fit_event`` and an arm that inherits a stub. Inheriting
``OnlineTrainable`` is an explicit assertion by the author (Java's
``implements``), and omitting the implementation is caught at construction time
with a ``TypeError`` rather than at the first sweep point.

It also carries ``online_equivalent``, a *semantic* fact about rule identity
that no structural check could express (see below).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import numpy as np


class UnsupportedRegime(Exception):
    """An ingestion regime was requested that this arm does not implement.

    Distinct from a failure *during* ingestion: a benchmark that catches this
    must record the sweep point as **not applicable**, never as a score. Writing
    a 0.0 here reports a capability gap as a performance failure.
    """


class OnlineTrainable(ABC):
    """Declares that an arm learns from one event at a time.

    Inheriting this is the declaration; ``isinstance(model, OnlineTrainable)``
    then reports what the author asserted rather than a coincidence of method
    names. Arms combine it with the base class::

        class ThetaPhaseSequenceNetwork(HippocampalModel, OnlineTrainable):
            online_equivalent = True

    ``fit_event`` and ``on_event_boundary`` live here rather than on
    ``HippocampalModel`` deliberately. A concrete stub anywhere in the MRO would
    satisfy the ``@abstractmethod`` and silently void the guarantee -- which is
    exactly the trap the previous stub-on-the-base design fell into.
    """

    #: True when ``fit_sequence`` applies *exactly* the per-transition rule that
    #: streaming ``fit_event`` applies, so the two paths differ only in
    #: ingestion. False when the batch path is a genuinely different rule -- the
    #: EP arms average their update by ``1 / n_transitions`` per epoch while
    #: ``fit_event`` takes a full step per transition, and Hopfield's default
    #: batch path is a one-shot pseudoinverse with no streaming counterpart.
    #:
    #: Read this before interpreting a batch-vs-streamed contrast: when it is
    #: False the contrast confounds ingestion with a change of learning rule.
    online_equivalent: bool = False

    @abstractmethod
    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None, **kwargs):
        """Train on a single event in a sensory stream.

        Args:
            event: The current event, a 1D array of shape ``(Features,)``.
            context: Optional contextual signal tied to the event, shape
                ``(ContextFeatures,)``.
            **kwargs: Arm-specific fitting arguments.
        """

    def on_event_boundary(self, boundary_context: Optional[np.ndarray] = None) -> None:
        """Signal a sequence seam, so no transition forms across it.

        Defaults to ``reset_context()``, which clears the transient state the
        arm carries between events (previous-event buffer, latent, reservoir)
        while leaving learned weights untouched.
        """
        self.reset_context()


def supports_online(model_or_class: Any) -> bool:
    """Whether an arm (instance *or* class) declares ``OnlineTrainable``."""
    if isinstance(model_or_class, type):
        return issubclass(model_or_class, OnlineTrainable)
    return isinstance(model_or_class, OnlineTrainable)


def is_online_equivalent(model_or_class: Any) -> bool:
    """Whether the arm's batch path is the streamed rule. False if not online."""
    return supports_online(model_or_class) and bool(
        getattr(model_or_class, "online_equivalent", False)
    )


class TemporallyClocked(ABC):
    """Declares that an arm is clocked by elapsed TIME, not by event ordinal.

    Two things follow, and a benchmark needs both:

    1. ``fit_sequence`` / ``fit_event`` accept the inter-event ``intervals`` and
       let them change the state, and
    2. ``predict_next`` accepts ``elapsed`` so the same cue can produce different
       continuations for different gaps.

    Declaring it is the assertion that the interval is *load-bearing*. Duck-typing
    cannot establish that: ``AsymmetricHopfieldNetwork.fit_sequence`` takes
    ``**kwargs`` and would happily swallow ``intervals=[...]`` while remaining
    clocked by ordinal, so a signature check would report a capability the arm
    does not have — silently, and in the direction that inflates the score.

    The benchmark this gates is interval-as-cue (see
    ``examples/dts_esn_interval_demo.py``): one prefix, two continuations, the
    elapsed gap the only thing telling them apart. An ordinal-clocked arm sees two
    contradictory transitions out of the same item and cannot do better than
    picking one.
    """

    @abstractmethod
    def predict_next(self, current_event: np.ndarray,
                     current_context: Optional[np.ndarray] = None,
                     elapsed: Optional[float] = None, **kwargs) -> np.ndarray:
        """Predict the next event given that ``elapsed`` time has passed."""


class TimingPredictive(ABC):
    """Declares that an arm generates *when* as well as *what*.

    Strictly stronger than :class:`TemporallyClocked`: conditioning on a supplied
    interval is not the same as producing one. An arm can absorb elapsed time into
    its state (clocked) and still have no read-out that emits a gap.

    The benchmark this gates is interval-as-output (see
    ``examples/dts_esn_generation_demo.py``): cue the first item and let the arm
    roll out both the items and the gaps between them, then score tempo and
    rhythm against the trained schedule. Note the demo's own control — the two
    benchmarks need *different* stimulus streams, because a gap that carries
    information cannot also be predictable from the items.
    """

    @abstractmethod
    def generate(self, prompt_event: np.ndarray, length: int, **kwargs):
        """Roll out ``(events, gaps)`` under the arm's own schedule."""


def supports_intervals(model_or_class: Any) -> bool:
    """Is this arm clocked by time? Nominal, never duck-typed — see the class."""
    return _declares(model_or_class, TemporallyClocked)


def predicts_timing(model_or_class: Any) -> bool:
    """Can this arm emit the gap as well as the event?"""
    return _declares(model_or_class, TimingPredictive)


def _declares(model_or_class: Any, cap: type) -> bool:
    return (isinstance(model_or_class, cap)
            or (isinstance(model_or_class, type) and issubclass(model_or_class, cap)))


class RolloutMode(str, Enum):
    """What an arm's own ``recall()`` feeds back at each autoregressive step.

    ``measure_recall_autoregressive`` drives ``predict_next`` itself and takes
    an explicit ``feedback_mode`` ("raw" / "l2" / "quantized"), so the protocol
    is a declared condition there. ``model.recall()`` is the *other* rollout
    path -- scored by ``benchmarks/pattern_completion.py`` -- and each arm
    implements it itself.
    This enum makes that per-arm choice explicit, because it is not a detail:
    it is the same axis the symbolic suite deliberately sweeps.
    """

    #: Raw prediction fed straight back as the next cue. Output magnitude is
    #: unconstrained, so state drifts off the encoder's manifold and error
    #: compounds. Equivalent to ``feedback_mode="raw"``.
    OBSERVATION = "observation"

    #: The prediction is decoded to a symbol and the symbol's *clean* codebook
    #: embedding is fed back. Equivalent to ``feedback_mode="quantized"``, which
    #: the symbolic suite documents as a "drift-free upper bound ... [that]
    #: hides sub-threshold representational degradation."
    ENCODER = "encoder"

    #: The rollout runs in latent space; observations are read out per step but
    #: never fed back. Error still compounds, but in the latent metric, not the
    #: observation metric.
    LATENT = "latent"

    #: A latent/recurrent state is carried forward *and* a decoded observation
    #: is fed back into it each step.
    HYBRID = "hybrid"


def rollout_mode_of(model_or_class: Any) -> Optional[RolloutMode]:
    """The arm's declared ``recall()`` rollout protocol, or None if undeclared."""
    return getattr(model_or_class, "rollout_mode", None)


def rollout_modes_comparable(a: Any, b: Any) -> bool:
    """Whether two arms' ``recall()`` scores were produced under the same protocol.

    An ENCODER arm gets codebook cleanup at every step that an OBSERVATION arm
    does not, so their pattern-completion and disambiguation numbers are not
    measuring the same thing. Undeclared (None) is never comparable.
    """
    ma, mb = rollout_mode_of(a), rollout_mode_of(b)
    return ma is not None and ma == mb


def consumes_prompt_trajectory(model_or_class: Any) -> bool:
    """Whether the arm's ``recall`` uses a 2D prompt, or reduces it to ``[-1]``.

    ``HippocampalModel.recall``'s contract says ``prompt_event`` may be "a 2D
    sequence of shape (Time, Features) representing a prompt trajectory", and
    ``benchmarks/pattern_completion.py`` passes the full prefix. Most arms
    then discard everything but the last
    row -- correctly, because their ``predict_next`` is pure and they carry no
    state a longer prompt could establish. This flag records which is which, so
    a floor on a history-dependent task can be read as a model property rather
    than a silently-dropped prompt.
    """
    return bool(getattr(model_or_class, "prompt_conditioned", False))


class StatePrimeable(ABC):
    """Declares a **non-learning** state update: advance carried state without
    touching weights.

    This is the primitive that `benchmarks/spatial_disambiguation.py` asks for
    in prose and cannot call::

        "An arm that DOES carry state must ingest the stem here through a
         state-updating call (not predict_next) before the rollout."

    `fit_event` advances state but also writes weights, so it cannot be used at
    probe time. `predict_next` is pure by contract (independent-probe safety).
    `observe` is the missing middle: state moves, plasticity does not.

    Only arms that genuinely carry state between steps should declare it.
    Declaring it on a memoryless arm would claim a capability that does nothing,
    which is exactly the ambiguity these declarations exist to remove.
    """

    @abstractmethod
    def observe(self, event: np.ndarray, context: Optional[np.ndarray] = None) -> None:
        """Advance carried state by one observed event. Must not modify weights."""

    def observe_sequence(self, events: np.ndarray, context: Optional[np.ndarray] = None) -> None:
        """Advance carried state over a prefix, in order."""
        for t, ev in enumerate(np.asarray(events, dtype=float)):
            self.observe(ev, None if context is None else context[t])


def supports_priming(model_or_class: Any) -> bool:
    """Whether the arm declares ``StatePrimeable`` (non-learning state update)."""
    return _declares(model_or_class, StatePrimeable)


def is_stochastic_forward(model_or_class: Any) -> bool:
    """Whether the arm's forward pass is random (repeat probes are new samples)."""
    return bool(getattr(model_or_class, "stochastic_forward", False))
