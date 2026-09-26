"""
Diverse-TimeScale Echo State Network (DTS-ESN).

Reference
---------
Tanaka et al., "Reservoir computing with diverse timescales for prediction of
multiscale dynamics", Phys. Rev. Research 4, L032014 (2022) — a leaky-integrator
ESN whose neurons carry *distributed* leak rates. Leaky-integrator ESN itself:
Jaeger, Lukosevicius, Popovici & Siewert, Neural Networks 20(3):335-352 (2007).

Ported to NumPy, no new dependency; the configuration corresponds to
reservoirpy's ``Reservoir(units=N, lr=<array of shape (N,)>)`` followed by an
``RLS`` readout.

Why this arm exists
-------------------
Every other arm in the suite is clocked by *event ordinal*. ``fit_event`` carries
no timestamp, so two streams with identical items and different spacing produce
byte-identical updates. The DTS-ESN is clocked by *time*. Each unit i integrates

    dx_i/dt = a_i * ( -x_i + tanh( W_in u + W x + b )_i )

with the leak rates ``a_i = 1/tau_i`` spread log-spaced over
``[1/tau_max, 1/tau_min]``. Between events the input is zero and the state keeps
evolving, so a gap of duration D leaves a signature: units with tau << D have
decayed, units with tau >> D have barely moved, and the *ratio* of activation
across timescales is a code for elapsed time. The readout reads that code.

This is the reservoir-side route to the same object as a Laplace/SITH temporal
basis — a bank of leaky integrators with geometrically spaced rate constants.

Only the readout is plastic, trained by online recursive least squares, so
``fit_event`` is the primary path rather than an adaptation of a batch rule.

Interval interface
------------------
Elapsed time enters as ``elapsed=`` on ``fit_event`` / ``predict_next``, and as
``intervals=`` on ``fit_sequence`` / ``recall``. Callers that pass nothing get
the old ordinal-clocked behaviour, so this arm stays drop-in compatible with
benchmarks that have no time axis.
"""

from typing import Optional, Sequence

import numpy as np

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode, StatePrimeable, TemporallyClocked, TimingPredictive


class DTSESNSequenceNetwork(HippocampalModel, OnlineTrainable, StatePrimeable,
                           TemporallyClocked, TimingPredictive):
    """Leaky-integrator reservoir with log-spaced per-unit time constants and an
    online RLS readout.

    The reservoir is fixed (random, never trained); all plasticity lives in
    ``W_out``. Elapsed time between events is absorbed by integrating the
    reservoir forward with zero input, never by appending a time feature to the
    input vector.
    """

    #: `_state_from` replays a 2D cue through the reservoir in order.
    prompt_conditioned = True

    #: `recall` threads its own reservoir state AND feeds `decode_prediction(y)`
    #: back into it at every step -- the only arm that honours the
    #: `decode_prediction` cleanup hook inside its own rollout.
    rollout_mode = RolloutMode.HYBRID

    #: ``fit_sequence`` is literally a loop over ``fit_event``.
    online_equivalent = True

    def __init__(
        self,
        n_features: int,
        n_units: int = 400,
        tau_min: float = 0.1,
        tau_max: float = 20.0,
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        connectivity: float = 0.1,
        dt: float = 0.05,
        event_duration: Optional[float] = None,
        rls_lambda: float = 0.999,
        rls_delta: float = 1.0,
        bias_scale: float = 0.0,
        predict_timing: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_features : int
            Dimensionality of each observation vector.
        n_units : int
            Reservoir size.
        tau_min, tau_max : float
            Range of unit time constants, in the same units as ``elapsed``. This
            is the parameter that sets which intervals the arm can resolve: the
            range must bracket the intervals in the benchmark. Rates are spaced
            log-uniformly between ``1/tau_max`` and ``1/tau_min``.
        spectral_radius : float
            Rescaling of the recurrent matrix. Must be < 1 for the zero state to
            be a stable fixed point, which is what makes gaps decay rather than
            wander.
        input_scaling : float
            Gain on ``W_in``.
        connectivity : float
            Density of the recurrent matrix.
        dt : float
            Integration step. Must be <= ``tau_min`` for the fastest units to be
            integrated stably.
        event_duration : float, optional
            How long an event is held on the input. Defaults to ``dt`` (a single
            step), i.e. events are impulses.
        rls_lambda : float
            RLS forgetting factor. 1.0 is no forgetting.
        rls_delta : float
            RLS inverse-covariance init; ``P = I / rls_delta``.
        bias_scale : float
            Std of the constant reservoir bias. 0.0 disables it.
        predict_timing : bool
            Add a second readout that predicts the interval to the next event,
            enabling autonomous rollout via ``generate``. Off by default; when
            off the arm behaves exactly as without this feature. Note the
            timing head can only be learned when the interval is *determined*
            by the history — on a stream where the same prefix is deliberately
            followed by different gaps (the interval-as-cue design) the least-
            squares fit collapses toward the mean gap.
        seed : int, optional
            Seed for reservoir construction.
        """
        super().__init__(**kwargs)

        if dt > tau_min:
            raise ValueError(
                f"dt={dt} exceeds tau_min={tau_min}; the fastest units would be "
                "integrated unstably. Lower dt or raise tau_min."
            )

        self.n_features = n_features
        self.n_units = n_units
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dt = dt
        self.event_duration = dt if event_duration is None else event_duration
        self.rls_lambda = rls_lambda
        self.rls_delta = rls_delta
        self.rng = np.random.default_rng(seed)

        # --- the defining feature: log-spaced time constants -----------------
        self.taus = np.logspace(np.log10(tau_min), np.log10(tau_max), n_units)
        self.a = 1.0 / self.taus                      # per-unit leak rates

        # --- fixed random reservoir -----------------------------------------
        W = self.rng.standard_normal((n_units, n_units))
        mask = self.rng.random((n_units, n_units)) < connectivity
        W *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        if radius > 0:
            W *= spectral_radius / radius
        self.W = W

        self.W_in = self.rng.uniform(-1.0, 1.0, (n_units, n_features)) * input_scaling
        self.bias = (
            self.rng.standard_normal(n_units) * bias_scale if bias_scale > 0
            else np.zeros(n_units)
        )

        # --- plastic readouts (the only things that learn) -------------------
        self.W_out = np.zeros((n_features, n_units + 1))   # +1 for bias term
        self.P = np.eye(n_units + 1) / rls_delta

        # Timing head. It gets its own inverse-covariance rather than extra rows
        # on ``P``: the two readouts are fitted on states sampled at *different*
        # moments (item readout post-gap, timing readout pre-gap), so a shared
        # P would track a mixture of two state distributions and degrade both.
        self.predict_timing = predict_timing
        if predict_timing:
            self.W_time = np.zeros((1, n_units + 1))
            self.P_time = np.eye(n_units + 1) / rls_delta
        else:
            self.W_time = None
            self.P_time = None

        # --- transient state --------------------------------------------------
        self._x = np.zeros(n_units)
        # Set only by `observe`, cleared only by `reset_context`. Lets a probe read
        # deliberately-primed state while un-primed probes stay pure (from rest),
        # which is the independent-probe guarantee measure_recall_associative needs.
        # Default passes per ``fit_sequence`` call when the caller does not name
        # one. Stored 2026-09-03 for the same reason as
        # AsymmetricHopfieldNetwork: MODEL_REGISTRY passes n_epochs to every
        # constructor, this arm dropped it into **kwargs, and
        # SpatialReversalBenchmark's exposure contract is the constructor value
        # (it calls fit_sequence with no epochs kwarg). This arm and AHN were
        # the only two in the registry not honouring it.
        self.n_epochs = int(kwargs.get("n_epochs", 1))
        self._primed = False
        self._seen = False

    # ------------------------------------------------------------------
    # Reservoir dynamics
    # ------------------------------------------------------------------
    def _advance(self, x: np.ndarray, u: Optional[np.ndarray], duration: float) -> np.ndarray:
        """Integrate the reservoir for ``duration`` holding input ``u`` (or zero).

        This single routine handles both event injection and inter-event gaps —
        a gap is simply an interval with no input, which is what lets elapsed
        time enter through the dynamics rather than through a feature.
        """
        if duration is None or duration <= 0:
            return x
        n_steps = max(1, int(round(duration / self.dt)))
        drive = self.W_in @ u if u is not None else 0.0
        for _ in range(n_steps):
            pre = self.W @ x + drive + self.bias
            x = x + self.dt * self.a * (-x + np.tanh(pre))
        return x

    def _state_from(
        self,
        cue: np.ndarray,
        elapsed: Optional[float] = None,
        prompt_intervals: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Build a reservoir state from a cue alone, starting from rest.

        ``cue`` may be 1D (a single event) or 2D (a prompt trajectory, replayed
        in order). ``prompt_intervals[t]`` is the gap preceding prompt item t —
        supply it whenever the prompt was ingested with non-uniform spacing,
        otherwise the replayed state will not match the state that was trained
        against. Keeping this a pure function of the cue is what allows
        ``predict_next`` to satisfy the independent-probe contract that
        ``measure_recall_associative`` assumes.
        """
        cue = np.asarray(cue, dtype=float)
        # A 1-D cue after an explicit `observe` prime continues from the carried
        # state -- that is the StatePrimeable contract this class declares. With
        # no prime since the last reset_context, start from rest exactly as
        # before, so independent probes cannot leak into one another.
        x = (self._x.copy() if (self._primed and cue.ndim == 1)
             else np.zeros(self.n_units))
        if cue.ndim > 1:
            for t, row in enumerate(cue):
                gap = prompt_intervals[t] if prompt_intervals is not None else None
                x = self._advance(x, None, gap)
                x = self._advance(x, row, self.event_duration)
        else:
            x = self._advance(x, cue, self.event_duration)
        return self._advance(x, None, elapsed)

    def _readout(self, x: np.ndarray) -> np.ndarray:
        return self.W_out @ np.append(x, 1.0)

    def _rls_update(self, x: np.ndarray, target: np.ndarray) -> None:
        """One recursive-least-squares step on the readout. Genuinely online:
        a single event, a single update, no epoch."""
        z = np.append(x, 1.0)
        Pz = self.P @ z
        gain = Pz / (self.rls_lambda + z @ Pz)
        err = target - self.W_out @ z
        self.W_out += np.outer(err, gain)
        self.P = (self.P - np.outer(gain, Pz)) / self.rls_lambda

    def _readout_time(self, x: np.ndarray) -> float:
        """Predicted interval until the next event, floored at one integration
        step so a rollout can always advance."""
        raw = float((self.W_time @ np.append(x, 1.0))[0])
        return max(self.dt, raw)

    def _rls_update_time(self, x: np.ndarray, gap: float) -> None:
        """One RLS step on the timing head, against its own covariance."""
        z = np.append(x, 1.0)
        Pz = self.P_time @ z
        gain = Pz / (self.rls_lambda + z @ Pz)
        err = np.array([gap]) - self.W_time @ z
        self.W_time += np.outer(err, gain)
        self.P_time = (self.P_time - np.outer(gain, Pz)) / self.rls_lambda

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def fit_event(
        self,
        event: np.ndarray,
        context: Optional[np.ndarray] = None,
        elapsed: Optional[float] = None,
        **kwargs,
    ):
        """Consume one streamed event, optionally preceded by a gap.

        Order matters: the gap is integrated *before* the item readout is
        trained, so the state that readout learns to map from is the post-gap
        state — i.e. the model learns "after this history and this much elapsed
        time, expect this". The timing head is trained on the other side of the
        gap. The first event of a stream only primes the reservoir.
        """
        e = np.asarray(event, dtype=float).ravel()
        # The timing head reads the *pre-gap* state: from where the stream was
        # after the last event, how long until the next one arrives.
        if self._seen and self.predict_timing and elapsed is not None:
            self._rls_update_time(self._x, float(elapsed))
        self._x = self._advance(self._x, None, elapsed)
        # The item head reads the *post-gap* state, so elapsed time is an input
        # to it rather than a target.
        if self._seen:
            self._rls_update(self._x, e)
        self._x = self._advance(self._x, e, self.event_duration)
        self._seen = True

    def observe(
        self,
        event: np.ndarray,
        context: Optional[np.ndarray] = None,
        elapsed: Optional[float] = None,
    ) -> None:
        """Advance the reservoir by one observed event, without learning.

        Exactly `fit_event`'s state path with both RLS updates omitted: let the
        gap elapse, then drive the reservoir with the event. Readout weights and
        the RLS inverse-covariance are untouched.
        """
        e = np.asarray(event, dtype=float).ravel()
        self._x = self._advance(self._x, None, elapsed)
        self._x = self._advance(self._x, e, self.event_duration)
        self._primed = True
        self._seen = True

    def fit_sequence(
        self,
        sequence_data: np.ndarray,
        context_data: Optional[np.ndarray] = None,
        intervals: Optional[Sequence[float]] = None,
        epochs: Optional[int] = None,
        **kwargs,
    ):
        """Stream a sequence through ``fit_event``.

        ``intervals[t]`` is the gap *preceding* event t; ``intervals[0]`` is
        normally 0 or None. Omitting ``intervals`` reproduces ordinal clocking.
        """
        sequence_data = np.asarray(sequence_data, dtype=float)
        if sequence_data.shape[0] < 2:
            return
        for _ in range(self.n_epochs if epochs is None else int(epochs)):
            self.reset_context()
            for t, event in enumerate(sequence_data):
                gap = intervals[t] if intervals is not None else None
                self.fit_event(event, elapsed=gap)
        self.on_event_boundary()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(
        self,
        current_event: np.ndarray,
        current_context: Optional[np.ndarray] = None,
        elapsed: Optional[float] = None,
        prompt_intervals: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Pure function of the cue: rebuild the reservoir state from rest, let
        ``elapsed`` pass, read out. Does not mutate the streaming state, so
        independent probes cannot leak into one another.

        ``current_event`` may be a single vector or a prompt trajectory. Passing
        the trajectory matters for a stateful arm: a bare one-item cue discards
        the history the reservoir would have held during training. Pass
        ``prompt_intervals`` alongside it if that history was non-uniformly
        spaced.
        """
        x = self._state_from(current_event, elapsed, prompt_intervals)
        return self._readout(x)

    def recall(
        self,
        prompt_event: np.ndarray,
        length: int,
        prompt_context: Optional[np.ndarray] = None,
        intervals: Optional[Sequence[float]] = None,
        prompt_intervals: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Autoregressive rollout, threading its own reservoir state.

        ``intervals[t]`` is the gap to let elapse before recalling step t.
        Predictions are passed through ``decode_prediction`` before being fed
        back, so a caller can install codebook cleanup without touching this.
        """
        x = self._state_from(prompt_event, prompt_intervals=prompt_intervals)
        recalled = np.zeros((length, self.n_features))
        for t in range(length):
            gap = intervals[t] if intervals is not None else None
            x = self._advance(x, None, gap)
            y = self._readout(x)
            recalled[t] = y
            x = self._advance(x, self.decode_prediction(y), self.event_duration)
        return recalled

    # ------------------------------------------------------------------
    # Autonomous generation (requires predict_timing=True)
    # ------------------------------------------------------------------
    def _require_timing(self) -> None:
        if not self.predict_timing:
            raise RuntimeError(
                "This model was built with predict_timing=False; construct it "
                "with predict_timing=True to use the timing head."
            )

    def predict_time_to_next(
        self,
        current_event: np.ndarray,
        prompt_intervals: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> float:
        """Predicted interval until the next event, as a pure function of the cue.

        The timing counterpart of ``predict_next``: same purity contract, same
        cue conventions. Note there is no ``elapsed`` argument — this reads the
        pre-gap state, because the gap is what it is predicting.
        """
        self._require_timing()
        x = self._state_from(current_event, prompt_intervals=prompt_intervals)
        return self._readout_time(x)

    def generate(
        self,
        prompt_event: np.ndarray,
        length: int,
        prompt_intervals: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        """Autonomous rollout: the model supplies its own schedule.

        Unlike ``recall``, which is interval-*conditioned* (the caller passes the
        gaps), this is interval-*generating* — at each step the timing head says
        how long to wait, the reservoir is advanced by exactly that, and only
        then does the item head read out. Predictions are passed through
        ``decode_prediction`` before being fed back.

        Returns
        -------
        (events, intervals) : (np.ndarray of shape (length, n_features),
                               np.ndarray of shape (length,))
            The generated events and the gap the model chose to precede each.
        """
        self._require_timing()
        x = self._state_from(prompt_event, prompt_intervals=prompt_intervals)
        events = np.zeros((length, self.n_features))
        intervals = np.zeros(length)
        for t in range(length):
            gap = self._readout_time(x)          # decide when
            x = self._advance(x, None, gap)
            y = self._readout(x)                 # decide what
            events[t] = y
            intervals[t] = gap
            x = self._advance(x, self.decode_prediction(y), self.event_duration)
        return events, intervals

    # ------------------------------------------------------------------
    # Interface plumbing
    # ------------------------------------------------------------------
    def get_latent_state(self) -> dict:
        state = {
            "x": self._x.copy(),
            "W_out": self.W_out.copy(),
            "taus": self.taus.copy(),
        }
        if self.predict_timing:
            state["W_time"] = self.W_time.copy()
        return state

    def reset_context(self):
        """Clear the reservoir and the priming flag. Readout weights and the RLS
        inverse-covariance are long-term memory and stay untouched."""
        self._x = np.zeros(self.n_units)
        self._primed = False
        self._seen = False

    # ------------------------------------------------------------------
    # Diagnostics capability interface (parity with the other arms)
    # ------------------------------------------------------------------
    def named_representations(self, x: np.ndarray) -> dict:
        state = self._state_from(x)
        return {"input": np.asarray(x, dtype=float).ravel(),
                "reservoir": state,
                "output": self._readout(state)}

    def named_parameters(self) -> dict:
        return {"W_out": self.W_out}
