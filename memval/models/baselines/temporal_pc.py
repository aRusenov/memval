"""
Temporal Predictive Coding (tPC) Sequence Networks.

A biologically plausible sequence memory trained by local prediction-error
minimisation (Tang, Barron & Bogacz, *Sequential Memory with Temporal Predictive
Coding*, NeurIPS 2023).

Where EP learns from the *difference between two settled states* (a contrastive
rule requiring two relaxation phases), tPC learns from an explicit **prediction
error** carried by dedicated error units: at each step the network predicts the
next observation, and every weight update is the outer product of a local error
with a local activation. One phase, no nudging, no contrastive comparison.

Two models are provided, matching the reference implementation
(https://github.com/C16Mftang/sequential-memory, `src/models.py`):

``TemporalPCNetwork`` — single-layer
    x̂_t = W_r · f(x_{t-1}).  The paper's Theorem shows this is a classical
    **Asymmetric Hopfield Network with an implicit whitening** of the input
    covariance: where AHN sets W = Σ_t x_{t+1}x_tᵀ, gradient descent on the tPC
    energy converges to W = (Σ_t x_{t+1}x_tᵀ)(Σ_t x_t x_tᵀ)⁻¹. It is therefore the
    controlled comparison against `AsymmetricHopfieldNetwork` — same
    architecture, the whitening being the only difference.

``MultilayerTemporalPCNetwork`` — two-layer
    A recurrent *latent* z_t predicts both itself and the observation:
        pred_z = W_r · f(z_{t-1});   pred_x = W_out · f(z_t)
    z_t is inferred by relaxation (gradient descent on the energy) before the
    weights move. This is the paper's headline model; the latent gives it memory
    beyond a first-order Markov transition.

Energy (two-layer; the single-layer is the same without the first term)
----------------------------------------------------------------------
    F = ‖z_t − W_r f(z_{t-1})‖²  +  ‖x_t − W_out f(z_t)‖²
        └──── ε_z ────┘              └──── ε_x ────┘

Inference (relax z_t, weights fixed)
    Δz = ε_z − f'(z_t) ⊙ (W_outᵀ ε_x)
    z_t ← z_t − inf_lr · Δz

Learning (after relaxation; both rules are local Hebbian outer products)
    W_r   += lr · ε_z ⊗ f(z_{t-1})
    W_out += lr · ε_x ⊗ f(z_t)

Adaptation to the MemVal interface
----------------------------------
The reference trains on one long sequence and scores whole-sequence recall.
MemVal instead probes `predict_next` at each position independently with a noisy
cue (`measure_recall_associative`). ``predict_next`` is therefore implemented as
a **pure function of the cue** — it infers the latent from the cue and does not
mutate the carried state — so independent probes cannot leak state into one
another. ``recall`` threads the latent explicitly for its autoregressive rollout.
See the note on `MultilayerTemporalPCNetwork.predict_next`.

Deviations from the reference implementation (deliberate, documented)
---------------------------------------------------------------------
* **Plain SGD, not Adam.** The reference optimises with Adam; we apply the raw
  local update. This keeps the learning rule local/biologically plausible (the
  paper's own framing) and makes the comparison against EP's plain-SGD updates
  fair. It does mean absolute numbers are not directly comparable to the paper's.
* **NumPy, not PyTorch autograd.** The gradients above are analytic; no autograd
  is needed and the rest of the `baselines` package is NumPy.
* **First event primes, no update.** The reference seeds `prev` with a random
  (Kaiming) vector at t=0; here the first event of a sequence only primes the
  buffer, matching the transition semantics the rest of MemVal uses.
"""

from typing import Optional

import numpy as np

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode, StatePrimeable


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_deriv(x: np.ndarray) -> np.ndarray:
    """f'(x) for f = tanh, evaluated at the *pre-activation* x."""
    return 1.0 - np.tanh(x) ** 2


class TemporalPCNetwork(HippocampalModel, OnlineTrainable):
    """
    Single-layer temporal predictive coding.

    Architecture
    ------------
    x_{t-1} ──f──► W_r ──► x̂_t        (error ε_t = x_t − x̂_t drives learning)

    Equivalent to an Asymmetric Hopfield Network with implicit whitening of the
    input covariance (see module docstring), so it doubles as the controlled
    comparison against `AsymmetricHopfieldNetwork`.
    """

    #: `recall` feeds the raw `predict_next` output straight back as the next cue.
    rollout_mode = RolloutMode.OBSERVATION

    #: both paths apply the same ``_transition_delta`` per transition.
    online_equivalent = True

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        nonlinearity: str = "tanh",
        weight_scale: float = 0.0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_features : int
            Dimensionality of each observation vector.
        learning_rate : float
            Step size for the local error-driven update.
        n_epochs : int
            Passes over a sequence in `fit_sequence`.
        nonlinearity : {'tanh', 'linear'}
            f in x̂_t = W_r f(x_{t-1}). 'linear' recovers the exact AHN-with-
            whitening correspondence; 'tanh' matches the reference default.
        weight_scale : float
            Std of the Gaussian weight init. 0.0 (the default) starts from a
            blank memory, as an associative memory should.
        seed : int, optional
            Seed for weight init.
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.nonlinearity = nonlinearity
        self.rng = np.random.default_rng(seed)

        if weight_scale > 0:
            self.W_r = self.rng.normal(0.0, weight_scale, (n_features, n_features))
        else:
            self.W_r = np.zeros((n_features, n_features))

        self.current_state = np.zeros(n_features)
        # online-streaming buffer: previous event, so `fit_event` can form a
        # (prev -> current) transition. None means "start of stream".
        self._prev_event: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _f(self, x: np.ndarray) -> np.ndarray:
        return _tanh(x) if self.nonlinearity == "tanh" else x

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _transition_delta(self, x_t: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """Local error-driven update for one (x_t -> x_next) transition.

        Shared by the batch (`fit_sequence`) and online (`fit_event`) paths so
        both ingestion modes learn with identical math; only the update schedule
        differs. Returns the un-scaled delta for W_r.
        """
        f_prev = self._f(x_t)
        err = x_next - self.W_r @ f_prev
        return np.outer(err, f_prev)

    def fit_sequence(self, sequence_data: np.ndarray, context_data=None, epochs: Optional[int] = None, **kwargs):
        """Train on one sequence: for each epoch, one local update per transition.

        ``epochs`` overrides ``self.n_epochs`` for this call — the symbolic
        pipeline drives its duration sweeps that way."""
        seq_len = sequence_data.shape[0]
        if seq_len < 2:
            return

        # Fence the batch path with the same seam semantics streaming uses, so
        # `fit_sequence` is exactly `[fit_event(x) for x in seq]` and no stale
        # carried state leaks in or out (online_equivalent = True).
        self.on_event_boundary()

        for _epoch in range(self.n_epochs if epochs is None else int(epochs)):
            for t in range(seq_len - 1):
                self.W_r += self.learning_rate * self._transition_delta(
                    sequence_data[t], sequence_data[t + 1]
                )
        self.on_event_boundary()

    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None, **kwargs):
        """Consume one streamed event; apply a single local update for the
        (prev -> event) transition. The first event of a stream only primes the
        buffer. Call `on_event_boundary` at a sequence seam so no spurious
        transition forms across it."""
        x = np.asarray(event, dtype=float).ravel()
        if self._prev_event is not None:
            self.W_r += self.learning_rate * self._transition_delta(self._prev_event, x)
        self._prev_event = x

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(self, current_event, current_context=None, **kwargs) -> np.ndarray:
        """Pure function of the cue: x̂ = W_r f(x). `current_context` is accepted
        for interface parity and unused."""
        x = np.asarray(current_event, dtype=float).ravel()
        pred = self.W_r @ self._f(x)
        self.current_state = pred.copy()
        return pred

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context=None, **kwargs) -> np.ndarray:
        """Autoregressive rollout from the last vector of the prompt."""
        prompt_event = np.asarray(prompt_event, dtype=float)
        current = prompt_event[-1].copy() if prompt_event.ndim > 1 else prompt_event.copy()

        recalled = np.zeros((length, self.n_features))
        for t in range(length):
            current = self.predict_next(current)
            recalled[t] = current
        return recalled

    def get_latent_state(self) -> dict:
        return {"W_r": self.W_r.copy()}

    def reset_context(self):
        """Clear transient state and the online previous-event buffer; weights
        untouched."""
        self.current_state = np.zeros(self.n_features)
        self._prev_event = None

    # ------------------------------------------------------------------
    # Diagnostics capability interface (consumed by memval.diagnostics)
    # ------------------------------------------------------------------
    def named_representations(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float).ravel()
        return {"input": x, "output": self.W_r @ self._f(x)}

    def named_parameters(self) -> dict:
        return {"W_r": self.W_r}

    def transition_grads(self, x_t: np.ndarray, x_next: np.ndarray) -> dict:
        return {
            "W_r": self._transition_delta(
                np.asarray(x_t, dtype=float).ravel(),
                np.asarray(x_next, dtype=float).ravel(),
            )
        }


class MultilayerTemporalPCNetwork(HippocampalModel, OnlineTrainable, StatePrimeable):
    """
    Two-layer temporal predictive coding with an inferred recurrent latent.

    Architecture
    ------------
    z_{t-1} ──f──► W_r ──► pred_z ─┐
                                   ├──► z_t (relaxed) ──f──► W_out ──► x̂_t
    x_t ───────────────────────────┘   (ε_x feeds back into the relaxation)

    Unlike the single-layer model — and unlike the EP baselines, which settle
    from zero every transition — this model carries a latent across events, so
    it is not restricted to a first-order Markov transition map.
    """

    #: `recall` threads the whole prompt through `_infer`, so the latent it rolls
    #: forward from is built from the entire prefix, not just its last row.
    prompt_conditioned = True

    #: `recall` infers the cue's latent once, then runs the latent forward
    #: generatively, reading out an observation per step without ever feeding
    #: one back (the reference's generative recall).
    rollout_mode = RolloutMode.LATENT

    #: both paths thread the latent through the same ``_step``.
    online_equivalent = True

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 128,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        inf_iters: int = 100,
        inf_lr: float = 0.01,
        nonlinearity: str = "tanh",
        weight_scale: float = 0.05,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_hidden : int
            Latent width.
        inf_iters, inf_lr : int, float
            Relaxation steps and step size for inferring z_t. Reference defaults
            are 100 / 1e-2; the paper notes these need little tuning.
        weight_scale : float
            Std of the Gaussian weight init. Must be > 0 here: a zero W_out
            gives an identically-zero prediction and no error to learn from.
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.inf_iters = inf_iters
        self.inf_lr = inf_lr
        self.nonlinearity = nonlinearity
        self.rng = np.random.default_rng(seed)

        self.W_r = self.rng.normal(0.0, weight_scale, (n_hidden, n_hidden))
        self.W_out = self.rng.normal(0.0, weight_scale, (n_features, n_hidden))

        # carried latent (the model's transient memory across events)
        self._prev_z = np.zeros(n_hidden)
        self.current_state = np.zeros(n_features)

    # ------------------------------------------------------------------
    def _f(self, x: np.ndarray) -> np.ndarray:
        return _tanh(x) if self.nonlinearity == "tanh" else x

    def _f_deriv(self, x: np.ndarray) -> np.ndarray:
        return _tanh_deriv(x) if self.nonlinearity == "tanh" else np.ones_like(x)

    # ------------------------------------------------------------------
    # Inference: relax z_t given the observation and the carried latent
    # ------------------------------------------------------------------
    def _infer(self, x: np.ndarray, prev_z: np.ndarray) -> np.ndarray:
        """Relax z_t to a minimum of F = ‖z − W_r f(prev_z)‖² + ‖x − W_out f(z)‖².

        Initialised at the top-down prediction (a forward pass), as in the
        reference, then descended for `inf_iters` steps.
        """
        pred_z = self.W_r @ self._f(prev_z)
        z = pred_z.copy()

        for _ in range(self.inf_iters):
            err_z = z - pred_z
            err_x = x - self.W_out @ self._f(z)
            delta_z = err_z - self._f_deriv(z) * (self.W_out.T @ err_x)
            z = z - self.inf_lr * delta_z

        return z

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _step(self, x: np.ndarray, prev_z: np.ndarray, apply: bool = True):
        """One tPC timestep: infer z_t, then apply the two local updates.

        Returns ``(z_t, dW_r, dW_out)``. With ``apply=False`` the weights are
        left untouched and only the deltas are returned (used by the diagnostics
        hook).
        """
        z = self._infer(x, prev_z)

        f_prev = self._f(prev_z)
        f_z = self._f(z)
        err_z = z - self.W_r @ f_prev
        err_x = x - self.W_out @ f_z

        dW_r = np.outer(err_z, f_prev)
        dW_out = np.outer(err_x, f_z)

        if apply:
            self.W_r += self.learning_rate * dW_r
            self.W_out += self.learning_rate * dW_out

        return z, dW_r, dW_out

    def fit_sequence(self, sequence_data: np.ndarray, context_data=None, epochs: Optional[int] = None, **kwargs):
        """Train on one sequence. Each epoch sweeps the sequence, threading the
        latent forward and applying the local updates at every step.

        ``epochs`` overrides ``self.n_epochs`` for this call."""
        seq_len = sequence_data.shape[0]
        if seq_len < 1:
            return

        # Fence the batch path with the same seam semantics streaming uses, so
        # `fit_sequence` is exactly `[fit_event(x) for x in seq]` and no stale
        # carried state leaks in or out (online_equivalent = True).
        self.on_event_boundary()

        for _epoch in range(self.n_epochs if epochs is None else int(epochs)):
            prev_z = np.zeros(self.n_hidden)
            for t in range(seq_len):
                z, _, _ = self._step(np.asarray(sequence_data[t], dtype=float).ravel(), prev_z)
                prev_z = z
        self.on_event_boundary()

    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None, **kwargs):
        """Consume one streamed event: infer its latent given the carried one,
        apply the local updates, and carry the new latent forward.

        Note the contrast with the single-layer model and with EP: there is no
        previous-*event* buffer, because the latent itself is the carried state.
        Every event produces an update, including the first of a stream."""
        x = np.asarray(event, dtype=float).ravel()
        z, _, _ = self._step(x, self._prev_z)
        self._prev_z = z

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(self, current_event, current_context=None, **kwargs) -> np.ndarray:
        """Predict x_{t+1} from the cue x_t.

        **Pure by design.** MemVal's `measure_recall_associative` probes each
        position independently with a noisy cue, so a `predict_next` that
        mutated the carried latent would leak state between probes that are
        meant to be independent. This method therefore infers the cue's latent
        from `self._prev_z` (zeroed by `reset_context`) *without* updating it —
        the reference's "offline query" mode. `recall` threads the latent
        explicitly for the autoregressive case."""
        x = np.asarray(current_event, dtype=float).ravel()
        z = self._infer(x, self._prev_z)
        pred = self.W_out @ self._f(self.W_r @ self._f(z))
        self.current_state = pred.copy()
        return pred

    def observe(self, event: np.ndarray, context: Optional[np.ndarray] = None) -> None:
        """Advance the carried latent by one observed event, without learning.

        `_infer` relaxes the latent to a minimum of the free energy; it touches
        no weights. This is the probe-time counterpart of `fit_event`, which
        does the same inference and then applies the local updates.
        """
        x = np.asarray(event, dtype=float).ravel()
        self._prev_z = self._infer(x, self._prev_z)

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context=None, **kwargs) -> np.ndarray:
        """Autoregressive rollout: infer the latent of the cue, then run the
        latent forward generatively, reading out an observation at each step.

        This is the reference's generative recall — the rollout happens in
        *latent* space rather than by feeding predictions back through the
        input, so it does not suffer the magnitude drift that autoregressive
        feedback causes in the observation-space models."""
        prompt_event = np.asarray(prompt_event, dtype=float)

        # Thread the WHOLE prompt through inference, not just its last row. The
        # latent is this arm's carried state, so a prefix is exactly the thing
        # that can disambiguate two routes sharing their final observation --
        # and `benchmarks/pattern_completion.py` already passes the full
        # prefix. Reducing it to `prompt_event[-1]` (the previous
        # behaviour) discarded that, which made the only latent-state arm in the
        # suite behave like a memoryless one at retrieval. Inference does not
        # touch weights, so this stays a non-learning probe.
        prompt = prompt_event if prompt_event.ndim > 1 else prompt_event[np.newaxis, :]
        z = self._prev_z
        for row in prompt:
            z = self._infer(row, z)

        recalled = np.zeros((length, self.n_features))
        for t in range(length):
            z = self.W_r @ self._f(z)
            recalled[t] = self.W_out @ self._f(z)

        self.current_state = recalled[-1].copy() if length else np.zeros(self.n_features)
        return recalled

    def get_latent_state(self) -> dict:
        return {"W_r": self.W_r.copy(), "W_out": self.W_out.copy(), "z": self._prev_z.copy()}

    def reset_context(self):
        """Zero the carried latent and the transient readout; weights untouched.
        This is the event-boundary semantics: the latent is exactly the state
        that must not bridge two different sequences."""
        self._prev_z = np.zeros(self.n_hidden)
        self.current_state = np.zeros(self.n_features)

    # ------------------------------------------------------------------
    # Diagnostics capability interface (consumed by memval.diagnostics)
    # ------------------------------------------------------------------
    def named_representations(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float).ravel()
        z = self._infer(x, self._prev_z)
        return {"input": x, "hidden": z, "output": self.W_out @ self._f(self.W_r @ self._f(z))}

    def named_parameters(self) -> dict:
        return {"W_r": self.W_r, "W_out": self.W_out}

    def transition_grads(self, x_t: np.ndarray, x_next: np.ndarray) -> dict:
        """Local update at the current weights for the step that *observes*
        `x_next`, having just seen `x_t`. Weights are not modified."""
        z_t = self._infer(np.asarray(x_t, dtype=float).ravel(), self._prev_z)
        _, dW_r, dW_out = self._step(
            np.asarray(x_next, dtype=float).ravel(), z_t, apply=False
        )
        return {"W_r": dW_r, "W_out": dW_out}
