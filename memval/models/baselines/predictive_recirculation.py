"""
Predictive recirculation sequence memory (Chen, Zhang, Cameron & Sejnowski 2024).

Reference
---------
Chen, Zhang, Cameron & Sejnowski, "Predictive sequence learning in the
hippocampal formation", Neuron 112:2645-2658 (2024). Ported from the published
equations -- Equation 2 of the Results and Equations 5-11 of the STAR Methods
("Predictive recirculation: a biologically plausible learning algorithm").
The authors' repository (github.com/yschen13/HCPrediction, ``Main_local.py``)
carries no license, so nothing from it is used; this is a from-equations port
in the same sense as ``theta_phase.py``.

What is ported, and what is not
-------------------------------
The paper's headline results (Figures 3-6) train the predictive recurrent
autoencoder with backpropagation through time. This arm implements only the
paper's *local* alternative -- the three-rule predictive recirculation scheme of
its final Results section -- because a rule that needs BPTT is not commensurable
with the rest of the roster. The paper validates the local rule on one short
MNIST-PC sequence (STAR Methods: "a single sequence consisting of 6 time steps");
its behaviour under load, delay and overlap is what this benchmark measures, and
the authors themselves flag that the truncated gradient "might ... be difficult
to accumulate ... for long sequences".

The mechanism (Equation 2)
--------------------------
Anatomy: x is the EC / DG input (DG "serving solely as a delay operator"), h is
CA3, and CA1 carries the prediction error dx alongside the prediction.

    h_t     = tanh(W h_{t-1} + U x_{t-1})          CA3 recurrent state
    xhat_t  = V h_t                                 prediction of x_t
    dx_t    = x_t - xhat_t                          CA1 error units
    dh_t    = U dx_t                                error RECIRCULATED through
                                                    the input weights (not V^T)
    dW  ~  diag(1 - h_t^2) dh_t  h_{t-1}^T
    dV  ~  dx_t h_t^T
    dU  ~  diag(1 - h_t^2) dh_t  x_{t-1}^T

``dV`` is the exact gradient of the squared prediction error (Eq. 5). ``dU`` and
``dW`` are the gradient with two approximations (Eqs. 7-11): the temporal term
through ``W`` is truncated at the current step, and the error is carried back to
the hidden layer through ``U`` rather than ``V^T``, on the recirculation
argument (Hinton & McClelland 1987) that ``U`` converges toward ``V^T`` up to
scale (the paper's Figure S7). Every update is a local outer product; nothing
is transposed and nothing is relaxed.

Where it sits against ``MultilayerTemporalPCNetwork``
-----------------------------------------------------
Both arms are rate-coded, prediction-error-driven, carry a tanh latent across
events and learn with local Hebbian outer products. They differ on exactly two
design decisions, which is what makes them a controlled pair:

* **How the latent is obtained.** tPC *infers* ``z_t`` by relaxing an energy
  (``inf_iters`` gradient steps per event). This arm computes ``h_t`` in one
  forward pass; there is no inference loop.
* **How the error reaches the hidden layer.** tPC feeds ``eps_x`` back through
  ``W_out^T`` during relaxation. This arm feeds ``dx`` back through ``U``,
  the forward input weights.

Adaptation to the MemVal interface
----------------------------------
Under the sequence mapping the cue is ``x_t`` and the target ``x_{t+1}``, so
``predict_next(x_t)`` advances a *copy* of the carried CA3 state by ``x_t`` and
reads out ``V h``. It is a pure function of (cue, carried state): the carried
state is never written by a probe, so ``measure_recall_associative``'s
independent probes cannot leak into one another. ``observe`` is the
non-learning state advance (``StatePrimeable``), ``fit_event`` the learning
one. ``recall`` is the paper's own completion protocol -- "following a partial
input sequence to the network, it completed the remaining sequences" -- the
readout is fed back as the next input while the CA3 state is carried, which is
``RolloutMode.HYBRID``.

Deviations from the paper, to carry into any writeup
-----------------------------------------------------
- Plain SGD on the raw local update; no optimiser. Same choice as the tPC arm.
- The first event of a stream primes ``x_{t-1}`` and forms no transition (the
  paper's sequence starts from ``h_0 = 0`` and predicts from ``t = 1``).
- The learning rate and hidden width have no paper value for the local rule;
  the registry follows the tPC arm's so the pair differs only where the
  equations differ.
"""

from typing import Optional

import numpy as np

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode, StatePrimeable


class PredictiveRecirculationNetwork(HippocampalModel, OnlineTrainable, StatePrimeable):
    """Chen et al. 2024's predictive recirculation rule as a sequence arm."""

    #: `recall` threads the whole prompt through the CA3 state before rolling.
    prompt_conditioned = True

    #: `recall` carries the CA3 state AND feeds the decoded readout back as the
    #: next input -- the paper's completion protocol.
    rollout_mode = RolloutMode.HYBRID

    #: both paths apply the same ``_step`` per transition.
    online_equivalent = True

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 60,
        learning_rate: float = 0.05,
        n_epochs: int = 100,
        weight_scale: float = 0.05,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_features : int
            Dimensionality of each observation vector.
        n_hidden : int
            CA3 width.
        learning_rate : float
            Step size shared by the three local updates.
        n_epochs : int
            Passes over a sequence in `fit_sequence`.
        weight_scale : float
            Std of the Gaussian init of U, W and V. Must be > 0: with ``U = 0``
            the hidden layer is silent and ``dU`` is identically zero.
        seed : int, optional
            Seed for weight init.
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(seed)

        self.U = self.rng.normal(0.0, weight_scale, (n_hidden, n_features))
        self.W = self.rng.normal(0.0, weight_scale, (n_hidden, n_hidden))
        self.V = self.rng.normal(0.0, weight_scale, (n_features, n_hidden))

        # carried CA3 state and the previous input it will be advanced by
        self._h = np.zeros(n_hidden)
        self._prev_x: Optional[np.ndarray] = None
        self.current_state = np.zeros(n_features)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _advance(self, h_prev: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        """``h_t = tanh(W h_{t-1} + U x_{t-1})``."""
        return np.tanh(self.W @ h_prev + self.U @ x_prev)

    def _deltas(self, h_prev, x_prev, h, x_target):
        """The three local updates of Equation 2 for one transition."""
        dx = x_target - self.V @ h                  # CA1 error units
        dh = self.U @ dx                            # recirculated error
        gate = (1.0 - h ** 2) * dh                  # diag(1 - h^2) dh
        return {
            "W": np.outer(gate, h_prev),
            "V": np.outer(dx, h),
            "U": np.outer(gate, x_prev),
        }

    def _step(self, h_prev: np.ndarray, x_prev: np.ndarray, x_target: np.ndarray,
              apply: bool = True):
        """Advance CA3 by ``x_prev``, predict ``x_target``, apply the updates.

        Returns ``(h_t, deltas)``. With ``apply=False`` the weights are left
        untouched (diagnostics hook)."""
        h = self._advance(h_prev, x_prev)
        d = self._deltas(h_prev, x_prev, h, x_target)
        if apply:
            self.W += self.learning_rate * d["W"]
            self.V += self.learning_rate * d["V"]
            self.U += self.learning_rate * d["U"]
        return h, d

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def fit_sequence(self, sequence_data: np.ndarray, context_data=None,
                     epochs: Optional[int] = None, **kwargs):
        """Train on one sequence: each epoch threads the CA3 state from zero and
        applies the three local updates at every transition.

        ``epochs`` overrides ``self.n_epochs`` for this call."""
        seq = np.asarray(sequence_data, dtype=float)
        if seq.shape[0] < 2:
            return

        # Same seam semantics as streaming, so `fit_sequence` is exactly
        # `[fit_event(x) for x in seq]` per epoch (online_equivalent = True).
        self.on_event_boundary()
        for _epoch in range(self.n_epochs if epochs is None else int(epochs)):
            h = np.zeros(self.n_hidden)
            for t in range(seq.shape[0] - 1):
                h, _ = self._step(h, seq[t], seq[t + 1])
        self.on_event_boundary()

    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None, **kwargs):
        """Consume one streamed event. The first event of a stream primes the
        input buffer; each later event closes a (prev -> event) transition,
        applies the updates, and carries the CA3 state forward."""
        x = np.asarray(event, dtype=float).ravel()
        if self._prev_x is not None:
            self._h, _ = self._step(self._h, self._prev_x, x)
        self._prev_x = x

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(self, current_event, current_context=None, **kwargs) -> np.ndarray:
        """``V tanh(W h + U x)`` from the carried ``h`` -- pure: the carried
        state is read, never written. `current_context` is accepted for
        interface parity and unused."""
        x = np.asarray(current_event, dtype=float).ravel()
        pred = self.V @ self._advance(self._h, x)
        self.current_state = pred.copy()
        return pred

    def observe(self, event: np.ndarray, context: Optional[np.ndarray] = None) -> None:
        """Advance the carried CA3 state by one observed event; no learning."""
        x = np.asarray(event, dtype=float).ravel()
        self._h = self._advance(self._h, x)
        self._prev_x = x

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context=None, **kwargs) -> np.ndarray:
        """The paper's sequence completion: drive CA3 with the prompt, then let
        the readout become the next input while the state is carried."""
        prompt = np.asarray(prompt_event, dtype=float)
        prompt = prompt if prompt.ndim > 1 else prompt[np.newaxis, :]

        h = self._h.copy()
        for row in prompt:
            h = self._advance(h, row)
        x = self.V @ h

        recalled = np.zeros((length, self.n_features))
        for t in range(length):
            recalled[t] = x
            h = self._advance(h, x)
            x = self.V @ h

        self.current_state = recalled[-1].copy() if length else np.zeros(self.n_features)
        return recalled

    def get_latent_state(self) -> dict:
        return {"U": self.U.copy(), "W": self.W.copy(), "V": self.V.copy(),
                "h": self._h.copy()}

    def reset_context(self):
        """Zero the carried CA3 state and the input buffer; weights untouched."""
        self._h = np.zeros(self.n_hidden)
        self._prev_x = None
        self.current_state = np.zeros(self.n_features)

    # ------------------------------------------------------------------
    # Diagnostics capability interface (consumed by memval.diagnostics)
    # ------------------------------------------------------------------
    def named_representations(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float).ravel()
        h = self._advance(self._h, x)
        return {"input": x, "hidden": h, "output": self.V @ h}

    def named_parameters(self) -> dict:
        return {"U": self.U, "W": self.W, "V": self.V}

    def transition_grads(self, x_t: np.ndarray, x_next: np.ndarray) -> dict:
        """Local update at the current weights for the (x_t -> x_next)
        transition from the carried state. Weights are not modified."""
        _, d = self._step(self._h, np.asarray(x_t, dtype=float).ravel(),
                          np.asarray(x_next, dtype=float).ravel(), apply=False)
        return d
