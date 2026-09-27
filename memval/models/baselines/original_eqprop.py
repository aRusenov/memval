"""
Equilibrium Propagation Sequence Network.

An energy-based model adapted for sequence learning using the two-phase
(free / nudge) Equilibrium Propagation learning rule (Scellier & Bengio, 2017).

The core idea: instead of back-propagating errors through a computation graph,
this network finds steady-state equilibria of an energy function and uses tiny
perturbations ("nudges") toward the target to compute a biologically plausible
gradient signal.

Adaptation for Sequences
------------------------
Standard EP operates on static inputs. For temporal / sequence learning we
iterate over consecutive pairs (x_t, x_{t+1}) within a trajectory:
  - x_t  is clamped to the input (visible) layer
  - x_{t+1} is the target used to nudge the output layer
Weight updates accumulate across all transitions and across training epochs.
"""

from typing import Any, Optional

import numpy as np

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode


class OriginalEqPropSequenceNetwork(HippocampalModel, OnlineTrainable):
    """
    A multi-layer energy-based network trained with Equilibrium Propagation,
    adapted for sequential / trajectory prediction tasks.

    Architecture
    ------------
    input  (n_features)  →  hidden (n_hidden, tanh)  →  output (n_features)

    Energy
    ------
    E  = -½ sₕᵀ W₁ sᵢ  -  ½ sₒᵀ W₂ sₕ  -  bₕᵀ sₕ  -  bₒᵀ sₒ  +  ½ ‖sₒ - y‖² · β

    where β is the nudge strength (0 in the free phase, >0 in the nudge phase).

    Learning Rule
    -------------
    ΔW ∝ (1/β) · (s_nudge ⊗ s_pre_nudge  −  s_free ⊗ s_pre_free)
    """

    #: `recall` feeds the raw prediction back via `self.current_state`.
    rollout_mode = RolloutMode.OBSERVATION

    #: ``fit_sequence`` averages the EP delta by ``1 / n_transitions`` and
    #: applies one step per epoch; ``fit_event`` takes a full step per
    #: transition (online SGD). Different rules, not just different ingestion.
    online_equivalent = False

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 64,
        learning_rate: float = 0.01,
        beta: float = 0.5,
        n_settle_steps: int = 50,
        n_epochs: int = 100,
        dt: float = 0.5,
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        hidden_sparsity: Optional[float] = None,
        use_output_bias: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_features : int
            Dimensionality of each observation vector (e.g. 2 for 2D trajectories).
        n_hidden : int
            Number of hidden units.
        learning_rate : float
            Step size for weight updates.
        beta : float
            Nudging strength.  Should be small but positive.
        n_settle_steps : int
            Number of iterative steps to reach steady-state in each phase.
        n_epochs : int
            Number of full passes over the sequence during training.
        dt : float
            Integration step size for the settling dynamics (controls stability).
        activation : str
            Activation function for hidden & output units ('tanh' or 'sigmoid').
        seed : int, optional
            Random seed for reproducibility.
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.beta = beta
        self.n_settle_steps = n_settle_steps
        self.n_epochs = n_epochs
        self.dt = dt
        self.activation_type = activation
        self.output_activation_type = output_activation if output_activation else activation
        # A shared additive output bias is a catastrophic-forgetting liability in
        # continual learning: it captures the targets' shared mean, is written by
        # every task, and is expressively redundant when W_ho is full-rank. Set
        # False to drop it (b_o stays 0, never updated) — improves retention with
        # no learning cost. Inherited by DG/XdG subclasses.
        self.use_output_bias = use_output_bias
        self.rng = np.random.default_rng(seed)

        # ----- initialise parameters -----
        # Glorot (Xavier) Uniform initialization
        limit_ih = np.sqrt(6.0 / (n_features + n_hidden))
        limit_ho = np.sqrt(6.0 / (n_hidden + n_features))

        self.W_ih = self.rng.uniform(-limit_ih, limit_ih, (n_hidden, n_features))   # input  → hidden
        self.W_ho = self.rng.uniform(-limit_ho, limit_ho, (n_features, n_hidden))   # hidden → output
        self.b_h = np.zeros(n_hidden)
        self.b_o = np.zeros(n_features)

        # transient state used during recall
        self.current_state = np.zeros(n_features)

        # online-streaming buffer: the previously seen event, held so that
        # fit_event can form a (prev -> current) transition and apply a single
        # EP update per event. None means "no transition yet" (start of stream).
        self._prev_event: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------
    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation_type == "tanh":
            return np.tanh(x)
        elif self.activation_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return x  # linear fallback

    def _activate_output(self, x: np.ndarray) -> np.ndarray:
        """Output-layer activation (may differ from hidden)."""
        if self.output_activation_type == "clip":
            return np.clip(x, 0.0, 1.0)
        elif self.output_activation_type == "tanh":
            return np.tanh(x)
        elif self.output_activation_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return x  # linear fallback

    def _activate_deriv(self, activated: np.ndarray) -> np.ndarray:
        """Derivative of activation given the *already-activated* value."""
        if self.activation_type == "tanh":
            return 1.0 - activated ** 2
        elif self.activation_type == "sigmoid":
            return activated * (1.0 - activated)
        return np.ones_like(activated)

    # ------------------------------------------------------------------
    # Settling dynamics
    # ------------------------------------------------------------------
    def _settle(
        self,
        x_input: np.ndarray,
        target: Optional[np.ndarray] = None,
        beta: float = 0.0,
    ):
        """
        Run the network dynamics until (approximate) steady-state.

        Parameters
        ----------
        x_input : (n_features,) clamped input.
        target  : (n_features,) nudge target (None → free phase).
        beta    : nudge strength (0 → free phase).

        Returns
        -------
        s_h, s_o : steady-state hidden and output activations.
        """
        # initialise layers at zero (pre-activations)
        s_h = np.zeros(self.n_hidden)
        s_o = np.zeros(self.n_features)

        for _ in range(self.n_settle_steps):
            # Evaluate activations (in Original EP, the state is clipped to [0,1])
            # So the state itself serves as its own activation
            
            # ----- hidden unit dynamics -----
            current_h = self.W_ih @ x_input + self.W_ho.T @ s_o + self.b_h
            s_h = np.clip(s_h + self.dt * (-s_h + current_h), 0.0, 1.0)

            # ----- output unit dynamics -----
            current_o = self.W_ho @ s_h + self.b_o
            
            nudge_term = np.zeros_like(s_o)
            if target is not None and beta > 0:
                # The gradient of cost = ||s_o - target||^2 adds a factor of 2
                nudge_term = 2.0 * beta * (target - s_o)
                
            s_o = np.clip(s_o + self.dt * (-s_o + current_o + nudge_term), 0.0, 1.0)

        return s_h, s_o

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _transition_deltas(self, x_t: np.ndarray, x_next: np.ndarray):
        """Compute the raw EP weight/bias deltas for one (x_t -> x_next) pair.

        This is the shared learning rule used by both the batch trainer
        (``fit_sequence``, which accumulates and averages these deltas) and the
        online trainer (``fit_event``, which applies one immediately). Keeping a
        single implementation guarantees the two ingestion modes learn with
        identical math; only the *update schedule* differs.

        Returns ``(dW_ih, dW_ho, db_h, db_o)`` — the un-scaled EP deltas.
        """
        # free phase: clamp x_t, settle with no target
        s_h_free, s_o_free = self._settle(x_t, target=None, beta=0.0)
        # nudge phase: clamp x_t, nudge the output toward x_next
        s_h_nudge, s_o_nudge = self._settle(x_t, target=x_next, beta=self.beta)

        # states are already [0,1]-clipped, so they double as their activations
        inv_beta = 1.0 / self.beta
        dW_ho = inv_beta * (
            np.outer(s_o_nudge, s_h_nudge) - np.outer(s_o_free, s_h_free)
        )
        dW_ih = inv_beta * (
            np.outer(s_h_nudge, x_t) - np.outer(s_h_free, x_t)
        )
        db_h = inv_beta * (s_h_nudge - s_h_free)
        db_o = inv_beta * (s_o_nudge - s_o_free)
        return dW_ih, dW_ho, db_h, db_o

    def fit_sequence(self, sequence_data: np.ndarray, **kwargs):
        """
        Train the model on a single sequence using Equilibrium Propagation.

        Each consecutive pair (x_t, x_{t+1}) produces one weight update:
          free  phase:  clamp x_t, settle freely          → s_h_free, s_o_free
          nudge phase:  clamp x_t, nudge toward x_{t+1}   → s_h_nudge, s_o_nudge

        ΔW_ho ∝ (1/β)(s_o_nudge ⊗ s_h_nudge − s_o_free ⊗ s_h_free)
        ΔW_ih ∝ (1/β)(s_h_nudge ⊗ x_t_nudge − s_h_free ⊗ x_t_free)
        """
        seq_len = sequence_data.shape[0]
        if seq_len < 2:
            return

        n_transitions = seq_len - 1

        # Honour a call-level ``epochs=``; fall back to the constructor value.
        # ``epochs_to_criterion`` and every exposure staircase document
        # "trains for exactly ``epochs`` passes", and 13 of the 26
        # ``fit_sequence`` call sites in the suite rely on it without also
        # pinning ``n_epochs`` on the instance. Reading ``self.n_epochs``
        # unconditionally silently multiplied every such step by up to 100x
        # (continual_chain steps one epoch at a time; each step ran 100).
        for _epoch in range(int(kwargs.get('epochs', self.n_epochs))):
            # accumulators
            dW_ih = np.zeros_like(self.W_ih)
            dW_ho = np.zeros_like(self.W_ho)
            db_h = np.zeros_like(self.b_h)
            db_o = np.zeros_like(self.b_o)

            for t in range(n_transitions):
                d_ih, d_ho, d_bh, d_bo = self._transition_deltas(
                    sequence_data[t], sequence_data[t + 1]
                )
                dW_ih += d_ih
                dW_ho += d_ho
                db_h += d_bh
                db_o += d_bo

            # average over transitions and apply
            lr = self.learning_rate / n_transitions
            self.W_ho += lr * dW_ho
            self.W_ih += lr * dW_ih
            self.b_h += lr * db_h
            if self.use_output_bias:
                self.b_o += lr * db_o

            # extension hook (no-op here; used e.g. by EWC to pull weights
            # toward a consolidated anchor after each epoch's data update)
            self._on_epoch_end()

    def _on_epoch_end(self) -> None:
        """Called once per training epoch after weights are updated. No-op by
        default; subclasses/mixins may override to add per-epoch regularisation."""
        pass

    # ------------------------------------------------------------------
    # Online / streaming training
    # ------------------------------------------------------------------
    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None, **kwargs):
        """Consume one streamed event and apply a single online EP update.

        This is the streaming counterpart of ``fit_sequence``: instead of
        buffering a whole sequence and updating once per epoch over averaged
        deltas, it forms the transition ``(prev_event -> event)`` from the last
        buffered event and applies that transition's EP delta immediately, then
        stores ``event`` as the new ``prev``. The first event of a stream only
        primes the buffer (no pair yet), so it produces no update.

        Update magnitude is ``learning_rate`` per transition — the online SGD
        convention of one full step per sample. Note this is un-averaged, unlike
        ``fit_sequence`` which divides by ``n_transitions``; so a single streamed
        presentation of an L-item list applies ~L aggressive updates. To bring a
        one-shot online pass into the same regime as one offline epoch, lower
        ``learning_rate`` (roughly by 1/L) or raise ``n_presentations``.

        This architecture carries no recurrent hidden state across events (each
        settle starts from zero; ``predict_next`` is a pure function of the
        clamped event), so ``prev_event`` is the only cross-call state online
        training needs. ``context`` is accepted for interface parity but is
        unused by this model. Call ``on_event_boundary`` (or ``reset_context``)
        at a sequence boundary to avoid forming a spurious transition across the
        seam between two different sequences.
        """
        x = np.asarray(event).flatten()
        if self._prev_event is not None:
            d_ih, d_ho, d_bh, d_bo = self._transition_deltas(self._prev_event, x)
            self.W_ih += self.learning_rate * d_ih
            self.W_ho += self.learning_rate * d_ho
            self.b_h += self.learning_rate * d_bh
            if self.use_output_bias:
                self.b_o += self.learning_rate * d_bo
            self._on_update_end()
        self._prev_event = x

    def _on_update_end(self) -> None:
        """Called after each online (per-event) weight update. No-op by default;
        the streaming analogue of ``_on_epoch_end`` for mixins (e.g. EWC) that
        regularise per update rather than per epoch."""
        pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(self, current_event: np.ndarray, **kwargs) -> np.ndarray:
        """
        Single-step prediction: clamp current_event, run free-phase settle,
        return the output-layer steady state as the predicted next event.
        """
        x = current_event.flatten()
        _s_h, s_o = self._settle(x, target=None, beta=0.0)
        self.current_state = s_o.copy()
        return self.current_state.copy()

    def recall(self, prompt_event: np.ndarray, length: int) -> np.ndarray:
        """
        Auto-regressive recall: seed with the last vector of the prompt,
        then iteratively predict_next for `length` steps.
        """
        if len(prompt_event.shape) > 1:
            current = prompt_event[-1].copy()
        else:
            current = prompt_event.copy()

        self.current_state = current
        recalled = np.zeros((length, self.n_features))

        for t in range(length):
            next_val = self.predict_next(self.current_state)
            recalled[t] = next_val
            self.current_state = next_val

        return recalled

    def get_latent_state(self) -> np.ndarray:
        """Return current weight matrices as a diagnostic-friendly dict."""
        return {
            "W_ih": self.W_ih.copy(),
            "W_ho": self.W_ho.copy(),
            "b_h": self.b_h.copy(),
            "b_o": self.b_o.copy(),
        }

    def reset_context(self):
        """Reset transient state without touching learned weights. Also clears
        the online previous-event buffer, so a new stream (or the segment after
        an event boundary) starts a fresh chain of transitions rather than
        bridging across the seam. The inherited ``on_event_boundary`` calls this."""
        self.current_state = np.zeros(self.n_features)
        self._prev_event = None

    # ------------------------------------------------------------------
    # Diagnostics capability interface (consumed by memval.diagnostics)
    # ------------------------------------------------------------------
    # These thin wrappers expose the model to the model-agnostic diagnostics
    # (representation geometry, weight-update interference, Fisher attribution)
    # without those utilities reaching into private internals. Subclasses that
    # change the input pathway (e.g. a DG separator) or the settling dynamics
    # (e.g. XdG gating) are picked up automatically: `named_representations`
    # routes through `_separate`/`_settle` if present, and `transition_grads`
    # wraps the overridable `_transition_deltas`.

    def named_representations(self, x: np.ndarray) -> dict:
        """Free-phase activations per named layer for a single raw input.

        Returns ``{"input", "hidden", "output"}``. ``input`` is the code actually
        clamped to the visible layer — the raw input here, or the sparse DG code
        in DG-based subclasses (via `_separate`). This is what the geometry tier
        (per-layer Jaccard / RSA / sparsity) consumes."""
        x = np.asarray(x, dtype=float).ravel()
        code = self._separate(x) if hasattr(self, "_separate") else x
        s_h, s_o = self._settle(code, target=None, beta=0.0)
        return {"input": code, "hidden": s_h, "output": s_o}

    def named_parameters(self) -> dict:
        """Live (uncopied) references to the trainable parameter groups. The
        weight tier snapshots/copies these itself when it needs to."""
        return {"W_ih": self.W_ih, "W_ho": self.W_ho, "b_h": self.b_h, "b_o": self.b_o}

    def transition_grads(self, x_t: np.ndarray, x_next: np.ndarray) -> dict:
        """The per-transition EP update (gradient estimate) per parameter group,
        at the current weights. Consumed by the weight tier for update-direction
        cosine interference and diagonal-Fisher / attribution. DG/XdG get the
        right (separated / gated) gradients for free via `_transition_deltas`."""
        d_ih, d_ho, d_bh, d_bo = self._transition_deltas(
            np.asarray(x_t, dtype=float).ravel(), np.asarray(x_next, dtype=float).ravel()
        )
        return {"W_ih": d_ih, "W_ho": d_ho, "b_h": d_bh, "b_o": d_bo}
