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
from ..capabilities import RolloutMode


class EqPropSequenceNetwork(HippocampalModel):
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
            # Evaluate activations
            rho_h = self._activate(s_h)
            rho_o = self._activate_output(s_o)
            
            # ----- hidden unit dynamics -----
            current_h = self.W_ih @ x_input + self.W_ho.T @ rho_o + self.b_h
            s_h = s_h + self.dt * (-s_h + current_h)

            # ----- output unit dynamics -----
            current_o = self.W_ho @ rho_h + self.b_o
            
            nudge_term = np.zeros_like(s_o)
            if target is not None and beta > 0:
                nudge_term = beta * (target - rho_o)
                
            s_o = s_o + self.dt * (-s_o + current_o + nudge_term)

        return s_h, s_o

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
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
        # Matches DGOriginalEqPropSequenceNetwork, which already did this.
        for _epoch in range(int(kwargs.get('epochs', self.n_epochs))):
            # accumulators
            dW_ih = np.zeros_like(self.W_ih)
            dW_ho = np.zeros_like(self.W_ho)
            db_h = np.zeros_like(self.b_h)
            db_o = np.zeros_like(self.b_o)

            for t in range(n_transitions):
                x_t = sequence_data[t]
                x_next = sequence_data[t + 1]

                # --- free phase ---
                s_h_free, s_o_free = self._settle(x_t, target=None, beta=0.0)

                # --- nudge phase ---
                s_h_nudge, s_o_nudge = self._settle(x_t, target=x_next, beta=self.beta)

                # --- get activations ---
                rho_h_free = self._activate(s_h_free)
                rho_o_free = self._activate_output(s_o_free)
                rho_h_nudge = self._activate(s_h_nudge)
                rho_o_nudge = self._activate_output(s_o_nudge)

                # --- EP weight updates ---
                inv_beta = 1.0 / self.beta

                dW_ho += inv_beta * (
                    np.outer(rho_o_nudge, rho_h_nudge) - np.outer(rho_o_free, rho_h_free)
                )
                dW_ih += inv_beta * (
                    np.outer(rho_h_nudge, x_t) - np.outer(rho_h_free, x_t)
                )
                db_h += inv_beta * (rho_h_nudge - rho_h_free)
                db_o += inv_beta * (rho_o_nudge - rho_o_free)

            # average over transitions and apply
            lr = self.learning_rate / n_transitions
            self.W_ho += lr * dW_ho
            self.W_ih += lr * dW_ih
            self.b_h += lr * db_h
            self.b_o += lr * db_o

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
        self.current_state = self._activate_output(s_o)
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
        """Reset transient state without touching learned weights."""
        self.current_state = np.zeros(self.n_features)
