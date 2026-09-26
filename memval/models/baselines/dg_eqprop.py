import numpy as np
from typing import Optional

from ..base import HippocampalModel
from ..capabilities import RolloutMode
from .eq_prop import EqPropSequenceNetwork


class DentateGyrusKWTA:
    """
    Biological Dentate Gyrus expansion layer.
    Orthogonalizes dense sensory inputs + context into non-overlapping sparse codes
    using random projections and k-Winners-Take-All lateral inhibition.
    """
    def __init__(self, in_features: int, out_features: int = 1000, sparsity: float = 0.05, seed: Optional[int] = None):
        self.in_features = in_features
        self.out_features = out_features
        self.k = max(1, int(out_features * sparsity))

        rng = np.random.default_rng(seed)
        limit = np.sqrt(3.0 / in_features)
        # Fixed, non-plastic random expansion weights
        self.W = rng.uniform(-limit, limit, (out_features, in_features))

    def transform(self, x: np.ndarray) -> np.ndarray:
        raw_activations = self.W @ x
        threshold = np.partition(raw_activations, -self.k)[-self.k]
        sparse_out = np.zeros(self.out_features)
        sparse_out[raw_activations >= threshold] = 1.0
        return sparse_out


class DGEqPropSequenceNetwork(HippocampalModel):
    """
    Equilibrium Propagation model enhanced with a Dentate Gyrus (DG) module.

    The architecture uses:
      1. Dentate Gyrus: Expands the raw feature vector (optionally concatenated
         with a caller-supplied context array) into an orthogonal sparse code.
      2. EP CA3/CA1 layer: Learns the temporal transition
         ``sparse_code_t -> dense_state_{t+1}``.

    Context is purely external — the model does not auto-generate any positional
    encoding or temporal clock.  Callers that need such signals should compute them
    and pass them in via the ``context`` arguments.

    **Activation and readout strategy**:

    The EP settling dynamics use tanh throughout (hidden + output). Tanh is optimal
    for EP learning with sparse population-coded targets because it is zero-centered:
    inactive target neurons (the vast majority in sparse codes) are already at tanh's
    natural resting point (0), so the EP gradient focuses entirely on the few active
    neurons rather than wasting capacity.

    At readout, outputs are clipped to ``[0, ∞)`` to prevent negative activations from
    poisoning center-of-mass decoders used with population codes (e.g. place cells).
    Without this clip, tanh produces ~45% negative outputs that pull decoded positions
    in wrong directions, causing cascading errors during autoregressive recall.
    """

    #: `recall` feeds the raw prediction back as the next cue.
    rollout_mode = RolloutMode.OBSERVATION

    def __init__(
        self,
        n_features: int,
        n_context: int = 0,
        n_dg: int = 1000,
        sparsity: float = 0.05,
        noise_scale: float = 0.05,
        n_hidden: int = 128,
        learning_rate: float = 0.05,
        beta: float = 0.5,
        n_settle_steps: int = 50,
        n_epochs: int = 100,
        dt: float = 0.5,
        activation: str = "tanh",
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_features : int
            Dimensionality of each observation vector.
        n_context : int
            Dimensionality of the external context array that callers will supply
            at each timestep (e.g. a category tag, task ID, or positional encoding
            computed externally).  Defaults to 0 (no context).
        n_dg : int
            Number of DG units (expansion layer size).
        sparsity : float
            Fraction of DG units active after k-WTA inhibition.
        noise_scale : float
            Std-dev of Gaussian noise added to inputs during training (0 = off).
        n_hidden : int
            Number of EP hidden units.
        learning_rate : float
            Step size for EP weight updates.
        beta : float
            Nudging strength for the EP nudge phase.
        n_settle_steps : int
            Number of iterative steps to reach steady-state in each EP phase.
        n_epochs : int
            Number of full passes over the sequence(s) during training.
        dt : float
            Integration step size for the settling dynamics.
        activation : str
            Activation function for EP hidden & output units ('tanh' or 'sigmoid').
        seed : int, optional
            Random seed for reproducibility.
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_context = n_context
        self.n_dg = n_dg
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta = beta
        self.noise_scale = noise_scale

        self.dg = DentateGyrusKWTA(
            in_features=n_features + n_context,
            out_features=n_dg,
            sparsity=sparsity,
            seed=seed,
        )

        # Wrap the standard EP net but override the input projection to accept
        # the DG sparse codes rather than raw features.
        self.ep_net = EqPropSequenceNetwork(
            n_features=n_features,
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            beta=beta,
            n_settle_steps=n_settle_steps,
            n_epochs=n_epochs,
            dt=dt,
            activation=activation,
            seed=seed,
        )

        # Overwrite W_ih to connect n_dg sparse inputs to n_hidden nodes.
        scale_ih = np.sqrt(2.0 / (n_dg + n_hidden))
        self.ep_net.W_ih = self.ep_net.rng.normal(0, scale_ih, (n_hidden, n_dg))

    # ------------------------------------------------------------------
    # Readout helper
    # ------------------------------------------------------------------

    @staticmethod
    def _readout(s_o: np.ndarray) -> np.ndarray:
        """Convert EP output pre-activations to a non-negative population code.

        Applies tanh (matching the settling activation) then clips to ``[0, ∞)``.
        The clip removes negative activations that would otherwise poison
        center-of-mass decoders used with population codes like place cells.

        Without this clip, ~45% of tanh outputs are negative, pulling the
        decoded position in arbitrary wrong directions and causing cascading
        autoregressive errors during recall.
        """
        return np.clip(np.tanh(s_o), 0.0, None)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _make_dg_input(self, x: np.ndarray, context: Optional[np.ndarray]) -> np.ndarray:
        """Concatenate feature vector and optional context for DG expansion."""
        if context is not None and self.n_context > 0:
            return np.concatenate([x.flatten(), context.flatten()])
        return x.flatten()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_sequence(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, **kwargs):
        """Train the model on a single sequence using Equilibrium Propagation.

        Each consecutive pair ``(x_t, x_{t+1})`` is passed through the DG and
        contributes to a gradient accumulation buffer.  Weights are updated once
        per epoch after all transitions have been processed.

        Args:
            sequence_data: Shape ``(T, n_features)``. The encoded trajectory.
            context_data: Optional shape ``(T, n_context)``. External context
                for each timestep. If ``None`` and ``n_context == 0``, the DG
                receives only the feature vector.
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
            dW_ih = np.zeros_like(self.ep_net.W_ih)
            dW_ho = np.zeros_like(self.ep_net.W_ho)
            db_h = np.zeros_like(self.ep_net.b_h)
            db_o = np.zeros_like(self.ep_net.b_o)

            for t in range(n_transitions):
                x_t = sequence_data[t]
                x_next = sequence_data[t + 1]
                ctx_t = context_data[t] if context_data is not None else None

                if self.noise_scale > 0.0:
                    x_t_noisy = x_t + self.ep_net.rng.normal(
                        0, self.noise_scale, size=x_t.shape
                    )
                else:
                    x_t_noisy = x_t

                dg_in = self._make_dg_input(x_t_noisy, ctx_t)
                dg_code = self.dg.transform(dg_in)

                s_h_free, s_o_free = self.ep_net._settle(
                    x_input=dg_code, target=None, beta=0.0
                )
                s_h_nudge, s_o_nudge = self.ep_net._settle(
                    x_input=dg_code, target=x_next, beta=self.beta
                )

                rho_h_free  = self.ep_net._activate(s_h_free)
                rho_o_free  = self.ep_net._activate(s_o_free)
                rho_h_nudge = self.ep_net._activate(s_h_nudge)
                rho_o_nudge = self.ep_net._activate(s_o_nudge)

                inv_beta = 1.0 / self.beta
                dW_ho += inv_beta * (
                    np.outer(rho_o_nudge, rho_h_nudge)
                    - np.outer(rho_o_free, rho_h_free)
                )
                dW_ih += inv_beta * (
                    np.outer(rho_h_nudge, dg_code)
                    - np.outer(rho_h_free, dg_code)
                )
                db_h += inv_beta * (rho_h_nudge - rho_h_free)
                db_o += inv_beta * (rho_o_nudge - rho_o_free)

            lr = self.learning_rate / n_transitions
            self.ep_net.W_ho += lr * dW_ho
            self.ep_net.W_ih += lr * dW_ih
            self.ep_net.b_h  += lr * db_h
            self.ep_net.b_o  += lr * db_o

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_next(self, current_event: np.ndarray, current_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Predict the next event from a single timestep using the DG + EP pipeline.

        The current event is (optionally) concatenated with an external context
        vector, expanded through the Dentate Gyrus, and the EP network settles
        to a free-phase equilibrium.  The output is read out with non-negative
        clipping.

        Args:
            current_event: Shape ``(n_features,)``. Current encoded observation.
            current_context: Shape ``(n_context,)``. Optional external context.
                Required when ``n_context > 0``; ignored otherwise.

        Returns:
            Predicted next event, shape ``(n_features,)``, non-negative.
        """
        dg_in = self._make_dg_input(current_event, current_context)
        dg_code = self.dg.transform(dg_in)

        _s_h, s_o = self.ep_net._settle(x_input=dg_code, target=None, beta=0.0)

        if np.any(np.isnan(s_o)):
            print("Warning: EP network diverged. Resetting output.")
            return np.zeros(self.n_features)

        self.ep_net.current_state = self._readout(s_o)
        return self.ep_net.current_state.copy()

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Autoregressively recall a sequence from a prompt.

        Seeds with the last vector of the prompt, then iteratively calls
        ``predict_next`` for ``length`` steps.

        Args:
            prompt_event: Shape ``(T_prompt, n_features)`` or ``(n_features,)``.
                The cue to seed recall from.
            length: Number of steps to recall forward.
            prompt_context: Optional shape ``(length, n_context)``. External
                context for each recall step.  Required when ``n_context > 0``.

        Returns:
            Recalled sequence, shape ``(length, n_features)``, non-negative.
        """
        if len(prompt_event.shape) > 1:
            current = prompt_event[-1].copy()
        else:
            current = prompt_event.copy()

        recalled = np.zeros((length, self.n_features))

        for t in range(length):
            ctx = prompt_context[t] if (prompt_context is not None and t < len(prompt_context)) else None
            next_val = self.predict_next(current, current_context=ctx)
            recalled[t] = next_val
            current = next_val

        return recalled

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_latent_state(self) -> dict:
        state = self.ep_net.get_latent_state()
        state['W_dg'] = self.dg.W.copy()
        return state

    def reset_context(self):
        """Reset transient EP state without touching learned weights."""
        self.ep_net.reset_context()
