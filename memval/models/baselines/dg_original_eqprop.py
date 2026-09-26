"""
Dentate-Gyrus Equilibrium Propagation Sequence Network.

Extends the plain Equilibrium Propagation net (`OriginalEqPropSequenceNetwork`)
with a biologically-motivated **dentate gyrus (DG) pattern separator** in front of
the trainable EP core.

Motivation
----------
The plain EP net suffers catastrophic forgetting: training on sequence B
overwrites the dense weights that encoded sequence A. Forgetting is governed by
*weight-update overlap* — a weight only moves when both of its endpoints are
active. If two inputs map to sparse, near-disjoint codes, then their input-side
weight updates land in disjoint columns and interfere far less.

The DG achieves this via **pattern separation**: a large expansion of the input
into a sparse, decorrelated code. Here the DG is a *frozen random expansion*
(perforant-path-like, not trained → stable, non-drifting separation) followed by
**feedback (lateral) inhibition** that drives the code sparse.

Biological route to sparsity
-----------------------------
Sparsity in the real DG is the equilibrium of a competitive inhibitory circuit —
feedforward inhibition (modelled here as a threshold) and feedback inhibition
from granule cells (modelled as subtractive global inhibition) — not a top-k
sort. We use that competitive-settling route by default because:
  * it is the biological mechanism (GABAergic interneurons), and
  * the inhibition term is the gradient of a real energy term (½·γ·(Σs)²), so the
    settling still descends a well-defined energy and EP stays valid.
An optional hard `dg_k` cap (top-k) is available as the γ→∞ limiting case, kept
off by default for use as an ablation / control knob.

Scope
-----
The DG only separates the *input* side. The downstream hidden→output readout is
still dense and shared; protecting that matrix (maintained hidden sparsity /
DG-derived gating à la XdG, slow readout) is a deliberate next step, not covered
here.
"""

from typing import Optional

import numpy as np

from .original_eqprop import OriginalEqPropSequenceNetwork


class DGOriginalEqPropSequenceNetwork(OriginalEqPropSequenceNetwork):
    """
    `OriginalEqPropSequenceNetwork` with a frozen, competitively-inhibited DG
    pattern separator on the input.

    Architecture
    ------------
    input (n_features)
        → DG separator (n_dg, frozen random expansion + feedback inhibition, sparse)
        → hidden (n_hidden, trained EP)
        → output (n_features, trained EP; predicts x_{t+1})

    Only the DG→hidden and hidden→output weights are trained (standard two-phase
    EP). The input→DG projection is fixed.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 64,
        n_dg: int = 256,
        dg_target_sparsity: Optional[float] = 0.05,
        dg_inhibition: float = 1.0,
        dg_threshold: float = 0.0,
        dg_settle_steps: int = 15,
        dg_k: Optional[int] = None,
        dg_seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_dg : int
            Number of DG units. Should be ≫ n_features (expansion is the DG's
            defining feature).
        dg_target_sparsity : float, optional
            Target fraction of active DG units. Implements *feedforward*
            inhibition as a gain-controlled adaptive threshold: the threshold is
            set from the input-drive distribution so that ~this fraction of units
            start above threshold, independent of input scale. This is what keeps
            sparsity stable (a fixed `dg_threshold` is fragile w.r.t. input
            magnitude). Set to None to disable and use a fixed `dg_threshold`.
        dg_inhibition : float
            Feedback (lateral) inhibition strength γ. Population-normalised
            (γ·(Σr − r)/n_dg), so it does not blow up with n_dg. Sharpens the
            code among the above-threshold units. Higher → sparser / more graded.
        dg_threshold : float
            Fixed feedforward inhibition (constant subtracted from the drive).
            Only used when `dg_target_sparsity` is None.
        dg_settle_steps : int
            Iterations of the DG competitive settling dynamics.
        dg_k : int, optional
            Optional hard cap on the number of active DG units (top-k / k-WTA
            limit). Off by default; a control / ablation knob.
        dg_seed : int, optional
            Seed for the frozen random DG projection (independent of the EP
            weight-init seed so the separation basis is reproducible on its own).
        **kwargs
            Forwarded to `OriginalEqPropSequenceNetwork` (learning_rate, beta,
            n_settle_steps, n_epochs, dt, activation, seed, ...).
        """
        super().__init__(n_features=n_features, n_hidden=n_hidden, **kwargs)

        self.n_dg = n_dg
        self.dg_target_sparsity = dg_target_sparsity
        self.dg_inhibition = dg_inhibition
        self.dg_threshold = dg_threshold
        self.dg_settle_steps = dg_settle_steps
        self.dg_k = dg_k

        # ----- frozen random DG expansion (perforant-path-like) -----
        # Own RNG so the separation basis is reproducible independently of the
        # EP weight init.
        dg_rng = np.random.default_rng(dg_seed)
        W_dg = dg_rng.normal(0.0, 1.0, (n_dg, n_features))
        # Unit-norm rows keep the DG drive scale ~O(1) so that the threshold /
        # inhibition defaults are meaningful regardless of n_features.
        norms = np.linalg.norm(W_dg, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.W_dg = W_dg / norms  # (n_dg, n_features), frozen — never updated

        # ----- rebuild input→hidden weights to accept the DG code -----
        # The parent initialised W_ih as (n_hidden, n_features); the EP core now
        # reads the DG code (n_dg) instead of the raw input.
        limit_ih = np.sqrt(6.0 / (n_dg + n_hidden))
        self.W_ih = self.rng.uniform(-limit_ih, limit_ih, (n_hidden, n_dg))

    # ------------------------------------------------------------------
    # Dentate-gyrus pattern separation
    # ------------------------------------------------------------------
    def _separate(self, x: np.ndarray) -> np.ndarray:
        """
        Map a raw input (n_features,) to a sparse DG code (n_dg,).

        Feedforward inhibition sets an adaptive threshold (gain control) targeting
        a stable population activity; feedback inhibition then competitively
        sharpens the code. Clipped to [0, 1] to match the EP state convention.
        """
        x = np.asarray(x, dtype=float).flatten()
        drive = self.W_dg @ x

        # --- feedforward inhibition: adaptive, scale-invariant threshold ---
        if self.dg_target_sparsity is not None:
            k = int(round(self.dg_target_sparsity * self.n_dg))
            k = min(max(k, 1), self.n_dg)
            theta = np.partition(drive, -k)[-k]  # drive of the k-th strongest unit
        else:
            theta = self.dg_threshold
        drive = drive - theta

        r = np.zeros(self.n_dg)
        for _ in range(self.dg_settle_steps):
            # feedback (lateral) inhibition, population-normalised so it scales
            # with the *fraction* active rather than the raw count — gradient of
            # ½·(γ/n_dg)·(Σr)². Avoids runaway suppression as n_dg grows.
            inhib = self.dg_inhibition * (r.sum() - r) / self.n_dg
            r = np.clip(r + self.dt * (-r + drive - inhib), 0.0, 1.0)

        # optional hard k-WTA cap (γ→∞ limit) — off by default
        if self.dg_k is not None and self.dg_k < self.n_dg:
            if np.count_nonzero(r) > self.dg_k:
                thresh = np.partition(r, -self.dg_k)[-self.dg_k]
                r = np.where(r >= thresh, r, 0.0)

        return r

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _transition_deltas(self, x_t: np.ndarray, x_next: np.ndarray):
        """DG-aware override of the parent's per-transition rule: pattern-separate
        the presynaptic input to its sparse DG code before forming the EP deltas,
        so the inherited online trainer (``fit_event``) drives the EP core with DG
        codes exactly as ``fit_sequence`` does here. The nudge target stays in
        feature space (``x_next``). Returns ``(dW_ih, dW_ho, db_h, db_o)``."""
        dg_t = self._separate(x_t)
        s_h_free, s_o_free = self._settle(dg_t, target=None, beta=0.0)
        s_h_nudge, s_o_nudge = self._settle(dg_t, target=x_next, beta=self.beta)
        inv_beta = 1.0 / self.beta
        dW_ho = inv_beta * (np.outer(s_o_nudge, s_h_nudge) - np.outer(s_o_free, s_h_free))
        dW_ih = inv_beta * (np.outer(s_h_nudge, dg_t) - np.outer(s_h_free, dg_t))
        db_h = inv_beta * (s_h_nudge - s_h_free)
        db_o = inv_beta * (s_o_nudge - s_o_free)
        return dW_ih, dW_ho, db_h, db_o

    def fit_sequence(self, sequence_data: np.ndarray, context_data=None, **kwargs):
        """
        Train the EP core on a sequence, clamping DG codes (not raw inputs) to
        the visible layer.

        Mirrors the parent's two-phase EP update, with two differences:
          * the clamped input and the pre-synaptic factor of ΔW_ih are the sparse
            DG code (so input-side updates are gated by DG sparsity), and
          * `epochs` passed by the caller is honoured (the benchmark passes it).
        The nudge target stays in feature space (x_{t+1}).
        """
        seq_len = sequence_data.shape[0]
        if seq_len < 2:
            return

        n_transitions = seq_len - 1
        n_epochs = int(kwargs.get("epochs", self.n_epochs))

        # DG projection is frozen → codes are constant across epochs; precompute.
        dg_codes = [self._separate(sequence_data[t]) for t in range(seq_len)]

        for _epoch in range(n_epochs):
            dW_ih = np.zeros_like(self.W_ih)
            dW_ho = np.zeros_like(self.W_ho)
            db_h = np.zeros_like(self.b_h)
            db_o = np.zeros_like(self.b_o)

            for t in range(n_transitions):
                dg_t = dg_codes[t]
                x_next = sequence_data[t + 1]

                # --- free phase ---
                s_h_free, s_o_free = self._settle(dg_t, target=None, beta=0.0)
                # --- nudge phase ---
                s_h_nudge, s_o_nudge = self._settle(dg_t, target=x_next, beta=self.beta)

                inv_beta = 1.0 / self.beta

                dW_ho += inv_beta * (
                    np.outer(s_o_nudge, s_h_nudge) - np.outer(s_o_free, s_h_free)
                )
                # pre-synaptic factor is the sparse DG code → sparse, gated update
                dW_ih += inv_beta * (
                    np.outer(s_h_nudge, dg_t) - np.outer(s_h_free, dg_t)
                )
                db_h += inv_beta * (s_h_nudge - s_h_free)
                db_o += inv_beta * (s_o_nudge - s_o_free)

            lr = self.learning_rate / n_transitions
            self.W_ho += lr * dW_ho
            self.W_ih += lr * dW_ih
            self.b_h += lr * db_h
            if self.use_output_bias:
                self.b_o += lr * db_o

            # extension hook (inherited no-op; EWC overrides it to pull weights
            # toward a consolidated anchor after each epoch's data update)
            self._on_epoch_end()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(self, current_event: np.ndarray, current_context=None, **kwargs) -> np.ndarray:
        """Single-step prediction: separate → free-phase settle → output state."""
        x = np.asarray(current_event, dtype=float).flatten()
        dg = self._separate(x)
        _s_h, s_o = self._settle(dg, target=None, beta=0.0)
        self.current_state = s_o.copy()
        return self.current_state.copy()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_latent_state(self) -> dict:
        """Weight matrices incl. the frozen DG projection, for analysis."""
        state = super().get_latent_state()
        state["W_dg"] = self.W_dg.copy()
        return state
