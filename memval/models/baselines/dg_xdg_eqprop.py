"""
DG + XdG Equilibrium Propagation Sequence Network.

Extends `DGOriginalEqPropSequenceNetwork` with **context-dependent gating (XdG)**
of the hidden layer, where the gating context is *derived from the DG code* rather
than supplied as an external label.

Why
---
The DG separator alone only gates the input-side matrix (DG→hidden). The dense
hidden→output readout stays shared, so training on B still overwrites A there —
empirically the DG-only model forgot *more*, not less. XdG fixes the bottleneck:
a binary mask restricts each item to a sparse subset of hidden units, so different
items update near-disjoint rows/columns of **both** trained matrices — including
the readout.

DG-derived gate (no external context needed)
--------------------------------------------
Vanilla XdG looks up a fixed random mask per discrete task label. This benchmark
provides no distinguishing context signal at recall time, so instead the mask is
a deterministic function of the DG code:

    gate_drive = M · dg_code           # M: frozen random (n_hidden, n_dg)
    mask       = top-`gate_k` units of gate_drive  (binary)

Because dg_code is a frozen function of the input, the *same item reproduces the
same mask* during the free phase, the nudge phase, and at recall — so the gated
subnetwork is self-selecting from the cue. Different items → different DG codes →
near-disjoint masks → protected associations.

How the gate protects the readout
---------------------------------
The EP updates are outer products with the hidden activity `h`:
    ΔW_ih ∝ outer(h, dg_code)          → masked rows of W_ih don't update
    ΔW_ho ∝ outer(o, h)                → masked columns of W_ho don't update
Since `h` is zero outside the mask, only the item's subnetwork rows/columns move,
in *both* matrices. That is the piece the DG-only model was missing.
"""

from typing import Optional

import numpy as np

from .dg_original_eqprop import DGOriginalEqPropSequenceNetwork


class DGXdGEqPropSequenceNetwork(DGOriginalEqPropSequenceNetwork):
    """
    `DGOriginalEqPropSequenceNetwork` plus a DG-derived context-dependent gate
    (XdG) on the hidden layer.

    Architecture
    ------------
    input (n_features)
        → DG separator (n_dg, frozen, sparse)         [inherited]
        → hidden (n_hidden, trained EP) gated by a DG-derived binary mask
        → output (n_features, trained EP; predicts x_{t+1})
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 128,
        n_dg: int = 1000,
        gate_sparsity: float = 0.2,
        gate_seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        gate_sparsity : float
            Fraction of hidden units left active per item (the size of each
            item's subnetwork). Lower → more disjoint subnetworks / less
            interference, but less representational capacity per item. 0.2 is the
            XdG-typical value.
        gate_seed : int, optional
            Seed for the frozen DG→hidden gate projection `M` (independent of the
            EP weight-init and DG-projection seeds).
        **kwargs
            Forwarded to `DGOriginalEqPropSequenceNetwork` (n_dg via `n_dg`,
            dg_target_sparsity, dg_inhibition, learning_rate, beta, seed, ...).
        """
        super().__init__(n_features=n_features, n_hidden=n_hidden, n_dg=n_dg, **kwargs)

        self.gate_sparsity = gate_sparsity
        self.gate_k = int(round(gate_sparsity * n_hidden))
        self.gate_k = min(max(self.gate_k, 1), n_hidden)

        # frozen random DG→hidden gate projection (own RNG for reproducibility)
        gate_rng = np.random.default_rng(gate_seed)
        self.M = gate_rng.normal(0.0, 1.0, (n_hidden, n_dg))  # (n_hidden, n_dg), frozen

    # ------------------------------------------------------------------
    # DG-derived context gate
    # ------------------------------------------------------------------
    def _hidden_gate(self, dg_code: np.ndarray) -> np.ndarray:
        """
        Binary hidden mask (~gate_k active) selected by the DG code. Deterministic
        in the input, so identical across free/nudge phases and at recall.
        """
        if self.gate_k >= self.n_hidden:
            return np.ones(self.n_hidden)
        gate_drive = self.M @ dg_code
        thresh = np.partition(gate_drive, -self.gate_k)[-self.gate_k]
        return (gate_drive >= thresh).astype(float)

    # ------------------------------------------------------------------
    # Gated settling dynamics
    # ------------------------------------------------------------------
    def _settle(self, x_input, target=None, beta=0.0):
        """
        Same dynamics as the parent, but the hidden state is confined to the
        DG-derived subnetwork (masked every step). `x_input` is the DG code
        (the DG model clamps DG codes as the visible layer), so the mask is
        derived directly from it.
        """
        mask = self._hidden_gate(x_input)

        s_h = np.zeros(self.n_hidden)
        s_o = np.zeros(self.n_features)

        for _ in range(self.n_settle_steps):
            # ----- hidden unit dynamics (gated) -----
            current_h = self.W_ih @ x_input + self.W_ho.T @ s_o + self.b_h
            s_h = mask * np.clip(s_h + self.dt * (-s_h + current_h), 0.0, 1.0)

            # ----- output unit dynamics -----
            current_o = self.W_ho @ s_h + self.b_o

            nudge_term = np.zeros_like(s_o)
            if target is not None and beta > 0:
                nudge_term = 2.0 * beta * (target - s_o)

            s_o = np.clip(s_o + self.dt * (-s_o + current_o + nudge_term), 0.0, 1.0)

        return s_h, s_o

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_latent_state(self) -> dict:
        state = super().get_latent_state()
        state["M"] = self.M.copy()
        return state
