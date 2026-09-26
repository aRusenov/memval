"""
EWC + DG + XdG Equilibrium Propagation Sequence Network.

Online Elastic Weight Consolidation (Kirkpatrick et al., 2017) on top of the
DG(+XdG) EP stack. The EWC machinery lives in `EWCMixin`; this class only
supplies the base-specific gradient (the input is pattern-separated by the DG
before the EP two-phase gradient is formed) and wires XdG gating through by
inheriting `DGXdGEqPropSequenceNetwork`.

See `EWCMixin` for the consolidation math and `EWCOriginalEqPropSequenceNetwork`
for the plain-net counterpart (same mixin, raw input, for A/B comparison).

Why EWC beats widening the hidden layer: it protects the weights important for an
old sequence directly (regardless of which hidden units are shared), so retention
does not require a wide/sparse hidden layer — only a diagonal Fisher and one extra
pass per sequence. Set `gate_sparsity=1.0` to disable XdG and study EWC on the
plain DG separator (the best-performing config so far).
"""

import numpy as np

from .dg_xdg_eqprop import DGXdGEqPropSequenceNetwork
from .ewc_mixin import EWCMixin


class EWCDGXdGEqPropSequenceNetwork(EWCMixin, DGXdGEqPropSequenceNetwork):
    """`DGXdGEqPropSequenceNetwork` + online EWC (via `EWCMixin`)."""

    def __init__(
        self,
        n_features: int,
        ewc_lambda: float = 30000.0,
        ewc_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(n_features=n_features, **kwargs)
        self._init_ewc(ewc_lambda, ewc_decay)

    def _fisher_transition_grads(self, x_t: np.ndarray, x_next: np.ndarray) -> dict:
        """EP gradient of one transition, on the DG-separated input."""
        dg = self._separate(x_t)
        s_h_free, s_o_free = self._settle(dg, target=None, beta=0.0)
        s_h_nudge, s_o_nudge = self._settle(dg, target=x_next, beta=self.beta)
        inv_beta = 1.0 / self.beta
        return {
            "W_ho": inv_beta * (np.outer(s_o_nudge, s_h_nudge) - np.outer(s_o_free, s_h_free)),
            "W_ih": inv_beta * (np.outer(s_h_nudge, dg) - np.outer(s_h_free, dg)),
            "b_h": inv_beta * (s_h_nudge - s_h_free),
            "b_o": inv_beta * (s_o_nudge - s_o_free),
        }
