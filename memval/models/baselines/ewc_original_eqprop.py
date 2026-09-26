"""
EWC + OriginalEqProp Sequence Network.

Online Elastic Weight Consolidation directly on the plain
`OriginalEqPropSequenceNetwork` — no DG pattern separation, no XdG gating. Same
`EWCMixin` as the DG variant; the only difference is that the EP gradient is
formed on the raw input (there is no `_separate`). Exists as the controlled
comparison point for "does the DG front-end add anything over plain EWC?".
"""

import numpy as np

from .original_eqprop import OriginalEqPropSequenceNetwork
from .ewc_mixin import EWCMixin


class EWCOriginalEqPropSequenceNetwork(EWCMixin, OriginalEqPropSequenceNetwork):
    """`OriginalEqPropSequenceNetwork` + online EWC (via `EWCMixin`)."""

    def __init__(
        self,
        n_features: int,
        ewc_lambda: float = 30000.0,
        ewc_decay: float = 0.0,
        ewc_per_task: bool = False,
        ewc_normalize: bool = False,
        **kwargs,
    ):
        super().__init__(n_features=n_features, **kwargs)
        self._init_ewc(ewc_lambda, ewc_decay, per_task=ewc_per_task, normalize=ewc_normalize)

    def _fisher_transition_grads(self, x_t: np.ndarray, x_next: np.ndarray) -> dict:
        """EP gradient of one transition, on the raw (un-separated) input."""
        s_h_free, s_o_free = self._settle(x_t, target=None, beta=0.0)
        s_h_nudge, s_o_nudge = self._settle(x_t, target=x_next, beta=self.beta)
        inv_beta = 1.0 / self.beta
        return {
            "W_ho": inv_beta * (np.outer(s_o_nudge, s_h_nudge) - np.outer(s_o_free, s_h_free)),
            "W_ih": inv_beta * (np.outer(s_h_nudge, x_t) - np.outer(s_h_free, x_t)),
            "b_h": inv_beta * (s_h_nudge - s_h_free),
            "b_o": inv_beta * (s_o_nudge - s_o_free),
        }
