import numpy as np
from typing import Optional, Dict
from ..base import HippocampalModel
from .dg_eqprop import DGEqPropSequenceNetwork


class EWCDGEqPropSequenceNetwork(DGEqPropSequenceNetwork):
    """
    Equilibrium Propagation sequence network enhanced with a Dentate Gyrus (DG) 
    expansion layer and Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.

    Supports three methods of weight importance estimation:
      1. 'canonical': Standard EWC (nudge target = x_next)
      2. 'lr': Logits Reversal/Target Reversal (nudge target = -x_next)
      3. 'mas': Memory Aware Synapses equivalent (nudge target = 0)
    """

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
        ewc_lambda: float = 100.0,
        ewc_method: str = "lr",
        **kwargs,
    ):
        super().__init__(
            n_features=n_features,
            n_context=n_context,
            n_dg=n_dg,
            sparsity=sparsity,
            noise_scale=noise_scale,
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            beta=beta,
            n_settle_steps=n_settle_steps,
            n_epochs=n_epochs,
            dt=dt,
            activation=activation,
            seed=seed,
            **kwargs,
        )
        self.ewc_lambda = ewc_lambda
        self.ewc_method = ewc_method.lower()
        if self.ewc_method not in ["canonical", "lr", "mas"]:
            raise ValueError(
                f"Invalid ewc_method: {self.ewc_method}. Must be 'canonical', 'lr', or 'mas'."
            )

        # consolidated parameters
        self.fisher: Optional[Dict[str, np.ndarray]] = None
        self.anchor_params: Optional[Dict[str, np.ndarray]] = None

    def fit_sequence(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, **kwargs):
        """Train the model on a single sequence using Equilibrium Propagation and EWC penalty.

        At each epoch, weights are updated with the sum of the standard EP gradients
        and the EWC weight consolidation quadratic penalty.
        """
        ewc_lambda = kwargs.get("ewc_lambda", self.ewc_lambda)

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

            # Apply EWC updates if fisher matrices and anchors exist
            if self.fisher is not None and self.anchor_params is not None and ewc_lambda > 0.0:
                ewc_penalty_ho = self.learning_rate * ewc_lambda * self.fisher["W_ho"] * (self.ep_net.W_ho - self.anchor_params["W_ho"])
                ewc_penalty_ih = self.learning_rate * ewc_lambda * self.fisher["W_ih"] * (self.ep_net.W_ih - self.anchor_params["W_ih"])
                ewc_penalty_bh = self.learning_rate * ewc_lambda * self.fisher["b_h"] * (self.ep_net.b_h - self.anchor_params["b_h"])
                ewc_penalty_bo = self.learning_rate * ewc_lambda * self.fisher["b_o"] * (self.ep_net.b_o - self.anchor_params["b_o"])
            else:
                ewc_penalty_ho = 0.0
                ewc_penalty_ih = 0.0
                ewc_penalty_bh = 0.0
                ewc_penalty_bo = 0.0

            self.ep_net.W_ho += lr * dW_ho - ewc_penalty_ho
            self.ep_net.W_ih += lr * dW_ih - ewc_penalty_ih
            self.ep_net.b_h  += lr * db_h  - ewc_penalty_bh
            self.ep_net.b_o  += lr * db_o  - ewc_penalty_bo

    def estimate_fisher(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Estimate the diagonal of the Fisher Information Matrix (FIM) for the given sequence.
        
        Runs a single forward pass over the sequence, computes the EP gradients (i.e. derivative
        of energy nudge vs free) for each transition using the selected `ewc_method`, squares
        them, and averages over transitions.
        """
        seq_len = sequence_data.shape[0]
        if seq_len < 2:
            return {}
        n_transitions = seq_len - 1

        fisher = {
            "W_ih": np.zeros_like(self.ep_net.W_ih),
            "W_ho": np.zeros_like(self.ep_net.W_ho),
            "b_h": np.zeros_like(self.ep_net.b_h),
            "b_o": np.zeros_like(self.ep_net.b_o),
        }

        for t in range(n_transitions):
            x_t = sequence_data[t]
            x_next = sequence_data[t + 1]
            ctx_t = context_data[t] if context_data is not None else None

            # Calculate DG code (no noise when estimating Fisher)
            dg_in = self._make_dg_input(x_t, ctx_t)
            dg_code = self.dg.transform(dg_in)

            # Free phase
            s_h_free, s_o_free = self.ep_net._settle(
                x_input=dg_code, target=None, beta=0.0
            )

            # Define nudge target based on the EWC method
            if self.ewc_method == "canonical":
                target = x_next
            elif self.ewc_method == "lr":
                target = -x_next
            elif self.ewc_method == "mas":
                target = np.zeros_like(x_next)
            else:
                target = x_next

            # Nudge phase
            s_h_nudge, s_o_nudge = self.ep_net._settle(
                x_input=dg_code, target=target, beta=self.beta
            )

            rho_h_free  = self.ep_net._activate(s_h_free)
            rho_o_free  = self.ep_net._activate(s_o_free)
            rho_h_nudge = self.ep_net._activate(s_h_nudge)
            rho_o_nudge = self.ep_net._activate(s_o_nudge)

            inv_beta = 1.0 / self.beta
            
            # Compute gradient for this transition (negative of EP update direction)
            g_ho = inv_beta * (np.outer(rho_o_nudge, rho_h_nudge) - np.outer(rho_o_free, rho_h_free))
            g_ih = inv_beta * (np.outer(rho_h_nudge, dg_code) - np.outer(rho_h_free, dg_code))
            g_bh = inv_beta * (rho_h_nudge - rho_h_free)
            g_bo = inv_beta * (rho_o_nudge - rho_o_free)

            fisher["W_ho"] += g_ho ** 2
            fisher["W_ih"] += g_ih ** 2
            fisher["b_h"] += g_bh ** 2
            fisher["b_o"] += g_bo ** 2

        # Average over transitions
        for key in fisher:
            fisher[key] /= n_transitions

        return fisher

    def consolidate(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, decay_rate: float = 0.0):
        """
        Consolidate the current model weights by:
        1. Estimating the Fisher Information Matrix (FIM) diagonal for the given sequence.
        2. Updating the cumulative Fisher matrix: F_new = (1 - decay_rate) * F_old + F_seq.
        3. Saving the current weights as anchor weights for future EWC training.
        """
        seq_fisher = self.estimate_fisher(sequence_data, context_data)
        if not seq_fisher:
            return

        if self.fisher is None:
            self.fisher = {key: val.copy() for key, val in seq_fisher.items()}
        else:
            for key in self.fisher:
                self.fisher[key] = (1.0 - decay_rate) * self.fisher[key] + seq_fisher[key]

        self.anchor_params = {
            "W_ih": self.ep_net.W_ih.copy(),
            "W_ho": self.ep_net.W_ho.copy(),
            "b_h": self.ep_net.b_h.copy(),
            "b_o": self.ep_net.b_o.copy(),
        }

    def reset_ewc(self):
        """Reset the running Fisher information and anchor parameters."""
        self.fisher = None
        self.anchor_params = None
