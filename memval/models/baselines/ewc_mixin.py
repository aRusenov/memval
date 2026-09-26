"""
Base-agnostic online Elastic Weight Consolidation (EWC) for EP networks.

`EWCMixin` provides the whole EWC machinery — running diagonal Fisher, a
consolidated anchor, the per-epoch penalty, and `consolidate()` — independent of
which EP base it is mixed into. The *only* base-specific piece is
`_fisher_transition_grads`, which computes the EP two-phase gradient for a single
(x_t → x_{t+1}) transition. That is exactly where the difference between bases
lives: a DG-based net pattern-separates the input first; a plain net clamps the
raw input. Subclasses implement that one method.

Usage:
    class EWCFoo(EWCMixin, FooEqPropNetwork):
        def __init__(self, *a, ewc_lambda=..., ewc_decay=..., **kw):
            super().__init__(*a, **kw)          # builds the EP base
            self._init_ewc(ewc_lambda, ewc_decay)
        def _fisher_transition_grads(self, x_t, x_next):
            ...  # return dict of per-weight EP gradients

The penalty is delivered through the base class's `_on_epoch_end()` hook, so no
training-loop code is duplicated.

Fisher under Equilibrium Propagation
------------------------------------
EP has no back-propagated gradient, but its two-phase update
    g(theta) = (1/beta)(rho_post_nudge (x) rho_pre_nudge - rho_post_free (x) rho_pre_free)
IS the per-sample gradient estimate of the loss. Fisher diagonal ~ mean of
g(theta)^2 over the sequence's transitions.
"""

import numpy as np


class EWCMixin:
    """Online-EWC machinery, agnostic to the EP base it is mixed into."""

    _WEIGHTS = ("W_ih", "W_ho", "b_h", "b_o")

    # ------------------------------------------------------------------
    def _init_ewc(self, ewc_lambda: float = 30000.0, ewc_decay: float = 0.0,
                  per_task: bool = False, normalize: bool = False) -> None:
        """Initialise EWC state. Call from the subclass __init__ after super().__init__.

        ewc_lambda : consolidation strength (stability-plasticity knob). Fisher is
            tiny for EP, so this is large (~1e4-1e5).
        ewc_decay  : decay of accumulated Fisher on each consolidate (single-anchor
            mode only): F <- (1 - ewc_decay)*F + F_new. 0.0 keeps all past importance.
        per_task   : anchor strategy.
            False (default) — SINGLE running anchor: one Fisher (decayed-accumulated)
                and one anchor that is overwritten to the *current* weights on every
                consolidate. Cheap, but the anchor drifts toward the already-degraded
                current weights across a chain, so it defends a moving target.
            True — PER-TASK anchors: each consolidate freezes a separate
                (anchor, Fisher) snapshot and never overwrites past ones. The penalty
                sums one spring per past task, each pulling toward where that task was
                actually good (classic multi-task EWC). Costs O(#tasks) memory; removes
                the drifting-target failure mode.
        """
        self.ewc_lambda = ewc_lambda
        self.ewc_decay = ewc_decay
        self.ewc_per_task = per_task
        # per_task only: divide the summed penalty by #tasks so total spring force
        # stays ~constant as the chain grows (decouples lambda from chain length).
        self.ewc_normalize = normalize
        # single-anchor state (per_task=False)
        self.fisher = None          # running diagonal Fisher (dict) or None
        self.anchor_params = None   # consolidated weights (dict) or None
        # per-task state (per_task=True): list of {"fisher": {...}, "anchor": {...}}
        self.tasks = []

    # ------------------------------------------------------------------
    # Base-specific: compute the EP gradient of one transition.
    # ------------------------------------------------------------------
    def _fisher_transition_grads(self, x_t: np.ndarray, x_next: np.ndarray) -> dict:
        """Return {W_ih, W_ho, b_h, b_o} EP gradients for one transition.
        Must be implemented by the concrete subclass (DG-separated vs raw input).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Per-epoch penalty (delivered via the base class hook)
    # ------------------------------------------------------------------
    def _apply_ewc_penalty(self) -> None:
        """One gradient step on the EWC quadratic penalty: pull each weight toward
        its consolidated anchor with per-weight strength lambda*Fisher. Delivered
        once per batch epoch (offline) or once per event update (online) — via
        whichever hook the active ingestion path calls — so EWC is applied under
        both regimes with no duplicated training-loop code."""
        if self.ewc_lambda <= 0.0:
            return
        lr = self.learning_rate
        if self.ewc_per_task:
            # one spring per frozen task snapshot, summed (optionally averaged so
            # total force is chain-length-independent)
            if not self.tasks:
                return
            scale = (1.0 / len(self.tasks)) if self.ewc_normalize else 1.0
            for task in self.tasks:
                F, anchor = task["fisher"], task["anchor"]
                for name in self._WEIGHTS:
                    W = getattr(self, name)
                    W -= lr * self.ewc_lambda * scale * F[name] * (W - anchor[name])
        else:
            if self.fisher is None or self.anchor_params is None:
                return
            for name in self._WEIGHTS:
                W = getattr(self, name)
                W -= lr * self.ewc_lambda * self.fisher[name] * (W - self.anchor_params[name])

    def _on_epoch_end(self) -> None:
        # batch ingestion (fit_sequence) delivers the penalty once per epoch
        self._apply_ewc_penalty()

    def _on_update_end(self) -> None:
        # online ingestion (fit_event) delivers the penalty once per event update
        self._apply_ewc_penalty()

    # ------------------------------------------------------------------
    # Fisher estimation & consolidation
    # ------------------------------------------------------------------
    def estimate_fisher(self, sequence_data: np.ndarray) -> dict:
        """Diagonal Fisher ~ mean over transitions of the squared EP gradient."""
        F = {name: np.zeros_like(getattr(self, name)) for name in self._WEIGHTS}
        n = 0
        for t in range(sequence_data.shape[0] - 1):
            g = self._fisher_transition_grads(sequence_data[t], sequence_data[t + 1])
            for name in self._WEIGHTS:
                F[name] += g[name] ** 2
            n += 1
        if n > 0:
            for name in F:
                F[name] /= n
        return F

    def consolidate(self, sequence_data: np.ndarray, context_data=None, **kwargs) -> None:
        """Estimate this sequence's Fisher and snapshot the current weights. Called
        by the benchmark after each task. In per-task mode this appends a frozen
        (anchor, Fisher) snapshot; in single-anchor mode it accumulates Fisher and
        overwrites the one anchor with the current weights."""
        F_new = self.estimate_fisher(sequence_data)
        anchor = {name: getattr(self, name).copy() for name in self._WEIGHTS}
        if self.ewc_per_task:
            # freeze a separate snapshot for this task; never touch earlier ones
            self.tasks.append({"fisher": F_new, "anchor": anchor})
        else:
            if self.fisher is None:
                self.fisher = F_new
            else:
                for name in F_new:
                    self.fisher[name] = (1.0 - self.ewc_decay) * self.fisher[name] + F_new[name]
            self.anchor_params = anchor

    def reset_ewc(self) -> None:
        """Clear consolidated state (Fisher + anchors, both modes)."""
        self.fisher = None
        self.anchor_params = None
        self.tasks = []
