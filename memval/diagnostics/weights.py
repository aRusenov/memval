"""
Weight-space tier: interference and importance in parameter space.

Three views, all consuming the `transition_grads` / `named_parameters` capability
and all model-agnostic (they operate on dicts of arrays keyed by parameter group,
never on hardcoded matrix names):

  * `update_interference`  — cosine of per-transition update DIRECTIONS, per group.
    High off-diagonal (task A rows x task B cols) == the two tasks steer the same
    synapses. Answers "which matrix collides."
  * `fisher_diagonal`      — diagonal Fisher per group = mean squared gradient.
    A's importance map (valid as a Fisher because the EP update IS a gradient est).
  * `fisher_attribution`   — the EWC 2nd-order forgetting estimate
    sum_i F_A[i] (theta_B[i] - theta_A[i])^2, split per group into the product and
    its two factors (importance mass vs displacement). Answers "where does the
    realized forgetting live" — including the biases the cosine view can't see.

Diagonal Fisher (no cross-weight terms) and a local 2nd-order estimate are the
standing caveats; see the session notes.
"""

import numpy as np

from .protocols import supports_gradients

__all__ = [
    "transition_signatures", "update_interference",
    "fisher_diagonal", "snapshot_parameters", "fisher_attribution",
]


def _block_mean_abs(J, rows, cols, exclude_diag=False):
    sub = np.abs(J[np.ix_(list(rows), list(cols))])
    if exclude_diag and sub.shape[0] == sub.shape[1]:
        mask = ~np.eye(sub.shape[0], dtype=bool)
        return float(sub[mask].mean()) if mask.any() else float("nan")
    return float(sub.mean()) if sub.size else float("nan")


def transition_signatures(model, transitions) -> dict:
    """Per-group matrix of L2-normalised, flattened update directions, one row per
    transition. `transitions` is an iterable of (x_t, x_next) vector pairs."""
    grads = [model.transition_grads(xt, xn) for xt, xn in transitions]
    out = {}
    for g in grads[0]:
        M = np.vstack([np.asarray(gr[g]).ravel() for gr in grads])
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        out[g] = M / norms
    return out


def update_interference(model, transitions_a, transitions_b) -> dict:
    """Per-group cosine interference between two task's update directions.

    Returns ``{group: {ab, within_a, within_b}}`` where each value is a mean
    |cos|. Evaluate at init (all transitions un-learned) for the cleanest read;
    a learned transition's update shrinks to ~0 and its direction is meaningless.
    """
    if not supports_gradients(model):
        return {}
    ta, tb = list(transitions_a), list(transitions_b)
    sigs = transition_signatures(model, ta + tb)
    na = len(ta)
    a_idx, b_idx = range(na), range(na, na + len(tb))
    out = {}
    for g, M in sigs.items():
        J = M @ M.T
        out[g] = {
            "ab": _block_mean_abs(J, a_idx, b_idx),
            "within_a": _block_mean_abs(J, a_idx, a_idx, exclude_diag=True),
            "within_b": _block_mean_abs(J, b_idx, b_idx, exclude_diag=True),
        }
    return out


def fisher_diagonal(model, transitions) -> dict:
    """Diagonal Fisher per group = mean over transitions of the squared gradient,
    at the current weights. `transitions`: iterable of (x_t, x_next) vector pairs."""
    params = model.named_parameters()
    F = {g: np.zeros_like(np.asarray(p, dtype=float)) for g, p in params.items()}
    n = 0
    for xt, xn in transitions:
        grad = model.transition_grads(xt, xn)
        for g in F:
            F[g] += np.asarray(grad[g], dtype=float) ** 2
        n += 1
    if n:
        for g in F:
            F[g] /= n
    return F


def snapshot_parameters(model) -> dict:
    """Deep copy of the named parameters (for before/after displacement)."""
    return {g: np.asarray(p, dtype=float).copy() for g, p in model.named_parameters().items()}


def fisher_attribution(fisher_map, params_before, params_after) -> dict:
    """Per-group second-order forgetting attribution.

    ``contrib[g] = sum_i F[g][i] * (after[g][i] - before[g][i])**2`` — an estimate
    of how much moving group g's weights (from `before` to `after`) raised the
    Fisher-anchored task's loss. Also returns the two factors so you can see
    whether a group dominates via importance (`fmass`) or displacement (`disp`),
    plus each as a share of the total.
    """
    groups = list(fisher_map.keys())
    contrib, fmass, disp = {}, {}, {}
    for g in groups:
        d2 = (params_after[g] - params_before[g]) ** 2
        contrib[g] = float((fisher_map[g] * d2).sum())
        fmass[g] = float(fisher_map[g].sum())
        disp[g] = float(d2.sum())

    def shares(d):
        tot = sum(d.values()) or 1.0
        return {g: d[g] / tot for g in groups}

    return {
        "contrib": contrib, "contrib_share": shares(contrib),
        "fmass": fmass, "fmass_share": shares(fmass),
        "disp": disp, "disp_share": shares(disp),
    }
