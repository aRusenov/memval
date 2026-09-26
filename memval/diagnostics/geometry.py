"""
Representation-geometry tier: how much do two groups of items share the SAME
active units, per named layer.

Given a model exposing `named_representations(x)`, this computes — for every
layer the model reports — the mean pairwise Jaccard between an active-unit set of
a group-A item and a group-B item, alongside within-group overlap, the measured
density, and the random-chance Jaccard floor for that density (so a low number is
interpretable). This is the metric that localizes WHERE separation is achieved or
lost (e.g. input Jaccard low but hidden Jaccard high == separation didn't
propagate).
"""

import numpy as np

from .protocols import supports_representations

__all__ = ["representation_overlap"]


def _active_sets(vectors, eps):
    return [frozenset(np.nonzero(np.asarray(v).ravel() > eps)[0].tolist()) for v in vectors]


def _mean_pairwise_jaccard(sets_a, sets_b, exclude_identity=False):
    vals = []
    for i, A in enumerate(sets_a):
        for j, B in enumerate(sets_b):
            if exclude_identity and A is B and i == j:
                continue
            u = len(A | B)
            vals.append(len(A & B) / u if u else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def _density(vectors, eps):
    vs = [np.asarray(v).ravel() for v in vectors]
    n = vs[0].size
    return float(np.mean([np.count_nonzero(v > eps) for v in vs]) / n)


def _jaccard_floor(pa, pb):
    denom = pa + pb - pa * pb
    return (pa * pb) / denom if denom > 0 else 0.0


def representation_overlap(model, items_a, items_b, eps: float = 1e-6) -> dict:
    """Per-layer Jaccard overlap between two groups of raw inputs.

    Parameters
    ----------
    model : must implement `named_representations`.
    items_a, items_b : iterables of raw input vectors (one per item).
    eps : activation threshold defining an "active" unit.

    Returns
    -------
    ``{layer_name: {jaccard_ab, jaccard_within_a, jaccard_within_b,
    density_a, density_b, floor}}``. Empty dict if the model lacks the capability.
    """
    if not supports_representations(model):
        return {}

    reps_a = [model.named_representations(x) for x in items_a]
    reps_b = [model.named_representations(x) for x in items_b]
    layers = list(reps_a[0].keys())

    out = {}
    for layer in layers:
        va = [r[layer] for r in reps_a]
        vb = [r[layer] for r in reps_b]
        sa, sb = _active_sets(va, eps), _active_sets(vb, eps)
        pa, pb = _density(va, eps), _density(vb, eps)
        out[layer] = {
            "jaccard_ab": _mean_pairwise_jaccard(sa, sb),
            "jaccard_within_a": _mean_pairwise_jaccard(sa, sa, exclude_identity=True),
            "jaccard_within_b": _mean_pairwise_jaccard(sb, sb, exclude_identity=True),
            "density_a": pa,
            "density_b": pb,
            "floor": _jaccard_floor(pa, pb),
        }
    return out
