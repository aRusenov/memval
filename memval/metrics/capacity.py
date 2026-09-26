"""
Capacity / stable-state metrics for sequence-associative memory models.

These metrics treat a trained model as a *transition associator*: ``predict_next``
maps an item embedding ``x`` to a settled output ``x' = f(x)``.  A stored episode
``w0 -> w1 -> ... -> wL`` is a trajectory under ``f``; autoregressive recall is
iterating ``f``.  That framing makes three questions well-posed:

  * **Interference** — which stored memories collide?  Measured behaviourally with
    :func:`crosstalk_matrix` (cue -> target cosine) and mechanistically with
    :func:`synaptic_interference_matrix` (overlap of the EP weight updates each
    transition would apply).
  * **Spurious attractors** — where does recall go *instead* when it collides?
    :func:`attractor_scan` iterates ``f`` from many seeds, detects fixed points /
    cycles, and reports the basin-weighted fraction that are not stored memories.
  * **Retained information** — how many bits survive?  :func:`recall_confusion`
    builds a noisy-channel confusion matrix and :func:`channel_mi` turns it into a
    threshold-free, chance-corrected bit count (with a category decomposition via
    :func:`category_information`).

All functions are read-only with respect to the model (they never call
``fit_sequence`` and never write weights) and depend only on numpy.

The public entry points are model-agnostic but assume the ``HippocampalModel``
interface: ``predict_next(vec) -> vec``.  The synaptic and attractor helpers
additionally use EqProp internals (``_settle``, ``beta``, ``W_ih``, ``W_ho``),
which are guarded so they degrade gracefully on models that lack them.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "crosstalk_matrix",
    "diagonal_dominance",
    "recall_confusion",
    "channel_mi",
    "category_information",
    "ep_update_signature",
    "synaptic_interference_matrix",
    "attractor_scan",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector, safe against the zero vector."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _predict(model: Any, vec: np.ndarray) -> np.ndarray:
    """Call the model's single-step map, tolerating the optional context kwarg."""
    return np.asarray(model.predict_next(vec)).ravel()


# ---------------------------------------------------------------------------
# a) interference — behavioural crosstalk
# ---------------------------------------------------------------------------
def crosstalk_matrix(
    model: Any,
    cue_words: Sequence[str],
    target_words: Sequence[str],
    encoder: Any,
) -> np.ndarray:
    """Behavioural interference matrix ``S[i, j]``.

    ``S[i, j] = cos( f(embed(cue_i)), embed(target_j) )`` where ``f`` is one
    ``predict_next`` step.  When ``cue_words[i]`` is stored with successor
    ``target_words[i]``, the diagonal is self-recall fidelity and off-diagonal
    entries are pull toward other memories' targets.  With category-structured
    embeddings the off-diagonal reveals block structure by category.

    Parameters
    ----------
    model : trained model exposing ``predict_next``.
    cue_words : words to present as cues (rows).
    target_words : candidate successor words to score against (columns).
    encoder : ``SymbolicEncoder`` (uses ``.encode``).

    Returns
    -------
    (len(cue_words), len(target_words)) cosine-similarity matrix.
    """
    T = encoder.encode(list(target_words))                     # (Nt, d)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    S = np.empty((len(cue_words), len(target_words)), dtype=float)
    for i, w in enumerate(cue_words):
        cue = encoder.encode([w])[0]
        out = _unit(_predict(model, cue))
        S[i] = T @ out
    return S


def diagonal_dominance(S: np.ndarray) -> float:
    """Mean margin between each cue's own target and its strongest competitor.

    ``mean_i ( S[i, i] - max_{j != i} S[i, j] )``.  Positive => memories are
    separable; near zero / negative => interference.  Assumes ``S`` is square and
    aligned so that row ``i`` pairs with column ``i``.
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    if n < 2:
        return float("nan")
    off = S.copy()
    np.fill_diagonal(off, -np.inf)
    return float(np.mean(np.diag(S) - off.max(axis=1)))


# ---------------------------------------------------------------------------
# c) information-theoretic — noisy-channel confusion + mutual information
# ---------------------------------------------------------------------------
def recall_confusion(
    model: Any,
    transitions: Sequence[Tuple[str, str]],
    encoder: Any,
    decoder: Any,
    noise_scale: float,
    n_trials: int,
    rng: np.random.Generator,
    y_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Estimate the recall channel's confusion counts under cue noise.

    For each stored ``(cue, target)`` transition and each trial, the cue embedding
    is perturbed with isotropic Gaussian noise (matching ``measure_recall_*`` in
    the symbolic pipeline: additive, no renormalisation), pushed one step through
    the model, and decoded.  Row = intended target word, column = decoded word.

    Returns a dict with ``counts`` (n_x, n_y), ``x_labels`` (intended targets,
    de-duplicated & sorted) and ``y_labels`` (decode vocabulary; defaults to the
    full decoder vocabulary so nothing falls outside the matrix).
    """
    x_labels = sorted({t for _, t in transitions})
    if y_labels is None:
        y_labels = list(decoder.idx_to_word)
    else:
        y_labels = list(y_labels)
    x_idx = {w: i for i, w in enumerate(x_labels)}
    y_idx = {w: i for i, w in enumerate(y_labels)}

    counts = np.zeros((len(x_labels), len(y_labels)), dtype=float)
    dim = encoder.embedding_dim
    for cue_w, tgt_w in transitions:
        clean = encoder.encode([cue_w])[0]
        xi = x_idx[tgt_w]
        for _ in range(n_trials):
            noisy = clean + rng.normal(0.0, noise_scale, size=dim)
            pred = _predict(model, noisy)
            word, _score = decoder.decode_with_score(pred)
            yj = y_idx.get(word)
            if yj is not None:                                # in-vocab by construction
                counts[xi, yj] += 1.0
    return {"counts": counts, "x_labels": x_labels, "y_labels": y_labels}


def channel_mi(counts: np.ndarray) -> Dict[str, float]:
    """Mutual information (bits) of a joint/confusion count matrix.

    Works for rectangular ``counts`` (intended X vs decoded Y).  Returns:

      * ``mi_bits``       – I(X; Y) in bits (chance-corrected, threshold-free).
      * ``h_x_bits``      – source entropy H(X); the ceiling MI can reach.
      * ``efficiency``    – ``mi_bits / h_x_bits`` in [0, 1]; fraction of the
        stored distinction that survives recall.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return {"mi_bits": 0.0, "h_x_bits": 0.0, "efficiency": 0.0}
    P = counts / total
    Px = P.sum(axis=1, keepdims=True)
    Py = P.sum(axis=0, keepdims=True)
    mask = P > 0
    denom = (Px @ Py)
    mi = float(np.sum(P[mask] * np.log2(P[mask] / denom[mask])))
    px = Px.ravel()
    hx = float(-np.sum(px[px > 0] * np.log2(px[px > 0])))
    eff = mi / hx if hx > 0 else 0.0
    return {"mi_bits": mi, "h_x_bits": hx, "efficiency": eff}


def category_information(
    confusion: Dict[str, Any],
    word_to_category: Dict[str, str],
) -> Dict[str, float]:
    """Decompose retained information into category- and item-level terms.

    Collapsing the item confusion onto categories gives ``mi_category_bits`` (do we
    still know it's a *fruit*?).  The full item MI minus the category MI is the
    within-category information (do we know *which* fruit?) — this is what tends to
    erode first under interference.  Also returns coarse accuracies.

    Requires ``word_to_category`` covering all x/y labels in ``confusion``.
    """
    counts = np.asarray(confusion["counts"], dtype=float)
    x_labels: List[str] = list(confusion["x_labels"])
    y_labels: List[str] = list(confusion["y_labels"])

    x_cats = sorted({word_to_category[w] for w in x_labels})
    y_cats = sorted({word_to_category[w] for w in y_labels})
    xc_idx = {c: i for i, c in enumerate(x_cats)}
    yc_idx = {c: i for i, c in enumerate(y_cats)}

    cat_counts = np.zeros((len(x_cats), len(y_cats)), dtype=float)
    for i, xw in enumerate(x_labels):
        ci = xc_idx[word_to_category[xw]]
        for j, yw in enumerate(y_labels):
            cat_counts[ci, yc_idx[word_to_category[yw]]] += counts[i, j]

    item = channel_mi(counts)
    cat = channel_mi(cat_counts)

    total = counts.sum()
    # coarse accuracies (argmax decode already baked into the counts)
    correct_item = 0.0
    correct_cat = 0.0
    for i, xw in enumerate(x_labels):
        for j, yw in enumerate(y_labels):
            if xw == yw:
                correct_item += counts[i, j]
            if word_to_category[xw] == word_to_category[yw]:
                correct_cat += counts[i, j]

    return {
        "mi_item_bits": item["mi_bits"],
        "mi_category_bits": cat["mi_bits"],
        "mi_within_category_bits": max(item["mi_bits"] - cat["mi_bits"], 0.0),
        "acc_item": correct_item / total if total > 0 else 0.0,
        "acc_category": correct_cat / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# a) interference — mechanistic / synaptic (EqProp-specific, non-invasive)
# ---------------------------------------------------------------------------
def ep_update_signature(
    model: Any,
    cue_vec: np.ndarray,
    target_vec: np.ndarray,
) -> np.ndarray:
    """The one-shot EP weight update a single transition would apply *now*.

    Reproduces the free/nudge contrast from ``fit_sequence`` for one (cue, target)
    pair at the model's current weights, without modifying them.  Returns the
    concatenated, flattened ``(dW_ih, dW_ho)`` — a direction in weight space.
    Overlap of these directions across transitions is destructive interference.

    Raises ``AttributeError`` if the model is not an EqProp settling network.
    """
    beta = float(model.beta)
    cue = np.asarray(cue_vec).ravel()
    tgt = np.asarray(target_vec).ravel()
    s_h_free, s_o_free = model._settle(cue, target=None, beta=0.0)
    s_h_nudge, s_o_nudge = model._settle(cue, target=tgt, beta=beta)
    inv = 1.0 / beta
    dW_ho = inv * (np.outer(s_o_nudge, s_h_nudge) - np.outer(s_o_free, s_h_free))
    dW_ih = inv * (np.outer(s_h_nudge, cue) - np.outer(s_h_free, cue))
    return np.concatenate([dW_ih.ravel(), dW_ho.ravel()])


def synaptic_interference_matrix(
    model: Any,
    transitions: Sequence[Tuple[str, str]],
    encoder: Any,
) -> np.ndarray:
    """Cosine-overlap matrix ``J[a, b]`` of transitions' EP update signatures.

    ``J[a, b] = cos( g_a, g_b )`` where ``g`` is :func:`ep_update_signature`.
    High off-diagonal => transitions compete for the same synapses (interference
    that pattern separation is meant to reduce).  ``J`` is symmetric with unit
    diagonal.
    """
    sigs = []
    for cue_w, tgt_w in transitions:
        cue = encoder.encode([cue_w])[0]
        tgt = encoder.encode([tgt_w])[0]
        sigs.append(_unit(ep_update_signature(model, cue, tgt)))
    G = np.vstack(sigs)
    return G @ G.T


# ---------------------------------------------------------------------------
# b) spurious attractor scan
# ---------------------------------------------------------------------------
def attractor_scan(
    model: Any,
    seeds: np.ndarray,
    decoder: Any,
    stored_words: Sequence[str],
    max_steps: int = 40,
    tol: float = 1e-3,
    cycle_tol: float = 1e-2,
) -> Dict[str, Any]:
    """Iterate ``f = predict_next`` from each seed and classify where it lands.

    From every seed we run the autoregressive map until it reaches a fixed point
    (``||x_{k+1} - x_k|| < tol``), re-enters a previously visited state (a cycle),
    or hits ``max_steps``.  Each endpoint is decoded; an endpoint is *genuine* if
    it settles (fixed point or short cycle) onto a stored word, otherwise
    *spurious*.  Norm trajectories are tracked to expose the classic EqProp
    failure where chains drift off the unit sphere into an absorbing state.

    Parameters
    ----------
    seeds : (n_seeds, d) initial states (e.g. noise-perturbed stored items).
    stored_words : the set of words that count as genuine memories.
    tol, cycle_tol : fixed-point and cycle detection thresholds (cosine-distance
        style Euclidean on the settled vectors).

    Returns
    -------
    dict with:
      * ``records`` – per-seed dicts (endpoint word, kind, residual, steps,
        final_norm, norm_trace).
      * ``spurious_rate`` – basin-weighted fraction of seeds ending spurious.
      * ``n_distinct_attractors`` / ``n_spurious_attractors`` – distinct decoded
        endpoints among fixed points.
      * ``basin_counts`` – ``{endpoint_word: n_seeds}`` over settled seeds.
      * ``dominant_word`` / ``dominant_basin_frac`` – the most absorbing endpoint
        and the fraction of *all* seeds it swallows.  A single stored word with a
        near-1.0 fraction is the "black-hole" collapse — degenerate even though it
        is technically a genuine memory.
      * ``norm_traces`` – list of per-seed ``||x_k||`` arrays (for plotting).
    """
    stored = set(stored_words)
    records: List[Dict[str, Any]] = []
    norm_traces: List[np.ndarray] = []
    distinct: set = set()
    spurious_attr: set = set()
    basin_counts: Dict[str, int] = {}

    for seed in seeds:
        x = np.asarray(seed, dtype=float).ravel()
        history = [x.copy()]
        norms = [float(np.linalg.norm(x))]
        kind = "no_converge"
        steps = max_steps
        for k in range(max_steps):
            nxt = _predict(model, x)
            norms.append(float(np.linalg.norm(nxt)))
            if np.linalg.norm(nxt - x) < tol:                 # fixed point
                kind = "fixed_point"
                steps = k + 1
                x = nxt
                break
            hit_cycle = False
            for prev in history[:-1]:                         # cycle (period >= 2)
                if np.linalg.norm(nxt - prev) < cycle_tol:
                    hit_cycle = True
                    break
            x = nxt
            history.append(x.copy())
            if hit_cycle:
                kind = "cycle"
                steps = k + 1
                break

        residual = float(np.linalg.norm(_predict(model, x) - x))
        word, score = decoder.decode_with_score(x)
        settled = kind in ("fixed_point", "cycle")
        genuine = settled and (word in stored)
        label = "genuine" if genuine else "spurious"
        if settled:
            distinct.add(word)
            basin_counts[word] = basin_counts.get(word, 0) + 1
            if not genuine:
                spurious_attr.add(word)

        records.append(
            {
                "endpoint_word": word,
                "endpoint_score": float(score),
                "kind": kind,
                "label": label,
                "residual": residual,
                "steps": steps,
                "final_norm": norms[-1],
                "norm_trace": np.asarray(norms),
            }
        )
        norm_traces.append(np.asarray(norms))

    n = len(records)
    spurious_rate = sum(r["label"] == "spurious" for r in records) / n if n else 0.0
    if basin_counts:
        dominant_word = max(basin_counts, key=basin_counts.get)
        dominant_frac = basin_counts[dominant_word] / n if n else 0.0
    else:
        dominant_word, dominant_frac = None, 0.0
    return {
        "records": records,
        "spurious_rate": spurious_rate,
        "n_distinct_attractors": len(distinct),
        "n_spurious_attractors": len(spurious_attr),
        "basin_counts": basin_counts,
        "dominant_word": dominant_word,
        "dominant_basin_frac": dominant_frac,
        "norm_traces": norm_traces,
    }
