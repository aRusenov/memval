"""Cognitive phenomena: the L0 behavioural signatures of list recall.

This section reads the *shape* of recall off the suite's own probes and reports
it against the human reference band from the list-recall literature (Kahana
2020, Annu. Rev. Psychol. 71, section 4, "Benchmark recall phenomena"). It is
**descriptive, not evaluative**: nothing here enters the capacity scorecard, no
key is written to ``results["metrics"]``, and the output carries the human
signature beside the model's read-out so a reader can compare direction and
magnitude without the section deciding for them. The point is the L0 rung of
the correspondence ladder (docs/cognitive_phenomena_design.md): a model that
declares a behavioural correspondence is obliged to these read-outs, and a
model that does not is merely described by them.

Five phenomena, three of which fall out of ONE material grid:

  recall[position, list length, exposure]

* **Serial position** -- the curve at the headline length. Human: immediate
  serial recall shows strong primacy and modest recency (Kahana 2020 Fig. 2b).
* **List length** -- the curve at every length. Human: longer lists cost the
  early and middle positions, not the last few (Murdock 1962; Kahana 2020
  Fig. 2a).
* **Presentation rate** -- the curve at every exposure. Human: a slower rate
  helps early and middle items and leaves the final items unchanged (Murdock
  1962). In the batch suite "rate" is passes over the list; the section says
  so and reads the dissociation, not the absolute level.

and two from their own material:

* **Prior-list intrusion recency** -- a chain of lists trained in order; when
  the just-trained list is probed, where do its errors come from? Human: most
  prior-list intrusions come from the last two lists and fall off with list
  lag (Zaromb et al. 2006; Kahana 2020 Fig. 5a). Under an exact between-list
  cosine every prior list is equidistant, so the geometric null is FLAT and a
  recency gradient is the memory's doing.
* **Semantic clustering** -- semantic partners placed at non-adjacent list
  positions so that the temporal successor, the cue's partner, the target's
  partner, and the successor of the cue's partner are four different items.
  An error is then classified by which of them it is, which separates
  organisation of retrieval (cue -> its semantic associate; the semantic-CRP
  analogue, Howard & Kahana 2002) from output-side confusability (landing on
  the target's sibling) and input-side generalisation (treating the cue as its
  sibling and following the chain). The within-category cosine is swept down
  to no structure as the calibration rung: every semantic share must fall to
  chance there or the measure is contaminated.

Two things a reader must not do with these curves, stated once here and again
in the design note. (1) The autoregressive (rollout) curve cannot show recency:
once a chained rollout errs it is off the manifold, so it decays monotonically
by construction; primacy there is error propagation. The serial-position
read-outs therefore come from the independently-cued probe, and the rollout
curve is drawn beside it precisely to make the contrast visible. (2) The
reference data are human word lists recalled with rehearsal and strategic
search; no arm here rehearses. The claim is never "arm X reproduces the human
curve", only "the shape of arm X's curve under this protocol, against the
human shape as a band".

Probe protocol: CLEAN_SINGLE (one clean cue per transition, margin beside
accuracy). Replication is over material (categories / lists), never over cue
noise, per the house rule. Exposure is criterion-referenced on cued recall and
the read-outs are taken at the criterion rung AND at a short fixed ladder of
exposures below it, because a curve read only at criterion is at ceiling by
construction and its shape lives in the margin alone.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from memval.encoders.symbolic import SymbolicDecoder, SymbolicEncoder
from memval.benchmarks.exposure import epochs_to_criterion
from memval.benchmarks.continual_chain import build_chain_material, consolidate_if_supported

SECTION = "cognitive_phenomena"
PHENOMENA = ("serial_position", "list_length", "presentation_rate",
             "pli_recency", "semantic_clustering")

#: The human signature each read-out is reported against. Qualitative on
#: purpose: the quantities are not commensurable with human recall
#: probabilities, only their direction and ordering are.
HUMAN_REFERENCE: Dict[str, Dict[str, str]] = {
    "serial_position": {
        "signature": "Immediate serial recall: strong primacy, modest recency; "
                     "immediate free recall: strong recency, smaller primacy.",
        "read_as": "primacy_index > 0 and recency_index > 0 on the CUED curve; "
                   "the rollout curve is structurally unable to show recency.",
        "source": "Kahana 2020 sec 4.1, Fig. 2b (Kahana et al. 2010); Murdock 1962.",
    },
    "list_length": {
        "signature": "Longer lists reduce recall of early and middle positions and "
                     "do not reliably change recall of the last few items.",
        "read_as": "early_slope < 0 and late_slope ~ 0, i.e. dissociation = "
                   "late_slope - early_slope > 0.",
        "source": "Kahana 2020 sec 4.1, Fig. 2a (Murdock 1962).",
    },
    "presentation_rate": {
        "signature": "A slower presentation rate increases recall of early and middle "
                     "items and has no discernible effect on the final items.",
        "read_as": "early_slope > 0 and late_slope ~ 0 against log2(passes), i.e. "
                   "dissociation = early_slope - late_slope > 0. 'Rate' is passes "
                   "over the list in the batch suite, not seconds per item.",
        "source": "Kahana 2020 sec 4.1 (Murdock 1962).",
    },
    "pli_recency": {
        "signature": "Prior-list intrusions show a striking recency effect: most come "
                     "from the last two lists and the proportion falls with list lag.",
        "read_as": "pli_lag1_share high and pli_recency_spearman < 0 (rate per "
                   "opportunity decreasing in lag). The competitor read-out is the "
                   "same question above the accuracy floor.",
        "source": "Kahana 2020 sec 4.3, Fig. 5a (Zaromb et al. 2006).",
    },
    "semantic_clustering": {
        "signature": "Recall transitions and intrusions are biased toward items "
                     "semantically similar to the just-recalled item, even on lists "
                     "without obvious associates.",
        "read_as": "cue_partner share above chance_specific and semantic_factor_cue "
                   "> 0.5 at the structured rungs, both falling to chance at the "
                   "no-structure calibration rung. target_partner and "
                   "cue_partner_successor are confusability, not organisation.",
        "source": "Kahana 2020 sec 4.2-4.3, Fig. 4b, 5c (Howard & Kahana 2002; "
                  "Zaromb et al. 2006); semantic factor after Polyn et al. 2009.",
    },
}

#: Semantic-clustering list layout. Letters are categories, positions are list
#: slots; each letter appears twice, its two slots are semantic partners. The
#: layout is chosen so that for every cue i (target i+1) the four items
#: {target, partner(cue), partner(target), successor(partner(cue))} are all
#: distinct and none is a temporal neighbour of the target; see
#: ``validate_semantic_pattern`` and tests/test_cognitive_phenomena.py.
SEMANTIC_PATTERN = "ABCDBADC"

#: Error classes for the semantic section, in the order they are tested.
SEMANTIC_CLASSES = (
    "cue_partner",             # S1: the cue's semantic associate  (organisation)
    "target_partner",          # S2: the target's semantic sibling (output confusability)
    "cue_partner_successor",   # S3: chain followed from the cue's sibling (input generalisation)
    "cue_repeat",
    "temporal_neighbour",
    "other_in_list",
    "eli_cue_category",
    "eli_target_category",
    "eli_list_category",
    "eli_other_category",
)


# ---------------------------------------------------------------------------
# model construction, training, probing
# ---------------------------------------------------------------------------

def _make_model(model_class: Type, enc: Any, model_kwargs: Dict[str, Any], epochs: int):
    kw = dict(model_kwargs)
    kw["n_epochs"] = int(epochs)
    kw["epochs"] = int(epochs)
    if "encoder" in model_class.__init__.__code__.co_varnames:
        return model_class(encoder=enc, n_features=enc.embedding_dim, **kw)
    return model_class(n_features=enc.embedding_dim, **kw)


def _fit(model: Any, X: np.ndarray, epochs: int) -> None:
    if hasattr(model, "n_epochs"):
        model.n_epochs = int(epochs)
    model.fit_sequence(X, epochs=int(epochs))


def train_at(policy: Dict[str, Any], build: Callable[[int], Any],
             fit: Callable[[Any, int], None], score: Callable[[Any], float]
             ) -> Tuple[Any, Dict[str, Any]]:
    """Train under an exposure policy and report what it took.

    Mirrors ``train_at_exposure`` in the symbolic pipeline: a pinned budget
    trains once and records whether the criterion was met there; otherwise the
    staircase settles the exposure and returns the model at it.
    """
    if policy["mode"] == "fixed":
        m = build(policy["epochs"])
        fit(m, policy["epochs"])
        sc = float(score(m))
        return m, {"epochs": int(policy["epochs"]), "reached": sc >= policy["criterion"],
                   "mode": "fixed", "score": sc}
    res = epochs_to_criterion(
        lambda: build(policy["max_epochs"]), fit, score,
        criterion=float(policy["criterion"]), max_epochs=int(policy["max_epochs"]))
    return res["model"], {"epochs": int(res["epochs"]), "reached": bool(res["reached"]),
                          "mode": "criterion", "score": float(res["score"])}


def _unit_rows(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def cued_probe(model: Any, words: Sequence[str], encoder: Any) -> Dict[str, Any]:
    """One clean cue per transition; everything the section reads, at once.

    Position 0 is the cue and is NaN in every per-position array. For scored
    positions ``p = 1 .. L-1``: ``hit[p]`` (nearest-neighbour decode over the
    encoder's whole vocabulary equals the target), ``margin[p]`` (cos to target
    minus cos to the best competitor), ``decoded[p]`` and ``competitor[p]``
    (words). ``sims`` is the ``(L-1, V)`` cosine row per probe.
    """
    words = list(words)
    L = len(words)
    E = _unit_rows(encoder.embeddings)
    vocab = list(encoder.idx_to_word)
    idx_of = {w: k for k, w in enumerate(vocab)}
    X = np.asarray(encoder.encode(words), dtype=float)
    ctx = np.array([1.0])

    hit = np.full(L, np.nan)
    margin = np.full(L, np.nan)
    decoded: List[Optional[str]] = [None] * L
    competitor: List[Optional[str]] = [None] * L
    sims_all = np.zeros((L - 1, len(vocab)))
    if hasattr(model, "reset_context"):
        model.reset_context()
    for i in range(L - 1):
        model.current_t = i
        pred = np.asarray(model.predict_next(X[i].copy(), current_context=ctx), dtype=float).ravel()
        pn = float(np.linalg.norm(pred))
        sims = E @ (pred / pn) if pn > 1e-12 else np.zeros(len(vocab))
        sims_all[i] = sims
        ti = idx_of[words[i + 1]]
        others = np.delete(np.arange(len(vocab)), ti)
        ci = int(others[np.argmax(sims[others])])
        di = int(np.argmax(sims)) if pn > 1e-12 else ci   # a null prediction loses
        decoded[i + 1] = vocab[di]
        competitor[i + 1] = vocab[ci]
        hit[i + 1] = float(di == ti)
        margin[i + 1] = float(sims[ti] - sims[ci])
    return {"hit": hit, "margin": margin, "decoded": decoded,
            "competitor": competitor, "sims": sims_all}


def rollout_probe(model: Any, words: Sequence[str], encoder: Any,
                  feedback_mode: str = "l2") -> np.ndarray:
    """Autoregressive curve from a clean initial cue (harness-controlled
    feedback). Position 0 is NaN. Drawn beside the cued curve only to show
    that it cannot carry recency -- see the module note."""
    from memval.benchmarks.symbolic_pipeline import measure_recall_autoregressive
    dec = SymbolicDecoder(encoder)
    curve = np.asarray(measure_recall_autoregressive(
        model, list(words), encoder, dec, n_trials=1, noise_scale=0.0,
        feedback_mode=feedback_mode), dtype=float)
    curve[0] = np.nan
    return curve


# ---------------------------------------------------------------------------
# serial-position statistics
# ---------------------------------------------------------------------------

def position_bands(L: int) -> Dict[str, List[int]]:
    """Early / middle / late bands over the scored positions ``1 .. L-1``.

    Two positions per end when at least five are scored, otherwise one; the
    middle may be empty for short lists (its mean is then NaN).
    """
    scored = list(range(1, L))
    k = 2 if len(scored) >= 5 else 1
    return {"early": scored[:k], "middle": scored[k:len(scored) - k], "late": scored[-k:]}


def band_means(curve: Sequence[float]) -> Dict[str, float]:
    """Band means and the two serial-position indices on one curve."""
    c = np.asarray(curve, dtype=float)
    bands = position_bands(len(c))
    out = {}
    for name, idx in bands.items():
        out[name] = float(np.nanmean(c[idx])) if idx else float("nan")
    out["primacy_index"] = out["early"] - out["middle"]
    out["recency_index"] = out["late"] - out["middle"]
    return out


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(x[ok]) == 0:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(y[ok]) == 0 or np.ptp(x[ok]) == 0:
        return float("nan")
    from memval.benchmarks.continual_chain import spearman
    return float(spearman(x[ok], y[ok]))


def _nanmean(values: Sequence[float]) -> float:
    """NaN-safe mean that is NaN (silently) on an empty or all-NaN input."""
    v = np.asarray(list(values), dtype=float)
    ok = np.isfinite(v)
    return float(v[ok].mean()) if ok.any() else float("nan")


def _nanmean_curves(curves: Sequence[Sequence[float]]) -> List[float]:
    A = np.asarray(curves, dtype=float)
    return [_nanmean(A[:, j]) for j in range(A.shape[1])]


def _nanstd_curves(curves: Sequence[Sequence[float]]) -> List[float]:
    A = np.asarray(curves, dtype=float)
    out = []
    for j in range(A.shape[1]):
        col = A[:, j][np.isfinite(A[:, j])]
        out.append(float(col.std()) if col.size else float("nan"))
    return out


# ---------------------------------------------------------------------------
# phenomena 1-3: the position x length x exposure grid
# ---------------------------------------------------------------------------

def _readouts(model: Any, words: Sequence[str], enc: Any, feedback_mode: str) -> Dict[str, Any]:
    pr = cued_probe(model, words, enc)
    roll = rollout_probe(model, words, enc, feedback_mode)
    return {
        "cued_hit": [float(v) for v in pr["hit"]],
        "cued_margin": [float(v) for v in pr["margin"]],
        "rollout_hit": [float(v) for v in roll],
        "bands_hit": band_means(pr["hit"]),
        "bands_margin": band_means(pr["margin"]),
        "bands_rollout": band_means(roll),
    }


def run_position_grid(
    model_class: Type,
    model_kwargs: Dict[str, Any],
    vocab: Dict[str, str],
    policy: Dict[str, Any],
    lengths: Sequence[int] = (5, 7, 10),
    categories: Sequence[str] = ("fruit", "animal", "number"),
    ladder: Sequence[int] = (1, 2, 4, 8),
    n_seeds: int = 5,
    embedding_dim: int = 100,
    category_variance: float = 0.2,
    seed: int = 42,
    feedback_mode: str = "l2",
) -> Dict[str, Any]:
    """Cued accuracy, margin and rollout accuracy at every (category, seed,
    length, exposure) cell.

    **The replicate unit is (category, encoder seed), not category.** With the
    six-category vocabulary only three categories hold ten words, so categories
    alone cap the sample at three -- and at n = 3 every band index below flips
    sign across replicates (measured 2026-09-07 on AHN, theta and DTS-ESN), so
    no direction could be claimed from it. Re-drawing the same category's
    embeddings under a fresh seed is another draw of the same material process,
    which is the quantity these curves average over. ``n_seeds = 5`` gives 15
    replicates; the summary reports the standard error and refuses a direction
    when the interval spans zero.

    One encoder per (category, seed) over ALL of that category's words, so the
    decode candidate set is the same at every length (chance does not move with
    L) and the first L words carry identical vectors at every length (the
    encoder draws one sample per word in vocabulary order).
    """
    cells: Dict[str, Any] = {}
    for cat in categories:
        cat_words = [w for w, c in vocab.items() if c == cat]
        if len(cat_words) < max(lengths):
            raise ValueError(f"category {cat!r} has {len(cat_words)} words, "
                             f"need >= {max(lengths)} for lengths {tuple(lengths)}")
        for si in range(int(n_seeds)):
            # Seed per (category, seed index). A one-category encoder draws its
            # prototype first and one sample per word after, so two categories
            # under the SAME seed get identical vectors with different labels
            # and the "replicates" collapse to one sample (caught on the first
            # AHN run: zero SD). Seeds are offset by category so no two
            # (category, seed) pairs collide either.
            enc_seed = int(seed) + 1000 * (list(categories).index(cat) + 1) + si
            enc = SymbolicEncoder({w: cat for w in cat_words}, embedding_dim=embedding_dim,
                                  category_variance=category_variance, seed=enc_seed)
            for L in lengths:
                words = cat_words[:L]
                X = enc.encode(words)

                def _build(ep, _e=enc):
                    return _make_model(model_class, _e, model_kwargs, ep)

                def _fitx(m, ep, _X=X):
                    _fit(m, _X, ep)

                def _score(m, _w=words, _e=enc):
                    return _nanmean(cued_probe(m, _w, _e)["hit"][1:])

                model_c, rec = train_at(policy, _build, _fitx, _score)
                rungs: Dict[str, Any] = {}
                for ep in ladder:
                    m = _build(ep)
                    _fitx(m, ep)
                    rungs[str(int(ep))] = _readouts(m, words, enc, feedback_mode)
                rungs["criterion"] = _readouts(model_c, words, enc, feedback_mode)
                cells[f"{cat}:s{si}:{L}"] = {
                    "category": cat, "seed_index": int(si), "length": int(L),
                    "words": list(words), "n_candidates": len(cat_words),
                    "encoder_seed": enc_seed, "exposure": rec, "rungs": rungs,
                }
            print(f"  [grid] {cat:>7} seed {si} criterion at "
                  f"{[cells[f'{cat}:s{si}:{L}']['exposure']['epochs'] for L in lengths]} epochs "
                  f"for L={list(lengths)}")
    return {"cells": cells, "lengths": [int(L) for L in lengths],
            "categories": list(categories), "ladder": [int(e) for e in ladder],
            "n_seeds": int(n_seeds), "n_replicates": len(categories) * int(n_seeds),
            "rungs": [str(int(e)) for e in ladder] + ["criterion"],
            "geometry": {"embedding_dim": int(embedding_dim),
                         "category_variance": float(category_variance),
                         "within_cos_law": 1.0 / (1.0 + embedding_dim * category_variance ** 2)}}


def _ci(values: Sequence[float]) -> Dict[str, float]:
    """Mean, SD, standard error and a normal 95% interval over replicates.

    ``spans_zero`` is the gate on every direction claim: a band index whose
    interval covers zero states no direction, however large its mean.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = int(v.size)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0, "spans_zero": True}
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    lo, hi = (m - 1.96 * se, m + 1.96 * se) if n > 1 else (float("nan"), float("nan"))
    return {"mean": m, "sd": sd, "se": float(se) if n > 1 else float("nan"),
            "ci_lo": float(lo), "ci_hi": float(hi), "n": n,
            "spans_zero": bool(not np.isfinite(lo) or lo <= 0.0 <= hi)}


def derive_serial_position(grid: Dict[str, Any], headline_length: int) -> Dict[str, Any]:
    """Replicate-averaged curves and indices at the headline length, per rung."""
    out: Dict[str, Any] = {"length": int(headline_length), "human_reference": HUMAN_REFERENCE["serial_position"],
                           "rungs": {}}
    for rung in grid["rungs"]:
        reps = [c["rungs"][rung] for c in grid["cells"].values() if c["length"] == headline_length]
        if not reps:
            continue
        entry: Dict[str, Any] = {"n_replicates": len(reps)}
        for key in ("cued_hit", "cued_margin", "rollout_hit"):
            curves = [r[key] for r in reps]
            entry[key] = _nanmean_curves(curves)
            entry[key + "_sd"] = _nanstd_curves(curves)
        for key in ("bands_hit", "bands_margin", "bands_rollout"):
            entry[key] = {k: _nanmean([r[key][k] for r in reps]) for k in reps[0][key]}
            # Uncertainty over replicates, beside every band index. The n = 3
            # run could not support a direction on any of these; the interval
            # is what says so.
            entry[key + "_ci"] = {k: _ci([r[key][k] for r in reps]) for k in reps[0][key]}
        out["rungs"][rung] = entry
    return out


def derive_list_length(grid: Dict[str, Any]) -> Dict[str, Any]:
    """Early / middle / late band means against list length, per rung, with the
    slopes and the dissociation the human data show."""
    out: Dict[str, Any] = {"lengths": grid["lengths"], "human_reference": HUMAN_REFERENCE["list_length"],
                           "rungs": {}}
    for rung in grid["rungs"]:
        entry: Dict[str, Any] = {}
        for readout, key in (("hit", "bands_hit"), ("margin", "bands_margin")):
            bands = {b: [] for b in ("early", "middle", "late")}
            for L in grid["lengths"]:
                reps = [c["rungs"][rung][key] for c in grid["cells"].values() if c["length"] == L]
                for b in bands:
                    bands[b].append(float(np.nanmean([r[b] for r in reps])) if reps else float("nan"))
            slopes = {b: _slope(grid["lengths"], v) for b, v in bands.items()}
            # The dissociation per replicate, so it carries an interval too:
            # a slope taken through the replicate MEAN hides how often the
            # sign flips underneath it.
            per_rep = []
            for cat, si in sorted({(c["category"], c["seed_index"])
                                   for c in grid["cells"].values()}):
                by_L = {c["length"]: c["rungs"][rung][key] for c in grid["cells"].values()
                        if c["category"] == cat and c["seed_index"] == si}
                Ls = sorted(by_L)
                per_rep.append(_slope(Ls, [by_L[L]["late"] for L in Ls])
                               - _slope(Ls, [by_L[L]["early"] for L in Ls]))
            entry[readout] = {"bands": bands, "slopes": slopes,
                              "dissociation": slopes["late"] - slopes["early"],
                              "dissociation_ci": _ci(per_rep)}
        out["rungs"][rung] = entry
    return out


def derive_presentation_rate(grid: Dict[str, Any], headline_length: int) -> Dict[str, Any]:
    """Band means against log2(passes) over the fixed ladder at the headline
    length. The criterion rung is excluded: its exposure differs per cell."""
    ladder = grid["ladder"]
    x = [float(np.log2(e)) for e in ladder]
    out: Dict[str, Any] = {"length": int(headline_length), "passes": ladder, "log2_passes": x,
                           "human_reference": HUMAN_REFERENCE["presentation_rate"]}
    for readout, key in (("hit", "bands_hit"), ("margin", "bands_margin")):
        bands = {b: [] for b in ("early", "middle", "late")}
        for ep in ladder:
            reps = [c["rungs"][str(ep)][key] for c in grid["cells"].values()
                    if c["length"] == headline_length]
            for b in bands:
                bands[b].append(_nanmean([r[b] for r in reps]) if reps else float("nan"))
        slopes = {b: _slope(x, v) for b, v in bands.items()}
        per_rep = []
        for cat, si in sorted({(c["category"], c["seed_index"]) for c in grid["cells"].values()}):
            cell = next((c for c in grid["cells"].values()
                         if c["category"] == cat and c["seed_index"] == si
                         and c["length"] == headline_length), None)
            if cell is None:
                continue
            e = [cell["rungs"][str(ep)][key]["early"] for ep in ladder]
            la = [cell["rungs"][str(ep)][key]["late"] for ep in ladder]
            per_rep.append(_slope(x, e) - _slope(x, la))
        out[readout] = {"bands": bands, "slopes": slopes,
                        "dissociation": slopes["early"] - slopes["late"],
                        "dissociation_ci": _ci(per_rep)}
    return out


# ---------------------------------------------------------------------------
# phenomenon 4: prior-list intrusion recency
# ---------------------------------------------------------------------------

PLI_REGIMES = ("criterion", "one_pass")


def run_pli_recency(
    model_class: Type,
    model_kwargs: Dict[str, Any],
    vocab: Dict[str, str],
    policy: Dict[str, Any],
    n_tasks: int = 6,
    seq_len: int = 5,
    between_cosines: Sequence[Optional[float]] = (0.0, 0.5, 0.8),
    regimes: Sequence[str] = PLI_REGIMES,
    embedding_dim: int = 100,
    category_variance: float = 0.2,
    seed: int = 42,
    min_pli: int = 5,
) -> Dict[str, Any]:
    """A chain of lists, one category each, trained in order on ONE model with
    no reset; after each list is trained, that list is probed and every error
    and every runner-up is attributed to a source list by lag.

    Two exposure regimes, both reported. ``criterion`` trains each list to the
    suite's criterion on its own cued recall, which is how every other section
    trains -- but it also means the just-trained list is at ceiling by
    construction, so an arm can only show prior-list intrusions there through
    its runner-ups. ``one_pass`` studies each list exactly once, which is the
    human paradigm (a list is seen once and recalled) and is where a slower
    arm's current-list errors live.

    ``between_cosines`` is the between-list prototype cosine (``None`` is the
    quasi-orthogonal legacy geometry). Under an exact value every prior list is
    equidistant from the current one, so the stimulus predicts a FLAT lag
    profile and any gradient is the memory's. The rung with the most
    prior-list intrusions is the headline; a rung with none has not posed the
    question and is reported as such.
    """
    rungs: Dict[str, Any] = {}
    for rho_b in between_cosines:
      for regime in regimes:
        if regime not in PLI_REGIMES:
            raise ValueError(f"regime must be one of {PLI_REGIMES}, got {regime!r}")
        key = ("none" if rho_b is None else f"{float(rho_b):.2f}") + f"/{regime}"
        material = build_chain_material(vocab, n_tasks, seq_len, embedding_dim=embedding_dim,
                                        category_variance=category_variance,
                                        between_category_cosine=rho_b, seed=seed)
        seqs = material["sequences"]
        names = list(seqs.keys())
        enc = material["encoder"]
        task_index = {c: i for i, c in enumerate(names)}
        task_of = {w: task_index[c] for w, c in material["vocab"].items()}
        T = len(names)
        budget = int(policy["epochs"]) if policy["mode"] == "fixed" else int(policy["max_epochs"])
        model = _make_model(model_class, enc, model_kwargs, budget)

        def _score(i: int) -> float:
            return _nanmean(cued_probe(model, seqs[names[i]], enc)["hit"][1:])

        R = np.full((T, T), np.nan)
        exposures: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        for j in range(T):
            X = material["embeddings"][names[j]]
            if regime == "one_pass":
                _fit(model, X, 1)
                used, reached = 1, _score(j) >= float(policy["criterion"])
            elif policy["mode"] == "fixed":
                _fit(model, X, policy["epochs"])
                used, reached = int(policy["epochs"]), _score(j) >= policy["criterion"]
            else:
                used, reached = 0, False
                while used < int(policy["max_epochs"]):
                    _fit(model, X, 1)
                    used += 1
                    if _score(j) >= float(policy["criterion"]):
                        reached = True
                        break
            consolidate_if_supported(model, X)
            exposures.append({"task": names[j], "epochs": used, "reached": bool(reached)})
            for i in range(j + 1):
                R[j, i] = _score(i)
            pr = cued_probe(model, seqs[names[j]], enc)
            for p in range(1, seq_len):
                d, c = pr["decoded"][p], pr["competitor"][p]
                d_task, c_task = task_of[d], task_of[c]
                records.append({
                    "stage": j, "position": p, "target": seqs[names[j]][p],
                    "hit": bool(pr["hit"][p]), "margin": float(pr["margin"][p]),
                    "decoded": d, "decoded_lag": (j - d_task) if d_task != j else 0,
                    "competitor": c, "competitor_lag": (j - c_task) if c_task != j else 0,
                })

        # Final state: every list probed, errors attributed to their source list.
        source = np.zeros((T, T))
        for i in range(T):
            pr = cued_probe(model, seqs[names[i]], enc)
            for p in range(1, seq_len):
                if not pr["hit"][p]:
                    source[i, task_of[pr["decoded"][p]]] += 1

        lags = list(range(1, T))
        opportunities = {k: (seq_len - 1) * (T - k) for k in lags}
        pli_counts = {k: sum(1 for r in records if (not r["hit"]) and r["decoded_lag"] == k) for k in lags}
        within = sum(1 for r in records if (not r["hit"]) and r["decoded_lag"] == 0)
        n_pli = sum(pli_counts.values())
        comp_counts = {k: sum(1 for r in records if r["competitor_lag"] == k) for k in lags}
        n_comp_prior = sum(comp_counts.values())
        pli_rate = {k: pli_counts[k] / opportunities[k] for k in lags}
        comp_rate = {k: comp_counts[k] / opportunities[k] for k in lags}
        rungs[key] = {
            "between_category_cosine": None if rho_b is None else float(rho_b),
            "regime": regime,
            "geometry": material["geometry"],
            "task_labels": names, "retention_matrix": R.tolist(),
            "avg_forgetting": float(np.nanmean([R[j, j] - R[T - 1, j] for j in range(T - 1)]))
            if T > 1 else float("nan"),
            "exposures": exposures,
            "n_current_probes": len(records),
            "n_errors": sum(1 for r in records if not r["hit"]),
            "within_list_errors": within,
            "pli_total": n_pli,
            "pli_rate": n_pli / max(1, len(records)),
            "lags": lags,
            "opportunities": [opportunities[k] for k in lags],
            "pli_counts": [pli_counts[k] for k in lags],
            "pli_share_by_lag": [pli_counts[k] / n_pli if n_pli else float("nan") for k in lags],
            "pli_rate_by_lag": [pli_rate[k] for k in lags],
            "pli_lag1_share": (pli_counts[1] / n_pli) if n_pli else float("nan"),
            "pli_recency_spearman": _spearman(lags, [pli_rate[k] for k in lags]),
            "competitor_counts": [comp_counts[k] for k in lags],
            "competitor_prior_share": n_comp_prior / max(1, len(records)),
            "competitor_rate_by_lag": [comp_rate[k] for k in lags],
            "competitor_lag1_share": (comp_counts[1] / n_comp_prior) if n_comp_prior else float("nan"),
            "competitor_recency_spearman": _spearman(lags, [comp_rate[k] for k in lags]),
            "final_source_matrix": source.tolist(),
            "records": records,
        }
        print(f"  [pli] {key}: errors={rungs[key]['n_errors']}/{len(records)} "
              f"PLIs={n_pli} lag-1 share={rungs[key]['pli_lag1_share']:.2f} "
              f"runner-up prior share={rungs[key]['competitor_prior_share']:.2f} "
              f"forgetting={rungs[key]['avg_forgetting']:.3f}")

    # A lag profile needs intrusions to be a profile: below `min_pli` the
    # accuracy read-out has not posed the question (one intrusion is not a
    # gradient) and the runner-up profile carries the direction instead. The
    # headline rung is the one with the most intrusions when posed, else the
    # one with the most prior-list runner-ups.
    best_pli = max(rungs, key=lambda k: rungs[k]["pli_total"])
    posed = rungs[best_pli]["pli_total"] >= min_pli
    headline = best_pli if posed else max(
        rungs, key=lambda k: (rungs[k]["competitor_prior_share"], rungs[k]["pli_total"]))
    return {"human_reference": HUMAN_REFERENCE["pli_recency"], "rungs": rungs,
            "headline_rung": headline, "question_posed": bool(posed), "min_pli": int(min_pli),
            "direction_source": "intrusions" if posed else "runner-up",
            "regimes": list(regimes), "n_tasks": int(n_tasks), "seq_len": int(seq_len)}


# ---------------------------------------------------------------------------
# phenomenon 5: semantic clustering
# ---------------------------------------------------------------------------

def validate_semantic_pattern(pattern: str) -> Dict[int, int]:
    """Check the layout's separation property and return ``partner[pos]``.

    For every cue position i with target i+1, the items {target, partner(i),
    partner(i+1), partner(i)+1} must be pairwise distinct, and neither partner
    nor the successor-of-partner may be a temporal neighbour of the target.
    Adjacent positions must differ in category.
    """
    L = len(pattern)
    partner: Dict[int, int] = {}
    for p, letter in enumerate(pattern):
        same = [q for q, l in enumerate(pattern) if l == letter and q != p]
        if len(same) != 1:
            raise ValueError(f"letter {letter!r} must occur exactly twice in {pattern!r}")
        partner[p] = same[0]
    for p in range(L - 1):
        if pattern[p] == pattern[p + 1]:
            raise ValueError(f"adjacent positions {p},{p + 1} share a category")
        t = p + 1
        items = {"target": t, "S1": partner[p], "S2": partner[t]}
        if partner[p] + 1 < L:
            items["S3"] = partner[p] + 1
        vals = list(items.values())
        if len(set(vals)) != len(vals):
            raise ValueError(f"cue {p}: {items} are not distinct")
        for name in ("S1", "S2", "S3"):
            if name in items and abs(items[name] - t) == 1 and items[name] != p:
                raise ValueError(f"cue {p}: {name} is a temporal neighbour of the target")
    return partner


def build_semantic_material(
    vocab: Dict[str, str],
    categories: Sequence[str],
    embedding_dim: int = 100,
    category_variance: float = 0.2,
    seed: int = 42,
    pattern: str = SEMANTIC_PATTERN,
    n_eli_same: int = 2,
    n_eli_other: int = 2,
) -> Dict[str, Any]:
    """One list laid out on ``pattern`` from ``categories`` (one per letter),
    plus extra-list words: ``n_eli_same`` further words of every list category
    and ``n_eli_other`` words of every non-list category, so that intrusions
    from outside the list are possible and classifiable."""
    letters = sorted(set(pattern))
    if len(categories) != len(letters):
        raise ValueError(f"pattern {pattern!r} needs {len(letters)} categories, got {len(categories)}")
    partner = validate_semantic_pattern(pattern)
    by_cat: Dict[str, List[str]] = {}
    for w, c in vocab.items():
        by_cat.setdefault(c, []).append(w)
    cat_of_letter = dict(zip(letters, categories))
    seen: Dict[str, int] = {}
    words: List[str] = []
    for letter in pattern:
        cat = cat_of_letter[letter]
        k = seen.get(cat, 0)
        words.append(by_cat[cat][k])
        seen[cat] = k + 1
    eli_same = {w for cat in categories for w in by_cat[cat][2:2 + n_eli_same]}
    others = [c for c in sorted(by_cat) if c not in categories]
    eli_other = {w for cat in others for w in by_cat[cat][:n_eli_other]}
    ordered = list(words) + sorted(eli_same) + sorted(eli_other)
    enc_vocab = {w: vocab[w] for w in dict.fromkeys(ordered)}
    enc = SymbolicEncoder(enc_vocab, embedding_dim=embedding_dim,
                          category_variance=category_variance, seed=seed)
    E = _unit_rows(enc.embeddings)
    G = E @ E.T
    idx = {w: enc.word_to_idx[w] for w in words}
    within = [G[idx[words[p]], idx[words[partner[p]]]] for p in range(len(words))]
    nn_partner = 0
    for p in range(len(words)):
        row = G[idx[words[p]]].copy()
        row[idx[words[p]]] = -np.inf
        nn_partner += int(enc.idx_to_word[int(np.argmax(row))] == words[partner[p]])
    return {
        "words": words, "partner": partner, "pattern": pattern,
        "category_of_pos": [cat_of_letter[l] for l in pattern],
        "categories": list(categories), "other_categories": others,
        "eli_same": eli_same, "eli_other": eli_other, "encoder": enc,
        "n_candidates": len(enc.idx_to_word),
        "geometry": {"within_partner_cos_measured": float(np.mean(within)),
                     "within_cos_law": 1.0 / (1.0 + embedding_dim * category_variance ** 2),
                     "nn_is_partner_fraction": nn_partner / len(words)},
    }


def classify_semantic(cue_pos: int, word: str, mat: Dict[str, Any]) -> str:
    """Class of a decoded / competitor word for the probe at ``cue_pos``."""
    words, partner = mat["words"], mat["partner"]
    L = len(words)
    t = cue_pos + 1
    if word == words[partner[cue_pos]]:
        return "cue_partner"
    if word == words[partner[t]]:
        return "target_partner"
    if partner[cue_pos] + 1 < L and word == words[partner[cue_pos] + 1]:
        return "cue_partner_successor"
    if word == words[cue_pos]:
        return "cue_repeat"
    if word in words:
        pos = words.index(word)
        return "temporal_neighbour" if abs(pos - t) == 1 else "other_in_list"
    cat = mat["encoder_vocab"][word]
    if cat == mat["category_of_pos"][cue_pos]:
        return "eli_cue_category"
    if cat == mat["category_of_pos"][t]:
        return "eli_target_category"
    if cat in mat["categories"]:
        return "eli_list_category"
    return "eli_other_category"


def _percentile_rank(sims_row: np.ndarray, word_idx: int, exclude: Sequence[int]) -> float:
    """Where the decoded word's similarity sits among the candidates: 1 = most
    similar candidate, 0 = least. NaN if the word is excluded."""
    if word_idx in exclude:
        return float("nan")
    cand = np.array([k for k in range(len(sims_row)) if k not in exclude])
    s = sims_row[cand]
    v = sims_row[word_idx]
    below = float(np.sum(s < v))
    ties = float(np.sum(s == v)) - 1.0
    return (below + 0.5 * ties) / max(1.0, len(cand) - 1.0)


def run_semantic_clustering(
    model_class: Type,
    model_kwargs: Dict[str, Any],
    vocab: Dict[str, str],
    policy: Dict[str, Any],
    category_sets: Sequence[Sequence[str]] = (("fruit", "animal", "vehicle", "color"),
                                              ("number", "tool", "fruit", "animal"),
                                              ("vehicle", "color", "number", "tool")),
    variances: Sequence[float] = (0.1, 0.2, 1.0, 3.0),
    ladder: Sequence[int] = (1, 2, 4),
    embedding_dim: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Errors and runners-up classified by their relation to the cue and the
    target, on lists whose semantic partners are separated in time, across a
    within-category cosine sweep whose last rung has no semantic structure."""
    rungs: Dict[str, Any] = {}
    for sigma in variances:
        skey = f"{float(sigma):.2f}"
        per_list: List[Dict[str, Any]] = []
        for li, cats in enumerate(category_sets):
            mat = build_semantic_material(vocab, cats, embedding_dim=embedding_dim,
                                          category_variance=sigma, seed=seed + li)
            enc = mat["encoder"]
            mat["encoder_vocab"] = {w: vocab[w] for w in enc.idx_to_word}
            words = mat["words"]
            X = enc.encode(words)
            idx_of = {w: k for k, w in enumerate(enc.idx_to_word)}

            def _build(ep, _e=enc):
                return _make_model(model_class, _e, model_kwargs, ep)

            def _fitx(m, ep, _X=X):
                _fit(m, _X, ep)

            def _score(m, _w=words, _e=enc):
                return float(np.nanmean(cued_probe(m, _w, _e)["hit"][1:]))

            model_c, rec = train_at(policy, _build, _fitx, _score)
            models = {str(int(ep)): None for ep in ladder}
            for ep in ladder:
                m = _build(ep)
                _fitx(m, ep)
                models[str(int(ep))] = m
            models["criterion"] = model_c
            reads: Dict[str, Any] = {}
            for rung, m in models.items():
                pr = cued_probe(m, words, enc)
                err_cls: List[str] = []
                comp_cls: List[str] = []
                err_rank_cue: List[float] = []
                err_rank_target: List[float] = []
                comp_rank_cue: List[float] = []
                for p in range(1, len(words)):
                    i = p - 1
                    excl = [idx_of[words[i]], idx_of[words[p]]]
                    c = pr["competitor"][p]
                    comp_cls.append(classify_semantic(i, c, mat))
                    cue_sims = _unit_rows(enc.embeddings) @ _unit_rows(enc.embeddings)[idx_of[words[i]]]
                    comp_rank_cue.append(_percentile_rank(cue_sims, idx_of[c], excl))
                    if not pr["hit"][p]:
                        d = pr["decoded"][p]
                        err_cls.append(classify_semantic(i, d, mat))
                        tgt_sims = _unit_rows(enc.embeddings) @ _unit_rows(enc.embeddings)[idx_of[words[p]]]
                        err_rank_cue.append(_percentile_rank(cue_sims, idx_of[d], excl))
                        err_rank_target.append(_percentile_rank(tgt_sims, idx_of[d], excl))
                reads[rung] = {
                    "n_probes": len(words) - 1, "n_errors": len(err_cls),
                    "error_classes": err_cls, "competitor_classes": comp_cls,
                    "error_rank_cue": err_rank_cue, "error_rank_target": err_rank_target,
                    "competitor_rank_cue": comp_rank_cue,
                    "accuracy": float(np.nanmean(pr["hit"][1:])),
                    "margin": float(np.nanmean(pr["margin"][1:])),
                }
            per_list.append({"categories": list(cats), "words": words,
                             "geometry": mat["geometry"], "n_candidates": mat["n_candidates"],
                             "exposure": rec, "rungs": reads})

        # pool over lists per rung
        pooled: Dict[str, Any] = {}
        n_cand = per_list[0]["n_candidates"]
        chance_specific = 1.0 / (n_cand - 2)
        for rung in list(per_list[0]["rungs"].keys()):
            errs = [c for pl in per_list for c in pl["rungs"][rung]["error_classes"]]
            comps = [c for pl in per_list for c in pl["rungs"][rung]["competitor_classes"]]
            n_probes = sum(pl["rungs"][rung]["n_probes"] for pl in per_list)
            rk_cue = [v for pl in per_list for v in pl["rungs"][rung]["error_rank_cue"]]
            rk_tgt = [v for pl in per_list for v in pl["rungs"][rung]["error_rank_target"]]
            crk = [v for pl in per_list for v in pl["rungs"][rung]["competitor_rank_cue"]]
            pooled[rung] = {
                "n_probes": n_probes, "n_errors": len(errs),
                "accuracy": float(np.mean([pl["rungs"][rung]["accuracy"] for pl in per_list])),
                "margin": float(np.mean([pl["rungs"][rung]["margin"] for pl in per_list])),
                "error_share": {c: (errs.count(c) / len(errs) if errs else float("nan"))
                                for c in SEMANTIC_CLASSES},
                "error_rate": {c: errs.count(c) / max(1, n_probes) for c in SEMANTIC_CLASSES},
                "competitor_share": {c: comps.count(c) / max(1, n_probes) for c in SEMANTIC_CLASSES},
                "semantic_factor_cue": _nanmean(rk_cue),
                "semantic_factor_target": _nanmean(rk_tgt),
                "competitor_semantic_factor_cue": _nanmean(crk),
            }
        rungs[skey] = {
            "category_variance": float(sigma),
            "within_cos_law": float(np.mean([pl["geometry"]["within_cos_law"] for pl in per_list])),
            "within_partner_cos_measured": float(np.mean(
                [pl["geometry"]["within_partner_cos_measured"] for pl in per_list])),
            "nn_is_partner_fraction": float(np.mean(
                [pl["geometry"]["nn_is_partner_fraction"] for pl in per_list])),
            "n_candidates": n_cand, "chance_specific": chance_specific,
            "exposures": [pl["exposure"] for pl in per_list],
            "pooled": pooled, "lists": per_list,
        }
        r1 = pooled[str(int(ladder[0]))]
        print(f"  [semantic] sigma={skey} cos={rungs[skey]['within_partner_cos_measured']:.3f} "
              f"errors@{ladder[0]}={r1['n_errors']}/{r1['n_probes']} "
              f"S1={r1['error_share']['cue_partner']:.2f} S2={r1['error_share']['target_partner']:.2f} "
              f"S3={r1['error_share']['cue_partner_successor']:.2f} "
              f"factor_cue={r1['semantic_factor_cue']:.2f}")

    # calibration: at the least-structured rung the semantic read-outs must be at chance
    keys = sorted(rungs, key=lambda k: rungs[k]["within_cos_law"])
    nostruct, house = keys[0], (f"{0.2:.2f}" if f"{0.2:.2f}" in rungs else keys[-1])
    first = str(int(ladder[0]))
    ns = rungs[nostruct]["pooled"][first]
    calibrated = (np.isnan(ns["semantic_factor_cue"])
                  or abs(ns["semantic_factor_cue"] - 0.5) <= 0.15)
    return {"human_reference": HUMAN_REFERENCE["semantic_clustering"], "rungs": rungs,
            "rung_order": keys, "no_structure_rung": nostruct, "house_rung": house,
            "first_rung": first, "classes": list(SEMANTIC_CLASSES),
            "calibrated": bool(calibrated), "pattern": SEMANTIC_PATTERN}


# ---------------------------------------------------------------------------
# summary + plots + entry point
# ---------------------------------------------------------------------------

def _emit_ci(s: Dict[str, Any], base: str, ci: Dict[str, float]) -> None:
    """Write a band index as mean + interval, never as a bare number.

    Reporting the mean alone is what let the first (n = 3) run read a direction
    off a quantity whose replicates flipped sign; `<base>_spans_zero` is the
    gate that the direction flags below consult.
    """
    s[base] = ci["mean"]
    s[f"{base}_sd"] = ci["sd"]
    s[f"{base}_se"] = ci["se"]
    s[f"{base}_ci_lo"] = ci["ci_lo"]
    s[f"{base}_ci_hi"] = ci["ci_hi"]
    s[f"{base}_n"] = ci["n"]
    s[f"{base}_spans_zero"] = ci["spans_zero"]


def _summarise(sp, ll, pr, pli, sem, first_rung: str) -> Dict[str, Any]:
    s: Dict[str, Any] = {}
    r1, rc = sp["rungs"].get(first_rung), sp["rungs"].get("criterion")
    if r1:
        _emit_ci(s, "cog_spc_primacy_acc_first", r1["bands_hit_ci"]["primacy_index"])
        _emit_ci(s, "cog_spc_recency_acc_first", r1["bands_hit_ci"]["recency_index"])
        _emit_ci(s, "cog_spc_rollout_recency_first", r1["bands_rollout_ci"]["recency_index"])
    if rc:
        _emit_ci(s, "cog_spc_primacy_margin_criterion", rc["bands_margin_ci"]["primacy_index"])
        _emit_ci(s, "cog_spc_recency_margin_criterion", rc["bands_margin_ci"]["recency_index"])
        s["cog_spc_accuracy_criterion"] = _nanmean(rc["cued_hit"][1:])
    if first_rung in ll["rungs"]:
        s["cog_ll_early_slope_acc_first"] = ll["rungs"][first_rung]["hit"]["slopes"]["early"]
        s["cog_ll_late_slope_acc_first"] = ll["rungs"][first_rung]["hit"]["slopes"]["late"]
        _emit_ci(s, "cog_ll_dissociation_acc_first",
                 ll["rungs"][first_rung]["hit"]["dissociation_ci"])
    if "criterion" in ll["rungs"]:
        s["cog_ll_early_slope_margin_criterion"] = ll["rungs"]["criterion"]["margin"]["slopes"]["early"]
        s["cog_ll_late_slope_margin_criterion"] = ll["rungs"]["criterion"]["margin"]["slopes"]["late"]
        _emit_ci(s, "cog_ll_dissociation_margin_criterion",
                 ll["rungs"]["criterion"]["margin"]["dissociation_ci"])
    s["cog_pr_early_slope_acc"] = pr["hit"]["slopes"]["early"]
    s["cog_pr_late_slope_acc"] = pr["hit"]["slopes"]["late"]
    _emit_ci(s, "cog_pr_dissociation_acc", pr["hit"]["dissociation_ci"])
    s["cog_pr_early_slope_margin"] = pr["margin"]["slopes"]["early"]
    s["cog_pr_late_slope_margin"] = pr["margin"]["slopes"]["late"]
    _emit_ci(s, "cog_pr_dissociation_margin", pr["margin"]["dissociation_ci"])
    h = pli["rungs"][pli["headline_rung"]]
    s["cog_pli_headline_rung"] = pli["headline_rung"]
    s["cog_pli_headline_rho_b"] = h["between_category_cosine"]
    s["cog_pli_headline_regime"] = h["regime"]
    s["cog_pli_question_posed"] = pli["question_posed"]
    s["cog_pli_total"] = h["pli_total"]
    s["cog_pli_rate"] = h["pli_rate"]
    s["cog_pli_lag1_share"] = h["pli_lag1_share"]
    s["cog_pli_recency_spearman"] = h["pli_recency_spearman"]
    s["cog_pli_avg_forgetting"] = h["avg_forgetting"]
    s["cog_comp_prior_share"] = h["competitor_prior_share"]
    s["cog_comp_lag1_share"] = h["competitor_lag1_share"]
    s["cog_comp_recency_spearman"] = h["competitor_recency_spearman"]
    hp = sem["rungs"][sem["house_rung"]]["pooled"][sem["first_rung"]]
    ns = sem["rungs"][sem["no_structure_rung"]]["pooled"][sem["first_rung"]]
    hc = sem["rungs"][sem["house_rung"]]["pooled"]["criterion"]
    s["cog_sem_house_sigma"] = sem["rungs"][sem["house_rung"]]["category_variance"]
    s["cog_sem_chance_specific"] = sem["rungs"][sem["house_rung"]]["chance_specific"]
    s["cog_sem_n_errors_house_first"] = hp["n_errors"]
    s["cog_sem_s1_cue_partner_share"] = hp["error_share"]["cue_partner"]
    s["cog_sem_s2_target_partner_share"] = hp["error_share"]["target_partner"]
    s["cog_sem_s3_cue_partner_successor_share"] = hp["error_share"]["cue_partner_successor"]
    s["cog_sem_factor_cue_house"] = hp["semantic_factor_cue"]
    s["cog_sem_factor_target_house"] = hp["semantic_factor_target"]
    s["cog_sem_factor_cue_nostructure"] = ns["semantic_factor_cue"]
    s["cog_sem_comp_s1_share_criterion"] = hc["competitor_share"]["cue_partner"]
    s["cog_sem_comp_s2_share_criterion"] = hc["competitor_share"]["target_partner"]
    s["cog_sem_comp_factor_cue_criterion"] = hc["competitor_semantic_factor_cue"]
    s["cog_sem_calibrated"] = sem["calibrated"]

    # Direction against the human reference: a fact per signature, never summed.
    # None means UNDETERMINED, not "no effect" -- either the read-out is missing
    # or its replicate interval spans zero, in which case the run has not
    # measured a direction and must not report one.
    def _sgn(v, want):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else bool((v > 0) == want)

    def _dir(base, want=True):
        if s.get(f"{base}_spans_zero", True):
            return None
        return _sgn(s.get(base), want)

    s["cog_dir_spc_primacy"] = _dir("cog_spc_primacy_margin_criterion")
    s["cog_dir_spc_recency"] = _dir("cog_spc_recency_margin_criterion")
    s["cog_dir_ll_dissociation"] = _dir("cog_ll_dissociation_margin_criterion")
    s["cog_dir_pr_dissociation"] = _dir("cog_pr_dissociation_margin")
    s["cog_pli_direction_source"] = pli["direction_source"]
    s["cog_dir_pli_recency"] = _sgn(
        -(s["cog_pli_recency_spearman"] if pli["question_posed"]
          else s["cog_comp_recency_spearman"]), True)
    s["cog_dir_sem_organisation"] = (
        None if np.isnan(hp["semantic_factor_cue"]) else bool(hp["semantic_factor_cue"] > 0.5))
    return s


#: Qualitative reference shapes, drawn beside every model figure. These are
#: SCHEMATICS after the cited figures -- the ordering and direction of the
#: human effects, on an unlabelled recall axis -- not digitised data, and the
#: panels say so. Their job is to make "what the expected curve looks like"
#: visible on the same page as the model's curve.
REFERENCE_SCHEMATIC: Dict[str, Dict[str, Any]] = {
    "serial_position": {
        "after": "Kahana 2020 Fig. 2b (serial), Fig. 2a (free)",
        "x": [1, 2, 3, 4, 5, 6, 7],
        "immediate serial recall (this protocol's reference)":
            [0.95, 0.86, 0.74, 0.62, 0.54, 0.51, 0.58],
        "immediate free recall (for contrast)":
            [0.62, 0.46, 0.38, 0.35, 0.40, 0.62, 0.86],
    },
    "list_length": {
        "after": "Kahana 2020 Fig. 2a (Murdock 1962)",
        "x": [7, 13, 19],
        "early": [0.88, 0.74, 0.62], "middle": [0.60, 0.42, 0.30], "late": [0.60, 0.59, 0.60],
    },
    "presentation_rate": {
        "after": "Kahana 2020 sec 4.1 (Murdock 1962)",
        "x": [0.0, 1.0],   # log2 of study time per item, 1 s -> 2 s
        "early": [0.55, 0.76], "middle": [0.35, 0.56], "late": [0.70, 0.70],
    },
    "pli_recency": {
        "after": "Kahana 2020 Fig. 5a (Zaromb et al. 2006)",
        "x": [1, 2, 3, 4, 5],
        "share": [0.46, 0.24, 0.13, 0.09, 0.08],
    },
    "semantic_clustering": {
        "after": "Kahana 2020 Fig. 4b (Howard & Kahana 2002)",
        "x": [0.0, 0.17, 0.33, 0.5],   # semantic similarity of the pair
        "organisation (transition bias)": [0.5, 0.58, 0.68, 0.78],
    },
}


def _reference_panel(ax, kind: str, chance: Optional[float] = None) -> None:
    """Draw the schematic reference for ``kind`` on ``ax`` in a deliberately
    different style (grey, dashed, no y tick labels) so it cannot be mistaken
    for data on the model's scale."""
    ref = REFERENCE_SCHEMATIC[kind]
    x = ref["x"]
    styles = {"early": "tab:blue", "middle": "tab:grey", "late": "tab:red"}
    fallback = ["0.25", "tab:olive", "tab:cyan"]
    series = [(k, v) for k, v in ref.items() if k not in ("after", "x")]
    for n, (k, v) in enumerate(series):
        col = styles.get(k, fallback[n % len(fallback)])
        ax.plot(x, v, "--", marker="o", ms=4, color=col, alpha=0.85, label=k)
    if kind == "pli_recency":
        ax.bar(x, ref["share"], color="0.6", alpha=0.6, width=0.7)
        ax.set_xlabel("list lag")
        ax.set_ylabel("share of PLIs (schematic)")
    elif kind == "serial_position":
        ax.set_xlabel("serial position")
        ax.set_ylabel("recall probability (schematic)")
    elif kind == "list_length":
        ax.set_xlabel("list length")
        ax.set_ylabel("recall probability (schematic)")
    elif kind == "presentation_rate":
        ax.set_xlabel("log2(study time per item)")
        ax.set_ylabel("recall probability (schematic)")
    elif kind == "semantic_clustering":
        ax.axhline(0.5, color="k", ls=":", lw=0.8, label="no organisation")
        ax.set_xlabel("semantic similarity of the pair")
        ax.set_ylabel("transition bias (schematic)")
    if chance is not None:
        ax.axhline(chance, color="k", ls=":", lw=0.8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_facecolor("0.96")
    ax.set_title(f"HUMAN REFERENCE (schematic)\nafter {ref['after']}", fontsize=8, color="0.3")
    ax.legend(fontsize=6.5, loc="best")
    ax.grid(alpha=0.25)


def _plot_serial_position(sp: Dict[str, Any], path: str, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rungs = list(sp["rungs"].keys())
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    _reference_panel(axes[3], "serial_position")
    panels = (("cued_hit", "cued accuracy (clean single probe)", (-0.05, 1.05)),
              ("cued_margin", "cued margin = cos(target) - cos(runner-up)", None),
              ("rollout_hit", "rollout accuracy (cannot show recency: see note)", (-0.05, 1.05)))
    cmap = plt.get_cmap("viridis")
    for ax, (key, title, ylim) in zip(axes[:3], panels):
        for k, rung in enumerate(rungs):
            y = np.asarray(sp["rungs"][rung][key], dtype=float)
            x = np.arange(len(y))
            col = "k" if rung == "criterion" else cmap(k / max(1, len(rungs) - 1))
            ax.plot(x[1:], y[1:], "-o", ms=4, color=col,
                    label=("criterion" if rung == "criterion" else f"{rung} pass(es)"))
        ax.set_xlabel("serial position (0 = cue)")
        ax.set_title(title, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("recall")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Serial position, L={sp['length']} ({model_name}); human serial recall: "
                 "strong primacy, modest recency", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_bands(entry: Dict[str, Any], x: Sequence[float], xlabel: str, title: str,
                path: str, reference: str = "presentation_rate") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _reference_panel(axes[2], reference)
    for ax, readout in zip(axes[:2], ("hit", "margin")):
        b = entry[readout]["bands"]
        for band, col in (("early", "tab:blue"), ("middle", "tab:grey"), ("late", "tab:red")):
            ax.plot(x, b[band], "-o", color=col, label=f"{band} (slope {entry[readout]['slopes'][band]:+.3f})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cued accuracy" if readout == "hit" else "cued margin")
        ax.set_title(f"{readout}: dissociation {entry[readout]['dissociation']:+.3f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_list_length(ll: Dict[str, Any], first_rung: str, path: str, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    _reference_panel(axes[0, 2], "list_length")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.95, HUMAN_REFERENCE["list_length"]["signature"] + "\n\nRead as: "
                    + HUMAN_REFERENCE["list_length"]["read_as"], va="top", ha="left", fontsize=8,
                    wrap=True, transform=axes[1, 2].transAxes, color="0.3")
    for row, rung in enumerate((first_rung, "criterion")):
        e = ll["rungs"][rung]
        for col, readout in enumerate(("hit", "margin")):
            ax = axes[row, col]
            b = e[readout]["bands"]
            for band, c in (("early", "tab:blue"), ("middle", "tab:grey"), ("late", "tab:red")):
                ax.plot(ll["lengths"], b[band], "-o", color=c,
                        label=f"{band} (slope {e[readout]['slopes'][band]:+.3f})")
            ax.set_xlabel("list length")
            ax.set_ylabel("cued accuracy" if readout == "hit" else "cued margin")
            ax.set_title(f"{'1 pass' if rung == first_rung else 'criterion'}, {readout}: "
                         f"late - early slope {e[readout]['dissociation']:+.3f}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle(f"List length ({model_name}); human: longer lists cost early/middle "
                 "positions, not the last few", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pli(pli: Dict[str, Any], path: str, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(pli["rungs"].keys())
    fig, axes = plt.subplots(len(keys), 3, figsize=(16, 3.6 * len(keys)), squeeze=False)
    _reference_panel(axes[0, 2], "pli_recency")
    for r in range(1, len(keys)):
        axes[r, 2].axis("off")
    if len(keys) > 1:
        axes[1, 2].text(0.02, 0.95, HUMAN_REFERENCE["pli_recency"]["signature"] + "\n\nRead as: "
                        + HUMAN_REFERENCE["pli_recency"]["read_as"]
                        + f"\n\nHeadline rung: {pli['headline_rung']} "
                        f"(direction from {pli['direction_source']}; "
                        f"question posed in accuracy: {pli['question_posed']}, "
                        f"needs >= {pli['min_pli']} intrusions)",
                        va="top", ha="left", fontsize=8, wrap=True,
                        transform=axes[1, 2].transAxes, color="0.3")
    for r, k in enumerate(keys):
        e = pli["rungs"][k]
        lags = e["lags"]
        ax = axes[r, 0]
        ax.bar(lags, e["pli_rate_by_lag"], color="tab:blue", width=0.7,
               label=f"PLI rate per opportunity (n={e['pli_total']})")
        ax.set_xlabel("list lag (1 = the list just before)")
        ax.set_ylabel("intrusions / opportunities")
        ax.set_title(f"{k}: errors {e['n_errors']}/{e['n_current_probes']}, "
                     f"PLIs {e['pli_total']}, lag-1 share "
                     f"{e['pli_lag1_share'] if e['pli_total'] else float('nan'):.2f}, "
                     f"forgetting {e['avg_forgetting']:.2f}", fontsize=8)
        if e["pli_total"] == 0:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "no prior-list intrusions:\nquestion not posed at this rung\n"
                    "(read the runner-up panel)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="tab:grey")
        ax.legend(fontsize=7)
        ax = axes[r, 1]
        ax.bar(lags, e["competitor_rate_by_lag"], color="tab:orange", width=0.7,
               label="runner-up in a prior list, per opportunity")
        ax.set_xlabel("list lag of the runner-up")
        ax.set_ylabel("probes / opportunities")
        ax.set_title(f"pressure above the floor: prior-list runner-up share "
                     f"{e['competitor_prior_share']:.2f}, rho {e['competitor_recency_spearman']:+.2f}",
                     fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(f"Prior-list intrusion recency ({model_name}); human: most PLIs from the "
                 "last two lists, falling with lag; geometric null is flat", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.12 / len(keys)))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_semantic(sem: Dict[str, Any], path: str, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sem["rung_order"]
    x = [sem["rungs"][k]["within_partner_cos_measured"] for k in keys]
    first = sem["first_rung"]
    chance = sem["rungs"][keys[0]]["chance_specific"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    _reference_panel(axes[3], "semantic_clustering")
    cls = (("cue_partner", "tab:green", "S1 cue -> its partner (organisation)"),
           ("target_partner", "tab:red", "S2 target's partner (output confusability)"),
           ("cue_partner_successor", "tab:purple", "S3 successor of cue's partner (input generalisation)"),
           ("temporal_neighbour", "tab:grey", "temporal neighbour"))
    ax = axes[0]
    for c, col, lab in cls:
        ax.plot(x, [sem["rungs"][k]["pooled"][first]["error_share"][c] for k in keys], "-o",
                color=col, label=lab)
    ax.axhline(chance, color="k", ls="--", lw=0.8, label=f"chance for one specific item ({chance:.3f})")
    n_err = [sem["rungs"][k]["pooled"][first]["n_errors"] for k in keys]
    ax.set_title(f"share of ERRORS at {first} pass(es); n errors per rung {n_err}", fontsize=8)
    if sum(n_err) == 0:
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.6, f"no errors at {first} pass(es):\nquestion not posed in accuracy\n"
                "(read the runner-up panel)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="tab:grey")
    ax.set_xlabel("measured within-category cosine (partner pairs)")
    ax.set_ylabel("share of errors")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for c, col, lab in cls:
        ax.plot(x, [sem["rungs"][k]["pooled"]["criterion"]["competitor_share"][c] for k in keys],
                "-o", color=col, label=lab)
    ax.axhline(chance, color="k", ls="--", lw=0.8)
    ax.set_title("share of PROBES whose runner-up is ... (criterion exposure)", fontsize=8)
    ax.set_xlabel("measured within-category cosine (partner pairs)")
    ax.set_ylabel("share of probes")
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(x, [sem["rungs"][k]["pooled"][first]["semantic_factor_cue"] for k in keys], "-o",
            color="tab:green", label=f"errors, cue-relative ({first} pass)")
    ax.plot(x, [sem["rungs"][k]["pooled"][first]["semantic_factor_target"] for k in keys], "-o",
            color="tab:red", label=f"errors, target-relative ({first} pass)")
    ax.plot(x, [sem["rungs"][k]["pooled"]["criterion"]["competitor_semantic_factor_cue"] for k in keys],
            "-s", color="tab:olive", label="runner-up, cue-relative (criterion)")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="0.5 = no semantic organisation")
    ax.set_ylim(0, 1)
    ax.set_title("semantic factor (percentile rank of the produced item's similarity)", fontsize=8)
    ax.set_xlabel("measured within-category cosine (partner pairs)")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    fig.suptitle(f"Semantic clustering ({model_name}); layout {sem['pattern']}; "
                 f"calibrated at no structure: {sem['calibrated']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_cognitive_phenomena(
    model_class: Type,
    model_kwargs: Optional[Dict[str, Any]] = None,
    vocab: Optional[Dict[str, str]] = None,
    exposure: Optional[Dict[str, Any]] = None,
    lengths: Sequence[int] = (5, 7, 10),
    headline_length: int = 7,
    categories: Sequence[str] = ("fruit", "animal", "number"),
    ladder: Sequence[int] = (1, 2, 4, 8),
    n_seeds: int = 5,
    pli_n_tasks: int = 6,
    pli_seq_len: int = 5,
    pli_between_cosines: Sequence[Optional[float]] = (0.0, 0.5, 0.8),
    pli_regimes: Sequence[str] = PLI_REGIMES,
    semantic_variances: Sequence[float] = (0.1, 0.2, 1.0, 3.0),
    semantic_ladder: Sequence[int] = (1, 2, 4),
    embedding_dim: int = 100,
    category_variance: float = 0.2,
    feedback_mode: str = "l2",
    seed: int = 42,
    run_dir: Optional[str] = None,
    metrics_filename: str = "cognitive_phenomena_metrics.json",
    run_name: Optional[str] = None,
    phenomena: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run the five L0 phenomena and write them to their own file.

    Returns ``{"metrics": {}, "series": {"cognitive_phenomena": ...}}``: the
    section contributes NOTHING to ``results["metrics"]`` by design (it is
    descriptive and must not enter the capacity scorecard); the summary
    scalars live under ``series`` and in ``metrics_filename``.
    """
    model_kwargs = dict(model_kwargs or {})
    if vocab is None:
        from memval.benchmarks.symbolic_pipeline import load_vocab
        vocab = load_vocab()
    policy = exposure or {"mode": "criterion", "epochs": None, "criterion": 0.95,
                          "max_epochs": 512, "fallback_epochs": 300}
    selected = list(phenomena) if phenomena else list(PHENOMENA)
    name = run_name or model_class.__name__
    first_rung = str(int(ladder[0]))
    out: Dict[str, Any] = {
        "metadata": {"section": SECTION, "model": name, "seed": int(seed),
                     "exposure_mode": policy["mode"], "criterion": policy["criterion"],
                     "probe": "clean_single", "feedback_mode": feedback_mode,
                     "replication": "material (categories / lists), not cue noise",
                     "phenomena": selected},
        "phenomena": {}, "summary": {},
    }

    grid = None
    if any(p in selected for p in ("serial_position", "list_length", "presentation_rate")):
        print("  [cognitive] position x length x exposure grid ...")
        grid = run_position_grid(model_class, model_kwargs, vocab, policy, lengths=lengths,
                                 categories=categories, ladder=ladder, n_seeds=n_seeds,
                                 embedding_dim=embedding_dim, category_variance=category_variance,
                                 seed=seed, feedback_mode=feedback_mode)
        out["phenomena"]["grid"] = {k: v for k, v in grid.items() if k != "cells"}
        out["phenomena"]["grid"]["cells"] = {
            k: {kk: vv for kk, vv in c.items()} for k, c in grid["cells"].items()}
        sp = derive_serial_position(grid, headline_length)
        ll = derive_list_length(grid)
        pr = derive_presentation_rate(grid, headline_length)
        out["phenomena"]["serial_position"] = sp
        out["phenomena"]["list_length"] = ll
        out["phenomena"]["presentation_rate"] = pr

    pli = None
    if "pli_recency" in selected:
        print("  [cognitive] prior-list intrusion recency ...")
        pli = run_pli_recency(model_class, model_kwargs, vocab, policy, n_tasks=pli_n_tasks,
                              seq_len=pli_seq_len, between_cosines=pli_between_cosines,
                              regimes=pli_regimes, embedding_dim=embedding_dim,
                              category_variance=category_variance, seed=seed)
        out["phenomena"]["pli_recency"] = pli

    sem = None
    if "semantic_clustering" in selected:
        print("  [cognitive] semantic clustering ...")
        sem = run_semantic_clustering(model_class, model_kwargs, vocab, policy,
                                      variances=semantic_variances, ladder=semantic_ladder,
                                      embedding_dim=embedding_dim, seed=seed)
        out["phenomena"]["semantic_clustering"] = sem

    if grid is not None and pli is not None and sem is not None:
        out["summary"] = _summarise(sp, ll, pr, pli, sem, first_rung)
    out["human_reference"] = HUMAN_REFERENCE

    if run_dir is not None:
        plots = os.path.join(run_dir, "plots")
        os.makedirs(plots, exist_ok=True)
        if grid is not None:
            _plot_serial_position(sp, os.path.join(plots, "cognitive_serial_position.png"), name)
            _plot_list_length(ll, first_rung, os.path.join(plots, "cognitive_list_length.png"), name)
            _plot_bands(pr, pr["log2_passes"], "log2(passes over the list)",
                        f"Presentation rate, L={headline_length} ({name}); human: slower rate "
                        "helps early/middle items, not the last few",
                        os.path.join(plots, "cognitive_presentation_rate.png"))
        if pli is not None:
            _plot_pli(pli, os.path.join(plots, "cognitive_pli_recency.png"), name)
        if sem is not None:
            _plot_semantic(sem, os.path.join(plots, "cognitive_semantic_clustering.png"), name)
        with open(os.path.join(run_dir, metrics_filename), "w") as f:
            json.dump(out, f, indent=2, default=_json_default)
    series = {"summary": out["summary"], "human_reference": HUMAN_REFERENCE,
              "metadata": out["metadata"]}
    return {"metrics": {}, "series": {SECTION: series}}


def replot_cognitive_phenomena(json_path: str, plots_dir: Optional[str] = None,
                               model_name: Optional[str] = None) -> List[str]:
    """Regenerate the five figures from a saved ``cognitive_phenomena_metrics.json``."""
    with open(json_path) as f:
        payload = json.load(f)
    ph = payload["phenomena"]
    name = model_name or payload["metadata"]["model"]
    plots_dir = plots_dir or os.path.join(os.path.dirname(os.path.abspath(json_path)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out: List[str] = []
    if "serial_position" in ph:
        sp, ll, pr = ph["serial_position"], ph["list_length"], ph["presentation_rate"]
        first_rung = str(int(ph["grid"]["ladder"][0]))
        p = os.path.join(plots_dir, "cognitive_serial_position.png")
        _plot_serial_position(sp, p, name); out.append(p)
        p = os.path.join(plots_dir, "cognitive_list_length.png")
        _plot_list_length(ll, first_rung, p, name); out.append(p)
        p = os.path.join(plots_dir, "cognitive_presentation_rate.png")
        _plot_bands(pr, pr["log2_passes"], "log2(passes over the list)",
                    f"Presentation rate, L={sp['length']} ({name}); human: slower rate "
                    "helps early/middle items, not the last few", p); out.append(p)
    if "pli_recency" in ph:
        p = os.path.join(plots_dir, "cognitive_pli_recency.png")
        _plot_pli(ph["pli_recency"], p, name); out.append(p)
    if "semantic_clustering" in ph:
        p = os.path.join(plots_dir, "cognitive_semantic_clustering.png")
        _plot_semantic(ph["semantic_clustering"], p, name); out.append(p)
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, dict):
        return {str(k): v for k, v in o.items()}
    return str(o)
