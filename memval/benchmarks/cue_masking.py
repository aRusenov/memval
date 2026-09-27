"""Cue-masking benchmark — the *structural* half of pattern completion.

`noise_invariance` degrades every feature of the cue a little (corruption).
This section removes some features entirely and leaves the rest exact
(partiality). They are the two manipulations `docs/capacities/capacities.md` ¶17 names,
and until this section landed only the corruption half had an instrument: the
`cue_completeness` dimension was UNBUILT and the spatial suite's fixed 30%
prompt is a *prefix*, neither swept nor dimension-masking.

Why the hierarchical encoder
----------------------------
Masking dimensions of a ``SymbolicEncoder`` word would measure almost nothing
about the model. Its embeddings are ``category_base + noise``, L2-normalised —
arbitrary coordinates in a random basis with no per-dimension meaning — so
zeroing a random subset acts like a random projection: it shrinks cosine to
*every* embedding at roughly the same rate and largely preserves their rank
order. That is the failure mode that got `letter_noise` deleted (audit D5):
a manipulation that scores the vocabulary's geometry rather than the arm.

``HierarchicalEncoder`` has the property the manipulation needs — its columns
are *owned by nodes of a tree*, so an item's active features partition into two
blocks that mean different things:

  * **shared**   — columns owned by the item's ancestors. Category-level
    structure, held in common with its siblings. "It is a bird."
  * **identity** — columns owned by the item's own leaf node. What distinguishes
    it from its siblings. "It is *this* bird."

That partition is what makes masking a *modality-like* manipulation rather than
a fraction of a random basis, and it is what lets the section ask a question no
scalar completeness sweep can: does it matter *which* features are missing, or
only how many? Losing category context and losing identity are different kinds
of fragment, and an arm that treats the cue as a bag of dimensions cannot tell
them apart. That is the holistic-retrieval claim of Horner et al. (2015) in the
form this suite can actually test.

Three masking modes, one grid
-----------------------------
``random`` masks a uniformly random fraction of the active features; ``shared``
and ``identity`` mask their own block first and spill into the other once it is
exhausted, so all three sweep the full [0, 1] range and stay comparable
point-for-point. The block modes only *differ* from each other while both blocks
are still non-empty (``mask_block_comparable_frac``), and the asymmetry read-out
is narrower still: it is taken over the window where the two modes remove the
same number of features **and** both masked cues are still identifiable
(``mask_block_window_frac``). Outside that window the modes differ in how much
they cost the stimulus, not only in which block they took.

Renormalisation is not cosmetic
-------------------------------
A masked cue is re-L2-normalised by default. Without it, masking mostly shrinks
the cue's norm; decoding here is cosine-based and many arms are linear, so a
smaller cue yields a smaller prediction in nearly the same direction and masking
would look almost free — for a reason that has nothing to do with completion.
Renormalised, the masked cue is a genuine projection onto a coordinate subspace,
on the unit-norm manifold the encoder and the ``l2``/``quantized`` feedback modes
already assume. ``renormalize=False`` is kept so the confound can be shown.

The model-free reference is the point of the section
----------------------------------------------------
Masking is only a model measurement to the extent that the masked cue still
*names* its item. ``cue_identifiability(f)`` — the fraction of masked cues whose
nearest studied embedding is still the true item — is computed from the encoder
alone, before any arm runs, and it does two jobs.

**Below it**, it says which failures are not the arm's. Once several studied
items' masked cues are bit-identical the arm receives the same input for each and
cannot emit different successors, so recall there is collision luck. Every scored
metric is restricted to the region where the reference still holds
(``mask_*_recall_in_bound``), and ``mask_resolved`` gates the dimension. This is
the discipline ``projection_ratio`` provides for `schema_consistency`.

**Above it**, it is a baseline worth beating. It is *not* a ceiling: it asks
whether a plain matched filter on the raw cue picks out the right item, and an
associative memory need not work that way. An arm whose recall runs above the
reference is recovering a target the degraded cue no longer specifies on its own
— which is pattern completion in the strict sense, as against cue matching.
``mask_*_completion_advantage`` is that contrast, and on the first arm run
through this section (AHN) it is positive at the sparse end of the sweep.

Both probes, per ¶17
--------------------
Cued recall masks the cue at every position independently. Autoregressive
rollout masks **only the initial cue** and then free-runs, which is ¶17's
"we only manipulate the initial cue, which additionally allows us to score how
an initial error propagates". ``mask_completion_gap`` is that contrast.
"""
import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..encoders.hierarchical import HierarchicalEncoder
from ..encoders.symbolic import SymbolicDecoder
from .exposure import epochs_to_criterion

#: Which features are removed first. ``random`` is the structure-agnostic graded
#: fragment; the other two are the block ("modality") manipulations.
MASK_MODES = ("random", "shared", "identity")

#: How the study list is drawn from the tree. ``across`` spans categories, so
#: both blocks carry information about which list item the cue is; ``within``
#: draws from one category, where every item has the *same* shared block and
#: identity alone discriminates. ``across`` is the default because under
#: ``within`` the block asymmetry is settled by construction rather than measured.
LIST_SCOPES = ("across", "within")

_MODE_LABEL = {
    "random": "random features",
    "shared": "shared (category) features first",
    "identity": "identity features first",
}


def feature_blocks(enc: HierarchicalEncoder, item: str) -> Dict[str, np.ndarray]:
    """Split ``item``'s active feature columns into ``shared`` and ``identity``.

    ``identity`` is the columns owned by the item's own leaf node; ``shared`` is
    everything it inherits from its ancestors. Both are intersected with the
    row's actually-active columns, so the split stays correct if a caller hands
    in an encoder whose matrix was built differently from its node table.
    """
    if item not in enc.word_to_idx:
        raise ValueError(f"Item '{item}' not in this hierarchy.")
    active = np.flatnonzero(enc.feature_matrix[enc.word_to_idx[item]] > 0)
    if item not in enc.name_to_node:
        # Augmented items have no node; treat their whole cue as identity.
        return {"active": active, "shared": np.array([], dtype=int),
                "identity": active}
    own = np.asarray(enc.nodes[enc.name_to_node[item]]["columns"], dtype=int)
    identity = np.intersect1d(active, own)
    shared = np.setdiff1d(active, identity)
    return {"active": active, "shared": shared, "identity": identity}


def mask_order(enc: HierarchicalEncoder, item: str, mode: str,
               rng: np.random.Generator) -> np.ndarray:
    """Order ``item``'s active columns, first-to-be-masked first.

    The block modes exhaust their own block before spilling into the other, so
    every mode covers the full fraction range and the curves stay comparable at
    every grid point. Order *within* a block is shuffled, so a block mode still
    averages over draws rather than reporting one arbitrary column order.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mode must be one of {MASK_MODES}, got {mode!r}")
    b = feature_blocks(enc, item)
    if mode == "random":
        return rng.permutation(b["active"])
    first, second = (b["shared"], b["identity"]) if mode == "shared" \
        else (b["identity"], b["shared"])
    return np.concatenate([rng.permutation(first), rng.permutation(second)])


def n_masked(n_active: int, fraction: float) -> int:
    """Columns removed at ``fraction``, rounded.

    The section's own grid is built from the realisable counts
    (``arange(n_active + 1) / n_active``) precisely so that no two grid points
    round to the same number of features and imply a resolution the substrate
    does not have. The rounding here is for callers passing an arbitrary
    fraction, and for items whose active-feature count differs from the modal
    one in an unbalanced tree.
    """
    return int(np.clip(round(fraction * n_active), 0, n_active))


def masked_cue(enc: HierarchicalEncoder, item: str, fraction: float, mode: str,
               rng: np.random.Generator, renormalize: bool = True) -> np.ndarray:
    """``item``'s embedding with a ``fraction`` of its active features removed."""
    vec = enc.embeddings[enc.word_to_idx[item]].copy()
    order = mask_order(enc, item, mode, rng)
    k = n_masked(len(order), fraction)
    if k:
        vec[order[:k]] = 0.0
    if renormalize:
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
    return vec


def cue_identifiability(enc: HierarchicalEncoder, items: Sequence[str],
                        fraction: float, mode: str, rng: np.random.Generator,
                        n_draws: int = 16, renormalize: bool = True,
                        candidates: Optional[Sequence[str]] = None) -> float:
    """Model-free: does the masked cue still name its own item?

    Fraction of masked cues whose nearest embedding by cosine is still the item
    the cue came from: a matched filter run on the raw cue, with no model in it.
    Where two items' masked cues collide, the arm receives the same input for
    both and cannot emit different successors, so a failure at that fraction
    belongs to the stimulus.

    **A reference curve, not a ceiling.** Recall above it is possible and
    informative — this asks whether the cue alone still picks out the item, and
    a trained associative memory can recover a target the raw cue no longer
    specifies. Read it as the line separating cue matching from completion, not
    as a bound a correct arm must sit under.

    ``candidates`` defaults to ``items`` — **the studied list, not the whole
    vocabulary** — because the studied list is the only set whose cues can
    collide in a way that costs the arm anything. A masked cue drifting nearer
    to an item that was never trained is harmless: the arm does not decode its
    own input, it maps it through weights encoding only the studied transitions.
    On the default substrate the two candidate sets happen to give the identical
    curve, because an item's identity columns are unique to it and a collision
    therefore needs the whole identity block gone — at which point the cue
    collides with every sibling, studied or not. The distinction still matters
    for any tree where partial identity loss can cause a partial collision.
    A tie counts as a **failure**, not a hit. Ties are not a corner case here:
    masking an item's whole identity block leaves exactly its ancestors'
    columns, which every sibling in the category shares *bit for bit*, so the
    masked cues of all six siblings become identical. Under ``argmax`` the
    sibling with the lowest vocabulary index would collect the credit for all of
    them and the bound would read ~0.5 where the true value is 0 -- an artefact
    of index order that would have flattered every arm scored against it.
    """
    pool = list(candidates if candidates is not None else items)
    E = enc.encode(pool)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    where = {w: j for j, w in enumerate(pool)}
    hits = total = 0
    for item in items:
        target = where[item]
        for _ in range(n_draws):
            cue = masked_cue(enc, item, fraction, mode, rng, renormalize)
            norm = float(np.linalg.norm(cue))
            total += 1
            if norm <= 1e-12:
                continue                      # an empty cue names nothing
            sims = E @ (cue / norm)
            others = np.delete(sims, target)
            if others.size and float(sims[target]) > others.max() + 1e-9:
                hits += 1
    return hits / total if total else float("nan")


def build_hierarchy(branching: Sequence[int] = (2, 12),
                    features_per_node: int = 6) -> HierarchicalEncoder:
    """Two-level tree: 2 categories x 12 items over 156 feature dimensions.

    **Shallower than `schema_consistency`'s (2, 2, 6), deliberately.** This
    section does not need a deep hierarchy; it needs two feature blocks that are
    both meaningful and *comparably sized*, because the block contrast is only
    valid where the two masks remove the same number of features. Under a
    three-level tree a leaf inherits from two ancestors and owns one node, so
    the blocks are 2:1 and the identity block is exhausted — taking the
    model-free bound to 0 with it — at a third of the sweep. The contrast then
    has a three-point window and reads 0.0 for want of range. At two levels the
    blocks are 1:1 and the window is half the grid.

    ``features_per_node=6`` sets both the resolution of that window (one grid
    point per feature removed) and the load. The tree gives 26 non-root nodes,
    so N = 26 * 6 = 156 and the default 8-item list sits at alpha = P/N ~ 0.051,
    well under the Hopfield-class capacity knee alpha_c ~ 0.138. Keeping that
    margin is not fussiness: reading a substrate-capacity ceiling as a model
    property is exactly what put `sequence_length` in CAVEAT. Raising `seq_len`
    is what spends it.
    """
    return HierarchicalEncoder(branching=branching,
                               features_per_node=features_per_node)


def study_list(enc: HierarchicalEncoder, seq_len: int, scope: str) -> List[str]:
    """Draw the list to be memorised.

    ``across`` takes items round-robin over the mid-level categories, so
    consecutive items differ in their shared block as well as their identity
    block and *both* blocks carry information about which item the cue is. Under
    ``within`` every list item shares one category, so the shared block is
    constant across the list and cannot discriminate — masking it is then free
    by construction and the block asymmetry is decided by the stimulus rather
    than measured on the arm.
    """
    if scope not in LIST_SCOPES:
        raise ValueError(f"scope must be one of {LIST_SCOPES}, got {scope!r}")
    by_cat: Dict[str, List[str]] = {}
    for item in enc.items:
        by_cat.setdefault(enc.category_of(item), []).append(item)
    cats = sorted(by_cat)
    if scope == "within":
        pool = by_cat[cats[0]]
        if seq_len > len(pool):
            raise ValueError(
                f"scope='within' has only {len(pool)} items in category "
                f"'{cats[0]}'; seq_len={seq_len} would silently borrow from "
                f"another category and confound load with category structure.")
        return pool[:seq_len]
    out: List[str] = []
    depth = 0
    while len(out) < seq_len:
        added = False
        for c in cats:
            if len(out) >= seq_len:
                break
            if depth < len(by_cat[c]):
                out.append(by_cat[c][depth])
                added = True
        if not added:
            raise ValueError(f"Hierarchy has only {len(out)} items; "
                             f"seq_len={seq_len} is not drawable.")
        depth += 1
    return out


# ==========================================================================
# Probes
# ==========================================================================

def _unit_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def cued_probe(model: Any, enc: HierarchicalEncoder, decoder: SymbolicDecoder,
               words: Sequence[str], fraction: float, mode: str,
               rng: np.random.Generator, n_draws: int = 8,
               renormalize: bool = True) -> Tuple[float, float]:
    """One-step cued recall under a masked cue: ``(recall, margin)``.

    Every position is probed independently with its own masked cue, so this is
    the association read-out with no rollout drift in it. The margin is
    ``cos(pred, target) - max cos(pred, other)`` over the whole vocabulary the
    decoder ranks; it is reported beside recall because it keeps resolving after
    recall has floored at 0, which is where the interesting part of a masking
    curve lives.
    """
    E = _unit_rows(enc.embeddings)
    ctx = np.array([1.0])
    hits = margins = 0.0
    total = 0
    for _ in range(n_draws):
        model.reset_context()
        for i in range(len(words) - 1):
            model.current_t = i
            cue = masked_cue(enc, words[i], fraction, mode, rng, renormalize)
            pred = model.predict_next(cue, current_context=ctx)
            hits += float(decoder.decode(pred, top_k=1)[0] == words[i + 1])
            pv = np.asarray(pred, dtype=float)
            pv = pv / (np.linalg.norm(pv) + 1e-12)
            sims = E @ pv
            ti = enc.word_to_idx[words[i + 1]]
            others = np.delete(sims, ti)
            margins += float(sims[ti] - (others.max() if others.size else -1.0))
            total += 1
    if not total:
        return float("nan"), float("nan")
    return hits / total, margins / total


def rollout_probe(model: Any, enc: HierarchicalEncoder, decoder: SymbolicDecoder,
                  words: Sequence[str], fraction: float, mode: str,
                  rng: np.random.Generator, n_draws: int = 8,
                  renormalize: bool = True,
                  feedback_mode: str = "l2") -> float:
    """Autoregressive recall from a masked *initial* cue only.

    ¶17's second probe. Only item 0's cue is manipulated; every later step is
    fed the model's own prediction, so what this measures beyond `cued_probe` is
    how far an initially-fragmentary cue propagates. ``feedback_mode="l2"``
    keeps the fed-back state on the encoder's unit-norm manifold, so the fall
    across the sweep is the masking rather than the magnitude drift that
    ``"raw"`` feedback adds on top of it (see `measure_recall_autoregressive`).
    """
    ctx = np.array([1.0])
    hits = 0.0
    total = 0
    for _ in range(n_draws):
        model.reset_context()
        current = masked_cue(enc, words[0], fraction, mode, rng, renormalize)
        for i in range(len(words) - 1):
            model.current_t = i
            pred = model.predict_next(current, current_context=ctx)
            decoded = decoder.decode(pred, top_k=1)[0]
            hits += float(decoded == words[i + 1])
            total += 1
            if feedback_mode == "quantized":
                current = enc.encode([decoded])[0]
            elif feedback_mode == "l2":
                norm = float(np.linalg.norm(pred))
                current = pred / norm if norm > 1e-12 else pred
            else:
                current = pred
    return hits / total if total else float("nan")


def _threshold(fractions: Sequence[float], curve: Sequence[float],
               level: float = 0.5) -> float:
    """Largest masked fraction still recalled at or above ``level``.

    Read as "how much of the cue can be thrown away". Mirrors
    `noise_tolerance_threshold`, which reports the sigma at which the curve
    first crosses 0.5; the sign convention is the same (higher is more robust).
    A curve already below ``level`` at f = 0 returns 0.0.
    """
    out = 0.0
    for f, y in zip(fractions, curve):
        if np.isnan(y) or y < level:
            break
        out = float(f)
    return out


# ==========================================================================
# Section runner
# ==========================================================================

def _make_model(model_class: Type, enc: HierarchicalEncoder,
                model_kwargs: Dict[str, Any], epochs: int):
    kwargs = dict(model_kwargs)
    kwargs["n_epochs"] = epochs
    kwargs["epochs"] = epochs
    if "encoder" in model_class.__init__.__code__.co_varnames:
        return model_class(encoder=enc, n_features=enc.embedding_dim, **kwargs)
    return model_class(n_features=enc.embedding_dim, **kwargs)


def _fit(model: Any, encoded: np.ndarray, epochs: int) -> None:
    if hasattr(model, "n_epochs"):
        model.n_epochs = epochs
    model.fit_sequence(encoded, epochs=epochs)


def run_cue_masking(
    model_class: Type,
    model_kwargs: Optional[Dict[str, Any]] = None,
    branching: Sequence[int] = (2, 12),
    features_per_node: int = 6,
    seq_len: int = 8,
    list_scope: str = "across",
    n_draws: int = 8,
    renormalize: bool = True,
    feedback_mode: str = "l2",
    recall_level: float = 0.5,
    identifiable_level: float = 0.95,
    exposure: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    output_dir: str = "results",
    run_name: Optional[str] = None,
    run_dir: Optional[str] = None,
    metrics_filename: str = "metrics.json",
) -> Dict[str, Any]:
    """Sweep cue completeness and score what survives.

    Protocol
    --------
    0. **Model-free reference.** ``cue_identifiability`` is swept first, before
       any arm is built, so the region in which the manipulation is a model
       question at all is known independently of the result.
    1. **Acquisition.** The list is trained to criterion on *clean* cued recall
       (``fraction = 0``). Exposure is a result — ``mask_epochs_to_criterion`` —
       not a parameter, so every arm is probed at the same functional point.
       This is the same pairing `noise_invariance` uses: settle at the
       undegraded end, score the sweep.
    2. **Sweep.** For each masking mode, cued recall + margin + autoregressive
       rollout at every realisable masked-feature count.

    ``exposure`` is the policy dict from ``memval.benchmarks.exposure``; omitted,
    the section runs criterion-referenced with the module defaults.

    ``run_dir`` / ``metrics_filename`` let a host suite embed this section in its
    own output tree, exactly as `schema_consistency` does.
    """
    if list_scope not in LIST_SCOPES:
        raise ValueError(f"list_scope must be one of {LIST_SCOPES}, got {list_scope!r}")
    model_kwargs = dict(model_kwargs or {})
    # Seeded per probe below rather than from one shared stream, so that adding
    # or reordering a mask mode cannot shift the draws every other mode sees.
    # np.random.seed covers the arms that use the legacy global state internally.
    np.random.seed(seed)

    enc = build_hierarchy(branching, features_per_node)
    words = study_list(enc, seq_len, list_scope)
    decoder = SymbolicDecoder(enc)
    encoded = enc.encode(words)

    blocks = [feature_blocks(enc, w) for w in words]
    n_active = int(max(len(b["active"]) for b in blocks))
    n_shared = int(np.median([len(b["shared"]) for b in blocks]))
    n_identity = int(np.median([len(b["identity"]) for b in blocks]))
    # Sweep the realisable masked-feature counts rather than an arbitrary grid,
    # so no two grid points silently mask the same number of features.
    fractions = (np.arange(n_active + 1) / n_active).tolist()
    counts = list(range(n_active + 1))
    # The two block modes mask the same COUNT and differ only in WHICH features
    # until the smaller block runs out; past that they have both spilled into the
    # other block and the contrast is no longer clean. The asymmetry read-out is
    # restricted to this prefix.
    comparable_k = min(n_shared, n_identity)
    comparable_frac = comparable_k / n_active if n_active else float("nan")

    metrics: Dict[str, Any] = {}
    series: Dict[str, Any] = {"fractions": fractions, "counts": counts}

    # --- 0. model-free reference, computed before any model exists -----------
    # Reseeded at every grid point, matching the model probes below. Sharing one
    # stream across the sweep would let the reference at f drift on draws
    # consumed at f-1, so the reference curve and the recall curve would be
    # measured against different masks and `completion_advantage` -- their
    # difference -- would carry that drift as signal.
    ident: Dict[str, List[float]] = {}
    for mode in MASK_MODES:
        ident[mode] = [cue_identifiability(enc, words, f, mode,
                                           np.random.default_rng(seed + 17),
                                           n_draws=max(n_draws, 16),
                                           renormalize=renormalize)
                       for f in fractions]
    series["identifiability"] = ident
    metrics["mask_identifiability_auc"] = float(np.mean(ident["random"]))
    metrics["mask_identifiable_to"] = _threshold(fractions, ident["random"],
                                                 identifiable_level)
    for mode in MASK_MODES:
        metrics[f"mask_{mode}_identifiable_to"] = _threshold(
            fractions, ident[mode], identifiable_level)

    # --- 1. acquisition to criterion on the clean cue ------------------------
    policy = exposure or {"mode": "criterion", "epochs": None, "criterion": 0.95,
                          "max_epochs": 512, "fallback_epochs": 300}

    def _score_clean(m) -> float:
        r = np.random.default_rng(seed + 1)
        return cued_probe(m, enc, decoder, words, 0.0, "random", r,
                          n_draws=1, renormalize=renormalize)[0]

    if policy["mode"] == "fixed":
        n_ep = int(policy["epochs"])
        model = _make_model(model_class, enc, model_kwargs, n_ep)
        _fit(model, encoded, n_ep)
        clean = float(_score_clean(model))
        exp_rec = {"epochs": n_ep, "reached": clean >= policy["criterion"]}
    else:
        res = epochs_to_criterion(
            lambda: _make_model(model_class, enc, model_kwargs,
                                int(policy["max_epochs"])),
            lambda m, ep: _fit(m, encoded, ep),
            _score_clean,
            criterion=float(policy["criterion"]),
            max_epochs=int(policy["max_epochs"]))
        model, exp_rec = res["model"], {"epochs": res["epochs"],
                                        "reached": res["reached"]}
    metrics["mask_epochs_to_criterion"] = float(exp_rec["epochs"])
    metrics["mask_criterion_reached"] = bool(exp_rec["reached"])
    metrics["mask_exposure_mode"] = policy["mode"]

    # --- 2. the sweep --------------------------------------------------------
    recall: Dict[str, List[float]] = {}
    margin: Dict[str, List[float]] = {}
    rollout: Dict[str, List[float]] = {}
    for mode in MASK_MODES:
        rec, mar, rol = [], [], []
        for f in fractions:
            # Same seed at every grid point on purpose: the mask drawn at f is
            # then a prefix of the one drawn at f+1, so the sweep is paired
            # rather than independent at each point. A monotone curve is the
            # thing being measured; independent draws would add variance to
            # exactly the comparison the curve exists to make.
            r = np.random.default_rng(seed + 101)
            a, m = cued_probe(model, enc, decoder, words, f, mode, r,
                              n_draws=n_draws, renormalize=renormalize)
            r = np.random.default_rng(seed + 202)
            b = rollout_probe(model, enc, decoder, words, f, mode, r,
                              n_draws=n_draws, renormalize=renormalize,
                              feedback_mode=feedback_mode)
            rec.append(a); mar.append(m); rol.append(b)
        recall[mode], margin[mode], rollout[mode] = rec, mar, rol
        metrics[f"mask_{mode}_tolerance"] = _threshold(fractions, rec, recall_level)
        metrics[f"mask_{mode}_auc"] = float(np.mean(rec))
        metrics[f"mask_{mode}_margin_auc"] = float(np.mean(mar))
        metrics[f"mask_{mode}_rollout_auc"] = float(np.mean(rol))
        # Scored companion to the full-grid AUCs, restricted to the region where
        # the masked cue still names its item. Past the reference the arm gets
        # identical inputs for several studied items, so what the full-grid AUC
        # picks up there is which of the colliding successors it happens to
        # favour -- collision luck, not completion. The `identity` mode spends
        # half its grid in that state, so its AUC and its in-bound rate are
        # different questions and only the second is scoreable.
        keep = [j for j, y in enumerate(ident[mode]) if y >= identifiable_level]
        metrics[f"mask_{mode}_recall_in_bound"] = float(
            np.mean([rec[j] for j in keep])) if keep else float("nan")
        metrics[f"mask_{mode}_margin_in_bound"] = float(
            np.mean([mar[j] for j in keep])) if keep else float("nan")
        metrics[f"mask_{mode}_in_bound_points"] = int(len(keep))
    series.update({"recall": recall, "margin": margin, "rollout": rollout})

    # --- 3. contrasts --------------------------------------------------------
    # Completion proper, as against cue matching: how far the arm's recall runs
    # above the matched filter on the same raw cue, over the whole sweep.
    #
    #   > 0  the arm recovers targets the degraded cue no longer specifies on its
    #        own -- the attractor is doing work the stimulus does not do for it.
    #   ~ 0  recall tracks what the cue still names; the arm is a cue matcher.
    #   < 0  the arm fails where the cue was still unambiguous.
    #
    # Read over the FULL grid deliberately. Inside the identifiable region the
    # reference is 1.0 by construction and the difference is just
    # `recall_in_bound` restated; all the signal is at the sparse end.
    for mode in MASK_MODES:
        metrics[f"mask_{mode}_completion_advantage"] = float(np.mean(
            np.asarray(recall[mode]) - np.asarray(ident[mode])))
    # Deliberately no un-suffixed alias: three modes, three numbers, and a bare
    # `mask_completion_advantage` silently meaning "the random one" is how a
    # reader ends up quoting a block-mode result as the headline.

    # Propagation cost of an initially-fragmentary cue: what rollout loses
    # relative to independently-probed association at the same completeness.
    metrics["mask_completion_gap"] = float(np.mean(
        [metrics[f"mask_{m}_auc"] - metrics[f"mask_{m}_rollout_auc"]
         for m in MASK_MODES]))

    # Does it matter WHICH features are missing, at an equal number missing?
    #
    # Raw recall cannot answer this, and the model-free sweep is why. Masking an
    # item's shared block never costs identifiability -- its identity columns are
    # unique, so the cue still names it -- whereas masking its identity block
    # makes the cue bit-identical to every sibling and the reference drops
    # straight to 0. Comparing the two recall curves directly would therefore
    # report a property of the stimulus with the arm's contribution buried in it.
    #
    # The contrast is read only in the window where the two masks are matched on
    # both counts that matter: the SAME NUMBER of features removed (k <=
    # comparable_k, before either block is exhausted) and BOTH still fully
    # identifiable. In that window the modes differ in exactly one thing --
    # which block was taken -- so the difference is the arm's.
    #
    #   > 0  removing identity features costs less than removing shared ones:
    #        the arm leans on category structure it did not need, since identity
    #        alone always sufficed to name the cue.
    #   < 0  the arm leans on the identity features.
    #   ~ 0  the cue is a bag of dimensions to it -- the holistic-retrieval null.
    window = [j for j, k in enumerate(counts)
              if k <= comparable_k
              and ident["shared"][j] >= identifiable_level
              and ident["identity"][j] >= identifiable_level]
    metrics["mask_block_asymmetry"] = float(np.mean(
        [recall["identity"][j] - recall["shared"][j] for j in window])) \
        if window else float("nan")
    metrics["mask_block_window_frac"] = float(fractions[window[-1]]) if window else float("nan")
    metrics["mask_block_window_points"] = int(len(window))
    metrics["mask_block_comparable_frac"] = float(comparable_frac)

    # --- 4. guards -----------------------------------------------------------
    clean_recall = float(recall["random"][0])
    chance_level = 1.0 / len(enc.items)
    acquired = bool(clean_recall >= max(recall_level, 2 * chance_level))
    identifiable_region = metrics["mask_identifiable_to"] >= 0.25
    # Aimed at the SCORED recall metrics -- the in-bound rates -- not at the
    # full-grid curves. An arm can look like it has range because the identity
    # mode collapses past its bound while every in-bound rate sits at 1.00,
    # which is exactly AHN: a perfect completer wherever the cue is unambiguous.
    # A ceiling flag computed off the full grid would have called that "ranging"
    # and hidden the fact that the recall half of the dimension said nothing
    # about the arm.
    at_ceiling = bool(min(metrics[f"mask_{m}_recall_in_bound"]
                          for m in MASK_MODES) >= 0.95)
    # The dimension is still informative at that ceiling if something else
    # scored ranges: the margin resolves below the accuracy floor (that is why
    # it is scored) and the tolerance is read off the full grid.
    margin_ranges = max(float(np.nanmax(margin[m]) - np.nanmin(margin[m]))
                        for m in MASK_MODES) > 0.05
    tolerance_ranges = min(metrics[f"mask_{m}_tolerance"]
                           for m in MASK_MODES) < 1.0
    metrics.update({
        "mask_clean_recall": clean_recall,
        "mask_chance_level": float(chance_level),
        "mask_acquired": acquired,
        "mask_at_ceiling": at_ceiling,
        "mask_at_floor": bool(not acquired),
        "mask_identifiable_region": bool(identifiable_region),
        "mask_margin_ranges": bool(margin_ranges),
        "mask_resolved": bool(acquired and identifiable_region
                              and (not at_ceiling or margin_ranges
                                   or tolerance_ranges)),
    })

    # --- 5. configuration, recorded because every read-out depends on it -----
    metrics.update({
        "mask_seq_len": int(seq_len),
        "mask_n_features": int(enc.embedding_dim),
        "mask_active_features": n_active,
        "mask_shared_features": n_shared,
        "mask_identity_features": n_identity,
        "mask_alpha": float(seq_len / enc.embedding_dim),
        "mask_list_scope": list_scope,
        "mask_renormalize": bool(renormalize),
        "mask_feedback_mode": feedback_mode,
        "mask_n_items": int(len(enc.items)),
    })

    if not acquired:
        print(f"WARNING: cue_masking at floor -- clean-cue recall is "
              f"{clean_recall:.2f} (chance {chance_level:.3f}). The list was "
              f"never learned, so the sweep measures nothing about masking. "
              f"Raise the exposure budget or lower seq_len "
              f"(alpha = {metrics['mask_alpha']:.3f}).")
    elif at_ceiling:
        print("NOTE: cue_masking in-bound recall is saturated -- this arm "
              "completes correctly at every masked fraction where the cue still "
              "names its item, in all three modes. Not a failure of the section, "
              "but the recall metrics carry no information about this arm; the "
              "dimension is carried by mask_random_tolerance and "
              "mask_random_margin_in_bound. Raise seq_len to push it off the "
              "ceiling.")
    elif not identifiable_region:
        print(f"WARNING: cue_masking stimulus is degenerate -- masked cues stop "
              f"naming their own item at f = {metrics['mask_identifiable_to']:.2f}. "
              f"Below that the sweep scores the encoder's geometry, not the arm. "
              f"Raise features_per_node.")

    model_name = run_name or model_class.__name__
    if run_dir is None:
        run_dir = os.path.join(output_dir, model_name, "cue_masking")
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    _plot(series, metrics, model_name, run_dir, recall_level)

    results = {
        "metadata": {"model_name": model_name, "suite": "cue_masking",
                     "branching": list(branching),
                     "features_per_node": features_per_node,
                     "seq_len": seq_len, "list_scope": list_scope,
                     "n_draws": n_draws, "renormalize": renormalize,
                     "feedback_mode": feedback_mode, "seed": seed,
                     "words": list(words),
                     "model_kwargs": {k: str(v) for k, v in model_kwargs.items()}},
        "metrics": metrics,
        "series": {"mask_sweep": series},
    }
    with open(os.path.join(run_dir, metrics_filename), "w") as f:
        json.dump(results, f, indent=2)
    return results


def _plot(series, metrics, model_name, run_dir, recall_level) -> None:
    """Three panels, because the three read-outs answer different questions and
    overlaying them hides the one that matters most.

    The per-mode reference curves are drawn in each mode's own colour rather than
    as one shared line. With a single reference plotted, the identity curve's
    cliff reads as the arm failing, when in fact that mode's own reference fell
    at the same point: the cue stopped naming its item and every arm is at
    collision luck past it. The gap between a solid line and the dashed line of
    the same colour is the only vertical distance on the panel that is about the
    model.
    """
    f = series["fractions"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for k, mode in enumerate(MASK_MODES):
        ax.plot(f, series["recall"][mode], marker="o", ms=4, color=f"C{k}",
                label=f"recall — {_MODE_LABEL[mode]}")
        ax.plot(f, series["identifiability"][mode], color=f"C{k}", lw=1.1,
                ls="--", alpha=0.55)
    ax.axhline(recall_level, ls=":", c="grey", lw=0.8)
    ax.set_xlabel("fraction of cue features masked")
    ax.set_ylabel("cued recall")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Completion from a partial cue\n(dashed: same-colour model-free reference)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    for k, mode in enumerate(MASK_MODES):
        ax.plot(f, series["margin"][mode], marker="s", ms=4, color=f"C{k}",
                label=_MODE_LABEL[mode])
    ax.axhline(0, c="k", lw=0.8)
    cmp_f = metrics.get("mask_block_window_frac")
    if cmp_f is not None and np.isfinite(cmp_f):
        ax.axvspan(0, cmp_f, color="grey", alpha=0.10)
        ax.annotate("blocks matched on\ncount and identifiability",
                    xy=(0.01, 0.80), xycoords="axes fraction", fontsize=7,
                    color="dimgrey", va="top")
    ax.set_xlabel("fraction of cue features masked")
    ax.set_ylabel("cosine margin (target − best competitor)")
    ax.set_title("Margin, which resolves below the recall floor", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[2]
    for k, mode in enumerate(MASK_MODES):
        rol = series["rollout"][mode]
        ax.plot(f, rol, marker="^", ms=4, color=f"C{k}", label=_MODE_LABEL[mode])
        if k == 0 and rol:
            # The level here is set by free-running ability, not by masking, so
            # the unmasked value is drawn as the line the curve falls away from.
            ax.axhline(rol[0], ls=":", c="dimgrey", lw=0.9)
            ax.annotate("unmasked rollout", xy=(0.02, rol[0] + 0.02), fontsize=7,
                        color="dimgrey")
    ax.set_xlabel("fraction of the INITIAL cue masked")
    ax.set_ylabel("autoregressive recall")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Rollout from a masked initial cue\n(propagation, not association)",
                 fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle(f"Cue masking — {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plots", "cue_masking.png"), dpi=150)
    plt.close(fig)
