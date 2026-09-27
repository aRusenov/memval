"""Schema-consistency benchmark.

Scores how the *rate* of new learning and the *extent* of interference depend on
how consistent new material is with what the model already knows. Motivated by
McClelland, McNaughton & Lampinen (2020), which retracts the 1995 claim that
cortical learning is slow per se: learning rate is prior-knowledge dependent, and
interference is a matter of degree set by consistency. See
docs/sections/schema_benchmark_design.md for the full design rationale, including why the
`duplicate` rung deviates from the paper and why the schema must be acquired
rather than declared.

Protocol
--------
A. **Schema acquisition** - the model learns one sequence per mid-level category
   of a `HierarchicalEncoder` tree, so the schema is the transition structure it
   actually acquires, not a label supplied by the generator.
B. **Schema probe** - category-level vs item-level recall accuracy. A model that
   has acquired the schema predicts into the right category even when it gets the
   exact item wrong, so `category_accuracy >> item_accuracy` is the signature.
   Without this check a model that never learned the schema would be scored as
   though it had one, and the consistency manipulation would be vacuous.
C. **New item** - for each rung of the consistency ladder independently, the
   schema is retrained from scratch (rather than deep-copied, which is not safe
   across all model backends) and the model then learns a sequence in which the
   final slot is occupied by the new item.
D. **Readout** - the new sequence extends the host sequence by one item, so only
   its *final* transition is new; the rest is material the schema already
   taught. Acquisition is therefore scored on that transition alone
   (`new_item_recall`), not on the sequence-mean MRR, which mixes one new
   transition into N-1 known ones and so cannot fall below (N-2)/(N-1). The
   already-known prefix is reported separately (`prefix_recall`) so learning the
   new item at the cost of the host sequence is visible rather than netted out.
   Interference with the *other* base sequences is split by relatedness (same
   category / sibling category / far category), which is the graded profile that
   motivates SWIL.

`projection_ratio` is reported alongside as a model-free measure of each rung's
consistency, so the ladder can be verified as correctly ordered independently of
any model's behaviour.
"""
import os
import json
import inspect
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..encoders.hierarchical import HierarchicalEncoder
from ..encoders.symbolic import SymbolicDecoder
from .symbolic_pipeline import measure_recall_associative, mean_recall_rate


# Ladder rungs, most-consistent first. Names are the paper's exemplars.
RUNGS = ("duplicate", "within", "across", "random")

#: Rungs the headline correlations are computed over. ``duplicate`` is excluded
#: and reported separately: its row is bit-identical to an existing item in the
#: base columns (cosine 0.87 to `sparrow`; the next rung is 0.72), so under a
#: same-space ``predict_next`` read-out the decoder must separate two
#: near-identical targets. Across 5 seeds on EP it is the *slowest* rung
#: (5.0 +- 2.0 trials vs 2.2-2.6) and the only rung with any spread, and it alone
#: flips both correlations to the wrong sign. That is a discriminability cost the
#: paper's architecture never pays -- it has a dedicated input unit per item and
#: reads out attributes, not identity -- not a consistency effect. Keeping it in
#: the correlation would let the decoder vote on the schema question. See
#: ``schema_duplicate_excess_trials`` and ``schema_<rung>_nearest_base_cosine``.
STRUCTURED_RUNGS = ("within", "across", "random")

#: How the new item is trained, once the schema has been acquired.
#:
#: ``"extended"`` -- train on the whole host list with the new item appended
#:     (``sparrow -> ... -> wren -> sparrowhawk``). Every base transition is
#:     retrained alongside the new one, so this is a *partially interleaved*
#:     condition: the new item is interleaved with its own category and nothing
#:     else. McClelland (2013) reports this variant too (its Figure 3, lower
#:     panels) and notes that interleaving *retards* acquisition even of
#:     consistent items.
#: ``"focused"`` -- train on the new association alone
#:     (``wren -> sparrowhawk``). This is McClelland's main protocol: "only the
#:     two experiences involving one of the new items were presented, without
#:     interleaving".
#:
#: **The probe is identical under both**; only the training data differs, so
#: every read-out stays directly comparable across the two.
INTERFERENCE_PROTOCOLS = ("extended", "focused")

_RUNG_LABEL = {
    "duplicate": "consistent (cardinal)",
    "within": "within-branch (sparrowhawk)",
    "across": "cross-branch (penguin)",
    "random": "unstructured (random)",
}


def build_hierarchy(branching: Sequence[int] = (2, 2, 6),
                    features_per_node: int = 2) -> HierarchicalEncoder:
    """Balanced tree: 2 superordinate x 2 mid-level x 6 items = 24 items.

    Balanced by default because branch size sets singular value and singular
    value sets learning order; an unbalanced tree makes the differentiation
    ordering an artefact of vocabulary construction rather than a model property.
    """
    return HierarchicalEncoder(branching=branching,
                               features_per_node=features_per_node)


def category_sequences(enc: HierarchicalEncoder) -> Dict[str, List[str]]:
    """One sequence per mid-level category, visiting that category's items."""
    out: Dict[str, List[str]] = {}
    for item in enc.base_items if hasattr(enc, "base_items") else enc.items:
        out.setdefault(enc.category_of(item), []).append(item)
    return out


def _relatedness(enc: HierarchicalEncoder, cat_a: str, cat_b: str) -> str:
    """same / sibling / far, by depth of the categories' common ancestor."""
    if cat_a == cat_b:
        return "same"
    a = enc.nodes[enc.name_to_node[cat_a]]
    b = enc.nodes[enc.name_to_node[cat_b]]
    return "sibling" if a["parent"] == b["parent"] else "far"


def _sequence_identity_available(model: Any) -> bool:
    """Can this arm be *told* which sequence a training pair belongs to?

    Two things are required, and four of the six arms in the taxonomy have
    exactly one of them:

    1. a context dimension on the instance (``n_context > 0``), and
    2. a ``fit_sequence`` that accepts ``context_data``.

    Condition 2 alone is worthless and actively dangerous: ``TemporalPCNetwork``,
    ``ThetaPhaseSequenceNetwork`` and ``DTSESNSequenceNetwork`` all *accept* ``context_data`` and silently ignore
    it, so passing a sequence identifier to them would look like it worked and
    change nothing. ``OriginalEqPropSequenceNetwork`` does not accept the
    argument at all. Only ``AsymmetricHopfieldNetwork`` declares ``n_context``,
    and even there it defaults to 0 -- so this is a property of the *instance*
    as configured, not of the class.

    Nothing in this benchmark passes a sequence identifier today; the flag is
    reported so a reader can see that the extended/focused distinction below is
    a statement about *what was fed*, never about what the model was told. See
    docs/sections/schema_benchmark_design.md §5.5 and the audit's pending item 18: doing
    this properly needs a nominal capability declaration alongside
    ``OnlineTrainable``, not a duck-typed check like this one.
    """
    n_ctx = getattr(model, "n_context", 0) or 0
    try:
        accepts = "context_data" in inspect.signature(
            type(model).fit_sequence).parameters
    except (TypeError, ValueError):          # C-implemented or unintrospectable
        accepts = False
    return bool(n_ctx > 0 and accepts)


def _fresh_model(model_class: Type, n_features: int, model_kwargs: Dict[str, Any],
                 epochs_per_trial: int):
    kwargs = dict(model_kwargs)
    kwargs["n_epochs"] = epochs_per_trial
    kwargs["epochs"] = epochs_per_trial
    return model_class(n_features=n_features, **kwargs)


def _fit(model, encoded: np.ndarray, epochs: int) -> None:
    """One training trial. fit_sequence is incremental (no weight reset), which
    is what the online-continual pipeline already relies on."""
    if hasattr(model, "n_epochs"):
        model.n_epochs = epochs
    model.fit_sequence(encoded, epochs=epochs)


def _train_schema(model, enc, seqs: Dict[str, List[str]], trials: int,
                  epochs_per_trial: int) -> None:
    for _ in range(trials):
        for words in seqs.values():
            model.reset_context()
            _fit(model, enc.encode(words), epochs_per_trial)


def _train_schema_to_criterion(model, enc, decoder, seqs: Dict[str, List[str]],
                               ceiling: int, epochs_per_trial: int,
                               criterion: float, window: int,
                               n_probe_trials: int, noise_scale: float
                               ) -> Dict[str, Any]:
    """Train the schema one trial at a time until base recall holds at criterion.

    This is what makes the section's cost read-out **unit-free**. A "trial" is
    one pass over each category list (``epochs_per_trial`` epochs), and one pass
    is not the same amount of learning for every arm: AHN applies one LMS
    update per transition, EP accumulates the transitions and applies a single
    averaged update at ``lr / n_transitions``. At the default one epoch per
    trial EP's new-item recall sits at exactly 0.0 for all 15 trials while AHN
    resolves at 5-10 -- the raw ``trials_to_criterion`` of the two arms are in
    different units. Dividing by the trials the *same arm* needed to acquire
    the schema cancels the unit (``schema_<rung>_relative_cost``), which is how
    McClelland reads Fig. 3: new-item error against the arm's own end-of-
    acquisition reference line, never raw epochs across architectures.

    Returns ``{"trials", "reached", "curve"}``; ``trials`` is the ceiling when
    unmet, with ``reached=False`` -- report both.
    """
    curve: List[float] = []
    for t in range(1, ceiling + 1):
        for words in seqs.values():
            model.reset_context()
            _fit(model, enc.encode(words), epochs_per_trial)
        mrrs = []
        for words in seqs.values():
            model.reset_context()
            mrrs.append(mean_recall_rate(measure_recall_associative(
                model, words, enc, decoder, n_trials=n_probe_trials,
                noise_scale=noise_scale)))
        curve.append(float(np.mean(mrrs)))
        if len(curve) >= window and all(v >= criterion for v in curve[-window:]):
            return {"trials": float(t), "reached": True, "curve": curve}
    return {"trials": float(ceiling), "reached": False, "curve": curve}


def _schema_probe(model, enc, decoder, seqs: Dict[str, List[str]],
                  n_trials: int, noise_scale: float) -> Dict[str, float]:
    """Item-level and category-level one-step accuracy.

    Category-level credit is given when the predicted item's category matches
    the target's, which is what generalisation to the schema looks like when the
    exact item is not recoverable.
    """
    item_hits = cat_hits = total = 0
    ctx = np.array([1.0])
    for words in seqs.values():
        for _ in range(n_trials):
            for i in range(len(words) - 1):
                model.current_t = i
                cue = enc.encode([words[i]])[0]
                cue = cue + np.random.normal(0, noise_scale, enc.embedding_dim)
                pred = model.predict_next(cue, current_context=ctx)
                got = decoder.decode(pred, top_k=1)[0]
                total += 1
                if got == words[i + 1]:
                    item_hits += 1
                try:
                    if enc.category_of(got) == enc.category_of(words[i + 1]):
                        cat_hits += 1
                except (KeyError, ValueError):
                    pass          # a new item has no category in the base tree
    if total == 0:
        return {"item_accuracy": float("nan"), "category_accuracy": float("nan")}
    return {"item_accuracy": item_hits / total,
            "category_accuracy": cat_hits / total}


def _trials_to_criterion(curve: Sequence[float], criterion: float,
                         window: int) -> Dict[str, Any]:
    """First trial after which the criterion holds for `window` consecutive
    trials. An unmet criterion returns len(curve) so the scalar stays plottable;
    `reached` records whether it was actually met -- report both, since
    trials-to-criterion is meaningless without it."""
    for i in range(len(curve) - window + 1):
        chunk = curve[i:i + window]
        if all((not np.isnan(v)) and v >= criterion for v in chunk):
            return {"trials": float(i + 1), "reached": True}
    return {"trials": float(len(curve)), "reached": False}


def _run_schema_consistency_once(
    model_class: Type,
    model_kwargs: Optional[Dict[str, Any]] = None,
    branching: Sequence[int] = (2, 2, 6),
    features_per_node: int = 2,
    schema_trials: int = 40,
    schema_criterion: float = 0.75,
    epochs_per_trial: int = 1,
    new_item_trials: int = 15,
    criterion_frac: float = 0.75,
    criterion_window: int = 2,
    n_probe_trials: int = 1,
    noise_scale: float = 0.0,
    interference_protocol: str = "extended",
    seed: int = 0,
):
    """One seed of the schema-consistency protocol. Returns
    ``(metrics, series, proj, host_cat)``; :func:`run_schema_consistency`
    replicates this over seeds, aggregates, plots and persists.

    ``interference_protocol`` selects how the new item is trained; see
    :data:`INTERFERENCE_PROTOCOLS`. It is a **harness-side switch with no model
    interface**: all it changes is the array handed to ``fit_sequence`` -- a
    ``(len(host)+1, dim)`` array under ``"extended"`` against a ``(2, dim)``
    array under ``"focused"``. The arm cannot detect it, condition on it, or
    behave differently because of it; it is recorded so a reader of
    ``metrics.json`` can tell which of two different questions the interference
    numbers answer.

    Choose it by which read-out you are after:

    * **Acquisition** (``trials_to_criterion``, ``new_item_recall``,
      ``new_item_auc``) is measured under either. ``"extended"`` is the harder,
      partially-interleaved condition.
    * **Interference** (``schema_<rung>_interference_*``) is only valid under
      ``"focused"``. Under ``"extended"`` the new list *is* the host list plus
      one item, so learning the new item rehearses all five base transitions of
      the host category -- exactly the category where the paper predicts the
      largest damage. The ``same`` bucket then measures rehearsal, not
      interference, and comes out at or below zero. ``schema_interference_valid``
      records this; it is ``False`` under ``"extended"``.

    Verified 2026-09-02 on ``OriginalEqPropSequenceNetwork``: switching to
    ``"focused"`` recovers the paper's local-interference gradient
    (same/sibling/far = +0.84 / +0.16 / +0.04 for the ``within`` rung, against
    -0.08 / +0.08 / +0.04 under ``"extended"``), and the effect is not a dose
    artefact -- it is unchanged at a 40x smaller update budget.

    The default stays ``"extended"`` so previously reported acquisition numbers
    do not move underneath anyone. Interference results should be regenerated
    under ``"focused"``.

    ``schema_trials`` is a **ceiling**, not a budget. Schema acquisition is
    staircased: one trial at a time until mean base-list MRR holds at
    ``schema_criterion`` for ``criterion_window`` consecutive trials, capped at
    ``schema_trials``. The count is ``schema_acquisition_trials``; every rung's
    model is then trained for exactly that many trials so all rungs start from
    the same exposure. ``schema_<rung>_relative_cost`` = trials to acquire the
    new item / trials to acquire the schema -- the cross-arm-comparable cost,
    since the per-trial unit cancels (see ``_train_schema_to_criterion``). Raw
    ``trials_to_criterion`` stays reported; it is within-arm only.

    ``criterion_frac`` is applied to ``base_mrr`` and the result is checked
    against ``schema_<rung>_new_item_recall`` -- recall of the single new
    transition -- **not** against the sequence-mean MRR. See the module
    docstring, section D: the sequence mean has an arithmetic floor of
    (N-2)/(N-1) and a criterion below it is met on trial 1 regardless of
    learning. ``schema_dilution_floor`` and
    ``schema_criterion_below_dilution_floor`` record that floor explicitly.

    ``run_dir`` and ``metrics_filename`` exist so a host suite can embed this
    section in its own output tree: ``run_symbolic_pipeline`` passes its run
    directory and a distinct filename, so the plot lands beside the other
    symbolic plots without clobbering the suite's ``metrics.json``. Called
    directly (both unset) the section keeps its self-contained layout under
    ``<output_dir>/<model>/schema_consistency/``.
    """
    if interference_protocol not in INTERFERENCE_PROTOCOLS:
        raise ValueError(f"interference_protocol must be one of "
                         f"{INTERFERENCE_PROTOCOLS}, got {interference_protocol!r}")
    model_kwargs = dict(model_kwargs or {})
    rng_seed = seed
    np.random.seed(rng_seed)

    base = build_hierarchy(branching, features_per_node)
    base_seqs = category_sequences(base)
    categories = sorted(base_seqs)
    host_cat = categories[0]                       # new items join this category

    specs = [
        {"name": "new_duplicate", "kind": "duplicate", "target": base_seqs[host_cat][0]},
        {"name": "new_within",    "kind": "within",    "parent": host_cat},
        {"name": "new_across",    "kind": "across",    "parent": host_cat,
         "other": categories[1]},
        {"name": "new_random",    "kind": "random"},
    ]
    enc = base.augment(specs, seed=rng_seed)
    decoder = SymbolicDecoder(enc)
    n_features = enc.embedding_dim

    metrics: Dict[str, Any] = {}
    series: Dict[str, Any] = {}

    # --- model-free ladder check: is the ladder ordered as intended? ---
    proj = {}
    for rung, spec in zip(RUNGS, specs):
        row = enc.feature_matrix[enc.word_to_idx[spec["name"]], :base.n_features]
        proj[rung] = base.projection_ratio(row)
        metrics[f"schema_{rung}_projection_ratio"] = proj[rung]
    # Discriminability covariate: how close the new item sits to its nearest
    # base item in the space the decoder actually searches. This is what
    # separates `duplicate` from the rest (0.87 vs <= 0.72) and why it is
    # reported outside the correlation.
    base_rows = enc.embeddings[[enc.word_to_idx[w] for w in base.items]]
    for rung, spec in zip(RUNGS, specs):
        v = enc.embeddings[enc.word_to_idx[spec["name"]]]
        metrics[f"schema_{rung}_nearest_base_cosine"] = float(np.max(base_rows @ v))
    ordered = all(proj[RUNGS[i]] >= proj[RUNGS[i + 1]] - 1e-9
                  for i in range(len(RUNGS) - 1))
    metrics["schema_ladder_ordered"] = bool(ordered)
    if not ordered:
        print(f"WARNING: consistency ladder is not monotone: {proj}. "
              f"Model results are not interpretable as a consistency effect.")

    # --- A/B: schema acquisition and probe (once, for reporting) ---
    probe_model = _fresh_model(model_class, n_features, model_kwargs, epochs_per_trial)
    acq = _train_schema_to_criterion(
        probe_model, enc, decoder, base_seqs, schema_trials, epochs_per_trial,
        schema_criterion, criterion_window, n_probe_trials, noise_scale)
    n_acq = int(acq["trials"])
    metrics["schema_acquisition_trials"] = acq["trials"]
    metrics["schema_acquisition_reached"] = acq["reached"]
    metrics["schema_acquisition_criterion"] = schema_criterion
    series["schema_acquisition_curve"] = acq["curve"]
    print(f"  schema acquired in {n_acq} trial(s)"
          f"{'' if acq['reached'] else ' (CEILING -- criterion not met)'} "
          f"at {epochs_per_trial} epoch(s)/trial; base MRR {acq['curve'][-1]:.2f}")
    if not acq["reached"]:
        print(f"WARNING: schema did not reach {schema_criterion:.2f} within "
              f"{schema_trials} trials. relative_cost is nan for every rung; raise "
              f"schema_trials or epochs_per_trial.")
    probe = _schema_probe(probe_model, enc, decoder, base_seqs,
                          n_probe_trials, noise_scale)
    n_cat = len(categories)
    n_items = len(base.items)
    metrics["schema_probe_item_accuracy"] = probe["item_accuracy"]
    metrics["schema_probe_category_accuracy"] = probe["category_accuracy"]
    metrics["schema_probe_item_chance"] = 1.0 / n_items
    metrics["schema_probe_category_chance"] = 1.0 / n_cat
    metrics["schema_acquired"] = bool(
        probe["category_accuracy"] > 2.0 / n_cat
        and probe["category_accuracy"] >= probe["item_accuracy"]
    )
    if not metrics["schema_acquired"]:
        print(f"WARNING: schema probe failed "
              f"(category acc {probe['category_accuracy']:.2f} vs chance {1.0/n_cat:.2f}). "
              f"Consistency conditions below are not interpretable.")

    # --- C/D: one independent run per rung ---
    for rung, spec in zip(RUNGS, specs):
        np.random.seed(rng_seed)
        model = _fresh_model(model_class, n_features, model_kwargs, epochs_per_trial)
        # Exactly the staircased count, so every rung starts from the same
        # exposure the schema needed -- not from a fixed budget.
        _train_schema(model, enc, base_seqs, n_acq, epochs_per_trial)

        pre = {}
        for cat, words in base_seqs.items():
            model.reset_context()
            pre[cat] = mean_recall_rate(measure_recall_associative(
                model, words, enc, decoder, n_trials=n_probe_trials,
                noise_scale=noise_scale))
        base_mrr = float(np.mean(list(pre.values())))
        criterion = criterion_frac * base_mrr
        metrics[f"schema_{rung}_base_mrr"] = base_mrr
        metrics[f"schema_{rung}_criterion"] = criterion

        # New sequence: the host category's items, then the new item appended.
        # It must EXTEND rather than replace -- substituting for an existing item
        # would contradict a transition the base sequence already teaches, and
        # every rung would then take the same mechanical MRR hit regardless of
        # consistency, masking the effect being measured. The paper's
        # sparrowhawk is likewise added to the dataset, not swapped in for the
        # sparrow. The association is still arbitrary, placed in a familiar
        # structure: the Tse et al. flavour-place analogue.
        new_words = base_seqs[host_cat] + [spec["name"]]
        # What we TRAIN on. What we PROBE is always `new_words`, under both
        # protocols, so new_item_recall / prefix_recall / final_mrr keep the
        # same definition and stay comparable across them. Under "focused" the
        # prefix positions are simply not retrained, which is what makes their
        # decay readable as same-category interference rather than being
        # masked by rehearsal.
        train_words = (new_words if interference_protocol == "extended"
                       else [base_seqs[host_cat][-1], spec["name"]])
        curve: List[float] = []        # sequence-mean MRR -- diluted, see below
        new_curve: List[float] = []    # the one new transition: host[-1] -> new item
        prefix_curve: List[float] = [] # the already-known transitions in the same list
        for _ in range(new_item_trials):
            model.reset_context()
            _fit(model, enc.encode(train_words), epochs_per_trial)
            model.reset_context()
            rc = measure_recall_associative(
                model, new_words, enc, decoder, n_trials=n_probe_trials,
                noise_scale=noise_scale)
            curve.append(mean_recall_rate(rc))
            new_curve.append(float(rc[-1]))
            prefix_curve.append(float(np.mean(rc[1:-1])) if len(rc) > 2
                                else float("nan"))

        # Scored on the new transition alone. Scoring `curve` instead makes
        # trials-to-criterion uninterpretable: with N-1 scored transitions of
        # which N-2 are already learned, the sequence mean cannot fall below
        # (N-2)/(N-1), which for the default 7-item list is 0.833 -- above the
        # 0.75 x base_mrr criterion, so every rung "reaches criterion" on trial 1
        # whether or not the new item was ever acquired.
        ttc = _trials_to_criterion(new_curve, criterion, criterion_window)
        metrics[f"schema_{rung}_trials_to_criterion"] = ttc["trials"]
        metrics[f"schema_{rung}_criterion_reached"] = ttc["reached"]
        # Unit-free cost: censored on either side -> nan, never a number.
        metrics[f"schema_{rung}_relative_cost"] = (
            float(ttc["trials"] / n_acq)
            if ttc["reached"] and acq["reached"] and n_acq > 0 else float("nan"))
        metrics[f"schema_{rung}_new_item_recall"] = (
            float(new_curve[-1]) if new_curve else float("nan"))
        metrics[f"schema_{rung}_new_item_auc"] = (
            float(np.mean(new_curve)) if new_curve else float("nan"))
        metrics[f"schema_{rung}_prefix_recall"] = (
            float(prefix_curve[-1]) if prefix_curve else float("nan"))
        metrics[f"schema_{rung}_final_mrr"] = float(curve[-1]) if curve else float("nan")

        post, drops = {}, {"same": [], "sibling": [], "far": []}
        for cat, words in base_seqs.items():
            model.reset_context()
            post[cat] = mean_recall_rate(measure_recall_associative(
                model, words, enc, decoder, n_trials=n_probe_trials,
                noise_scale=noise_scale))
            drops[_relatedness(base, host_cat, cat)].append(pre[cat] - post[cat])
        for rel, vals in drops.items():
            metrics[f"schema_{rung}_interference_{rel}"] = (
                float(np.mean(vals)) if vals else float("nan"))

        series[rung] = {"learning_curve": curve,
                        "new_item_curve": new_curve,
                        "prefix_curve": prefix_curve,
                        "pre_mrr": pre, "post_mrr": post}
        print(f"  {rung:<10} proj={proj[rung]:.2f}  "
              f"trials={ttc['trials']:.0f}{'' if ttc['reached'] else '+'} "
              f"(x{metrics[f'schema_{rung}_relative_cost']:.2f} schema)  "
              f"new={metrics[f'schema_{rung}_new_item_recall']:.2f} "
              f"(auc {metrics[f'schema_{rung}_new_item_auc']:.2f})  "
              f"prefix={metrics[f'schema_{rung}_prefix_recall']:.2f}  "
              f"interference same/sib/far = "
              f"{metrics[f'schema_{rung}_interference_same']:+.2f}/"
              f"{metrics[f'schema_{rung}_interference_sibling']:+.2f}/"
              f"{metrics[f'schema_{rung}_interference_far']:+.2f}")

    # --- protocol provenance -------------------------------------------------
    # Neither of these reaches the model. `interference_valid` says whether the
    # interference columns above answer the question they appear to; the
    # identity flag says whether this arm could even be told which sequence a
    # pair belongs to (no arm is told, by any protocol -- see
    # _sequence_identity_available and audit pending item 18).
    metrics["schema_interference_valid"] = bool(interference_protocol == "focused")
    metrics["schema_sequence_identity_available"] = _sequence_identity_available(
        probe_model)
    if not metrics["schema_interference_valid"]:
        print("NOTE: interference_protocol='extended' -- learning the new item "
              "rehearses every base transition of the host category, so the "
              "`same` bucket measures rehearsal, not interference. Re-run with "
              "interference_protocol='focused' before quoting any "
              "schema_*_interference_* value.")

    # --- headline: does consistency predict acquisition speed? ---
    # Two readouts of the same question. `speed_corr` is the criterion-based one
    # and saturates whenever every rung is one-shot; `auc_corr` is continuous and
    # keeps ranging after that, so report both and prefer auc_corr when
    # `schema_at_ceiling` is set.
    # Over STRUCTURED_RUNGS only -- see that constant for why `duplicate` is out.
    xs = [proj[r] for r in STRUCTURED_RUNGS]
    ys = [metrics[f"schema_{r}_trials_to_criterion"] for r in STRUCTURED_RUNGS]
    aucs = [metrics[f"schema_{r}_new_item_auc"] for r in STRUCTURED_RUNGS]
    if len(set(xs)) > 1 and len(set(ys)) > 1:
        metrics["schema_consistency_speed_corr"] = float(np.corrcoef(xs, ys)[0, 1])
    else:
        metrics["schema_consistency_speed_corr"] = float("nan")
    if len(set(xs)) > 1 and len(set(np.round(aucs, 12))) > 1:
        metrics["schema_consistency_auc_corr"] = float(np.corrcoef(xs, aucs)[0, 1])
    else:
        metrics["schema_consistency_auc_corr"] = float("nan")
    metrics["schema_corr_rungs"] = "/".join(STRUCTURED_RUNGS)
    # `duplicate`, reported on its own: trials beyond the structured rungs' mean.
    # Positive = the exact-copy item was HARDER to acquire than novel ones.
    dup_ttc = metrics["schema_duplicate_trials_to_criterion"]
    metrics["schema_duplicate_excess_trials"] = (
        float(dup_ttc - np.mean(ys)) if metrics["schema_duplicate_criterion_reached"]
        and all(metrics[f"schema_{r}_criterion_reached"] for r in STRUCTURED_RUNGS)
        else float("nan"))
    # The weight-space headline. Predicted POSITIVE: consistent material induces
    # updates that reinforce (EI -> 1), inconsistent material updates that cancel
    # (EI -> 0). McClelland (2013) reports 0.93 vs 0.27 for the two extremes.

    # Dilution diagnostic: the floor the *sequence-mean* MRR cannot fall below,
    # and whether the criterion sits under it. Both true means a run scored on
    # `final_mrr` rather than `new_item_recall` is degenerate by arithmetic. Kept
    # as an explicit metric so the fix cannot silently regress.
    n_scored = len(base_seqs[host_cat])          # (N-1) transitions in the new list
    dilution_floor = (n_scored - 1) / n_scored if n_scored else float("nan")
    metrics["schema_dilution_floor"] = float(dilution_floor)
    metrics["schema_criterion_below_dilution_floor"] = bool(
        float(np.mean([metrics[f"schema_{r}_criterion"] for r in RUNGS])) <= dilution_floor)

    # Ceiling/floor are over ALL rungs (a state of the section); `resolved`
    # follows the rung set the correlation is computed over, so a spread that
    # exists only in `duplicate` cannot mark the consistency effect resolved.
    reached_all = [metrics[f"schema_{r}_criterion_reached"] for r in RUNGS]
    ys_all = [metrics[f"schema_{r}_trials_to_criterion"] for r in RUNGS]
    reached = [metrics[f"schema_{r}_criterion_reached"] for r in STRUCTURED_RUNGS]
    new_finals = [metrics[f"schema_{r}_new_item_recall"] for r in RUNGS]
    at_ceiling = all(reached_all) and all(y <= 1.0 for y in ys_all)
    at_floor = not any(reached_all)
    metrics["schema_at_ceiling"] = bool(at_ceiling)
    metrics["schema_at_floor"] = bool(at_floor)
    metrics["schema_resolved"] = bool(not at_ceiling and not at_floor
                                      and len(set(ys)) > 1)
    metrics["schema_auc_resolved"] = bool(
        len(set(np.round(aucs, 3))) > 1 and not all(np.isnan(aucs)))
    if at_ceiling:
        print(f"WARNING: at ceiling -- every rung acquires the new item on trial 1 "
              f"(probe item accuracy {probe['item_accuracy']:.2f}). This is now a "
              f"one-shot result, not a dilution artefact; read "
              f"schema_consistency_auc_corr, which still ranges. To recover "
              f"criterion range, shorten epochs_per_trial or raise criterion_frac "
              f"-- do NOT lengthen the host sequence, which only dilutes "
              f"final_mrr further.")
    elif at_floor:
        print(f"WARNING: at floor -- no rung reaches criterion in "
              f"{len(ys)} trials (final new-item recall "
              f"{'/'.join(f'{v:.2f}' for v in new_finals)}). Raise "
              f"new_item_trials, or the model cannot learn this task at all.")
    elif not metrics["schema_resolved"]:
        print("WARNING: rungs are not separated; the consistency effect is "
              "unresolved at this configuration.")

    return metrics, series, proj, host_cat


def run_schema_consistency(
    model_class: Type,
    model_kwargs: Optional[Dict[str, Any]] = None,
    branching: Sequence[int] = (2, 2, 6),
    features_per_node: int = 2,
    schema_trials: int = 40,
    schema_criterion: float = 0.75,
    epochs_per_trial: int = 1,
    new_item_trials: int = 15,
    criterion_frac: float = 0.75,
    criterion_window: int = 2,
    n_probe_trials: int = 1,
    noise_scale: float = 0.0,
    interference_protocol: str = "extended",
    seed: int = 0,
    n_seeds: int = 5,
    output_dir: str = "results",
    run_name: Optional[str] = None,
    run_dir: Optional[str] = None,
    metrics_filename: str = "metrics.json",
) -> Dict[str, Any]:
    """Run the schema-consistency protocol over ``n_seeds`` replicates.

    Protocol and every read-out: see :func:`_run_schema_consistency_once`.
    This wrapper only adds replication, following the suite convention
    (``spatial_disambiguation.guidance_sweep``, ``cognitive_phenomena``):

    * Replicate ``i`` runs with section seed ``seed + i`` -- this drives the
      ``augment()`` column draws for the ``within`` / ``across`` / ``random``
      rungs and any ``np.random`` use -- and, **only if the arm takes a
      ``seed`` kwarg**, model seed ``model_kwargs["seed"] + i``. An arm without
      a seed kwarg is deterministic given the data, so its model is identical
      across replicates and only the stimulus varies; ``metadata["model_seeded"]``
      records which case applied.
    * Every numeric metric is reported under its **original key as the mean
      over seeds** (so the scorecard, guards and tests read it unchanged) plus
      ``<key>_sd``, the s.d. over seeds (nan when ``n_seeds == 1``).
    * Every boolean guard is ``all()`` over seeds -- a guard passes only if it
      passed on every replicate -- plus ``<key>_frac``, the fraction that did.
    * ``series`` is replicate 0's (what the plot draws) plus
      ``series["per_seed"]`` with all of them.

    ``n_seeds=1`` reproduces the single-run behaviour exactly.
    """
    if int(n_seeds) < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds!r}")
    model_kwargs = dict(model_kwargs or {})
    model_seeded = "seed" in model_kwargs
    base_model_seed = int(model_kwargs["seed"]) if model_seeded else None

    per_metrics: List[Dict[str, Any]] = []
    per_series: List[Dict[str, Any]] = []
    proj = host_cat = None
    for i in range(int(n_seeds)):
        kw = dict(model_kwargs)
        if model_seeded:
            kw["seed"] = base_model_seed + i
        if n_seeds > 1:
            print(f"[schema_consistency] seed {i + 1}/{n_seeds}: section seed "
                  f"{seed + i}" + (f", model seed {kw['seed']}" if model_seeded else
                                   " (arm takes no seed; model identical across replicates)"))
        m, ser, proj, host_cat = _run_schema_consistency_once(
            model_class, kw, branching, features_per_node, schema_trials,
            schema_criterion, epochs_per_trial, new_item_trials, criterion_frac,
            criterion_window, n_probe_trials, noise_scale, interference_protocol,
            seed=seed + i)
        per_metrics.append(m)
        per_series.append(ser)

    # --- aggregate -----------------------------------------------------------
    metrics: Dict[str, Any] = {}
    for key, v0 in per_metrics[0].items():
        vals = [pm.get(key) for pm in per_metrics]
        if isinstance(v0, bool):
            metrics[key] = bool(all(bool(v) for v in vals))
            metrics[key + "_frac"] = float(np.mean([bool(v) for v in vals]))
        elif isinstance(v0, (int, float)) and not isinstance(v0, bool):
            arr = np.array([float(v) for v in vals], dtype=float)
            metrics[key] = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")
            metrics[key + "_sd"] = (float(np.nanstd(arr))
                                    if len(arr) > 1 and np.sum(np.isfinite(arr)) > 1
                                    else float("nan"))
        else:
            metrics[key] = v0
    series: Dict[str, Any] = dict(per_series[0])
    series["per_seed"] = per_series

    model_name = run_name or model_class.__name__
    if run_dir is None:
        run_dir = os.path.join(output_dir, model_name, "schema_consistency")
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    _plot(series, metrics, proj, model_name, run_dir,
          float(np.mean([metrics[f'schema_{r}_criterion'] for r in RUNGS])))

    results = {
        "metadata": {"model_name": model_name, "suite": "schema_consistency",
                     "branching": list(branching),
                     "features_per_node": features_per_node,
                     "schema_trials": schema_trials,
                     "schema_criterion": schema_criterion,
                     "epochs_per_trial": epochs_per_trial,
                     "new_item_trials": new_item_trials,
                     "criterion_frac": criterion_frac, "seed": seed,
                     "n_seeds": int(n_seeds),
                     "seeds": [seed + i for i in range(int(n_seeds))],
                     "model_seeded": model_seeded,
                     "model_seeds": ([base_model_seed + i for i in range(int(n_seeds))]
                                     if model_seeded else None),
                     "interference_protocol": interference_protocol,
                     "host_category": host_cat,
                     "model_kwargs": {k: str(v) for k, v in model_kwargs.items()}},
        "metrics": metrics,
        "series": series,
    }
    with open(os.path.join(run_dir, metrics_filename), "w") as f:
        json.dump(results, f, indent=2)
    return results


def _plot(series, metrics, proj, model_name, run_dir, criterion) -> None:
    """Two panels. Left: acquisition of the new transition per rung, mean over
    seeds with a +-1 s.d. band when ``series["per_seed"]`` is present. Right:
    interference by relatedness, mean with s.d. error bars. ``duplicate`` is
    drawn dashed with hollow markers and hatched bars: it is reported
    separately and excluded from the headline correlations (see
    ``STRUCTURED_RUNGS``). The suptitle carries the correlations and the rung
    set they were computed over, so the figure cannot be read as if
    ``duplicate`` were in them.
    """
    per_seed = series.get("per_seed") or [series]
    n_seeds = len(per_seed)

    def _band(key, rung):
        curves = [np.asarray(ps[rung][key], dtype=float) for ps in per_seed
                  if rung in ps and len(ps[rung].get(key, [])) > 0]
        if not curves:
            return None, None
        L = min(len(c) for c in curves)
        arr = np.stack([c[:L] for c in curves])
        return arr.mean(axis=0), (arr.std(axis=0) if n_seeds > 1 else None)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    for k, rung in enumerate(RUNGS):
        sep = rung not in STRUCTURED_RUNGS
        style = dict(ls="--", marker="s", mfc="none") if sep else dict(ls="-", marker="o")
        mean, sd = _band("new_item_curve", rung)
        if mean is None:
            continue
        x = np.arange(1, len(mean) + 1)
        label = f"{_RUNG_LABEL[rung]}  (proj={proj[rung]:.2f})"
        if sep:
            label += "  -- reported separately"
        ax.plot(x, mean, ms=4, color=f"C{k}", label=label, **style)
        if sd is not None:
            ax.fill_between(x, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                            color=f"C{k}", alpha=0.12, lw=0)
        # The diluted sequence-mean, faint, so the gap between what is scored
        # and the old read-out stays visible on the same axes.
        smean, _ = _band("learning_curve", rung)
        if smean is not None:
            ax.plot(np.arange(1, len(smean) + 1), smean, color=f"C{k}", lw=1,
                    alpha=0.25, ls=":")
    ax.axhline(criterion, ls="--", c="grey", lw=1)
    floor = metrics.get("schema_dilution_floor")
    if floor is not None and np.isfinite(floor):
        ax.axhline(floor, ls="-.", c="firebrick", lw=1, alpha=0.6)
        ax.text(1, floor + 0.015, "sequence-mean floor", fontsize=7,
                color="firebrick")
    ax.set_xlabel("training trial on the new material")
    ax.set_ylabel("recall of the new transition (dotted: sequence-mean MRR)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Acquisition by schema consistency"
                 + (f"  (mean +- s.d., {n_seeds} seeds)" if n_seeds > 1 else ""))
    ax.legend(fontsize=7.5)

    ax = axes[1]
    rels = ["same", "sibling", "far"]
    width = 0.2
    x = np.arange(len(rels))
    for k, rung in enumerate(RUNGS):
        vals = [metrics[f"schema_{rung}_interference_{r}"] for r in rels]
        errs = [metrics.get(f"schema_{rung}_interference_{r}_sd", float("nan")) for r in rels]
        errs = None if all(not np.isfinite(e) for e in errs) else [
            (e if np.isfinite(e) else 0.0) for e in errs]
        sep = rung not in STRUCTURED_RUNGS
        ax.bar(x + (k - 1.5) * width, vals, width, label=_RUNG_LABEL[rung],
               color=f"C{k}", yerr=errs, capsize=2 if errs else 0,
               hatch="//" if sep else None, edgecolor="k" if sep else None,
               lw=0.6 if sep else 0)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["same\ncategory", "sibling\ncategory", "far\ncategory"])
    ax.set_ylabel("MRR drop on base sequences")
    valid = metrics.get("schema_interference_valid", True)
    ax.set_title("Interference, graded by relatedness"
                 + ("" if valid else "  [INVALID: extended protocol rehearses `same`]"))
    ax.legend(fontsize=7.5)

    sc = metrics.get("schema_consistency_speed_corr", float("nan"))
    ac = metrics.get("schema_consistency_auc_corr", float("nan"))
    rungs = metrics.get("schema_corr_rungs", "/".join(STRUCTURED_RUNGS))
    ex = metrics.get("schema_duplicate_excess_trials", float("nan"))
    fig.suptitle(f"Schema consistency - {model_name}    |    "
                 f"speed_corr {sc:+.2f}, auc_corr {ac:+.2f} over {rungs}    |    "
                 f"duplicate excess {ex:+.1f} trials", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plots", "schema_consistency.png"), dpi=150)
    plt.close(fig)
