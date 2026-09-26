#!/usr/bin/env python
"""Serial-order read-outs that need the decoded identity, not just a boolean.

`measure_recall_associative` and `measure_recall_autoregressive` both decode a
word at every step and then discard which word it was, keeping only whether it
was right. That single loss is what blocks three of the four Serial-order rows in
`docs/capacity_coverage_audit.md` sec 5: `memory_span` zeroes at the first error
and cannot tell a transposition from an omission from an intrusion.

This probe re-runs the same cued and rollout protocols over the same length grid
and keeps the identities, so the specified metrics can be computed:

  establishment      list_membership_rate, order_given_item, establishment_break_length
  binding_ordinal    order_error_fraction, transposition_locality,
                     transposition_asymmetry, intrusion_rate
  unrolling          unrolling_gap, cascade_recovery_rate,
                     cascade_conditional_ratio, rollout_margin_at_break

Displacement is d = (study position of the decoded word) - (position of the true
target), so d = 0 is correct and |d| = 1 is a neighbour swap. A decode outside the
probed list is an intrusion and has no d. Intrusions come in two kinds, as in the
recall literature:

  other_list_intrusion   a word from ANOTHER studied list (the prior-list intrusion
                         of free recall; here it can also come from a later list).
                         Only the multi-list condition can produce one.
  extra_list_intrusion   a word that was never studied at all.

**Material.** Every condition decodes against ONE fixed vocabulary of VOCAB_SIZE
items, larger than any studied material, so an extra-list intrusion is possible
everywhere and chance is the same constant across conditions. The earlier probe
rebuilt the encoder from the studied words alone, which made both intrusion kinds
and a non-unit list_membership_rate structurally impossible. Items are synthetic
labels in one category at category_variance 0.2 -- data/vocab.json has only 12
fruits -- and the encoder's per-word draws do not depend on the label, so item i
is the same vector the old fruit list gave position i. Within-list cosine is
therefore constant across L; only the load changes.

Conditions:
  single list   L in LENGTHS (10, 20, 30). The registry budget is sized to bring
                ONE 10-item list to criterion, so L = 10 is the calibrated point
                and 20 / 30 add load at the same exposure.
  3 x 10        three disjoint 10-item lists trained blocked (A, then B, then C)
                at the same per-list budget, then every list probed. Same item
                count as L = 30, one third of the chain length.

**This is a probe, not the fix.** The real change (audit S-O2) is to return the
identities from the two shared helpers so every section gets them and these
become suite metrics rather than a side-car. This duplicates their decode loop
deliberately, to avoid touching a function seven sections depend on.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.benchmarks.symbolic_pipeline import (
    measure_recall_associative, measure_recall_autoregressive,
    mean_recall_rate, memory_span)
from run_benchmark import MODEL_REGISTRY

LENGTHS = (10, 20, 30)
N_LISTS, MULTI_LIST_LEN = 3, 10
#: Decode vocabulary shared by every condition. Must exceed both max(LENGTHS) and
#: N_LISTS * MULTI_LIST_LEN, or the largest condition has no unstudied item left
#: to intrude with.
VOCAB_SIZE = 40
OUTCOME_KINDS = ("correct", "transposition", "other_list_intrusion",
                 "extra_list_intrusion")

#: Arm under probe. Set by main() from --model; the probe is per-arm, and the
#: staircase's whole point is that E_cued / E_roll are self-referenced, so an
#: arm-specific epoch budget must not leak in from AHN's defaults.
MODEL_NAME = "hopfield"


def _fit_one_epoch(model, X):
    """Advance ``model`` by exactly one pass over ``X``.

    See ``staircase``'s docstring: the EP family ignores a call-level
    ``epochs=``, so the count has to be pinned on the instance as well.
    """
    if hasattr(model, "n_epochs"):
        model.n_epochs = 1
    model.fit_sequence(X, epochs=1)


def _make_model(lr, seed):
    """Build the arm under probe with its registry defaults.

    ``learning_rate`` is overridden by the probe's own ``lr`` only when the arm
    actually exposes one. ``n_epochs`` is deliberately NOT forwarded: both entry
    points here drive exposure themselves (``probe`` passes ``epochs=`` to
    ``fit_sequence``, ``staircase`` steps one epoch at a time), so the registry
    default would otherwise be applied twice.
    """
    entry = MODEL_REGISTRY[MODEL_NAME]
    kwargs = {k: v for k, v in entry["default_kwargs"].items() if k != "n_epochs"}
    if "learning_rate" in kwargs:
        kwargs["learning_rate"] = lr
    kwargs["seed"] = seed
    return entry["class"](n_features=100, **kwargs)


def _material(seed):
    """The fixed decode vocabulary: VOCAB_SIZE synthetic items in one category.

    Built once per seed and shared by every condition, so an item is the same
    vector whichever list it is studied in, and the decoder always ranks against
    the full vocabulary -- studied or not.
    """
    pool = [f"item{i:02d}" for i in range(VOCAB_SIZE)]
    enc = SymbolicEncoder({w: "item" for w in pool}, embedding_dim=100,
                          category_variance=0.2, seed=seed)
    return pool, enc, SymbolicDecoder(enc)


def _classify(decoded, words, target_idx, other_lists=()):
    """(outcome, displacement, source list).

    displacement is None for either intrusion kind; source is the index of the
    studied list an other-list intrusion came from, else None.
    """
    if decoded == words[target_idx]:
        return "correct", 0, None
    if decoded in words:
        return "transposition", words.index(decoded) - target_idx, None
    for j, other in other_lists:
        if decoded in other:
            return "other_list_intrusion", None, j
    return "extra_list_intrusion", None, None


def _cued(model, words, enc, dec, n_trials, noise_scale, rng, other_lists=()):
    """One-step cued recall, keeping every decoded identity."""
    n = len(words)
    hits = np.zeros(n)
    outcomes, disps, sources = Counter(), Counter(), Counter()
    for _ in range(n_trials):
        for i in range(n - 1):
            model.current_t = i
            cue = enc.encode([words[i]])[0] + rng.normal(0, noise_scale, enc.embedding_dim)
            w = dec.decode(model.predict_next(cue, current_context=np.array([1.0])),
                           top_k=1)[0]
            kind, d, src = _classify(w, words, i + 1, other_lists)
            outcomes[kind] += 1
            if kind == "correct":
                hits[i + 1] += 1
            elif kind == "transposition":
                disps[d] += 1
            elif kind == "other_list_intrusion":
                sources[src] += 1
    hits[0] = n_trials
    return hits / n_trials, outcomes, disps, sources


def _outcome_rates(outcomes, disps):
    """The establishment / binding_ordinal read-outs off one outcome tally."""
    total = sum(outcomes.values())
    failures = total - outcomes["correct"]
    in_list = outcomes["correct"] + outcomes["transposition"]
    n_trans = outcomes["transposition"]
    fwd, bwd = disps.get(1, 0), disps.get(-1, 0)
    nan = float("nan")
    return dict(
        n_probes=int(total), n_failures=int(failures),
        outcomes={k: int(outcomes.get(k, 0)) for k in OUTCOME_KINDS},
        displacements={str(k): int(v) for k, v in sorted(disps.items())},
        list_membership_rate=in_list / total if total else nan,
        order_given_item=outcomes["correct"] / in_list if in_list else nan,
        order_error_fraction=n_trans / failures if failures else nan,
        transposition_locality=(fwd + bwd) / n_trans if n_trans else nan,
        transposition_asymmetry=fwd / (fwd + bwd) if (fwd + bwd) else nan,
        extra_list_intrusion_rate=outcomes["extra_list_intrusion"] / total if total else nan,
        other_list_intrusion_rate=outcomes["other_list_intrusion"] / total if total else nan,
        intrusion_rate=((outcomes["extra_list_intrusion"] + outcomes["other_list_intrusion"])
                        / total if total else nan),
    )


def _rollout(model, words, enc, dec, n_trials, noise_scale, rng):
    """Free rollout, keeping per-step correctness so the conditionals can be formed."""
    n = len(words)
    hits = np.zeros(n)
    pairs = Counter()          # (correct_at_k, correct_at_k+1)
    # Margin ranks the target against the FULL decode vocabulary, exactly as the
    # decoder does; ranking against the list alone would ignore the unstudied
    # competitors that an extra-list intrusion comes from.
    Xn = enc.embeddings / np.linalg.norm(enc.embeddings, axis=1, keepdims=True)
    tgt = [enc.word_to_idx[w] for w in words]
    margins = np.zeros(n - 1)
    for _ in range(n_trials):
        model.reset_context()
        cur = enc.encode([words[0]])[0] + rng.normal(0, noise_scale, enc.embedding_dim)
        prev_ok = None
        for i in range(n - 1):
            model.current_t = i
            pred = model.predict_next(cur, current_context=np.array([1.0]))
            w = dec.decode(pred, top_k=1)[0]
            ok = (w == words[i + 1])
            if ok:
                hits[i + 1] += 1
            if prev_ok is not None:
                pairs[(prev_ok, ok)] += 1
            prev_ok = ok
            pn = pred / max(float(np.linalg.norm(pred)), 1e-12)
            sims = Xn @ pn
            margins[i] += float(sims[tgt[i + 1]] - np.max(np.delete(sims, tgt[i + 1])))
            cur = pred
    hits[0] = n_trials
    return hits / n_trials, pairs, margins / n_trials


def staircase(lengths=LENGTHS, n_trials=20, criterion=0.95, budget=512,
              lr=0.1, seed=42, noise_scale=0.05):
    """Give every arm the exposure it needs, and report the cost in its own units.

    The suite resolves a section's epoch count as CLI > model default > section
    fallback, and the model defaults in MODEL_REGISTRY differ by 300x across the
    taxonomy (hopfield 1, EP arms 100-300). Any span measured at "the default" is
    therefore measured at a different point on every arm's learning curve, and
    the cross-arm comparison is void.

    Two self-referenced quantities fix that, neither of which needs a budget
    chosen from outside:

      E_cued  smallest exposure at which CUED recall reaches criterion. This is
              the point at which "the associations are present" becomes true, so
              it is where the paragraph-11 clause's antecedent holds and where
              the gap should be read.
      E_roll  smallest exposure at which ROLLOUT reaches the same criterion.

      unrolling_gap_at_criterion  1 - span(rollout)/span(cued), measured at
                                  E_cued. How much unrolling costs at the moment
                                  the associations exist.
      unrolling_exposure_ratio    E_roll / E_cued. How much extra exposure buys
                                  it back. 1.0 = unrolling comes free with the
                                  associations; large = a sample-efficiency gap;
                                  censored = a genuine collapse.

    Both are ratios of an arm against itself, so they are comparable across arms
    with wildly different convergence rates — the same reason the schema section
    sets its criterion as a fraction of the model's own base MRR rather than an
    absolute.

    ``fit_sequence`` is incremental for every arm the suite runs (the schema and
    continual sections already rely on it), so the staircase trains one epoch at
    a time rather than retraining from scratch at each checkpoint.
    
    **One epoch per step means forcing it.** Nine of the thirteen registered arms
    -- the whole EP family -- declare ``fit_sequence(self, sequence_data,
    **kwargs)`` and loop over ``self.n_epochs``, so a call-level ``epochs=1``
    lands in ``**kwargs`` and is discarded; the step silently trains
    ``self.n_epochs`` (100 for original_eqprop) epochs instead of one. That makes
    E_cued and E_roll units of 100 epochs for those arms while remaining units of
    1 for the rest, which is exactly the cross-arm confound this staircase
    exists to remove -- and it costs ~100x the wall-clock. Every ``fit`` closure
    in symbolic_pipeline.py already guards against it by assigning ``n_epochs``
    before the call; this does the same. No-op for the four arms that honour the
    kwarg.
    """
    pool, enc, dec = _material(seed)

    out = []
    for L in lengths:
        words = pool[:L]
        X = enc.encode(words)
        np.random.seed(0)
        m = _make_model(lr, seed)
        e_cued = e_roll = None
        span_c = span_r = None
        ep = 0
        while ep < budget and (e_cued is None or e_roll is None):
            _fit_one_epoch(m, X)
            ep += 1
            m.reset_context()
            c = measure_recall_associative(m, words, enc, dec, n_trials=n_trials,
                                           noise_scale=noise_scale)
            m.reset_context()
            r = measure_recall_autoregressive(m, words, enc, dec, n_trials=n_trials,
                                              noise_scale=noise_scale,
                                              feedback_mode="raw")
            if e_cued is None and mean_recall_rate(c) >= criterion:
                e_cued, span_c, span_r = ep, memory_span(c), memory_span(r)
            if e_roll is None and mean_recall_rate(r) >= criterion:
                e_roll = ep
        out.append(dict(
            length=L, e_cued=e_cued, e_rollout=e_roll,
            cued_reached=e_cued is not None, rollout_reached=e_roll is not None,
            budget=budget, criterion=criterion,
            span_cued_at_criterion=span_c, span_rollout_at_criterion=span_r,
            gap_at_criterion=(1.0 - span_r / span_c) if span_c else float("nan"),
            exposure_ratio=(e_roll / e_cued) if (e_cued and e_roll) else float("inf"),
        ))
    return out


def probe(lengths=LENGTHS, n_trials=30, epochs=300, lr=0.1, seed=42, noise_scale=0.05):
    pool, enc, dec = _material(seed)

    rows = []
    for L in lengths:
        words = pool[:L]
        X = enc.encode(words)
        np.random.seed(0)
        m = _make_model(lr, seed)
        # Same pinning as the staircase: the EP family discards a call-level
        # `epochs=`, so it has to be set on the instance too or this trains at
        # the class default rather than at the requested exposure.
        if hasattr(m, "n_epochs"):
            m.n_epochs = epochs
        m.fit_sequence(X, epochs=epochs)

        rng = np.random.default_rng(11)
        cued_curve, outcomes, disps, _ = _cued(m, words, enc, dec, n_trials,
                                               noise_scale, rng)
        m.reset_context()
        roll_curve, pairs, margins = _rollout(m, words, enc, dec, n_trials,
                                              noise_scale, rng)

        cued_span, roll_span = memory_span(cued_curve), memory_span(roll_curve)
        cc, ce = pairs[(True, True)], pairs[(False, True)]
        n_after_ok = pairs[(True, True)] + pairs[(True, False)]
        n_after_err = pairs[(False, True)] + pairs[(False, False)]
        p_ok_after_ok = cc / n_after_ok if n_after_ok else float("nan")
        p_ok_after_err = ce / n_after_err if n_after_err else float("nan")

        rows.append(dict(
            length=L, words=words,
            cued_curve=[float(v) for v in cued_curve],
            rollout_curve=[float(v) for v in roll_curve],
            cued_span=int(cued_span), rollout_span=int(roll_span),
            **_outcome_rates(outcomes, disps),
            p_correct_after_correct=p_ok_after_ok,
            cascade_recovery_rate=p_ok_after_err,
            rollout_margin_at_break=float(margins[roll_span]) if roll_span < len(margins)
                                    else float(margins[-1]),
            step_margins=[float(v) for v in margins],
        ))

    def _m(key):
        vs = [r[key] for r in rows if r[key] == r[key]]
        return float(np.mean(vs)) if vs else float("nan")

    stair = staircase(lengths=lengths, seed=seed, noise_scale=noise_scale, lr=lr)
    gaps = [t["gap_at_criterion"] for t in stair if t["gap_at_criterion"] == t["gap_at_criterion"]]
    ratios_x = [t["exposure_ratio"] for t in stair if np.isfinite(t["exposure_ratio"])]
    all_reached = all(t["rollout_reached"] for t in stair)
    ratios = [r["p_correct_after_correct"] / r["cascade_recovery_rate"]
              for r in rows
              if r["cascade_recovery_rate"] and r["cascade_recovery_rate"] == r["cascade_recovery_rate"]]
    ok = [r["length"] for r in rows if r["order_given_item"] >= 0.75]
    n_fail = sum(r["n_failures"] for r in rows)
    summary = {
        # Read before any binding_ordinal number below. Every one of them is a
        # conditional on a cued failure, so with no failures they are undefined
        # rather than zero -- the arm sits above the instrument's range.
        "failures_observed": bool(n_fail),
        "n_cued_failures": int(n_fail),
        "n_cued_probes": int(sum(r["n_probes"] for r in rows)),
        "list_membership_rate": _m("list_membership_rate"),
        "order_given_item": _m("order_given_item"),
        "establishment_break_length": float(max(ok)) if ok else 0.0,
        "order_error_fraction": _m("order_error_fraction"),
        "transposition_locality": _m("transposition_locality"),
        "transposition_asymmetry": _m("transposition_asymmetry"),
        "intrusion_rate": _m("intrusion_rate"),
        # Single-list rows can only intrude from outside the study set, so this
        # is the whole of intrusion_rate above; other-list intrusions live in
        # the multi_list block.
        "extra_list_intrusion_rate": _m("extra_list_intrusion_rate"),
        # Measured at each length's own cued criterion, not at whatever epoch
        # count the registry happens to carry for this arm.
        "unrolling_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "unrolling_exposure_ratio": (float(np.mean(ratios_x)) if ratios_x
                                     else float("inf")),
        "rollout_criterion_reached": bool(all_reached),
        "cascade_recovery_rate": _m("cascade_recovery_rate"),
        "cascade_conditional_ratio": float(np.mean(ratios)) if ratios else float("inf"),
        "rollout_margin_at_break": _m("rollout_margin_at_break"),
    }
    return rows, summary, stair


def multi_list(n_lists=N_LISTS, list_len=MULTI_LIST_LEN, n_trials=30, epochs=1,
               lr=0.1, seed=42, noise_scale=0.05):
    """Several disjoint lists trained blocked on one model, then each probed.

    This is the only condition in which an other-list intrusion can occur: a cue
    from list A answered with an item studied in list B or C. Every list gets the
    same fixed per-list budget as the single-list rows (the registry's, sized on
    one 10-item list), trained in order A, B, C exactly as `multiple_sequences`
    trains its blocked condition: `n_epochs` pinned on the instance, then
    `fit_sequence`, no reset between lists.

    Each list is probed twice with the full outcome taxonomy:
      immediate  right after its own training, before the next list -- what was
                 acquired, with only EARLIER lists able to intrude;
      final      after the last list -- what survived, with every other list able
                 to intrude. Intrusion source is recorded relative to the probed
                 list (-1 = the list before it, +1 = the list after it).
    """
    pool, enc, dec = _material(seed)
    lists = [pool[k * list_len:(k + 1) * list_len] for k in range(n_lists)]
    np.random.seed(0)
    m = _make_model(lr, seed)
    if hasattr(m, "n_epochs"):
        m.n_epochs = epochs

    def _others(k, upto):
        return [(j, lists[j]) for j in range(upto) if j != k]

    def _probe_list(k, upto):
        m.reset_context()
        rng = np.random.default_rng(11 + k)
        curve, oc, disps, src = _cued(m, lists[k], enc, dec, n_trials, noise_scale,
                                      rng, _others(k, upto))
        m.reset_context()
        roll, _, _ = _rollout(m, lists[k], enc, dec, n_trials, noise_scale, rng)
        return dict(
            list_index=k, cued_mrr=mean_recall_rate(curve),
            cued_span=int(memory_span(curve)), rollout_span=int(memory_span(roll)),
            cued_curve=[float(v) for v in curve],
            rollout_curve=[float(v) for v in roll],
            **_outcome_rates(oc, disps),
            intrusion_sources={str(j - k): int(v) for j, v in sorted(src.items())},
        )

    immediate = []
    for k, words in enumerate(lists):
        m.fit_sequence(enc.encode(words), epochs=epochs)
        immediate.append(_probe_list(k, k + 1))
    final = [_probe_list(k, n_lists) for k in range(n_lists)]

    def _pooled(block):
        oc, ds, src = Counter(), Counter(), Counter()
        for r in block:
            oc.update(r["outcomes"])
            ds.update({int(d): v for d, v in r["displacements"].items()})
            src.update(r["intrusion_sources"])
        out = _outcome_rates(oc, ds)
        out["intrusion_sources"] = dict(sorted(src.items(), key=lambda kv: int(kv[0])))
        return out

    return dict(n_lists=n_lists, list_len=list_len, epochs_per_list=epochs,
                lists=lists, immediate=immediate, final=final,
                summary=dict(immediate=_pooled(immediate), final=_pooled(final),
                             retained_mrr=[f["cued_mrr"] for f in final],
                             acquired_mrr=[i["cued_mrr"] for i in immediate]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="hopfield", choices=list(MODEL_REGISTRY),
                    help="Arm to probe (default: hopfield, the AHN reference).")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Exposure for the FIXED-epoch rows (establishment, "
                         "binding_ordinal). Defaults to the arm's own "
                         "MODEL_REGISTRY n_epochs, which is the same operating "
                         "point the suites use -- registry defaults span 1..300 "
                         "across the taxonomy, so a single shared number would "
                         "put every arm at a different place on its learning "
                         "curve. The staircase rows (unrolling) ignore this: "
                         "they are self-referenced by construction.")
    a = ap.parse_args()

    global MODEL_NAME
    MODEL_NAME = a.model
    if a.epochs is None:
        a.epochs = int(MODEL_REGISTRY[a.model]["default_kwargs"].get("n_epochs", 1))
        print(f"[probe] --epochs defaulted to {a.model}'s registry n_epochs = {a.epochs}")

    rows, summary, stair = probe(n_trials=a.n_trials, epochs=a.epochs)
    ml = multi_list(n_trials=a.n_trials, epochs=a.epochs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(probe="serial_order",
                       model=MODEL_REGISTRY[a.model]["class"].__name__,
                       epochs=a.epochs, n_trials=a.n_trials,
                       lengths=list(LENGTHS), vocab_size=VOCAB_SIZE,
                       rows=rows, staircase=stair, multi_list=ml,
                       summary=summary), f, indent=2)

    def _line(tag, r):
        o, n = r["outcomes"], max(r["n_probes"], 1)
        print(f"{tag:>9} {o['correct']/n:>6.3f} {o['transposition']/n:>6.3f} "
              f"{o['other_list_intrusion']/n:>6.3f} {o['extra_list_intrusion']/n:>6.3f} "
              f"{r['order_given_item']:>9.2f} {r['transposition_locality']:>6.2f} "
              f"{r['transposition_asymmetry']:>6.2f} {r['cued_span']:>5} {r['rollout_span']:>5}")

    hdr = (f"{'':>9} {'corr':>6} {'trans':>6} {'othL':>6} {'extL':>6} {'ord|item':>9} "
           f"{'local':>6} {'asym':>6} {'cued':>5} {'roll':>5}")
    print(f"single list, {a.epochs} epochs, decode vocabulary {VOCAB_SIZE}:")
    print(hdr)
    for r in rows:
        _line(f"L={r['length']}", r)
    print(f"\n{ml['n_lists']} x {ml['list_len']} blocked, {ml['epochs_per_list']} "
          f"epochs per list:")
    print(hdr)
    for phase in ("immediate", "final"):
        for r in ml[phase]:
            _line(f"{'ABC'[r['list_index']]} {phase[:5]}", r)
    print(f"  intrusion source (probed list +offset), final: "
          f"{ml['summary']['final']['intrusion_sources']}")
    print("\nstaircase to criterion (each arm judged against its own convergence):")
    print(f"  {'L':>3} {'E_cued':>7} {'E_roll':>7} {'ratio':>7} {'gap@Ec':>7}")
    for t in stair:
        er = str(t["e_rollout"]) if t["rollout_reached"] else f'>{t["budget"]}'
        rr = f'{t["exposure_ratio"]:.1f}' if np.isfinite(t["exposure_ratio"]) else "cens."
        print(f"  {t['length']:>3} {str(t['e_cued']):>7} {er:>7} {rr:>7} "
              f"{t['gap_at_criterion']:>7.2f}")
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k:<28} {v:.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
