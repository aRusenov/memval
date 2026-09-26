#!/usr/bin/env python
"""Capacity curve on sparse codes: cued recall and raw autoregressive unroll.

The simplest question the suite can ask an arm -- *how much can it hold, and can
it walk the chain unaided?* -- on the sparse coding regime rather than the dense
Gaussian one. Two conditions, one x-axis (number of items ingested), so they are
read against each other:

  A  single_sequence  100 sparse items with NO category structure, one chain,
                      streamed in blocks of 10. Block b re-presents the previous
                      block's last item as its first cue, so exactly the 10 new
                      transitions are trained and the chain stays ONE sequence.
  B  multi_sequence   10 sparse categories x 10 sparse items. Each category is
                      its own sequence, ingested one at a time with no weight
                      reset -- so this asks for multiple sequences held at once,
                      which A never does.

Both report, at every checkpoint N = 10, 20, ... 100 items:

  cued recall     one clean cue per stored position, one step, averaged over
                  every item ingested so far (`measure_recall_associative`).
                  Position 0 excluded (it is the cue).
  raw unroll      autoregressive rollout with `feedback_mode="raw"`: the
                  prediction is fed back verbatim, so magnitude drift and error
                  compound. HARNESS-CONTROLLED -- the harness imposes this
                  protocol on every arm rather than using the arm's own
                  `recall()`; that is what "raw" means and any reading of the
                  figure has to say so.

In A the unroll is one chain of length N from item 0. In B it is one chain per
category, each from its own first item, averaged.

Both readouts are ALSO kept per position, not only as their mean over positions.
The mean is positionally blind: an MRR of 0.2 over a 9-step chain is equally
consistent with a clean 2-item prefix and with 20% scattered across the chain,
and those are different claims about an arm. Two things are therefore recorded
beside each mean:

  <key>_curve     P(correct) at each position, index 0 being the cue (forced to
                  1 by the probes and excluded from the mean).
  unroll_prefix   length of the CONTIGUOUS run of fully-correct positions from
                  position 1 -- how far the chain is actually walked before the
                  first break. Read it against the mean: where the mean exceeds
                  the prefix, the surplus is off-prefix hits, which for a
                  collapsed rollout are usually a frozen attractor coinciding
                  with the target at the one position whose index matches it,
                  not recall.

The per-position curves are drawn in a second figure,
`sparse_capacity_positions.png`.

B is also run a second time with ``category_core = 0`` and its unroll drawn as
a control line, because otherwise the condition confounds two things: holding
several sequences at once (what B is for) and holding a sequence whose own
items overlap (what the category core does). The control separates them.

Decoding always ranks against the FULL 100-item vocabulary, not just the items
ingested so far, so chance is a constant 1/100 across the whole x-axis and the
curve is not flattered by having fewer distractors early on.

Exposure is a FIXED budget per ingestion call (the arm's registry `n_epochs`
unless `--epochs`), not criterion-referenced: the question here is how much is
retained at equal exposure, so equalising exposure is the point. That makes the
curve a capacity-at-fixed-cost reading, and an arm that simply needs more passes
will look worse than one that does not -- state it when comparing arms.

    python bin/sparse_capacity_benchmark.py --model hopfield --out results/sparse_capacity
    python bin/sparse_capacity_benchmark.py --model hopfield --active 20 --category-core 0.8

Writes <out>/<ClassName>/sparse_capacity/plots/sparse_capacity_curves.png and
<out>/<ClassName>/sparse_capacity/sparse_capacity.json -- the same shape as a
suite directory, so a capacity run tree (`bin/run_capacity_arm.sh`, which runs
this as a step) shows it in the results index beside the suites.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

from run_benchmark import MODEL_REGISTRY  # noqa: E402

from memval.benchmarks.symbolic_pipeline import (  # noqa: E402
    _get_model_kwargs, mean_recall_rate, measure_recall_associative,
    measure_recall_autoregressive)
from memval.encoders import SparseSymbolicEncoder, SymbolicDecoder  # noqa: E402


def build_model(model_class, default_kwargs: Dict[str, Any], encoder, epochs: int, seed: int):
    kw = _get_model_kwargs(default_kwargs, epochs)
    if "seed" in kw:
        kw["seed"] = seed
    if "encoder" in model_class.__init__.__code__.co_varnames:
        return model_class(encoder=encoder, n_features=encoder.embedding_dim, **kw)
    return model_class(n_features=encoder.embedding_dim, **kw)


def _prefix_len(curve: np.ndarray) -> float:
    """Contiguous fully-correct positions from position 1.

    "Fully correct" is every trial correct at that position, so at the default
    `n_trials=1` this is just the run of 1s. A partial rate breaks the run: the
    prefix is the length of chain the arm walks *reliably*, and anything less
    than reliable is what the mean is already reporting.
    """
    n = 0
    for v in np.asarray(curve, dtype=float)[1:]:
        if v >= 1.0:
            n += 1
        else:
            break
    return float(n)


def _score(model, words, enc, dec, n_trials, noise) -> Dict[str, Any]:
    """Cued recall and raw unroll over one word list, per position and as means.

    The curves are kept, not just their means: see the module docstring on why
    an unroll mean cannot be read as a prefix length.
    """
    model.reset_context()
    cued_curve = measure_recall_associative(
        model, words, enc, dec, n_trials=n_trials, noise_scale=noise)
    model.reset_context()
    unroll_curve = measure_recall_autoregressive(
        model, words, enc, dec, n_trials=n_trials, noise_scale=noise,
        feedback_mode="raw")
    return {"cued": mean_recall_rate(cued_curve),
            "unroll": mean_recall_rate(unroll_curve),
            "cued_curve": [float(v) for v in cued_curve],
            "unroll_curve": [float(v) for v in unroll_curve],
            "unroll_prefix": _prefix_len(unroll_curve)}


def run_single_sequence(model_class, kwargs, *, n_items, block, dim, active, epochs,
                        seed, n_trials, noise) -> Dict[str, Any]:
    """A: one chain of `n_items` sparse items, streamed in blocks."""
    words = [f"w{i:03d}" for i in range(n_items)]
    enc = SparseSymbolicEncoder(words, embedding_dim=dim, active=active, seed=seed)
    dec = SymbolicDecoder(enc)
    model = build_model(model_class, kwargs, enc, epochs, seed)

    points = []
    for start in range(0, n_items, block):
        # Include the previous block's last item so its outgoing transition is
        # trained: the 100 items are one sequence, not ten disjoint ones.
        lo = max(0, start - 1)
        model.fit_sequence(enc.encode(words[lo:start + block]))
        seen = words[:start + block]
        s = _score(model, seen, enc, dec, n_trials, noise)
        points.append({"n_items": len(seen), **s})
    return {"points": points, "geometry": enc.geometry()}


def run_multi_sequence(model_class, kwargs, *, n_cats, per_cat, dim, active, core,
                       epochs, seed, n_trials, noise) -> Dict[str, Any]:
    """B: `n_cats` sparse categories of `per_cat` items, one sequence each."""
    vocab = {f"c{c:02d}_i{i:02d}": f"c{c:02d}" for c in range(n_cats) for i in range(per_cat)}
    enc = SparseSymbolicEncoder(vocab, embedding_dim=dim, active=active,
                                category_core=core, seed=seed)
    dec = SymbolicDecoder(enc)
    seqs = [[f"c{c:02d}_i{i:02d}" for i in range(per_cat)] for c in range(n_cats)]
    model = build_model(model_class, kwargs, enc, epochs, seed)

    points = []
    for c in range(n_cats):
        model.fit_sequence(enc.encode(seqs[c]))
        per = [_score(model, seqs[j], enc, dec, n_trials, noise) for j in range(c + 1)]
        points.append({
            "n_items": (c + 1) * per_cat,
            "n_sequences": c + 1,
            "cued": float(np.mean([p["cued"] for p in per])),
            "unroll": float(np.mean([p["unroll"] for p in per])),
            # Every sequence is `per_cat` long, so the per-position curves are
            # commensurable and average directly.
            "cued_curve": np.mean([p["cued_curve"] for p in per], axis=0).tolist(),
            "unroll_curve": np.mean([p["unroll_curve"] for p in per], axis=0).tolist(),
            "unroll_prefix": float(np.mean([p["unroll_prefix"] for p in per])),
            "per_sequence": per,
        })
    return {"points": points, "geometry": enc.geometry()}


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Seed mean and sd of each curve."""
    xs = [p["n_items"] for p in runs[0]["points"]]
    out = {"n_items": xs}
    for key in ("cued", "unroll", "unroll_prefix"):
        M = np.array([[p[key] for p in r["points"]] for r in runs], dtype=float)
        out[f"{key}_mean"] = M.mean(axis=0).tolist()
        out[f"{key}_sd"] = M.std(axis=0).tolist()
    # Chain length grows with the checkpoint in condition A, so the curves are
    # ragged ACROSS checkpoints -- but at a given checkpoint every seed ran the
    # same length, so each one averages on its own.
    for key in ("cued", "unroll"):
        out[f"{key}_curve_mean"] = [
            np.mean([r["points"][i][f"{key}_curve"] for r in runs], axis=0).tolist()
            for i in range(len(xs))]
    return out


def plot(conditions: List[Dict[str, Any]], path: str, title: str, chance: float,
         independent_ylim: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(conditions), figsize=(7.2 * len(conditions), 5.0),
                             squeeze=False)
    for ax, cond in zip(axes[0], conditions):
        agg = cond["agg"]
        x = np.array(agg["n_items"], dtype=float)
        cm, cs = np.array(agg["cued_mean"]), np.array(agg["cued_sd"])
        um, us = np.array(agg["unroll_mean"]), np.array(agg["unroll_sd"])

        ax.errorbar(x, cm, yerr=cs, fmt="-o", color="tab:blue", ms=5, lw=1.8,
                    capsize=3, label="cued recall (one clean cue, one step)")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("items ingested")
        ax.set_ylabel("mean cued recall", color="tab:blue")
        ax.tick_params(axis="y", colors="tab:blue")
        ax.axhline(chance, ls=":", color="grey", lw=1.0)
        ax.text(x[0], chance + 0.015, f"chance = {chance:g}", fontsize=7, color="grey")
        ax.set_xticks(x)
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        ax2.errorbar(x, um, yerr=us, fmt="--s", color="tab:red", ms=5, lw=1.8,
                     capsize=3, label="autoregressive raw unroll")
        ctrl = cond.get("control")
        if ctrl is not None:
            ax2.errorbar(np.array(ctrl["n_items"], dtype=float), ctrl["unroll_mean"],
                         yerr=ctrl["unroll_sd"], fmt=":^", color="tab:red", ms=4,
                         lw=1.3, alpha=0.55, capsize=2,
                         label="raw unroll, control: no category core")
        if independent_ylim:
            hi = max(0.05, float(np.nanmax(um + us)) * 1.25)
            ax2.set_ylim(-0.03 * hi / 1.05, hi)
            note = "INDEPENDENT scale"
        else:
            ax2.set_ylim(-0.03, 1.05)
            note = "same 0-1 scale as left"
        ax2.set_ylabel(f"mean raw unroll recall  ({note})", color="tab:red")
        ax2.tick_params(axis="y", colors="tab:red")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
        ax.set_title(cond["title"], fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_positions(conditions: List[Dict[str, Any]], path: str, title: str,
                   chance: float) -> None:
    """P(correct) against POSITION, one line per checkpoint.

    The companion to the capacity curves: those collapse each rollout to its
    mean over positions, and this is what that mean was made of. A curve that
    steps down once and stays down is a prefix; a curve that is flat and low is
    scattered, and scattered hits out of a raw rollout are usually a collapsed
    attractor rather than recall.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(conditions), figsize=(7.2 * len(conditions), 5.0),
                             squeeze=False)
    for ax, cond in zip(axes[0], conditions):
        agg = cond["agg"]
        xs, curves = agg["n_items"], agg["unroll_curve_mean"]
        cmap = plt.get_cmap("viridis")
        for i, (n, cv) in enumerate(zip(xs, curves)):
            shade = cmap(i / max(1, len(xs) - 1))
            pos = np.arange(1, len(cv))          # position 0 is the cue, always 1
            ax.plot(pos, np.asarray(cv)[1:], "-o", color=shade, ms=3.5, lw=1.4,
                    label=f"{n} items")
        ctrl = cond.get("control")
        if ctrl is not None:
            cv = np.asarray(ctrl["unroll_curve_mean"][-1])
            ax.plot(np.arange(1, len(cv)), cv[1:], "--", color="tab:red", lw=1.6,
                    alpha=0.7, label=f"control, no category core ({xs[-1]} items)")
        # Annotations in axes coordinates, so they land in free space rather
        # than on top of whichever curve happens to pass through a data point.
        from matplotlib.transforms import blended_transform_factory
        ax.axhline(chance, ls=":", color="grey", lw=1.0)
        ax.text(0.995, chance + 0.015, f"chance = {chance:g}", fontsize=7, color="grey",
                ha="right", transform=blended_transform_factory(ax.transAxes, ax.transData))
        pref = agg["unroll_prefix_mean"][-1]
        ax.axvline(pref + 0.5, ls="-.", color="black", lw=1.0, alpha=0.6)
        ax.text(0.5, 0.60,
                f"mean reliable prefix at {xs[-1]} items: {pref:.2f}\n"
                f"(mean over positions: {agg['unroll_mean'][-1]:.3f})",
                fontsize=8, color="black", ha="center", va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("position in the chain (0 = cue, excluded)")
        ax.set_ylabel("P(correct) at this position, raw unroll")
        ax.set_title(cond["title"], fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="hopfield", choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--out", default="results/sparse_capacity")
    ap.add_argument("--n-items", type=int, default=100)
    ap.add_argument("--block", type=int, default=10, help="items per ingestion call in A")
    ap.add_argument("--n-cats", type=int, default=10)
    ap.add_argument("--per-cat", type=int, default=10)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--active", type=int, default=10, help="active units per item (k of d)")
    ap.add_argument("--category-core", type=float, default=0.5,
                    help="fraction of an item's units shared with its category (condition B)")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=None,
                    help="fixed passes per ingestion call; default = the arm's registry n_epochs")
    ap.add_argument("--n-trials", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--independent-ylim", action="store_true",
                    help="autoscale the unroll axis to its own range. Off by default: both "
                         "readouts are recall fractions on 0-1, and stretching one of them "
                         "makes a collapsed curve look like a healthy one.")
    args = ap.parse_args()

    spec = MODEL_REGISTRY[args.model]
    model_class, default_kwargs = spec["class"], dict(spec["default_kwargs"])
    class_name = model_class.__name__
    epochs = int(args.epochs if args.epochs is not None else default_kwargs.get("n_epochs", 1))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    # Suite-shaped layout, so bin/build_results_index.py picks the figure up as
    # the `sparse_capacity` pseudo-suite of the arm: <out>/<Class>/sparse_capacity/
    # {plots/sparse_capacity_curves.png, sparse_capacity.json}.
    out_dir = os.path.join(args.out, class_name, "sparse_capacity")
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    t0 = time.time()

    common = dict(dim=args.dim, active=args.active, epochs=epochs,
                  n_trials=args.n_trials, noise=args.noise)
    runs_a = [run_single_sequence(model_class, default_kwargs, n_items=args.n_items,
                                  block=args.block, seed=sd, **common) for sd in seeds]
    runs_b = [run_multi_sequence(model_class, default_kwargs, n_cats=args.n_cats,
                                 per_cat=args.per_cat, core=args.category_core,
                                 seed=sd, **common) for sd in seeds]
    # Control for B: same protocol, no shared units inside a sequence.
    runs_b0 = [run_multi_sequence(model_class, default_kwargs, n_cats=args.n_cats,
                                  per_cat=args.per_cat, core=0.0,
                                  seed=sd, **common) for sd in seeds]

    conditions = [
        {"key": "A_single_sequence",
         "title": f"A · one chain of {args.n_items} sparse items\n"
                  f"(k={args.active} of d={args.dim}, no categories, blocks of {args.block})",
         "agg": aggregate(runs_a), "geometry": runs_a[0]["geometry"],
         "runs": [r["points"] for r in runs_a]},
        {"key": "B_multi_sequence",
         "title": f"B · {args.n_cats} sequences x {args.per_cat} sparse items\n"
                  f"(k={args.active} of d={args.dim}, category core {args.category_core:g})",
         "agg": aggregate(runs_b), "geometry": runs_b[0]["geometry"],
         "control": aggregate(runs_b0), "control_geometry": runs_b0[0]["geometry"],
         "runs": [r["points"] for r in runs_b]},
    ]

    for cond in conditions:
        a = cond["agg"]
        print(f"\n{cond['key']}   (mean of {len(seeds)} seeds)")
        print("  items   cued recall        raw unroll         reliable prefix")
        for i, n in enumerate(a["n_items"]):
            print(f"  {n:5d}   {a['cued_mean'][i]:.3f} ± {a['cued_sd'][i]:.3f}      "
                  f"{a['unroll_mean'][i]:.3f} ± {a['unroll_sd'][i]:.3f}      "
                  f"{a['unroll_prefix_mean'][i]:.2f} ± {a['unroll_prefix_sd'][i]:.2f}")
        g = cond["geometry"]
        print(f"  geometry: within {g['within_cos_measured']:.3f} (law {g['within_cos_law']:.3f})  "
              f"between {g['between_cos_measured']:.3f} (law {g['between_cos_law']:.3f})")
        if cond.get("control"):
            cc = cond["control"]
            print("  control (no category core), raw unroll: "
                  + "  ".join(f"{n}:{v:.3f}" for n, v in zip(cc["n_items"], cc["unroll_mean"])))

    chance = 1.0 / (args.n_items if args.n_items >= args.n_cats * args.per_cat
                    else args.n_cats * args.per_cat)
    png = os.path.join(out_dir, "plots", "sparse_capacity_curves.png")
    plot(conditions, png,
         f"Sparse-code capacity: cued recall vs raw autoregressive unroll "
         f"({class_name}, mean of {len(seeds)} seeds, {epochs} passes per ingestion)",
         chance, args.independent_ylim)
    png_pos = os.path.join(out_dir, "plots", "sparse_capacity_positions.png")
    plot_positions(conditions, png_pos,
                   f"Sparse-code capacity, raw unroll BY POSITION "
                   f"({class_name}, mean of {len(seeds)} seeds, {epochs} passes per ingestion)",
                   chance)
    with open(os.path.join(out_dir, "sparse_capacity.json"), "w") as f:
        json.dump({"model": args.model, "class": class_name, "seeds": seeds,
                   "epochs": epochs, "args": vars(args), "chance": chance,
                   "conditions": conditions}, f, indent=1)
    print(f"\nwrote {png}\n      {png_pos}\n"
          f"      {os.path.join(out_dir, 'sparse_capacity.json')}\n"
          f"{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
