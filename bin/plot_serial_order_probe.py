#!/usr/bin/env python3
"""Draw the Serial-order failure figures from a saved ``serial_order_probe.json``.

`bin/probe_serial_order.py` keeps the decoded identity at every cued and rollout
step over the length grid and writes the three ordinal rows of Serial order
(establishment, binding_ordinal, unrolling) as a JSON side-car. Until now it
had no figure: its numbers reached the scorecard as SVG only, never the run
tree. This writes two PNGs beside the symbolic suite's plots so they can be
declared in ``MAIN_FIGURES``:

  serial_order_failures.png   (a) was order established -- order_given_item and
                              list_membership_rate against L, with chance;
                              (b) what a cued failure is -- correct /
                              transposition / intrusion per L;
                              (c) where transpositions land -- displacement
                              histogram, with locality and forward share.
  serial_order_unrolling.png  (a) cued span against raw-rollout span, the
                              unrolling_gap shaded; (b) exposure to cued vs
                              rollout criterion (the staircase), censored open;
                              (c) P(correct next | correct) vs P(correct next |
                              error) -- the cascade.

Arms with no cued failure get panel (b) all-correct and panel (c) marked
undefined: every binding_ordinal metric is a conditional on a failure.

    python bin/plot_serial_order_probe.py --results-dir results/zoo_capacity_run
"""
import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTCOMES = (("correct", "tab:blue"), ("transposition", "tab:orange"),
            ("other_list_intrusion", "tab:purple"),
            ("extra_list_intrusion", "tab:red"),
            # probe files written before the intrusion split (2026-09-25)
            ("intrusion", "tab:red"))


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def plot_failures(d, path):
    rows = d["rows"]
    sm = d.get("summary") or {}
    Ls = [r["length"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))

    # (a) establishment
    ax = axes[0]
    ax.plot(Ls, [r["list_membership_rate"] for r in rows], '-s', color="tab:grey",
            label="list membership (any studied item)")
    ax.plot(Ls, [r["order_given_item"] for r in rows], '-o', color="tab:blue",
            label="order | item (correct next, given in-list)")
    ax.plot(Ls, [1.0 / (L - 1) for L in Ls], ':', color="tab:blue", lw=0.9,
            label="chance for order | item")
    ax.axhline(0.75, ls="--", color="lightgrey", lw=0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(Ls)
    ax.set_xlabel("list length L")
    ax.set_ylabel("rate")
    ebl = sm.get("establishment_break_length")
    ax.set_title(f"Order established?  (break length = {ebl:g})" if _finite(ebl)
                 else "Order established?")
    ax.legend(fontsize=7.5, loc="lower left")

    # (b) outcome taxonomy per L, then each list of the blocked multi-list
    # condition after the last list was trained. The same model error is
    # labelled by which list its answer belongs to, so read the orange and
    # purple bars together across the gap.
    ax = axes[1]
    ml = d.get("multi_list") or {}
    cols = [(str(L), r) for L, r in zip(Ls, rows)]
    if ml.get("final"):
        tag = f"{ml['n_lists']}×{ml['list_len']}"
        cols += [(f"{tag}\n{'ABCDEFGH'[r['list_index']]}", r) for r in ml["final"]]
    xs = [i if i < len(rows) else i + 0.6 for i in range(len(cols))]
    bottom = np.zeros(len(cols))
    for kind, color in OUTCOMES:
        frac = np.array([(r["outcomes"].get(kind, 0) / max(r["n_probes"], 1))
                         for _, r in cols])
        if not frac.any():
            continue
        ax.bar(xs, frac, bottom=bottom, color=color, width=0.6,
               label=kind.replace("_", " "))
        bottom += frac
    ax.set_xticks(xs)
    ax.set_xticklabels([c for c, _ in cols], fontsize=8)
    ax.set_xlabel("single list: length L   |   blocked lists: after the last one"
                  if ml.get("final") else "list length L")
    ax.set_ylabel("fraction of cued probes")
    ax.set_ylim(0, 1.0)
    n_fail = sm.get("n_cued_failures", 0)
    ax.set_title(f"What a cued failure is  ({n_fail:,} failures / "
                 f"{sm.get('n_cued_probes', 0):,} probes)")
    ax.legend(fontsize=8, loc="lower left")

    # (c) displacement histogram, pooled over L
    ax = axes[2]
    pooled = {}
    for r in rows:
        for k, v in (r.get("displacements") or {}).items():
            pooled[int(k)] = pooled.get(int(k), 0) + int(v)
    if pooled:
        ds = sorted(pooled)
        tot = sum(pooled.values())
        ax.bar(ds, [pooled[x] / tot for x in ds],
               color=["tab:green" if abs(x) == 1 else "tab:orange" for x in ds])
        ax.axvline(0, color="black", lw=0.8)
        span = max(ds) - min(ds)
        step = 1 if span <= 16 else 5
        ax.set_xticks([x for x in range(min(ds) - min(ds) % step, max(ds) + 1, step)])
        ax.set_xlabel("displacement d = study pos(decoded) − pos(target)")
        ax.set_ylabel("share of transpositions")
        loc, asym = sm.get("transposition_locality"), sm.get("transposition_asymmetry")
        ax.set_title(f"Transposition displacement  (|d|=1: {loc:.2f}, fwd: {asym:.2f})"
                     if _finite(loc) and _finite(asym) else "Transposition displacement")
    else:
        ax.text(0.5, 0.5, "no cued failures observed\nbinding_ordinal is undefined, not 1.00",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
                color="tab:grey")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Where transpositions land")

    fig.suptitle(f"Serial order, failures of order — {d.get('model', '')}  "
                 f"(probe: cued recall at a fixed {d.get('epochs', '?')} epochs per list, "
                 f"the arm's registry default; decoded against "
                 f"{d.get('vocab_size', 'the studied')} items)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_unrolling(d, path):
    rows = d["rows"]
    st = d.get("staircase") or []
    sm = d.get("summary") or {}
    Ls = [r["length"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))

    # (a) cued vs rollout span, read where cued recall first reaches criterion
    # (the staircase); the fixed-exposure rows are drawn faint for comparison.
    # `unrolling_gap` in the summary is the mean of gap_at_criterion, so the
    # shaded wedge is the scored quantity, not the fixed-budget one.
    ax = axes[0]
    ax.plot(Ls, [r["cued_span"] for r in rows], '-', color="tab:blue", alpha=0.25, lw=1)
    ax.plot(Ls, [r["rollout_span"] for r in rows], '-', color="tab:red", alpha=0.25, lw=1,
            label=f"faint: at fixed exposure ({d.get('epochs', '?')} epochs)")
    ok = [t for t in st if t.get("cued_reached") and t.get("span_cued_at_criterion")]
    if ok:
        oL = [t["length"] for t in ok]
        cs = [t["span_cued_at_criterion"] for t in ok]
        rs = [t["span_rollout_at_criterion"] or 0 for t in ok]
        ax.plot(oL, cs, '-o', color="tab:blue", label="cued span, at cued criterion")
        ax.plot(oL, rs, '-x', color="tab:red", label="rollout span (raw), same exposure")
        ax.fill_between(oL, rs, cs, color="tab:red", alpha=0.12)
    miss = [t["length"] for t in st if not t.get("cued_reached")]
    half = 0.4 * (min(np.diff(sorted(Ls))) if len(Ls) > 1 else 1)
    for L in miss:
        ax.axvspan(L - half, L + half, color="lightgrey", alpha=0.5)
    if miss:
        ax.plot([], [], 's', color="lightgrey", label="cued never reached criterion")
    ax.plot(Ls, [L - 1 for L in Ls], ':', color="grey", lw=0.9, label="ceiling (L−1)")
    ax.set_xticks(Ls)
    ax.set_xlabel("list length L")
    ax.set_ylabel("memory span")
    gap = sm.get("unrolling_gap")
    ax.set_title(f"Associations present, unrolling collapses  (gap = {gap:.2f})"
                 if _finite(gap) else "Cued vs rollout span")
    ax.legend(fontsize=7.5, loc="upper left")

    # (b) staircase: exposure to cued vs rollout criterion
    ax = axes[1]
    if st:
        budget = max(t.get("budget") or 512 for t in st)
        sL = [t["length"] for t in st]
        for key, reached_key, color, lab in (
                ("e_cued", "cued_reached", "tab:blue", "cued reaches criterion"),
                ("e_rollout", "rollout_reached", "tab:red", "rollout reaches criterion")):
            ys = [t[key] if t[key] else budget for t in st]
            hit = [i for i, t in enumerate(st) if t.get(reached_key)]
            miss = [i for i, t in enumerate(st) if not t.get(reached_key)]
            ax.plot(sL, ys, '-', color=color, lw=1.1, zorder=1)
            if hit:
                ax.plot([sL[i] for i in hit], [ys[i] for i in hit], 'o', color=color,
                        label=lab, zorder=2)
            if miss:
                ax.plot([sL[i] for i in miss], [ys[i] for i in miss], 'o', mfc="white",
                        mec=color, label=f"{lab.split(' ')[0]}: censored at budget", zorder=2)
        ax.axhline(budget, ls="--", color="lightgrey", lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(sL)
        ax.set_xlabel("list length L")
        ax.set_ylabel("presentations to criterion (log)")
        ratio = sm.get("unrolling_exposure_ratio")
        ax.set_title(f"What unrolling costs  (E_roll / E_cued = {ratio:.2f})"
                     if _finite(ratio) else "What unrolling costs")
        ax.legend(fontsize=7.5, loc="upper left")
    else:
        ax.set_visible(False)

    # (c) cascade conditionals
    ax = axes[2]
    cr = [r for r in rows if _finite(r.get("cascade_recovery_rate"))]
    if cr:
        x = np.arange(len(cr))
        w = 0.38
        ax.bar(x - w / 2, [r["p_correct_after_correct"] for r in cr], w,
               color="tab:blue", label="P(next correct | this correct)")
        ax.bar(x + w / 2, [r["cascade_recovery_rate"] for r in cr], w,
               color="tab:red", label="P(next correct | this wrong)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["length"]) for r in cr])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("list length L (only lengths where rollout broke)")
        ax.set_ylabel("conditional per-step accuracy")
        rec, rat = sm.get("cascade_recovery_rate"), sm.get("cascade_conditional_ratio")
        ax.set_title(f"Is an error terminal?  (recovery {rec:.2f}, ratio "
                     f"{rat:.1f}×)" if _finite(rec) and _finite(rat)
                     else f"Is an error terminal?  (recovery {rec:.2f})" if _finite(rec)
                     else "Is an error terminal?")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "rollout never broke on this grid:\nnothing to condition on",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
                color="tab:grey")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Is an error terminal?")

    fig.suptitle(f"Serial order, autonomous unrolling — {d.get('model', '')}  "
                 f"(probe: staircase, one epoch at a time, spans read where cued recall "
                 f"first reaches 0.95; cascade at the fixed {d.get('epochs', '?')}-epoch rows)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--arm", default="*",
                    help="Class-name directory to plot (default: every arm).")
    a = ap.parse_args()
    n = 0
    for p in sorted(glob.glob(os.path.join(a.results_dir, a.arm, "serial_order_probe.json"))):
        with open(p) as f:
            d = json.load(f)
        if not d.get("rows"):
            continue
        plots = os.path.join(os.path.dirname(p), "symbolic", "plots")
        os.makedirs(plots, exist_ok=True)
        plot_failures(d, os.path.join(plots, "serial_order_failures.png"))
        plot_unrolling(d, os.path.join(plots, "serial_order_unrolling.png"))
        print(f"wrote {plots}/serial_order_{{failures,unrolling}}.png")
        n += 1
    print(f"{n} arm(s) plotted")


if __name__ == "__main__":
    main()
