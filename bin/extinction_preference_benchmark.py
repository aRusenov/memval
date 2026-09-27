#!/usr/bin/env python
"""Run the adaptive extinction-preference protocol on one arm, across seeds.

    python bin/extinction_preference_benchmark.py --model ahn --out results/extinction_preference

Writes ``<out>/<ClassName>/extinction_preference.png`` and ``.json``. See
``memval/benchmarks/extinction_preference.py`` for the protocol and the sign
convention: every stage score is relative to the arm extinguished in that
stage; negative = preference flees it (naturalistic), positive = follows it.
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

from run_benchmark import MODEL_REGISTRY, resolve_model_name  # noqa: E402

from memval.benchmarks.extinction_preference import ARMS, ExtinctionPreferenceBenchmark  # noqa: E402
from memval.generators.t_maze_extinction import TMazeExtinctionGenerator  # noqa: E402

STAGES = ("acquisition", "extinction_1", "extinction_2")


def run_seed(model_class, kwargs, seed, args) -> Dict[str, Any]:
    task = TMazeExtinctionGenerator(seed=seed).generate(seed=seed)
    kw = dict(kwargs); kw["seed"] = seed; kw["n_epochs"] = args.epochs
    if args.learning_rate is not None and "learning_rate" in kw:
        kw["learning_rate"] = args.learning_rate
    model = model_class(n_features=task["n_features"], **kw)
    bench = ExtinctionPreferenceBenchmark(
        acquisition_pairs=args.acquisition_pairs, extinction_trials=args.extinction_trials,
        second_stage=args.second_stage, nudge_max=args.nudge_max,
        nudge_points=args.nudge_points, reward_noise=args.reward_noise,
        measure_savings=not args.no_savings, savings_criterion=args.savings_criterion,
        seed=seed)
    end_on = "left" if seed % 2 == 0 else "right"
    return bench.evaluate(model, task, epochs=args.epochs, end_on=end_on)


def _series(runs, stage, key, sub=None):
    """(trials, matrix seeds x trials) of curves[stage][*][key] (or [key][sub])."""
    rows = []
    for r in runs:
        recs = r["curves"][stage]
        rows.append([(rec[key][sub] if sub else rec[key]) for rec in recs])
    n = min(len(x) for x in rows)
    M = np.array([x[:n] for x in rows], dtype=float)
    return np.array([rec["trial"] for rec in runs[0]["curves"][stage][:n]]), M


def plot(runs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    n = len(runs)

    # --- (0,0) preference toward the extinguished arm, per stage ------------
    ax = axes[0, 0]
    off = 0
    for stage, col in (("extinction_1", "tab:red"), ("extinction_2", "tab:purple")):
        t, P = _series(runs, stage, "pref_ext")
        for row in P:
            ax.plot(t + off, row, color=col, alpha=0.18, lw=0.8)
        ax.errorbar(t + off, P.mean(0), yerr=P.std(0), fmt="-o", color=col, ms=4, capsize=2,
                    label=f"{stage.replace('_', ' ')} (arm extinguished in that stage)")
        off += t[-1] + 1
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("extinction trial (stage 2, then stage 3)")
    ax.set_ylabel("preference toward the extinguished arm\n(+1 = commits to it, −1 = commits to the other)")
    ax.set_title("Free rollout from the stem")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.25)

    # --- (0,1) fraction choosing extinguished arm + graded bias --------------
    ax = axes[0, 1]; ax2 = ax.twinx()
    off = 0
    for stage, col in (("extinction_1", "tab:red"), ("extinction_2", "tab:purple")):
        t, C = _series(runs, stage, "chose_ext")
        _, B = _series(runs, stage, "bias_ext")
        ax.plot(t + off, C.mean(0), "-s", color=col, ms=4, label=f"{stage}: fraction of seeds choosing it")
        ax2.errorbar(t + off, B.mean(0), yerr=B.std(0), fmt=":o", color=col, ms=3, alpha=0.7,
                     capsize=2, label=f"{stage}: psychometric bias toward it")
        off += t[-1] + 1
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("fraction of seeds committing to the extinguished arm")
    ax2.axhline(0, color="grey", lw=0.6, ls="--")
    ax2.set_ylabel("psychometric bias toward extinguished arm\n(nudge units; saturates at ±nudge_max)")
    ax.set_xlabel("extinction trial")
    ax.set_title("Binary choice across seeds, and the graded bias")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="center left")
    ax.grid(alpha=0.25)

    # --- (0,2) signed stage scores -------------------------------------------
    ax = axes[0, 2]
    keys = [("extinction_1_delta_pref_ext", "Δ preference\n(ext 1)"),
            ("extinction_1_delta_bias_ext", "Δ bias\n(ext 1)"),
            ("extinction_2_delta_pref_ext", "Δ preference\n(ext 2)"),
            ("extinction_2_delta_bias_ext", "Δ bias\n(ext 2)")]
    vals = [np.array([r["metrics"][k] for r in runs], dtype=float) for k, _ in keys]
    x = np.arange(len(keys))
    ax.bar(x, [v.mean() for v in vals], yerr=[v.std() for v in vals], capsize=4,
           color=["tab:red", "tab:red", "tab:purple", "tab:purple"], alpha=0.75)
    for xi, v in zip(x, vals):
        ax.scatter(np.full(len(v), xi) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)),
                   v, color="k", s=9, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=8)
    ax.set_ylabel("change over the stage, toward the extinguished arm")
    ax.set_title("Signed stage score: < 0 flees the extinguished arm,\n> 0 follows it, ≈ 0 unmoved", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # --- (1,0) reward prediction: extinguished vs other ------------------------
    def content_series(stage, field):
        E, O = [], []
        for r in runs:
            arm = r["metrics"][f"{stage}_arm"]
            recs = r["curves"][stage]
            E.append([rec["content"][arm][field] for rec in recs])
            O.append([rec["content"]["right" if arm == "left" else "left"][field] for rec in recs])
        n_ = min(len(e) for e in E)
        t = np.array([rec["trial"] for rec in runs[0]["curves"][stage][:n_]])
        return t, np.array([e[:n_] for e in E]), np.array([o[:n_] for o in O])

    for ax, field, lab in ((axes[1, 0], "reward_pred", "predicted reward in goal zone / gain"),
                           (axes[1, 1], "odour_retained", "predicted cheese odour along arm / original")):
        off = 0
        for stage, col in (("extinction_1", "tab:red"), ("extinction_2", "tab:purple")):
            t, E, O = content_series(stage, field)
            ax.errorbar(t + off, E.mean(0), yerr=E.std(0), fmt="-o", color=col, ms=4, capsize=2,
                        label=f"{stage}: extinguished arm")
            ax.errorbar(t + off, O.mean(0), yerr=O.std(0), fmt="--^", color=col, ms=4, capsize=2,
                        alpha=0.55, label=f"{stage}: other arm")
            off += t[-1] + 1
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("extinction trial"); ax.set_ylabel(lab)
        ax.set_title("Content of the trace (teacher-forced along the original route)", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # --- (1,2) savings -----------------------------------------------------------
    ax = axes[1, 2]
    a0 = np.array([r["metrics"].get("savings_acquisition_trials") or np.nan for r in runs], dtype=float)
    a1 = np.array([r["metrics"].get("savings_trials_to_reacquire") or np.nan for r in runs], dtype=float)
    if np.isfinite(a0).any():
        xs = np.arange(n)
        ax.bar(xs - 0.2, a0, 0.4, label="acquisition: pairs to reward criterion", color="tab:blue")
        ax.bar(xs + 0.2, a1, 0.4, label="re-acquisition after two extinctions", color="tab:green")
        ratio = np.nanmean(a1 / a0)
        ax.set_title(f"Savings: mean re-acquisition / acquisition = {ratio:.2f}\n(< 1 latent trace survived, ≈ 1 erased)", fontsize=10)
        ax.set_xlabel("seed"); ax.set_ylabel("presentations to reward criterion")
        ax.legend(fontsize=7)
    else:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ahn", type=resolve_model_name, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--out", default="results/extinction_preference")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--epochs", type=int, default=3, help="passes per presentation (reversal section uses 3)")
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="override the registry learning rate (savings needs acquisition to take >1 pair)")
    ap.add_argument("--acquisition-pairs", type=int, default=12)
    ap.add_argument("--extinction-trials", type=int, default=12)
    ap.add_argument("--second-stage", default="preferred", choices=["preferred", "other"])
    ap.add_argument("--nudge-max", type=float, default=1.0)
    ap.add_argument("--nudge-points", type=int, default=21)
    ap.add_argument("--reward-noise", type=float, default=1.0)
    ap.add_argument("--savings-criterion", type=float, default=0.8,
                    help="fraction of the arm's own end-of-acquisition reward prediction")
    ap.add_argument("--no-savings", action="store_true")
    args = ap.parse_args()

    spec = MODEL_REGISTRY[args.model]
    cls, kwargs = spec["class"], dict(spec["default_kwargs"])
    seeds = [int(s) for s in args.seeds.split(",")]
    out = os.path.join(args.out, cls.__name__); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    runs = []
    for sd in seeds:
        r = run_seed(cls, kwargs, sd, args); runs.append(r)
        m = r["metrics"]
        print(f"seed {sd:2d}: acq end_on={m['acquisition_end_on']:<5} preferred={m['acquisition_preferred_arm']:<5} | "
              f"ext1 {m['extinction_1_arm']:<5} Δpref {m['extinction_1_delta_pref_ext']:+.2f} Δbias {m['extinction_1_delta_bias_ext']:+.2f} "
              f"reward {m['extinction_1_ext_reward_pred_start']:.2f}→{m['extinction_1_ext_reward_pred_end']:.2f} "
              f"odour {m['extinction_1_ext_odour_retained_start']:.2f}→{m['extinction_1_ext_odour_retained_end']:.2f} | "
              f"ext2 {m['extinction_2_arm']:<5} Δpref {m['extinction_2_delta_pref_ext']:+.2f} Δbias {m['extinction_2_delta_bias_ext']:+.2f}"
              + (f" | savings {m.get('savings_acquisition_trials')}→{m.get('savings_trials_to_reacquire')}" if not args.no_savings else "")
              + ("  DIVERGED" if m["diverged"] else ""))
    def agg(k):
        v = np.array([r["metrics"][k] for r in runs], dtype=float); return v.mean(), v.std()
    nd = sum(r["metrics"]["diverged"] for r in runs)
    print(f"\nSTAGE SCORES (mean ± sd over seeds; negative = preference flees the extinguished arm; {nd}/{len(runs)} seeds diverged)")
    for st in ("extinction_1", "extinction_2"):
        dp, dps = agg(f"{st}_delta_pref_ext"); db, dbs = agg(f"{st}_delta_bias_ext")
        c0, _ = agg(f"{st}_chose_ext_start"); c1, _ = agg(f"{st}_chose_ext_end")
        r0, _ = agg(f"{st}_ext_reward_pred_start"); r1, _ = agg(f"{st}_ext_reward_pred_end")
        o0, _ = agg(f"{st}_ext_odour_retained_start"); o1, _ = agg(f"{st}_ext_odour_retained_end")
        print(f"  {st}: Δpref_ext {dp:+.3f}±{dps:.3f}  Δbias_ext {db:+.3f}±{dbs:.3f}  "
              f"P(choose ext) {c0:.2f}→{c1:.2f}  reward_pred(ext) {r0:.2f}→{r1:.2f}  odour(ext) {o0:.2f}→{o1:.2f}")
    if not args.no_savings:
        a0 = np.array([r["metrics"]["savings_acquisition_trials"] or np.nan for r in runs], float)
        a1 = np.array([r["metrics"]["savings_trials_to_reacquire"] or np.nan for r in runs], float)
        print(f"  savings: acquisition {np.nanmean(a0):.1f} pairs, re-acquisition {np.nanmean(a1):.1f} presentations, ratio {np.nanmean(a1/a0):.2f}")
    tag = f"{args.second_stage}" + (f"_lr{args.learning_rate:g}_ep{args.epochs}" if args.learning_rate is not None else "")
    png = os.path.join(out, f"extinction_preference_{tag}.png")
    plot(runs, png, f"Adaptive extinction–preference ({cls.__name__}, {len(seeds)} seeds, {args.epochs} passes/presentation, "
                    f"stage 3 = {args.second_stage} arm)")
    with open(os.path.join(out, f"extinction_preference_{tag}.json"), "w") as f:
        json.dump({"model": args.model, "class": cls.__name__, "args": vars(args), "seeds": seeds,
                   "runs": [{"metrics": r["metrics"],
                             "curves": {s: [{k: v for k, v in rec.items() if k != "psychometric"} for rec in recs]
                                        for s, recs in r["curves"].items()}} for r in runs]}, f, indent=1)
    print(f"\nwrote {png}\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
