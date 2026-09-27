#!/usr/bin/env python
"""Extinction timeline: recall of A and B across train A -> train B -> extinguish one.

One x-axis (presentation), two traces (A = circles, B = squares), three phases,
drawn twice: once extinguishing B (the most recently trained arm) and once
extinguishing A. Several candidate y-axes are drawn as rows so the most
informative one can be chosen:

  choice margin       cos(choice-point prediction, this arm) - cos(., other arm);
                      graded, signed, A = -B by construction
  free rollout        fraction of free-rollout steps (from the stem) on this arm
  route from entry    rollout from this arm's first step: fraction of steps on it
  reward in rollout   reward predicted in the goal zone of that arm-entry rollout
  reward (forced)     reward predicted teacher-forced along the original route
  odour (forced)      cheese odour predicted along the arm / original

    python bin/extinction_timeline.py --model ahn --learning-rate 0.05 --epochs 1
"""
from __future__ import annotations

import argparse, json, os, sys, time
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from run_benchmark import MODEL_REGISTRY, resolve_model_name  # noqa: E402
from memval.benchmarks.extinction_preference import (  # noqa: E402
    ExtinctionPreferenceBenchmark, arm_rollout, choice_margin, choice_margin_ceiling, other)
from memval.generators.t_maze_extinction import TMazeExtinctionGenerator as G  # noqa: E402

VARIANTS = [
    ("choice_margin", "choice-point margin, rollout\n(toward this arm)", (-1.05, 1.05)),
    ("choice_margin_forced", "choice-point margin, forced stem\n(toward this arm)", (-1.05, 1.05)),
    ("free_rollout", "free rollout from stem:\nfraction of steps on this arm", (-0.05, 1.05)),
    ("trace_fidelity", "trace fidelity from the fork\n(1 = exact, 0 = as wrong as other arm)", (-0.05, 1.05)),
    ("trace_error", "trace error from the fork\n(arena units)", (-0.02, 0.6)),
    ("route", "rollout from arm entry:\nfraction of steps on this arm", (-0.05, 1.05)),
    ("reward_in_rollout", "reward reached in\narm-entry rollout / gain", (-0.1, 1.1)),
    ("reward_entry", "reward anticipated entering\nthe goal zone / gain", (-0.1, 1.1)),
    ("reward_forced", "reward predicted across\nthe goal zone / gain", (-0.1, 1.1)),
    ("odour_forced", "cheese odour predicted,\nteacher-forced / original", (-0.1, 1.1)),
]
ARM = {"A": "left", "B": "right"}

#: Trace fidelity an arm must reach on its OWN training phase before its
#: extinction curves mean anything. Below it the arm never learned the route,
#: so a flat extinction line is a floor, not a result.
ACQ_GUARD = 0.25


def other_lab(lab: str) -> str:
    return "A" if lab == "B" else "B"


def probe(bench, model, task) -> Dict[str, Dict[str, float]]:
    out = {}
    preds = bench._rollout(model, task)
    _, fl, fr = bench._preference(preds, task)
    for lab, arm in ARM.items():
        ro = arm_rollout(model, task, arm)
        ct = bench._content(model, task, arm)
        out[lab] = {"choice_margin": choice_margin(model, task, arm, "rollout"),
                    "choice_margin_forced": choice_margin(model, task, arm, "forced"),
                    "free_rollout": fl if arm == "left" else fr,
                    "route": ro["route"], "reward_in_rollout": ro["reward_in_rollout"],
                    "trace_fidelity": ro["trace_fidelity"], "trace_error": ro["trace_error"],
                    "trace_error_norm": ro["trace_error_norm"],
                    "reward_entry": ct["reward_entry"],
                    "reward_forced": ct["reward_pred"], "odour_forced": ct["odour_retained"]}
    return out


def run(model_class, kwargs, seed, args, extinguish: str) -> List[Dict[str, Any]]:
    task = G(seed=seed).generate(seed=seed)
    kw = dict(kwargs); kw["seed"] = seed; kw["n_epochs"] = args.epochs
    if args.learning_rate is not None and "learning_rate" in kw:
        kw["learning_rate"] = args.learning_rate
    model = model_class(n_features=task["n_features"], **kw)
    bench = ExtinctionPreferenceBenchmark(reward_noise=args.reward_noise, seed=seed)
    rng = np.random.default_rng(seed)
    n = args.per_phase
    phases = [("train A", ARM["A"], "on"), ("train B", ARM["B"], "on"),
              (f"extinguish {extinguish}", ARM[extinguish], "noise")]
    recs = [{"t": 0, "phase": "start", "ceiling": choice_margin_ceiling(task), **probe(bench, model, task)}]
    t = 0
    for name, arm, reward in phases:
        for k in range(n):
            fade = 1.0 - (k + 1) / n if reward == "noise" else 1.0
            model.fit_sequence(G.trajectory(task, arm, fade, reward, rng, args.reward_noise),
                               epochs=args.epochs)
            t += 1
            recs.append({"t": t, "phase": name, **probe(bench, model, task)})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ahn", type=resolve_model_name, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--out", default="results/extinction_preference")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--per-phase", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--reward-noise", type=float, default=1.0)
    args = ap.parse_args()
    spec = MODEL_REGISTRY[args.model]; cls, kwargs = spec["class"], dict(spec["default_kwargs"])
    seeds = [int(s) for s in args.seeds.split(",")]
    # Suite-shaped, so bin/build_results_index.py picks the figures up:
    # <out>/<Class>/extinction/{plots/*.png, extinction_timeline.json}
    out = os.path.join(args.out, cls.__name__, "extinction")
    plots = os.path.join(out, "plots"); os.makedirs(plots, exist_ok=True)
    t0 = time.time()
    data = {ext: [run(cls, kwargs, sd, args, ext) for sd in seeds] for ext in ("B", "A")}

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = args.per_phase
    lr = args.learning_rate if args.learning_rate is not None else kwargs.get("learning_rate")
    ceiling = data["B"][0][0]["ceiling"]
    RED, GREY = "#C0392B", "#7f8c8d"
    stamp = (f"{cls.__name__} · {len(seeds)} seeds · {args.epochs} pass/presentation"
             + (f" · lr {lr:g}" if lr is not None else ""))

    def row_figure(key, ylab, head, blurb, ylim, path, show_other):
        """One readout, both mirror conditions, with the explanation beside it."""
        fig = plt.figure(figsize=(14.2, 4.35))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1, 1], wspace=0.12,
                              left=0.015, right=0.985, top=0.80, bottom=0.16)
        axt = fig.add_subplot(gs[0, 0]); axt.axis("off")
        axt.text(0.0, 0.55, f"$\\bf{{{head}}}$\n\n" + blurb, fontsize=7.7, va="center",
                 ha="left", linespacing=1.55, color="#222222")
        ax0 = None
        for c, ext in enumerate(("B", "A")):
            runs = data[ext]; keep, drop = ext, other_lab(ext)
            ts = np.array([r["t"] for r in runs[0]])
            ax = fig.add_subplot(gs[0, c + 1], sharey=ax0)
            ax0 = ax0 or ax
            ax.axvspan(2 * n + 0.5, ts[-1] + 0.4, color=RED, alpha=0.07, lw=0)
            for bnd in (n, 2 * n):
                ax.axvline(bnd + 0.5, color="k", ls="--", lw=0.8)
            M = np.array([[rec[keep][key] for rec in run_] for run_ in runs], float)
            ax.errorbar(ts, M.mean(0), yerr=M.std(0), fmt="-o", color=RED, ms=4.5, lw=2.0,
                        capsize=2, zorder=3, label=f"{keep} — extinguished")
            # Acquisition guard: nothing downstream is interpretable if the arm
            # never learned the route in its own training phase. A flat line at
            # the floor is "never acquired", not "extinction did nothing".
            own = n if keep == "A" else 2 * n
            acq = float(np.mean([r[own][keep]["trace_fidelity"] for r in runs]))
            if acq < ACQ_GUARD:
                ax.text(0.5, 0.5, f"acquisition guard FAILED\ntrace fidelity {acq:.2f} at the end of\n"
                                  f"train {keep} (threshold {ACQ_GUARD:.2f})\nnothing below is interpretable",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9,
                        color="#7a1f14", bbox=dict(fc="#fdece9", ec="#C0392B", lw=1.0, pad=6))
            if show_other:
                O = np.array([[rec[drop][key] for rec in run_] for run_ in runs], float)
                ax.errorbar(ts, O.mean(0), yerr=O.std(0), fmt="--^", color=GREY, ms=4, lw=1.4,
                            capsize=2, alpha=0.85, label=f"{drop} — untouched")
            ax.axhline(0, color="k", lw=0.7); ax.grid(alpha=0.2); ax.set_ylim(*ylim)
            ax.set_xlabel("presentation"); ax.set_xticks(ts)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=9)
            else:
                ax.tick_params(labelleft=False)
            ax.legend(fontsize=7.5, loc="lower right" if key == "trace_fidelity" else "lower left")
            ax.set_title(f"train A → train B → extinguish {ext}", fontsize=10, pad=16)
            from matplotlib.transforms import blended_transform_factory as _btf
            tr = _btf(ax.transData, ax.transAxes)
            for x, s_ in ((n / 2 + 0.5, "train A"), (1.5 * n + 0.5, "train B"),
                          (2.5 * n + 0.5, f"extinguish {ext}")):
                ax.text(x, 1.012, s_, ha="center", va="bottom", transform=tr,
                        fontsize=8.2, color="dimgrey")
        fig.suptitle(f"{head.replace(chr(92)+chr(92), '').replace('\\ ', ' ')}   —   {stamp}",
                     fontsize=11)
        fig.savefig(path, dpi=150); plt.close(fig)

    row_figure(
        "trace_fidelity",
        "trace fidelity from the fork\n(1 = exact route, 0 = as wrong as the other arm)",
        "Is\\ the\\ route\\ still\\ there?",
        "The direction is imposed: the stem is walked in and\nthe model is placed on this arm's first step, then it\n"
        "runs free on its own predictions to the cheese, drift\naccumulating.\n\n"
        "Each predicted position is decoded and compared to\nwhere it should be; the mean error is divided by how\n"
        "far apart the two arms are over those steps, so 0\nmeans 'as wrong as naming the other arm' and 1 means\n"
        "exact.\n\n"
        "No reward channel is involved: purely whether the\ntrajectory survives, independent of which way the\nmodel would choose to go.",
        (-0.03, 1.03), os.path.join(plots, "extinction_route_survival.png"), True)

    row_figure(
        "choice_margin",
        "choice-point margin\ntoward the extinguished arm",
        "What\\ comes\\ after\\ the\\ choice\\ point?",
        "The model is cued with the maze entrance only and\nunrolls the stem on its own predictions. The state it\n"
        "produces after the last stem position is scored\nagainst the only two possible answers — the first step\n"
        "of each arm:\n\n"
        "    margin = cos(pred, this arm) − cos(pred, other)\n\n"
        "> 0 means it points down this arm; 0 is a real\ndecision boundary. Antisymmetric, so the other arm is\n"
        f"this line mirrored. Ceiling {ceiling:.2f}: the candidates\nare the arms' first divergent step and sit at cosine\n"
        f"{1 - ceiling:.2f}, not orthogonal. Rolling the stem includes\ndrift, so a latent-state arm builds its own state.",
        (-0.45, 0.45), os.path.join(plots, "extinction_choice_margin.png"), False)

    # ---- supplementary: every candidate readout as its own row --------------
    figv, axes = plt.subplots(len(VARIANTS), 2, figsize=(13, 2.6 * len(VARIANTS) + 1), sharex=True)
    for c, ext in enumerate(("B", "A")):
        runs = data[ext]
        ts = np.array([r["t"] for r in runs[0]])
        for r_i, (key, ylab, ylim) in enumerate(VARIANTS):
            ax = axes[r_i, c]
            for lab, mk, col in (("A", "o", "tab:blue"), ("B", "s", "tab:orange")):
                M = np.array([[rec[lab][key] for rec in run_] for run_ in runs], float)
                ax.errorbar(ts, M.mean(0), yerr=M.std(0), fmt=f"-{mk}", color=col, ms=4,
                            capsize=2, lw=1.4, label=f"{lab} ({ARM[lab]})")
            for bnd in (n, 2 * n):
                ax.axvline(bnd + 0.5, color="k", ls="--", lw=0.8)
            ax.axhline(0, color="grey", lw=0.6); ax.set_ylim(*ylim); ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if r_i == 0:
                ax.set_title(f"train A → train B → extinguish {ext}", fontsize=10)
                ax.legend(fontsize=7, loc="lower left")
            if r_i == len(VARIANTS) - 1:
                ax.set_xlabel("presentation"); ax.set_xticks(ts)
    figv.suptitle(f"Extinction timeline — every candidate readout ({stamp}, {n} presentations per phase)",
                  fontsize=11)
    figv.tight_layout(rect=(0, 0, 1, 0.975))
    figv.savefig(os.path.join(plots, "extinction_timeline_variants.png"), dpi=150); plt.close(figv)

    with open(os.path.join(out, "extinction_timeline.json"), "w") as f:
        json.dump({"args": vars(args), "seeds": seeds, "ceiling": ceiling, "data": data}, f, indent=1)

    for ext in ("B", "A"):
        runs = data[ext]; print(f"\nextinguish {ext}  (mean over seeds; t = presentation)")
        hdr = "  t  phase         " + "  ".join(f"{k[:11]:>11s}A {k[:11]:>11s}B" for k, _, _ in VARIANTS)
        print(hdr)
        for i, rec in enumerate(runs[0]):
            row = []
            for key, _, _ in VARIANTS:
                for lab in ("A", "B"):
                    row.append(np.mean([r[i][lab][key] for r in runs]))
            print(f"  {rec['t']:2d}  {rec['phase']:<13s} " + "  ".join(f"{v:12.2f}" for v in row))
    print(f"\nwrote {plots}/extinction_{{route_survival,choice_margin,timeline_variants}}.png\n"
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
