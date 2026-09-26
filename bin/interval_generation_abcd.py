#!/usr/bin/env python
"""Standalone interval-generation probe: A -(1.2s)-> B -(10s)-> C -(1.2s)-> D.

A one-off re-run of the Serial-order 5.5 protocol (``interval_generation``) on
DTSESNSequenceNetwork with a single custom training stream. It reuses the shipped
section's conventions exactly -- codebook cleanup on the rollout's feedback path,
autonomous ``generate()`` for the items AND the gaps, the zoo ESN parameters --
but is deliberately NOT wired into ``memval/benchmarks/interval_timing.py`` or the
symbolic pipeline, so the benchmark's own stimuli and metrics are untouched.

Read-outs, per generated step (1..3):
  * target gap      -- the trained interval preceding that item
  * generated gap   -- what the timing head chose during autonomous rollout
  * (recorded, not plotted) teacher-forced ``predict_time_to_next`` on the true
    history, the same "peak time" read-out the section reports.

Run:  python bin/interval_generation_abcd.py [--n-reps 40] [--seed 42] [--out DIR]
      python bin/interval_generation_abcd.py --seeds 42,43,44,45,46

``--seeds`` repeats the run per seed (the seed moves both the item embeddings
and the reservoir) and writes a SUITE-SHAPED directory, the layout
``bin/build_results_index.py`` lists as a standalone pseudo-suite:
``<out>/interval_generation_abcd_seeds.json``, ``<out>/seeds/seed_<s>.json`` and
``<out>/plots/interval_generation_abcd_seeds.png`` (per-step mean and SD of the
generated gap, every seed drawn as a point). Point ``--out`` at
``results/zoo_capacity_run/DTSESNSequenceNetwork/interval_generation_abcd`` to
put it in the zoo index.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from memval.benchmarks.interval_timing import _attach_codebook_cleanup  # noqa: E402
from memval.benchmarks.symbolic_pipeline import INTERVAL_EMBED_DIM  # noqa: E402
from memval.encoders.symbolic import SymbolicDecoder, SymbolicEncoder  # noqa: E402
from memval.models.baselines import DTSESNSequenceNetwork  # noqa: E402

# The stream. ``GAPS[t]`` is the gap PRECEDING item t (the section's convention).
ITEMS = ["A", "B", "C", "D"]
GAPS = [0.0, 1.2, 10.0, 1.2]

# The zoo run's ESN parameters (results/zoo_capacity_run/.../capacity_scorecard.md),
# with n_epochs stripped exactly as _build_interval_model does: exposure is
# driven by n_reps, not the registry epoch count.
ESN_KWARGS = dict(n_units=400, tau_min=0.1, tau_max=20.0, spectral_radius=0.9,
                  dt=0.05, predict_timing=True)


def run(n_reps: int, seed: int) -> dict:
    enc = SymbolicEncoder(ITEMS, embedding_dim=INTERVAL_EMBED_DIM, seed=seed)
    dec = SymbolicDecoder(enc)
    ev = enc.encode(ITEMS)

    model = DTSESNSequenceNetwork(n_features=INTERVAL_EMBED_DIM, seed=seed,
                                  **ESN_KWARGS)
    _attach_codebook_cleanup(model, enc)
    for _ in range(n_reps):
        model.fit_sequence(ev, intervals=list(GAPS))

    # Autonomous rollout from A: the model supplies both the item and the wait.
    events, gen_gaps = model.generate(ev[0], length=len(ITEMS) - 1)
    gen_gaps = np.asarray(gen_gaps, dtype=float).ravel()
    words = [dec.decode(np.asarray(e)[None, :], top_k=1)[0] for e in np.asarray(events)]
    target = np.asarray(GAPS[1:], dtype=float)

    # Teacher-forced: given the TRUE history up to item t, when is item t+1 due?
    forced = []
    for t in range(len(ITEMS) - 1):
        forced.append(float(model.predict_time_to_next(
            ev[:t + 1], prompt_intervals=list(GAPS[:t + 1]))))
    forced = np.asarray(forced)

    rel_err = np.abs(gen_gaps - target) / target
    return {
        "stream": {"items": ITEMS, "gaps_preceding": GAPS},
        "model": {"class": "DTSESNSequenceNetwork", "kwargs": ESN_KWARGS,
                  "seed": seed, "embedding_dim": INTERVAL_EMBED_DIM,
                  "n_reps": n_reps},
        "steps": [
            {"step": i + 1, "transition": f"{ITEMS[i]}->{ITEMS[i + 1]}",
             "target_gap": float(target[i]), "generated_gap": float(gen_gaps[i]),
             "teacher_forced_gap": float(forced[i]),
             "generated_item": words[i], "expected_item": ITEMS[i + 1],
             "rel_error": float(rel_err[i])}
            for i in range(len(target))
        ],
        "metrics": {
            "tempo_reproduction_error": float(rel_err.mean()),
            "item_acc": float(np.mean([w == e for w, e in zip(words, ITEMS[1:])])),
            "pause_position_acc": float(int(np.argmax(gen_gaps) == np.argmax(target))),
            "peak_time_error": float((np.abs(forced - target) / target).mean()),
        },
    }


def plot(out: dict, path: str) -> None:
    steps = out["steps"]
    xs = np.arange(1, len(steps) + 1)
    target = np.array([s["target_gap"] for s in steps])
    gen = np.array([s["generated_gap"] for s in steps])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w = 0.36
    # Fixed categorical slots: neutral for the target, series-1 blue for the arm.
    b_t = ax.bar(xs - w / 2 - 0.01, target, width=w, color="#c9c9c9",
                 label="target delay", zorder=2)
    b_g = ax.bar(xs + w / 2 + 0.01, gen, width=w, color="#2a78d6",
                 label="generated delay (ESN)", zorder=2)
    for bars in (b_t, b_g):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}s",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color="#0b0b0b")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"step {s['step']}\n{s['transition']}"
                        f"\n(got {s['generated_item']})" for s in steps],
                       fontsize=9)
    ax.set_ylabel("inter-item delay (s)")
    ax.set_ylim(0, max(target.max(), gen.max()) * 1.18)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    m = out["metrics"]
    ax.set_title(
        "Interval generation, DTS-ESN, autonomous rollout from A\n"
        f"train: A -(1.2s)-> B -(10s)-> C -(1.2s)-> D, "
        f"n_reps={out['model']['n_reps']}   "
        f"mean rel. error {m['tempo_reproduction_error']:.3f}   "
        f"items {m['item_acc']:.2f}",
        fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aggregate(runs: list) -> dict:
    """Per-step and per-metric mean / SD / min / max across seed runs."""
    def stats(v):
        v = np.asarray(v, dtype=float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()), "values": v.tolist()}
    steps = []
    for i, st in enumerate(runs[0]["steps"]):
        per = [r["steps"][i] for r in runs]
        steps.append({"step": st["step"], "transition": st["transition"],
                      "target_gap": st["target_gap"],
                      "generated_gap": stats([p["generated_gap"] for p in per]),
                      "teacher_forced_gap": stats([p["teacher_forced_gap"] for p in per]),
                      "rel_error": stats([p["rel_error"] for p in per]),
                      "item_correct": [p["generated_item"] == p["expected_item"] for p in per]})
    return {"stream": runs[0]["stream"],
            "model": {k: v for k, v in runs[0]["model"].items() if k != "seed"},
            "seeds": [r["model"]["seed"] for r in runs],
            "steps": steps,
            "metrics": {k: stats([r["metrics"][k] for r in runs]) for k in runs[0]["metrics"]}}


def plot_seeds(agg: dict, path: str) -> None:
    steps = agg["steps"]
    xs = np.arange(1, len(steps) + 1)
    target = np.array([s["target_gap"] for s in steps])
    mean = np.array([s["generated_gap"]["mean"] for s in steps])
    sd = np.array([s["generated_gap"]["sd"] for s in steps])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w = 0.36
    ax.bar(xs - w / 2 - 0.01, target, width=w, color="#c9c9c9", label="target delay", zorder=2)
    ax.bar(xs + w / 2 + 0.01, mean, width=w, color="#2a78d6", yerr=sd, capsize=4,
           error_kw=dict(ecolor="#0b0b0b", lw=1), label="generated delay, seed mean ± SD",
           zorder=2)
    rng = np.random.default_rng(0)
    for i, s in enumerate(steps):
        v = np.asarray(s["generated_gap"]["values"])
        ax.scatter(xs[i] + w / 2 + 0.01 + rng.uniform(-0.08, 0.08, len(v)), v, s=14,
                   color="white", edgecolor="#0b0b0b", lw=0.7, zorder=3)
    for x, t in zip(xs, target):
        ax.annotate(f"{t:.2f}s", xy=(x - w / 2 - 0.01, t), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for x, m_, sd_ in zip(xs, mean, sd):
        ax.annotate(f"{m_:.2f}s ± {sd_:.2f}", xy=(x + w / 2 + 0.01, m_ + sd_), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"step {s['step']}\n{s['transition']}\n"
                        f"(items {sum(s['item_correct'])}/{len(s['item_correct'])})"
                        for s in steps], fontsize=9)
    ax.set_ylabel("inter-item delay (s)")
    ax.set_ylim(0, max(target.max(), (mean + sd).max()) * 1.45)   # headroom for the legend
    ax.grid(axis="y", color="#e6e6e3", lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    m = agg["metrics"]
    ax.set_title(
        f"Interval generation, DTS-ESN, autonomous rollout from A — {len(agg['seeds'])} seeds\n"
        f"train: A -(1.2s)-> B -(10s)-> C -(1.2s)-> D, n_reps={agg['model']['n_reps']}\n"
        f"mean rel. error {m['tempo_reproduction_error']['mean']:.3f} "
        f"± {m['tempo_reproduction_error']['sd']:.3f}   "
        f"items {m['item_acc']['mean']:.2f}", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-reps", type=int, default=40,
                   help="presentations of the stream (section default 40)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default=None,
                   help="comma list; runs every seed and writes the aggregate instead")
    p.add_argument("--out", default=os.path.join(
        "results", "interval_generation_abcd", "DTSESNSequenceNetwork"))
    a = p.parse_args()

    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        runs = [run(a.n_reps, sd) for sd in seeds]
        os.makedirs(os.path.join(a.out, "seeds"), exist_ok=True)
        for sd, r in zip(seeds, runs):
            with open(os.path.join(a.out, "seeds", f"seed_{sd}.json"), "w") as f:
                json.dump(r, f, indent=2)
        agg = aggregate(runs)
        with open(os.path.join(a.out, "interval_generation_abcd_seeds.json"), "w") as f:
            json.dump(agg, f, indent=2)
        os.makedirs(os.path.join(a.out, "plots"), exist_ok=True)
        plot_seeds(agg, os.path.join(a.out, "plots", "interval_generation_abcd_seeds.png"))
        print(f"{'seed':>4}  " + "  ".join(f"{s['transition']:>8}" for s in runs[0]["steps"])
              + "  items  rel.err  peak")
        for sd, r in zip(seeds, runs):
            print(f"{sd:>4}  " + "  ".join(f"{st['generated_gap']:>8.3f}" for st in r["steps"])
                  + f"  {r['metrics']['item_acc']:>5.2f}  "
                    f"{r['metrics']['tempo_reproduction_error']:>7.4f}  "
                    f"{r['metrics']['peak_time_error']:.4f}")
        print("mean  " + "  ".join(f"{st['generated_gap']['mean']:>8.3f}" for st in agg["steps"]))
        print("  sd  " + "  ".join(f"{st['generated_gap']['sd']:>8.3f}" for st in agg["steps"]))
        for k, v in agg["metrics"].items():
            print(f"{k:>26}: {v['mean']:.4f} ± {v['sd']:.4f}  [{v['min']:.4f}, {v['max']:.4f}]")
        print(f"wrote {a.out}/interval_generation_abcd_seeds.json, plots/, seeds/")
        return

    out = run(a.n_reps, a.seed)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "interval_generation_abcd.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot(out, os.path.join(a.out, "interval_generation_abcd.png"))

    print(f"{'step':>4}  {'transition':>10}  {'target':>7}  {'generated':>9}  "
          f"{'forced':>7}  item")
    for s in out["steps"]:
        print(f"{s['step']:>4}  {s['transition']:>10}  {s['target_gap']:>7.2f}  "
              f"{s['generated_gap']:>9.2f}  {s['teacher_forced_gap']:>7.2f}  "
              f"{s['generated_item']} (expected {s['expected_item']})")
    for k, v in out["metrics"].items():
        print(f"{k:>26}: {v:.3f}")
    print(f"wrote {a.out}/interval_generation_abcd.{{json,png}}")


if __name__ == "__main__":
    main()
