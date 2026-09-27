#!/usr/bin/env python
"""The continual chain along the two overlap dials of the symbolic encoder.

The ``continual_chain`` section builds one task per category, which puts the
between-task cosine at ~0 (quasi-orthogonal prototypes) and holds it there.
That isolates the *load* axis, and it is also why a linear store shows a flat
retention matrix on it: task B disturbs task A in proportion to ``x_B . x_A``,
and the section never lets that product leave zero. This script sweeps the
two dials the encoder now exposes and draws the retention matrix at every rung:

  between  ``between_category_cosine`` (rho_b): cosine between task prototypes,
           exact. Word-level between-task cosine is ``rho * rho_b``. This is
           the forgetting lever. Within-task overlap held at the section's
           default (``--sigma 0.2``, rho = 0.20).
  within   ``category_variance`` (sigma): within-task cosine
           ``rho = 1 / (1 + d sigma^2)``. Makes each list internally
           confusable. Between-task cosine held exactly at ``--rho-b-fixed``
           (default 0, exactly orthogonal categories).

Protocol, scoring and statistics are the section's own (``build_chain_material``,
``ContinualChainBenchmark`` in criterion mode, ``measure_recall_associative``
under the clean-single probe), so a rung at the section's defaults reproduces
the section's matrix. Rungs are replicated over encoder+model seeds; the
matrices drawn are seed means, the curves carry the seed spread.

Two figures, into the arm's symbolic plots directory so the results index
picks them up beside ``continual_chain_retention.png``:

  continual_chain_overlap_matrices.png   small multiples: one full matrix per
                                         rung, one row per dial, shared scale
  continual_chain_overlap_axes.png       per rung x task: exposure to criterion,
                                         learned, final, retained-by-gap; and
                                         LA / ACC / ratio / SPI / forgetting
                                         against each dial
  continual_chain_overlap_intrusions.png per rung: where the errors go (same
                                         task / earlier task / later task) and
                                         who the runner-up is, with the margin --
                                         the measurement behind "cross-task
                                         confusion" vs "within-task confusion"

and ``continual_chain_overlap_sweep.json`` with every per-seed matrix, summary,
exposure record and the measured geometry of each rung.

    python bin/continual_chain_overlap_sweep.py --model ahn --out results/ahn_capacity_run
    python bin/continual_chain_overlap_sweep.py --model ahn --dials between --between-rungs 0,0.5,0.9
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

from run_benchmark import MODEL_REGISTRY, resolve_model_name  # noqa: E402  (bin/ on the path above)

from memval.benchmarks._selection import resolve_epochs_default  # noqa: E402
from memval.benchmarks.continual_chain import (  # noqa: E402
    COMPETITOR_CLASSES, INTRUSION_CLASSES, ContinualChainBenchmark,
    build_chain_material, probe_intrusions)
from memval.benchmarks.probe import resolve_probe  # noqa: E402
from memval.benchmarks.symbolic_pipeline import (  # noqa: E402
    _get_model_kwargs, load_vocab, mean_recall_rate, measure_recall_associative)
from memval.metrics.retention import (  # noqa: E402
    plot_intrusion_breakdown, plot_retention_matrix_grid, plot_retention_sweep_axes,
    retention_summary)

EMBED_DIM = 100


def _floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_model(model_class, default_kwargs: Dict[str, Any], encoder, epochs: int, seed: int):
    """Construct the arm the way the symbolic pipeline does for this section."""
    kw = _get_model_kwargs(default_kwargs, epochs)
    if "seed" in kw:
        kw["seed"] = seed
    if "encoder" in model_class.__init__.__code__.co_varnames:
        return model_class(encoder=encoder, n_features=encoder.embedding_dim, **kw)
    return model_class(n_features=encoder.embedding_dim, **kw)


def fresh_epochs_to_criterion(model_class, default_kwargs, enc, dec, words, *, seed, n_trials,
                              noise_scale, epochs, criterion=0.95, max_epochs=512):
    """Passes a FRESH model needs on one task alone, same criterion loop as the chain.

    This is the denominator for intransigence: dividing the chain's cost for
    task i by this cancels both the rung's material cost and any task-specific
    difficulty, leaving only what the already-stored tasks added."""
    model = build_model(model_class, default_kwargs, enc, epochs, seed)
    emb = enc.encode(words)
    used, reached = 0, False
    while used < max_epochs:
        model.fit_sequence(emb, epochs=1)
        used += 1
        model.reset_context()
        if mean_recall_rate(measure_recall_associative(
                model, words, enc, dec, n_trials=n_trials, noise_scale=noise_scale)) >= criterion:
            reached = True
            break
    return used, reached


def run_rung(model_class, default_kwargs, vocab, *, n_tasks, seq_len, sigma, rho_b,
             seed, n_trials, noise_scale, epochs, rehearse) -> Dict[str, Any]:
    material = build_chain_material(vocab, n_tasks, seq_len, embedding_dim=EMBED_DIM,
                                    category_variance=sigma, between_category_cosine=rho_b,
                                    seed=seed)
    enc, dec = material["encoder"], material["decoder"]
    seqs = material["sequences"]
    names = list(seqs.keys())
    model = build_model(model_class, default_kwargs, enc, epochs, seed)

    def score_task(i: int) -> float:
        model.reset_context()
        return mean_recall_rate(measure_recall_associative(
            model, seqs[names[i]], enc, dec, n_trials=n_trials, noise_scale=noise_scale))

    out = ContinualChainBenchmark(n_trials=n_trials, noise_scale=noise_scale,
                                  rehearse=rehearse).evaluate(
        model=model,
        datasets={"sequences": seqs, "embeddings": material["embeddings"],
                  "score_fn": score_task},
        epochs=epochs,
        chance_level=1.0 / len(material["vocab"]),
    )
    out["geometry"] = material["geometry"]
    out["task_labels"] = names
    out["chance_level"] = 1.0 / len(material["vocab"])
    # Fresh-model baseline for every task at this rung (intransigence denominator).
    out["fresh"] = [dict(zip(("epochs", "reached"),
                             fresh_epochs_to_criterion(model_class, default_kwargs, enc, dec,
                                                       seqs[n], seed=seed, n_trials=n_trials,
                                                       noise_scale=noise_scale, epochs=epochs)))
                    for n in names]
    # Final-state readout: what is recalled instead, and who the runner-up is.
    # (After rehearsal if --rehearse ran, so it describes the model as left.)
    out["intrusions"] = probe_intrusions(model, seqs, enc, dec)
    return out


def aggregate(runs: List[Dict[str, Any]], chance_level: float) -> Dict[str, Any]:
    """Seed mean of the matrix and the exposure; the per-seed summaries kept."""
    Rs = np.array([r["series"]["retention_matrix"] for r in runs], dtype=float)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # upper triangle is all-NaN by design
            R_mean = np.nanmean(Rs, axis=0)
    expo = np.array([[e["epochs"] for e in r["series"]["exposures"]] for r in runs], dtype=float)
    chain_cens = np.array([[e["reached"] is False for e in r["series"]["exposures"]] for r in runs])
    fresh = np.array([[f["epochs"] for f in r["fresh"]] for r in runs], dtype=float)
    fresh_cens = np.array([[not f["reached"] for f in r["fresh"]] for r in runs])
    intr = [r["intrusions"] for r in runs]
    n_probes = sum(i["n_probes"] for i in intr)
    outcomes = {c: sum(i["outcome_counts"][c] for i in intr) for c in INTRUSION_CLASSES}
    competitors = {c: sum(i["competitor_counts"][c] for i in intr) for c in COMPETITOR_CLASSES}
    all_recs = [rec for i in intr for rec in i["records"]]
    margin_by_comp = {c: (float(np.mean([r["margin"] for r in all_recs if r["competitor_class"] == c]))
                          if competitors[c] else float("nan")) for c in COMPETITOR_CLASSES}
    offsets: Dict[str, int] = {}
    for i in intr:
        for k, v in i["within_offsets"].items():
            offsets[k] = offsets.get(k, 0) + v
    return {
        "intrusions": {
            "n_probes": n_probes, "outcome_counts": outcomes,
            "competitor_counts": competitors,
            "margin_mean": float(np.mean([r["margin"] for r in all_recs])),
            "margin_by_competitor": margin_by_comp, "within_offsets": offsets,
        },
        "R_mean": R_mean,
        "summary_of_mean": retention_summary(R_mean, chance_level=chance_level),
        "summaries": [retention_summary(np.array(r["series"]["retention_matrix"]),
                                        chance_level=chance_level) for r in runs],
        "exposure_mean": expo.mean(axis=0),
        "exposure_seeds": expo, "chain_censored_seeds": chain_cens,
        "fresh_seeds": fresh, "fresh_censored_seeds": fresh_cens,
        "censored": bool(any(not r["metrics"]["chain_criterion_reached"] for r in runs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ahn", type=resolve_model_name, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--out", default="results/overlap_sweep",
                    help="run directory; figures go to <out>/<ClassName>/symbolic/plots/")
    ap.add_argument("--dials", default="between,within",
                    help="comma list from {between, within}")
    ap.add_argument("--between-rungs", default="0,0.25,0.5,0.75,0.95",
                    help="rho_b values for the between dial")
    ap.add_argument("--within-rungs", default="0.05,0.1,0.2,0.5,1.0",
                    help="sigma values for the within dial (the suite's ladder)")
    ap.add_argument("--sigma", type=float, default=0.2,
                    help="within-task sigma held fixed on the between dial (section default 0.2)")
    ap.add_argument("--rho-b-fixed", type=float, default=0.0,
                    help="between-task cosine held fixed on the within dial (0 = exactly orthogonal)")
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="encoder+model seeds; the section itself uses 42")
    ap.add_argument("--n-tasks", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=5)
    ap.add_argument("--n-trials", type=int, default=30,
                    help="requested probe trials; the section's protocol resolves the real count")
    ap.add_argument("--epochs", type=int, default=None,
                    help="per-stage budget for fixed-mode paths and rehearsal; default = registry n_epochs, else 300")
    ap.add_argument("--rehearse", action="store_true",
                    help="also run the selective-retention phase at every rung")
    args = ap.parse_args()

    spec = MODEL_REGISTRY[args.model]
    model_class, default_kwargs = spec["class"], dict(spec["default_kwargs"])
    class_name = model_class.__name__
    epochs = int(resolve_epochs_default(args.epochs, default_kwargs, 300))
    n_trials, noise_scale = resolve_probe("continual_chain", model_class, args.n_trials)
    seeds = _ints(args.seeds)
    vocab = load_vocab()
    dials = [d.strip() for d in args.dials.split(",") if d.strip()]

    plots_dir = os.path.join(args.out, class_name, "symbolic", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    common = dict(n_tasks=args.n_tasks, seq_len=args.seq_len, n_trials=n_trials,
                  noise_scale=noise_scale, epochs=epochs, rehearse=args.rehearse)
    record: Dict[str, Any] = {
        "model": args.model, "class": class_name, "seeds": seeds, "epochs": epochs,
        "probe": {"n_trials": n_trials, "noise_scale": noise_scale},
        "n_tasks": args.n_tasks, "seq_len": args.seq_len, "dials": {},
    }
    grid_rows, axes_sweeps = [], []
    t0 = time.time()

    for dial in dials:
        if dial == "between":
            rungs = _floats(args.between_rungs)
            settings = [(args.sigma, rb) for rb in rungs]
            name = f"between-task cosine ρ_b\n(σ={args.sigma:g}, within ρ={1/(1+EMBED_DIM*args.sigma**2):.2f})"
            xlabel = "between-task prototype cosine ρ_b"
        elif dial == "within":
            rungs = _floats(args.within_rungs)
            settings = [(s, args.rho_b_fixed) for s in rungs]
            name = f"within-task cosine ρ\n(ρ_b={args.rho_b_fixed:g} exactly)"
            xlabel = "within-task cosine ρ = 1/(1+dσ²)   (overlap increases →)"
        else:
            raise SystemExit(f"unknown dial {dial!r}")

        per_rung, panels, Rs, expos, sums, xs, labels = [], [], [], [], [], [], []
        intr_rows: List[Dict[str, Any]] = []
        for rung, (sigma, rho_b) in zip(rungs, settings):
            runs = [run_rung(model_class, default_kwargs, vocab, sigma=sigma, rho_b=rho_b,
                             seed=sd, **common) for sd in seeds]
            chance = runs[0]["chance_level"]
            agg = aggregate(runs, chance)
            geo = runs[0]["geometry"]
            if dial == "between":
                x = rho_b
                lab = f"ρ_b={rho_b:.2f}  word cos {geo['between_word_cos_law']:.2f}"
            else:
                x = geo["within_cos_law"]
                lab = f"σ={sigma:g}  ρ={x:.2f}"
            som = agg["summary_of_mean"]
            med_chain = np.median(agg["exposure_seeds"], axis=0)
            med_fresh = np.median(agg["fresh_seeds"], axis=0)
            intr = np.median(agg["exposure_seeds"] / np.maximum(agg["fresh_seeds"], 1), axis=0)
            print(f"[{dial}] {lab:<34} ACC {som['avg_accuracy']:.3f}  LA {som['avg_learning']:.3f}  "
                  f"forget {som['avg_forgetting']:+.3f}  ratio {som['retention_ratio']:.3f}  "
                  f"SPI {som['stability_plasticity_index']:.3f}  "
                  f"epochs(median) chain {med_chain.mean():.1f} fresh {med_fresh.mean():.1f}  "
                  f"intransigence {intr.mean():.2f}{'  CENSORED' if agg['censored'] else ''}")
            ic = agg["intrusions"]
            oc, cc, npb = ic["outcome_counts"], ic["competitor_counts"], max(ic["n_probes"], 1)
            print(f"           errors: within {oc['within_task']/npb:.2f}  earlier {oc['cross_earlier']/npb:.2f}  "
                  f"later {oc['cross_later']/npb:.2f} | runner-up: same {cc['same_task']/npb:.2f}  "
                  f"earlier {cc['earlier']/npb:.2f}  later {cc['later']/npb:.2f} | margin {ic['margin_mean']:+.3f}")
            intr_rows.append(ic)
            per_rung.append({
                "rung": rung, "sigma": sigma, "rho_b": rho_b, "geometry": geo, "x": x,
                "label": lab, "chance_level": chance,
                "R_mean": agg["R_mean"].tolist(),
                "summary_of_mean": som,
                "exposure_mean": agg["exposure_mean"].tolist(),
                "exposure_median": np.median(agg["exposure_seeds"], axis=0).tolist(),
                "fresh_median": np.median(agg["fresh_seeds"], axis=0).tolist(),
                "intransigence_ratio_median": np.median(
                    agg["exposure_seeds"] / np.maximum(agg["fresh_seeds"], 1), axis=0).tolist(),
                "censored": agg["censored"],
                "intrusions": agg["intrusions"],
                "per_seed": [{"seed": sd, "R": r["series"]["retention_matrix"],
                              "metrics": r["metrics"], "exposures": r["series"]["exposures"],
                              "fresh": r["fresh"],
                              "intrusion_records": r["intrusions"]["records"]}
                             for sd, r in zip(seeds, runs)],
            })
            panels.append({"label": lab, "R": agg["R_mean"]})
            Rs.append(agg["R_mean"]); expos.append(agg)
            sums.append(agg["summaries"]); xs.append(x); labels.append(lab.split("  ")[0])
        record["dials"][dial] = {"name": name, "rungs": per_rung}
        grid_rows.append({"name": name, "panels": panels})
        axes_sweeps.append({"name": name, "xlabel": xlabel, "x": xs, "rung_labels": labels,
                            "task_labels": runs[0]["task_labels"], "Rs": Rs,
                            "exposures": [a["exposure_mean"] for a in expos],
                            "exposure_seeds": [a["exposure_seeds"] for a in expos],
                            "chain_censored_seeds": [a["chain_censored_seeds"] for a in expos],
                            "fresh_seeds": [a["fresh_seeds"] for a in expos],
                            "fresh_censored_seeds": [a["fresh_censored_seeds"] for a in expos],
                            "summaries": sums,
                            "log_x": dial == "within",
                            "outcomes": [i["outcome_counts"] for i in intr_rows],
                            "competitors": [i["competitor_counts"] for i in intr_rows],
                            "margin_mean": [i["margin_mean"] for i in intr_rows],
                            "margin_by_competitor": [i["margin_by_competitor"] for i in intr_rows]})

    task_labels = axes_sweeps[0]["task_labels"]
    chance = record["dials"][dials[0]]["rungs"][0]["chance_level"]
    seeds_note = f"mean of {len(seeds)} seeds" if len(seeds) > 1 else f"seed {seeds[0]}"
    p1 = os.path.join(plots_dir, "continual_chain_overlap_matrices.png")
    plot_retention_matrix_grid(
        grid_rows, p1,
        title=f"Retention matrix along the overlap dials ({class_name}, {seeds_note}, "
              f"{args.n_tasks} tasks × {args.seq_len}, trained to criterion)",
        value_label="cued recall (mean recall rate)", task_labels=task_labels,
        chance_level=chance)
    p2 = os.path.join(plots_dir, "continual_chain_overlap_axes.png")
    plot_retention_sweep_axes(
        axes_sweeps, p2,
        title=f"Plasticity and stability along the overlap dials ({class_name}, {seeds_note})",
        value_label="cued recall (mean recall rate)")
    p3 = os.path.join(plots_dir, "continual_chain_overlap_intrusions.png")
    plot_intrusion_breakdown(
        [{"name": sw["name"].split("\n")[0], "rung_labels": sw["rung_labels"],
          "outcomes": sw["outcomes"], "competitors": sw["competitors"],
          "margin_mean": sw["margin_mean"], "margin_by_competitor": sw["margin_by_competitor"]}
         for sw in axes_sweeps], p3,
        title=f"Intrusion source and runner-up identity along the overlap dials "
              f"({class_name}, {seeds_note}, final stage)")
    pj = os.path.join(args.out, class_name, "symbolic", "continual_chain_overlap_sweep.json")
    with open(pj, "w") as f:
        json.dump(record, f, indent=1)
    print(f"\nwrote {p1}\n      {p2}\n      {p3}\n      {pj}\n{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
