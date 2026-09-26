#!/usr/bin/env python3
"""Regenerate ``continual_chain_axes.png`` for every arm of a saved run.

The chain section stores its full series (retention matrix, per-task
diagonal/final, forgetting by gap, and the per-task ``exposures`` list) in
``<arm>/symbolic/metrics.json``, so the axes figure can be redrawn from disk
without re-running any arm. Used when the figure changes shape -- e.g. the
epochs-to-criterion panel added on 2026-09-06 -- so an existing run tree stays
the source of truth.

    python bin/replot_continual_chain.py --results-dir results/zoo_capacity_run
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from memval.benchmarks.continual_chain import plot_chain_axes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.results_dir, "*", "symbolic", "metrics.json")))
    done = 0
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        series = (d.get("series") or {}).get("continual_chain") or {}
        if not series.get("retention_matrix"):
            continue
        out = os.path.join(os.path.dirname(p), "plots", "continual_chain_axes.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        metrics = d.get("metrics") or {}
        plot_chain_axes(series, metrics, out,
                        baseline_epochs=metrics.get("convergence_epochs"))
        print(f"wrote {out}")
        done += 1
    print(f"{done} arm(s) replotted")


if __name__ == "__main__":
    main()
