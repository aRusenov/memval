#!/usr/bin/env python3
"""Redraw ``cue_availability.png`` for every arm of a saved run from its side file.

`memval/benchmarks/cue_availability.py` writes ``cue_availability_metrics.json``
(metrics + series) beside the symbolic suite, so the figure can be redrawn
without re-running the section.

    python bin/replot_cue_availability.py --results-dir results/zoo_capacity_run
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from memval.benchmarks.cue_availability import _plot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    a = ap.parse_args()
    n = 0
    for p in sorted(glob.glob(os.path.join(a.results_dir, "*", "symbolic",
                                           "cue_availability_metrics.json"))):
        with open(p) as f:
            d = json.load(f)
        run_dir = os.path.dirname(p)
        name = (d.get("metadata") or {}).get("model") or p.split(os.sep)[-3]
        _plot(d["series"], d["metrics"], name, run_dir)
        print(f"wrote {run_dir}/plots/cue_availability.png")
        n += 1
    print(f"{n} arm(s) replotted")


if __name__ == "__main__":
    main()
