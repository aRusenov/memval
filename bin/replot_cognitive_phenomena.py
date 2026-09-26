#!/usr/bin/env python
"""Regenerate the cognitive-phenomena (L0) figures from a saved run.

    python bin/replot_cognitive_phenomena.py results/cognitive_l0/AsymmetricHopfieldNetwork/symbolic/cognitive_phenomena_metrics.json

Writes plots/cognitive_*.png next to the JSON (or into --plots-dir). Every
figure carries a schematic human-reference panel after the Kahana (2020)
figures; see docs/cognitive_phenomena_design.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from memval.benchmarks.cognitive_phenomena import replot_cognitive_phenomena  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("--plots-dir", default=None)
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()
    for p in replot_cognitive_phenomena(args.json_path, args.plots_dir, args.model_name):
        print(p)


if __name__ == "__main__":
    main()
