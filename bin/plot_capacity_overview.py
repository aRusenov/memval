#!/usr/bin/env python
"""Capacity-by-arm overview figure (the Results headline).

One figure, two panels, read from the per-arm ``capacity_scorecard.json``
files (never from ``_compare/zoo_profiles.json``, which is a cache and goes
stale the moment one arm is re-scored).

  A. capacity x arm matrix -- the score as a single-hue heatmap, coverage as a
     separate strip inside the cell. Score and coverage are two claims and are
     never multiplied (see compare_capacity_arms.py).
  B. dimension x arm matrix -- the same scores one level down, with the
     non-scored statuses drawn as distinct glyphs so a missing instrument is
     never read as a model failure.

Usage
-----
    python bin/plot_capacity_overview.py \
        --root results/zoo_capacity_run \
        --out  results/zoo_capacity_run/_compare/capacity_overview
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np

CAPACITIES = ["Continual retention", "One-shot learning", "Pattern completion",
              "Sequence disambiguation", "Serial order"]
CAP_SHORT = {"Continual retention": "Continual\nretention",
             "One-shot learning": "One-shot\nlearning",
             "Pattern completion": "Pattern\ncompletion",
             "Sequence disambiguation": "Sequence\ndisambiguation",
             "Serial order": "Serial\norder"}
DIM_SHORT = {"load": "load", "plasticity_under_load": "plasticity\nunder load",
             "contingency": "contingency", "relevance": "relevance",
             "schema_consistency": "schema\nconsistency",
             "presentations": "presenta-\ntions",
             "presentations_streamed": "streamed",
             "cue_corruption": "cue\ncorruption", "cue_point": "cue point",
             "cue_completeness": "cue\ncompleteness",
             "contextual_overlap": "contextual\noverlap",
             "item_similarity": "item\nsimilarity",
             "unrolling": "unrolling", "binding_ordinal": "binding\nordinal",
             "establishment": "establish-\nment",
             "interval_retention": "interval\nretention",
             "interval_generation": "interval\ngeneration"}

# (label, results sub-directory, unit grain, plasticity locus)
ARMS = [
    ("AHN",        "AsymmetricHopfieldNetwork",    "rate",    "one matrix"),
    ("theta",      "ThetaPhaseSequenceNetwork",    "rate",    "one matrix"),
    ("tPC",        "MultilayerTemporalPCNetwork",  "rate",    "all-plastic"),
    ("recirc",     "PredictiveRecirculationNetwork","rate",   "all-plastic"),
    ("EP",         "OriginalEqPropSequenceNetwork","rate",    "all-plastic"),
    ("DTS-ESN",    "DTSESNSequenceNetwork",        "rate",    "readout only"),
]

# Sequential blue ramp, steps 100 -> 700 (dataviz reference palette).
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)
STATUS = {  # same colours as compare_capacity_arms.py
    "unresolved": "#B07A16", "not_applicable": "#9a9a9a",
    "unbuilt": "#5B4B8A", "unrunnable": "#C0392B", "protocol_limited": "#AD3F6B",
}
INK, INK2, INK3, LINE = "#1b1b1b", "#52514e", "#8a8a8a", "#d8d8d8"


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def load(root):
    out = []
    for label, sub, grain, locus in ARMS:
        p = os.path.join(root, sub, "capacity_scorecard.json")
        if not os.path.exists(p):
            print(f"[skip] {label}: no scorecard at {p}")
            continue
        with open(p) as f:
            out.append((label, grain, locus, json.load(f)))
    return out


def cell_text_color(score):
    return "white" if score is not None and score >= 0.55 else INK


def draw(arms, out, title):
    n = len(arms)
    dims = [(cap, d) for cap in CAPACITIES
            for d in arms[0][3]["capacities"][cap]["dimensions"]]
    weights = {(cap, d): arms[0][3]["dimension_weights"][cap][d] for cap, d in dims}

    fig = plt.figure(figsize=(14.2, 9.6), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[n * 0.62 + 1.2, n * 0.5 + 1.4],
                          hspace=0.48, left=0.11, right=0.985, top=0.855, bottom=0.06)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # --------------------------------------------------------------- panel A
    ax = axA
    ax.set_xlim(0, 5); ax.set_ylim(n, 0); ax.axis("off")
    for r, (label, grain, locus, d) in enumerate(arms):
        for c, cap in enumerate(CAPACITIES):
            s = _num(d["capacities"][cap]["score"]) or 0.0
            cov = float(d["capacities"][cap]["coverage"])
            ax.add_patch(Rectangle((c + 0.03, r + 0.05), 0.94, 0.9,
                                   facecolor=CMAP(s), edgecolor="white", lw=1.5))
            tc = cell_text_color(s)
            ax.text(c + 0.5, r + 0.36, f"{s:.2f}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=tc)
            # coverage strip: fraction of the capacity's weight that was measurable
            x0, w = c + 0.25, 0.5
            ax.add_patch(Rectangle((x0, r + 0.60), w, 0.07, facecolor="white",
                                   alpha=0.45, edgecolor="none"))
            ax.add_patch(Rectangle((x0, r + 0.60), w * cov, 0.07, facecolor=tc,
                                   alpha=0.9 if tc == "white" else 0.7, edgecolor="none"))
            if cov < 0.999:
                ax.text(c + 0.5, r + 0.82, f"coverage {cov:.2f}", ha="center",
                        va="center", fontsize=7.4, color=tc, alpha=0.95)
        ax.text(-0.06, r + 0.38, label, ha="right", va="center", fontsize=11.5,
                fontweight="bold", color=INK, transform=ax.transData)
        ax.text(-0.06, r + 0.68, f"{grain} · {locus}", ha="right", va="center",
                fontsize=8.2, color=INK3)
    for c, cap in enumerate(CAPACITIES):
        ax.text(c + 0.5, -0.12, CAP_SHORT[cap], ha="center", va="bottom",
                fontsize=10.5, color=INK, fontweight="semibold")
    ax.text(0, -0.9, "A   Capacity score per arm", ha="left", va="bottom",
            fontsize=12.5, fontweight="bold", color=INK, transform=ax.transData)

    # --------------------------------------------------------------- panel B
    ax = axB
    nd = len(dims)
    # x positions with a gap between capacity groups
    xs, x, prev = [], 0.0, None
    for cap, d in dims:
        if prev is not None and cap != prev:
            x += 0.35
        xs.append(x); x += 1.0; prev = cap
    ax.set_xlim(-0.05, x + 0.05); ax.set_ylim(n, 0); ax.axis("off")
    for r, (label, grain, locus, d) in enumerate(arms):
        for (cap, dim), xi in zip(dims, xs):
            info = d["capacities"][cap]["dimensions"].get(dim)
            st = info["status"] if info else "absent"
            s = _num(info["score"]) if info else None
            if st == "scored" and s is not None:
                ax.add_patch(Rectangle((xi + 0.04, r + 0.06), 0.92, 0.88,
                                       facecolor=CMAP(s), edgecolor="white", lw=1.2))
                ax.text(xi + 0.5, r + 0.5, f"{s:.2f}", ha="center", va="center",
                        fontsize=9.2, color=cell_text_color(s))
            else:
                col = STATUS.get(st, LINE)
                hatch = {"unresolved": "xx", "not_applicable": "//",
                         "unbuilt": "..", "unrunnable": "**"}.get(st, "")
                ax.add_patch(Rectangle((xi + 0.04, r + 0.06), 0.92, 0.88,
                                       facecolor="white", edgecolor=col, lw=1.1,
                                       hatch=hatch, alpha=0.9))
        ax.text(-0.25, r + 0.5, label, ha="right", va="center", fontsize=10.5,
                fontweight="bold", color=INK)
    ax.plot([-1.6, x], [first_spk, first_spk], color=INK3, lw=0.9, ls=(0, (4, 3)),
            clip_on=False)
    # dimension labels + weights, capacity group brackets
    for (cap, dim), xi in zip(dims, xs):
        ax.text(xi + 0.5, -0.1, DIM_SHORT[dim], ha="center", va="bottom",
                fontsize=7.6, color=INK2, linespacing=1.05)
        ax.text(xi + 0.5, n + 0.12, f"w {weights[(cap, dim)]:.2f}", ha="center",
                va="top", fontsize=7.0, color=INK3)
    for cap in CAPACITIES:
        cols = [xi for (c, _), xi in zip(dims, xs) if c == cap]
        x0, x1 = cols[0] + 0.04, cols[-1] + 0.96
        ax.plot([x0, x1], [-0.95, -0.95], color=INK, lw=1.2, clip_on=False)
        ax.text((x0 + x1) / 2, -1.02, CAP_SHORT[cap], ha="center", va="bottom", fontsize=9.2, linespacing=1.05,
                fontweight="semibold", color=INK, clip_on=False)
    ax.text(0, -2.1, "B   Dimension score per arm, with the reason a dimension was not scored",
            ha="left", va="bottom", fontsize=12.5, fontweight="bold", color=INK, clip_on=False)

    # legends: colour bar for score, status glyphs
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cax = fig.add_axes([0.895, 0.95, 0.085, 0.015])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=7.5, length=2, colors=INK2)
    cb.outline.set_edgecolor(LINE)
    cax.set_title("score (0–1)", fontsize=8, color=INK2, pad=3)
    present = {d["capacities"][cap]["dimensions"][dim]["status"]
               for _, _, _, d in arms for cap, dim in dims
               if dim in d["capacities"][cap]["dimensions"]}
    LEGEND = [("unresolved", "xx", "guard failed: section had no dynamic range for this arm"),
              ("not_applicable", "//", "not applicable: arm lacks the declared capability"),
              ("unbuilt", "..", "unbuilt: suite has no section for this dimension"),
              ("unrunnable", "**", "unrunnable: run emitted no finite metric"),
              ("protocol_limited", "xx", "protocol-limited")]
    handles = [Patch(facecolor="white", edgecolor=STATUS[k], hatch=h, label=lab)
               for k, h, lab in LEGEND if k in present]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.55, 0.0), handlelength=1.6)

    fig.text(0.11, 0.965, title, fontsize=16, fontweight="bold", color=INK, ha="left")
    fig.text(0.11, 0.925,
             "Cell colour and number: capacity score. In-cell strip: coverage, the share of the capacity's dimension weight\n"
             "that was measurable for that arm. Score and coverage are two claims and are never multiplied.",
             fontsize=8.8, color=INK2, ha="left", va="bottom", linespacing=1.3)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    # fresh numbers alongside the figure
    table = {label: {cap: {"score": _num(d["capacities"][cap]["score"]),
                           "coverage": d["capacities"][cap]["coverage"]}
                     for cap in CAPACITIES} for label, _, _, d in arms}
    with open(f"{out}.json", "w") as f:
        json.dump(table, f, indent=1)
    print(f"wrote {out}.png / .pdf / .json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/zoo_capacity_run")
    ap.add_argument("--out", default="results/zoo_capacity_run/_compare/capacity_overview")
    ap.add_argument("--title", default="MemVal capacity profile of the model zoo")
    a = ap.parse_args()
    arms = load(a.root)
    draw(arms, a.out, a.title)


if __name__ == "__main__":
    main()
