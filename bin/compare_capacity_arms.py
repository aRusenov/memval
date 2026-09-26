#!/usr/bin/env python
"""Cross-arm comparison figures for the capacity scorecard.

Reads one ``capacity_scorecard.json`` per arm and draws the four views the
per-arm radar cannot: every arm on one set of axes, the score/coverage split
made explicit, which dimensions were scored vs voided vs not-applicable, and a
table of the headline raw metrics behind the rollups.

Usage
-----
    python bin/compare_capacity_arms.py \
        --scorecard "AHN=results/ahn_capacity_run/AsymmetricHopfieldNetwork/capacity_scorecard.json" \
        --scorecard "theta=results/zoo_capacity_run/ThetaPhaseSequenceNetwork/capacity_scorecard.json" \
        --out-dir results/zoo_capacity_run/_compare

Reading note carried over from the per-arm radar: **score and coverage are not
one number and must not be multiplied into one when ranking.** A high score at
low coverage is a claim about a fraction of the capacity. Every figure here
shows both, and the bar chart plots them as separate panels for that reason.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

CAPACITIES = [
    "Continual retention",
    "One-shot learning",
    "Pattern completion",
    "Sequence disambiguation",
    "Serial order",
]
SHORT = {
    "Continual retention": "Continual\nretention",
    "One-shot learning": "One-shot\nlearning",
    "Pattern completion": "Pattern\ncompletion",
    "Sequence disambiguation": "Sequence\ndisambiguation",
    "Serial order": "Serial\norder",
}

# Colour-blind-safe qualitative set; the reference arm is deliberately grey so
# the new arms read against it rather than competing with it.
PALETTE = ["#8c8c8c", "#2e6296", "#B07A16", "#3F7A4F", "#AD3F6B", "#5B4B8A",
           "#1F7A8C"]

#: The five statuses ``score_capacities.py`` emits, plus ``absent`` for a
#: dimension one arm's scorecard carries and another's does not. The three
#: non-scored ones are deliberately different colours because they are
#: different claims and the report turns on telling them apart:
#:   unresolved     -- a guard failed. The section ran; its manipulation had no
#:                     dynamic range for this arm, so its numbers carry no
#:                     information. The ARM may be above or below the
#:                     instrument's range.
#:   not_applicable -- the arm does not declare the capability the row needs.
#:                     A property of the ARM.
#:   unbuilt        -- MemVal has no section wired for this row at all. A
#:                     property of the SUITE, identical for every arm.
#:   unrunnable     -- the feeding section produced no finite metric (it was not
#:                     run, or it errored). A property of the RUN, and the one
#:                     status that should be chased down rather than reported.
STATUS_COLOR = {
    "scored": "#3F7A4F",
    "unresolved": "#B07A16",
    "not_applicable": "#9a9a9a",
    "unbuilt": "#5B4B8A",
    "unrunnable": "#C0392B",
    "protocol_limited": "#AD3F6B",
    "absent": "#d8d8d8",
}


def load(specs: List[str]) -> List[Tuple[str, dict]]:
    out = []
    for s in specs:
        name, _, path = s.partition("=")
        if not path:
            path = name
            name = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            out.append((name, json.load(f)))
    return out


def _score(d, cap):
    v = d["capacities"][cap]["score"]
    return 0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


def _cov(d, cap):
    return float(d["capacities"][cap]["coverage"])


# ------------------------------------------------------------------ radar ---
def radar(arms, path):
    """Every arm's capacity score on one set of axes.

    Coverage is NOT drawn on the polygon -- a coverage-discounted radar invites
    reading a missing instrument as a model failure, which is the one confusion
    the scorecard exists to prevent. It is annotated per vertex instead, and the
    bar figure gives it its own panel.
    """
    N = len(CAPACITIES)
    ang = [n / N * 2 * math.pi for n in range(N)] + [0.0]

    fig = plt.figure(figsize=(11.5, 10.4), facecolor="white")
    ax = fig.add_axes([0.235, 0.235, 0.53, 0.53], polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")

    theta = np.linspace(0, 2 * math.pi, 240)
    for r in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.plot(theta, [r] * 240, color="#e2e2e2", lw=0.7, zorder=0)
    for a in ang[:-1]:
        ax.plot([a, a], [0, 1.0], color="#e8e8e8", lw=0.7, zorder=0)

    for i, (name, d) in enumerate(arms):
        col = PALETTE[i % len(PALETTE)]
        v = [_score(d, c) for c in CAPACITIES]
        v += v[:1]
        ref = i == 0
        ax.plot(ang, v, color=col, lw=3.0 if ref else 2.1,
                ls=(0, (5, 3)) if ref else "-", zorder=3 + i, label=name)
        if ref:
            ax.fill(ang, v, color=col, alpha=0.10, zorder=2)
        ax.scatter(ang[:-1], v[:-1], s=42, color=col, zorder=10 + i,
                   edgecolor="white", linewidth=1.2)

    ax.set_xticks([])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8,
                       color="#a8a8a8")
    ax.set_rlabel_position(36)
    ax.grid(False)
    ax.spines["polar"].set_color("#d2d2d2")

    for a, c in zip(ang[:-1], CAPACITIES):
        ux, uy = math.sin(a), math.cos(a)
        ha = "center" if abs(ux) < 1e-6 else ("left" if ux > 0 else "right")
        covs = {round(_cov(d, c), 3) for _, d in arms}
        cov_txt = (f"coverage {list(covs)[0]:.0%}" if len(covs) == 1
                   else f"coverage {min(covs):.0%}–{max(covs):.0%}")
        ax.annotate(SHORT[c].replace("\n", " "), xy=(a, 1.0), xycoords="data",
                    xytext=(30 * ux, 30 * uy + 8), textcoords="offset points",
                    ha=ha, va="center", fontsize=12.5, fontweight="bold",
                    color="#1b1b1b")
        ax.annotate(cov_txt, xy=(a, 1.0), xycoords="data",
                    xytext=(30 * ux, 30 * uy - 8), textcoords="offset points",
                    ha=ha, va="center", fontsize=9, color="#8a8a8a")

    fig.text(0.5, 0.955, "MemVal capacity profile — the model zoo",
             ha="center", fontsize=19, fontweight="bold", color="#1b1b1b")
    fig.text(0.5, 0.925,
             "capacity score: weighted mean over VALIDATED metrics only, on "
             "identical axes. AHN is the reference (dashed grey).",
             ha="center", fontsize=10, color="#7a7a7a")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), fontsize=10.5,
              frameon=False, ncol=3, handlelength=2.6, columnspacing=2.2)
    fig.text(0.5, 0.028,
             "Coverage is annotated, never multiplied in: a low-coverage vertex "
             "means the suite lacks the instrument or the arm lacks the\n"
             "capability, NOT that the arm failed. Read the bar figure's second "
             "panel before ranking anything on the polygon above.",
             ha="center", va="bottom", fontsize=8.8, color="#8a8a8a",
             linespacing=1.7)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------- bars ---
def bars(arms, path):
    """Score and coverage as two panels, never as one product."""
    n = len(arms)
    x = np.arange(len(CAPACITIES))
    w = 0.8 / n

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 9.2), facecolor="white",
                             gridspec_kw=dict(height_ratios=[2.0, 1.0],
                                              hspace=0.30))
    for i, (name, d) in enumerate(arms):
        col = PALETTE[i % len(PALETTE)]
        off = (i - (n - 1) / 2) * w
        s = [_score(d, c) for c in CAPACITIES]
        cv = [_cov(d, c) for c in CAPACITIES]
        axes[0].bar(x + off, s, w * 0.92, color=col, label=name,
                    edgecolor="white", linewidth=0.8)
        axes[1].bar(x + off, cv, w * 0.92, color=col, edgecolor="white",
                    linewidth=0.8)
        for xi, v in zip(x + off, s):
            axes[0].text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=7.6, color="#4a4a4a", rotation=90)

    for ax, ylab, title in (
        (axes[0], "capacity score",
         "Capacity score — weighted mean over validated metrics"),
        (axes[1], "coverage",
         "Coverage — share of the capacity's intended dimension weight that "
         "produced a score"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[c] for c in CAPACITIES], fontsize=10.5)
        ax.set_ylim(0, 1.12 if ax is axes[0] else 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left",
                     color="#1b1b1b", pad=8)
        ax.grid(axis="y", color="#ececec", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#cccccc")

    axes[1].axhline(1.0, color="#bbbbbb", lw=0.9, ls=(0, (3, 3)))
    axes[0].legend(fontsize=10, frameon=False, ncol=min(n, 4),
                   loc="upper center", bbox_to_anchor=(0.5, 1.30))
    fig.text(0.5, 0.015,
             "The two panels answer different questions and a single ranking "
             "cannot serve both. A tall bar above a short one means "
             "“scored well on the part of the capacity that could be "
             "measured”.",
             ha="center", fontsize=8.8, color="#8a8a8a")
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- dimension status ---
def dimension_status(arms, path):
    """Which dimension of which capacity produced a score, for each arm.

    This is the figure that keeps the scores honest: the same capacity number
    can rest on four dimensions for one arm and two for another, and only this
    view shows which.
    """
    cells, rows = [], []
    for cap in CAPACITIES:
        dims = []
        for _, d in arms:
            dims += list(d["capacities"][cap]["dimensions"].keys())
        seen = []
        for dm in dims:
            if dm not in seen:
                seen.append(dm)
        for dm in seen:
            rows.append((cap, dm))

    fig, ax = plt.subplots(figsize=(11.6, 0.42 * len(rows) + 2.6),
                           facecolor="white")
    for r, (cap, dm) in enumerate(rows):
        for c, (_, d) in enumerate(arms):
            info = d["capacities"][cap]["dimensions"].get(dm)
            st = info["status"] if info else "absent"
            col = STATUS_COLOR.get(st, "#d8d8d8")
            ax.add_patch(plt.Rectangle((c, r), 0.94, 0.88, color=col,
                                       linewidth=0))
            if info and st == "scored":
                ax.text(c + 0.47, r + 0.44, f"{info['score']:.2f}",
                        ha="center", va="center", fontsize=8.4, color="white",
                        fontweight="bold")
            w_ = info["weight"] if info else None
            if c == 0 and w_ is not None:
                ax.text(-0.12, r + 0.44, f"{dm}  (w {w_:.2f})", ha="right",
                        va="center", fontsize=9.2, color="#3a3a3a")
    # capacity separators
    prev = None
    for r, (cap, _) in enumerate(rows):
        if cap != prev:
            ax.axhline(r, color="#bbbbbb", lw=1.1)
            ax.text(-0.12, r + 0.44 + 0.0, "", ha="right")
            prev = cap
    # capacity band labels on the right
    prev, start = rows[0][0], 0
    for r in range(len(rows) + 1):
        cap = rows[r][0] if r < len(rows) else None
        if cap != prev:
            ax.text(len(arms) + 0.15, (start + r) / 2, prev.replace(" ", "\n"),
                    ha="left", va="center", fontsize=10.5, fontweight="bold",
                    color="#1b1b1b")
            prev, start = cap, r

    ax.set_xlim(-3.5, len(arms) + 1.9)
    ax.set_ylim(len(rows), -0.4)
    ax.set_xticks([c + 0.47 for c in range(len(arms))])
    ax.set_xticklabels([n for n, _ in arms], fontsize=9.6, rotation=28,
                       ha="left")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    handles = [
        Patch(color=STATUS_COLOR["scored"], label="scored"),
        Patch(color=STATUS_COLOR["unresolved"],
              label="guard failed — section had no dynamic range for this arm"),
        Patch(color=STATUS_COLOR["not_applicable"],
              label="not applicable — the ARM lacks the declared capability"),
        Patch(color=STATUS_COLOR["unbuilt"],
              label="unbuilt — the SUITE has no section wired for this row"),
        Patch(color=STATUS_COLOR["unrunnable"],
              label="unrunnable — the RUN emitted no finite metric (chase it)"),
        Patch(color=STATUS_COLOR["absent"],
              label="dimension absent from this arm's scorecard"),
    ]
    ax.legend(handles=handles, fontsize=9, frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.45, -0.02 - 1.4 / len(rows)))
    fig.suptitle("What each capacity score actually rests on",
                 fontsize=15, fontweight="bold", color="#1b1b1b", y=0.995)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- headline table --
#: (capacity, metric key, label, "hi"|"lo", format). Chosen because each is the
#: *raw* quantity a capacity paragraph in the report is written about, so a
#: reader can check a claim without opening five JSON files. Keys are read from
#: the full metric list, not just role="score", so a read-out a guard demoted to
#: diagnostic (e.g. mask_identity_tolerance under mask_at_ceiling) still shows
#: -- it is excluded from the ROLLUP, which is not a reason to hide the number.
HEADLINE = [
    # Emitted as after - before, so 0 is ideal and NEGATIVE is forgetting.
    # Scored "hi" for that reason; "lo" would shade the worst arm as the best.
    ("Continual retention", "delta_mrr_forgetting",
     "ΔMRR on A after B (0 = none)", "hi", "{:+.3f}"),
    ("Continual retention", "chain_avg_accuracy",
     "chain ACC (6 tasks)", "hi", "{:.3f}"),
    ("Continual retention", "chain_avg_forgetting",
     "chain avg forgetting", "lo", "{:.3f}"),
    ("Continual retention", "pa_abac_ab_recall_final",
     "AB retention after AC", "hi", "{:.3f}"),
    ("Continual retention", "pa_cue_competition_cost",
     "cue-competition cost", "lo", "{:.3f}"),
    ("Continual retention", "reversal_direct_reversal_trials_to_criterion",
     "reversal trials to criterion", "lo", "{:.1f}"),
    ("Continual retention", "reversal_direct_reversal_final_perseveration",
     "final perseveration", "lo", "{:.3f}"),
    ("One-shot learning", "exposure_baseline_epochs",
     "baseline exposure (epochs)", "lo", "{:.0f}"),
    ("One-shot learning", "convergence_epochs",
     "epochs to convergence", "lo", "{:.0f}"),
    ("One-shot learning", "convergence_mrr", "MRR at convergence", "hi",
     "{:.3f}"),
    ("One-shot learning", "online_mrr_1shot",
     "MRR, one streamed pass", "hi", "{:.3f}"),
    ("One-shot learning", "online_mrr_20pass",
     "MRR, 20 streamed passes", "hi", "{:.3f}"),
    ("One-shot learning", "schema_consistency_speed_corr",
     "schema-consistency x speed (corr)", "lo", "{:+.3f}"),
    ("Pattern completion", "noise_tolerance_threshold",
     "σ-tolerance", "hi", "{:.3f}"),
    ("Pattern completion", "mask_random_tolerance",
     "mask tolerance (random)", "hi", "{:.3f}"),
    ("Pattern completion", "mask_identity_tolerance",
     "mask tolerance (identity)", "hi", "{:.3f}"),
    ("Pattern completion", "mask_shared_tolerance",
     "mask tolerance (category)", "hi", "{:.3f}"),
    ("Pattern completion", "tmaze_pc_coverage",
     "route coverage from a fragment", "hi", "{:.3f}"),
    ("Sequence disambiguation", "symdis_max_episodes_above_chance",
     "confusable episodes above chance", "hi", "{:.0f}"),
    ("Sequence disambiguation", "symdis_similarity_tolerance_threshold",
     "discriminator-similarity crossing", "hi", "{:.3f}"),
    # 4.3, the ⚑ StatePrimeable row. Zero for every arm under the memoryless
    # one-step probe, so a non-zero value here is the whole point of the tag.
    ("Sequence disambiguation", "symdis_max_delay_at_fixed_support",
     "delay the discriminator survives", "hi", "{:.0f}"),
    ("Sequence disambiguation", "symdis_middle_endogenous_accuracy",
     "accuracy from carried state alone", "hi", "{:.3f}"),
    ("Sequence disambiguation", "symdis_load_disambiguation_cost",
     "overlap cost vs orthogonal control", "lo", "{:.3f}"),
    ("Sequence disambiguation", "similarity_exposure_cost",
     "exposure cost at high similarity", "lo", "{:.1f}x"),
    ("Sequence disambiguation", "tmaze_disamb_full_branch_acc",
     "T-maze branch accuracy", "hi", "{:.3f}"),
    ("Serial order", "max_memory_span", "max memory span (rollout)", "hi",
     "{:.1f}"),
    ("Serial order", "max_memory_span_quantized",
     "max span, quantized feedback", "hi", "{:.1f}"),
    ("Serial order", "unrolling_gap", "unrolling gap at criterion", "lo",
     "{:.3f}"),
    ("Serial order", "unrolling_exposure_ratio", "unrolling exposure ratio",
     "lo", "{:.1f}x"),
    ("Serial order", "establishment_break_length",
     "length at which order breaks", "hi", "{:.0f}"),
]


def headline_table(arms, path_png, path_md):
    """The raw metrics behind the rollups, one table, all arms."""
    index = [{m["key"]: m for m in d["metrics"]} for _, d in arms]

    rows = []
    for cap, key, label, direction, fmt in HEADLINE:
        vals = [ix.get(key, {}).get("raw") for ix in index]
        if all(v is None for v in vals):
            continue
        rows.append((cap, label, direction, fmt, vals))

    def cell(v, fmt):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        # An infinite exposure ratio is a CENSORED measurement -- the criterion
        # was never reached inside the budget -- not a very large number, and
        # "infx" is not a reading. Say so.
        if math.isinf(f):
            return "censored"
        return fmt.format(f)

    # ---- markdown
    with open(path_md, "w") as f:
        f.write("| capacity | metric | " + " | ".join(n for n, _ in arms) + " |\n")
        f.write("|---" * (len(arms) + 2) + "|\n")
        prev = None
        for cap, label, direction, fmt, vals in rows:
            arrow = "↑" if direction == "hi" else "↓"
            f.write(f"| {cap if cap != prev else ''} | {label} {arrow} | "
                    + " | ".join(cell(v, fmt) or "—" for v in vals) + " |\n")
            prev = cap

    # ---- png: shading is the rank WITHIN a row, so the colour never compares
    # quantities that are not commensurable; the printed number is always raw.
    fig, ax = plt.subplots(figsize=(2.05 * len(arms) + 6.4,
                                    0.50 * len(rows) + 2.8), facecolor="white")
    prev = None
    for r, (cap, label, direction, fmt, vals) in enumerate(rows):
        fin = [float(v) for v in vals
               if isinstance(v, (int, float)) and math.isfinite(float(v))]
        lo_, hi_ = (min(fin), max(fin)) if fin else (0.0, 1.0)
        # A row whose arms are all within a hair of each other has no rank to
        # show. Shading it anyway turns 0.007-vs-0.000 into "best in row", which
        # is how a reader ends up believing a rounding difference is a result.
        flat = (hi_ - lo_) <= 0.02 * max(1.0, abs(hi_))
        for c, v in enumerate(vals):
            txt = cell(v, fmt)
            if txt == "censored":
                ax.add_patch(plt.Rectangle((c, r), 0.94, 0.86, color="#f6e3e1"))
                ax.text(c + 0.47, r + 0.43, "censored", ha="center",
                        va="center", fontsize=8.4, color="#9c3a2e")
                continue
            if txt is None:
                ax.add_patch(plt.Rectangle((c, r), 0.94, 0.86, color="#f2f2f2"))
                ax.text(c + 0.47, r + 0.43, "—", ha="center", va="center",
                        fontsize=9, color="#b0b0b0")
                continue
            if flat:
                t = 0.5
            else:
                t = (float(v) - lo_) / (hi_ - lo_)
                if direction == "lo":
                    t = 1.0 - t
            col = plt.get_cmap("BuGn")(0.10 + 0.72 * t)
            ax.add_patch(plt.Rectangle((c, r), 0.94, 0.86, color=col))
            ax.text(c + 0.47, r + 0.43, txt, ha="center", va="center",
                    fontsize=9.2, color="white" if t > 0.62 else "#2a2a2a",
                    fontweight="bold" if t > 0.62 else "normal")
        arrow = "↑" if direction == "hi" else "↓"
        ax.text(-0.15, r + 0.43, f"{label}  {arrow}", ha="right", va="center",
                fontsize=9.6, color="#2a2a2a")
        if cap != prev:
            ax.axhline(r - 0.07, color="#b8b8b8", lw=1.0)
            prev = cap

    # capacity band labels on the right
    prev, start = rows[0][0], 0
    for r in range(len(rows) + 1):
        cap = rows[r][0] if r < len(rows) else None
        if cap != prev:
            ax.text(len(arms) + 0.2, (start + r) / 2 + 0.43,
                    prev.replace(" ", "\n"), ha="left", va="center",
                    fontsize=10.5, fontweight="bold", color="#1b1b1b")
            prev, start = cap, r

    ax.set_xlim(-6.0, len(arms) + 2.0)
    ax.set_ylim(len(rows), -0.5)
    ax.set_xticks([c + 0.47 for c in range(len(arms))])
    ax.set_xticklabels([n for n, _ in arms], fontsize=10, rotation=22, ha="left")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.suptitle("Headline raw metrics behind the rollups",
                 fontsize=15, fontweight="bold", color="#1b1b1b", y=1.005)
    fig.text(0.5, -0.015,
             "Shading is the rank WITHIN a row (darker = better in that row's "
             "own direction: ↑ higher-is-better, ↓ lower-is-better); the "
             "printed number is always the raw value.\nA dash is a metric the "
             "arm's run did not emit — a guard voided its section, or the arm "
             "lacks the capability. It is never a zero.",
             ha="center", fontsize=8.8, color="#8a8a8a", linespacing=1.7)
    fig.savefig(path_png, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorecard", action="append", required=True,
                    help="NAME=path/to/capacity_scorecard.json (repeatable). "
                         "The FIRST one is drawn as the reference arm.")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    arms = load(a.scorecard)
    os.makedirs(a.out_dir, exist_ok=True)
    radar(arms, os.path.join(a.out_dir, "zoo_radar.png"))
    bars(arms, os.path.join(a.out_dir, "zoo_capacity_bars.png"))
    dimension_status(arms, os.path.join(a.out_dir, "zoo_dimension_status.png"))
    headline_table(arms, os.path.join(a.out_dir, "zoo_headline_metrics.png"),
                   os.path.join(a.out_dir, "zoo_headline_metrics.md"))

    # the profile table, as the report consumes it
    prof = {n: {c: dict(score=_score(d, c), coverage=_cov(d, c))
                for c in CAPACITIES} for n, d in arms}
    with open(os.path.join(a.out_dir, "zoo_profiles.json"), "w") as f:
        json.dump(prof, f, indent=2)

    print(f"wrote 4 figures + zoo_profiles.json to {a.out_dir}")


if __name__ == "__main__":
    main()
