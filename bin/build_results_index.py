#!/usr/bin/env python
"""Static index for a capacity run: one browsable entry point over every artefact.

A run writes ~120 PNGs across nested per-arm directories plus a self-contained
scorecard page each. That is complete but not explorable -- there is no page
that puts an arm's section plots next to the scorecard those plots produced.
This builds one, with no dependencies and no copying: every link is relative to
the output directory, so the tree stays the single source of truth and the index
can be regenerated at any time.

    python bin/build_results_index.py --results-dir results/zoo_capacity_run

Writes ``<results-dir>/index.html``. Open it directly; no server needed.

Why the sections are grouped by *suite* and not by capacity: the scorecard page
already presents metrics by capacity, and it is built from ``metrics.json``
rather than from the plots. Keeping this index in run order means the two views
stay independent, which is what makes them useful as a cross-check -- if a
scorecard number disagrees with the section plot that fed it, that is a scoring
bug rather than a model result.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

CAPACITIES = ["Continual retention", "One-shot learning", "Pattern completion",
              "Sequence disambiguation", "Serial order"]

#: Display name for a class directory. Anything not listed falls back to the
#: directory name, so a new arm appears without editing this.
DISPLAY = {
    "AsymmetricHopfieldNetwork": "AHN (reference)",
    "ThetaPhaseSequenceNetwork": "theta",
    "OriginalEqPropSequenceNetwork": "EP",
    "SpikingEqPropSequenceNetwork": "spiking EP",
    "MultilayerTemporalPCNetwork": "tPC",
    "PredictiveRecirculationNetwork": "recirc (Chen)",
    "DTSESNSequenceNetwork": "DTS-ESN",
}

#: One line per section plot, so a reader knows what they are looking at before
#: they open it. Keyed by PNG basename.
CAPTION = {
    "tmaze_completion": "Route completion from a fragment, open-loop.",
    "tmaze_disambiguation": "Branch choice with the odour cue present.",
    "tmaze_disambiguation_graded": "Branch choice as the cue is withdrawn earlier.",
    "tmaze_reversal": "Place→reward reversal: extinction and re-acquisition.",
    "convergence_curve": "MRR against exposure — the presentation-duration sweep.",
    "length_curves": "Serial-position curves by list length.",
    "cognitive_serial_position": "L0: cued serial-position curve at L=7 per exposure; rollout beside it (cannot show recency).",
    "cognitive_list_length": "L0: early/middle/late recall against list length; human: longer lists cost early/middle, not the last few.",
    "cognitive_presentation_rate": "L0: early/middle/late recall against passes; human: slower rate helps early/middle, not the last few.",
    "cognitive_pli_recency": "L0: prior-list intrusions and runners-up by list lag; human: most from the last two lists.",
    "cognitive_semantic_clustering": "L0: errors classified cue-partner / target-partner / chain-from-partner across a within-cosine sweep.",
    "multiple_seq_forgetting": "List A before and after learning list B.",
    "continual_chain_retention": "Retention matrix over the 6-task chain.",
    "continual_chain_axes": "Load curve, diagonal vs final row, and epochs-to-criterion at every "
                            "chain position: acquisition paired with what it cost.",
    "continual_chain_overlap_matrices": "The chain's retention matrix along both overlap dials: "
                                        "between-task cosine (top) and within-task cosine (bottom).",
    "continual_chain_overlap_axes": "Exposure, acquisition, survival and the load curve at every "
                                    "rung of each overlap dial, with LA / ACC / SPI against the dial.",
    "continual_chain_overlap_intrusions": "Where the errors go (same task, earlier task, later task) "
                                          "and who the runner-up is, per rung of each overlap dial.",
    "continual_chain_usage": "Does retention track future use? Margin per task along a use-structured "
                             "stream, with the disuse control and the recency condition.",
    "continual_chain_selectivity": "Rehearsal: what is kept under capacity pressure.",
    "paired_associate_abac": "AB/AC interference and cue competition.",
    "noise_invariance": "Cued recall against cue-noise sigma, and rollout from a corrupted "
                        "initial cue only (how the initial error propagates).",
    "cue_masking": "Recall as cue features are removed, by block type.",
    "semantic_similarity": "Exposure and recall against within-category overlap.",
    "symbolic_disambiguation": "Six axes at the fork: discriminator similarity, load (with orthogonal "
                               "control), withdrawal, support vs delay decoupled, shared-stretch "
                               "length under persistent and onset-only support, shared middle.",
    "schema_consistency": "Acquisition rate against consistency with prior knowledge.",
    "online_convergence": "Streamed presentations to criterion.",
    "serial_order_failures": "From the serial-order probe: was order established (order | item vs "
                             "list membership), what a cued failure is, and where transpositions land.",
    "serial_order_overlap_recall_length": "Three blocked 10-item lists along the within- and "
                                          "between-list overlap dials: steps a free rollout gets "
                                          "right before its first error, for every list after "
                                          "every stage (lower triangle, like the retention matrix).",
    "serial_order_overlap_failures": "The same sweep's cued failures after the last list: "
                                     "transposition, prior- / later-list intrusion, or a "
                                     "never-studied lure, per list and rung.",
    "interval_generation_abcd_seeds": "The 5.5 protocol on one custom stream, A -1.2s- B -10s- "
                                      "C -1.2s- D, autonomous rollout from A over 5 seeds: "
                                      "generated gap per step, seed mean ± SD with every seed "
                                      "drawn. Not scored (bin/interval_generation_abcd.py).",
    "serial_order_unrolling": "From the serial-order probe: cued vs rollout span (the unrolling gap), "
                              "exposure to cued vs rollout criterion, and whether an error is terminal.",
    "extinction_route_survival": "Is the route still there? Direction imposed at the fork, then a free "
                                 "rollout to the cheese: how far each predicted position lands from the "
                                 "true one, scaled so 0 = as wrong as naming the other arm. No reward "
                                 "channel involved.",
    "extinction_choice_margin": "What comes after the choice point? Cued at the maze entrance, the model "
                                "unrolls the stem itself; the state it produces after the last stem "
                                "position is scored against the two possible answers, the first step of "
                                "each arm. Signed toward the extinguished arm.",
    "extinction_timeline_variants": "Every candidate readout for the same timeline, one per row, "
                                    "including the outcome channel and the free-rollout commit.",
    "sparse_capacity_curves": "Cued recall against raw autoregressive unroll as sparse items are "
                              "ingested: one 100-item chain vs ten 10-item sequences (with a "
                              "no-category-core control). Fixed exposure, not a scored section.",
    "sparse_capacity_positions": "The same raw unroll resolved BY POSITION, one line per "
                                 "checkpoint. The capacity curves average these over positions, "
                                 "which cannot distinguish a short reliable prefix from scattered "
                                 "hits; the marked prefix is the contiguous fully-correct run.",
    "cue_availability": "A whole modality absent from the cue (item without its sound, sound "
                        "without its item; one tone per item), one sequence and three; exact "
                        "next-item recall, cued and rolled out, with margins.",
    "interval_retention": "Branch chosen from the elapsed gap alone, against the ordinal-twin control.",
    "interval_generation": "Generated vs trained gap per step: tempo and the placed pause.",
    "isi_tolerance": "Recall against inter-stimulus noise steps.",
}

#: Main-text figures, per capacity, in reading order, keyed by PNG stem. Every
#: plot NOT listed here is rendered under "Supplementary" for its suite. This is
#: the one place the main/supplementary split is declared; edit it here and
#: rebuild. Capacities with an empty list have not had their main set decided
#: yet and contribute nothing to the Main block.
MAIN_FIGURES = {
    "Continual retention": [
        "continual_chain_overlap_matrices",     # stability along both overlap dials
        "continual_chain_overlap_axes",         # intransigence, survival, load curve
        "paired_associate_abac",                # revision plasticity: forgetting scored as success
        "continual_chain_overlap_intrusions",   # where the errors go (attribution)
        # Extinction: the trace that stopped paying. Route survival first --
        # it needs no reward channel, so it is the half that means the same
        # thing for every arm (bin/extinction_timeline.py).
        "extinction_route_survival",            # does the invalidated route survive being extinguished?
        "extinction_choice_margin",             # and is it still the one the model would take?
        "sparse_capacity_curves",               # retention at load: 1x100 vs 10x10 sparse items
    ],
    "One-shot learning": [
        "convergence_curve",                    # exposure x recall: the presentation sweep
        "online_convergence",                   # the strict single pass: streamed, event by event
        "continual_chain_axes",                 # acquisition under load, paired with what it cost
        "schema_consistency",                   # is acquisition faster when the item fits?
    ],
    "Pattern completion": [
        "noise_invariance",                     # corrupted cue: cued sweep + rollout from a noisy initial cue
        "cue_masking",                          # partial cue: cued, margin, rollout from a masked initial cue
        "cue_availability",                     # absent modality: item-only and tone-only cues, single and multi
        "tmaze_completion",                     # route fragment, open-loop: where the rollout leaves the route
    ],
    "Sequence disambiguation": [
        "symbolic_disambiguation",              # support, load, delay, stretch length, shared middle
        "tmaze_disambiguation",                 # spatial: branch choice with the odour cue present
        "tmaze_disambiguation_graded",          # spatial: odour withdrawn earlier and earlier
    ],
    "Serial order": [
        "serial_order_failures",                # order never established; recalled but out of sequence
        "serial_order_unrolling",               # associations present, autonomous unrolling collapses
        "interval_retention",                   # is the interval between two items retained (clocked arms)
    ],
}
MAIN_STEMS = {stem for stems in MAIN_FIGURES.values() for stem in stems}

SUITES = [("spatial", "Spatial suite"), ("symbolic", "Symbolic suite"),
          ("online_symbolic", "Streamed regime"),
          # Standalone, not scored sections: each writes a suite-shaped directory
          # so its figures land here.
          ("sparse_capacity", "Sparse-code capacity (standalone)"),
          ("extinction", "Extinction \u2014 preference and trace (standalone)"),
          ("interval_generation_abcd",
           "Interval generation, A\u2013B\u2013C\u2013D rhythm (standalone)")]

CSS = """
:root{--bg:#fbfbfc;--fg:#1b1b1b;--mut:#6b7280;--line:#e2e5ea;--card:#fff;
      --ok:#3F7A4F;--warn:#B07A16;--na:#9a9a9a;--bad:#C0392B;--acc:#2e6296}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 72px}
h1{font-size:25px;margin:0 0 4px} h2{font-size:18px;margin:38px 0 12px}
h4{font-size:13px;margin:16px 0 6px;color:var(--fg);font-weight:600}
h3{font-size:14px;margin:22px 0 10px;color:var(--mut);font-weight:600;
   text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--mut);margin:0 0 26px}
a{color:var(--acc);text-decoration:none} a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%;background:var(--card);
      border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;font-size:13px}
th{background:#f4f6f8;font-weight:600;color:#4a5462}
tr:last-child td{border-bottom:none}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.cov{color:var(--mut);font-size:11px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
      padding:10px;display:block}
.card img{width:100%;border-radius:4px;display:block;background:#fff}
.card .t{font-weight:600;margin:8px 2px 2px;font-size:13px}
.card .c{color:var(--mut);font-size:11.5px;margin:0 2px}
.arm{background:var(--card);border:1px solid var(--line);border-radius:10px;
     padding:20px 22px;margin:0 0 22px}
.arm h2{margin-top:0}
.links a{display:inline-block;margin:0 14px 0 0;font-size:13px}
.pill{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;
      color:#fff;margin-left:6px}
.note{background:#fffdf5;border:1px solid #eadfc0;border-radius:8px;
      padding:12px 15px;margin:14px 0;font-size:13px;color:#5c4d2a}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
"""


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _load(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _plots(arm_dir: str, suite: str) -> List[str]:
    d = os.path.join(arm_dir, suite, "plots")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))


def _cards(paths: List[str], root: str, extra: Dict[str, str] = None) -> str:
    extra = extra or {}
    out = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        cap = extra.get(stem) or CAPTION.get(stem, "")
        r = _rel(p, root)
        out.append(
            f'<a class="card" href="{html.escape(r)}" target="_blank">'
            f'<img src="{html.escape(r)}" loading="lazy" alt="{html.escape(stem)}">'
            f'<div class="t">{html.escape(stem)}</div>'
            f'<div class="c">{html.escape(cap)}</div></a>')
    return f'<div class="grid">{"".join(out)}</div>' if out else ""


def _profile_row(name: str, caps: dict) -> str:
    cells = []
    for c in CAPACITIES:
        v = caps.get(c)
        if not v:
            cells.append("<td>—</td>")
            continue
        s, cov = v.get("score"), v.get("coverage")
        s = "—" if s is None or s != s else f"{s:.3f}"
        cells.append(f'<td class="num">{s}<div class="cov">cov {cov:.0%}</div></td>')
    return f"<tr><td><b>{html.escape(name)}</b></td>{''.join(cells)}</tr>"


def build(results_dir: str) -> str:
    root = os.path.abspath(results_dir)
    arms: List[Tuple[str, str, Optional[dict]]] = []
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if not os.path.isdir(d) or entry.startswith("_") or entry.startswith("."):
            continue
        arms.append((entry, DISPLAY.get(entry, entry), _load(os.path.join(d, "capacity_scorecard.json"))))

    # deterministic order: DISPLAY order first, then anything new alphabetically
    order = {k: i for i, k in enumerate(DISPLAY)}
    arms.sort(key=lambda a: (order.get(a[0], 10_000), a[0]))

    parts: List[str] = []
    parts.append(f"<h1>MemVal capacity run — {html.escape(os.path.basename(root))}</h1>")
    parts.append(f'<p class="sub">{len(arms)} arms · '
                 f'generated {datetime.now():%Y-%m-%d %H:%M} · '
                 f'every link is relative to this directory, so the run tree stays '
                 f'the source of truth.</p>')

    # ---- cross-arm -------------------------------------------------------
    cmp_dir = os.path.join(root, "_compare")
    if os.path.isdir(cmp_dir):
        parts.append("<h2>Cross-arm comparison</h2>")
        pngs = sorted(os.path.join(cmp_dir, f) for f in os.listdir(cmp_dir)
                      if f.endswith(".png"))
        parts.append(_cards(pngs, root, {
            "zoo_radar": "All arms on identical axes; AHN dashed grey.",
            "zoo_capacity_bars": "Score and coverage as separate panels — never multiplied.",
            "zoo_dimension_status": "What each capacity score rests on, dimension by dimension.",
            "zoo_headline_metrics": "The raw metrics behind the rollups.",
        }))
        side = [f for f in ("zoo_headline_metrics.md", "zoo_profiles.json")
                if os.path.exists(os.path.join(cmp_dir, f))]
        if side:
            links = " · ".join(
                f'<a href="_compare/{f}">{f}</a>' for f in side)
            parts.append(f'<p class="sub" style="margin-top:14px">{links}</p>')

    # ---- profile table ---------------------------------------------------
    scored = [(disp, sc) for _, disp, sc in arms if sc]
    if scored:
        parts.append("<h2>Profiles</h2><table><tr><th>arm</th>"
                     + "".join(f"<th>{html.escape(c)}</th>" for c in CAPACITIES)
                     + "</tr>"
                     + "".join(_profile_row(d, sc.get("capacities", {}))
                               for d, sc in scored)
                     + "</table>")
        parts.append('<div class="note"><b>Read score and coverage together.</b> '
                     'A high score at low coverage is a claim about the part of the '
                     'capacity that could be measured, not about the whole of it. '
                     'The dimension-status figure above says which parts those were.</div>')

    # ---- per arm ---------------------------------------------------------
    parts.append("<h2>By arm</h2>")
    for cls, disp, sc in arms:
        d = os.path.join(root, cls)
        parts.append('<div class="arm">')
        parts.append(f'<h2>{html.escape(disp)} '
                     f'<span class="mono" style="color:var(--mut);font-weight:400">'
                     f'{html.escape(cls)}</span></h2>')

        links = []
        for label, rel in (("interactive scorecard", "capacity_scorecard.html"),
                           ("per-metric scorecard (md)", "capacity_scorecard.md"),
                           ("scorecard JSON", "capacity_scorecard.json"),
                           ("serial-order probe", "serial_order_probe.json")):
            if os.path.exists(os.path.join(d, rel)):
                links.append(f'<a href="{cls}/{rel}">{label}</a>')
        parts.append(f'<div class="links">{"".join(links)}</div>')

        radar = os.path.join(d, "capacity_scorecard_radar.png")
        radar_m = os.path.join(d, "capacity_scorecard_radar_margin.png")
        if os.path.exists(radar):
            parts.append("<h3>Capacity profile</h3>")
            parts.append(_cards([p for p in (radar, radar_m) if os.path.exists(p)], root,
                                {"capacity_scorecard_radar":
                                 "Score, inclusive, and score×coverage on one set of axes.",
                                 "capacity_scorecard_radar_margin":
                                 "ALTERNATE scoring: Continual retention's forgetting read from "
                                 "delta_margin_forgetting_lists (graded margin over list pairs) "
                                 "instead of delta_mrr_forgetting. Not the published profile."}))

        # ---- main figures, grouped by capacity ---------------------------
        by_stem = {os.path.splitext(os.path.basename(p))[0]: p
                   for suite, _ in SUITES for p in _plots(d, suite)}
        main_blocks = []
        for cap, stems in MAIN_FIGURES.items():
            present = [by_stem[st] for st in stems if st in by_stem]
            if present:
                main_blocks.append(f'<h4>{html.escape(cap)}</h4>' + _cards(present, root))
        if main_blocks:
            parts.append("<h3>Main figures</h3>")
            parts.append('<p class="sub">The figures that answer each capacity\'s promise, '
                         'in reading order. Everything else is under Supplementary. '
                         'Declared in <span class="mono">MAIN_FIGURES</span> in '
                         '<span class="mono">bin/build_results_index.py</span>.</p>')
            parts.extend(main_blocks)
            undecided = [c for c, st in MAIN_FIGURES.items() if not st]
            if undecided:
                parts.append('<p class="sub">Main set not yet decided for: '
                             + ", ".join(html.escape(c) for c in undecided) + '.</p>')

        for suite, title in SUITES:
            pngs = [p for p in _plots(d, suite)
                    if os.path.splitext(os.path.basename(p))[0] not in MAIN_STEMS]
            mpath = os.path.join(d, suite, "metrics.json")
            title = f"Supplementary — {title}"
            if not pngs:
                m = _load(mpath)
                status = (m or {}).get("metadata", {}).get("status")
                if m is None:
                    continue
                parts.append(f"<h3>{title}</h3>")
                parts.append('<p class="sub">No plots — the suite reported '
                             f'<span class="mono">{html.escape(str(status or "no sections"))}</span>. '
                             'For an arm that does not declare the capability this is a '
                             'capability gap, never a score of zero.</p>')
                continue
            parts.append(f"<h3>{title}</h3>")
            parts.append(_cards(pngs, root))

        foc = os.path.join(root, "_schema_focused", cls, "symbolic", "plots",
                           "schema_consistency.png")
        if os.path.exists(foc):
            parts.append("<h3>Schema, focused protocol</h3>")
            parts.append('<p class="sub">The suite default (<span class="mono">extended</span>) '
                         'is valid for acquisition only; interference and prefix-recall are '
                         'valid only here.</p>')
            parts.append(_cards([foc], root,
                                {"schema_consistency":
                                 "Interference by relatedness, focused protocol."}))
        parts.append("</div>")

    body = "\n".join(parts)
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>MemVal — {html.escape(os.path.basename(root))}</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>{body}</div></body></html>")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default=None,
                    help="Default: <results-dir>/index.html")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results_dir, "index.html")
    with open(out, "w") as f:
        f.write(build(a.results_dir))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
