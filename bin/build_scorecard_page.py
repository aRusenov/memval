#!/usr/bin/env python
"""Render a capacity scorecard JSON (from bin/score_capacities.py) as a single
self-contained HTML page: radar, per-capacity dimension breakdown, the full
metric inventory, and the sweeps the derived metrics come from."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from typing import Any, Dict, List

CAP_ORDER = ["Continual retention", "One-shot learning", "Pattern completion",
             "Sequence disambiguation", "Serial order"]

CAP_BLURB = {
    "Continual retention": "Does earlier material survive later material, can new "
        "material still get in once the substrate is full, can an association that "
        "is no longer true be overwritten — and is what survives the part still "
        "being used?",
    "One-shot learning": "How much exposure does acquisition need, and is it faster "
        "when the new item fits what is already known?",
    "Pattern completion": "Is a stored trace recoverable from a degraded cue?",
    "Sequence disambiguation": "When two episodes share observations, does the right "
        "one come back?",
    "Serial order": "Is order itself stored, and does an unrolled sequence stay on "
        "its trajectory?",
}

ROLE_META = {
    "score":            ("scored", "Enters the weighted sum."),
    "protocol_limited": ("protocol-limited", "Measured, but the probe cannot express the test."),
    "control":          ("control", "An ablation or floor whose low value is the intended result."),
    "reference":        ("reference", "A comparison condition: the baseline an experimental arm is read against."),
    "guard":            ("guard", "A validity check on its own section."),
    "config":           ("config", "A protocol setting, not a measurement."),
    "diagnostic":       ("diagnostic", "Measured, excluded from scoring for a stated reason."),
}

STATUS_META = {
    "scored":         ("scored", "ok"),
    "unresolved":     ("guard failed", "bad"),
    "unbuilt":        ("no instrument", "none"),
    "not_applicable": ("not applicable", "none"),
    "unrunnable":     ("produced nothing", "none"),
}


def e(x):
    return html.escape(str(x), quote=True)


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        f = float(x)
        if math.isnan(f):
            return "—"
        if isinstance(x, int) or f.is_integer() and abs(f) < 1e4 and nd == 0:
            return str(int(f))
        return f"{f:.{nd}f}"
    return str(x)


# ---------------------------------------------------------------- radar ----
def radar_svg(caps: Dict[str, Any], incl: Dict[str, Any]) -> str:
    W = H = 560
    cx, cy, R = W / 2, H / 2 + 6, 176
    n = len(CAP_ORDER)

    def pt(i, r):
        a = -math.pi / 2 + i * 2 * math.pi / n
        return cx + math.cos(a) * R * r, cy + math.sin(a) * R * r

    def val(d, c):
        v = d[c]["score"]
        return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v

    def poly(vals):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in
                        (pt(i, v) for i, v in enumerate(vals)))

    s_prim = [val(caps, c) for c in CAP_ORDER]
    s_incl = [val(incl, c) for c in CAP_ORDER]
    s_disc = [val(caps, c) * caps[c]["coverage"] for c in CAP_ORDER]

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="radar" '
           f'aria-label="Radar chart of five capacity scores">']
    out.append('<title>Capacity scores across the five capacities</title>')

    for r in (0.25, 0.5, 0.75, 1.0):
        ring = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, r) for i in range(n)))
        out.append(f'<polygon points="{ring}" class="ring"/>')
    for i in range(n):
        x, y = pt(i, 1.0)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" class="spoke"/>')
    a_mid = -math.pi / 2 + 0.5 * 2 * math.pi / n     # between two spokes, clear of labels
    for r in (0.25, 0.5, 0.75, 1.0):
        x = cx + math.cos(a_mid) * R * r
        y = cy + math.sin(a_mid) * R * r
        out.append(f'<text x="{x + 6:.1f}" y="{y - 4:.1f}" class="rtick">{r:.2f}</text>')

    out.append(f'<polygon points="{poly(s_incl)}" class="area-incl"/>')
    out.append(f'<polyline points="{poly(s_incl)} {poly(s_incl).split(" ")[0]}" class="line-incl"/>')
    out.append(f'<polygon points="{poly(s_prim)}" class="area-prim"/>')
    out.append(f'<polyline points="{poly(s_prim)} {poly(s_prim).split(" ")[0]}" class="line-prim"/>')
    out.append(f'<polyline points="{poly(s_disc)} {poly(s_disc).split(" ")[0]}" class="line-disc"/>')

    for i, c in enumerate(CAP_ORDER):
        x, y = pt(i, s_disc[i])
        out.append(f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" class="dot-disc"/>')
    for i, c in enumerate(CAP_ORDER):
        x, y = pt(i, s_prim[i])
        cov = caps[c]["coverage"]
        out.append(
            f'<g class="vtx" tabindex="0" role="listitem" '
            f'aria-label="{e(c)}: score {s_prim[i]:.2f}, coverage {cov:.0%}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" class="dot-prim"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" class="hit" '
            f'data-cap="{e(c)}" data-score="{s_prim[i]:.2f}" '
            f'data-incl="{s_incl[i]:.2f}" data-disc="{s_disc[i]:.2f}" '
            f'data-cov="{cov:.0%}"/></g>')

    for i, c in enumerate(CAP_ORDER):
        lx, ly = pt(i, 1.0)
        ux, uy = (lx - cx) / R, (ly - cy) / R
        tx, ty = cx + ux * (R + 30), cy + uy * (R + 30)
        anchor = "middle" if abs(ux) < 0.02 else ("start" if ux > 0 else "end")
        cov = caps[c]["coverage"]
        klass = "cov-low" if cov < 0.6 else ("cov-mid" if cov < 0.85 else "cov-hi")
        words, line, lines = c.split(" "), "", []
        for w in words:
            if len(line + " " + w) > 13 and line:
                lines.append(line); line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        # Anchor the block by the edge nearest the chart, so a label above the
        # circle grows upward and one below grows downward -- never inward over
        # the plot.
        block = (len(lines) - 1) * 16 + 42
        if uy < -0.3:
            y0 = ty - block
        elif uy > 0.3:
            y0 = ty
        else:
            y0 = ty - block / 2
        for k, ln in enumerate(lines):
            out.append(f'<text x="{tx:.1f}" y="{y0 + k*16:.1f}" text-anchor="{anchor}" '
                       f'class="vlabel">{e(ln)}</text>')
        base = y0 + (len(lines) - 1) * 16
        out.append(f'<text x="{tx:.1f}" y="{base + 25:.1f}" '
                   f'text-anchor="{anchor}" class="vscore">{s_prim[i]:.2f}</text>')
        out.append(f'<text x="{tx:.1f}" y="{base + 42:.1f}" '
                   f'text-anchor="{anchor}" class="vcov {klass}">coverage {cov:.0%}</text>')
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------- sweeps ----
def line_chart(series, xlabel, ylabel, caption, note, marks=None, log_x=False,
               y_max=1.0):
    W, H = 560, 230
    pad_l, pad_r, pad_t, pad_b = 46, 16, 14, 40
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    x0, x1 = min(xs), max(xs)
    if log_x:
        tx = lambda v: pad_l + (math.log10(max(v, 1e-4)) - math.log10(max(x0, 1e-4))) / \
            (math.log10(x1) - math.log10(max(x0, 1e-4))) * (W - pad_l - pad_r)
    else:
        tx = lambda v: pad_l + (v - x0) / (x1 - x0 or 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v / y_max) * (H - pad_t - pad_b)

    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" aria-label="{e(caption)}">']
    for g in [y_max * f for f in (0, 0.25, 0.5, 0.75, 1.0)]:
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">'
                 f'{g:g}</text>')
    pts = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in series)
    o.append(f'<polyline points="{pts}" class="sline"/>')
    for x, y in series:
        o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="4" class="sdot"/>')
        o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="12" class="hit" '
                 f'data-x="{x:g}" data-y="{y:.3f}" data-xl="{e(xlabel)}" '
                 f'data-yl="{e(ylabel)}"/>')
    for m in (marks or []):
        mx = tx(m["x"])
        o.append(f'<line x1="{mx:.1f}" y1="{pad_t}" x2="{mx:.1f}" y2="{ty(0)}" class="mark"/>')
        o.append(f'<text x="{mx+5:.1f}" y="{pad_t+11}" class="marklab">{e(m["label"])}</text>')
    for x in (xs[0], xs[len(xs)//2], xs[-1]):
        o.append(f'<text x="{tx(x):.1f}" y="{H-pad_b+16}" text-anchor="middle" '
                 f'class="atick">{x:g}</text>')
    o.append(f'<text x="{(W+pad_l-pad_r)/2:.0f}" y="{H-6}" text-anchor="middle" '
             f'class="axlabel">{e(xlabel)}</text>')
    o.append("</svg>")
    return (f'<figure class="chart"><figcaption><strong>{e(caption)}</strong>'
            f'<span>{note}</span></figcaption>{"".join(o)}</figure>')


RUNGS = ("duplicate", "within", "across", "random")
RUNG_LABEL = {"duplicate": "duplicate", "within": "within-branch",
              "across": "cross-branch", "random": "unstructured"}


def _rung_cls(i):
    return f"r{i}"


def schema_acquisition_chart(series, m):
    """New-transition recall per trial, with the superseded sequence-mean
    underneath it and the arithmetic floor that made the old readout vacuous."""
    W, H = 980, 330
    pad_l, pad_r, pad_t, pad_b = 58, 212, 16, 46
    curves = {r: series[r]["new_item_curve"] for r in RUNGS if r in series}
    if not curves:
        return ""
    n = max(len(c) for c in curves.values())
    tx = lambda i: pad_l + i / max(n - 1, 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)

    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="New-transition recall per trial for each consistency rung">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')

    floor = m.get("schema_dilution_floor")
    crit = m.get("schema_duplicate_criterion")
    if floor:
        o.append(f'<line x1="{pad_l}" y1="{ty(floor):.1f}" x2="{W-pad_r}" '
                 f'y2="{ty(floor):.1f}" class="floorline"/>')
        o.append(f'<text x="{pad_l+6}" y="{ty(floor)-6:.1f}" class="floorlab">'
                 f'sequence-mean floor {floor:.2f}</text>')
    if crit:
        o.append(f'<line x1="{pad_l}" y1="{ty(crit):.1f}" x2="{W-pad_r}" '
                 f'y2="{ty(crit):.1f}" class="critline"/>')
        o.append(f'<text x="{W-pad_r-4}" y="{ty(crit)+13:.1f}" text-anchor="end" '
                 f'class="critlab">criterion {crit:.2f}</text>')

    # superseded sequence-mean, faint
    for i, r in enumerate(RUNGS):
        if r not in series:
            continue
        seq = series[r]["learning_curve"]
        pts = " ".join(f"{tx(k):.1f},{ty(v):.1f}" for k, v in enumerate(seq))
        o.append(f'<polyline points="{pts}" class="seqline {_rung_cls(i)}"/>')
    # the scored read-out
    for i, r in enumerate(RUNGS):
        if r not in curves:
            continue
        c = curves[r]
        pts = " ".join(f"{tx(k):.1f},{ty(v):.1f}" for k, v in enumerate(c))
        o.append(f'<polyline points="{pts}" class="rline {_rung_cls(i)}"/>')
        for k, v in enumerate(c):
            o.append(f'<circle cx="{tx(k):.1f}" cy="{ty(v):.1f}" r="3.2" '
                     f'class="rdot {_rung_cls(i)}"/>')
            o.append(f'<circle cx="{tx(k):.1f}" cy="{ty(v):.1f}" r="10" class="hit" '
                     f'data-x="{k+1}" data-y="{v:.2f}" data-xl="trial" '
                     f'data-yl="{RUNG_LABEL[r]} new-item recall"/>')
        ttc = m.get(f"schema_{r}_trials_to_criterion")
        o.append(f'<text x="{W-pad_r+12}" y="{pad_t+24+i*36:.1f}" class="rlab {_rung_cls(i)}">'
                 f'{RUNG_LABEL[r]}</text>')
        o.append(f'<text x="{W-pad_r+12}" y="{pad_t+40+i*36:.1f}" class="rsub">'
                 f'proj {m.get(f"schema_{r}_projection_ratio", 0):.2f} · '
                 f'{ttc:.0f} trials · auc {m.get(f"schema_{r}_new_item_auc", 0):.2f}</text>')
    for k in range(0, n, 2):
        o.append(f'<text x="{tx(k):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{k+1}</text>')
    o.append(f'<text x="{(pad_l + W-pad_r)/2:.0f}" y="{H-8}" text-anchor="middle" '
             f'class="axlabel">training trial (one presentation each)</text>')
    o.append(f'<text transform="translate(14,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">recall of the new transition</text>')
    o.append('</svg>')
    return ('<figure class="chart chart-wide"><figcaption><strong>The readout that '
            'unblocked the section</strong><span>Solid: recall of the single new '
            'transition, which is what acquisition is now scored on. Faint: the '
            'sequence-mean MRR it replaced — it starts at the dilution floor and '
            'stays there, because only 1 of 6 scored transitions is new. The '
            'criterion sits <em>below</em> that floor, so under the old readout '
            'every rung passed on trial 1.</span></figcaption>'
            + "".join(o) + '</figure>')


def schema_ladder_chart(m):
    """Consistency (projection ratio) against trials to criterion."""
    W, H = 468, 250
    pad_l, pad_r, pad_t, pad_b = 52, 24, 20, 46
    pts = [(m[f"schema_{r}_projection_ratio"], m[f"schema_{r}_trials_to_criterion"], r)
           for r in RUNGS if f"schema_{r}_projection_ratio" in m]
    if not pts:
        return ""
    ymax = max(p[1] for p in pts) + 2
    tx = lambda v: pad_l + v * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v / ymax) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Trials to criterion against rung consistency">']
    step = 2 if ymax <= 12 else 4
    for g in range(0, int(ymax) + 1, step):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g}</text>')
    order = sorted(pts, key=lambda p: -p[0])
    line = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y, _ in order)
    o.append(f'<polyline points="{line}" class="ladderline"/>')
    for x, y, r in pts:
        i = RUNGS.index(r)
        o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="7" class="rdot big {_rung_cls(i)}"/>')
        dy = -14 if i % 2 == 0 else 21
        anch = "end" if x > 0.9 else ("start" if x < 0.12 else "middle")
        lx = tx(x) + (-9 if anch == "end" else (9 if anch == "start" else 0))
        o.append(f'<text x="{lx:.1f}" y="{ty(y)+dy:.1f}" text-anchor="{anch}" '
                 f'class="pointlab">{RUNG_LABEL[r]}</text>')
        o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="14" class="hit" '
                 f'data-x="{x:.2f}" data-y="{y:.0f}" data-xl="projection ratio" '
                 f'data-yl="trials to criterion"/>')
    for g in (0.4, 0.6, 0.8, 1.0):
        o.append(f'<text x="{tx(g):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{g:g}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-8}" text-anchor="middle" '
             f'class="axlabel">projection ratio (consistency with the schema) →</text>')
    o.append(f'<text transform="translate(12,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">trials to criterion</text>')
    o.append('</svg>')
    sc = m.get("schema_consistency_speed_corr", float("nan"))
    ac = m.get("schema_consistency_auc_corr", float("nan"))
    return ('<figure class="chart"><figcaption><strong>The ladder is non-monotone at '
            'the consistent end</strong><span>Predicted: fewer trials as consistency '
            f'rises. The unstructured rung is slowest as predicted, but '
            f'<em>duplicate</em> — the most consistent rung — is slower than both '
            f'middle rungs. speed corr {sc:+.2f} (predicted negative), '
            f'AUC corr {ac:+.2f} (predicted positive).</span></figcaption>'
            + "".join(o) + '</figure>')


def schema_interference_chart(m):
    """Recall lost by prior material, split by relatedness to the host category."""
    W, H = 468, 250
    pad_l, pad_r, pad_t, pad_b = 54, 16, 18, 46
    rels = ("same", "sibling", "far")
    vals = {rel: [m.get(f"schema_{r}_interference_{rel}", 0.0) for r in RUNGS]
            for rel in rels}
    ymax = max(0.16, max(max(v) for v in vals.values()) * 1.2)
    gw = (W - pad_l - pad_r) / len(rels)
    bw = gw / len(RUNGS) - 4
    ty = lambda v: pad_t + (1 - v / ymax) * (H - pad_t - pad_b)
    base = ty(0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Interference by relatedness for each consistency rung">']
    for g in (0, 0.05, 0.10, 0.15):
        if g > ymax:
            continue
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:.2f}</text>')
    for gi, rel in enumerate(rels):
        gx = pad_l + gi * gw
        for i, r in enumerate(RUNGS):
            v = vals[rel][i]
            x = gx + 3 + i * (bw + 4)
            h = max(base - ty(v), 1.2)
            o.append(f'<rect x="{x:.1f}" y="{ty(v):.1f}" width="{bw:.1f}" '
                     f'height="{h:.1f}" rx="2.5" class="rbar {_rung_cls(i)}"/>')
            o.append(f'<rect x="{x:.1f}" y="{pad_t}" width="{bw:.1f}" '
                     f'height="{base-pad_t:.1f}" class="hit" data-x="{RUNG_LABEL[r]}" '
                     f'data-y="{v:+.2f}" data-xl="{rel} category" data-yl="recall lost"/>')
        o.append(f'<text x="{gx+gw/2:.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{rel}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-8}" text-anchor="middle" '
             f'class="axlabel">relatedness of the damaged material to the host category</text>')
    o.append(f'<text transform="translate(12,{(pad_t+base)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">recall lost (pre − post)</text>')
    o.append('</svg>')
    pf = " / ".join(f"{m.get(f'schema_{r}_prefix_recall', float('nan')):.2f}" for r in RUNGS)
    return ('<figure class="chart"><figcaption><strong>Damage is local, and graded by '
            'consistency</strong><span>Only the host category loses anything, and it '
            'loses 7&times; more to the unstructured item than to the consistent one. '
            'Sibling and far categories are untouched at 0.00 — the SWIL-predicted '
            f'profile. Host-prefix recall falls the same way: {pf}. '
            'Valid only under <code>interference_protocol=\'focused\'</code>.'
            '</span></figcaption>' + "".join(o) + '</figure>')


# ------------------------------------------------------- one-shot learning ----

def exposure_chart(curve, m):
    """MRR against number of presentations — the literal time-to-acquisition."""
    if not curve:
        return ""
    W, H = 468, 262
    pad_l, pad_r, pad_t, pad_b = 52, 22, 22, 52
    eps = [pt["epoch"] for pt in curve]
    mrrs = [pt["mrr"] for pt in curve]
    lo_, hi_ = math.log2(max(eps[0], 1)), math.log2(max(eps[-1], 2))
    tx = lambda v: pad_l + (math.log2(max(v, 1)) - lo_) / max(hi_ - lo_, 1e-9) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Recall against number of presentations">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    crit = m.get("mrr_threshold", 0.95)
    o.append(f'<line x1="{pad_l}" y1="{ty(crit):.1f}" x2="{W-pad_r}" y2="{ty(crit):.1f}" '
             f'class="critline"/>')
    o.append(f'<text x="{W-pad_r-3}" y="{ty(crit)+14:.1f}" text-anchor="end" '
             f'class="critlab">criterion {crit:g}</text>')
    ce = m.get("convergence_epochs")
    if ce:
        o.append(f'<line x1="{tx(ce):.1f}" y1="{pad_t}" x2="{tx(ce):.1f}" '
                 f'y2="{ty(0):.1f}" class="floorline"/>')
        o.append(f'<text x="{tx(ce)+7:.1f}" y="{ty(0.42):.0f}" class="floorlab">'
                 f'converged at {ce:g} presentation{"s" if ce != 1 else ""}</text>')
    o.append('<polyline points="' + " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in zip(eps, mrrs))
             + '" class="sline"/>')
    for x, y in zip(eps, mrrs):
        o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="5" class="sdot"/>')
        o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="13" class="hit" '
                 f'data-x="{x:g}" data-y="{y:.3f}" data-xl="presentations" data-yl="MRR"/>')
        o.append(f'<text x="{tx(x):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{x:g}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-10}" text-anchor="middle" '
             f'class="axlabel">presentations of the list (log₂)</text>')
    o.append(f'<text transform="translate(12,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">cued recall (MRR)</text>')
    o.append('</svg>')
    span = m.get("convergence_span")
    return ('<figure class="chart"><figcaption><strong>Time to acquisition: one '
            'presentation</strong><span>The sweep starts at a single pass and the '
            f'list is already fully recalled — MRR 1.00, span {span:g} of 6, at the '
            'first checkpoint. The curve has no rising limb to measure, so this '
            'section can report that acquisition is one-shot but cannot resolve '
            '<em>how</em> one-shot. The two panels beside it are what put load on '
            'the claim.</span></figcaption>' + "".join(o) + '</figure>')


def load_acquisition_chart(cc):
    """The chain diagonal against its final row: acquired vs still there."""
    learned = cc.get("per_task_learned") or []
    final = cc.get("per_task_final") or []
    labels = cc.get("task_labels") or []
    if not learned:
        return ""
    W, H = 468, 262
    pad_l, pad_r, pad_t, pad_b = 52, 22, 22, 58
    n = len(learned)
    tx = lambda i: pad_l + (i / max(n - 1, 1)) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Acquisition under load against final retention, per task">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    for vals, cls, lab in ((learned, "s-acc", "acquired (diagonal)"),
                           (final, "s-per", "still there at the end")):
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(vals))
                 + f'" class="revline {cls}"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="4" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="12" class="hit" '
                     f'data-x="{i} prior task{"s" if i != 1 else ""}" data-y="{v:.2f}" '
                     f'data-xl="load" data-yl="{lab}"/>')
    for i in range(n):
        o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+17}" text-anchor="middle" '
                 f'class="atick">{i}</text>')
        if i < len(labels):
            o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+30}" text-anchor="middle" '
                     f'class="rsub">{e(labels[i][:6])}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-8}" text-anchor="middle" '
             f'class="axlabel">tasks already in the substrate</text>')
    o.append(f'<text transform="translate(12,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">cued recall (MRR)</text>')
    o.append('</svg>')
    legend = ('<div class="legend"><span><i class="sw sw-acc"></i>acquired, at the '
              'moment it was learned</span><span><i class="sw sw-per"></i>still there '
              'after the whole chain</span></div>')
    return ('<figure class="chart"><figcaption><strong>One-shot survives load — and '
            'is paired with what it cost</strong><span>Each task gets one pass, with '
            'every earlier task already stored. Acquisition does not decay along the '
            'chain (intransigence 0.00), and the two curves coincide, so nothing was '
            'bought by forgetting. This is the exposure-against-retention pairing '
            'that stops a retention score being gamed by not learning — newly '
            'available now the chain section exists.</span></figcaption>'
            + "".join(o) + legend + '</figure>')


def scale_probe_chart(probe):
    """MRR, aim and trace strength across four orders of magnitude of learning rate."""
    rows = (probe or {}).get("rows") or []
    if not rows:
        return ""
    W, H = 980, 300
    pad_l, pad_r, pad_t, pad_b = 56, 196, 24, 54
    lrs = [r["learning_rate"] for r in rows]
    lo_, hi_ = math.log10(min(lrs)), math.log10(max(lrs))
    tx = lambda v: pad_l + (math.log10(v) - lo_) / max(hi_ - lo_, 1e-9) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Score, aim and trace strength against learning rate">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    series = (("mrr", "p-mrr", "the benchmark score", "MRR at one pass"),
              ("cos", "p-cos", "aim", "cosine to the true next item"),
              ("mag", "p-mag", "trace strength", "|prediction| / |target|"))
    for key, cls, lab, sub in series:
        vals = [r[key] for r in rows]
        o.append('<polyline points="' + " ".join(f"{tx(x):.1f},{ty(v):.1f}"
                                                 for x, v in zip(lrs, vals))
                 + f'" class="revline {cls}"/>')
        for x, v in zip(lrs, vals):
            o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(v):.1f}" r="3.6" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(v):.1f}" r="10" class="hit" '
                     f'data-x="{x:g}" data-y="{v:.4f}" data-xl="learning rate" '
                     f'data-yl="{sub}"/>')
    for i, (key, cls, lab, sub) in enumerate(series):
        o.append(f'<text x="{W-pad_r+14}" y="{pad_t+26+i*38:.0f}" class="revlab {cls}">{lab}</text>')
        o.append(f'<text x="{W-pad_r+14}" y="{pad_t+40+i*38:.0f}" class="rsub">{sub}</text>')
    for x in lrs:
        o.append(f'<text x="{tx(x):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{x:g}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-10}" text-anchor="middle" '
             f'class="axlabel">learning rate (log₁₀), one pass throughout</text>')
    o.append(f'<text transform="translate(14,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">value (all three are ratios on [0,1])</text>')
    o.append('</svg>')
    mags = [r["mag"] for r in rows]
    span = max(mags) / max(min(mags), 1e-12)
    mrrs = [r["mrr"] for r in rows]
    return ('<figure class="chart chart-wide"><figcaption><strong>What "one-shot" '
            f'here does and does not mean</strong><span>Across four orders of '
            f'magnitude of learning rate the stored trace varies {span:,.0f}× in '
            f'strength, and the score does not move ({min(mrrs):.3f}–{max(mrrs):.3f}). '
            'Aim barely moves either: the readout points at the right item with '
            'cosine ≈ 0.59 throughout, which is all a cosine decoder needs to pick it '
            'out of twelve. So the section establishes that an association forms in '
            'one pass, not that the trace is usable — and the same unconstrained '
            'magnitude is why raw-feedback rollout collapses in the length sweep '
            'while cued recall stays perfect. A property of the read-out, not a model '
            'failure, but it bounds what the 1.00 supports.</span></figcaption>'
            + "".join(o) + '</figure>')


def cascade_position_chart(pr, epochs=1):
    """Per-position accuracy: cued against the three rollout feedback modes."""
    rows = (pr or {}).get("rows") or []
    row = next((r for r in rows if r["epochs"] == epochs), None)
    if not row:
        return ""
    W, H = 980, 320
    pad_l, pad_r, pad_t, pad_b = 56, 196, 24, 54
    n = len(row["cued_curve"])
    tx = lambda i: pad_l + i / max(n - 1, 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Recall at each serial position, cued against three rollout modes">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    o.append(f'<line x1="{pad_l}" y1="{ty(0.75):.1f}" x2="{W-pad_r}" y2="{ty(0.75):.1f}" '
             f'class="critline"/>')
    o.append(f'<text x="{W-pad_r-3}" y="{ty(0.75)-6:.1f}" text-anchor="end" '
             f'class="critlab">span threshold 0.75</text>')
    series = [("cued", row["cued_curve"], "c-cued", "cued (one step)",
               "each position probed from the true previous item"),
              ("quantized", row["rollout_curve"]["quantized"], "c-quant",
               "rollout, codebook", "prediction snapped to a clean embedding"),
              ("l2", row["rollout_curve"]["l2"], "c-l2", "rollout, L2",
               "magnitude removed, direction kept"),
              ("raw", row["rollout_curve"]["raw"], "c-raw", "rollout, raw",
               "prediction fed back verbatim")]
    for key, vals, cls, lab, sub in series:
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(vals))
                 + f'" class="revline {cls}"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="3.4" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="10" class="hit" '
                     f'data-x="{i}" data-y="{v:.2f}" data-xl="position" data-yl="{lab}"/>')
    for i, (key, vals, cls, lab, sub) in enumerate(series):
        o.append(f'<text x="{W-pad_r+14}" y="{pad_t+24+i*38:.0f}" class="revlab {cls}">{lab}</text>')
        o.append(f'<text x="{W-pad_r+14}" y="{pad_t+38+i*38:.0f}" class="rsub">{sub}</text>')
    for i in range(n):
        o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{i}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-10}" text-anchor="middle" '
             f'class="axlabel">serial position in the list</text>')
    o.append(f'<text transform="translate(14,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">recall accuracy</text>')
    o.append('</svg>')
    sr = row["rollout_span"]["raw"]; sq = row["rollout_span"]["quantized"]
    return ('<figure class="chart chart-wide"><figcaption><strong>The chain breaks at '
            f'position {sr + 1}, and the associations are not what break</strong>'
            f'<span>One presentation, {pr["seq_len"]}-item list: position 0 is the cue, '
            f'so {pr["max_span"]} items are generated and scored. Cueing each position '
            'from the true previous item recovers all of them; letting the model '
            f'drive itself recovers {sr}. Codebook feedback recovers {sq} from the '
            'same weights, so every one-step association is intact and what fails is '
            'the state handed forward. L2 tracks raw exactly — renormalising the '
            'fed-back vector changes nothing — so the drift is angular, not '
            'magnitude. The L2 mode exists in <code>sequence_length</code> precisely as the '
            'control that isolates magnitude drift, and running it here shows that is '
            'not what limits this arm — the control did its job.</span></figcaption>' + "".join(o) + '</figure>')


def cascade_exposure_chart(pr):
    """Cued against rollout recall along the exposure axis."""
    rows = (pr or {}).get("rows") or []
    if not rows:
        return ""
    W, H = 468, 300
    pad_l, pad_r, pad_t, pad_b = 52, 22, 24, 66
    eps = [r["epochs"] for r in rows]
    lo_, hi_ = math.log2(max(eps[0], 1)), math.log2(max(eps[-1], 2))
    tx = lambda v: pad_l + (math.log2(max(v, 1)) - lo_) / max(hi_ - lo_, 1e-9) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Cued and rollout recall against number of presentations">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    sets = [("cued", [r["cued_mrr"] for r in rows], "c-cued"),
            ("quantized", [r["rollout_mrr"]["quantized"] for r in rows], "c-quant"),
            ("raw", [r["rollout_mrr"]["raw"] for r in rows], "c-raw")]
    for lab, vals, cls in sets:
        o.append('<polyline points="' + " ".join(f"{tx(x):.1f},{ty(v):.1f}"
                                                 for x, v in zip(eps, vals))
                 + f'" class="revline {cls}"/>')
        for x, v in zip(eps, vals):
            o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(v):.1f}" r="3.6" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(x):.1f}" cy="{ty(v):.1f}" r="10" class="hit" '
                     f'data-x="{x:g}" data-y="{v:.2f}" data-xl="presentations" '
                     f'data-yl="{lab} MRR"/>')
    closure = (pr or {}).get("closure") or {}
    grid = next((r["epochs"] for r in rows if r["rollout_mrr"]["raw"] >= 0.99), None)
    first = closure.get("rollout_raw") or grid
    if first:
        o.append(f'<line x1="{tx(first):.1f}" y1="{pad_t}" x2="{tx(first):.1f}" '
                 f'y2="{ty(0):.1f}" class="floorline"/>')
        o.append(f'<text x="{tx(first)-7:.1f}" y="{ty(0.30):.0f}" text-anchor="end" '
                 f'class="floorlab">rollout closes at {first}</text>')
    for x in eps:
        o.append(f'<text x="{tx(x):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{x:g}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-32}" text-anchor="middle" '
             f'class="axlabel">presentations of the list (log₂)</text>')
    o.append(f'<text transform="translate(12,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">mean recall (MRR)</text>')
    o.append('</svg>')
    legend = ('<div class="legend"><span><i class="sw c-cued-sw"></i>cued</span>'
              '<span><i class="sw c-quant-sw"></i>rollout, codebook</span>'
              '<span><i class="sw c-raw-sw"></i>rollout, raw</span></div>')
    note = ""
    if grid and first and grid != first:
        note = (f' The plotted points double, so the curve first reaches 1.00 at '
                f'{grid}; {first} is the resolved minimum from a scan of the '
                f'interval below it.')
    return ('<figure class="chart"><figcaption><strong>Cued recall is one-shot; '
            f'rollout needs {first}×&nbsp;more</strong><span>The same weights, scored '
            'two ways. Cued recall is at ceiling from the first presentation and '
            'stays there, so the exposure axis looks flat and one-shot. Driving the '
            f'model from its own output needs {first} presentations before all '
            f'{pr["max_span"]} generated items come back correct on every trial. '
            'Everything the acquisition score reports as learned after one pass is '
            'learned — it just is not yet good enough to survive being fed its own '
            'output.' + note + '</span></figcaption>' + "".join(o) + legend + '</figure>')


PLOT_EXPOSURES = (1, 4, 8, 16, 32)

#: Heatmap bin counts. Values are binned rather than mapped to a continuous fill
#: so the colours can live in CSS custom properties and follow the viewer's
#: theme; the cell also carries its number, so the binning never hides a value.
SEQ_BINS, DIV_BINS = 6, 7


def _seq_bin(v, vmin, vmax):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    return max(0, min(SEQ_BINS - 1, int(t * SEQ_BINS)))


def _div_bin(v, vabs, eps=0.02):
    """7 bins symmetric about zero.

    The sign is assigned before the magnitude, so any value outside a narrow
    dead band around zero lands on its own side of the ramp. Rounding the
    signed ratio instead would put +0.10 and -0.10 both in the neutral bin,
    which would erase exactly the boundary this scale exists to show.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if abs(v) <= eps:
        return 3
    mag = min(1.0, abs(v) / vabs) if vabs else 0.0
    step = 1 + min(2, int(mag * 3))
    return 3 + step if v > 0 else 3 - step


def _heatmap(values, row_labels, col_labels, *, mode, vmin=0.0, vmax=1.0,
             x_label="", y_label="", row_title="", fmt="{:.2f}",
             cell=38, gap=2, pad_l=76, pad_t=26, hit_yl=""):
    """One grid, binned into theme-aware CSS classes, every cell labelled."""
    nrow, ncol = len(values), len(values[0])
    W = pad_l + ncol * (cell + gap) + 12
    H = pad_t + nrow * (cell + gap) + 44
    vabs = max(abs(vmin), abs(vmax))
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark hm" role="img" '
         f'aria-label="{e(y_label)} against {e(x_label)}">']
    for c in range(ncol):
        o.append(f'<text x="{pad_l + c*(cell+gap) + cell/2:.0f}" y="{pad_t-8}" '
                 f'text-anchor="middle" class="mlab">{e(col_labels[c])}</text>')
    for r in range(nrow):
        y = pad_t + r*(cell+gap)
        o.append(f'<text x="{pad_l-9}" y="{y + cell/2 + 4:.0f}" text-anchor="end" '
                 f'class="mlab">{e(row_labels[r])}</text>')
        for c in range(ncol):
            v = values[r][c]
            x = pad_l + c*(cell+gap)
            b = _seq_bin(v, vmin, vmax) if mode == "seq" else _div_bin(v, vabs)
            if b is None:
                o.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'rx="3" class="hcell-na"/>')
                continue
            cls = f"sq{b}" if mode == "seq" else f"dv{b}"
            dark = (mode == "seq" and b >= 3) or (mode == "div" and b in (0, 1, 5, 6))
            o.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" '
                     f'class="hcell {cls}"/>')
            o.append(f'<text x="{x+cell/2:.0f}" y="{y+cell/2+3.5:.0f}" '
                     f'text-anchor="middle" class="hval2 {"on-dark" if dark else "on-light"}">'
                     f'{fmt.format(v)}</text>')
            o.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" class="hit" '
                     f'data-x="{e(col_labels[c])}" data-y="{fmt.format(v)}" '
                     f'data-xl="{e(x_label)}" data-yl="{e(hit_yl or y_label)}, '
                     f'{e(row_labels[r])}"/>')
    o.append(f'<text x="{pad_l + ncol*(cell+gap)/2:.0f}" y="{H-10}" '
             f'text-anchor="middle" class="axlabel">{e(x_label)}</text>')
    o.append(f'<text transform="translate(15,{pad_t + nrow*(cell+gap)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">{e(row_title)}</text>')
    o.append('</svg>')
    return "".join(o)


def _hm_scale(mode, vmin, vmax, label):
    bins = SEQ_BINS if mode == "seq" else DIV_BINS
    pre = "sq" if mode == "seq" else "dv"
    sw = "".join(f'<i class="hsw {pre}{i}"></i>' for i in range(bins))
    return (f'<div class="hmscale"><span class="hend">{vmin:+.2f}</span>{sw}'
            f'<span class="hend">{vmax:+.2f}</span>'
            f'<span class="legend-note">{label}</span></div>')


def _exp_rows(pr, exposures=PLOT_EXPOSURES):
    rows = (pr or {}).get("rows") or []
    out = []
    for k, ep in enumerate(exposures):
        r = next((x for x in rows if x["epochs"] == ep), None)
        if r:
            out.append((k, ep, r))
    return out


def _exp_legend(items, x, y0, step=36):
    o = []
    for i, (k, ep, _r, sub) in enumerate(items):
        o.append(f'<text x="{x:.0f}" y="{y0 + i*step:.0f}" class="revlab e{k}">'
                 f'{ep} presentation{"s" if ep != 1 else ""}</text>')
        if sub:
            o.append(f'<text x="{x:.0f}" y="{y0 + i*step + 16:.0f}" class="rsub">{sub}</text>')
    return o


def cascade_accuracy_heatmap(pr):
    """Rollout recall for every (exposure, position) cell."""
    got = _exp_rows(pr)
    if not got:
        return ""
    vals = [g[2]["rollout_curve"]["raw"][1:] for g in got]     # position 0 is the cue
    ncol = len(vals[0])
    svg = _heatmap(vals,
                   [f'{ep}' for _, ep, _r in got],
                   [str(i + 1) for i in range(ncol)],
                   mode="seq", vmin=0.0, vmax=1.0,
                   x_label="serial position", y_label="recall accuracy",
                   row_title="presentations", cell=36,
                   hit_yl="recall")
    return ('<figure class="chart"><figcaption><strong>The cascade recedes along the '
            'list as exposure grows</strong><span>Every cell is one (exposure, '
            'position) pair under raw feedback; position 0 is the cue and is not '
            'scored. The dark region — recall intact — spreads rightward as training '
            'increases. Cued recall is 1.00 in every cell of this grid, at every '
            'exposure.</span></figcaption>' + svg
            + _hm_scale("seq", 0.0, 1.0, "recall accuracy") + '</figure>')


def cascade_margin_heatmap(pr):
    """The same grid in the quantity that decides it."""
    got = _exp_rows(pr)
    if not got or "margin_rollout" not in got[0][2]:
        return ""
    vals = [g[2]["margin_rollout"] for g in got]
    ncol = len(vals[0])
    svg = _heatmap(vals,
                   [f'{ep}' for _, ep, _r in got],
                   [str(i + 1) for i in range(ncol)],
                   mode="div", vmin=-0.55, vmax=0.55,
                   x_label="rollout step", y_label="margin",
                   row_title="presentations", cell=36, fmt="{:+.2f}",
                   hit_yl="margin")
    return ('<figure class="chart"><figcaption><strong>The margin is what predicts '
            'recall</strong><span>Cosine to the true item minus cosine to the best '
            'competing item, on the same grid as the accuracy panel. The colour turns '
            'at zero because zero is a real decision boundary — below it a different '
            'item outranks the true one — and the neutral band traces the same edge '
            'the accuracy panel shows. Note step 8: positive at every exposure, an '
            'item the readout finds easy regardless of '
            'training.</span></figcaption>' + svg
            + _hm_scale("div", -0.55, 0.55, "margin: target − best other") + '</figure>')


def cascade_geometry_heatmaps(pr):
    """Target and competitor cosine, one grid each, on a shared scale."""
    got = _exp_rows(pr)
    if not got or "cos_rollout_competitor" not in got[0][2]:
        return ""
    tgt = [g[2]["cos_rollout"] for g in got]
    cmp_ = [g[2]["cos_rollout_competitor"] for g in got]
    lo = min(min(r) for r in tgt + cmp_)
    hi = max(max(r) for r in tgt + cmp_)
    rows_lab = [f'{ep}' for _, ep, _r in got]
    cols = [str(i + 1) for i in range(len(tgt[0]))]
    kw = dict(mode="seq", vmin=lo, vmax=hi, x_label="rollout step",
              row_title="presentations", cell=34)
    a = _heatmap(tgt, rows_lab, cols, y_label="cosine to the true item",
                 hit_yl="cosine to the true item", **kw)
    b = _heatmap(cmp_, rows_lab, cols, y_label="cosine to the best other item",
                 hit_yl="cosine to the best other item", **kw)
    lo_r, hi_r = got[0][2], got[-1][2]
    return ('<figure class="chart chart-wide"><figcaption><strong>Exposure does not aim '
            'the trace better — it stops it pointing at everything else</strong>'
            '<span>Both grids share one scale, so they are directly comparable. Left, '
            'cosine to the correct next item: at the end of the rollout it is '
            f'{lo_r["cos_rollout"][-1]:.2f} after one presentation and '
            f'{hi_r["cos_rollout"][-1]:.2f} after {got[-1][1]}, barely moving, while '
            'span goes 2 → 10. Right, cosine to the best competitor: it collapses '
            f'from {lo_r["cos_rollout_competitor"][-1]:.2f} to '
            f'{hi_r["cos_rollout_competitor"][-1]:.2f}. Cosine to the target is '
            'therefore not a sufficient reading of trace fidelity — decoding ranks '
            'eleven non-orthogonal items (mean pairwise cosine 0.18), so what matters '
            'is the gap between the two grids.</span></figcaption>'
            '<div class="hmpair"><div><p class="hmtitle">cosine to the TRUE item</p>'
            + a + '</div><div><p class="hmtitle">cosine to the BEST OTHER item</p>'
            + b + '</div></div>'
            + _hm_scale("seq", lo, hi, "cosine, shared scale") + '</figure>')


def cascade_margin_trajectory(pr):
    """The margin as curves, with only the two extremes carrying colour."""
    got = _exp_rows(pr)
    if not got or "margin_rollout" not in got[0][2]:
        return ""
    W, H = 468, 310
    pad_l, pad_r, pad_t, pad_b = 54, 22, 24, 62
    n = len(got[0][2]["margin_rollout"])
    tx = lambda i: pad_l + i / max(n - 1, 1) * (W - pad_l - pad_r)
    lo_, hi_ = -0.35, 0.65
    ty = lambda v: pad_t + (1 - (v - lo_) / (hi_ - lo_)) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Margin along the rollout at the lowest and highest exposure">']
    o.append(f'<rect x="{pad_l}" y="{ty(0):.1f}" width="{W-pad_l-pad_r}" '
             f'height="{ty(lo_)-ty(0):.1f}" class="failband"/>')
    for g in (-0.2, 0.0, 0.2, 0.4, 0.6):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" '
                 f'class="{"zeroline" if g == 0 else "gl"}"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    o.append(f'<text x="{W-pad_r-4}" y="{ty(-0.06):.1f}" text-anchor="end" '
             f'class="floorlab">wrong item wins</text>')
    for k, ep, r in got[1:-1]:
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(r["margin_rollout"]))
                 + '" class="ctxline"/>')
    for (k, ep, r), cls in ((got[0], "m-lo"), (got[-1], "m-hi")):
        vals = r["margin_rollout"]
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(vals))
                 + f'" class="revline {cls}"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="3.6" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="10" class="hit" '
                     f'data-x="step {i+1}" data-y="{v:+.2f}" data-xl="rollout" '
                     f'data-yl="margin after {ep} presentation{"s" if ep != 1 else ""}"/>')
    o.append(f'<text x="{W-pad_r-4}" y="{ty(got[-1][2]["margin_rollout"][-1])-11:.1f}" '
             f'text-anchor="end" class="revlab m-hi">{got[-1][1]} presentations</text>')
    o.append(f'<text x="{W-pad_r-4}" y="{ty(got[0][2]["margin_rollout"][-1])+18:.1f}" '
             f'text-anchor="end" class="revlab m-lo">{got[0][1]} presentation</text>')
    for i in range(0, n, 2):
        o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{i+1}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-30}" text-anchor="middle" '
             f'class="axlabel">step of the free rollout</text>')
    o.append(f'<text transform="translate(13,{(pad_t+ty(lo_))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">margin: target − best other</text>')
    o.append('</svg>')
    return ('<figure class="chart"><figcaption><strong>The same margins as '
            'trajectories</strong><span>Only the two extremes carry colour; the three '
            'intermediate exposures sit behind them as context, so the ordering stays '
            'visible without five lines competing. One presentation crosses into the '
            'fail band at step 3 and stays there; thirty-two never '
            'crosses.</span></figcaption>' + "".join(o) + '</figure>')


def cascade_alignment_chart(pr, lo_ep=1, hi_ep=32):
    """Cosine to the true trajectory at each step of a free rollout."""
    rows = (pr or {}).get("rows") or []
    a = next((r for r in rows if r["epochs"] == lo_ep), None)
    b = next((r for r in rows if r["epochs"] == hi_ep), None)
    if not a or not b:
        return ""
    W, H = 468, 300
    pad_l, pad_r, pad_t, pad_b = 52, 22, 24, 66
    n = len(a["cos_rollout"])
    tx = lambda i: pad_l + i / max(n - 1, 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Alignment with the true trajectory along a free rollout">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    for row, cls, lab in ((b, "c-quant", f"{hi_ep} presentations"),
                          (a, "c-raw", f"{lo_ep} presentation")):
        vals = row["cos_rollout"]
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(vals))
                 + f'" class="revline {cls}"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="3.4" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="10" class="hit" '
                     f'data-x="step {i+1}" data-y="{v:.2f}" data-xl="rollout" '
                     f'data-yl="cosine to the true item ({lab})"/>')
    for i in range(0, n, 2):
        o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{i+1}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-32}" text-anchor="middle" '
             f'class="axlabel">step of the free rollout</text>')
    o.append(f'<text transform="translate(12,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">cosine to the true item</text>')
    o.append('</svg>')
    legend = (f'<div class="legend"><span><i class="sw c-quant-sw"></i>{hi_ep} '
              f'presentations</span><span><i class="sw c-raw-sw"></i>{lo_ep} '
              f'presentation</span></div>')
    a0, an = a["cos_rollout"][0], a["cos_rollout"][-1]
    b0, bn = b["cos_rollout"][0], b["cos_rollout"][-1]
    return ('<figure class="chart"><figcaption><strong>What exposure actually '
            f'buys</strong><span>Alignment decays along the rollout at both exposures '
            f'— {a0:.2f}→{an:.2f} after one presentation, {b0:.2f}→{bn:.2f} after '
            f'{hi_ep}. Exposure does not stop the decay; it raises where the decay '
            'starts. Because a nearest-neighbour decode only needs the true item to '
            'stay ahead of ten competitors, that small head start is worth many extra '
            'steps before the ranking flips — which is why the span jumps from 2 to '
            '10 while the one-step cosine moves only 0.57→0.67.</span></figcaption>'
            + "".join(o) + legend + '</figure>')


# --------------------------------------------------- sequence disambiguation ----

def symdis_load_chart(sd, m):
    """Branch accuracy against the NUMBER of confusable episodes."""
    ns = sd.get("episode_counts") or []
    if not ns:
        return ""
    acc, ch = sd["load_accuracy"], sd["load_chance"]
    orth = sd.get("load_orthogonal_accuracy") or []
    W, H = 980, 330
    pad_l, pad_r, pad_t, pad_b = 58, 200, 24, 56
    tx = lambda i: pad_l + i / max(len(ns) - 1, 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Branch accuracy against the number of confusable episodes">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    o.append('<polygon points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}" for i, v in enumerate(ch))
             + " " + f"{tx(len(ns)-1):.1f},{ty(0):.1f} {pad_l},{ty(0):.1f}" + '" class="chanceband"/>')
    o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}" for i, v in enumerate(ch))
             + '" class="chanceline"/>')
    if orth:
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(orth))
                 + '" class="revline s-orth"/>')
        for i, v in enumerate(orth):
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="3.4" class="revdot s-orth"/>')
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="10" class="hit" '
                     f'data-x="{ns[i]} episodes" data-y="{v:.2f}" data-xl="load" '
                     f'data-yl="orthogonal control"/>')
    o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}" for i, v in enumerate(acc))
             + '" class="revline s-acc"/>')
    for i, v in enumerate(acc):
        o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="4.6" class="revdot s-acc"/>')
        o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="12" class="hit" '
                 f'data-x="{ns[i]} episodes" data-y="{v:.2f}" data-xl="load" '
                 f'data-yl="branch accuracy (chance {ch[i]:.2f})"/>')
    for i, n in enumerate(ns):
        o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{n}</text>')
    labels = (("s-acc", "overlapping episodes", "the shared-stretch task"),
              ("s-orth", "orthogonal control", "same load, no shared stretch"),
              ("chance", "chance", "1/N, falling as N grows"))
    for i, (cls, lab, sub) in enumerate(labels):
        o.append(f'<text x="{W-pad_r+14}" y="{pad_t+26+i*46:.0f}" '
                 f'class="revlab {cls}">{e(lab)}</text>')
        o.append(f'<text x="{W-pad_r+14}" y="{pad_t+41+i*46:.0f}" class="rsub">{e(sub)}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-10}" text-anchor="middle" '
             f'class="axlabel">number of episodes sharing the stretch</text>')
    o.append(f'<text transform="translate(14,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">accuracy at the divergence point</text>')
    o.append('</svg>')
    above = m.get("symdis_max_episodes_above_chance", float("nan"))
    cost = m.get("symdis_load_disambiguation_cost", float("nan"))
    return ('<figure class="chart chart-wide"><figcaption><strong>The load axis, which '
            'the spatial section cannot reach</strong><span>The T-maze is capped at two '
            'routes; this section runs up to eight episodes through one shared stretch, '
            'which is the axis the audit called the cheapest real gain. Accuracy falls '
            'from 1.00 to 0.63 as the episodes pile up, but chance falls faster, so the '
            f'arm stays above it to N&nbsp;=&nbsp;{above:.0f}. The control settles what '
            'kind of failure it is: with the same N and no shared stretch, accuracy '
            'holds near 1.00 — so the cost is overlap, not capacity. That difference is '
            f'<code>load_disambiguation_cost</code> = {cost:.2f}.</span></figcaption>'
            + "".join(o) + '</figure>')


def symdis_axis_chart(sd, m, axis):
    """Accuracy above, margin below, against one manipulated axis."""
    cfg = {
        "similarity": dict(
            xs="discriminator_similarities", acc="variance_accuracy",
            mar="variance_margin", xlab="discriminator similarity between episodes",
            fmt="{:.2f}", rev=True,
            title="Accuracy steps; the margin slides",
            note=("Axis B: how alike the discriminating signals are. Accuracy takes one "
                  "step, 0.25 to 0.75, and tells you nothing about where between those "
                  "points the arm actually turns. The margin moves smoothly through "
                  "zero — negative means a competing episode outranks the true one — "
                  "and puts the crossing at similarity {thr:.2f}.")),
        "delay": dict(
            xs="delays", acc="delay_accuracy", mar="delay_margin",
            xlab="steps of shared stretch after the discriminator is withdrawn",
            fmt="{:g}", rev=False,
            title="Nothing survives the discriminator being withdrawn",
            note=("Axis A: how long the arm must hold the signal alone. It is correct "
                  "while the discriminator is present and at chance from the first "
                  "withdrawn step, with the margin going negative and staying there. "
                  "A protocol finding, not a ranking: this arm carries no state across "
                  "the cue-free stretch, so the rows past 0 are excluded from the "
                  "score.")),
    }[axis]
    xs = sd.get(cfg["xs"]) or []
    if not xs:
        return ""
    acc, mar = sd[cfg["acc"]], sd[cfg["mar"]]
    W, H = 468, 360
    pad_l, pad_r, pad_t = 56, 22, 22
    h_top, gap_mid, h_bot, pad_b = 118, 40, 96, 60
    tx = lambda i: pad_l + i / max(len(xs) - 1, 1) * (W - pad_l - pad_r)
    ty_a = lambda v: pad_t + (1 - v) * h_top
    m_lo, m_hi = -0.35, 0.45
    y0_b = pad_t + h_top + gap_mid
    ty_m = lambda v: y0_b + (1 - (v - m_lo) / (m_hi - m_lo)) * h_bot
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="{e(cfg["title"])}">']
    for g in (0, 0.5, 1.0):
        y = ty_a(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    ch = (sd.get("load_chance") or [0.25])[0] if axis == "delay" else 0.25
    o.append(f'<line x1="{pad_l}" y1="{ty_a(0.25):.1f}" x2="{W-pad_r}" '
             f'y2="{ty_a(0.25):.1f}" class="critline"/>')
    o.append(f'<text x="{W-pad_r-3}" y="{ty_a(0.25)-5:.1f}" text-anchor="end" '
             f'class="critlab">chance 0.25</text>')
    o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty_a(v):.1f}"
                                             for i, v in enumerate(acc))
             + '" class="revline s-acc"/>')
    for i, v in enumerate(acc):
        o.append(f'<circle cx="{tx(i):.1f}" cy="{ty_a(v):.1f}" r="4" class="revdot s-acc"/>')
        o.append(f'<circle cx="{tx(i):.1f}" cy="{ty_a(v):.1f}" r="11" class="hit" '
                 f'data-x="{cfg["fmt"].format(xs[i])}" data-y="{v:.2f}" '
                 f'data-xl="{e(cfg["xlab"])}" data-yl="accuracy"/>')
    o.append(f'<text x="{pad_l}" y="{pad_t-6}" class="stagelab">accuracy</text>')
    for g in (-0.2, 0.0, 0.2, 0.4):
        y = ty_m(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" '
                 f'class="{"zeroline" if g == 0 else "gl"}"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty_m(v):.1f}"
                                             for i, v in enumerate(mar))
             + '" class="revline s-per"/>')
    for i, v in enumerate(mar):
        o.append(f'<circle cx="{tx(i):.1f}" cy="{ty_m(v):.1f}" r="4" class="revdot s-per"/>')
        o.append(f'<circle cx="{tx(i):.1f}" cy="{ty_m(v):.1f}" r="11" class="hit" '
                 f'data-x="{cfg["fmt"].format(xs[i])}" data-y="{v:+.2f}" '
                 f'data-xl="{e(cfg["xlab"])}" data-yl="margin"/>')
    o.append(f'<text x="{pad_l}" y="{y0_b-8}" class="stagelab">margin</text>')
    for i, x in enumerate(xs):
        o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+30}" text-anchor="middle" '
                 f'class="atick">{cfg["fmt"].format(x)}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-14}" text-anchor="middle" '
             f'class="axlabel">{e(cfg["xlab"])}</text>')
    o.append('</svg>')
    note = cfg["note"].format(
        thr=m.get("symdis_similarity_tolerance_threshold", float("nan")))
    return (f'<figure class="chart"><figcaption><strong>{e(cfg["title"])}</strong>'
            f'<span>{note}</span></figcaption>' + "".join(o) + '</figure>')


def symdis_confusion_chart(sd, m):
    """Which wrong episode, not merely how often."""
    cm = sd.get("confusion_matrix_largest_n")
    if not cm:
        return ""
    n = len(cm)
    svg = _heatmap(cm, [str(i) for i in range(n)], [str(i) for i in range(n)],
                   mode="seq", vmin=0.0, vmax=1.0, x_label="episode predicted",
                   y_label="confusion", row_title="true episode", cell=30,
                   pad_l=88, hit_yl="share of trials")
    idx = m.get("symdis_context_graded_confusion_index", float("nan"))
    return ('<figure class="chart"><figcaption><strong>Confusion has structure</strong>'
            f'<span>At the largest load (N&nbsp;=&nbsp;{n}), where the arm lands when it '
            'is wrong. Rate alone would say 3 of 8; the matrix says the errors '
            'collapse onto two particular episodes rather than scattering, which is '
            'what a shared representation looks like as against random guessing. '
            f'The context-graded index is {idx:.2f} — how far the confusions sit from '
            'the true episode in context space, where 0 would mean falling into the '
            'nearest neighbour every time.</span></figcaption>' + svg
            + _hm_scale("seq", 0.0, 1.0, "share of trials") + '</figure>')


# ------------------------------------------------------- pattern completion ----

MASK_MODES = (("random", "m-rand", "random"),
              ("shared", "m-shar", "category (shared)"),
              ("identity", "m-iden", "identity"))


def _mask_axes(o, W, H, pad_l, pad_r, pad_t, pad_b, ty, tx, fracs, ylab,
               y_ticks=(0, 0.25, 0.5, 0.75, 1.0)):
    for g in y_ticks:
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" '
                 f'class="atick">{g:g}</text>')
    for i, f in enumerate(fracs):
        if i % 3 == 0 or i == len(fracs) - 1:
            o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                     f'class="atick">{f:.2f}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-10}" text-anchor="middle" '
             f'class="axlabel">fraction of the cue\'s active features removed</text>')
    o.append(f'<text transform="translate(14,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">{e(ylab)}</text>')


def mask_recall_chart(ms, m):
    """Recall against masked fraction, with the model-free reference beneath it."""
    if not ms:
        return ""
    fr = ms["fractions"]
    W, H = 980, 340
    pad_l, pad_r, pad_t, pad_b = 58, 214, 24, 54
    tx = lambda i: pad_l + i / max(len(fr) - 1, 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Recall and cue identifiability against masked fraction">']
    _mask_axes(o, W, H, pad_l, pad_r, pad_t, pad_b, ty, tx, fr, "recall")
    for key, cls, _lab in MASK_MODES:
        ident = ms["identifiability"][key]
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(ident))
                 + f'" class="identline {cls}"/>')
    for key, cls, lab in MASK_MODES:
        rec = ms["recall"][key]
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(rec))
                 + f'" class="revline {cls}"/>')
        for i, v in enumerate(rec):
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="3.4" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="10" class="hit" '
                     f'data-x="{fr[i]:.2f} masked" data-y="{v:.2f}" '
                     f'data-xl="fraction" data-yl="recall, {lab} masking"/>')
    for i, (key, cls, lab) in enumerate(MASK_MODES):
        tol = m.get(f"mask_{key}_tolerance", float("nan"))
        adv = m.get(f"mask_{key}_completion_advantage", float("nan"))
        o.append(f'<text x="{W-pad_r+14}" y="{pad_t+26+i*52:.0f}" class="revlab {cls}">'
                 f'{e(lab)}</text>')
        o.append(f'<text x="{W-pad_r+14}" y="{pad_t+42+i*52:.0f}" class="rsub">'
                 f'tolerance {tol:.2f}</text>')
        o.append(f'<text x="{W-pad_r+14}" y="{pad_t+58+i*52:.0f}" class="rsub">'
                 f'advantage {adv:+.2f}</text>')
    o.append(f'<text x="{W-pad_r+14}" y="{pad_t+26+3*52+8:.0f}" class="rsub">'
             f'faint line = model-free</text>')
    o.append(f'<text x="{W-pad_r+14}" y="{pad_t+26+3*52+24:.0f}" class="rsub">'
             f'cue identifiability</text>')
    o.append('</svg>')
    return ('<figure class="chart chart-wide"><figcaption><strong>Which features are '
            'missing matters more than how many</strong><span>Solid: recall. Faint: '
            'the model-free reference — the fraction of masked cues whose nearest '
            'studied embedding is still the right item, computed from the encoder '
            'before any arm runs. Removing category features is nearly free: recall '
            'holds at 1.00 with 92% of the shared block gone, because identity alone '
            'still names the item. Removing identity features costs everything at '
            'f&nbsp;=&nbsp;0.5 — and there the reference is <em>0.00</em> while recall '
            'is 0.29, so the arm is recovering a target the cue no longer specifies. '
            'That gap is pattern completion in the strict sense, as against cue '
            'matching.</span></figcaption>' + "".join(o) + '</figure>')


def mask_margin_chart(ms, m):
    """The margin, which keeps resolving where recall saturates."""
    if not ms:
        return ""
    fr = ms["fractions"]
    W, H = 468, 310
    pad_l, pad_r, pad_t, pad_b = 56, 22, 24, 66
    lo_, hi_ = -0.12, 0.60
    tx = lambda i: pad_l + i / max(len(fr) - 1, 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - (v - lo_) / (hi_ - lo_)) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Decision margin against masked fraction">']
    o.append(f'<rect x="{pad_l}" y="{ty(0):.1f}" width="{W-pad_l-pad_r}" '
             f'height="{ty(lo_)-ty(0):.1f}" class="failband"/>')
    for g in (0.0, 0.2, 0.4, 0.6):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" '
                 f'class="{"zeroline" if g == 0 else "gl"}"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    for key, cls, lab in MASK_MODES:
        vals = ms["margin"][key][:-1]          # f = 1.0 is an empty cue
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(vals))
                 + f'" class="revline {cls}"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="3.2" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="9" class="hit" '
                     f'data-x="{fr[i]:.2f} masked" data-y="{v:+.2f}" '
                     f'data-xl="fraction" data-yl="margin, {lab} masking"/>')
    for i, f in enumerate(fr[:-1]):
        if i % 3 == 0:
            o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                     f'class="atick">{f:.2f}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-34}" text-anchor="middle" '
             f'class="axlabel">fraction of active features removed</text>')
    o.append(f'<text transform="translate(13,{(pad_t+ty(lo_))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">margin: target − best other</text>')
    o.append('</svg>')
    asym = m.get("mask_block_asymmetry", float("nan"))
    return ('<figure class="chart"><figcaption><strong>The margin separates the blocks '
            'where recall cannot</strong><span>In-bound recall saturates for this arm, '
            f'so the recall asymmetry reads {asym:+.2f}. The margin does not: masking '
            'category features <em>raises</em> it to 0.52 — fewer shared dimensions '
            'means fewer competitors — while masking identity drives it to −0.01, '
            'below the boundary where a different item outranks the true one. Same '
            'quantity as the rollout margin, same reason for using '
            'it.</span></figcaption>' + "".join(o) + '</figure>')


def mask_probe_chart(m):
    """Cued masking against rollout with only the initial cue masked."""
    modes = [(k, cls, lab) for k, cls, lab in MASK_MODES
             if f"mask_{k}_auc" in m]
    if not modes:
        return ""
    W, H = 468, 310
    pad_l, pad_r, pad_t, pad_b = 54, 20, 24, 74
    gw = (W - pad_l - pad_r) / len(modes)
    bw = gw / 2 - 9
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    base = ty(0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Cued recall against autoregressive rollout under masking">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    for i, (key, cls, lab) in enumerate(modes):
        x0 = pad_l + i * gw
        for j, (suffix, bcls, what) in enumerate(
                (("auc", "bar-ctl", "cued, every position masked"),
                 ("rollout_auc", "bar-abac", "rollout, initial cue only"))):
            v = m.get(f"mask_{key}_{suffix}", float("nan"))
            if v != v:
                continue
            bx = x0 + 5 + j * (bw + 6)
            o.append(f'<rect x="{bx:.1f}" y="{ty(v):.1f}" width="{bw:.1f}" '
                     f'height="{max(base-ty(v),1.5):.1f}" rx="2.5" class="{bcls}"/>')
            o.append(f'<text x="{bx+bw/2:.1f}" y="{ty(v)-6:.1f}" text-anchor="middle" '
                     f'class="barval sm">{v:.2f}</text>')
            o.append(f'<rect x="{bx:.1f}" y="{pad_t}" width="{bw:.1f}" '
                     f'height="{base-pad_t:.1f}" class="hit" data-x="{e(lab)}" '
                     f'data-y="{v:.2f}" data-xl="masking mode" data-yl="{what}"/>')
        for li, part in enumerate(lab.split(" ")):
            o.append(f'<text x="{x0+gw/2:.1f}" y="{H-pad_b+18+li*12:.0f}" '
                     f'text-anchor="middle" class="atick">{e(part)}</text>')
    o.append(f'<text transform="translate(13,{(pad_t+base)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">mean recall over the sweep</text>')
    o.append('</svg>')
    gap = m.get("mask_completion_gap", float("nan"))
    legend = ('<div class="legend"><span><i class="sw sw-ctl"></i>cued — every '
              'position masked</span><span><i class="sw sw-abac"></i>rollout — only '
              'the initial cue</span></div>')
    return ('<figure class="chart"><figcaption><strong>Both probes, as paragraph 17 '
            'asks</strong><span>Cued masking degrades every cue independently; rollout '
            'masks only the first item and then free-runs, so it scores how an initial '
            f'error propagates. The gap between them is {gap:.2f} — an initial '
            'fragment costs more than the same fragment supplied at every step, '
            'because under rollout the arm has to survive its own output as well as '
            'the missing features.</span></figcaption>' + "".join(o) + legend + '</figure>')


# ------------------------------------------------------------ serial order ----

def serial_order_weight_chart(d):
    """Where Serial order's weight sits, and how much of it this arm can reach."""
    v = (d.get("capacities") or {}).get("Serial order") or {}
    dims = v.get("dimensions") or {}
    if not dims:
        return ""
    order = ["unrolling", "binding_ordinal", "establishment",
             "interval_retention", "interval_generation"]
    rows = [(k, dims[k]) for k in order if k in dims]
    W = 468
    bar_h, gap = 34, 10
    pad_l, pad_t, pad_r = 152, 30, 74
    H = pad_t + len(rows) * (bar_h + gap) + 42
    span = W - pad_l - pad_r
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Serial order dimension weights and their status">']
    y = pad_t
    for k, dv in rows:
        w_ = dv["weight"]
        st = dv["status"]
        cls = {"scored": "wb-ok", "unresolved": "wb-guard",
               "not_applicable": "wb-na", "unbuilt": "wb-na"}.get(st, "wb-na")
        bw = span * (w_ / 0.25)
        o.append(f'<text x="{pad_l-10}" y="{y+bar_h/2+4:.0f}" text-anchor="end" '
                 f'class="wlab">{e(k)}</text>')
        o.append(f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_h}" '
                 f'rx="4" class="wbar {cls}"/>')
        sc = dv["score"]
        txt = (f'{sc:.2f}' if isinstance(sc, (int, float)) and not math.isnan(sc)
               else {"not_applicable": "n/a", "unresolved": "guard", "unbuilt": "—"}.get(st, "—"))
        o.append(f'<text x="{pad_l+bw+8:.1f}" y="{y+bar_h/2+4:.0f}" '
                 f'class="wval {cls}">{txt}</text>')
        o.append(f'<text x="{pad_l+7}" y="{y+bar_h/2+4:.0f}" '
                 f'class="wnum {"on-pale" if cls == "wb-na" else ""}">{w_:.2f}</text>')
        o.append(f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_h}" '
                 f'class="hit" data-x="{e(k)}" data-y="weight {w_:.2f}" '
                 f'data-xl="dimension" data-yl="{e(st)}"/>')
        y += bar_h + gap
    o.append(f'<text x="{pad_l}" y="{H-14}" class="axlabel">'
             f'bar length = share of the capacity</text>')
    o.append('</svg>')
    cov = v.get("coverage", 0.0)
    return ('<figure class="chart"><figcaption><strong>Half of Serial order is metric '
            'time, and this arm reaches none of it</strong><span>Paragraph 11 puts '
            'temporal distance on an equal footing with order, so the two interval '
            'read-outs carry half the capacity between them. Both are marked '
            '<em>not applicable</em> rather than scored 0: an ordinal-clocked arm has '
            'no state that elapsed time can change, so the benchmark has no meaning '
            f'for it. Coverage {cov:.0%} — the score above rests on that '
            'fraction.</span></figcaption>' + "".join(o)
            + '<div class="legend"><span><i class="sw wb-ok"></i>scored</span>'
              '<span><i class="sw wb-guard"></i>guard failed</span>'
              '<span><i class="sw wb-na"></i>not applicable / unbuilt</span></div>'
            + '</figure>')


def _so_grid(Ls):
    """'L = 10, 20, 30' -- the probe's grid as run, never a hard-coded one."""
    return "L = " + ", ".join(str(L) for L in Ls)


def unrolling_gap_chart(sp):
    """Cued span against rollout span across the length grid."""
    rows = (sp or {}).get("rows") or []
    if not rows:
        return ""
    W, H = 468, 300
    pad_l, pad_r, pad_t, pad_b = 52, 22, 24, 62
    Ls = [r["length"] for r in rows]
    ymax = max(max(r["cued_span"] for r in rows), 1) + 1
    tx = lambda i: pad_l + i / max(len(Ls) - 1, 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v / ymax) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Cued span against rollout span by list length">']
    for g in range(0, ymax + 1, 2):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g}</text>')
    # the gap itself, as the shaded area between the two
    up = " ".join(f"{tx(i):.1f},{ty(r['cued_span']):.1f}" for i, r in enumerate(rows))
    dn = " ".join(f"{tx(i):.1f},{ty(r['rollout_span']):.1f}"
                  for i, r in reversed(list(enumerate(rows))))
    o.append(f'<polygon points="{up} {dn}" class="gapband"/>')
    for key, cls, lab in (("cued_span", "s-acc", "cued"),
                          ("rollout_span", "s-per", "rollout, raw")):
        vals = [r[key] for r in rows]
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in enumerate(vals))
                 + f'" class="revline {cls}"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="4" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="11" class="hit" '
                     f'data-x="L={Ls[i]}" data-y="{v}" data-xl="list length" '
                     f'data-yl="{lab} span"/>')
    o.append(f'<text x="{tx(len(Ls)-1)-6:.1f}" y="{ty(rows[-1]["cued_span"])-10:.1f}" '
             f'text-anchor="end" class="revlab s-acc">cued</text>')
    o.append(f'<text x="{tx(len(Ls)-1)-6:.1f}" y="{ty(rows[-1]["rollout_span"])+18:.1f}" '
             f'text-anchor="end" class="revlab s-per">rollout</text>')
    for i, L in enumerate(Ls):
        o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{L}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-30}" text-anchor="middle" '
             f'class="axlabel">list length L</text>')
    o.append(f'<text transform="translate(13,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">memory span</text>')
    o.append('</svg>')
    gap = (sp.get("summary") or {}).get("unrolling_gap", float("nan"))
    st = sp.get("staircase") or []
    at = [t for t in st if t.get("cued_reached")]
    spans = "; ".join(f'L = {t["length"]}: cued {t["span_cued_at_criterion"]}, '
                      f'rollout {t["span_rollout_at_criterion"]}' for t in at)
    missing = [t["length"] for t in st if not t.get("cued_reached")]
    return ('<figure class="chart"><figcaption><strong>Associations present, unrolling '
            'collapses</strong><span>The shaded wedge is <code>unrolling_gap</code> = '
            f'{gap:.2f}, over {_so_grid([r["length"] for r in rows])}. Spans at cued '
            f'criterion — {spans or "none reached"}.'
            + (f' Cued recall never reached criterion at L = '
               f'{", ".join(map(str, missing))}, so no gap is read there.' if missing else '')
            + ' Both spans are read at the exposure where CUED recall first reaches '
            'criterion, so the clause\'s antecedent — the associations are present — '
            'holds by construction rather than by assumption.</span></figcaption>'
            + "".join(o) + '</figure>')


def staircase_chart(sp):
    """Exposure to cued criterion against exposure to rollout criterion."""
    st = (sp or {}).get("staircase") or []
    if not st:
        return ""
    W, H = 468, 310
    pad_l, pad_r, pad_t, pad_b = 56, 22, 24, 66
    Ls = [t["length"] for t in st]
    hi = max([t["e_rollout"] or t["budget"] for t in st] + [2])
    lo_ = math.log2(1); hi_ = math.log2(max(hi, 2))
    tx = lambda i: pad_l + i / max(len(Ls) - 1, 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - (math.log2(max(v, 1)) - lo_) / max(hi_ - lo_, 1e-9)) \
        * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Exposure to cued and rollout criterion by list length">']
    ticks = [2 ** k for k in range(0, 10) if 2 ** k <= hi] or [1, hi]
    ticks = ticks[::2] if len(ticks) > 6 else ticks
    for g in ticks:
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g}</text>')
    for i, t in enumerate(st):
        if t["e_cued"] and t["e_rollout"]:
            o.append(f'<line x1="{tx(i):.1f}" y1="{ty(t["e_cued"]):.1f}" '
                     f'x2="{tx(i):.1f}" y2="{ty(t["e_rollout"]):.1f}" class="dumbbell"/>')
    for key, cls, lab in (("e_cued", "s-acc", "cued reaches criterion"),
                          ("e_rollout", "s-per", "rollout reaches criterion")):
        pts = [(i, t[key]) for i, t in enumerate(st) if t[key]]
        o.append('<polyline points="' + " ".join(f"{tx(i):.1f},{ty(v):.1f}"
                                                 for i, v in pts)
                 + f'" class="revline {cls}"/>')
        for i, v in pts:
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="4.5" class="revdot {cls}"/>')
            o.append(f'<circle cx="{tx(i):.1f}" cy="{ty(v):.1f}" r="12" class="hit" '
                     f'data-x="L={st[i]["length"]}" data-y="{v}" data-xl="list length" '
                     f'data-yl="{lab}, epochs"/>')
    for i, t in enumerate(st):
        if t["exposure_ratio"] and math.isfinite(t["exposure_ratio"]) and t["exposure_ratio"] > 1.5:
            o.append(f'<text x="{tx(i)+7:.1f}" y="{(ty(t["e_cued"])+ty(t["e_rollout"]))/2+4:.1f}" '
                     f'class="ratiolab">{t["exposure_ratio"]:.0f}×</text>')
    o.append(f'<text x="{tx(0)+6:.1f}" y="{ty(1)-9:.1f}" class="revlab s-acc">cued</text>')
    o.append(f'<text x="{tx(len(Ls)-1)-6:.1f}" y="{ty(st[-1]["e_rollout"] or 2)-11:.1f}" '
             f'text-anchor="end" class="revlab s-per">rollout</text>')
    for i, L in enumerate(Ls):
        o.append(f'<text x="{tx(i):.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{L}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-32}" text-anchor="middle" '
             f'class="axlabel">list length L</text>')
    o.append(f'<text transform="translate(14,{(pad_t+ty(1))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">presentations to criterion (log₂)</text>')
    o.append('</svg>')
    sm = sp.get("summary") or {}
    ratio = sm.get("unrolling_exposure_ratio", float("nan"))
    budget = max(t.get("budget") or 0 for t in st)
    per_L = "; ".join(
        f'L = {t["length"]}: cued {t["e_cued"] if t.get("cued_reached") else f"&gt;{budget}"}'
        f', rollout {t["e_rollout"] if t.get("rollout_reached") else f"&gt;{budget}"}'
        for t in st)
    censored = [t["length"] for t in st if not t.get("rollout_reached")]
    if censored:
        head = (f'Rollout is censored at the {budget}-presentation budget from '
                f'L = {min(censored)}')
        verdict = (f'Where both reach criterion the mean cost is {ratio:.1f}×; beyond '
                   'that the budget cannot tell an expensive unroll from an impossible '
                   'one, so the censored lengths are not a sample-efficiency figure.')
    else:
        head = 'Unrolling is not impossible here — it is expensive'
        verdict = (f'Rollout reaches criterion at every length, at a mean {ratio:.1f}× '
                   'the cued exposure: a sample-efficiency gap, not an architectural '
                   'inability — a distinction a single fixed-budget span cannot make.')
    return (f'<figure class="chart"><figcaption><strong>{head}</strong><span>Each arm '
            'is staircased against its own convergence rather than run at whatever '
            'epoch count its registry entry carries. Presentations to criterion — '
            f'{per_L}. {verdict}</span></figcaption>' + "".join(o) + '</figure>')


def cascade_conditional_chart(sp):
    """P(correct next | correct now) against P(correct next | error now)."""
    rows = [r for r in (sp or {}).get("rows") or []
            if r.get("cascade_recovery_rate") == r.get("cascade_recovery_rate")]
    if not rows:
        return ""
    W, H = 468, 300
    pad_l, pad_r, pad_t, pad_b = 52, 22, 24, 62
    Ls = [r["length"] for r in rows]
    gw = (W - pad_l - pad_r) / len(Ls)
    bw = gw / 2 - 7
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    base = ty(0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Probability of a correct step after a correct step vs after an error">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    for i, r in enumerate(rows):
        x0 = pad_l + i * gw
        for j, (key, cls, lab) in enumerate(
                (("p_correct_after_correct", "bar-ctl", "after a correct step"),
                 ("cascade_recovery_rate", "bar-abac", "after an error"))):
            v = r[key]
            if v != v:
                continue
            bx = x0 + 4 + j * (bw + 5)
            o.append(f'<rect x="{bx:.1f}" y="{ty(v):.1f}" width="{bw:.1f}" '
                     f'height="{max(base-ty(v),1.5):.1f}" rx="2.5" class="{cls}"/>')
            o.append(f'<rect x="{bx:.1f}" y="{pad_t}" width="{bw:.1f}" '
                     f'height="{base-pad_t:.1f}" class="hit" data-x="L={r["length"]}" '
                     f'data-y="{v:.2f}" data-xl="list length" data-yl="{lab}"/>')
        o.append(f'<text x="{x0+gw/2:.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                 f'class="atick">{r["length"]}</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-30}" text-anchor="middle" '
             f'class="axlabel">list length L</text>')
    o.append(f'<text transform="translate(13,{(pad_t+base)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">P(next step correct)</text>')
    o.append('</svg>')
    legend = ('<div class="legend"><span><i class="sw sw-ctl"></i>after a correct '
              'step</span><span><i class="sw sw-abac"></i>after an error</span></div>')
    sm = sp.get("summary") or {}
    return ('<figure class="chart"><figcaption><strong>An error is close to '
            'terminal</strong><span>Conditional per-step accuracy under free rollout — '
            'the decomposition paragraph 11 asks for. Recovery after an error is '
            f'{sm.get("cascade_recovery_rate", float("nan")):.2f}, against '
            f'{sm.get("cascade_conditional_ratio", float("nan")):.1f}× that after a '
            'correct step. Lists short enough never to break contribute no bars: with '
            'no error there is nothing to condition on.</span></figcaption>'
            + "".join(o) + legend + '</figure>')


def order_taxonomy_strip(sp):
    """The binding_ordinal readouts. Each is a conditional on a cued failure, so
    with no failures they are shown undefined rather than perfect."""
    sm = (sp or {}).get("summary") or {}
    rows = (sp or {}).get("rows") or []
    if not rows:
        return ""
    n_probes, n_fail = sm.get("n_cued_probes", 0), sm.get("n_cued_failures", 0)

    def _v(k):
        x = sm.get(k)
        return (f"{x:.2f}" if isinstance(x, (int, float)) and math.isfinite(x)
                else "undefined")
    items = [
        ("cued probes", f"{n_probes:,.0f}", f"across {_so_grid([r['length'] for r in rows])}"),
        ("failures to classify", f"{n_fail:,.0f}", "the denominator of every metric below"),
        ("order_error_fraction", _v("order_error_fraction"), "transpositions ÷ failures"),
        ("transposition_locality", _v("transposition_locality"),
         "share of order errors at |d| = 1"),
        ("transposition_asymmetry", _v("transposition_asymmetry"),
         "forward share; the architecture fingerprint"),
        ("extra_list_intrusion_rate", _v("extra_list_intrusion_rate"),
         "decodes to a never-studied item"),
        ("order_given_item", f'{sm.get("order_given_item", float("nan")):.2f}',
         "P(correct next | decode is a studied item)"),
        ("list_membership_rate", f'{sm.get("list_membership_rate", float("nan")):.2f}',
         "decodes landing inside the studied list"),
        ("establishment_break_length", f'{sm.get("establishment_break_length", 0):.0f}',
         "longest L with order_given_item ≥ 0.75"
         + (" — the grid maximum, so a ceiling"
            if sm.get("establishment_break_length", 0) >= max(r["length"] for r in rows)
            else "")),
    ]
    cells = "".join(
        f'<div class="statcell{" undef" if v == "undefined" else ""}">'
        f'<p class="statk">{e(k)}</p><p class="statv">{e(v)}</p>'
        f'<p class="statn">{e(n)}</p></div>' for k, v, n in items)
    if not n_fail:
        head, body = ('Order errors: the instrument has nothing to bite on',
                      'Every <code>binding_ordinal</code> metric is a conditional on a '
                      'cued failure — a transposition gradient needs transpositions. '
                      f'This arm produced none in {n_probes:,.0f} probes, so they are '
                      'undefined rather than perfect, and the dimension is excluded '
                      'from the score rather than counted as 1.00.')
    else:
        loc = sm.get("transposition_locality")
        body = ('Every <code>binding_ordinal</code> metric is a conditional on a '
                f'cued failure; this arm produced {n_fail:,.0f}. A transposition is '
                'labelled by where its answer was studied, not by why it was given.')
        if isinstance(loc, (int, float)) and math.isfinite(loc) and loc < 0.5:
            body += (f' Locality is {loc:.2f}: most answers land far from the target, '
                     'which is not the neighbour-swap signature of lost order.')
        if (sp or {}).get("multi_list"):
            body += (' The blocked multi-list condition in '
                     '<code>serial_order_failures.png</code> puts the same items in '
                     'separate lists, so errors landing there count as other-list '
                     'intrusions instead.')
        head = 'Order errors: what a cued failure is'
    return ('<figure class="chart chart-wide"><figcaption><strong>' + head +
            '</strong><span>' + body + '</span></figcaption>'
            f'<div class="statgrid">{cells}</div></figure>')


# ---------------------------------------------------- continual retention ----

def retention_matrix_chart(cc, m):
    """R[j, i]: recall of task i measured after task j. Lower triangle is
    stability, the diagonal is acquisition under load."""
    R = cc.get("retention_matrix")
    labels = cc.get("task_labels") or []
    if not R:
        return ""
    T = len(R)
    cell, gap = 46, 3
    pad_l, pad_t = 96, 46
    W = pad_l + T * (cell + gap) + 16
    H = pad_t + T * (cell + gap) + 46

    def ramp(v):
        for k, thr in enumerate((0.2, 0.4, 0.6, 0.8)):
            if v < thr:
                return f"h{k}"
        return "h4"

    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Retention matrix: recall of each task after each later task">']
    for i in range(T):
        o.append(f'<text x="{pad_l + i*(cell+gap) + cell/2:.0f}" y="{pad_t-14}" '
                 f'text-anchor="middle" class="mlab">{e(labels[i][:6] if i < len(labels) else i)}</text>')
        o.append(f'<text x="{pad_l-10}" y="{pad_t + i*(cell+gap) + cell/2 + 4:.0f}" '
                 f'text-anchor="end" class="mlab">after {e(labels[i][:6] if i < len(labels) else i)}</text>')
    for j in range(T):
        for i in range(T):
            v = R[j][i]
            x = pad_l + i*(cell+gap)
            y = pad_t + j*(cell+gap)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                o.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'rx="3" class="hcell-na"/>')
                continue
            diag = " diag" if i == j else ""
            o.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" '
                     f'class="hcell {ramp(v)}{diag}"/>')
            o.append(f'<text x="{x+cell/2}" y="{y+cell/2+4}" text-anchor="middle" '
                     f'class="hval {"on-dark" if v >= 0.6 else "on-light"}">{v:.2f}</text>')
            o.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" class="hit" '
                     f'data-x="{e(labels[i][:8] if i < len(labels) else i)}" data-y="{v:.2f}" '
                     f'data-xl="task" data-yl="recall after task '
                     f'{e(labels[j][:8] if j < len(labels) else j)}"/>')
    o.append(f'<text x="{pad_l + T*(cell+gap)/2:.0f}" y="{H-10}" text-anchor="middle" '
             f'class="axlabel">task, in the order it was learned →</text>')
    o.append('</svg>')
    acc = m.get("chain_avg_accuracy"); fg = m.get("chain_avg_forgetting")
    la = m.get("chain_avg_learning")
    return ('<figure class="chart"><figcaption><strong>Stability: nothing decays, '
            'because nothing competes</strong><span>Outlined cells are the diagonal '
            '— each task scored the moment it was learned, with every earlier task '
            f'already stored (acquisition under load, LA {la:.2f}). Below it is what '
            f'survived: ACC {acc:.2f}, forgetting {fg:.3f}. The matrix is flat '
            'because the chain cannot load this arm — 6 categories at ≤6 words is 30 '
            'associations in a 100-dimensional store, and a re-run at the '
            'vocabulary ceiling (6×6) still holds ACC 0.999. Read it as an untested '
            'axis, not a perfect score.</span></figcaption>' + "".join(o) + '</figure>')


def abac_chart(m):
    """AB retention under colliding cues against the disjoint-cue control."""
    W, H = 468, 268
    pad_l, pad_r, pad_t, pad_b = 54, 16, 22, 58
    conds = [("control", "disjoint cues\n(d → e)", "bar-ctl"),
             ("abac", "same cues re-paired\n(a → c)", "bar-abac")]
    vals = [m.get(f"pa_{c}_ab_retention", float("nan")) for c, _, _ in conds]
    gw = (W - pad_l - pad_r) / len(conds)
    bw = min(120, gw * 0.5)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    base = ty(0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="AB retention under cue collision against a disjoint-cue control">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    for k, ((cond, lab, cls), v) in enumerate(zip(conds, vals)):
        cx = pad_l + k*gw + gw/2
        h = max(base - ty(v), 1.5)
        o.append(f'<rect x="{cx-bw/2:.1f}" y="{ty(v):.1f}" width="{bw:.1f}" '
                 f'height="{h:.1f}" rx="3" class="{cls}"/>')
        o.append(f'<text x="{cx:.1f}" y="{ty(v)-9:.1f}" text-anchor="middle" '
                 f'class="barval">{v:.2f}</text>')
        for li, part in enumerate(lab.split("\n")):
            o.append(f'<text x="{cx:.1f}" y="{H-pad_b+18+li*13:.0f}" text-anchor="middle" '
                     f'class="atick">{e(part)}</text>')
        o.append(f'<rect x="{cx-bw/2:.1f}" y="{pad_t}" width="{bw:.1f}" '
                 f'height="{base-pad_t:.1f}" class="hit" data-x="{e(cond)}" '
                 f'data-y="{v:.2f}" data-xl="phase 2" data-yl="AB retention"/>')
    o.append(f'<text transform="translate(13,{(pad_t+base)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">AB retention after phase 2</text>')
    o.append('</svg>')
    cost = m.get("pa_cue_competition_cost", float("nan"))
    other = m.get("pa_abac_other_rate_final", float("nan"))
    return ('<figure class="chart"><figcaption><strong>Stability: everything decays, '
            'the moment cues collide</strong><span>Equal pair count, equal trials, '
            f'equal epochs — only the cues differ. Cue-competition cost {cost:.2f}, '
            'the maximum. The old target is not corrupted but cleanly replaced: '
            f'intrusion rate into neither-target is {other:.2f}, and A→C reaches '
            'criterion in 3 trials. This is the load the retention matrix could not '
            'apply.</span></figcaption>' + "".join(o) + '</figure>')


def reversal_chart(curves, m, proto="extinction"):
    """Arm accuracy against perseveration across the reversal protocol's stages."""
    c = curves.get(f"reversal_{proto}_curves")
    if not c:
        return ""
    stages = [st for st in ("acquisition", "extinction", "reversal", "rereversal")
              if st in c]
    W, H = 980, 300
    pad_l, pad_r, pad_t, pad_b = 56, 186, 26, 52
    lens = [len(c[st]["arm_accuracy"]) for st in stages]
    total = sum(lens)
    tx = lambda k: pad_l + k / max(total - 1, 1) * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Arm accuracy and perseveration across the reversal stages">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
    off = 0
    for si, st in enumerate(stages):
        n = lens[si]
        x0, x1 = tx(off), tx(off + n - 1)
        if si % 2:
            o.append(f'<rect x="{x0-6:.1f}" y="{pad_t}" width="{x1-x0+12:.1f}" '
                     f'height="{ty(0)-pad_t:.1f}" class="stageband"/>')
        o.append(f'<text x="{(x0+x1)/2:.1f}" y="{pad_t-10}" text-anchor="middle" '
                 f'class="stagelab">{st}</text>')
        if si:
            o.append(f'<line x1="{x0-6:.1f}" y1="{pad_t}" x2="{x0-6:.1f}" '
                     f'y2="{ty(0):.1f}" class="stagedivide"/>')
        off += n
    for key, cls, lab in (("arm_accuracy", "s-acc", "arm accuracy (new contingency)"),
                          ("perseveration", "s-per", "perseveration (old arm)")):
        pts, off = [], 0
        for si, st in enumerate(stages):
            for k, v in enumerate(c[st][key]):
                pts.append((tx(off + k), ty(v), v, st, k))
            off += lens[si]
        o.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in pts)
                 + f'" class="revline {cls}"/>')
        for x, y, v, st, k in pts:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" class="revdot {cls}"/>')
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" class="hit" '
                     f'data-x="{st} trial {k+1}" data-y="{v:.2f}" data-xl="stage" '
                     f'data-yl="{lab}"/>')
    o.append(f'<text x="{W-pad_r+14}" y="{pad_t+26}" class="revlab s-acc">arm accuracy</text>')
    o.append(f'<text x="{W-pad_r+14}" y="{pad_t+40}" class="rsub">the new, now-correct arm</text>')
    o.append(f'<text x="{W-pad_r+14}" y="{pad_t+68}" class="revlab s-per">perseveration</text>')
    o.append(f'<text x="{W-pad_r+14}" y="{pad_t+82}" class="rsub">the old, now-invalid arm</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-10}" text-anchor="middle" '
             f'class="axlabel">trial, within stage</text>')
    o.append('</svg>')
    ttc = m.get(f"reversal_{proto}_reversal_trials_to_criterion", float("nan"))
    per = m.get(f"reversal_{proto}_reversal_final_perseveration", float("nan"))
    return ('<figure class="chart chart-wide"><figcaption><strong>Plasticity: the old '
            'response has to go before the new one appears</strong><span>The two '
            'curves are mirror images through the reversal stage — perseveration '
            f'falls exactly as arm accuracy rises, reaching criterion in {ttc:.0f} '
            f'trials with final perseveration {per:.2f}. This is the measure the '
            'retention matrix cannot substitute for: here, holding a learned '
            'association is the failure.</span></figcaption>' + "".join(o) + '</figure>')


def stability_plasticity_chart(m):
    """The joint position: retention against willingness to overwrite."""
    W, H = 468, 330
    pad_l, pad_r, pad_t, pad_b = 56, 24, 24, 56
    stab = m.get("chain_retention_ratio", float("nan"))
    pers = m.get("reversal_direct_reversal_final_perseveration", float("nan"))
    plast = 1.0 - pers if pers == pers else float("nan")
    tx = lambda v: pad_l + v * (W - pad_l - pad_r)
    ty = lambda v: pad_t + (1 - v) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Joint stability and plasticity position">']
    o.append(f'<rect x="{tx(0.5):.1f}" y="{pad_t}" width="{tx(1)-tx(0.5):.1f}" '
             f'height="{ty(0.5)-pad_t:.1f}" class="goodquad"/>')
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<line x1="{pad_l}" y1="{ty(g):.1f}" x2="{W-pad_r}" y2="{ty(g):.1f}" class="gl"/>')
        o.append(f'<line x1="{tx(g):.1f}" y1="{pad_t}" x2="{tx(g):.1f}" y2="{ty(0):.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{ty(g)+3.5:.1f}" text-anchor="end" class="atick">{g:g}</text>')
        o.append(f'<text x="{tx(g):.1f}" y="{H-pad_b+18:.0f}" text-anchor="middle" class="atick">{g:g}</text>')
    for x, y, lab, anch in ((0.06, 0.96, "rigid", "start"), (0.94, 0.06, "labile", "end"),
                            (0.06, 0.06, "broken", "start")):
        o.append(f'<text x="{tx(x):.1f}" y="{ty(y):.1f}" text-anchor="{anch}" '
                 f'class="quadlab">{lab}</text>')
    o.append(f'<text x="{tx(0.53):.1f}" y="{ty(0.55):.1f}" text-anchor="start" '
             f'class="quadlab good">balanced</text>')
    if stab == stab and plast == plast:
        o.append(f'<circle cx="{tx(plast):.1f}" cy="{ty(stab):.1f}" r="9" class="jointdot"/>')
        o.append(f'<circle cx="{tx(plast):.1f}" cy="{ty(stab):.1f}" r="17" class="hit" '
                 f'data-x="{plast:.2f}" data-y="{stab:.2f}" data-xl="plasticity" '
                 f'data-yl="stability"/>')
        o.append(f'<text x="{tx(plast):.1f}" y="{ty(stab)+28:.1f}" text-anchor="end" '
                 f'class="pointlab">AHN</text>')
    o.append(f'<text x="{(pad_l+W-pad_r)/2:.0f}" y="{H-10}" text-anchor="middle" '
             f'class="axlabel">plasticity → 1 − perseveration after reversal</text>')
    o.append(f'<text transform="translate(14,{(pad_t+ty(0))/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" class="axlabel">stability → chain retention ratio</text>')
    o.append('</svg>')
    return ('<figure class="chart"><figcaption><strong>The joint position, and what it '
            'cannot settle</strong><span>The two axes come from different suites on '
            'purpose: no single policy on weight change scores well on both, so the '
            'pair cannot be gamed the way either half can. AHN lands in the balanced '
            'corner — but its stability coordinate comes from a chain that never '
            'loaded it, and the AB/AC panel puts the same arm at 0.00 retention. Read '
            'the corner as <em>not yet contradicted</em>, not as '
            'established.</span></figcaption>' + "".join(o) + '</figure>')


def span_chart(ls):
    W, H = 560, 230
    pad_l, pad_r, pad_t, pad_b = 46, 16, 14, 40
    lengths, raw, quant = ls["lengths"], ls["span_raw"], ls["span_quantized"]
    ymax = max(max(quant), max(raw)) or 1
    bw = (W - pad_l - pad_r) / len(lengths)
    ty = lambda v: pad_t + (1 - v / ymax) * (H - pad_t - pad_b)
    o = [f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" '
         f'aria-label="Memory span by list length under two feedback modes">']
    for g in range(0, ymax + 1, 2):
        y = ty(g)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="gl"/>')
        o.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" text-anchor="end" class="atick">{g}</text>')
    base = ty(0)
    for i, (L, r, q) in enumerate(zip(lengths, raw, quant)):
        x = pad_l + i * bw
        w = bw / 2 - 5
        for j, (v, cls, lab) in enumerate(((q, "bar-q", "codebook feedback"),
                                           (r, "bar-r", "raw feedback"))):
            bx = x + 4 + j * (w + 3)
            o.append(f'<rect x="{bx:.1f}" y="{ty(v):.1f}" width="{w:.1f}" '
                     f'height="{max(base-ty(v),1):.1f}" rx="3" class="{cls}"/>')
            o.append(f'<rect x="{bx:.1f}" y="{pad_t}" width="{w:.1f}" '
                     f'height="{base-pad_t:.1f}" class="hit" data-x="L={L}" '
                     f'data-y="{v}" data-xl="{lab}" data-yl="span"/>')
        o.append(f'<text x="{x+bw/2:.1f}" y="{H-pad_b+16}" text-anchor="middle" '
                 f'class="atick">{L}</text>')
    o.append(f'<text x="{(W+pad_l-pad_r)/2:.0f}" y="{H-6}" text-anchor="middle" '
             f'class="axlabel">list length L</text></svg>')
    legend = ('<div class="legend"><span><i class="sw sw-q"></i>codebook feedback '
              '(drift-free bound)</span><span><i class="sw sw-r"></i>raw feedback</span></div>')
    return ('<figure class="chart chart-wide"><figcaption><strong>Span collapses under raw '
            'feedback, not under capacity</strong><span>Codebook feedback holds at '
            'the ceiling L&minus;1 at every length; raw feedback tracks it to '
            'L&nbsp;=&nbsp;7 and then falls to 2. The gap is readout drift, not a '
            'storage limit.</span></figcaption>' + "".join(o) + legend + "</figure>")


# ------------------------------------------------------------ page body ----
def build(d: Dict[str, Any], series: Dict[str, Any]) -> str:
    caps, incl = d["capacities"], d["capacities_inclusive"]
    rows: List[Dict[str, Any]] = d["metrics"]
    weights = d["dimension_weights"]
    meta = d["suite_metadata"]

    P: List[str] = []
    w = P.append

    # ---- masthead
    w('<header class="mast">')
    w('<p class="eyebrow">MemVal &middot; capacity scorecard</p>')
    w(f'<h1>{e(d["model"])}</h1>')
    w('<p class="lede">Every metric the spatial and symbolic suites emit, mapped to '
      'the dimension it manipulates and the capacity it evidences, then rolled up '
      'by a two-level weighted sum. Metrics that cannot mean what their number says '
      'are named and excluded rather than averaged in.</p>')
    w('<div class="runbar">')
    for suite, m in meta.items():
        ok = not m["status"].startswith("NOT APPLICABLE")
        w(f'<div class="run {"" if ok else "run-na"}">'
          f'<span class="run-k">{e(suite)}</span>'
          f'<span class="run-v">{m["sections"]} sections &middot; '
          f'{m["n_trials"]} trials</span>'
          f'<span class="run-s">{"ran" if ok else "not applicable"}</span></div>')
    w('</div>')
    w('</header>')

    # ---- summary strip
    w('<section class="band"><h2 class="sec">Capacity scores</h2>')
    w('<div class="cards">')
    for c in CAP_ORDER:
        p = caps[c]
        cov = p["coverage"]
        klass = "cov-low" if cov < 0.6 else ("cov-mid" if cov < 0.85 else "cov-hi")
        gap = abs((incl[c]["score"] or 0) - (p["score"] or 0))
        w(f'<article class="card"><h3>{e(c)}</h3>')
        w(f'<p class="score"><span class="num">{p["score"]:.2f}</span>'
          f'<span class="of">of 1.00</span></p>')
        w(f'<div class="meter"><div class="meter-fill {klass}" '
          f'style="width:{cov*100:.0f}%"></div></div>')
        w(f'<p class="covline {klass}">coverage {cov:.0%} &middot; '
          f'{sum(1 for k,v in p["dimensions"].items() if v["status"]=="scored")}'
          f'/{len(p["dimensions"])} dimensions</p>')
        w(f'<p class="blurb">{e(CAP_BLURB[c])}</p>')
        if gap > 0.02:
            w(f'<p class="flag">inclusive score {incl[c]["score"]:.2f} '
              f'&mdash; {gap:.2f} lower once excluded metrics are re-admitted</p>')
        w('</article>')
    w('</div></section>')

    # ---- radar
    w('<section class="band"><h2 class="sec">Profile</h2>')
    w('<div class="radar-wrap">')
    w(radar_svg(caps, incl))
    w('<div class="radar-key">')
    w('<p class="k"><i class="sw sw-prim"></i><strong>capacity score</strong>'
      '<span>validated metrics only &mdash; the headline number</span></p>')
    w('<p class="k"><i class="sw sw-incl"></i><strong>inclusive</strong>'
      '<span>protocol-limited metrics re-admitted at face value</span></p>')
    w('<p class="k"><i class="sw sw-disc"></i><strong>score &times; coverage</strong>'
      '<span>discounted by how much of the capacity the suite can measure</span></p>')
    w('<p class="foot">Where the solid and dashed rings coincide, nothing was '
      'excluded. Where the dotted ring falls far inside, the score rests on a '
      'fraction of the intended instruments.</p>')
    w('</div></div></section>')

    # ---- findings from the sweeps
    w('<section class="band"><h2 class="sec">What the sweeps show</h2>')
    w('<p class="secnote">Three scalar summaries in <code>metrics.json</code> read '
      '1.00 or 0.00 while the series underneath them is doing something. These '
      'charts are the source of the derived metrics in the tables below.</p>')
    w('<div class="charts">')
    ns = series.get("noise_sweep")
    if ns:
        w(line_chart(list(zip(ns["noise_levels"], ns["mrr"])),
                     "retrieval-cue Gaussian &sigma;", "MRR",
                     "Cued recall decays smoothly with cue corruption",
                     "Recall holds to &sigma;&nbsp;=&nbsp;0.2 and crosses the 0.5 "
                     "criterion at &sigma;&nbsp;=&nbsp;0.7. No cliff — this is the "
                     "graceful degradation an outer-product store predicts.",
                     marks=[{"x": 0.7, "label": "tolerance threshold"}]))
    ss = series.get("similarity_sweep")
    if ss and ss.get("epochs_to_criterion"):
        eps_ = ss["epochs_to_criterion"]
        pairs = sorted(zip(ss["cosine_similarities"], eps_))
        hardest = max(eps_) / max(min(eps_), 1)
        w(line_chart(pairs, "mean pairwise cosine between list items",
                     "presentations to criterion",
                     "Overlap costs exposure, not accuracy",
                     "Each variance is trained to its own recall criterion, so recall "
                     "no longer separates the rungs — every one of them gets there. "
                     f"What separates them is the cost: {max(eps_):g} presentations at "
                     f"cosine {max(ss['cosine_similarities']):.2f} against "
                     f"{min(eps_):g} at the easiest, a {hardest:.0f}&times; span. "
                     "Under the previous fixed 1-epoch budget this read as a collapse "
                     "in accuracy, which was the budget talking.",
                     log_x=True, y_max=max(eps_)))
    elif ss:
        w(line_chart(list(zip(ss["cosine_similarities"], ss["mrr"])),
                     "mean pairwise cosine between list items", "MRR",
                     "Recall against item overlap", "", log_x=True))
    ls = series.get("length_sweep")
    if ls:
        w(span_chart(ls))
    w('</div></section>')

    # ---- one-shot acquisition
    conv = series.get("convergence_curve")
    probe = d.get("_scale_probe")
    if conv or probe:
        m_all0 = d.get("_all_metrics", {})
        w('<section class="band"><h2 class="sec">One-shot acquisition</h2>')
        w('<p class="secnote">Acquisition is measured three ways, in order of how '
          'much they can be gamed: how many presentations a fresh list needs, '
          'whether that survives a substrate already full of other tasks, and '
          'whether the resulting trace is strong or merely correctly aimed.</p>')
        w('<div class="charts">')
        if conv:
            w(exposure_chart(conv, m_all0))
        cc0 = series.get("continual_chain") or {}
        if cc0:
            w(load_acquisition_chart(cc0))
        if probe:
            w(scale_probe_chart(probe))
        w('</div></section>')

    # ---- error cascade under autoregressive rollout
    casc = d.get("_cascade_probe")
    if casc:
        w('<section class="band"><h2 class="sec">Rollout and the error cascade</h2>')
        w('<p class="secnote">Every acquisition number above comes from <em>cued</em> '
          'recall, where each position is probed from the true previous item, so an '
          'error at one position cannot reach the next. Rollout removes that '
          'guardrail. It is the only protocol in the suite where an error propagates, '
          'and the gap between the two is what the one-shot score does not '
          'see.</p>')
        w('<div class="charts">')
        w(cascade_position_chart(casc, epochs=1))
        w(cascade_accuracy_heatmap(casc))
        w(cascade_margin_heatmap(casc))
        w(cascade_geometry_heatmaps(casc))
        w(cascade_margin_trajectory(casc))
        w(cascade_exposure_chart(casc))
        w('</div></section>')

    # ---- sequence disambiguation
    sd = series.get("symbolic_disambiguation")
    if sd:
        sm = d.get("_all_metrics", {})
        w('<section class="band"><h2 class="sec">Overlapping episodes</h2>')
        w('<p class="secnote">The symbolic twin of the T-maze, and it reaches two axes '
          'the spatial section cannot: more than two confusable episodes, and a '
          'discriminator whose <em>similarity</em> is swept rather than present or '
          'absent. Its probe also differs deliberately — a single '
          '<code>predict_next</code> at the divergence step, nothing fed back — so its '
          'accuracy is not confounded with rollout drift the way the spatial '
          'section\'s is. Read the two within their modality, never against each '
          'other.</p>')
        w('<div class="charts">')
        w(symdis_load_chart(sd, sm))
        w(symdis_axis_chart(sd, sm, "similarity"))
        w(symdis_axis_chart(sd, sm, "delay"))
        w(symdis_confusion_chart(sd, sm))
        w('</div></section>')

    # ---- pattern completion: the structural half
    ms = series.get("mask_sweep")
    if ms:
        mm = d.get("_all_metrics", {})
        w('<section class="band"><h2 class="sec">Partial cues</h2>')
        w('<p class="secnote">The <code>cue_masking</code> section supplies the half '
          'of paragraph 17 that had no instrument: <em>completeness</em>, as against '
          'the corruption the &sigma; sweep already measured. Features are removed '
          'entirely and the rest left exact, over a <code>HierarchicalEncoder</code> '
          'whose columns are owned by tree nodes — so an item\'s features split into '
          'a <strong>shared</strong> block it holds in common with its siblings and an '
          '<strong>identity</strong> block that tells it apart. That partition is what '
          'lets the section ask whether it matters <em>which</em> features are '
          'missing.</p>')
        w('<div class="charts">')
        w(mask_recall_chart(ms, mm))
        w(mask_margin_chart(ms, mm))
        w(mask_probe_chart(mm))
        w('</div></section>')

    # ---- serial order
    sop = d.get("_serial_order_probe")
    if sop:
        w('<section class="band"><h2 class="sec">Serial order</h2>')
        w('<p class="secnote">Rescoped on 2026-09-02: temporal contiguity is no longer '
          'a promise, so the lag-CRP is out and the capacity is exactly the four '
          'paragraph-11 clauses. Three of them need only the decoded identity that '
          '<code>measure_recall_associative</code> currently computes and throws away '
          '(audit S-O2); the numbers below come from a probe that keeps it, and are '
          'marked as such wherever they appear.</p>')
        w('<div class="charts">')
        w(serial_order_weight_chart(d))
        w(unrolling_gap_chart(sop))
        w(staircase_chart(sop))
        w(cascade_conditional_chart(sop))
        w(order_taxonomy_strip(sop))
        w('</div></section>')

    # ---- continual retention: the two sides
    cc = series.get("continual_chain") or {}
    m_all = d.get("_all_metrics", {})
    if cc or "reversal_direct_curves" in series:
        w('<section class="band"><h2 class="sec">Stability and plasticity</h2>')
        w('<p class="secnote">Continual retention is the one capacity that cannot be '
          'read off a single number, because its two halves pull in opposite '
          'directions on the same underlying quantity — how readily the substrate '
          'changes. A measure that rewards holding weights still and a measure that '
          'punishes it cannot both be gamed by one policy, which is what makes the '
          'pair worth reporting jointly. The panels are ordered by how hard each '
          'pushes on that: no competition, cue collision, then an association that '
          'has become false.</p>')
        w('<div class="charts">')
        if cc:
            w(retention_matrix_chart(cc, m_all))
        w(abac_chart(m_all))
        w(reversal_chart(series, m_all))
        w(stability_plasticity_chart(m_all))
        w('</div></section>')

    # ---- schema consistency
    if any(r in series for r in RUNGS):
        m = d.get("_schema_metrics", {})
        w('<section class="band"><h2 class="sec">Schema consistency</h2>')
        w('<p class="secnote">The section was reworked on 2026-09-02 and now '
          'resolves for this arm, where it previously read as vacuous. Two changes '
          'did it: acquisition moved off the sequence-mean MRR onto the single new '
          'transition, and the interference readout gained a protocol under which '
          'it measures damage rather than rehearsal. Acquisition below is the '
          'suite default (<code>extended</code>); damage is the '
          '<code>focused</code> run.</p>')
        w('<div class="charts">')
        w(schema_acquisition_chart(series, m))
        w(schema_ladder_chart(m))
        w(schema_interference_chart(m))
        w('</div>')
        w('<div class="legend rung-legend">')
        for i, r in enumerate(RUNGS):
            w(f'<span><i class="sw {_rung_cls(i)}"></i>{RUNG_LABEL[r]}</span>')
        w('<span class="legend-note">shaded from most to least consistent with '
          'the acquired schema</span>')
        w('</div>')
        w('</section>')

    # ---- per-capacity detail
    w('<section class="band"><h2 class="sec">Dimensions and metrics</h2>')
    w('<p class="secnote">A <em>dimension</em> is the axis a section manipulates. '
      'Material dimensions are properties of what is learned; probe dimensions are '
      'properties of the retrieval cue. Each dimension scores as a weighted mean of '
      'its metrics; each capacity scores as a weighted mean of its available '
      'dimensions.</p>')
    for c in CAP_ORDER:
        p = caps[c]
        w(f'<div class="cap"><div class="caphead"><h3>{e(c)}</h3>'
          f'<span class="capscore">{p["score"]:.3f}</span>'
          f'<span class="capcov">coverage {p["coverage"]:.0%}</span></div>')
        for dim, dw in weights[c].items():
            dv = p["dimensions"][dim]
            label, tone = STATUS_META.get(dv["status"], (dv["status"], "none"))
            drows = [r for r in rows if r["capacity"] == c and r["dimension"] == dim]
            kinds = sorted({r["kind"] for r in drows}) or ["—"]
            w(f'<details class="dim tone-{tone}"'
              f'{" open" if dv["status"] == "scored" else ""}>')
            w('<summary>')
            w(f'<span class="dname">{e(dim)}</span>')
            w(f'<span class="dkind">{e("/".join(kinds))}</span>')
            w(f'<span class="dbadge b-{tone}">{e(label)}</span>')
            sc = dv["score"]
            sc_txt = f"{sc:.3f}" if isinstance(sc, (int, float)) and not math.isnan(sc) else "—"
            w(f'<span class="dscore">{sc_txt}</span>')
            w(f'<span class="dweight">w&nbsp;{dw:.2f}</span>')
            w(f'<span class="dcount">{len(drows)} metric{"" if len(drows)==1 else "s"}</span>')
            w('</summary>')
            if dv["reason"]:
                w(f'<p class="reason">{e(dv["reason"])}</p>')
            if drows:
                w('<div class="tw"><table><thead><tr>'
                  '<th>metric</th><th>section</th><th class="r">raw</th>'
                  '<th class="r">normalised</th><th class="r">v</th><th>role</th>'
                  '<th>what it means</th></tr></thead><tbody>')
                for r in sorted(drows, key=lambda r: (r["role"] != "score", r["key"])):
                    rl, _ = ROLE_META.get(r["role"], (r["role"], ""))
                    slug = r["role"].replace("_", "-")
                    nval = r["n"]
                    nstr = ("—" if nval is None or (isinstance(nval, float) and math.isnan(nval))
                            else f"{nval:.3f}")
                    vstr = str(r["v"]) if r["role"] in ("score", "protocol_limited") else "—"
                    dtag = ' <span class="drv">derived</span>' if r.get("derived") else ""
                    w(f'<tr class="row-{slug}"><td class="mk"><code>{e(r["key"])}</code>{dtag}</td>'
                      f'<td class="sm">{e(r["suite"])} / <code>{e(r["section"])}</code></td>'
                      f'<td class="r num2">{fmt(r["raw"], 4)}</td>'
                      f'<td class="r num2">{nstr}</td><td class="r num2">{vstr}</td>'
                      f'<td><span class="rbadge rb-{slug}">{e(rl)}</span></td>'
                      f'<td class="note">{e(r["note"])}</td></tr>')
                w('</tbody></table></div>')
            else:
                w('<p class="reason empty">No metric is emitted for this dimension.</p>')
            w('</details>')
        w('</div>')
    w('</section>')

    # ---- formula + guards
    w('<section class="band split">')
    w('<div><h2 class="sec">The formula</h2>')
    w('<pre class="formula">n<sub>m</sub>  = normalise(raw<sub>m</sub>)          '
      'in [0,1], 1 = ideal\nD<sub>d</sub>  = &Sigma;<sub>m</sub> v<sub>m</sub>&middot;n'
      '<sub>m</sub> / &Sigma;<sub>m</sub> v<sub>m</sub>       dimension score\n'
      'C    = &Sigma;<sub>d</sub> w<sub>d</sub>&middot;D<sub>d</sub> / &Sigma;'
      '<sub>d</sub> w<sub>d</sub>       over available d\ncov  = &Sigma;'
      '<sub>available</sub> w<sub>d</sub> / &Sigma;<sub>intended</sub> w<sub>d</sub></pre>')
    w('<p class="secnote">Normalisers: identity for higher-better on [0,1]; '
      '<code>1&minus;x</code> for lower-better; chance correction '
      '<code>(x&minus;c)/(1&minus;c)</code>; <code>x/ref</code> against a stated '
      'attainable maximum; <code>1/n</code> for counts-to-criterion; '
      '<code>(budget&minus;n)/(budget&minus;1)</code> for trials against a protocol '
      'budget. Within a dimension <code>v</code> is 2 for a primary readout and 1 '
      'for a secondary one; dimension weights <code>w</code> are set per capacity '
      'and shown beside each dimension above.</p></div>')
    w('<div><h2 class="sec">Validity guards</h2>')
    w('<p class="secnote">Read before anything under them. A failed guard voids its '
      'whole section — the manipulation had no dynamic range, so the numbers beneath '
      'it carry no information about the model.</p>')
    w('<table class="guards"><tbody>')
    for k, v in sorted(d["guards"].items()):
        ok = bool(v) if isinstance(v, bool) else (isinstance(v, (int, float)) and v > 0)
        bad = k in ("schema_at_ceiling", "schema_at_floor")
        good = (ok and not bad) or (bad and not ok)
        if k == "schema_resolved" or k.startswith("iface:"):
            good = ok
        w(f'<tr><td><code>{e(k)}</code></td>'
          f'<td class="r"><span class="gb {"g-ok" if good else "g-bad"}">'
          f'{fmt(v)}</span></td></tr>')
    w('</tbody></table></div></section>')

    w('<footer class="foot-note"><p>Generated by '
      '<code>bin/score_capacities.py</code> and <code>bin/build_scorecard_page.py</code> '
      'from <code>results/&lt;model&gt;/&lt;suite&gt;/metrics.json</code>. '
      'Exclusion rationales cite <code>docs/capacity_coverage_audit.md</code>; the '
      'capacity definitions follow <code>paper/capacities.md</code>.</p></footer>')
    return "\n".join(P)


CSS = """
:root{
  --paper:#F4F6F8; --card:#FFFFFF; --sunk:#EDF0F3;
  --ink:#131A21; --ink2:#3D4954; --mute:#68757F; --faint:#DCE2E8; --hair:#E7ECF0;
  --s1:#2E6296; --s2:#B07A16; --s3:#AD3F6B;
  --ok:#2E6296; --warn:#B07A16; --bad:#AD3F6B; --none:#77838D;
  --s1-soft:rgba(46,98,150,.13); --s2-soft:rgba(176,122,22,.10);
  --shadow:0 1px 2px rgba(19,26,33,.05), 0 8px 24px -14px rgba(19,26,33,.18);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0E1216; --card:#161C22; --sunk:#1B222A;
    --ink:#E7ECF1; --ink2:#BCC7D1; --mute:#8D9AA6; --faint:#2A333C; --hair:#232B34;
    --s1:#5293CE; --s2:#B2851C; --s3:#C95F87;
    --ok:#5293CE; --warn:#B2851C; --bad:#C95F87; --none:#7E8B96;
    --s1-soft:rgba(82,147,206,.16); --s2-soft:rgba(178,133,28,.13);
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -16px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  --paper:#0E1216; --card:#161C22; --sunk:#1B222A;
  --ink:#E7ECF1; --ink2:#BCC7D1; --mute:#8D9AA6; --faint:#2A333C; --hair:#232B34;
  --s1:#5293CE; --s2:#B2851C; --s3:#C95F87;
  --ok:#5293CE; --warn:#B2851C; --bad:#C95F87; --none:#7E8B96;
  --s1-soft:rgba(82,147,206,.16); --s2-soft:rgba(178,133,28,.13);
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -16px rgba(0,0,0,.7);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  font-size:15px; line-height:1.6; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px; margin:0 auto; padding:56px 26px 80px;
  display:flex; flex-direction:column; gap:52px}
code,pre,.num2,.mk code{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
.num2{font-variant-numeric:tabular-nums}
h1,h2,h3{margin:0; text-wrap:balance}

/* masthead */
.mast{display:flex; flex-direction:column; gap:14px;
  border-bottom:1px solid var(--faint); padding-bottom:30px}
.eyebrow{margin:0; font-size:11.5px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--mute); font-weight:600}
.mast h1{font-family:"Newsreader",Georgia,serif; font-weight:400; font-size:clamp(30px,5vw,50px);
  line-height:1.08; letter-spacing:-.012em; overflow-wrap:anywhere}
.lede{margin:0; max-width:66ch; color:var(--ink2); font-size:16.5px}
.runbar{display:flex; flex-wrap:wrap; gap:10px; margin-top:6px}
.run{display:flex; align-items:baseline; gap:9px; padding:7px 13px; border-radius:7px;
  background:var(--card); border:1px solid var(--hair); font-size:12.5px}
.run-k{font-family:"IBM Plex Mono",monospace; font-weight:500; color:var(--ink)}
.run-v{color:var(--mute)}
.run-s{color:var(--ok); font-weight:600; font-size:11px; letter-spacing:.05em;
  text-transform:uppercase}
.run-na .run-s{color:var(--none)}
.run-na{opacity:.72}

.sec{font-family:"Newsreader",Georgia,serif; font-weight:400; font-size:27px;
  letter-spacing:-.008em; margin-bottom:6px}
.band{display:flex; flex-direction:column; gap:18px}
.secnote{margin:0; max-width:72ch; color:var(--ink2); font-size:14.5px}

/* summary cards */
.cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(min(190px,100%),1fr)); gap:14px}
.card{background:var(--card); border:1px solid var(--hair); border-radius:11px;
  padding:17px 17px 19px; display:flex; flex-direction:column; gap:9px; box-shadow:var(--shadow)}
.card h3{font-size:13.5px; font-weight:600; letter-spacing:.005em; line-height:1.3;
  min-height:2.6em; display:flex; align-items:flex-start}
.score{margin:0; display:flex; align-items:baseline; gap:7px}
.score .num{font-family:"Newsreader",Georgia,serif; font-size:42px; line-height:1;
  font-weight:400; font-variant-numeric:tabular-nums; color:var(--s1)}
.score .of{font-size:11.5px; color:var(--mute)}
.meter{height:5px; border-radius:3px; background:var(--sunk); overflow:hidden}
.meter-fill{height:100%; border-radius:3px}
.meter-fill.cov-hi{background:var(--ok)} .meter-fill.cov-mid{background:var(--warn)}
.meter-fill.cov-low{background:var(--bad)}
.covline{margin:0; font-size:11.5px; font-weight:600; font-variant-numeric:tabular-nums}
.cov-hi{color:var(--ok)} .cov-mid{color:var(--warn)} .cov-low{color:var(--bad)}
.blurb{margin:0; font-size:12.5px; color:var(--mute); line-height:1.5; flex:1}
.flag{margin:2px 0 0; font-size:11.5px; color:var(--s2); border-top:1px solid var(--hair);
  padding-top:8px; line-height:1.45}

/* radar */
.radar-wrap{display:grid; grid-template-columns:minmax(0,1.35fr) minmax(220px,1fr);
  gap:26px; align-items:center; background:var(--card); border:1px solid var(--hair);
  border-radius:13px; padding:20px 24px; box-shadow:var(--shadow)}
@media(max-width:760px){.radar-wrap{grid-template-columns:1fr}}
.radar{width:100%; height:auto; overflow:visible}
@media(max-width:520px){.radar{overflow:hidden}
  .radar-wrap{padding:16px 12px}}
.ring{fill:none; stroke:var(--faint); stroke-width:1}
.spoke{stroke:var(--faint); stroke-width:1}
.rtick{fill:var(--mute); font-size:9px; font-family:"IBM Plex Mono",monospace}
.area-incl{fill:var(--s2); fill-opacity:.10}
.line-incl{fill:none; stroke:var(--s2); stroke-width:1.7; stroke-dasharray:6 4}
.area-prim{fill:var(--s1); fill-opacity:.17}
.line-prim{fill:none; stroke:var(--s1); stroke-width:2.6; stroke-linejoin:round}
.line-disc{fill:none; stroke:var(--s3); stroke-width:1.9; stroke-dasharray:2 3.4}
.dot-prim{fill:var(--s1); stroke:var(--card); stroke-width:2}
.dot-disc{fill:var(--s3); stroke:var(--card); stroke-width:1.6}
.hit{fill:transparent; cursor:pointer}
.vtx:focus{outline:none} .vtx:focus .dot-prim{stroke:var(--ink); stroke-width:2.4}
.vlabel{fill:var(--ink); font-size:13px; font-weight:600}
.vscore{fill:var(--s1); font-size:17px; font-weight:700; font-variant-numeric:tabular-nums;
  font-family:"IBM Plex Sans",sans-serif}
.vcov{font-size:10.5px; font-weight:600; font-family:"IBM Plex Mono",monospace}
.vcov.cov-hi{fill:var(--ok)} .vcov.cov-mid{fill:var(--warn)} .vcov.cov-low{fill:var(--bad)}
.radar-key{display:flex; flex-direction:column; gap:13px}
.radar-key .k{margin:0; display:grid; grid-template-columns:auto 1fr; gap:2px 10px;
  align-items:baseline; font-size:13px}
.radar-key .k strong{font-weight:600}
.radar-key .k span{grid-column:2; color:var(--mute); font-size:12px; line-height:1.45}
.radar-key .foot{margin:4px 0 0; font-size:11.5px; color:var(--mute); line-height:1.5;
  border-top:1px solid var(--hair); padding-top:11px}
.sw{width:16px; height:3px; border-radius:2px; display:inline-block; grid-row:1}
.sw-prim{background:var(--s1); height:4px}
.sw-incl{background:linear-gradient(90deg,var(--s2) 60%,transparent 60%); background-size:8px 100%}
.sw-disc{background:linear-gradient(90deg,var(--s3) 45%,transparent 45%); background-size:5px 100%}
.sw-q{background:var(--s1)} .sw-r{background:var(--s3)}

/* charts */
.charts{display:grid; grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr)); gap:16px}
.chart-wide{grid-column:1/-1}
.chart{margin:0; background:var(--card); border:1px solid var(--hair); border-radius:11px;
  padding:16px 18px 12px; display:flex; flex-direction:column; gap:10px; box-shadow:var(--shadow)}
.chart figcaption{display:flex; flex-direction:column; gap:4px}
.chart figcaption strong{font-size:13.5px; font-weight:600; line-height:1.35}
.chart figcaption span{font-size:12px; color:var(--mute); line-height:1.5}
.spark{width:100%; height:auto}
.gl{stroke:var(--faint); stroke-width:1}
.atick{fill:var(--mute); font-size:9.5px; font-family:"IBM Plex Mono",monospace}
.axlabel{fill:var(--mute); font-size:10.5px}
.sline{fill:none; stroke:var(--s1); stroke-width:2; stroke-linejoin:round}
.sdot{fill:var(--s1); stroke:var(--card); stroke-width:1.6}
.mark{stroke:var(--s3); stroke-width:1.4; stroke-dasharray:3 3}
.marklab{fill:var(--s3); font-size:9.5px; font-weight:600}
.bar-q{fill:var(--s1)} .bar-r{fill:var(--s3)}
/* Sequential ramp for the consistency ladder: an ORDERED variable, so one hue
   light-to-dark rather than four categorical hues. Lightness monotonic, every
   step >= 3:1 on its surface. */
:root{--r0:#0F3050; --r1:#245C87; --r2:#3F7FAC; --r3:#6197C0}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){--r0:#BFD8EE; --r1:#8FB6D9; --r2:#5E93C4; --r3:#3D74A8}}
:root[data-theme="dark"]{--r0:#BFD8EE; --r1:#8FB6D9; --r2:#5E93C4; --r3:#3D74A8}
.rline{fill:none; stroke-width:2.2; stroke-linejoin:round}
.seqline{fill:none; stroke-width:1.3; stroke-dasharray:2.5 3; opacity:.34}
.ladderline{fill:none; stroke:var(--faint); stroke-width:1.6}
.rline.r0,.seqline.r0,.rbar.r0{stroke:var(--r0); fill:none} .rbar.r0{fill:var(--r0)}
.rline.r1,.seqline.r1{stroke:var(--r1)} .rbar.r1{fill:var(--r1)}
.rline.r2,.seqline.r2{stroke:var(--r2)} .rbar.r2{fill:var(--r2)}
.rline.r3,.seqline.r3{stroke:var(--r3)} .rbar.r3{fill:var(--r3)}
.rdot{stroke:var(--card); stroke-width:1.3}
.rdot.big{stroke-width:2}
.rdot.r0{fill:var(--r0)} .rdot.r1{fill:var(--r1)}
.rdot.r2{fill:var(--r2)} .rdot.r3{fill:var(--r3)}
.rlab{font-size:12px; font-weight:600}
.rlab.r0{fill:var(--r0)} .rlab.r1{fill:var(--r1)}
.rlab.r2{fill:var(--r2)} .rlab.r3{fill:var(--r3)}
.rsub{fill:var(--mute); font-size:10px; font-family:"IBM Plex Mono",monospace}
.pointlab{fill:var(--ink2); font-size:10.5px; font-weight:600}
.floorline{stroke:var(--s3); stroke-width:1.5; stroke-dasharray:5 3}
.floorlab{fill:var(--s3); font-size:10px; font-weight:600}
.critline{stroke:var(--mute); stroke-width:1; stroke-dasharray:2 3}
.critlab{fill:var(--mute); font-size:10px}
.sw.r0{background:var(--r0)} .sw.r1{background:var(--r1)}
.sw.r2{background:var(--r2)} .sw.r3{background:var(--r3)}
/* retention-matrix heatmap: sequential single-hue ramp, light -> dark */
:root{--h0:#EAF1F7; --h1:#C3D8E9; --h2:#8FB4D4; --h3:#5388B4; --h4:#22537E;
      --band:#EFF2F5; --quad:rgba(46,98,150,.09)}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){--h0:#1B242D; --h1:#263F55; --h2:#33597C;
      --h3:#4479A6; --h4:#6BA3D0; --band:#1B222A; --quad:rgba(82,147,206,.13)}}
:root[data-theme="dark"]{--h0:#1B242D; --h1:#263F55; --h2:#33597C; --h3:#4479A6;
      --h4:#6BA3D0; --band:#1B222A; --quad:rgba(82,147,206,.13)}
.hcell.h0{fill:var(--h0)} .hcell.h1{fill:var(--h1)} .hcell.h2{fill:var(--h2)}
.hcell.h3{fill:var(--h3)} .hcell.h4{fill:var(--h4)}
.hcell.diag{stroke:var(--ink); stroke-width:2}
.hcell-na{fill:none; stroke:var(--hair); stroke-width:1}
.hval{font-size:11px; font-variant-numeric:tabular-nums;
      font-family:"IBM Plex Mono",monospace}
.hval.on-dark{fill:#FFFFFF} .hval.on-light{fill:var(--ink)}
:root[data-theme="dark"] .hval.on-dark, :root:not([data-theme="light"]) .hval.on-dark{fill:#0E1216}
.mlab{fill:var(--mute); font-size:10.5px}
/* AB/AC */
.bar-ctl{fill:var(--s1)} .bar-abac{fill:var(--s3)}
.barval{fill:var(--ink); font-size:15px; font-weight:700;
        font-variant-numeric:tabular-nums}
/* reversal */
.stageband{fill:var(--band)}
.stagedivide{stroke:var(--faint); stroke-width:1}
.stagelab{fill:var(--mute); font-size:10.5px; letter-spacing:.07em;
          text-transform:uppercase; font-weight:600}
.revline{fill:none; stroke-width:2.3; stroke-linejoin:round}
.revline.s-acc{stroke:var(--s1)} .revline.s-per{stroke:var(--s3); stroke-dasharray:7 6}
.revdot{stroke:var(--card); stroke-width:1.2}
.revdot.s-acc{fill:var(--s1)} .revdot.s-per{fill:var(--s3)}
.revlab{font-size:12px; font-weight:600}
.revlab.s-acc{fill:var(--s1)} .revlab.s-per{fill:var(--s3)}
/* one-shot scale probe */
.revline.p-mrr{stroke:var(--s1)}
.revline.p-cos{stroke:var(--s2); stroke-dasharray:6 3}
.revline.p-mag{stroke:var(--s3); stroke-dasharray:2 3}
.revdot.p-mrr{fill:var(--s1)} .revdot.p-cos{fill:var(--s2)} .revdot.p-mag{fill:var(--s3)}
.revlab.p-mrr{fill:var(--s1)} .revlab.p-cos{fill:var(--s2)} .revlab.p-mag{fill:var(--s3)}
.sw-acc{background:var(--s1)}
.sw-per{background:linear-gradient(90deg,var(--s3) 60%,transparent 60%); background-size:9px 100%}
/* rollout cascade: cued reference + three feedback modes */
.revline.c-cued{stroke:var(--ink2); stroke-width:1.8; stroke-dasharray:1 3}
.revline.c-quant{stroke:var(--s1)}
.revline.c-l2{stroke:var(--s2); stroke-dasharray:7 4}
.revline.c-raw{stroke:var(--s3); stroke-dasharray:3 3}
.revdot.c-cued{fill:var(--ink2)} .revdot.c-quant{fill:var(--s1)}
.revdot.c-l2{fill:var(--s2)} .revdot.c-raw{fill:var(--s3)}
.revlab.c-cued{fill:var(--ink2)} .revlab.c-quant{fill:var(--s1)}
.revlab.c-l2{fill:var(--s2)} .revlab.c-raw{fill:var(--s3)}
.c-cued-sw{background:linear-gradient(90deg,var(--ink2) 30%,transparent 30%); background-size:4px 100%}
.c-quant-sw{background:var(--s1)}
.c-raw-sw{background:linear-gradient(90deg,var(--s3) 50%,transparent 50%); background-size:6px 100%}
/* exposure ramp: an ORDERED variable, so one hue light-to-dark. Monotonic in
   lightness, every step >= 3:1 on its surface, and every line carries a direct
   label so identity never rests on colour alone. */
:root{--e0:#07223A; --e1:#124366; --e2:#1E6390; --e3:#3080B0; --e4:#4F94BE;
      --failband:rgba(173,63,107,.07)}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){--e0:#DCEBF7; --e1:#B4D2EB; --e2:#8AB6DC;
      --e3:#6295C6; --e4:#3E76AB; --failband:rgba(201,95,135,.11)}}
:root[data-theme="dark"]{--e0:#DCEBF7; --e1:#B4D2EB; --e2:#8AB6DC; --e3:#6295C6;
      --e4:#3E76AB; --failband:rgba(201,95,135,.11)}
.revline.e0{stroke:var(--e0)} .revline.e1{stroke:var(--e1)}
.revline.e2{stroke:var(--e2)} .revline.e3{stroke:var(--e3)}
.revline.e4{stroke:var(--e4)}
.revdot.e0{fill:var(--e0)} .revdot.e1{fill:var(--e1)} .revdot.e2{fill:var(--e2)}
.revdot.e3{fill:var(--e3)} .revdot.e4{fill:var(--e4)}
.revlab.e0{fill:var(--e0)} .revlab.e1{fill:var(--e1)} .revlab.e2{fill:var(--e2)}
.revlab.e3{fill:var(--e3)} .revlab.e4{fill:var(--e4)}
.e0-sw{background:var(--e0)} .e1-sw{background:var(--e1)} .e2-sw{background:var(--e2)}
.e3-sw{background:var(--e3)} .e4-sw{background:var(--e4)}
.zeroline{stroke:var(--ink2); stroke-width:1.4}
.gapband{fill:var(--s3); fill-opacity:.13}
.chanceband{fill:var(--mute); fill-opacity:.07}
.chanceline{fill:none; stroke:var(--mute); stroke-width:1.3; stroke-dasharray:4 3}
.revline.s-orth{stroke:var(--s2); stroke-dasharray:6 3}
.revdot.s-orth{fill:var(--s2)}
.revlab.s-orth{fill:var(--s2)} .revlab.chance{fill:var(--mute)}
/* cue-masking modes: three distinct manipulations, so categorical */
.revline.m-rand{stroke:var(--s1)} .revline.m-shar{stroke:var(--s2)}
.revline.m-iden{stroke:var(--s3)}
.revdot.m-rand{fill:var(--s1)} .revdot.m-shar{fill:var(--s2)}
.revdot.m-iden{fill:var(--s3)}
.revlab.m-rand{fill:var(--s1)} .revlab.m-shar{fill:var(--s2)}
.revlab.m-iden{fill:var(--s3)}
.identline{fill:none; stroke-width:1.4; stroke-dasharray:2 3; opacity:.45}
.identline.m-rand{stroke:var(--s1)} .identline.m-shar{stroke:var(--s2)}
.identline.m-iden{stroke:var(--s3)}
.barval.sm{font-size:11.5px; font-weight:600}
.wbar.wb-ok{fill:var(--s1)} .wbar.wb-guard{fill:var(--s2)} .wbar.wb-na{fill:var(--faint)}
.wlab{fill:var(--ink2); font-size:12px; font-family:"IBM Plex Mono",monospace}
.wnum{fill:#FFFFFF; font-size:11.5px; font-weight:700; font-variant-numeric:tabular-nums}
.wnum.on-pale{fill:var(--ink2)}
.wval.wb-na{fill:var(--mute)}
.wval{font-size:12.5px; font-weight:600; font-variant-numeric:tabular-nums}
.wval.wb-ok{fill:var(--s1)} .wval.wb-guard{fill:var(--s2)}
.sw.wb-ok{background:var(--s1)} .sw.wb-guard{background:var(--s2)}
.sw.wb-na{background:var(--faint)}
.dumbbell{stroke:var(--faint); stroke-width:6; stroke-linecap:round}
.ratiolab{fill:var(--s3); font-size:11.5px; font-weight:700;
          font-variant-numeric:tabular-nums}
.sw-ctl{background:var(--s1)}
.sw-abac{background:var(--s3)}
.statgrid{display:grid; grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr));
          gap:10px}
.statcell{background:var(--sunk); border-radius:8px; padding:11px 13px;
          display:flex; flex-direction:column; gap:3px}
.statcell.undef{border-left:3px solid var(--s3)}
.statk{margin:0; font-size:10.5px; font-family:"IBM Plex Mono",monospace; color:var(--mute)}
.statv{margin:0; font-size:20px; font-weight:700; font-variant-numeric:tabular-nums;
       color:var(--s1); line-height:1.1}
.statcell.undef .statv{color:var(--s3); font-size:16px; font-weight:600}
.statn{margin:0; font-size:11px; color:var(--mute); line-height:1.4}
.ctxline{fill:none; stroke:var(--faint); stroke-width:1.8}
.revline.m-lo{stroke:var(--s3)} .revline.m-hi{stroke:var(--s1)}
.revdot.m-lo{fill:var(--s3)} .revdot.m-hi{fill:var(--s1)}
.revlab.m-lo{fill:var(--s3)} .revlab.m-hi{fill:var(--s1)}
/* Heatmap ramps. Sequential = one hue light-to-dark (magnitude); diverging =
   two hues about a neutral midpoint, used ONLY where zero is a real boundary.
   Binned into classes so the fills follow the viewer's theme; every cell also
   carries its number, so binning never hides a value. */
:root{
  --sq0:#EDF3F8; --sq1:#CBDDEC; --sq2:#A0C0DC; --sq3:#6E9CC6; --sq4:#3F76AC; --sq5:#1B4C76;
  --dv0:#A8385F; --dv1:#C77A96; --dv2:#E3BECC; --dv3:#F0F0EE; --dv4:#B7CBDD;
  --dv5:#6F9BC0; --dv6:#2E6296;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --sq0:#1A222B; --sq1:#23384C; --sq2:#2F5375; --sq3:#3C6889; --sq4:#5F94BC; --sq5:#8AB6DA;
    --dv0:#C95F87; --dv1:#A34A6B; --dv2:#6E3549; --dv3:#2A2E33; --dv4:#2F4E6E;
    --dv5:#39699B; --dv6:#6BA3D0;
  }}
:root[data-theme="dark"]{
  --sq0:#1A222B; --sq1:#23384C; --sq2:#2F5375; --sq3:#3C6889; --sq4:#5F94BC; --sq5:#8AB6DA;
  --dv0:#C95F87; --dv1:#A34A6B; --dv2:#6E3549; --dv3:#2A2E33; --dv4:#2F4E6E;
  --dv5:#39699B; --dv6:#6BA3D0;
}
.hcell.sq0{fill:var(--sq0)} .hcell.sq1{fill:var(--sq1)} .hcell.sq2{fill:var(--sq2)}
.hcell.sq3{fill:var(--sq3)} .hcell.sq4{fill:var(--sq4)} .hcell.sq5{fill:var(--sq5)}
.hcell.dv0{fill:var(--dv0)} .hcell.dv1{fill:var(--dv1)} .hcell.dv2{fill:var(--dv2)}
.hcell.dv3{fill:var(--dv3)} .hcell.dv4{fill:var(--dv4)} .hcell.dv5{fill:var(--dv5)}
.hcell.dv6{fill:var(--dv6)}
.hval2{font-size:9.5px; font-variant-numeric:tabular-nums;
       font-family:"IBM Plex Mono",monospace}
.hval2.on-dark{fill:#FFFFFF} .hval2.on-light{fill:var(--ink)}
:root[data-theme="dark"] .hval2.on-dark, :root:not([data-theme="light"]) .hval2.on-dark{fill:#0E1216}
.hm{max-width:100%}
.hmpair{display:grid; grid-template-columns:repeat(auto-fit,minmax(min(400px,100%),1fr));
        gap:14px}
.hmtitle{margin:0 0 2px; font-size:10.5px; letter-spacing:.07em; text-transform:uppercase;
         color:var(--mute); font-weight:600}
.hmscale{display:flex; align-items:center; gap:3px; font-size:11px; color:var(--mute);
         flex-wrap:wrap}
.hmscale .hend{font-family:"IBM Plex Mono",monospace; font-size:10px}
.hsw{width:20px; height:11px; display:inline-block; border-radius:2px}
.hsw.sq0{background:var(--sq0)} .hsw.sq1{background:var(--sq1)} .hsw.sq2{background:var(--sq2)}
.hsw.sq3{background:var(--sq3)} .hsw.sq4{background:var(--sq4)} .hsw.sq5{background:var(--sq5)}
.hsw.dv0{background:var(--dv0)} .hsw.dv1{background:var(--dv1)} .hsw.dv2{background:var(--dv2)}
.hsw.dv3{background:var(--dv3)} .hsw.dv4{background:var(--dv4)} .hsw.dv5{background:var(--dv5)}
.hsw.dv6{background:var(--dv6)}
.hmscale .legend-note{margin-left:6px}
.failband{fill:var(--failband)}
/* joint plane */
.goodquad{fill:var(--quad)}
.quadlab{fill:var(--mute); font-size:11px; font-style:italic}
.quadlab.good{fill:var(--s1); font-style:normal; font-weight:600}
.jointdot{fill:var(--s1); stroke:var(--card); stroke-width:2.5}
.rung-legend{gap:18px; padding-top:2px}
.rung-legend .legend-note{color:var(--mute); font-style:italic}
.legend{display:flex; flex-wrap:wrap; gap:16px; font-size:11.5px; color:var(--mute)}
.legend span{display:flex; align-items:center; gap:6px}

/* capacity detail */
.cap{background:var(--card); border:1px solid var(--hair); border-radius:12px;
  overflow:hidden; box-shadow:var(--shadow)}
.caphead{display:flex; align-items:baseline; gap:12px; padding:15px 18px;
  border-bottom:1px solid var(--hair); background:var(--sunk); flex-wrap:wrap}
.caphead h3{font-size:16px; font-weight:600; flex:1; min-width:150px}
.capscore{font-family:"Newsreader",Georgia,serif; font-size:23px; color:var(--s1);
  font-variant-numeric:tabular-nums}
.capcov{font-size:11.5px; color:var(--mute); font-family:"IBM Plex Mono",monospace}
.dim{border-bottom:1px solid var(--hair)}
.dim:last-child{border-bottom:none}
.dim summary{display:flex; align-items:center; gap:11px; padding:11px 18px; cursor:pointer;
  flex-wrap:wrap; list-style:none}
.dim summary::-webkit-details-marker{display:none}
.dim summary::before{content:"›"; color:var(--mute); font-size:16px; line-height:1;
  transition:transform .15s ease; display:inline-block}
@media(prefers-reduced-motion:reduce){.dim summary::before{transition:none}}
.dim[open] summary::before{transform:rotate(90deg)}
.dim summary:hover{background:var(--sunk)}
.dim summary:focus-visible{outline:2px solid var(--s1); outline-offset:-2px}
.dname{font-family:"IBM Plex Mono",monospace; font-size:13px; font-weight:500; flex:1;
  min-width:130px}
.dkind{font-size:10.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--mute)}
.dbadge{font-size:10.5px; font-weight:600; padding:2px 8px; border-radius:20px;
  letter-spacing:.02em; white-space:nowrap}
.b-ok{background:var(--s1-soft); color:var(--ok)}
.b-bad{background:rgba(173,63,107,.13); color:var(--bad)}
.b-none{background:var(--sunk); color:var(--none)}
.dscore{font-family:"IBM Plex Mono",monospace; font-size:13.5px; font-weight:600;
  font-variant-numeric:tabular-nums; min-width:44px; text-align:right}
.dweight,.dcount{font-size:11px; color:var(--mute); font-family:"IBM Plex Mono",monospace}
.reason{margin:0 18px 14px; padding:11px 14px; background:var(--sunk); border-radius:8px;
  border-left:3px solid var(--bad); font-size:12.5px; color:var(--ink2); line-height:1.55;
  max-width:82ch}
.reason.empty{border-left-color:var(--none)}
.tw{overflow-x:auto; margin:0 0 14px; padding:0 18px}
table{border-collapse:collapse; width:100%; font-size:12.5px; min-width:840px}
thead th{text-align:left; font-size:10.5px; letter-spacing:.07em; text-transform:uppercase;
  color:var(--mute); font-weight:600; padding:6px 9px; border-bottom:1px solid var(--faint);
  white-space:nowrap}
tbody td{padding:8px 9px; border-bottom:1px solid var(--hair); vertical-align:top}
tbody tr:last-child td{border-bottom:none}
th.r,td.r{text-align:right}
.mk code{font-size:11.5px; color:var(--ink)}
.sm{color:var(--mute); font-size:11.5px; white-space:nowrap}
.sm code{font-size:11px}
.note{color:var(--ink2); line-height:1.5; min-width:290px}
.drv{font-size:9.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--s2);
  font-weight:600; margin-left:5px}
.rbadge{font-size:10px; font-weight:600; padding:2px 7px; border-radius:20px; white-space:nowrap}
.rb-score{background:var(--s1-soft); color:var(--ok)}
.rb-protocol-limited{background:var(--s2-soft); color:var(--warn)}
.rb-control,.rb-guard,.rb-config,.rb-diagnostic{background:var(--sunk); color:var(--none)}
tr.row-score .mk code{font-weight:600}
tr:not(.row-score):not(.row-protocol-limited){opacity:.78}

/* formula + guards */
.split{display:grid; grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr)); gap:30px;
  align-items:start}
.split>div{display:flex; flex-direction:column; gap:12px}
.formula{margin:0; background:var(--card); border:1px solid var(--hair); border-radius:10px;
  padding:16px 18px; font-size:12.5px; line-height:1.9; overflow-x:auto; color:var(--ink2)}
.guards{font-size:12px; min-width:0; width:100%; table-layout:fixed}
.guards td{padding:5px 8px}
.guards td:first-child{overflow-wrap:anywhere}
.guards td.r{width:78px}
.guards code{font-size:11.5px}
.gb{font-family:"IBM Plex Mono",monospace; font-weight:600; padding:1px 8px;
  border-radius:20px; font-size:11px}
.g-ok{background:var(--s1-soft); color:var(--ok)}
.g-bad{background:rgba(173,63,107,.13); color:var(--bad)}
.foot-note{border-top:1px solid var(--faint); padding-top:20px; color:var(--mute); font-size:12px}
.foot-note p{margin:0; max-width:80ch}

#tip{position:fixed; pointer-events:none; opacity:0; transform:translate(-50%,-118%);
  background:var(--ink); color:var(--paper); padding:7px 11px; border-radius:7px;
  font-size:11.5px; line-height:1.5; white-space:nowrap; z-index:50;
  box-shadow:0 6px 18px rgba(0,0,0,.28); transition:opacity .1s ease}
@media(prefers-reduced-motion:reduce){#tip{transition:none}}
#tip b{font-variant-numeric:tabular-nums}
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(html,x,y){tip.innerHTML=html;tip.style.left=x+'px';tip.style.top=y+'px';
    tip.style.opacity='1';}
  function hide(){tip.style.opacity='0';}
  document.querySelectorAll('.hit').forEach(function(h){
    function render(ev){
      var r=h.getBoundingClientRect(), x=r.left+r.width/2, y=r.top;
      if(h.dataset.cap){
        show('<b>'+h.dataset.cap+'</b><br>score <b>'+h.dataset.score+
             '</b> &middot; coverage <b>'+h.dataset.cov+'</b><br>inclusive '+
             h.dataset.incl+' &middot; discounted '+h.dataset.disc, x, y);
      }else{
        show(h.dataset.xl+' <b>'+h.dataset.x+'</b> &middot; '+h.dataset.yl+
             ' <b>'+h.dataset.y+'</b>', x, y);
      }
    }
    h.addEventListener('mouseenter',render);
    h.addEventListener('mouseleave',hide);
    var g=h.closest('.vtx');
    if(g){g.addEventListener('focus',render);g.addEventListener('blur',hide);}
  });
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecard", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.scorecard) as f:
        d = json.load(f)

    series, schema_metrics = {}, {}
    for suite in ("spatial", "symbolic"):
        p = os.path.join(d["results_dir"], suite, "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                payload = json.load(f)
            series.update(payload.get("series", {}))
            schema_metrics.update({k: v for k, v in payload.get("metrics", {}).items()
                                   if k.startswith("schema")})
    # Damage read-outs come from the focused run, exactly as the scorecard does.
    ov = d.get("schema_focused_overlay") or {}
    if ov.get("applied") and ov.get("source"):
        fp = os.path.join(os.path.abspath(ov["source"]), "symbolic", "metrics.json")
        if os.path.exists(fp):
            with open(fp) as f:
                fpayload = json.load(f)
            for k, v in fpayload.get("metrics", {}).items():
                if any(t in k for t in ov.get("keys", [])):
                    schema_metrics[k] = v
    d["_schema_metrics"] = schema_metrics
    all_metrics = {}
    for suite in ("spatial", "symbolic"):
        mp = os.path.join(d["results_dir"], suite, "metrics.json")
        if os.path.exists(mp):
            with open(mp) as f:
                all_metrics.update(json.load(f).get("metrics", {}))
    d["_all_metrics"] = all_metrics
    for fname, key in (("oneshot_scale_probe.json", "_scale_probe"),
                       ("rollout_cascade_probe.json", "_cascade_probe"),
                       ("serial_order_probe.json", "_serial_order_probe")):
        pp = os.path.join(d["results_dir"], fname)
        if os.path.exists(pp):
            with open(pp) as f:
                d[key] = json.load(f)

    body = build(d, series)
    page = (
        '<title>AHN Capacity Scorecard</title>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500;600&'
        'family=IBM+Plex+Sans:wght@400;500;600;700&'
        'family=Newsreader:opsz,wght@6..72,300;6..72,400;6..72,500&display=swap">\n'
        f'<style>{CSS}</style>\n'
        f'<div class="wrap">\n{body}\n</div>\n'
        '<div id="tip" role="status" aria-live="polite"></div>\n'
        f'<script>{JS}</script>\n'
    )
    with open(a.out, "w") as f:
        f.write(page)
    print(f"wrote {a.out} ({len(page)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
