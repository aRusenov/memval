"""Generate the MemVal overview figure (docs/figures/memval_overview.svg).

The figure lays the framework out as one swimlane per **core capacity**. Each
lane reads left to right: the capacity, the question its suite asks, the
sequence dimensions manipulated to ask it, the metrics collected, and the shape
of the object those metrics produce. Underneath sit the diagnostics band and
three cross-cutting rails (regime, modality, oracle level) that apply to every
lane at once.

Chip styling encodes implementation status against the current tree:
solid = implemented in ``memval/``; dashed = proposed, not yet implemented;
amber + ✳ = defined only in the online (streaming) regime.

Run:  python docs/figures/overview_figure.py
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------
INK = "#1a1d21"
MUTED = "#697586"
FAINT = "#98a2b3"
RULE = "#dde1e7"
BG = "#ffffff"
LANE_ALT = "#f7f8fa"

IMPL_FILL, IMPL_STROKE, IMPL_TEXT = "#eaeff6", "#bcc8d8", "#26303f"
PROP_FILL, PROP_STROKE, PROP_TEXT = "#ffffff", "#c2c8d2", "#6b7280"
ONLINE_FILL, ONLINE_STROKE, ONLINE_TEXT = "#fdf4e5", "#e3c68d", "#6d5320"

CAP_FILL, CAP_STROKE = "#e3ebf5", "#93aac6"
CAP_PROP_FILL, CAP_PROP_STROKE = "#ffffff", "#b6bfcc"

DIAG_FILL, DIAG_STROKE, DIAG_TEXT = "#f3f1f7", "#cfc8dd", "#453a5c"
MODEL_FILL, MODEL_STROKE = "#f5f7f9", "#ccd4de"

OK_FILL, OK_STROKE, OK_TEXT = "#e7f2ec", "#93bfa6", "#1f4e37"
GAP_FILL, GAP_STROKE, GAP_TEXT = "#fdeceb", "#dda6a3", "#7a2f2a"

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "SF Mono, Menlo, Consolas, monospace"

# --------------------------------------------------------------------------
# text metrics (rough advance-width model, good enough for layout)
# --------------------------------------------------------------------------
_WIDE = set("mwMWO@%")
_NARROW = set("iljtfrI.,:;'|!()[] ")


def tw(s: str, size: float, bold: bool = False) -> float:
    w = 0.0
    for ch in s:
        if ch in _WIDE:
            w += 0.85
        elif ch in _NARROW:
            w += 0.35
        elif ch.isupper() or ch.isdigit():
            w += 0.62
        else:
            w += 0.545
    return w * size * (1.045 if bold else 1.0)


def wrap(text: str, size: float, max_w: float, bold: bool = False) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if tw(trial, size, bold) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------
# status codes: "impl" | "prop" | "online"
LANES = [
    dict(
        cap="Continual retention", cap_status="impl", new=False,
        q="What survives when new episodes overwrite older ones?",
        dims=[("Number of sequences", "impl"),
              ("Between-list similarity", "impl"),
              ("Shared subsequence", "impl"),
              ("Retention interval", "online"),
              ("Streaming order", "online")],
        metrics=[("Retention matrix R[j,i]", "impl"),
                 ("ACC", "impl"), ("BWT", "impl"),
                 ("Avg forgetting", "impl"),
                 ("Δ-MRR before / after", "impl")],
        out="Lower-triangular matrix, read down columns; three summary scalars",
    ),
    dict(
        cap="Pattern completion", cap_status="impl", new=False,
        q="Can a partial or degraded cue regenerate the whole episode?",
        dims=[("Cue fraction (prompt_fraction)", "impl"),
              ("Cue noise σ", "impl"),
              ("Sequence length", "impl")],
        metrics=[("MRR", "impl"), ("Recall fidelity", "impl"),
                 ("Memory span", "impl"),
                 ("Noise-tolerance threshold", "impl")],
        out="Completion curve vs. cue fraction and vs. σ",
    ),
    dict(
        cap="Sequence disambiguation", cap_status="impl", new=False,
        q="Are episodes that share a subsequence kept apart?",
        dims=[("Shared subsequence / overlap", "impl"),
              ("Semantic similarity", "impl"),
              ("High-order Markov", "prop"),
              ("Context-gated (XOR) transitions", "prop")],
        metrics=[("MRR at branch point", "impl"),
                 ("Crosstalk matrix", "impl"),
                 ("Diagonal dominance", "impl"),
                 ("Channel MI", "impl"),
                 ("Cross-episode intrusions", "prop"),
                 ("Oracle gap Δ", "prop")],
        out="Branch-point accuracy; confusion structure",
    ),
    dict(
        cap="One-shot learning", cap_status="impl", new=False,
        q="How much of an episode is acquired from a single exposure?",
        dims=[("Number of exposures", "impl"),
              ("Presentation duration", "impl"),
              ("Novelty / surprise gating", "online")],
        metrics=[("Convergence epochs", "impl"),
                 ("MRR after one pass", "impl"),
                 ("Span after one pass", "impl")],
        out="Acquisition curve vs. number of exposures",
    ),
    dict(
        cap="Autonomous generation", cap_status="impl", new=True,
        q="Can the dynamics be iterated once decoupled from input?",
        dims=[("Sequence length", "impl"),
              ("Cyclic / limit-cycle", "prop"),
              ("Delayed dependency (n-back)", "prop")],
        metrics=[("Memory span", "impl"), ("Attractor scan", "impl"),
                 ("Trajectory drift", "prop"), ("Error cascade", "prop"),
                 ("Conditional per-step accuracy", "prop"),
                 ("Serial position & transpositions", "prop")],
        out="Span vs. length; drift trace; position curve",
    ),
    dict(
        cap="Schema abstraction", cap_status="prop", new=True,
        q="Does a learned structure transfer to a novel instance?",
        dims=[("Schema consistency / grammar entropy", "prop"),
              ("Vocabulary size", "prop"),
              ("Transitive-inference depth", "prop")],
        metrics=[("Held-out instance accuracy", "prop"),
                 ("Forward transfer / savings", "prop"),
                 ("Linear-baseline gap", "prop")],
        out="Transfer curve; gap against the linear control",
    ),
    dict(
        cap="Temporal / interval memory", cap_status="prop", new=True,
        q="Is the interval itself encoded, not just what comes next?",
        dims=[("Inter-item interval", "online"),
              ("Rhythm / jitter", "online"),
              ("Presentation duration (dwell)", "online")],
        metrics=[("Interval reproduction error", "prop"),
                 ("Tempo invariance", "prop")],
        out="Timing curve; invariance under tempo rescaling",
    ),
]

MODEL_ITEMS = [
    ("Adapter API", ["fit_sequence", "fit_event", "recall", "on_event_boundary"]),
    ("Model zoo", ["Asymmetric Hopfield", "EP family (DG / XdG / EWC / CLS)", "GPT-2", "kNN episodic", "HICL"]),
    ("Controls", ["Linear next-item predictor", "Chance / ceiling"]),
]

DIAGNOSTICS = ["Crosstalk matrix", "Diagonal dominance", "Channel MI",
               "Category vs. item information", "Synaptic interference matrix",
               "Attractor scan", "Representational geometry / RSA", "Weight diagnostics"]

RAILS = [
    dict(label="REGIME",
         note="Gates which dimensions are definable at all. Chips marked ✳ exist only here.",
         chips=[("Offline — repeated batch epochs", "neutral"),
                ("Online — single-pass streaming ✳", "neutral")]),
    dict(label="MODALITY",
         note="Every dimension is manipulated within a modality; modality is orthogonal to all of it.",
         chips=[("Symbolic", "neutral"), ("Spatial", "neutral"),
                ("Temporal", "neutral"), ("Relational / audio", "neutral")]),
    dict(label="ORACLE LEVEL",
         note="How much of the problem the harness solves outside the model. Four instances of "
              "one design decision — two are declared and reported per run, two are not.",
         chips=[("Ground-truth item feedback — cued vs. free-running", "ok"),
                ("Clean codebook — raw / L2 / quantized", "ok"),
                ("Episode identity — context: veridical / degraded / in-band / absent / conflicting", "gap"),
                ("Episode segmentation — explicit reset / marker / inferable / none", "gap")]),
]

# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
W = 1760
MARGIN = 28
COLS = [
    ("CAPACITY", MARGIN, 208),
    ("THE QUESTION ITS SUITE ASKS", MARGIN + 216, 268),
    ("SEQUENCE DIMENSION MANIPULATED", MARGIN + 492, 372),
    ("METRIC COLLECTED", MARGIN + 872, 452),
    ("READ AS", MARGIN + 1332, W - MARGIN - (MARGIN + 1332)),
]

CHIP_H, CHIP_GAP, ROW_GAP, CHIP_PAD = 24.0, 7.0, 7.0, 10.0
CELL_PAD_X, CELL_PAD_Y = 13.0, 15.0
CHIP_FS, BODY_FS = 11.5, 12.5

out: list[str] = []


def rect(x, y, w, h, fill, stroke, rx=4, dash=None, sw=1.0, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' fill-opacity="{opacity}"' if opacity is not None else ""
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
               f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')


def text(x, y, s, size, fill=INK, weight="400", anchor="start", family=FONT, ls=None, italic=False):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    l = f' letter-spacing="{ls}"' if ls else ""
    i = ' font-style="italic"' if italic else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
               f'font-weight="{weight}" fill="{fill}"{a}{l}{i}>{s}</text>')


def chip_style(status):
    if status == "impl":
        return IMPL_FILL, IMPL_STROKE, IMPL_TEXT, None
    if status == "online":
        return ONLINE_FILL, ONLINE_STROKE, ONLINE_TEXT, None
    if status == "ok":
        return OK_FILL, OK_STROKE, OK_TEXT, None
    if status == "gap":
        return GAP_FILL, GAP_STROKE, GAP_TEXT, None
    if status == "neutral":
        return "#eef1f5", "#c6cdd8", "#2b3440", None
    return PROP_FILL, PROP_STROKE, PROP_TEXT, "3 2.5"


def layout_chips(items, max_w):
    """Pack (label, status) chips into rows. Returns (rows, total_height)."""
    rows, cur, cur_w = [], [], 0.0
    for label, status in items:
        label_txt = label + (" ✳" if status == "online" else "")
        cw = tw(label_txt, CHIP_FS) + 2 * CHIP_PAD
        if cur and cur_w + CHIP_GAP + cw > max_w:
            rows.append(cur)
            cur, cur_w = [], 0.0
        cur.append((label_txt, status, cw))
        cur_w += cw + (CHIP_GAP if len(cur) > 1 else 0)
    if cur:
        rows.append(cur)
    h = len(rows) * CHIP_H + max(0, len(rows) - 1) * ROW_GAP
    return rows, h


def draw_chips(rows, x, y):
    cy = y
    for row in rows:
        cx = x
        for label, status, cw in row:
            fill, stroke, txt, dash = chip_style(status)
            rect(cx, cy, cw, CHIP_H, fill, stroke, rx=CHIP_H / 2, dash=dash)
            text(cx + CHIP_PAD, cy + CHIP_H / 2 + 4.1, esc(label), CHIP_FS, txt)
            cx += cw + CHIP_GAP
        cy += CHIP_H + ROW_GAP


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
y = 46
text(MARGIN, y, "MemVal — building blocks", 27, INK, "600")
y += 24
text(MARGIN, y, "One lane per core capacity: what is claimed, the question that tests it, the knob turned to ask, "
                "what is measured, and the shape of the answer.", 13.5, MUTED)

# legend (right-aligned strip)
leg_y = 30
leg = [("Implemented in memval/", "impl"), ("Proposed", "prop"), ("Online regime only ✳", "online")]
lx = W - MARGIN
for label, status in reversed(leg):
    fill, stroke, txt, dash = chip_style(status)
    cw = tw(label, 11) + 2 * 9
    rect(lx - cw, leg_y, cw, 21, fill, stroke, rx=10.5, dash=dash)
    text(lx - cw + 9, leg_y + 14.6, esc(label), 11, txt)
    lx -= cw + 8

# --------------------------------------------------------------------------
# model band
# --------------------------------------------------------------------------
y += 26
band_y = y
band_h = 74
rect(MARGIN, band_y, W - 2 * MARGIN, band_h, MODEL_FILL, MODEL_STROKE, rx=6)
text(MARGIN + 16, band_y + 24, "MODEL UNDER TEST", 11.5, MUTED, "700", ls="0.9")
mx = MARGIN + 176
for i, (group, items) in enumerate(MODEL_ITEMS):
    text(mx, band_y + 24, group.upper(), 10, FAINT, "700", ls="0.7")
    rows, _ = layout_chips([(s, "impl") for s in items], 1000)
    draw_chips(rows, mx, band_y + 33)
    mx += max(tw(group, 10, True), sum(c[2] + CHIP_GAP for c in rows[0])) + 34
text(MARGIN, band_y + band_h + 17, "the adapter contract is what makes this a benchmark rather than a set of experiments",
     11, FAINT, italic=True)

# --------------------------------------------------------------------------
# column headers
# --------------------------------------------------------------------------
y = band_y + band_h + 44
for name, cx, cw in COLS:
    text(cx, y, name, 10.5, MUTED, "700", ls="0.8")
y += 10
out.append(f'<line x1="{MARGIN}" y1="{y}" x2="{W - MARGIN}" y2="{y}" stroke="{INK}" stroke-width="1.2"/>')

# --------------------------------------------------------------------------
# lanes
# --------------------------------------------------------------------------
lane_y = y
for idx, lane in enumerate(LANES):
    cap_w = COLS[0][2] - CELL_PAD_X
    q_w = COLS[1][2] - 2 * CELL_PAD_X
    d_w = COLS[2][2] - 2 * CELL_PAD_X
    m_w = COLS[3][2] - 2 * CELL_PAD_X
    o_w = COLS[4][2] - 2 * CELL_PAD_X

    cap_lines = wrap(lane["cap"], 13.5, cap_w - 22, bold=True)
    q_lines = wrap(lane["q"], BODY_FS, q_w)
    o_lines = wrap(lane["out"], 11.8, o_w)
    d_rows, d_h = layout_chips(lane["dims"], d_w)
    m_rows, m_h = layout_chips(lane["metrics"], m_w)

    cap_h = len(cap_lines) * 18 + 16 + (18 if lane["new"] else 0)
    content_h = max(cap_h + 5, len(q_lines) * 17, d_h, m_h, len(o_lines) * 16)
    lane_h = content_h + 2 * CELL_PAD_Y

    if idx % 2 == 1:
        rect(MARGIN, lane_y, W - 2 * MARGIN, lane_h, LANE_ALT, "none", rx=0)

    # capacity block
    cap_status = lane["cap_status"]
    cf, cs = (CAP_FILL, CAP_STROKE) if cap_status == "impl" else (CAP_PROP_FILL, CAP_PROP_STROKE)
    cap_box_y = lane_y + CELL_PAD_Y - 3
    rect(COLS[0][1], cap_box_y, cap_w, cap_h, cf, cs, rx=5,
         dash=None if cap_status == "impl" else "3.5 3")
    ty = cap_box_y + 20
    for ln in cap_lines:
        text(COLS[0][1] + 12, ty, esc(ln), 13.5, INK if cap_status == "impl" else MUTED, "600")
        ty += 18
    if lane["new"]:
        bw = tw("added here", 9.5) + 13
        rect(COLS[0][1] + 12, ty - 8, bw, 15, "#ffffff", cs, rx=7.5)
        text(COLS[0][1] + 18, ty + 3, "added here", 9.5, MUTED, "700")

    # question
    ty = lane_y + CELL_PAD_Y + 12
    for ln in q_lines:
        text(COLS[1][1] + CELL_PAD_X, ty, esc(ln), BODY_FS, "#2b3440")
        ty += 17

    draw_chips(d_rows, COLS[2][1] + CELL_PAD_X, lane_y + CELL_PAD_Y)
    draw_chips(m_rows, COLS[3][1] + CELL_PAD_X, lane_y + CELL_PAD_Y)

    ty = lane_y + CELL_PAD_Y + 12
    for ln in o_lines:
        text(COLS[4][1] + CELL_PAD_X, ty, esc(ln), 11.8, MUTED)
        ty += 16

    lane_y += lane_h
    out.append(f'<line x1="{MARGIN}" y1="{lane_y}" x2="{W - MARGIN}" y2="{lane_y}" '
               f'stroke="{RULE}" stroke-width="1"/>')

# column separators over the lane block
for _, cx, _ in COLS[1:]:
    out.append(f'<line x1="{cx - 8}" y1="{y}" x2="{cx - 8}" y2="{lane_y}" stroke="{RULE}" stroke-width="1"/>')

# --------------------------------------------------------------------------
# diagnostics band
# --------------------------------------------------------------------------
dy = lane_y + 26
drows, dh = layout_chips([(s, "diag") for s in DIAGNOSTICS], W - 2 * MARGIN - 250)
band_h = dh + 30
rect(MARGIN, dy, W - 2 * MARGIN, band_h, DIAG_FILL, DIAG_STROKE, rx=6)
text(MARGIN + 16, dy + 25, "DIAGNOSTICS", 11.5, DIAG_TEXT, "700", ls="0.9")
text(MARGIN + 16, dy + 42, "explain a result — never scored", 10.5, "#7a6f92", italic=True)
for row in drows:
    for i, (label, status, cw) in enumerate(row):
        row[i] = (label, "diag_chip", cw)
cy = dy + 15
for row in drows:
    cx = MARGIN + 216
    for label, _, cw in row:
        rect(cx, cy, cw, CHIP_H, "#ffffff", "#cfc8dd", rx=CHIP_H / 2)
        text(cx + CHIP_PAD, cy + CHIP_H / 2 + 4.1, esc(label), CHIP_FS, DIAG_TEXT)
        cx += cw + CHIP_GAP
    cy += CHIP_H + ROW_GAP

# --------------------------------------------------------------------------
# rails
# --------------------------------------------------------------------------
ry = dy + band_h + 30
text(MARGIN, ry, "CROSS-CUTTING RAILS", 10.5, MUTED, "700", ls="0.8")
text(MARGIN + 172, ry, "apply to every lane at once; declare with every run", 11, FAINT, italic=True)
ry += 12
out.append(f'<line x1="{MARGIN}" y1="{ry}" x2="{W - MARGIN}" y2="{ry}" stroke="{INK}" stroke-width="1.2"/>')

for rail in RAILS:
    is_oracle = rail["label"] == "ORACLE LEVEL"
    rrows, rh = layout_chips(rail["chips"], W - 2 * MARGIN - 260)
    note_lines = wrap(rail["note"], 11, W - 2 * MARGIN - 260)
    rail_h = max(rh, 0) + len(note_lines) * 15 + 26
    if is_oracle:
        rect(MARGIN, ry + 1, W - 2 * MARGIN, rail_h - 2, "#fcfaf8", "#e6ddd4", rx=6)
    text(MARGIN + (14 if is_oracle else 0), ry + 26, rail["label"], 11.5,
         INK if is_oracle else MUTED, "700", ls="0.9")
    draw_chips(rrows, MARGIN + 244, ry + 14)
    ny = ry + 14 + rh + 14
    for ln in note_lines:
        text(MARGIN + 244, ny, esc(ln), 11, MUTED if is_oracle else FAINT)
        ny += 15
    ry += rail_h
    out.append(f'<line x1="{MARGIN}" y1="{ry}" x2="{W - MARGIN}" y2="{ry}" stroke="{RULE}" stroke-width="1"/>')

H = ry + 34
text(MARGIN, H - 12, "Solid chips are implemented in memval/ as of this tree; dashed chips are proposed. "
                     "Generated by docs/figures/overview_figure.py", 10.5, FAINT)

# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}" width="{W}" height="{H:.0f}">'
       f'<rect x="0" y="0" width="{W}" height="{H:.0f}" fill="{BG}"/>' + "".join(out) + "</svg>")

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memval_overview.svg")
with open(path, "w", encoding="utf-8") as fh:
    fh.write(svg)
print(f"wrote {path}  ({W}x{H:.0f})")
