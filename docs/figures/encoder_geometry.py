"""Geometry of the symbolic encoders: PCA views and the similarity knob.

Renders two figures into docs/figures/:

  encoder_geometry_pca.png   -- SymbolicEncoder. Three PCA views of the same
      vocabulary at three settings of ``category_variance`` (sigma), plus the
      analytic similarity knob that those three panels are three points on.

  encoder_hierarchy_svd.png  -- HierarchicalEncoder. The cosine ladder the tree
      induces, and the singular spectrum whose dimensions ARE the splits.

Both figures are generated from the shipped encoders, not from hand-set values;
re-run this file after any change to memval/encoders/.

Vocabulary note: the panels use three categories of six items. The suite's own
fallback vocabulary carries only two tools, so the tool category is padded here
(drill, saw, chisel, pliers). The balance is deliberate -- with K = 3 categories
the between-category subspace is exactly K - 1 = 2 dimensional, so a 2-D
projection loses none of the category geometry and every bit of residual spread
inside a cluster is the noise ball, not a projection artefact.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from memval.encoders.symbolic import SymbolicEncoder
from memval.encoders.hierarchical import HierarchicalEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
D = 100                      # embedding_dim used throughout the symbolic suite
SEED = 42

VOCAB = {}
for w in ["apple", "banana", "orange", "grape", "pear", "peach"]:
    VOCAB[w] = "fruit"
for w in ["cat", "dog", "cow", "horse", "sheep", "pig"]:
    VOCAB[w] = "animal"
for w in ["hammer", "wrench", "drill", "saw", "chisel", "pliers"]:
    VOCAB[w] = "tool"

CAT_COLOR = {"fruit": "#1b9e77", "animal": "#7570b3", "tool": "#d95f02"}
PANEL_SIGMAS = [0.05, 0.20, 0.50]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def null_evr(n, d, reps=400, seed=0):
    """Top-2 explained-variance ratio expected from n isotropic points in d
    dimensions. With n << d the sample covariance has rank n-1, so the top two
    PCs capture ~2/(n-1) of the variance whether or not any structure exists.
    Printing the panel EVR without this reference invites reading pure sampling
    noise as category structure."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        X = rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
        out.append(np.sum(sv[:2] ** 2) / np.sum(sv ** 2))
    return float(np.mean(out))


def pca2(X):
    """Top-2 principal components of X. Returns (scores, explained_var_ratio)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt[:2].T
    evr = float(np.sum(S[:2] ** 2) / np.sum(S ** 2))
    return scores, evr


def align_to(reference, scores, labels, ref_labels):
    """Rotate/reflect `scores` onto `reference` by orthogonal Procrustes on the
    category centroids. The 2-D basis PCA returns is arbitrary up to O(2), so
    this only removes a nuisance and never moves a point relative to another."""
    cats = sorted(set(labels))
    A = np.array([reference[[i for i, l in enumerate(ref_labels) if l == c]].mean(0) for c in cats])
    B = np.array([scores[[i for i, l in enumerate(labels) if l == c]].mean(0) for c in cats])
    A = A - A.mean(0); B = B - B.mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    return scores @ (U @ Vt)


def _label_text(ax, tx, ty, angle, word):
    """One label, positioned so it reads away from the point it annotates."""
    return ax.annotate(
        word, (tx, ty), fontsize=7.5, color="#333333",
        ha="center" if abs(np.cos(angle)) < 0.5 else
           ("left" if np.cos(angle) > 0 else "right"),
        va="center", zorder=4,
        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


def _overlap(a, b):
    """Area of intersection of two display-space bboxes, 0 if disjoint."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def _marker_boxes(ax, scores, half=5.0):
    """Display-space boxes around the plotted markers, so labels avoid them."""
    from matplotlib.transforms import Bbox
    pts = ax.transData.transform(scores)
    return [Bbox([[x - half, y - half], [x + half, y + half]]) for x, y in pts]


def radial_labels(ax, scores, words, cats, pad_frac=0.19, mode="ring"):
    """Label every point, in one of two placement modes.

    ``ring``   -- labels sit on a circle around the category centroid, joined to
                  their point by a leader line. Right when the cluster is
                  near-degenerate (low sigma), where labels drawn at the points
                  themselves overplot into an unreadable smear.
    ``spread`` -- each label sits just outside its own point, pushed outward
                  along that point's bearing from the centroid. Right when the
                  cloud is already wide, where a ring would be drawn straight
                  through the points it is meant to annotate.

    In ``spread`` mode radial offsets alone are not enough -- two nearby points
    push their labels the same way -- so each label tries a fan of candidate
    angles and radii and takes the first that clears every box already placed,
    the markers, and the legend.
    """
    xs, ys = scores[:, 0], scores[:, 1]
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    R = pad_frac * span

    slots = []                    # (point index, target x, target y, bearing)
    bearings = []                 # (point index, bearing) for spread mode
    for c in sorted(set(cats)):
        idx = [i for i, l in enumerate(cats) if l == c]
        cen = scores[idx].mean(axis=0)
        # keep placement faithful to each point's own bearing from the centroid
        ang = np.arctan2(scores[idx, 1] - cen[1], scores[idx, 0] - cen[0])
        if mode == "ring":
            order = np.argsort(ang)
            ring = np.linspace(0, 2 * np.pi, len(idx), endpoint=False) + np.pi / len(idx)
            slots += [(idx[o], cen[0] + R * np.cos(a), cen[1] + R * np.sin(a), a)
                      for o, a in zip(order, ring)]
        else:
            for j, i in enumerate(idx):
                # a point sitting on the centroid has no bearing; give it a slot
                bearings.append((i, ang[j] if np.linalg.norm(scores[i] - cen) > 1e-9
                                    else 2 * np.pi * j / len(idx)))

    m = (0.34 if mode == "ring" else 0.30) * span
    ax.set_xlim(xs.min() - m, xs.max() + m)
    ax.set_ylim(ys.min() - m, ys.max() + m)

    if mode == "ring":
        for i, tx, ty, a in slots:
            ax.plot([scores[i, 0], tx], [scores[i, 1], ty], lw=0.5,
                    color="#bbbbbb", zorder=1)
            _label_text(ax, tx, ty, a, words[i])
        return

    fig = ax.figure
    fig.canvas.draw()                      # renderer + legend extent need a draw
    rend = fig.canvas.get_renderer()
    obstacles = _marker_boxes(ax, scores)
    leg = ax.get_legend()
    if leg is not None:
        obstacles.append(leg.get_window_extent())

    base = 0.055 * span
    for i, a0 in bearings:
        best = None                        # (overlap, x, y, angle)
        for step in range(12):
            a = a0 + (1 if step % 2 else -1) * ((step + 1) // 2) * (np.pi / 6)
            for rad in (base, 1.9 * base, 2.9 * base):
                tx = scores[i, 0] + rad * np.cos(a)
                ty = scores[i, 1] + rad * np.sin(a)
                probe = _label_text(ax, tx, ty, a, words[i])
                bb = probe.get_window_extent(rend)
                probe.remove()
                ov = sum(_overlap(bb, o) for o in obstacles)
                if best is None or ov < best[0]:
                    best = (ov, tx, ty, a, bb)
                if ov == 0:
                    break
            if best[0] == 0:
                break
        _, tx, ty, a, bb = best
        ax.plot([scores[i, 0], tx], [scores[i, 1], ty], lw=0.5,
                color="#bbbbbb", zorder=1)
        _label_text(ax, tx, ty, a, words[i])
        obstacles.append(bb)


def cosine_stats(E, cats):
    """Mean within-category and between-category cosine of unit-norm rows."""
    S = E @ E.T
    n = len(E)
    iu = np.triu_indices(n, 1)
    same = cats[iu[0]] == cats[iu[1]]
    return float(S[iu][same].mean()), float(S[iu][~same].mean())


# ----------------------------------------------------------------------
# Figure 1 -- SymbolicEncoder: PCA views + the similarity knob
# ----------------------------------------------------------------------
def figure_pca():
    cats = np.array([VOCAB[w] for w in VOCAB])          # construction order
    words = list(VOCAB)

    fig = plt.figure(figsize=(13.5, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], hspace=0.34, wspace=0.22)

    chance = null_evr(len(words), D)
    ref_scores = None
    for k, sigma in enumerate(PANEL_SIGMAS):
        enc = SymbolicEncoder(VOCAB, embedding_dim=D, category_variance=sigma, seed=SEED)
        order = [enc.word_to_idx[w] for w in words]
        E = enc.embeddings[order]
        scores, evr = pca2(E)
        if ref_scores is None:
            ref_scores = scores
        else:
            scores = align_to(ref_scores, scores, cats, cats)

        ax = fig.add_subplot(gs[0, k])
        for c in CAT_COLOR:
            m = cats == c
            ax.scatter(scores[m, 0], scores[m, 1], s=54, c=CAT_COLOR[c],
                       edgecolors="white", linewidths=0.8, zorder=3,
                       label=c if k in (0, 1) else None)
        # Panels 0 and 1 are the ones with structure to read, so both carry the
        # full apparatus. Panel 2 sits at chance (see its title) -- labelling
        # noise would only invite the reader to find groupings in it.
        #
        # The legend must exist BEFORE the labels are placed: spread-mode
        # placement treats it as an obstacle, and cannot avoid a legend that has
        # not been created yet.
        if k in (0, 1):
            ax.set_ylabel("PC 2", fontsize=9)
            ax.legend(frameon=False, fontsize=8.5,
                      loc="upper right" if k == 0 else "lower right",
                      handletextpad=0.4, borderaxespad=0.6)
            radial_labels(ax, scores, words, cats,
                          mode="ring" if k == 0 else "spread")
        within, between = cosine_stats(E, cats)
        ax.set_title(
            rf"$\sigma$ = {sigma:.2f}   ·   within-cat. cos = {within:.2f}"
            "\n" rf"top-2 PCs: {100*evr:.0f}% of variance "
            rf"(chance {100*chance:.0f}%)",
            fontsize=10,
            color="#111111" if evr > chance * 1.15 else "#8a3b00")
        ax.axhline(0, color="#dddddd", lw=0.7, zorder=0)
        ax.axvline(0, color="#dddddd", lw=0.7, zorder=0)
        ax.set_xlabel("PC 1", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=8)

    # --- the knob itself -------------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    sig_grid = np.geomspace(0.02, 2.0, 220)
    ax.plot(sig_grid, 1.0 / (1.0 + D * sig_grid ** 2), color="#1b9e77", lw=2.0,
            label=r"theory  $\rho_{\rm within} = 1/(1 + d\sigma^2)$")
    ax.axhline(0.0, color="#7570b3", lw=1.6, ls="-",
               label=r"theory  $\rho_{\rm between} \approx 0$")
    ax.fill_between(sig_grid, -1 / np.sqrt(D), 1 / np.sqrt(D), color="#7570b3",
                    alpha=0.13, lw=0,
                    label=r"$\pm 1/\sqrt{d}$  (between-cat. spread)")

    meas_s = np.geomspace(0.02, 2.0, 22)
    w_meas, b_meas = [], []
    for s in meas_s:
        enc = SymbolicEncoder(VOCAB, embedding_dim=D, category_variance=s, seed=SEED)
        order = [enc.word_to_idx[w] for w in words]
        w_, b_ = cosine_stats(enc.embeddings[order], cats)
        w_meas.append(w_); b_meas.append(b_)
    ax.plot(meas_s, w_meas, "o", ms=4.2, color="#1b9e77", mec="white", mew=0.6,
            label="measured, within (18 words)")
    ax.plot(meas_s, b_meas, "s", ms=3.6, color="#7570b3", mec="white", mew=0.6,
            label="measured, between")

    for s in PANEL_SIGMAS:                              # tie back to the panels
        ax.axvline(s, color="#999999", lw=0.8, ls=":", zorder=0)
        ax.annotate(rf"panel $\sigma$={s:g}", (s, -0.155), fontsize=7.8,
                    color="#666666", ha="center", va="bottom",
                    bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xscale("log")
    ax.set_xlim(0.02, 2.0); ax.set_ylim(-0.19, 1.02)
    ax.set_xlabel(r"category_variance  $\sigma$", fontsize=9.5)
    ax.set_ylabel("mean pairwise cosine", fontsize=9.5)
    ax.set_title("The knob: one scalar sets item overlap, and only overlap",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper right",
              borderaxespad=0.9)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # --- inverse design table -------------------------------------------
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.set_title("Designing a target overlap", fontsize=10.5, loc="left")
    ax.text(0.0, 0.87,
            r"invert the law:   $\sigma(\rho) = \sqrt{\dfrac{1-\rho}{\rho\, d}}$",
            fontsize=11, transform=ax.transAxes)
    rows = [(r, np.sqrt((1 - r) / (r * D))) for r in (0.8, 0.6, 0.4, 0.2, 0.1)]
    ax.text(0.02, 0.66, "target cos", fontsize=9, weight="bold", transform=ax.transAxes)
    ax.text(0.36, 0.66, r"required $\sigma$", fontsize=9, weight="bold", transform=ax.transAxes)
    ax.text(0.70, 0.66, "achieved", fontsize=9, weight="bold", transform=ax.transAxes)
    for i, (r, s) in enumerate(rows):
        enc = SymbolicEncoder(VOCAB, embedding_dim=D, category_variance=s, seed=SEED)
        order = [enc.word_to_idx[w] for w in words]
        got, _ = cosine_stats(enc.embeddings[order], cats)
        y = 0.55 - 0.105 * i
        ax.text(0.02, y, f"{r:.2f}", fontsize=9, transform=ax.transAxes)
        ax.text(0.36, y, f"{s:.3f}", fontsize=9, transform=ax.transAxes)
        ax.text(0.70, y, f"{got:.2f}", fontsize=9, color="#666666", transform=ax.transAxes)
    ax.text(0.0, -0.04,
            "'achieved' is this 18-word sample; the law is exact\n"
            "in the large-vocabulary limit (see docstring).\n\n"
            "'chance' above each PCA panel is the top-2 EVR of 18 isotropic\n"
            "points in 100-D. The right-hand panel sits AT chance: what is\n"
            "plotted there is sampling noise, not weak category structure.",
            fontsize=7.6, color="#777777", transform=ax.transAxes, va="top")

    fig.suptitle("Hierarchical symbolic encoder: what the 100-D code looks like, "
                 "and the one parameter that moves it",
                 fontsize=12.5, y=0.985)
    out = os.path.join(HERE, "encoder_geometry_pca.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 2 -- HierarchicalEncoder: the cosine ladder and the SVD
# ----------------------------------------------------------------------
def figure_hierarchy():
    names = {"animal": {"bird": ["sparrow", "hawk"], "fish": ["salmon", "sunfish"]},
             "plant": {"tree": ["oak", "maple"], "flower": ["rose", "daisy"]}}
    h = HierarchicalEncoder(tree=names, features_per_node=2)
    items = h.items
    E = h.embeddings
    C = E @ E.T
    L = max(n["level"] for n in h.nodes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})

    im = ax1.imshow(C, cmap="Greens", vmin=0, vmax=1)
    ax1.set_xticks(range(len(items))); ax1.set_xticklabels(items, rotation=45,
                                                           ha="right", fontsize=8)
    ax1.set_yticks(range(len(items))); ax1.set_yticklabels(items, fontsize=8)
    for i in range(len(items)):
        for j in range(len(items)):
            ax1.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if C[i, j] > 0.55 else "#444444")
    ax1.set_title(r"Cosine is quantised: $\cos(i,j) = \ell_{ij}\,/\,L$"
                  "\n" rf"here $L$ = {L}, so the ladder is "
                  + ", ".join(f"{k}/{L}" for k in range(L + 1)),
                  fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

    align = h.dimension_alignment()
    s = [d["singular_value"] for d in align]
    lv = [d["level"] for d in align]
    lvl_color = {1: "#08519c", 2: "#4292c6", 3: "#9ecae1"}
    bars = ax2.bar(range(len(s)), s, color=[lvl_color[l] for l in lv],
                   edgecolor="white", linewidth=0.8)
    for i, d in enumerate(align):
        ax2.text(i, s[i] + 0.08, d["match_node"], rotation=90, ha="center",
                 va="bottom", fontsize=7.4, color="#333333")
    ax2.set_ylim(0, max(s) * 1.55)
    ax2.set_xticks(range(len(s)))
    ax2.set_xticklabels([f"dim {i}" for i in range(len(s))], fontsize=8, rotation=45,
                        ha="right")
    ax2.set_ylabel("singular value  $s_k$", fontsize=9.5)
    ax2.set_title("Each singular dimension IS one split, and depth sets its strength\n"
                  "(alignment to the named split = "
                  f"{min(d['alignment'] for d in align):.2f}–"
                  f"{max(d['alignment'] for d in align):.2f})",
                  fontsize=10)
    ax2.legend(handles=[Line2D([], [], marker="s", ls="", ms=9, mec="white",
                               color=lvl_color[l], label=f"level-{l} split")
                        for l in (1, 2, 3)],
               frameon=False, fontsize=8.5, loc="upper right")
    ax2.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.suptitle("Deep-tree variant (HierarchicalEncoder): similarity is set by "
                 "tree depth, not by a noise scale", fontsize=12, y=1.0)
    fig.tight_layout()
    out = os.path.join(HERE, "encoder_hierarchy_svd.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", figure_pca())
    print("wrote", figure_hierarchy())
