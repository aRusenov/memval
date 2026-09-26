"""Continual-retention metrics, decoupled from any one suite.

Extracted from ``online_continual_pipeline`` so that any benchmark — symbolic,
spatial, streamed or batch — can report the same retention numbers. The caller
supplies two callbacks and owns everything modality-specific (encoders, task
construction, ingestion regime); this module owns only the matrix and the
summary statistics computed from it.

Convention
----------
``R[j, i]`` = score on task *i* after training **through** task *j*.
Rows are training stages, columns are evaluated tasks, so the matrix is
lower-triangular and ``NaN`` above the diagonal (a task cannot be scored before
it has been seen).

  - diagonal ``R[i, i]``  — how well task *i* was learned when fresh
  - bottom row ``R[T-1, i]`` — what survives of each task at the end
  - down a column        — how one task decays as later tasks overwrite it
  - down the diagonal    — how acquisition itself degrades as the substrate
    fills. This is the *plasticity* axis and it is the half a retention matrix
    is usually read without: a model that stops learning after task 0 has a
    perfect bottom row, ACC = 1, BWT = 0 and zero forgetting. ``avg_learning``
    and ``intransigence`` score it; ``stability_plasticity_index`` refuses to
    reward either half alone.

Note the transpose relative to ``bin/continual_chain_experiment.py``'s original
local convention (``R[i, j]``); that script now calls into here and reports the
same orientation as everything else.
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def retention_matrix(
    n_tasks: int,
    train_task: Callable[[int], Any],
    score_task: Callable[[int, int], float],
) -> np.ndarray:
    """Train a task chain on one model and score every task seen so far.

    Args:
        n_tasks: Number of tasks *T* in the chain.
        train_task: ``train_task(j)`` trains the model on task *j*. Called once
            per stage, in order, with **no weight reset between tasks** — that
            is the caller's responsibility and is what makes this continual.
            Any consolidation hook belongs here too.
        score_task: ``score_task(i, j)`` returns the scalar score on task *i*
            after training through task *j*. Called for every ``i <= j``.
            Transient state (e.g. ``reset_context``) should be cleared inside.

    Returns:
        ``(T, T)`` array, lower-triangular, ``NaN`` above the diagonal.
    """
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")

    R = np.full((n_tasks, n_tasks), np.nan)
    for j in range(n_tasks):
        train_task(j)
        for i in range(j + 1):
            R[j, i] = float(score_task(i, j))
    return R


def _chance_correct(x: float, chance_level: float) -> float:
    """Map a raw score onto [0, 1] where 0 is chance and 1 is perfect."""
    if chance_level >= 1.0:
        return float("nan")
    return float(np.clip((x - chance_level) / (1.0 - chance_level), 0.0, 1.0))


def retention_summary(R: np.ndarray, chance_level: float = 0.0) -> Dict[str, Any]:
    """Standard continual-learning statistics from a retention matrix.

    Args:
        R: ``(T, T)`` lower-triangular retention matrix from
            :func:`retention_matrix`.
        chance_level: score a chance-level responder would obtain on the
            caller's read-out (e.g. ``1/vocab_size`` for cued MRR). Used **only**
            by the joint index and its two coordinates, which are ratios and are
            meaningless on an uncorrected scale — a chance floor of 0.07 makes a
            dead model look 7% plastic. Every other statistic below is a
            difference and is reported raw.

    Stability — does earlier material survive later material?

    - ``avg_accuracy`` (ACC) — mean of the final row: what survives overall.
    - ``backward_transfer`` (BWT) — mean of ``final - just_learned`` over every
      task except the last. Negative means forgetting; positive means later
      tasks *helped* earlier ones.
    - ``avg_forgetting`` — mean drop from each task's best-ever score to its
      final score. Unlike BWT this uses the maximum rather than the diagonal, so
      a task that improved before decaying is scored against its peak. Following
      Chaudhry et al., the peak is taken over training stages *before* the final
      one, so this goes negative when a task's score improves at the last stage;
      that is backward transfer showing up in the forgetting column, not a bug.
    - ``forgetting_by_gap`` — mean *retained fraction* ``R[i+g, i] / R[i, i]``
      at each interposition gap ``g = 1 … T-1``. The forgetting **gradient**:
      the scalar summaries above read the trace at one point, this reads the
      decay. Entry ``g-1`` of the list is gap ``g``. Pairs whose diagonal is 0
      (nothing was learned, so nothing can decay) are dropped; a gap with no
      usable pair is ``nan``.

    Plasticity — can new material still get in?

    - ``avg_learning`` (LA) — mean of the diagonal: how well each task was
      acquired *at the moment it was trained*, with every earlier task already
      in the substrate. Chaudhry et al.'s Learning Accuracy.
    - ``intransigence`` — ``mean_{i>=1}(R[0,0] - R[i,i])``: how much worse
      acquisition became once the substrate was no longer empty. The reference
      is the arm's **own** first task rather than a separately trained joint
      model, so this is a within-run quantity and needs no second training run;
      it is correspondingly not comparable to Chaudhry's ``I`` in absolute
      terms. Positive means plasticity decayed under load, negative means later
      tasks were acquired *better* (forward transfer).
    - ``learning_slope`` — OLS slope of the diagonal against task index. The
      shape ``intransigence`` averages away: a cliff at task 1 and a steady
      decline give the same mean.

    The joint readout — neither half alone.

    - ``retention_ratio`` — chance-corrected ``ACC / LA``: **of what actually
      got in, how much survived**. This is the stability coordinate, and unlike
      ``avg_forgetting`` it cannot be gamed by failing to learn: a model that
      never acquires task *i* has no credit to keep for it.
    - ``stability_plasticity_index`` (SPI) — harmonic mean of ``avg_learning``
      (chance-corrected) and ``retention_ratio``. The harmonic mean is the point:
      it collapses toward 0 whenever *either* coordinate does, so the frozen
      model (perfect retention of the one thing it learned) and the overwriting
      model (perfect acquisition, nothing left) both score near 0, while the
      arithmetic mean would hand each of them 0.5. Read it with the two
      coordinates, never on its own — the index says *how balanced*, only the
      coordinates say *which way an arm fails*.

    ``avg_forgetting``, ``backward_transfer`` and ``intransigence`` are 0.0 for a
    single-task chain, where none of them is defined.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be a square matrix, got shape {R.shape}")
    T = R.shape[0]

    final = R[T - 1, :]
    diag = np.array([R[i, i] for i in range(T)])

    acc = float(np.nanmean(final))
    bwt = float(np.mean(final[:-1] - diag[:-1])) if T > 1 else 0.0
    per_task_forgetting = [
        float(np.nanmax(R[i:T - 1, i]) - R[T - 1, i]) for i in range(T - 1)
    ]
    avg_forget = float(np.mean(per_task_forgetting)) if per_task_forgetting else 0.0

    # --- plasticity axis: read down the diagonal ---------------------------
    avg_learning = float(np.nanmean(diag))
    intransigence = float(np.mean(diag[0] - diag[1:])) if T > 1 else 0.0
    if T > 1:
        idx = np.arange(T, dtype=float)
        learning_slope = float(np.polyfit(idx, diag, 1)[0])
    else:
        learning_slope = 0.0

    # --- forgetting gradient: retained fraction at each interposition gap ---
    forgetting_by_gap: List[float] = []
    for gap in range(1, T):
        kept = [R[i + gap, i] / diag[i] for i in range(T - gap) if diag[i] > 0]
        forgetting_by_gap.append(float(np.mean(kept)) if kept else float("nan"))

    # --- joint readout -----------------------------------------------------
    la_c = _chance_correct(avg_learning, chance_level)
    acc_c = _chance_correct(acc, chance_level)
    if la_c > 0:
        retention_ratio = float(np.clip(acc_c / la_c, 0.0, 1.0))
    else:
        # Nothing was ever acquired, so there is no trace whose survival could
        # be measured. 0.0, not nan: a chain that learned nothing has no
        # retention, and propagating nan would silently drop the arm from the
        # comparison it is meant to lose.
        retention_ratio = 0.0
    denom = la_c + retention_ratio
    spi = float(2.0 * la_c * retention_ratio / denom) if denom > 0 else 0.0

    return {
        "avg_accuracy": acc,
        "backward_transfer": bwt,
        "avg_forgetting": avg_forget,
        "avg_learning": avg_learning,
        "intransigence": intransigence,
        "learning_slope": learning_slope,
        "retention_ratio": retention_ratio,
        "stability_plasticity_index": spi,
        "chance_level": float(chance_level),
        "per_task_final": final.tolist(),
        "per_task_learned": diag.tolist(),
        "per_task_forgetting": per_task_forgetting,
        "forgetting_by_gap": forgetting_by_gap,
    }


def plot_retention_matrix(
    R: np.ndarray,
    path: str,
    title: str = "Retention",
    subtitle: Optional[str] = None,
    value_label: str = "score",
    task_labels: Optional[List[str]] = None,
    chance_level: float = 0.0,
) -> None:
    """Write the standard lower-triangular retention heatmap to ``path``.

    ``chance_level`` must be the one the caller scored with: the header's SPI
    is a chance-corrected ratio and disagrees with the scored
    ``stability_plasticity_index`` for any arm below ceiling if it is left at
    zero. ACC, LA, forgetting and intransigence are differences and do not
    depend on it.

    matplotlib is imported lazily so importing this module stays cheap for
    callers that only want the numbers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    summary = retention_summary(R, chance_level=chance_level)
    if subtitle is None:
        subtitle = (f"ACC={summary['avg_accuracy']:.2f}  "
                    f"LA={summary['avg_learning']:.2f}  "
                    f"Forget={summary['avg_forgetting']:.2f}  "
                    f"Intrans={summary['intransigence']:+.2f}  "
                    f"SPI={summary['stability_plasticity_index']:.2f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(R, vmin=0, vmax=1, cmap="viridis")
    ticks = range(T)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if task_labels is not None:
        ax.set_xticklabels(task_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(task_labels, fontsize=8)
    ax.set_xlabel("Evaluated task i")
    ax.set_ylabel("After training task j")
    ax.set_title(f"{title}\n{subtitle}")
    for j in range(T):
        for i in range(j + 1):
            ax.text(i, j, f"{R[j, i]:.2f}", ha="center", va="center",
                    color="w", fontsize=9)
    fig.colorbar(im, label=value_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stability_plasticity_plane(
    points: List[Dict[str, Any]],
    path: str,
    title: str = "Stability–plasticity plane",
) -> None:
    """Plot arms on the (plasticity, stability) plane with SPI iso-contours.

    The plane is the whole argument for reporting both halves: an arm's position
    says *which way* it fails, which no single scalar can. The two coordinates
    are the chance-corrected ones from :func:`retention_summary`:

        x — ``avg_learning``   : could new material still get in?
        y — ``retention_ratio``: of what got in, how much survived?

    Bottom-right is the overwriter (learns everything, keeps nothing), top-left
    the frozen model (keeps everything it has, which is one task), top-right the
    goal. The dotted contours are levels of ``stability_plasticity_index``, so
    the harmonic mean's refusal to reward a single axis is visible rather than
    asserted: both corners sit on low contours.

    Args:
        points: one dict per arm, with ``label`` plus either a ``summary`` dict
            from :func:`retention_summary` or explicit ``x`` / ``y`` floats.
        path: PNG destination.
        title: figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5.5))

    grid = np.linspace(1e-6, 1.0, 200)
    X, Y = np.meshgrid(grid, grid)
    SPI = 2.0 * X * Y / (X + Y)
    cs = ax.contour(X, Y, SPI, levels=[0.2, 0.4, 0.6, 0.8],
                    colors="grey", linestyles=":", linewidths=0.8)
    ax.clabel(cs, fmt="SPI %.1f", fontsize=7)

    for pt in points:
        s = pt.get("summary") or {}
        x = pt.get("x", s.get("avg_learning_corrected",
                              _chance_correct(s.get("avg_learning", float("nan")),
                                              s.get("chance_level", 0.0))))
        y = pt.get("y", s.get("retention_ratio", float("nan")))
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            continue
        ax.scatter([x], [y], s=60, zorder=3)
        ax.annotate(pt.get("label", ""), (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Plasticity — acquisition under load (chance-corrected LA)")
    ax.set_ylabel("Stability — retained fraction of what was acquired")
    ax.set_title(title)
    ax.annotate("overwriter", (0.97, 0.03), ha="right", fontsize=7, color="grey")
    ax.annotate("frozen", (0.03, 0.97), ha="left", va="top", fontsize=7, color="grey")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweeps: the same matrix at several settings of a stimulus dial
# ---------------------------------------------------------------------------

def _masked_matrix_image(ax, R, plt, vmin=0.0, vmax=1.0, cmap="viridis",
                         annotate=True, fontsize=7):
    """imshow with the untrained (upper) triangle drawn grey, not as zero."""
    import matplotlib.colors as mcolors
    R = np.asarray(R, dtype=float)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#e6e6ee")
    im = ax.imshow(np.ma.masked_invalid(R), vmin=vmin, vmax=vmax, cmap=cm)
    if annotate:
        T = R.shape[0]
        for j in range(T):
            for i in range(j + 1):
                v = R[j, i]
                if np.isfinite(v):
                    ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                            color="w" if v < 0.6 * vmax else "k", fontsize=fontsize)
    return im


def plot_retention_matrix_grid(
    rows: List[Dict[str, Any]],
    path: str,
    title: str = "Retention matrices along a stimulus dial",
    value_label: str = "score",
    task_labels: Optional[List[str]] = None,
    chance_level: float = 0.0,
) -> None:
    """Small multiples: one full retention matrix per rung of a dial.

    ``rows`` is a list of ``{"name": str, "panels": [{"label": str, "R": TxT}, ...]}``
    -- one row per dial, one panel per rung, so two dials can be read against
    each other on a shared 0-1 colour scale. Each panel is captioned with its
    rung label and the ACC / forgetting / LA it produces, which is what lets the
    reader see the triangle darken along the dial without a table.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_cols = max(len(r["panels"]) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.65 * n_cols + 1.2, 2.9 * n_rows + 0.9),
                             squeeze=False)
    im = None
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(row["panels"]):
                ax.axis("off")
                continue
            panel = row["panels"][c]
            R = np.asarray(panel["R"], dtype=float)
            T = R.shape[0]
            im = _masked_matrix_image(ax, R, plt)
            sm = retention_summary(R, chance_level=chance_level)
            ax.set_title(f"{panel['label']}\nACC {sm['avg_accuracy']:.2f}  "
                         f"forget {sm['avg_forgetting']:.2f}  LA {sm['avg_learning']:.2f}",
                         fontsize=8)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            if task_labels is not None and r == n_rows - 1:
                ax.set_xticklabels(task_labels, rotation=60, ha="right", fontsize=6.5)
            else:
                ax.set_xticklabels([])
            if task_labels is not None and c == 0:
                ax.set_yticklabels(task_labels, fontsize=6.5)
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=0)
        axes[r, 0].set_ylabel(f"{row['name']}\nafter training task j", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.005, "Evaluated task i  (grey = not yet trained)",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.86 if n_rows == 1 else 0.90,
                        bottom=0.13, wspace=0.12, hspace=0.45)
    cax = fig.add_axes([0.915, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label=value_label)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_retention_sweep_axes(
    sweeps: List[Dict[str, Any]],
    path: str,
    title: str = "Plasticity and stability along a stimulus dial",
    value_label: str = "score",
) -> None:
    """The quantitative companion to the small multiples.

    One row of four heatmaps per dial, rungs down the rows of every heatmap:

    1. **intransigence** -- epochs task i needed in the chain divided by the
       epochs a FRESH model needs on task i alone at the same rung, median
       over seeds. 1.0 means the stored tasks added nothing to the cost of
       acquiring this one; above 1 is plasticity lost to load. Dividing by the
       fresh cost cancels both the rung's material difficulty and any
       task-specific difficulty, which raw epochs conflate with load. Cells
       where any seed hit the epoch ceiling (chain or fresh) are hatched. The
       row label carries the median fresh cost, i.e. the material's own price.
       (Falls back to raw mean epochs when ``fresh_seeds`` is not supplied.)
    2. **learned** -- the diagonal ``R[i, i]`` (acquisition; censoring shows here).
    3. **final** -- the bottom row ``R[T-1, i]`` (what survives).
    4. **retained fraction by gap** -- ``R[i+g, i] / R[i, i]``, the load curve.

    Where *final* darkens while *learned* stays bright the arm is forgetting;
    where *exposure* brightens while both stay bright it is paying for
    acquisition and not forgetting. A last row of curves puts LA, ACC,
    retention ratio, SPI and forgetting (mean +- sd over seeds when given)
    against the dial.

    Each sweep is a dict with ``name``, ``xlabel``, ``x`` (rung values),
    ``rung_labels``, ``task_labels``, ``Rs`` (mean matrix per rung),
    ``exposures`` (per rung, per task epochs), ``summaries`` (per rung a list
    of per-seed summary dicts, or one dict) and optional ``log_x``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    def _epoch_scale(vmin, vmax):
        """Log colour/axis only when the range earns it; ticks always plain ints."""
        if vmax / vmin >= 8:
            ticks = sorted({int(round(t)) for t in np.geomspace(vmin, vmax, 5)})
            return LogNorm(vmin=vmin, vmax=vmax), ticks, "log"
        ticks = sorted({int(round(t)) for t in np.linspace(vmin, vmax, 4)})
        return Normalize(vmin=vmin, vmax=vmax), ticks, "linear"

    n = len(sweeps)
    fig = plt.figure(figsize=(17, 4.0 * n + 3.8))
    gs = fig.add_gridspec(n + 1, 4, height_ratios=[1.0] * n + [1.05],
                          hspace=0.55, wspace=0.42,
                          left=0.06, right=0.97, top=0.93, bottom=0.07)

    for r, sw in enumerate(sweeps):
        Rs = [np.asarray(R, dtype=float) for R in sw["Rs"]]
        T = Rs[0].shape[0]
        nr = len(Rs)
        learned = np.array([[R[i, i] for i in range(T)] for R in Rs])
        final = np.array([[R[T - 1, i] for i in range(T)] for R in Rs])
        gaps = list(range(1, T))
        kept = np.full((nr, T - 1), np.nan)
        for k, R in enumerate(Rs):
            d = np.array([R[i, i] for i in range(T)])
            for g in gaps:
                vals = [R[i + g, i] / d[i] for i in range(T - g) if d[i] > 0]
                kept[k, g - 1] = np.mean(vals) if vals else np.nan
        expo = np.asarray(sw["exposures"], dtype=float)
        have_fresh = "fresh_seeds" in sw
        if have_fresh:
            ratio = np.full((nr, T), np.nan)
            censored = np.zeros((nr, T), dtype=bool)
            fresh_med = np.full(nr, np.nan)
            for k in range(nr):
                E = np.asarray(sw["exposure_seeds"][k], dtype=float)
                F = np.asarray(sw["fresh_seeds"][k], dtype=float)
                ratio[k] = np.median(E / np.maximum(F, 1.0), axis=0)
                censored[k] = (np.asarray(sw["chain_censored_seeds"][k]).any(axis=0)
                               | np.asarray(sw["fresh_censored_seeds"][k]).any(axis=0))
                fresh_med[k] = float(np.median(np.median(F, axis=0)))
            first_panel = ("intransigence: epochs in chain / epochs fresh (median)", ratio,
                           "task", sw["task_labels"], "ratio")
        else:
            first_panel = ("exposure: epochs to criterion", expo, "task", sw["task_labels"], "log")

        panels = [
            first_panel,
            ("learned  R[i,i]", learned, "task", sw["task_labels"], None),
            ("final  R[T-1,i]", final, "task", sw["task_labels"], None),
            ("retained fraction  R[i+g,i]/R[i,i]", kept, "interposed tasks g",
             [str(g) for g in gaps], None),
        ]
        for c, (ptitle, M, xl, xt, scale) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            if scale == "ratio":
                # log2 ratio, diverging around 1.0: blue = cheaper than fresh,
                # white = no cost added, red = plasticity lost to load.
                from matplotlib.colors import TwoSlopeNorm
                from matplotlib.patches import Rectangle
                L = np.log2(np.ma.masked_invalid(M))
                lim = max(1.0, float(np.nanmax(np.abs(L))))
                im = ax.imshow(L, aspect="auto", cmap="RdBu_r",
                               norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
                cticks = [2.0 ** t for t in np.linspace(-lim, lim, 5)]
                for k in range(M.shape[0]):
                    for i in range(M.shape[1]):
                        if censored[k, i]:
                            ax.add_patch(Rectangle((i - 0.5, k - 0.5), 1, 1, fill=False,
                                                   hatch="////", edgecolor="k", lw=0.0))
                fmt = lambda v: f"{v:.1f}"
                thr = None
            elif scale == "log":
                pos = M[np.isfinite(M) & (M > 0)]
                vmin = max(1.0, float(pos.min())) if pos.size else 1.0
                vmax = max(vmin * 1.5, float(pos.max())) if pos.size else 2.0
                norm, cticks, _ = _epoch_scale(vmin, vmax)
                im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap="magma_r", norm=norm)
                fmt = lambda v: f"{v:.0f}"
                thr = None
            else:
                im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap="viridis",
                               vmin=0.0, vmax=1.0)
                fmt = lambda v: f"{v:.2f}"
                thr = 0.6
            for k in range(M.shape[0]):
                for i in range(M.shape[1]):
                    v = M[k, i]
                    if np.isfinite(v):
                        col = "k" if (thr is not None and v >= thr) else "w"
                        if scale == "log":
                            col = "w" if v > np.sqrt(vmin * vmax) else "k"
                        elif scale == "ratio":
                            col = "w" if abs(np.log2(v)) > 0.6 * lim else "k"
                        ax.text(i, k, fmt(v), ha="center", va="center", color=col, fontsize=7)
            ax.set_title(ptitle, fontsize=9)
            ax.set_xticks(range(M.shape[1]))
            ax.set_xticklabels(xt, rotation=60 if xl == "task" else 0, ha="right" if xl == "task" else "center",
                               fontsize=7)
            ax.set_yticks(range(nr))
            if c == 0 and have_fresh:
                ax.set_yticklabels([f"{lab}  (fresh {f:.0f} ep)"
                                    for lab, f in zip(sw["rung_labels"], fresh_med)], fontsize=7)
            else:
                ax.set_yticklabels(sw["rung_labels"] if c == 0 else [], fontsize=7)
            ax.set_xlabel(xl, fontsize=8)
            if c == 0:
                ax.set_ylabel(sw["name"], fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7)
            if c == 0 and scale == "ratio":
                cb.set_label("chain / fresh  (hatched = a seed hit the epoch ceiling)", fontsize=7)
                cb.set_ticks(np.log2(cticks))
                cb.set_ticklabels([f"{t:.2g}×" for t in cticks])
            elif c == 0:
                cb.set_label("epochs", fontsize=7)
                cb.set_ticks(cticks)
                cb.set_ticklabels([str(t) for t in cticks])
                cb.ax.yaxis.set_minor_formatter(NullFormatter())
            elif c == 2:
                cb.set_label(value_label, fontsize=7)

    # --- curves against the dial -------------------------------------------
    keys = [("avg_learning", "LA (learned)", "tab:blue", "-o"),
            ("avg_accuracy", "ACC (final)", "tab:red", "-x"),
            ("retention_ratio", "retention ratio", "tab:green", "-s"),
            ("stability_plasticity_index", "SPI", "k", "-d"),
            ("avg_forgetting", "avg forgetting", "crimson", "--^")]
    width = max(1, 4 // n)
    for r, sw in enumerate(sweeps):
        ax = fig.add_subplot(gs[n, r * width:(r + 1) * width])
        x = np.asarray(sw["x"], dtype=float)
        for key, lab, col, style in keys:
            mu, sd = [], []
            for per_rung in sw["summaries"]:
                seeds = per_rung if isinstance(per_rung, list) else [per_rung]
                vals = np.array([s[key] for s in seeds], dtype=float)
                mu.append(np.nanmean(vals))
                sd.append(np.nanstd(vals) if len(vals) > 1 else 0.0)
            ax.errorbar(x, mu, yerr=sd, fmt=style, color=col, label=lab, ms=4,
                        capsize=2, lw=1.2, alpha=0.9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(sw["xlabel"], fontsize=9)
        ax.set_ylabel("statistic", fontsize=9)
        if sw.get("log_x"):
            ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(sw["rung_labels"], fontsize=7)
        ax.minorticks_off()
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        if "fresh_seeds" in sw:
            ep = np.array([np.median(np.asarray(e, dtype=float)) for e in sw["exposure_seeds"]])
            fr = np.array([np.median(np.asarray(f, dtype=float)) for f in sw["fresh_seeds"]])
            ax2.plot(x, ep, ":", color="tab:purple", marker=".", label="median epochs, in chain")
            ax2.plot(x, fr, "--", color="grey", marker=".", lw=1.0, label="median epochs, fresh model")
        else:
            ep = np.array([np.nanmean(np.asarray(e, dtype=float)) for e in sw["exposures"]])
            ax2.plot(x, ep, ":", color="tab:purple", marker=".", label="mean epochs to criterion")
        _, eticks, escale = _epoch_scale(max(1.0, float(np.nanmin(ep))),
                                         max(1.5, float(np.nanmax(ep))))
        ax2.set_yscale(escale)
        ax2.set_yticks(eticks)
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax2.yaxis.set_minor_formatter(NullFormatter())
        ax2.set_ylabel(f"epochs to criterion ({escale})", color="tab:purple", fontsize=8)
        ax2.tick_params(axis="y", colors="tab:purple", labelsize=7)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left", ncol=2)
        ax.set_title(f"{sw['name']}: summary against the dial", fontsize=9)

    fig.suptitle(title, fontsize=12, y=0.985)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_intrusion_breakdown(
    sweeps: List[Dict[str, Any]],
    path: str,
    title: str = "What is recalled instead, and who the runner-up is",
) -> None:
    """Per rung of a dial: where the errors go, and where the pressure is.

    Left column: the error rate at each rung as a bar, split by the *source* of
    the intruding word (same task / an earlier task / a later task). Correct
    probes are not drawn -- they are ``1 - error rate`` and already live in the
    retention matrix -- so the axis is scaled to the errors. Right column: the best competitor of the target for
    every probe, failed or not, classed the same way, with the mean decision
    margin drawn over it. A rung whose errors are cross-task confusions and a
    rung whose errors are within-task confusions look different here even when
    their retention matrices read the same.

    Each sweep: ``name``, ``rung_labels``, ``outcomes`` / ``competitors``
    (per rung a dict of counts), ``margin_mean`` (per rung), optional
    ``margin_by_competitor`` (per rung a dict).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Errors only on the left: "correct" is 1 - errors and already lives in the
    # retention matrix, and drawing it makes the bars mostly blank grey.
    out_cls = [("within_task", "tab:orange", "intrusion from the same task"),
               ("cross_earlier", "tab:blue", "intrusion from an earlier task"),
               ("cross_later", "tab:red", "intrusion from a later task")]
    comp_cls = [("same_task", "tab:orange", "runner-up in the same task"),
                ("earlier", "tab:blue", "runner-up in an earlier task"),
                ("later", "tab:red", "runner-up in a later task")]

    n = len(sweeps)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.6 * n + 0.8), squeeze=False)
    for r, sw in enumerate(sweeps):
        labels = sw["rung_labels"]
        x = np.arange(len(labels))
        for c, (classes, key) in enumerate(((out_cls, "outcomes"), (comp_cls, "competitors"))):
            ax = axes[r, c]
            bottom = np.zeros(len(labels))
            # Fractions are always of ALL probes (denominator includes correct ones),
            # so bar height on the left is the error rate and the segments are
            # where those errors went.
            tot = np.array([sum(d.values()) for d in sw[key]], dtype=float)
            for cls, col, lab in classes:
                frac = np.array([d.get(cls, 0) for d in sw[key]], dtype=float) / np.maximum(tot, 1)
                ax.bar(x, frac, bottom=bottom, color=col, label=lab, width=0.7,
                       edgecolor="white", linewidth=0.5)
                thr = 0.02 if c == 0 else 0.06
                for xi, (b, f) in enumerate(zip(bottom, frac)):
                    if f >= thr:
                        ax.text(xi, b + f / 2, f"{f:.2f}", ha="center", va="center",
                                fontsize=7, color="w")
                bottom += frac
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            if c == 0:
                top = float(bottom.max())
                ax.set_ylim(0, max(0.05, top * 1.25))
                for xi, t in enumerate(bottom):
                    ax.text(xi, t + 0.012 * max(0.05, top * 1.25), f"{t:.2f}" if t > 0 else "0",
                            ha="center", va="bottom", fontsize=7, color="k")
                ax.set_ylabel("error rate (fraction of final-stage probes)", fontsize=8)
                ax.set_title(f"{sw['name']}: where the errors went", fontsize=9)
            else:
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("fraction of final-stage probes", fontsize=8)
            if c == 1:
                ax.set_title(f"{sw['name']}: best competitor of the target", fontsize=9)
                ax2 = ax.twinx()
                m = np.asarray(sw["margin_mean"], dtype=float)
                ax2.plot(x, m, "-o", color="k", ms=4, lw=1.4, label="mean margin (all probes)")
                if sw.get("margin_by_competitor"):
                    for cls, col, _ in comp_cls:
                        mv = [d.get(cls, np.nan) for d in sw["margin_by_competitor"]]
                        ax2.plot(x, mv, ":", marker=".", color=col, lw=1.0, alpha=0.9,
                                 label=f"margin when runner-up is {cls.replace('_', ' ')}")
                ax2.axhline(0.0, color="k", lw=0.7, ls="--")
                lo = min(-0.05, float(np.nanmin(m)) - 0.05)
                ax2.set_ylim(lo, max(0.5, float(np.nanmax(m)) + 0.1))
                ax2.set_ylabel("margin = cos(target) − cos(runner-up)", fontsize=8)
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper center",
                          bbox_to_anchor=(0.5, -0.12), ncol=3, framealpha=0.9)
            if c == 0:
                ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                          ncol=4, framealpha=0.9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_usage_protocol(
    cond: Dict[str, Any],
    path: str,
    title: str = "Retention against future use",
) -> None:
    """Four panels for the usage-rate protocol (seed-aggregated inputs).

    ``cond`` carries, per condition name (``frequency``, ``disuse``,
    ``recency``), the per-seed ``usage_phase`` outputs. Panels:

    (a) frequency under pressure -- margin trajectory per use-rate level,
        mean over seeds (tasks are grouped by the rate they were assigned,
        since the assignment is permuted per seed).
    (b) the same ladder with exactly orthogonal tasks: the disuse control. A
        flat never-used line here is "no passive forgetting".
    (c) final margin against realised presentations, every seed x task as a
        point, with the Spearman rho (mean +- sd over seeds, and pooled).
    (d) recency -- equal counts, early vs late placement: margin trajectory of
        the two groups.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (a, b), (c, d) = axes

    def _by_rate(runs):
        """{rate: mean trajectory over (seeds x tasks at that rate)}"""
        out = {}
        for r in runs:
            M = np.asarray(r["margins"], dtype=float)
            for i, rate in enumerate(r["rates"]):
                out.setdefault(float(rate), []).append(M[:, i])
        return {k: np.mean(np.array(v), axis=0) for k, v in sorted(out.items(), reverse=True)}, \
               {k: np.std(np.array(v), axis=0) for k, v in sorted(out.items(), reverse=True)}

    cmap = plt.get_cmap("viridis")
    for ax, key, ttl in ((a, "frequency", "frequency ladder under pressure"),
                         (b, "disuse", "same ladder, tasks exactly orthogonal (disuse control)")):
        runs = cond.get(key)
        if not runs:
            ax.axis("off")
            continue
        mu, sd = _by_rate(runs)
        rates = list(mu.keys())
        x = np.arange(len(next(iter(mu.values()))))
        for k, rate in enumerate(rates):
            col = cmap(1.0 - k / max(1, len(rates) - 1))
            lab = "never re-presented" if rate <= 0 else f"p = {rate:g} / block"
            ax.plot(x, mu[rate], "-o", ms=3, lw=1.5, color=col, label=lab)
            ax.fill_between(x, mu[rate] - sd[rate], mu[rate] + sd[rate], color=col, alpha=0.12)
        ax.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("block of the usage stream (0 = after the chain)")
        ax.set_ylabel("margin = cos(target) − cos(best competitor)")
        ax.set_title(f"{ttl}\n{runs[0].get('condition_note', '')}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")

    runs = cond.get("frequency") or []
    if runs:
        rhos = np.array([r["spearman_margin_vs_use"] for r in runs], dtype=float)
        xs, ys = [], []
        for r in runs:
            cnt = np.asarray(r["counts"], dtype=float)
            fin = np.asarray(r["final_margin"], dtype=float)
            xs.extend(cnt); ys.extend(fin)
            c.scatter(cnt + np.random.default_rng(0).normal(0, 0.12, len(cnt)), fin,
                      s=16, alpha=0.5, color="tab:blue")
        xs, ys = np.array(xs), np.array(ys)
        for u in np.unique(xs):
            c.plot(u, ys[xs == u].mean(), "_", color="k", ms=18, mew=2)
        from memval.benchmarks.continual_chain import spearman
        pooled = spearman(ys, xs)
        c.axhline(0.0, color="k", ls="--", lw=0.8)
        c.set_xlabel("realised presentations in the stream")
        c.set_ylabel("final margin")
        c.set_title(f"final margin vs use  —  Spearman ρ = {np.nanmean(rhos):.2f} ± "
                    f"{np.nanstd(rhos):.2f} per seed, {pooled:.2f} pooled\n"
                    f"misallocation = {np.nanmean([r['misallocation'] for r in runs]):+.3f}  "
                    f"(never-used − most-used margin; > 0 = over-stable)", fontsize=9)
        c.grid(alpha=0.25)
    else:
        c.axis("off")

    runs = cond.get("recency") or []
    if runs:
        early = np.mean([np.asarray(r["margins"])[:, r["early_tasks"]].mean(axis=1) for r in runs], axis=0)
        late = np.mean([np.asarray(r["margins"])[:, r["late_tasks"]].mean(axis=1) for r in runs], axis=0)
        x = np.arange(len(early))
        d.plot(x, early, "-o", ms=3, color="tab:orange", label="early group (presented in first third)")
        d.plot(x, late, "-s", ms=3, color="tab:green", label="late group (presented in last third)")
        d.axhline(0.0, color="k", ls="--", lw=0.8)
        diff = float(late[-1] - early[-1])
        d.set_title(f"recency: equal counts, different placement  —  "
                    f"final late − early = {diff:+.3f}\n{runs[0].get('condition_note', '')}", fontsize=9)
        d.set_xlabel("block of the usage stream")
        d.set_ylabel("mean margin of the group")
        d.grid(alpha=0.25)
        d.legend(fontsize=7, loc="lower left")
    else:
        d.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
