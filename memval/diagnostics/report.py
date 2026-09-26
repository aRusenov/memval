"""
Paired behavioural + geometric + weight report for a two-task (A then B)
forgetting protocol — the unified entry point.

`ab_forgetting_report` trains an untrained model on list a, then list b (no
reset), and assembles every tier the model supports into one structured dict:

  behavioural       : base_a (a after a), retain_a (a after b), b_learn
  geometry          : per-layer Jaccard A-vs-B (at init)
  interference_init : per-group update-cosine A-vs-B block (at init)
  fisher_attribution: per-group second-order forgetting attribution (a's Fisher
                      at post-a weights x the a->b displacement)

Capability-gated: a model exposing only the core interface still returns the
behavioural block. `plot_report` renders the paired panel (behaviour beside the
geometric/weight views) — the whole point being that the behavioural curve is
only interpretable next to where, in the network, it fails.
"""

import numpy as np

from .behavioral import transitions_of, one_step_accuracy, one_step_fidelity
from .geometry import representation_overlap
from .weights import (update_interference, fisher_diagonal, snapshot_parameters,
                      fisher_attribution)
from .protocols import supports_representations, supports_gradients

__all__ = ["ab_forgetting_report", "plot_report"]


def _vec_transitions(encoder, seq):
    return [(encoder.encode([x])[0], encoder.encode([y])[0]) for x, y in transitions_of(seq)]


def ab_forgetting_report(model, encoder, list_a, list_b, eps: float = 1e-6) -> dict:
    """Run the A->B protocol on a FRESH (untrained) model and collect all
    supported diagnostic tiers. Mutates `model` (it trains it)."""
    ta_sym, tb_sym = transitions_of(list_a), transitions_of(list_b)
    ta_vec, tb_vec = _vec_transitions(encoder, list_a), _vec_transitions(encoder, list_b)
    items_a = [encoder.encode([s])[0] for s in list_a]
    items_b = [encoder.encode([s])[0] for s in list_b]

    report = {
        "capabilities": {
            "representations": supports_representations(model),
            "gradients": supports_gradients(model),
        },
        "geometry": {},
        "interference_init": {},
        "fisher_attribution": {},
        "behavioral": {},
    }

    # --- init-time structural views (before any learning) ---
    report["geometry"] = representation_overlap(model, items_a, items_b, eps)
    report["interference_init"] = update_interference(model, ta_vec, tb_vec)

    # --- train list a, capture its importance ---
    model.fit_sequence(encoder.encode(list_a))
    report["behavioral"]["base_a"] = one_step_accuracy(model, encoder, ta_sym)
    report["behavioral"]["base_a_fid"] = one_step_fidelity(model, encoder, ta_sym)
    theta_A = F_A = None
    if supports_gradients(model):
        theta_A = snapshot_parameters(model)
        F_A = fisher_diagonal(model, ta_vec)

    # --- train list b on the same weights, measure forgetting ---
    model.fit_sequence(encoder.encode(list_b))
    report["behavioral"]["retain_a"] = one_step_accuracy(model, encoder, ta_sym)
    report["behavioral"]["retain_a_fid"] = one_step_fidelity(model, encoder, ta_sym)
    report["behavioral"]["b_learn"] = one_step_accuracy(model, encoder, tb_sym)
    if supports_gradients(model):
        theta_B = snapshot_parameters(model)
        report["fisher_attribution"] = fisher_attribution(F_A, theta_A, theta_B)

    return report


def format_report(report: dict) -> str:
    """Compact text rendering of a report dict."""
    b = report["behavioral"]
    lines = ["behavioural (one-step per-transition):",
             f"  base_a={b.get('base_a'):.2f}  retain_a={b.get('retain_a'):.2f}  "
             f"b_learn={b.get('b_learn'):.2f}"]
    if report["geometry"]:
        lines.append("geometry (A-vs-B Jaccard, at init):")
        for layer, g in report["geometry"].items():
            lines.append(f"  {layer:>7}: {g['jaccard_ab']:.3f}  "
                         f"(density {g['density_a']:.2f}, floor {g['floor']:.3f})")
    if report["interference_init"]:
        lines.append("update-cosine A<->B block (at init):")
        for g, v in report["interference_init"].items():
            lines.append(f"  {g:>5}: {v['ab']:.3f}")
    if report["fisher_attribution"]:
        lines.append("forgetting attribution (contrib% of F_A*displacement^2):")
        cs = report["fisher_attribution"]["contrib_share"]
        lines.append("  " + "  ".join(f"{g}={100*s:.1f}%" for g, s in cs.items()))
    return "\n".join(lines)


def plot_report(report: dict, path: str, title: str = "A→B forgetting diagnostics"):
    """Render the paired panel (behaviour + geometry + weight views) to `path`.
    Skips panels the model didn't support. Requires matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(title)

    # 1) behaviour
    b = report["behavioral"]
    names = ["base_a", "retain_a", "b_learn"]
    axes[0, 0].bar(names, [b.get(k, np.nan) for k in names],
                   color=["#4c72b0", "#c44e52", "#55a868"])
    axes[0, 0].set_ylim(0, 1); axes[0, 0].set_title("behaviour (one-step acc)")

    # 2) geometry: per-layer A-vs-B Jaccard vs floor
    ax = axes[0, 1]
    if report["geometry"]:
        layers = list(report["geometry"].keys())
        jac = [report["geometry"][l]["jaccard_ab"] for l in layers]
        flr = [report["geometry"][l]["floor"] for l in layers]
        xs = np.arange(len(layers))
        ax.bar(xs - 0.2, jac, 0.4, label="A↔B", color="#4c72b0")
        ax.bar(xs + 0.2, flr, 0.4, label="floor", color="#bbbbbb")
        ax.set_xticks(xs); ax.set_xticklabels(layers); ax.legend()
    ax.set_title("representation Jaccard (A vs B)")

    # 3) weight-update cosine A<->B block per group
    ax = axes[1, 0]
    if report["interference_init"]:
        gs = list(report["interference_init"].keys())
        ax.bar(gs, [report["interference_init"][g]["ab"] for g in gs], color="#8172b3")
    ax.set_title("update-cosine A↔B block")

    # 4) Fisher forgetting attribution (contrib share)
    ax = axes[1, 1]
    if report["fisher_attribution"]:
        cs = report["fisher_attribution"]["contrib_share"]
        gs = list(cs.keys())
        ax.bar(gs, [100 * cs[g] for g in gs], color="#c44e52")
        ax.set_ylabel("% of estimated forgetting")
    ax.set_title("Fisher forgetting attribution")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
