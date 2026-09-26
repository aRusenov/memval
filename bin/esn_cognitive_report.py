#!/usr/bin/env python
"""Three-panel L0 report for DTS-ESN: serial position, list length, presentation
rate -- pushed in list length until forgetting appears, each panel with the
human expectation drawn dotted for contrast.

    python bin/esn_cognitive_report.py                      # full report
    python bin/esn_cognitive_report.py --probe-only         # where does recall fall?

Material: L independent random unit vectors (no category structure), decoded
against the full L_max-word candidate set at every length so chance is
constant. One pass over the list (the human one-study regime) for panels A and
B; panel C sweeps passes at the length where forgetting first appears.

Two probes, because the arm is stateful. Training streams the list and the
readout learns to map the state AFTER items 0..t-1 onto item t. The matched
("warm") probe reproduces that: prime with items 0..t-2 via ``observe``, then
``predict_next`` on item t-1. The suite's default ("cold") probe rebuilds the
state from rest on the cue alone, which is the house independent-probe
contract but not the state the readout was trained against. Warm is primary
here; cold is reported for the gap.

Bands are a fixed count of items (first / last 3) rather than a fraction, since
primacy and recency in the human data are about the first and last few items
whatever the list length.

The human curves are SCHEMATICS after Kahana (2020) Fig. 2a/2b and sec 4.1
(Murdock 1962): direction and ordering on an unlabelled probability axis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from memval.encoders.symbolic import SymbolicEncoder  # noqa: E402
from memval.models.baselines import DTSESNSequenceNetwork  # noqa: E402

ARM_KW = dict(n_units=400, tau_min=0.1, tau_max=20.0, spectral_radius=0.9, dt=0.05,
              predict_timing=True, n_epochs=1)
DIM = 100
BAND = 3   # items per end band


def make_material(L_max: int, seed: int):
    names = [f"w{i:04d}" for i in range(L_max)]
    enc = SymbolicEncoder(names, embedding_dim=DIM, seed=seed)
    E = np.asarray(enc.embeddings, dtype=float)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return names, E


def make_model(seed: int):
    return DTSESNSequenceNetwork(n_features=DIM, seed=seed, **ARM_KW)


def probe(model, X: np.ndarray, E: np.ndarray, warm: bool):
    """Per-position hit and margin (position 0 = cue, NaN)."""
    L = X.shape[0]
    hit = np.full(L, np.nan)
    margin = np.full(L, np.nan)
    model.reset_context()
    for t in range(1, L):
        pred = np.asarray(model.predict_next(X[t - 1].copy()), dtype=float).ravel()
        pn = np.linalg.norm(pred)
        sims = E @ (pred / pn) if pn > 1e-12 else np.zeros(E.shape[0])
        others = np.delete(np.arange(E.shape[0]), t)
        best = sims[others].max()
        hit[t] = float(int(np.argmax(sims)) == t) if pn > 1e-12 else 0.0
        margin[t] = float(sims[t] - best)
        if warm:
            model.observe(X[t - 1].copy())
    return hit, margin


def run_cell(L: int, seed: int, passes: int, E: np.ndarray) -> Dict[str, Any]:
    model = make_model(seed)
    X = E[:L]
    model.fit_sequence(X, epochs=passes)
    hw, mw = probe(model, X, E, warm=True)
    hc, mc = probe(model, X, E, warm=False)
    return {"hit_warm": hw, "margin_warm": mw, "hit_cold": hc, "margin_cold": mc}


def bands(curve: np.ndarray, k: int = BAND) -> Dict[str, float]:
    c = np.asarray(curve, dtype=float)[1:]
    k = min(k, max(1, len(c) // 3))
    return {"early": float(np.nanmean(c[:k])), "middle": float(np.nanmean(c[k:-k])) if len(c) > 2 * k else float("nan"),
            "late": float(np.nanmean(c[-k:]))}


def deciles(curve: np.ndarray, n: int = 10) -> List[float]:
    c = np.asarray(curve, dtype=float)[1:]
    edges = np.linspace(0, len(c), n + 1).astype(int)
    return [float(np.nanmean(c[a:b])) if b > a else float("nan") for a, b in zip(edges[:-1], edges[1:])]


def ci(vals: Sequence[float]):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return float(v.mean()) if v.size else float("nan"), float("nan")
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(v.size))


# ---------------------------------------------------------------- schematics --

HUMAN = {
    "spc_deciles": [0.95, 0.86, 0.76, 0.66, 0.58, 0.53, 0.50, 0.50, 0.53, 0.58],  # serial recall
    "ll": {"early": (0.90, 0.45), "middle": (0.70, 0.30), "late": (0.60, 0.60)},   # (L_min, L_max)
    "pr": {"early": (0.50, 0.85), "middle": (0.35, 0.65), "late": (0.70, 0.70)},   # (1 pass, max)
}
COL = {"early": "tab:blue", "middle": "tab:grey", "late": "tab:red"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320, 640])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--passes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--fall-below", type=float, default=0.9,
                    help="L* is the smallest length whose warm one-pass recall is below this")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--out", default="results/cognitive_l0/DTSESNSequenceNetwork/symbolic")
    args = ap.parse_args()

    lengths = sorted(args.lengths)
    L_max = lengths[-1]
    t0 = time.time()

    # ---- panels A/B: one pass, every length, every seed ----------------------
    grid: Dict[int, List[Dict[str, Any]]] = {L: [] for L in lengths}
    for s in range(args.seeds):
        _, E = make_material(L_max, seed=100 + s)
        for L in lengths:
            r = run_cell(L, seed=100 + s, passes=1, E=E)
            grid[L].append(r)
        print(f"  seed {s}: " + "  ".join(
            f"L={L}: warm {np.nanmean(grid[L][-1]['hit_warm'][1:]):.2f}/cold "
            f"{np.nanmean(grid[L][-1]['hit_cold'][1:]):.2f}" for L in lengths)
            + f"   [{time.time() - t0:.0f}s]")

    mean_warm = {L: float(np.mean([np.nanmean(r["hit_warm"][1:]) for r in grid[L]])) for L in lengths}
    mean_cold = {L: float(np.mean([np.nanmean(r["hit_cold"][1:]) for r in grid[L]])) for L in lengths}
    print("\nmean one-pass recall by length:")
    for L in lengths:
        print(f"  L={L:5d}  warm {mean_warm[L]:.3f}   cold {mean_cold[L]:.3f}")
    if args.probe_only:
        return

    below = [L for L in lengths if mean_warm[L] < args.fall_below]
    L_star = below[0] if below else L_max
    forgot = bool(below)
    print(f"\nL* = {L_star} ({'forgetting appears' if forgot else 'NO forgetting inside the grid'})")

    # ---- panel C: passes sweep at L* ----------------------------------------
    rate: Dict[int, List[Dict[str, Any]]] = {p: [] for p in args.passes}
    for s in range(args.seeds):
        _, E = make_material(L_max, seed=100 + s)
        for p in args.passes:
            rate[p].append(run_cell(L_star, seed=100 + s, passes=p, E=E))
        print(f"  rate seed {s}: " + "  ".join(
            f"{p}p {np.nanmean(rate[p][-1]['hit_warm'][1:]):.2f}" for p in args.passes)
            + f"   [{time.time() - t0:.0f}s]")

    # ---- numbers -------------------------------------------------------------
    A = {probe_: {"deciles": np.array([deciles(r[f"hit_{probe_}"]) for r in grid[L_star]])}
         for probe_ in ("warm", "cold")}
    B = {b: [ci([bands(r["hit_warm"])[b] for r in grid[L]]) for L in lengths]
         for b in ("early", "middle", "late")}
    C = {b: [ci([bands(r["hit_warm"])[b] for r in rate[p]]) for p in args.passes]
         for b in ("early", "middle", "late")}
    out = {
        "arm": "DTSESNSequenceNetwork", "arm_kwargs": ARM_KW, "dim": DIM, "band_items": BAND,
        "lengths": lengths, "seeds": args.seeds, "passes": args.passes,
        "L_star": L_star, "forgetting_inside_grid": forgot, "fall_below": args.fall_below,
        "mean_recall_warm": mean_warm, "mean_recall_cold": mean_cold,
        "A_spc_deciles": {k: {"mean": A[k]["deciles"].mean(0).tolist(),
                              "ci": (1.96 * A[k]["deciles"].std(0, ddof=1) / np.sqrt(args.seeds)).tolist()}
                          for k in A},
        "B_list_length": {b: {"mean": [m for m, _ in v], "ci": [c for _, c in v]} for b, v in B.items()},
        "C_presentation_rate": {b: {"mean": [m for m, _ in v], "ci": [c for _, c in v]} for b, v in C.items()},
        "human_schematic": HUMAN,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "esn_capacity_report.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---- figure --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # A. serial position at L*
    x = np.arange(1, 11)
    for k, col, lab in (("warm", "k", "DTS-ESN, training-matched probe"),
                        ("cold", "0.55", "DTS-ESN, cold cue-only probe")):
        m = out["A_spc_deciles"][k]["mean"]; e = out["A_spc_deciles"][k]["ci"]
        ax[0].errorbar(x, m, yerr=e, fmt="-o", ms=4, color=col, lw=1.6 if k == "warm" else 1.0,
                       capsize=2, label=lab)
    ax[0].plot(x, HUMAN["spc_deciles"], ":", color="tab:green", lw=2, label="human serial recall (schematic)")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{d * 10}%" for d in range(1, 11)], fontsize=8)
    ax[0].set_xlabel("position in list (decile)"); ax[0].set_ylabel("recall probability")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_title(f"A  Serial position, L = {L_star}, one pass", fontsize=10)
    ax[0].legend(fontsize=7, loc="lower left")

    # B. list length
    lx = np.log2(lengths)
    for b in ("early", "middle", "late"):
        m = out["B_list_length"][b]["mean"]; e = out["B_list_length"][b]["ci"]
        ax[1].errorbar(lx, m, yerr=e, fmt="-o", ms=4, color=COL[b], capsize=2, label=f"{b} items")
        h0, h1 = HUMAN["ll"][b]
        ax[1].plot([lx[0], lx[-1]], [h0, h1], ":", color=COL[b], lw=2, alpha=0.9)
    ax[1].plot([], [], ":", color="k", lw=2, label="human (schematic, dotted)")
    ax[1].set_xticks(lx); ax[1].set_xticklabels([str(L) for L in lengths], fontsize=8)
    ax[1].set_xlabel("list length"); ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("B  List length, one pass: first/middle/last 3 items", fontsize=10)
    ax[1].legend(fontsize=7, loc="lower left")

    # C. presentation rate at L*
    px = np.log2(args.passes)
    for b in ("early", "middle", "late"):
        m = out["C_presentation_rate"][b]["mean"]; e = out["C_presentation_rate"][b]["ci"]
        ax[2].errorbar(px, m, yerr=e, fmt="-o", ms=4, color=COL[b], capsize=2, label=f"{b} items")
        h0, h1 = HUMAN["pr"][b]
        ax[2].plot([px[0], px[-1]], [h0, h1], ":", color=COL[b], lw=2, alpha=0.9)
    ax[2].plot([], [], ":", color="k", lw=2, label="human (schematic, dotted)")
    ax[2].set_xticks(px); ax[2].set_xticklabels([str(p) for p in args.passes], fontsize=8)
    ax[2].set_xlabel("passes over the list (study time)"); ax[2].set_ylim(-0.02, 1.02)
    ax[2].set_title(f"C  Presentation rate, L = {L_star}", fontsize=10)
    ax[2].legend(fontsize=7, loc="lower right")

    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"DTS-ESN (400 units, RLS readout), {args.seeds} material seeds, "
                 f"random-vector lists decoded over {L_max} candidates; dotted = human expectation",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.out, "plots", "esn_capacity_report.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
