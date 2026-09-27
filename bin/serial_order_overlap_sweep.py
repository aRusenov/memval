#!/usr/bin/env python
"""Serial order across three blocked lists, along the two overlap dials.

Three lists A, B, C of ``--list-len`` items, one category each, are trained
blocked (A, then B, then C) on one model. Each category also holds
``--n-lures`` items that are never studied, so a cued answer can land in four
places, exactly as in ``bin/probe_serial_order.py``'s multi-list condition:

  correct                 the true next item
  transposition           an item of the probed list, wrong position
  prior-list intrusion    an item of an EARLIER studied list (proactive)
  later-list intrusion    an item of a LATER studied list (retroactive)
  extra-list intrusion    a never-studied lure

Two dials, the same two the continual-chain sweep uses
(``bin/continual_chain_overlap_sweep.py``), each moved with the other held:

  within   within-list cosine rho = 1/(1 + d sigma^2), rungs set by TARGET rho
           and inverted to sigma (docs/sections/encoder_design.md: picking sigma by feel
           gives a ladder whose last rungs are the same stimulus). Lists held
           exactly orthogonal (rho_b = 0).
  between  prototype cosine rho_b, within held at sigma = 0.2 (rho = 0.20, the
           probe's material). Word-level between-list cosine is rho * rho_b, so
           rho_b -> 1 makes the lists as alike as the items inside one list --
           the full range the geometry allows at that within cosine.

The (rho = 0.2, rho_b = 0) stimulus appears on both dials; the two rungs must
agree, and the script prints both so that can be checked.

**Exposure is criterion-referenced on CUED recall** (``--train-to cued``, the
default): each list is trained one epoch at a time until its cued recall reaches
criterion, capped at ``--budget``, and the rollout is then read at that point --
"the associations are present", the antecedent of the paragraph-11 clause.
``--train-to both`` also waits for the rollout to reach criterion. It is NOT the
default because the rollout criterion is censored by the item draw alone: on
seed 42 at the easiest rung, list B cannot unroll even on a fresh model (step 6
jumps to B09 for all 512 epochs), and the censored list then trains for the
whole cap and overwrites the earlier lists' unroll (A: 9 -> 1). Epochs to each
criterion are recorded per list either way; a list that hits the cap on its
training criterion is marked censored.

Read-outs, per rung, per encoder+model seed (seeds move the geometry; for a
zero-init arm they are the only replication there is):

  recall length   steps a free rollout gets right before its first error
                  (max list_len - 1), for every list i after every stage j:
                  a lower-triangular 3x3 matrix, like the retention matrix
  cued outcomes   the five-way breakdown above, per list, after every stage

Probe is the suite's clean-single protocol (``sequence_length``): one clean
cue for a deterministic arm, repeats only for a stochastic forward pass.

Writes, beside the arm's symbolic plots:

  serial_order_overlap_recall_length.png   one 3x3 matrix per rung, one row per dial
  serial_order_overlap_failures.png        final cued outcome per list, per rung
  serial_order_overlap_sweep.json          everything, per seed

    python bin/serial_order_overlap_sweep.py --model ahn --out results/zoo_capacity_run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

import probe_serial_order as P  # noqa: E402
from continual_chain_overlap_sweep import build_model  # noqa: E402
from run_benchmark import MODEL_REGISTRY, resolve_model_name  # noqa: E402

from memval.benchmarks.probe import resolve_probe  # noqa: E402
from memval.encoders.symbolic import SymbolicDecoder, SymbolicEncoder  # noqa: E402

EMBED_DIM = 100
LIST_NAMES = "ABCDEFGH"
KINDS = ("correct", "transposition", "prior_list_intrusion", "later_list_intrusion",
         "extra_list_intrusion")
COLORS = {"correct": "tab:blue", "transposition": "tab:orange",
          "prior_list_intrusion": "#b39ddb", "later_list_intrusion": "tab:purple",
          "extra_list_intrusion": "tab:red"}


def _floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def sigma_for(rho: float, d: int = EMBED_DIM) -> float:
    """Invert rho = 1/(1 + d sigma^2) (docs/sections/encoder_design.md)."""
    return float(np.sqrt((1.0 - rho) / (rho * d)))


def build_material(n_lists, list_len, n_lures, sigma, rho_b, seed):
    """Lists A, B, C... each the first list_len items of its own category; the
    remaining n_lures items of each category are never studied. The mapping is
    built from lists, so the encoder's per-word draws are order-stable."""
    vocab: Dict[str, str] = {}
    lists = []
    for k in range(n_lists):
        cat = LIST_NAMES[k]
        words = [f"{cat}{i:02d}" for i in range(list_len + n_lures)]
        vocab.update({w: cat for w in words})
        lists.append(words[:list_len])
    enc = SymbolicEncoder(vocab, embedding_dim=EMBED_DIM, category_variance=sigma,
                          seed=seed, between_category_cosine=rho_b)
    E = enc.embeddings
    cat_of = np.array([vocab[w] for w in enc.idx_to_word])
    G = E @ E.T
    same = cat_of[:, None] == cat_of[None, :]
    off = ~np.eye(len(cat_of), dtype=bool)
    geo = dict(within_cos=float(G[same & off].mean()), between_cos=float(G[~same].mean()),
               within_cos_law=1.0 / (1.0 + EMBED_DIM * sigma ** 2),
               between_cos_law=(rho_b or 0.0) / (1.0 + EMBED_DIM * sigma ** 2))
    return lists, enc, SymbolicDecoder(enc), geo


def recall_length(model, words, enc, dec, n_trials, noise_scale, rng):
    """Mean number of steps a free (raw-feedback) rollout gets right before its
    first error, cued with the list's first item. list_len - 1 = never failed."""
    lengths = []
    for _ in range(n_trials):
        model.reset_context()
        cur = enc.encode([words[0]])[0] + rng.normal(0, noise_scale, enc.embedding_dim)
        n_ok = 0
        for i in range(len(words) - 1):
            model.current_t = i
            pred = model.predict_next(cur, current_context=np.array([1.0]))
            if dec.decode(pred, top_k=1)[0] != words[i + 1]:
                break
            n_ok += 1
            cur = pred
        lengths.append(n_ok)
    return float(np.mean(lengths))


def cued_outcomes(model, lists, k, n_trained, enc, dec, n_trials, noise_scale, rng):
    """Five-way cued breakdown for list k, with lists [0, n_trained) studied."""
    others = [(j, lists[j]) for j in range(n_trained) if j != k]
    model.reset_context()
    hits, oc, disps, src = P._cued(model, lists[k], enc, dec, n_trials, noise_scale, rng,
                                   others)
    out = Counter(correct=oc["correct"], transposition=oc["transposition"],
                  extra_list_intrusion=oc["extra_list_intrusion"])
    for j, v in src.items():
        out["prior_list_intrusion" if j < k else "later_list_intrusion"] += v
    return ({kd: int(out.get(kd, 0)) for kd in KINDS},
            float(np.mean(hits[1:])),
            {str(d): int(v) for d, v in sorted(disps.items())})


def run_rung(model_class, default_kwargs, *, sigma, rho_b, seed, n_lists, list_len,
             n_lures, n_trials, noise_scale, criterion, budget, train_to):
    lists, enc, dec, geo = build_material(n_lists, list_len, n_lures, sigma, rho_b, seed)
    np.random.seed(0)
    model = build_model(model_class, default_kwargs, enc, 1, seed)
    rng = np.random.default_rng(11)
    T = n_lists
    L = np.full((T, T), np.nan)                 # recall length of list i after stage j
    outcomes = [[None] * T for _ in range(T)]    # cued breakdown of list i after stage j
    cued_mrr = np.full((T, T), np.nan)
    disps = [[None] * T for _ in range(T)]
    exposure = []
    for j in range(T):
        X = enc.encode(lists[j])
        e_cued = e_roll = None
        ep = 0
        while ep < budget and (e_cued is None or (train_to == "both" and e_roll is None)):
            P._fit_one_epoch(model, X)
            ep += 1
            _, mrr, _ = cued_outcomes(model, lists, j, j + 1, enc, dec, n_trials,
                                      noise_scale, rng)
            if e_cued is None and mrr >= criterion:
                e_cued = ep
            if e_roll is None and recall_length(model, lists[j], enc, dec, n_trials,
                                                noise_scale, rng) >= criterion * (list_len - 1):
                e_roll = ep
        exposure.append(dict(e_cued=e_cued, e_rollout=e_roll, epochs=ep,
                             censored=e_cued is None or (train_to == "both" and e_roll is None)))
        for i in range(j + 1):
            L[j, i] = recall_length(model, lists[i], enc, dec, n_trials, noise_scale, rng)
            outcomes[j][i], cued_mrr[j, i], disps[j][i] = cued_outcomes(
                model, lists, i, j + 1, enc, dec, n_trials, noise_scale, rng)
    return dict(seed=seed, geometry=geo, recall_length=L.tolist(), cued_mrr=cued_mrr.tolist(),
                outcomes=outcomes, displacements=disps, exposure=exposure)


def plot_recall_length(dials, path, title, max_len, task_labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rows, n_cols = len(dials), max(len(d["rungs"]) for d in dials)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.9 * n_cols + 0.8, 3.0 * n_rows + 0.6),
                             squeeze=False)
    im = None
    for r, d in enumerate(dials):
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(d["rungs"]):
                ax.set_visible(False)
                continue
            rung = d["rungs"][c]
            M = np.array(rung["recall_length_mean"], dtype=float)
            T = M.shape[0]
            im = ax.imshow(np.ma.masked_invalid(M), vmin=0, vmax=max_len, cmap="viridis")
            cens = rung["censored_any"]
            for j in range(T):
                for i in range(j + 1):
                    v = M[j, i]
                    ax.text(i, j, f"{v:.1f}", ha="center", va="center", fontsize=9,
                            color="white" if v < 0.6 * max_len else "black")
                    if cens[j]:
                        ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False,
                                                   hatch="///", ec="white", lw=0))
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(task_labels, fontsize=8)
            ax.set_yticklabels([f"after {t}" for t in task_labels], fontsize=8)
            ax.set_title(rung["label"], fontsize=8.5)
            if r == n_rows - 1:
                ax.set_xlabel("recalled list", fontsize=8)
        axes[r][0].annotate(d["name"], xy=(-0.55, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center", fontsize=9.5)
    fig.subplots_adjust(wspace=0.45, hspace=0.35)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.04)
        cb.set_label(f"steps recalled before first error (max {max_len})", fontsize=8.5)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_failures(dials, path, title, task_labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(dials), 1, figsize=(13, 3.4 * len(dials)), squeeze=False)
    T = len(task_labels)
    for r, d in enumerate(dials):
        ax = axes[r][0]
        xs, labels, centers = [], [], []
        for g, rung in enumerate(d["rungs"]):
            base = g * (T + 1.2)
            centers.append(base + (T - 1) / 2)
            for i in range(T):
                xs.append(base + i)
                labels.append(task_labels[i])
        bottom = np.zeros(len(xs))
        for kd in KINDS:
            frac = []
            for rung in d["rungs"]:
                for i in range(T):
                    oc = rung["final_outcomes"][i]
                    frac.append(oc[kd] / max(sum(oc.values()), 1))
            frac = np.array(frac)
            if frac.any() or kd == "correct":
                ax.bar(xs, frac, bottom=bottom, width=0.8, color=COLORS[kd],
                       label=kd.replace("_", " "))
            bottom += frac
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        for g, rung in enumerate(d["rungs"]):
            ax.text(centers[g], -0.16, rung["label"], ha="center", va="top", fontsize=8.5,
                    transform=ax.get_xaxis_transform())
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of cued probes")
        ax.set_title(d["name"].replace("\n", "  "), fontsize=9.5)
        ax.legend(fontsize=8, loc="lower left", ncol=5, framealpha=0.9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.55)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ahn", type=resolve_model_name, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--out", default="results/zoo_capacity_run",
                    help="run directory; outputs go to <out>/<ClassName>/symbolic/")
    ap.add_argument("--dials", default="within,between")
    ap.add_argument("--within-rungs", default="0.05,0.2,0.4,0.6,0.8",
                    help="target within-list cosines rho (inverted to sigma)")
    ap.add_argument("--between-rungs", default="0,0.25,0.5,0.75,0.95",
                    help="prototype cosines rho_b")
    ap.add_argument("--sigma", type=float, default=0.2,
                    help="within sigma held on the between dial (rho = 0.20)")
    ap.add_argument("--rho-b-fixed", type=float, default=0.0,
                    help="prototype cosine held on the within dial (0 = exactly orthogonal)")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--n-lists", type=int, default=3)
    ap.add_argument("--list-len", type=int, default=10)
    ap.add_argument("--n-lures", type=int, default=4,
                    help="never-studied items per category (extra-list intrusion targets)")
    ap.add_argument("--criterion", type=float, default=0.95)
    ap.add_argument("--train-to", choices=("cued", "both"), default="cued",
                    help="train each list until cued recall (default) or cued AND rollout "
                         "reach criterion; see the module docstring for why 'both' is not "
                         "the default")
    ap.add_argument("--budget", type=int, default=512, help="epoch cap per list")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="requested; the clean-single protocol resolves the real count")
    a = ap.parse_args()

    spec = MODEL_REGISTRY[a.model]
    model_class, default_kwargs = spec["class"], dict(spec["default_kwargs"])
    P.MODEL_NAME = a.model
    n_trials, noise_scale = resolve_probe("sequence_length", model_class, a.n_trials)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    labels = list(LIST_NAMES[:a.n_lists])
    max_len = a.list_len - 1
    common = dict(n_lists=a.n_lists, list_len=a.list_len, n_lures=a.n_lures,
                  n_trials=n_trials, noise_scale=noise_scale, criterion=a.criterion,
                  budget=a.budget, train_to=a.train_to)
    record: Dict[str, Any] = dict(model=a.model, class_name=model_class.__name__,
                                  seeds=seeds, probe=dict(n_trials=n_trials,
                                                          noise_scale=noise_scale),
                                  **{k: v for k, v in common.items()
                                     if k not in ("n_trials", "noise_scale")}, dials=[])
    t0 = time.time()
    for dial in [x.strip() for x in a.dials.split(",") if x.strip()]:
        if dial == "within":
            settings = [(sigma_for(rho), a.rho_b_fixed) for rho in _floats(a.within_rungs)]
            name = f"within-list cosine ρ\n(lists orthogonal, ρ_b = {a.rho_b_fixed:g})"
        elif dial == "between":
            settings = [(a.sigma, rb) for rb in _floats(a.between_rungs)]
            name = (f"between-list prototype cosine ρ_b\n(within ρ = "
                    f"{1 / (1 + EMBED_DIM * a.sigma ** 2):.2f})")
        else:
            raise SystemExit(f"unknown dial {dial!r}")
        rungs = []
        for sigma, rho_b in settings:
            runs = [run_rung(model_class, default_kwargs, sigma=sigma, rho_b=rho_b, seed=sd,
                             **common) for sd in seeds]
            geo = {k: float(np.mean([r["geometry"][k] for r in runs])) for k in runs[0]["geometry"]}
            if dial == "within":
                label = f"ρ = {geo['within_cos_law']:.2f}  (σ = {sigma:.3f})"
            else:
                label = f"ρ_b = {rho_b:.2f}  (between cos {geo['between_cos_law']:.2f})"
            Lm = np.nanmean([r["recall_length"] for r in runs], axis=0)
            final = [{kd: sum(r["outcomes"][a.n_lists - 1][i][kd] for r in runs) for kd in KINDS}
                     for i in range(a.n_lists)]
            cens = [any(r["exposure"][j]["censored"] for r in runs) for j in range(a.n_lists)]
            e_c = [float(np.median([r["exposure"][j]["e_cued"] or a.budget for r in runs]))
                   for j in range(a.n_lists)]
            e_r = [float(np.median([r["exposure"][j]["e_rollout"] or np.nan for r in runs]))
                   for j in range(a.n_lists)]
            rungs.append(dict(sigma=sigma, rho_b=rho_b, label=label, geometry=geo,
                              recall_length_mean=Lm.tolist(), final_outcomes=final,
                              censored_any=cens, e_cued_median=e_c, e_rollout_median=e_r,
                              per_seed=runs))
            fl = ", ".join(f"{labels[i]} {Lm[-1][i]:.1f}" for i in range(a.n_lists))
            tot = [max(sum(f.values()), 1) for f in final]
            br = "  ".join(f"{labels[i]}: " + " ".join(
                f"{final[i][kd] / tot[i]:.2f}" for kd in KINDS) for i in range(a.n_lists))
            print(f"[{dial}] {label:<36} final recall length {fl} | epochs cued {e_c} "
                  f"roll {e_r}{'  CENSORED' if any(cens) else ''}\n"
                  f"   cued (corr/trans/prior/later/extra)  {br}", flush=True)
        record["dials"].append(dict(dial=dial, name=name, rungs=rungs))

    out_dir = os.path.join(a.out, model_class.__name__, "symbolic")
    plots = os.path.join(out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    seeds_note = f"mean of {len(seeds)} seeds" if len(seeds) > 1 else f"seed {seeds[0]}"
    crit = "cued criterion" if a.train_to == "cued" else "cued + rollout criterion"
    head = (f"{model_class.__name__}, {a.n_lists} lists × {a.list_len} blocked, each trained "
            f"to {crit} ({a.criterion:g}, cap {a.budget}), {seeds_note}")
    p1 = os.path.join(plots, "serial_order_overlap_recall_length.png")
    plot_recall_length(record["dials"], p1,
                       f"Autoregressive recall length along the overlap dials — {head}; "
                       f"hatched = a seed hit the cap at that stage", max_len, labels)
    p2 = os.path.join(plots, "serial_order_overlap_failures.png")
    plot_failures(record["dials"], p2,
                  f"What a cued failure is, after the last list — {head}; "
                  f"{a.n_lures} never-studied lures per category", labels)
    pj = os.path.join(out_dir, "serial_order_overlap_sweep.json")
    with open(pj, "w") as f:
        json.dump(record, f, indent=1)
    print(f"\nwrote {p1}\n      {p2}\n      {pj}\n{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
