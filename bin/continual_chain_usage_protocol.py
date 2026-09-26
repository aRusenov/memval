#!/usr/bin/env python
"""Retention against future use: the environmental-statistics protocol.

"Forgetting is functional" needs a definition of useful that a next-step
predictor can be tested on. This one is Anderson & Schooler's: a memory's
utility is how often the world re-presents it, and an ideal allocator retains
in proportion to expected future use. No reward channel; every arm can be run.

Three conditions, each on a chain loaded under pressure unless stated:

  frequency  after the chain, a stream of K blocks in which task i is
             re-presented with probability p_i from a ladder
             (1, 1/2, 1/4, 1/8, 1/16, 0), assignment permuted per seed so chain
             position is counterbalanced. Readout: Spearman(final margin,
             realised count) and misallocation (never-used minus most-used).
  disuse     the identical ladder with the task prototypes exactly orthogonal
             (between_category_cosine = 0). Orthogonal prototypes are not
             orthogonal words -- private deviations overlap by chance at
             sd ~ 1/sqrt(d) -- so this is the residual-cross-talk floor, not a
             zero-interference control. An arm with no decay term loses the
             never-used trace here only through that residual; the gap to the
             pressure condition is what between-task overlap adds.
  recency    every task gets the same number of presentations; half the tasks
             get them in the first third of the stream, half in the last third.
             Readout: final margin, late minus early.

The stream is scored with the clean single-cue MARGIN (target minus best
competitor) after every block, because accuracy saturates on the frequently
used tasks and the ranking lives above that floor. Chain, material and probe
are the continual_chain section's own.

    python bin/continual_chain_usage_protocol.py --model hopfield --out results/ahn_capacity_run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

from run_benchmark import MODEL_REGISTRY  # noqa: E402

from memval.benchmarks._selection import resolve_epochs_default  # noqa: E402
from memval.benchmarks.continual_chain import (  # noqa: E402
    ContinualChainBenchmark, build_chain_material, usage_phase)
from memval.benchmarks.probe import resolve_probe  # noqa: E402
from memval.benchmarks.symbolic_pipeline import (  # noqa: E402
    _get_model_kwargs, load_vocab, mean_recall_rate, measure_recall_associative)
from memval.metrics.retention import plot_usage_protocol  # noqa: E402

EMBED_DIM = 100


def build_model(model_class, default_kwargs, encoder, epochs, seed):
    kw = _get_model_kwargs(default_kwargs, epochs)
    if "seed" in kw:
        kw["seed"] = seed
    if "encoder" in model_class.__init__.__code__.co_varnames:
        return model_class(encoder=encoder, n_features=encoder.embedding_dim, **kw)
    return model_class(n_features=encoder.embedding_dim, **kw)


def load_chain(model_class, default_kwargs, vocab, *, n_tasks, seq_len, sigma, rho_b, seed,
               n_trials, noise_scale, epochs):
    """Run the chain (criterion mode, no rehearsal) and hand back the loaded model."""
    material = build_chain_material(vocab, n_tasks, seq_len, embedding_dim=EMBED_DIM,
                                    category_variance=sigma, between_category_cosine=rho_b,
                                    seed=seed)
    enc, dec, seqs = material["encoder"], material["decoder"], material["sequences"]
    names = list(seqs)
    model = build_model(model_class, default_kwargs, enc, epochs, seed)

    def score_task(i):
        model.reset_context()
        return mean_recall_rate(measure_recall_associative(
            model, seqs[names[i]], enc, dec, n_trials=n_trials, noise_scale=noise_scale))

    out = ContinualChainBenchmark(n_trials=n_trials, noise_scale=noise_scale, rehearse=False).evaluate(
        model=model, datasets={"sequences": seqs, "embeddings": material["embeddings"],
                               "score_fn": score_task},
        epochs=epochs, chance_level=1.0 / len(material["vocab"]))
    return model, material, score_task, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="hopfield", choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--out", default="results/usage_protocol")
    ap.add_argument("--rho-b", type=float, default=0.75,
                    help="between-task cosine for the pressure conditions (0.75 forgets 0.18 on AHN)")
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--rates", default="1,0.5,0.25,0.125,0.0625,0",
                    help="per-block re-presentation probabilities, one per task")
    ap.add_argument("--n-blocks", type=int, default=16)
    ap.add_argument("--recency-count", type=int, default=4,
                    help="presentations per task in the recency condition")
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--n-tasks", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=5)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    spec = MODEL_REGISTRY[args.model]
    model_class, default_kwargs = spec["class"], dict(spec["default_kwargs"])
    class_name = model_class.__name__
    epochs = int(resolve_epochs_default(args.epochs, default_kwargs, 300))
    n_trials, noise_scale = resolve_probe("continual_chain", model_class, args.n_trials)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rates = np.array([float(x) for x in args.rates.split(",")], dtype=float)
    if len(rates) != args.n_tasks:
        raise SystemExit(f"--rates needs {args.n_tasks} entries, got {len(rates)}")
    vocab = load_vocab()
    plots_dir = os.path.join(args.out, class_name, "symbolic", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    common = dict(n_tasks=args.n_tasks, seq_len=args.seq_len, sigma=args.sigma,
                  n_trials=n_trials, noise_scale=noise_scale, epochs=epochs)
    t0 = time.time()
    cond: Dict[str, List[Dict[str, Any]]] = {"frequency": [], "disuse": [], "recency": []}

    for sd in seeds:
        rng = np.random.default_rng(1000 + sd)
        perm = rng.permutation(args.n_tasks)          # counterbalance rate vs chain position
        assigned = rates[perm]

        # --- frequency, under pressure --------------------------------------
        model, mat, score, chain = load_chain(model_class, default_kwargs, vocab,
                                              rho_b=args.rho_b, seed=sd, **common)
        r = usage_phase(model, mat["sequences"], mat["embeddings"], mat["encoder"],
                        rates=assigned, n_blocks=args.n_blocks,
                        rng=np.random.default_rng(2000 + sd), score_fn=score)
        r["seed"] = sd
        r["chain_avg_forgetting"] = chain["metrics"]["chain_avg_forgetting"]
        r["condition_note"] = (f"ρ_b = {args.rho_b:g}, chain forgetting "
                               f"{chain['metrics']['chain_avg_forgetting']:.2f}")
        cond["frequency"].append(r)

        # --- disuse control: same ladder, orthogonal tasks -------------------
        model, mat, score, chain = load_chain(model_class, default_kwargs, vocab,
                                              rho_b=0.0, seed=sd, **common)
        r = usage_phase(model, mat["sequences"], mat["embeddings"], mat["encoder"],
                        rates=assigned, n_blocks=args.n_blocks,
                        rng=np.random.default_rng(2000 + sd), score_fn=score)
        r["seed"] = sd
        r["condition_note"] = ("ρ_b = 0 exactly: prototypes orthogonal; only chance word-level "
                               "cross-talk (sd ≈ 1/√d) reaches the unused task")
        cond["disuse"].append(r)

        # --- recency: equal counts, early vs late ----------------------------
        model, mat, score, chain = load_chain(model_class, default_kwargs, vocab,
                                              rho_b=args.rho_b, seed=sd, **common)
        K, T, m = args.n_blocks, args.n_tasks, args.recency_count
        grp = rng.permutation(T)
        early_tasks, late_tasks = sorted(grp[: T // 2].tolist()), sorted(grp[T // 2:].tolist())
        sched = np.zeros((K, T), dtype=bool)
        third = max(m, K // 3)
        for i in early_tasks:
            sched[rng.choice(np.arange(0, third), size=m, replace=False), i] = True
        for i in late_tasks:
            sched[rng.choice(np.arange(K - third, K), size=m, replace=False), i] = True
        r = usage_phase(model, mat["sequences"], mat["embeddings"], mat["encoder"],
                        schedule=sched, rng=np.random.default_rng(3000 + sd), score_fn=score)
        r["seed"] = sd
        r["early_tasks"], r["late_tasks"] = early_tasks, late_tasks
        r["late_minus_early"] = float(np.mean([r["final_margin"][i] for i in late_tasks])
                                      - np.mean([r["final_margin"][i] for i in early_tasks]))
        r["condition_note"] = f"ρ_b = {args.rho_b:g}, {m} presentations per task"
        cond["recency"].append(r)

    # --- summary ----------------------------------------------------------
    def _ms(vals):
        v = np.array(vals, dtype=float)
        return f"{np.nanmean(v):+.3f} ± {np.nanstd(v):.3f}"

    print(f"\n{class_name}, {len(seeds)} seeds, {args.n_blocks} blocks")
    for key in ("frequency", "disuse"):
        rs = cond[key]
        print(f"  {key:<9} Spearman(margin, use) {_ms([r['spearman_margin_vs_use'] for r in rs])}   "
              f"misallocation {_ms([r['misallocation'] for r in rs])}   "
              f"used-below-criterion {np.mean([r['used_below_criterion'] for r in rs]):.2f}")
        M = np.array([r["margins"] for r in rs], dtype=float)
        by_rate = {}
        for r, Mr in zip(rs, M):
            for i, rate in enumerate(r["rates"]):
                by_rate.setdefault(rate, []).append((Mr[0, i], Mr[-1, i]))
        for rate in sorted(by_rate, reverse=True):
            v = np.array(by_rate[rate])
            print(f"      p={rate:<7g} margin before {v[:,0].mean():+.3f} -> after {v[:,1].mean():+.3f}")
    print(f"  recency   late − early final margin {_ms([r['late_minus_early'] for r in cond['recency']])}")

    png = os.path.join(plots_dir, "continual_chain_usage.png")
    plot_usage_protocol(cond, png,
                        title=f"Retention against future use ({class_name}, {len(seeds)} seeds, "
                              f"{args.n_blocks}-block stream)")
    pj = os.path.join(args.out, class_name, "symbolic", "continual_chain_usage.json")
    with open(pj, "w") as f:
        json.dump({"model": args.model, "class": class_name, "args": vars(args),
                   "conditions": cond}, f, indent=1)
    print(f"\nwrote {png}\n      {pj}\n{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
