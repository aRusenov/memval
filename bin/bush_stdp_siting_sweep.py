#!/usr/bin/env python
"""Resolve the Bush arm's operating point at codec scale (spec S9.4, S9.5).

At D=64 / N=640 the arm reaches 1.00 one-step recall through the codec. At
D=156 / N=1560 on the hierarchical `(2, 12)` list every ridge point probed so far
floors. **Three things change together** between those configurations -- network
size, co-activity, and material overlap -- and no probe run so far separates
them. This one does.

Design
------
Everything is held at N=1560 (D=156, `n_per_feature=5`, `window_steps=50`).
Two materials, matched on *co-activity* so that overlap is the only difference:

* ``hierarchical`` -- the section's own 8-item across-branch list, mean abs
  cosine 0.21 between items.
* ``orthogonal``   -- 8 items with the same number of active dimensions, but
  disjoint, mean abs cosine 0.00.

If the orthogonal material works at N=1560 and the hierarchical one does not,
the limit is overlap. If neither works, the limit is scale, and D=64's success
was a size effect. That is the whole point of the pairing.

Sweep: ``g_syn`` x ``k_inh`` (along and across the stability ridge) at a fixed
exposure, then ``n_presentations`` at the best cell per material.

Ingestion is **streamed** (`fit_event`), which is how this arm is run --
`resample_per_pass` makes that byte-identical to the batch path, so the choice
costs nothing and matches the protocol it will be scored under.

Diagnostics, so a floor is attributable rather than merely observed:

* ``w_fwd`` / ``w_bwd`` -- mean weight between successive items' active
  populations. STDP's temporal asymmetry is supposed to produce ``w_fwd >>
  w_bwd``; if it does not, the arm never learned the sequence.
* ``spikes`` -- replay-window activity. Near zero means the chain died; near
  saturation means a synchronous burst with no order in it. The two failure
  modes sit on opposite sides of the ridge and are not distinguishable from the
  score alone.

Usage::

    python bin/bush_stdp_siting_sweep.py --json scratch/bush_siting.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.benchmarks.cue_masking import build_hierarchy, study_list
from memval.benchmarks.ingest import ingest
from memval.encoders.spike_codec import make_codec
from memval.models.baselines.bush_stdp import BushSTDPSpikingNetwork
from memval.models.codec_wrapper import wrap_with_codec

N_PER_FEATURE = 5
WINDOW_STEPS = 50
RECALL_STEPS = 12
# k_inh is a conductance (see BushSTDPSpikingNetwork): the working range sits
# roughly 15x below the current-based values it replaced, and shunting slows
# integration so the matching gains are correspondingly larger.
G_GRID = (160.0, 450.0, 550.0, 900.0, 1800.0)
K_GRID = (0.02, 0.05, 0.09, 0.15, 0.3)
P_GRID = (2, 5, 10, 20, 40)


def materials():
    """The section's list, and an overlap-free control matched on co-activity."""
    enc = build_hierarchy((2, 12), 6)
    words = study_list(enc, 8, "across")
    hier = enc.encode(words)
    D = hier.shape[1]
    # Same number of active dimensions per item, but disjoint across items.
    n_active = int(np.median((hier != 0).sum(axis=1)))
    ortho = np.zeros((len(hier), D))
    for i in range(len(hier)):
        cols = [(i * n_active + j) % D for j in range(n_active)]
        ortho[i, cols] = 1.0
    ortho /= np.linalg.norm(ortho, axis=1, keepdims=True)
    return {"hierarchical": hier, "orthogonal": ortho}, D, n_active


def mean_offdiag_cos(E):
    G = E @ E.T
    return float(np.abs(G[np.triu_indices(len(E), 1)]).mean())


def active_sets(codec, E):
    """Neurons a clean encoding of each item actually drives."""
    return [np.flatnonzero(codec.encode(v, rng=np.random.default_rng(0)).sum(axis=1) > 0)
            for v in E]


def evaluate(E, g_syn, k_inh, presentations, seed, D):
    codec, _ = make_codec(D, seed=seed, n_per_feature=N_PER_FEATURE,
                          window_steps=WINDOW_STEPS)
    inner = BushSTDPSpikingNetwork(n_neurons=codec.n_neurons, seed=seed,
                                   n_presentations=1, recall_steps=RECALL_STEPS,
                                   g_syn=g_syn, k_inh=k_inh)
    model = wrap_with_codec(inner, codec)
    for _ in range(presentations):
        ingest(model, E, regime="streamed")          # fit_event, per the protocol

    hits, spikes = 0, 0
    for i in range(len(E) - 1):
        raw = inner.predict_next(codec.encode(E[i], rng=np.random.default_rng(0)))
        spikes += int(raw.sum())
        pred = np.asarray(model.predict_next(E[i]), dtype=float)
        hits += int(np.argmax(E @ pred) == i + 1)

    A = active_sets(codec, E)
    fwd = float(np.mean([inner.W[np.ix_(A[i + 1], A[i])].mean()
                         for i in range(len(E) - 1)]))
    bwd = float(np.mean([inner.W[np.ix_(A[i], A[i + 1])].mean()
                         for i in range(len(E) - 1)]))
    return {"recall": hits / (len(E) - 1), "w_fwd": fwd, "w_bwd": bwd,
            "spikes": spikes / (len(E) - 1), "w_mean": float(inner.W.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--presentations", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    mats, D, n_active = materials()
    N = 2 * D * N_PER_FEATURE
    print(f"D={D}  N={N}  active dims/item={n_active}  "
          f"co-active neurons/item~{n_active * N_PER_FEATURE}")
    for name, E in mats.items():
        print(f"  {name:<13} mean|cos| between items = {mean_offdiag_cos(E):.2f}")
    print(f"ingestion: streamed (fit_event); exposure {args.presentations} passes; "
          f"{args.seeds} seeds\n")

    results = {"D": D, "N": N, "n_active": n_active, "grid": {}, "exposure": {}}
    t0 = time.time()

    for name, E in mats.items():
        print(f"=== {name}: one-step recall, g_syn x k_inh ===")
        print("  g \\ k " + "".join(f"{k:>8}" for k in K_GRID))
        best = None
        cells = {}
        for g in G_GRID:
            row = []
            for k in K_GRID:
                runs = [evaluate(E, g, k, args.presentations, s, D)
                        for s in range(args.seeds)]
                cell = {m: float(np.mean([r[m] for r in runs])) for m in runs[0]}
                cells[f"{g}/{k}"] = cell
                row.append(cell["recall"])
                if best is None or cell["recall"] > best["recall"]:
                    best = dict(cell, g_syn=g, k_inh=k)
            print(f"  {g:<6}" + "".join(f"{v:8.2f}" for v in row))
        d = best
        print(f"  best ({d['g_syn']:g}, {d['k_inh']:g}) recall={d['recall']:.2f}  "
              f"w_fwd={d['w_fwd']:.4f} w_bwd={d['w_bwd']:.4f}  "
              f"replay spikes/probe={d['spikes']:.0f}  [{time.time() - t0:.0f}s]")
        results["grid"][name] = {"cells": cells, "best": best}

        print(f"\n=== {name}: exposure at ({best['g_syn']:g}, {best['k_inh']:g}) ===")
        expo = {}
        for p in P_GRID:
            runs = [evaluate(E, best["g_syn"], best["k_inh"], p, s, D)
                    for s in range(args.seeds)]
            cell = {m: float(np.mean([r[m] for r in runs])) for m in runs[0]}
            expo[p] = cell
            print(f"  {p:>3} passes  recall={cell['recall']:.2f}  "
                  f"w_fwd={cell['w_fwd']:.4f} w_bwd={cell['w_bwd']:.4f}  "
                  f"spikes={cell['spikes']:.0f}")
        results["exposure"][name] = expo
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"wrote {args.json}  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
