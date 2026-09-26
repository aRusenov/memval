#!/usr/bin/env python
"""Is one-shot acquisition magnitude-free?

`presentation_duration` reports `convergence_epochs = 1` for one-shot associators,
which says an association formed in a single pass. It does not say the trace is
*strong*: `measure_recall_associative` decodes with `SymbolicDecoder`, which is
cosine-based and therefore scale-invariant, so a readout pointing the right way
scores 1.0 no matter how faint it is.

This probe separates the two by sweeping the learning rate at a fixed single
epoch and reporting three quantities on one [0, 1] scale:

  * `mrr`  -- the benchmark's own score.
  * `cos`  -- cosine between the prediction and the true next embedding: how well
    aimed the readout is.
  * `mag`  -- |prediction| / |target|: how strong the trace is.

If `mrr` is flat while `mag` moves over orders of magnitude, the section is
scoring direction only. That is a property of the read-out, not a model failure —
but it has to be stated, because the same unconstrained magnitude is what makes
autoregressive rollout collapse in `sequence_length` (raw feedback) while cued
recall stays perfect.

Diagnostic, not a suite section: it needs a per-arm hyperparameter (the learning
rate) that not every arm has, so it cannot be scored commensurably across the
taxonomy.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.benchmarks.symbolic_pipeline import (
    measure_recall_associative, mean_recall_rate)
from memval.models.baselines import AsymmetricHopfieldNetwork

DEFAULT_RATES = (1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0)
WORDS = ["apple", "banana", "orange", "grape", "pear", "peach", "plum"]


def probe(rates=DEFAULT_RATES, n_trials=30, epochs=1, seed=42, noise_scale=0.05):
    vocab_path = os.path.join(os.path.dirname(__file__), "..", "data", "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)
    fruit = {w: c for w, c in vocab.items() if c == "fruit"}
    enc = SymbolicEncoder(fruit, embedding_dim=100, category_variance=0.2, seed=seed)
    dec = SymbolicDecoder(enc)
    X = enc.encode(WORDS)

    rows = []
    for lr in rates:
        np.random.seed(0)
        m = AsymmetricHopfieldNetwork(n_features=100, learning_rate=lr, seed=seed)
        m.fit_sequence(X, epochs=epochs)
        m.reset_context()
        mrr = mean_recall_rate(measure_recall_associative(
            m, WORDS, enc, dec, n_trials=n_trials, noise_scale=noise_scale))
        cos, mag = [], []
        for i in range(len(WORDS) - 1):
            p, t = m.predict_next(X[i]), X[i + 1]
            d = np.linalg.norm(p) * np.linalg.norm(t)
            cos.append(float(p @ t / d) if d > 1e-12 else 0.0)
            mag.append(float(np.linalg.norm(p) / np.linalg.norm(t)))
        rows.append(dict(learning_rate=float(lr), mrr=float(mrr),
                         cos=float(np.mean(cos)), mag=float(np.mean(mag)),
                         w_norm=float(np.linalg.norm(m.W))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Path for the probe JSON.")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=1,
                    help="Passes over the sequence (default 1 = the one-shot point).")
    a = ap.parse_args()

    rows = probe(n_trials=a.n_trials, epochs=a.epochs)
    payload = {
        "probe": "oneshot_scale",
        "model": "AsymmetricHopfieldNetwork",
        "epochs": a.epochs,
        "n_trials": a.n_trials,
        "word_list": WORDS,
        "decoder": "SymbolicDecoder (cosine, scale-invariant)",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"{'lr':>8} {'MRR':>7} {'cos':>7} {'|pred|/|target|':>16}")
    for r in rows:
        print(f"{r['learning_rate']:>8g} {r['mrr']:>7.3f} {r['cos']:>7.3f} "
              f"{r['mag']:>16.4f}")
    span = max(r["mag"] for r in rows) / max(min(r["mag"] for r in rows), 1e-12)
    print(f"\ntrace magnitude spans {span:,.0f}x; MRR range "
          f"{min(r['mrr'] for r in rows):.3f}–{max(r['mrr'] for r in rows):.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
