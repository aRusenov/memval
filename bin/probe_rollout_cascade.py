#!/usr/bin/env python
"""Where does the recall chain break under its own output, and does exposure fix it?

`presentation_duration` reports one-shot acquisition because it probes **cued**
recall: every position is cued independently with the true previous item, so an
error at position i cannot reach position i+1. Autoregressive rollout removes
that guardrail — the model's own prediction becomes the next input — and it is
the only protocol in the suite where an error propagates.

This probe crosses the two along the exposure axis, holding the list fixed:

  * `cued`                 one-step recall; the no-propagation reference.
  * `rollout_raw`          feed the prediction back verbatim.
  * `rollout_l2`           feed it back L2-renormalised (removes magnitude drift).
  * `rollout_quantized`    snap to the decoded symbol's clean embedding
                           (removes direction error too; the drift-free bound).

and records the geometry that explains the gap:

  * `cos_one_step`   mean cosine between prediction and true next item.
  * `mag_one_step`   mean |prediction| / |target|.
  * `cos_rollout`             cosine to the true item at each step of a free
                              rollout.
  * `cos_rollout_competitor`  cosine to the best *other* item at the same step.
  * `margin_rollout`          the difference. Recall succeeds exactly where this is
                              positive, so it is the quantity to reason about;
                              cosine to the target alone is not, because the
                              decode is a ranking and the vocabulary is not
                              orthogonal.

Reading the modes against each other is what identifies the mechanism: if `l2`
recovers what `raw` loses, the drift is radial (magnitude); if `l2` tracks `raw`
and only `quantized` recovers, the drift is angular, and renormalising cannot
help because the fed-back direction is already wrong.

Diagnostic, not a suite section — `sequence_length` already owns the span x
feedback-mode grid. This adds the per-position and per-exposure resolution that
the scalar spans summarise away.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.benchmarks.symbolic_pipeline import (
    measure_recall_associative, measure_recall_autoregressive,
    mean_recall_rate, memory_span)
from memval.models.baselines import AsymmetricHopfieldNetwork

MODES = ("raw", "l2", "quantized")
DEFAULT_EXPOSURES = (1, 2, 4, 8, 16, 32, 64, 128)


def _geometry(model, X):
    """One-step aim and strength, and the geometry along a free rollout.

    Cosine to the target is not on its own a measure of trace fidelity: decoding
    is nearest-neighbour over the whole vocabulary, so what decides the read-out
    is the *margin* between the target and the best competing item. The
    embeddings are not orthogonal either (within-category mean pairwise cosine
    ~0.18 at `category_variance=0.2`), so there is a floor the target has to
    clear. All three are returned; the margin is the one that predicts recall.
    """
    n = len(X) - 1
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos1, mag1 = [], []
    for i in range(n):
        p, t = model.predict_next(X[i]), X[i + 1]
        d = np.linalg.norm(p) * np.linalg.norm(t)
        cos1.append(float(p @ t / d) if d > 1e-12 else 0.0)
        mag1.append(float(np.linalg.norm(p) / np.linalg.norm(t)))
    cur = X[0].copy()
    target, competitor = [], []
    for i in range(n):
        p = model.predict_next(cur)
        pn = p / max(float(np.linalg.norm(p)), 1e-12)
        sims = Xn @ pn
        target.append(float(sims[i + 1]))
        competitor.append(float(np.max(np.delete(sims, i + 1))))
        cur = p
    margin = [t - c for t, c in zip(target, competitor)]
    return (float(np.mean(cos1)), float(np.mean(mag1)),
            target, competitor, margin, cos1)


def _closure_epoch(fit_and_score, coarse_rows, key, threshold=0.999):
    """The smallest exposure that reaches `threshold`, not merely the first
    log-grid checkpoint that does.

    The coarse sweep doubles, so a value that first passes at 32 could have
    passed anywhere in (16, 32]. Reporting the checkpoint as though it were the
    threshold inflates the number by up to 2x, so the gap between the coarse
    grid and the true minimum is resolved by scanning the interval.
    """
    passing = [r for r in coarse_rows if r[key] >= threshold]
    if not passing:
        return None
    first = passing[0]["epochs"]
    below = [r["epochs"] for r in coarse_rows if r["epochs"] < first]
    lo = (below[-1] + 1) if below else 1
    for ep in range(lo, first + 1):
        if fit_and_score(ep) >= threshold:
            return ep
    return first


def probe(seq_len=11, exposures=DEFAULT_EXPOSURES, n_trials=30, lr=0.1,
          seed=42, noise_scale=0.05):
    vocab_path = os.path.join(os.path.dirname(__file__), "..", "data", "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)
    words = sorted(w for w, c in vocab.items() if c == "fruit")[:seq_len]
    if len(words) < seq_len:
        raise ValueError(f"need {seq_len} fruit words, vocab has {len(words)}")
    enc = SymbolicEncoder({w: "fruit" for w in words}, embedding_dim=100,
                          category_variance=0.2, seed=seed)
    dec = SymbolicDecoder(enc)
    X = enc.encode(words)

    rows = []
    for ep in exposures:
        np.random.seed(0)
        m = AsymmetricHopfieldNetwork(n_features=100, learning_rate=lr, seed=seed)
        m.fit_sequence(X, epochs=ep)

        m.reset_context()
        cued = measure_recall_associative(m, words, enc, dec, n_trials=n_trials,
                                          noise_scale=noise_scale)
        curves = {}
        for mode in MODES:
            m.reset_context()
            curves[mode] = measure_recall_autoregressive(
                m, words, enc, dec, n_trials=n_trials, noise_scale=noise_scale,
                feedback_mode=mode)
        cos1, mag1, tgt, comp, marg, cos1_per_pos = _geometry(m, X)
        rows.append(dict(
            epochs=int(ep),
            cued_curve=[float(v) for v in cued],
            cued_mrr=float(mean_recall_rate(cued)),
            cued_span=int(memory_span(cued)),
            rollout_curve={k: [float(v) for v in c] for k, c in curves.items()},
            rollout_mrr={k: float(mean_recall_rate(c)) for k, c in curves.items()},
            rollout_span={k: int(memory_span(c)) for k, c in curves.items()},
            cos_one_step=cos1, mag_one_step=mag1,
            cos_one_step_curve=cos1_per_pos,
            cos_rollout=tgt,
            cos_rollout_competitor=comp,
            margin_rollout=marg,
        ))

    def _raw_mrr(ep):
        np.random.seed(0)
        mm = AsymmetricHopfieldNetwork(n_features=100, learning_rate=lr, seed=seed)
        mm.fit_sequence(X, epochs=ep)
        mm.reset_context()
        return float(mean_recall_rate(measure_recall_autoregressive(
            mm, words, enc, dec, n_trials=n_trials, noise_scale=noise_scale,
            feedback_mode="raw")))

    def _cued_mrr(ep):
        np.random.seed(0)
        mm = AsymmetricHopfieldNetwork(n_features=100, learning_rate=lr, seed=seed)
        mm.fit_sequence(X, epochs=ep)
        mm.reset_context()
        return float(mean_recall_rate(measure_recall_associative(
            mm, words, enc, dec, n_trials=n_trials, noise_scale=noise_scale)))

    flat = [dict(epochs=r["epochs"], cued=r["cued_mrr"], raw=r["rollout_mrr"]["raw"])
            for r in rows]
    closure = {
        "cued": _closure_epoch(_cued_mrr, flat, "cued"),
        "rollout_raw": _closure_epoch(_raw_mrr, flat, "raw"),
        "rollout_raw_grid_first_pass": next(
            (r["epochs"] for r in flat if r["raw"] >= 0.999), None),
    }
    return words, rows, closure


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=11)
    ap.add_argument("--n-trials", type=int, default=30)
    a = ap.parse_args()

    words, rows, closure = probe(seq_len=a.seq_len, n_trials=a.n_trials)
    payload = dict(probe="rollout_cascade", model="AsymmetricHopfieldNetwork",
                   seq_len=a.seq_len, n_trials=a.n_trials, words=words,
                   max_span=a.seq_len - 1, rows=rows, closure=closure)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"L = {a.seq_len}, max span {a.seq_len - 1}")
    print(f"{'epochs':>7} {'cued':>6} {'raw':>6} {'l2':>6} {'quant':>6} "
          f"{'span_raw':>9} {'cos1':>6} {'|p|/|t|':>8}")
    for r in rows:
        print(f"{r['epochs']:>7} {r['cued_mrr']:>6.2f} {r['rollout_mrr']['raw']:>6.2f} "
              f"{r['rollout_mrr']['l2']:>6.2f} {r['rollout_mrr']['quantized']:>6.2f} "
              f"{r['rollout_span']['raw']:>9} {r['cos_one_step']:>6.3f} "
              f"{r['mag_one_step']:>8.3f}")
    c, r_ = closure["cued"], closure["rollout_raw"]
    if c and r_:
        print(f"\ncued closes at {c} presentation(s); raw rollout at {r_} "
              f"— a {r_ / c:.0f}x gap on identical weights.")
        g = closure["rollout_raw_grid_first_pass"]
        if g and g != r_:
            print(f"(the log grid first passes at {g}; {r_} is the resolved minimum)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
