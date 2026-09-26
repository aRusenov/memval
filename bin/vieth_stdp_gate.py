#!/usr/bin/env python
"""Correctness gate for the Vieth & Triesch arm: upstream's own task.

Before any MemVal score from ``ViethGabaSTDPNetwork`` is allowed to mean
anything, the reimplementation has to reproduce the behaviour the paper is about
-- a self-organised sequence memory that, once the input is switched off,
free-runs and regenerates the text it was trained on.

This is the paper's protocol, not a MemVal section: no encoder, no vocabulary,
no cosine, no MRR. Characters are one-hot over a block of ``width`` input
neurons, exactly as upstream's ``Grid(width=10, height=n_chars)``, and the
readout is upstream's ``TextReconstructor`` including its argmax.

The reference was measured by running the **real upstream code**
(``scratch/vieth_ref.py`` on ``gitmv/GABA_Modulated_STDP_Paper``) at
``n_exc=600``, 20000 training steps, 2000 recovery, 2000 free, over seeds 0-9.

**Upstream is bimodal across seeds, and the gate is built around that.** Five of
its ten seeds regenerate the training text almost perfectly (text score
3.99-4.13); the other five produce noise (1.72-2.07). Nothing lands in between.
So "does it reproduce the paper" cannot be asked of one run: the gate scores a
*distribution* and checks that the successful mode is reached at a rate
compatible with upstream's, and that when it is reached the text really is
regenerated.

Measured, on the same protocol:

======================  ==========  ==============================
implementation          successes   score range (success / failure)
======================  ==========  ==============================
upstream (seeds 0-9)    5 / 10      3.99-4.13 / 1.72-2.07
this port (seeds 0-7)   3 / 8       4.09-4.12 / 1.74-2.18
======================  ==========  ==============================

At upstream's *published* size (``n_exc=2400``, 60000 steps) both fail on this
one-sentence grammar -- upstream 1.81, port 1.81-1.86 -- so 600 units is the
operating point for the gate, not a reduction for speed. The published
hyperparameters were evolved against a three-sentence text, and they do not
transfer to a shorter one by making the network bigger.

Usage::

    python bin/vieth_stdp_gate.py                 # the gate, 5 seeds
    python bin/vieth_stdp_gate.py --exposure      # rate/score vs training steps
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.models.baselines.vieth_gaba_stdp import ViethGabaSTDPNetwork

TEXT = " fox eats meat."
WIDTH = 10
#: Upstream at n_exc=600 / 20000 steps, measured over seeds 0-9, not quoted.
UPSTREAM_SCORES = (4.127, 4.127, 1.847, 4.127, 1.826, 2.071, 1.777, 3.992,
                   4.058, 1.720)
UPSTREAM_RATE = 0.0664
#: A run "succeeded" when it reached the generating mode. The two modes are
#: separated by a factor of two with nothing between them, so the threshold is
#: not a tuned parameter -- anything in (2.2, 3.9) gives the same partition.
SUCCESS_SCORE = 3.5
#: Upstream reaches that mode on half its seeds. Requiring the port to do so
#: every time would be requiring it to be better than what it reimplements.
GATE_MIN_SUCCESS_FRAC = 0.25


def alphabet(text):
    return sorted(set(text))


def char_frames(text, width=WIDTH):
    """``text`` -> ``(len(text), n_inp)`` one-hot blocks, upstream's input grid."""
    alpha = alphabet(text)
    frames = np.zeros((len(text), width * len(alpha)), dtype=bool)
    for i, c in enumerate(text):
        j = alpha.index(c)
        frames[i, j * width:(j + 1) * width] = True
    return frames


def text_score(text, blocks, alpha, char_weighting):
    """Upstream ``TextGenerator.get_text_score``, reimplemented verbatim."""
    block_scores = [1.0 for _ in blocks]
    for i in range(len(text)):
        for bi, block in enumerate(blocks):
            block_score = 0.0
            comp_text = text[i:i + len(block)]
            comp_block = block[0:len(comp_text)]
            for t, b in zip(comp_text, comp_block):
                if t == b:
                    block_score += 1.0 / char_weighting[alpha.index(t)]
            block_scores[bi] += (block_score * block_score) / len(text)
    return float(sum(np.sqrt(bs) for bs in block_scores))


def run(seed=0, n_exc=600, train_steps=20000, recovery_steps=2000,
        free_steps=2000, text=TEXT, train=True, **model_kw):
    """Train on the repeated text, then free-run and read out the generation."""
    alpha = alphabet(text)
    frames = char_frames(text)
    counts = np.array([text.count(c) for c in alpha], dtype=float)
    char_weighting = counts / counts.mean()

    model = ViethGabaSTDPNetwork(n_neurons=frames.shape[1], n_exc=n_exc,
                                 target_activity=1.0 / len(text), seed=seed,
                                 **model_kw)
    # Upstream presents one continuous stream and never resets, so drive the
    # engine directly rather than through fit_sequence's fenced passes.
    for t in range(train_steps):
        model._iterate(frames[t % len(text)], plastic=train)

    silence = np.zeros(frames.shape[1], dtype=bool)
    for _ in range(recovery_steps):
        model._iterate(silence, plastic=train)

    generated, rates = [], []
    for _ in range(free_steps):
        act = model._reconstruct()
        if act.sum() == 0:
            generated.append("#")
        else:
            per_char = act.reshape(len(alpha), WIDTH).sum(axis=1)
            generated.append(alpha[int(np.argmax(per_char))])
        rates.append(float(model._state["exc_spike"].mean()))
        model._iterate(silence, plastic=train)

    gen = "".join(generated)
    return {"seed": seed, "rate": float(np.mean(rates)),
            "target": 1.0 / len(text),
            "score": text_score(gen, [text], alpha, char_weighting),
            "generated": gen[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--exc", type=int, default=600)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--exposure", action="store_true",
                    help="sweep training steps instead of running the gate")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.exposure:
        rows = []
        for steps in (1000, 2500, 5000, 10000, 20000, 40000):
            r = run(seed=0, n_exc=args.exc, train_steps=steps)
            rows.append(dict(r, train_steps=steps))
            print(f"steps={steps:6d}  rate={r['rate']:.4f} "
                  f"(target {r['target']:.4f})  score={r['score']:.3f}  "
                  f"{r['generated'][:48]!r}")
        if args.json:
            json.dump(rows, open(args.json, "w"), indent=2)
        return

    rows = [run(seed=s, n_exc=args.exc, train_steps=args.steps)
            for s in range(args.seeds)]

    scores = np.array([r["score"] for r in rows])
    ok = scores >= SUCCESS_SCORE
    frac = float(ok.mean())
    up = np.array(UPSTREAM_SCORES)
    up_frac = float((up >= SUCCESS_SCORE).mean())

    for r, good in zip(rows, ok):
        print(f"  seed {r['seed']}  rate={r['rate']:.4f}  score={r['score']:.3f}  "
              f"{'GENERATES' if good else 'noise    '}  {r['generated'][:48]!r}")

    print(f"\nsucceeded     {int(ok.sum())}/{len(rows)} seeds ({frac:.0%})  "
          f"upstream {int((up >= SUCCESS_SCORE).sum())}/{len(up)} ({up_frac:.0%})")
    if ok.any():
        print(f"  when it generates  score {scores[ok].min():.3f}-{scores[ok].max():.3f}"
              f"   (upstream {up[up >= SUCCESS_SCORE].min():.3f}-"
              f"{up[up >= SUCCESS_SCORE].max():.3f})")
    if (~ok).any():
        print(f"  when it does not   score {scores[~ok].min():.3f}-{scores[~ok].max():.3f}"
              f"   (upstream {up[up < SUCCESS_SCORE].min():.3f}-"
              f"{up[up < SUCCESS_SCORE].max():.3f})")
    print(f"  firing rate mean {np.mean([r['rate'] for r in rows]):.4f}  "
          f"target {rows[0]['target']:.4f}  (upstream {UPSTREAM_RATE:.4f})")

    passed = frac >= GATE_MIN_SUCCESS_FRAC and ok.any() and \
        scores[ok].min() >= SUCCESS_SCORE
    print(f"\nGATE: {'PASS' if passed else 'FAIL'}  (reaches upstream's generating "
          f"mode on at least {GATE_MIN_SUCCESS_FRAC:.0%} of seeds)")
    if args.json:
        json.dump({"runs": rows, "success_frac": frac,
                   "upstream_success_frac": up_frac, "pass": bool(passed)},
                  open(args.json, "w"), indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
