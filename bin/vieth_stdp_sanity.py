#!/usr/bin/env python
"""Sanity check for the Vieth arm: can it do cued and autoregressive recall?

Separate from ``bin/vieth_stdp_gate.py``, which reproduces upstream's own text
task. This asks the question MemVal actually scores an arm on -- present a short
sequence, then (a) cue item t and ask for item t+1, and (b) cue item 0 and roll
the whole sequence forward -- and asks it on three substrates so a failure is
attributable rather than just a floor.

======================  ===================================================
``native``              One-hot blocks straight into the arm, no codec. The
                        arm's best case: items are disjoint by construction,
                        so this isolates "does the sequence memory work" from
                        "does codec transport survive".
``codec_orthogonal``    One-hot vectors through the population spike codec.
                        Adds Poisson transport but keeps items disjoint.
``codec_hierarchical``  ``HierarchicalEncoder`` through the codec -- the
                        substrate ``docs/spike_codec_spec.md`` S3.1 names as
                        the reference, where items genuinely overlap.
======================  ===================================================

Two controls, because neither alone is enough:

* an **untrained** network, which bounds what the readout gives away for free
  (this arm's readout is a learned linear map, so an untrained one is not
  automatically at chance), and
* **chance**, ``1 / n_items``.

Seeds are reported individually and never averaged: the arm is bimodal across
seeds (``docs/vieth_stdp_port.md``), so a mean here would describe a state the
model is never in.

Usage::

    python bin/vieth_stdp_sanity.py                     # all three substrates
    python bin/vieth_stdp_sanity.py --substrate native --exposure
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.encoders.hierarchical import HierarchicalEncoder
from memval.encoders.spike_codec import PopulationSpikeEncoder, PopulationSpikeDecoder
from memval.models.baselines.vieth_gaba_stdp import ViethGabaSTDPNetwork

SUBSTRATES = ("native", "codec_orthogonal", "codec_hierarchical")


# --------------------------------------------------------------------- material
def make_material(substrate, n_items, width, seed):
    """Return ``(codebook, n_input_neurons, to_spikes, from_spikes)``.

    ``to_spikes`` turns item i into what the arm eats; ``from_spikes`` turns the
    arm's output back into an index, which is the only place the three
    substrates differ.
    """
    if substrate == "native":
        n_inp = n_items * width
        def to_spikes(i):
            s = np.zeros((n_inp, 1), dtype=bool)
            s[i * width:(i + 1) * width] = True
            return s
        def from_spikes(out):
            per_item = np.asarray(out).sum(axis=1).reshape(n_items, width).sum(axis=1)
            return -1 if per_item.sum() <= 0 else int(per_item.argmax())
        return None, n_inp, to_spikes, from_spikes

    if substrate == "codec_orthogonal":
        V = np.eye(n_items)
    else:
        enc = HierarchicalEncoder(branching=[2, 3], features_per_node=4, seed=seed)
        V = enc.encode(list(enc.items)[:n_items])
        V = V / np.linalg.norm(V, axis=1, keepdims=True)

    codec = PopulationSpikeEncoder(V.shape[1], n_per_feature=5, window_steps=50,
                                   seed=seed)
    decoder = PopulationSpikeDecoder(codec)
    probe_rng = lambda: np.random.default_rng(0)

    def to_spikes(i):
        return codec.encode(V[i], rng=probe_rng())

    def from_spikes(out):
        v = decoder.decode(out)
        if not np.any(v):
            return -1
        return int(np.argmax(V @ v))          # nearest item by cosine

    return V, codec.n_neurons, to_spikes, from_spikes


# ------------------------------------------------------------------------ run
def one_run(substrate="native", n_items=6, width=10, n_exc=300, passes=400,
            fenced=True, seed=0, train=True, cue_lag=2, recall_steps=1):
    """Train on one sequence, then probe cued and autoregressive recall."""
    _, n_inp, to_spikes, from_spikes = make_material(substrate, n_items, width, seed)
    items = [to_spikes(i) for i in range(n_items)]

    m = ViethGabaSTDPNetwork(n_neurons=n_inp, n_exc=n_exc,
                             target_activity=1.0 / n_items, cue_lag=cue_lag,
                             recall_steps=recall_steps, seed=seed)
    if train:
        if fenced:
            # MemVal's own path: fit_sequence fences each pass with
            # on_event_boundary, so no transition forms across the seam.
            m.fit_sequence(np.stack(items), epochs=passes)
        else:
            # Upstream's path: one continuous stream, never reset, so the
            # wrap-around transition is learned too.
            for t in range(passes * n_items):
                m._iterate(m._frames(items[t % n_items])[0], plastic=True)

    # (a) cued: probe each position independently, item t -> item t+1
    cued = [from_spikes(m.predict_next(items[i])) == i + 1
            for i in range(n_items - 1)]

    # (b) autoregressive: cue item 0, roll forward, compare to 1..L-1
    roll = m.recall(items[0], n_items - 1)
    auto, ok = [], True
    for t in range(n_items - 1):
        hit = from_spikes(roll[t]) == t + 1
        ok = ok and hit                      # first miss ends the correct prefix
        auto.append(hit)
    prefix = int(np.argmin(auto)) if not all(auto) else len(auto)

    return {"seed": seed, "cued": float(np.mean(cued)),
            "auto": float(np.mean(auto)), "auto_prefix": prefix,
            "cued_hits": "".join("1" if c else "." for c in cued),
            "auto_hits": "".join("1" if a else "." for a in auto)}


def summarise(label, rows, chance):
    print(f"\n{label}")
    for r in rows:
        print(f"  seed {r['seed']}   cued {r['cued']:.2f} [{r['cued_hits']}]   "
              f"autoregressive {r['auto']:.2f} [{r['auto_hits']}]  "
              f"correct prefix {r['auto_prefix']}")
    c = np.array([r["cued"] for r in rows])
    a = np.array([r["auto"] for r in rows])
    print(f"  cued           min {c.min():.2f}  median {np.median(c):.2f}  max {c.max():.2f}"
          f"   (chance {chance:.2f})")
    print(f"  autoregressive min {a.min():.2f}  median {np.median(a):.2f}  max {a.max():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=SUBSTRATES + ("all",), default="all")
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--exc", type=int, default=300)
    ap.add_argument("--passes", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--unfenced", action="store_true",
                    help="train as one continuous stream, upstream's protocol")
    ap.add_argument("--exposure", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    subs = SUBSTRATES if args.substrate == "all" else (args.substrate,)
    chance = 1.0 / args.items
    out = {}

    if args.exposure:
        for sub in subs:
            print(f"\n=== {sub}: exposure sweep (seed 0) ===")
            for passes in (25, 50, 100, 200, 400, 800, 1600):
                r = one_run(substrate=sub, n_items=args.items, n_exc=args.exc,
                            passes=passes, fenced=not args.unfenced, seed=0)
                print(f"  passes {passes:5d} ({passes*args.items:6d} iterations)  "
                      f"cued {r['cued']:.2f}  autoregressive {r['auto']:.2f}  "
                      f"prefix {r['auto_prefix']}")
        return

    for sub in subs:
        rows = [one_run(substrate=sub, n_items=args.items, n_exc=args.exc,
                        passes=args.passes, fenced=not args.unfenced, seed=s)
                for s in range(args.seeds)]
        ctrl = [one_run(substrate=sub, n_items=args.items, n_exc=args.exc,
                        passes=args.passes, fenced=not args.unfenced, seed=s,
                        train=False) for s in range(args.seeds)]
        summarise(f"=== {sub} (trained, {args.passes} passes) ===", rows, chance)
        summarise(f"--- {sub} (UNTRAINED control) ---", ctrl, chance)
        out[sub] = {"trained": rows, "untrained": ctrl, "chance": chance}

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
