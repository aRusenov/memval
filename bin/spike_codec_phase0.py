#!/usr/bin/env python
"""Phase 0 of ``docs/spike_codec_spec.md``: characterise the codec in isolation.

No model. This is the **go/no-go**: if round-trip fidelity has a low ceiling,
everything downstream is capped and the spiking tier is not worth building.

Acceptance (spec S6, Phase 0)
  * median round-trip cosine >= 0.95 at the defaults
  * fidelity monotone non-decreasing in ``n_per_feature`` and ``window_steps``
  * identical output for an identical seed
  * the global RNG is untouched

It also answers two of the spec's open questions with measurements rather than
argument:
  Q1  does ``HierarchicalEncoder``'s shared/identity block asymmetry survive
      transport, with no model attached?
  Q2  does renormalisation compose -- upstream L2 after masking, then the
      decoder's own L2 -- without double-shrinking?

Usage::

    python bin/spike_codec_phase0.py                     # full report
    python bin/spike_codec_phase0.py --json out.json     # machine-readable too
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.encoders.hierarchical import HierarchicalEncoder
from memval.encoders.symbolic import SymbolicEncoder
from memval.encoders.spike_codec import make_codec


def substrates(dim: int = 100, seed: int = 0):
    """The two candidate substrates, plus the sparse regime in isolation."""
    hier = HierarchicalEncoder(branching=(2, 2, 2), features_per_node=4)
    wide = HierarchicalEncoder(branching=(3, 3, 3), features_per_node=4)
    sym = SymbolicEncoder([f"w{i}" for i in range(32)], embedding_dim=dim, seed=seed)
    return {
        "hierarchical D=%d" % hier.embeddings.shape[1]: hier.embeddings,
        "hierarchical D=%d" % wide.embeddings.shape[1]: wide.embeddings,
        "symbolic D=%d" % dim: sym.embeddings,
    }


def roundtrip_cosines(E: np.ndarray, seed: int = 0, **codec_kw) -> np.ndarray:
    enc, dec = make_codec(E.shape[1], seed=seed, **codec_kw)
    return np.array([float(dec.decode(enc.encode(v)) @ v) for v in E])


def report_defaults(subs, seeds=(0, 1, 2)):
    print("\n=== round-trip fidelity at the defaults "
          "(n_per_feature=5, T=50, r_max=200Hz) ===")
    print(f"{'substrate':<22} {'median':>8} {'min':>8} {'gate 0.95':>10}")
    out = {}
    for name, E in subs.items():
        cos = np.concatenate([roundtrip_cosines(E, seed=s) for s in seeds])
        out[name] = {"median": float(np.median(cos)), "min": float(cos.min()),
                     "pass": bool(np.median(cos) >= 0.95)}
        print(f"{name:<22} {np.median(cos):8.3f} {cos.min():8.3f} "
              f"{'PASS' if np.median(cos) >= 0.95 else 'FAIL':>10}")
    return out


def report_curves(subs, seeds=(0, 1, 2)):
    """Fidelity vs the two knobs that buy Poisson events. Monotonicity is a gate."""
    out = {}
    for knob, values, kw in (("n_per_feature", (1, 2, 5, 10, 20), "n_per_feature"),
                             ("window_steps", (10, 25, 50, 100, 200), "window_steps"),
                             ("r_max", (50, 100, 200, 400), "r_max")):
        print(f"\n=== median round-trip cosine vs {knob} ===")
        header = f"{'substrate':<22}" + "".join(f"{v:>9}" for v in values)
        print(header)
        out[knob] = {}
        for name, E in subs.items():
            meds = []
            for v in values:
                cos = np.concatenate([roundtrip_cosines(E, seed=s, **{kw: v})
                                      for s in seeds])
                meds.append(float(np.median(cos)))
            mono = all(b >= a - 1e-9 for a, b in zip(meds, meds[1:]))
            out[knob][name] = {"values": list(values), "median": meds,
                               "monotone": bool(mono)}
            print(f"{name:<22}" + "".join(f"{m:9.3f}" for m in meds)
                  + ("" if mono else "   <- NOT monotone"))
    return out


def report_determinism(subs):
    """Spec constraint 4: same seed -> byte-identical; global RNG untouched."""
    print("\n=== determinism and RNG hygiene ===")
    E = next(iter(subs.values()))
    a = roundtrip_cosines(E, seed=7)
    b = roundtrip_cosines(E, seed=7)
    same = bool(np.array_equal(a, b))
    different = bool(not np.array_equal(a, roundtrip_cosines(E, seed=8)))

    before = np.random.get_state()
    roundtrip_cosines(E, seed=7)
    after = np.random.get_state()
    untouched = bool(before[0] == after[0] and np.array_equal(before[1], after[1])
                     and before[2:] == after[2:])
    print(f"  identical seed -> identical output : {'PASS' if same else 'FAIL'}")
    print(f"  different seed -> different output : {'PASS' if different else 'FAIL'}")
    print(f"  global RNG untouched              : {'PASS' if untouched else 'FAIL'}")
    return {"repeatable": same, "seed_sensitive": different,
            "global_rng_untouched": untouched}


def report_sensitivity(subs, seeds=(0, 1, 2, 3, 4)):
    """What upstream corruption survives transport (spec S1: sensitivity).

    Definability is guaranteed by construction -- the probe is built in vector
    space at full resolution. What is at risk is whether the codec washes a
    small corruption out before the model sees it. The number that matters is
    how much of the corruption's effect on cosine-to-clean is *retained* after
    the round trip.
    """
    print("\n=== probe sensitivity: does upstream corruption survive transport? ===")
    print("  cos(clean, noisy) upstream  ->  cos(clean, decode(encode(noisy))) "
          "downstream")
    out = {}
    for name, E in subs.items():
        rows = []
        for sigma in (0.0, 0.05, 0.1, 0.2, 0.4):
            up, down = [], []
            for s in seeds:
                rng = np.random.default_rng(1000 + s)
                enc, dec = make_codec(E.shape[1], seed=s)
                for v in E:
                    noisy = v + rng.standard_normal(v.shape) * sigma
                    noisy /= np.linalg.norm(noisy)
                    up.append(float(noisy @ v))
                    down.append(float(dec.decode(enc.encode(noisy)) @ v))
            rows.append((sigma, float(np.mean(up)), float(np.mean(down))))
        out[name] = rows
        print(f"  {name}")
        for sigma, u, d in rows:
            print(f"    sigma={sigma:<5} upstream {u:6.3f}   downstream {d:6.3f}"
                  f"   retained {(1 - d) / (1 - u) if u < 1 else float('nan'):6.2f}x")
    return out


def report_block_asymmetry(seeds=(0, 1, 2)):
    """Spec open question 1: do the hierarchical blocks survive transport?

    ``cue_masking`` distinguishes masking the *shared* columns (features a node
    gives all its descendants) from the *identity* columns (features unique to
    the leaf). If the codec flattens that distinction, the block modes cannot be
    run on spiking arms and that section is N/A for them.
    """
    print("\n=== open question 1: hierarchical shared/identity blocks after transport ===")
    h = HierarchicalEncoder(branching=(2, 2, 2), features_per_node=4)
    E = h.embeddings
    leaf_cols, shared_cols = _leaf_and_shared_columns(h)
    out = {}
    for label, cols in (("mask shared", shared_cols), ("mask identity", leaf_cols)):
        up, down = [], []
        for s in seeds:
            enc, dec = make_codec(E.shape[1], seed=s)
            for v in E:
                masked = v.copy()
                masked[cols] = 0.0
                n = np.linalg.norm(masked)
                if n == 0:
                    continue
                masked /= n
                up.append(float(masked @ v))
                down.append(float(dec.decode(enc.encode(masked)) @ v))
        out[label] = {"upstream": float(np.mean(up)), "downstream": float(np.mean(down))}
        print(f"  {label:<15} upstream {np.mean(up):6.3f}   downstream {np.mean(down):6.3f}")
    gap_up = abs(out["mask shared"]["upstream"] - out["mask identity"]["upstream"])
    gap_dn = abs(out["mask shared"]["downstream"] - out["mask identity"]["downstream"])
    print(f"  asymmetry (shared vs identity): upstream {gap_up:.3f} -> "
          f"downstream {gap_dn:.3f}  ({gap_dn / gap_up if gap_up else float('nan'):.2f}x retained)")
    out["asymmetry_upstream"], out["asymmetry_downstream"] = gap_up, gap_dn
    return out


def _leaf_and_shared_columns(h: HierarchicalEncoder):
    """Columns owned by leaf nodes, and columns owned by internal nodes."""
    leaf, shared = [], []
    for node in h.nodes:
        cols = node.get("columns", [])
        (leaf if not node["children"] else shared).extend(cols)
    return np.array(sorted(leaf), dtype=int), np.array(sorted(shared), dtype=int)


def report_renormalisation(seeds=(0, 1, 2)):
    """Spec open question 2: does the double L2 double-shrink?

    Upstream, a masked cue is re-L2-normalised; the decoder L2-normalises again.
    Composition would be a problem if the second normalisation changed the
    *direction* rather than only the scale. Measured as the norm of the decoded
    vector before its normalisation, relative to the input norm.
    """
    print("\n=== open question 2: does the double L2 double-shrink? ===")
    h = HierarchicalEncoder(branching=(2, 2, 2), features_per_node=4)
    ratios = []
    for s in seeds:
        enc, dec = make_codec(h.embeddings.shape[1], seed=s)
        for v in h.embeddings:
            c = dec.counts(enc.encode(v))
            raw = (c[:, 0] - c[:, 1]) / enc.spikes_per_unit    # pre-normalisation
            ratios.append(float(np.linalg.norm(raw) / np.linalg.norm(v)))
    print(f"  ||decode_raw(encode(v))|| / ||v||  mean {np.mean(ratios):.3f}  "
          f"sd {np.std(ratios):.3f}")
    print("  (a value near 1 means the estimator is unbiased in scale, so the "
          "decoder's L2\n   only fixes scale and cannot compound the upstream one)")
    return {"norm_ratio_mean": float(np.mean(ratios)),
            "norm_ratio_sd": float(np.std(ratios))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write results here")
    args = ap.parse_args()

    subs = substrates()
    results = {
        "defaults": report_defaults(subs),
        "curves": report_curves(subs),
        "determinism": report_determinism(subs),
        "sensitivity": report_sensitivity(subs),
        "block_asymmetry": report_block_asymmetry(),
        "renormalisation": report_renormalisation(),
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
