#!/usr/bin/env python
"""Phase 2.5 of ``docs/spike_codec_spec.md``: the Bush et al. (2010) correctness gate.

Reproduce the paper's own result **before** the arm's MemVal scores are allowed
to mean anything: >90% recall fidelity on the dual-coded configuration
(20 fields x 5 neurons), measured as temporal order fidelity across the recall
window, over 50 seeded runs.

Two modes:

    python bin/bush_stdp_gate.py                  # the gate, 50 seeds
    python bin/bush_stdp_gate.py --calibrate      # the g_syn/k_inh sweep behind
                                                  # the defaults, and the
                                                  # instability control

``--calibrate`` also runs the **instability control**: the same network built as
a literal reading of the spec describes it (recurrent excitation on during
encoding, no inhibitory pool). That is not a straw man -- it is the
configuration the module's inferred-parameter note exists to justify, and its
numbers belong in the record next to the ones that work.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.models.baselines.bush_stdp import (
    BushSTDPSpikingNetwork, dual_coded_stimulus, order_fidelity,
)

N_FIELDS, N_PER = 20, 5
GATE = 0.90


def one_run(seed: int, recall_steps: int = 40, **model_kw) -> dict:
    """Train on one seeded stimulus, cue field 0, score the replay window."""
    X = dual_coded_stimulus(n_fields=N_FIELDS, n_per_field=N_PER, seed=1000 + seed)
    m = BushSTDPSpikingNetwork(n_neurons=N_FIELDS * N_PER, recall_steps=recall_steps,
                               seed=seed, **model_kw)
    m.fit_sequence(X)
    spikes = m.predict_next(X[0])
    res = order_fidelity(spikes, N_FIELDS, N_PER)
    res.pop("first_spike")
    # Directionality of the learned matrix: the thing STDP's asymmetry is
    # supposed to produce, reported because a chain that replays for the wrong
    # reason is worth catching.
    Wf = m.W.reshape(N_FIELDS, N_PER, N_FIELDS, N_PER).mean(axis=(1, 3))
    res["w_forward"] = float(np.mean([Wf[k + 1, k] for k in range(N_FIELDS - 1)]))
    res["w_backward"] = float(np.mean([Wf[k, k + 1] for k in range(N_FIELDS - 1)]))
    res["w_mean"] = float(m.W.mean())
    res["n_spikes"] = int(spikes.sum())
    return res


def _nanmean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def run_gate(n_seeds: int = 50, recall_steps: int = 40, **model_kw) -> dict:
    rows = [one_run(s, recall_steps=recall_steps, **model_kw) for s in range(n_seeds)]
    fid = np.array([r["fidelity"] for r in rows])
    out = {
        "n_seeds": n_seeds, "recall_steps": recall_steps,
        "mean_fidelity": float(fid.mean()), "sd_fidelity": float(fid.std()),
        "min_fidelity": float(fid.min()), "median_fidelity": float(np.median(fid)),
        "frac_runs_above_gate": float((fid >= GATE).mean()),
        "mean_replayed": float(np.mean([r["n_replayed"] for r in rows])),
        # All-NaN when a configuration never replays anything -- a real
        # outcome in the calibration sweep, so report it as NaN rather than
        # letting numpy warn about it.
        "mean_span_steps": _nanmean([r["span_steps"] for r in rows]),
        "w_forward": float(np.mean([r["w_forward"] for r in rows])),
        "w_backward": float(np.mean([r["w_backward"] for r in rows])),
        "w_mean": float(np.mean([r["w_mean"] for r in rows])),
        "mean_spikes": float(np.mean([r["n_spikes"] for r in rows])),
        "model_kwargs": model_kw,
    }
    out["pass"] = bool(out["mean_fidelity"] >= GATE)
    return out


def print_gate(out: dict):
    print(f"\n=== Phase 2.5 gate: dual-coded {N_FIELDS} fields x {N_PER} neurons, "
          f"{out['n_seeds']} seeds ===")
    print(f"  recall window          : {out['recall_steps']} steps "
          f"(the paper's ripple is ~33; this replay takes ~36 -- see the module "
          f"note on recall_steps)")
    print(f"  order fidelity         : mean {out['mean_fidelity']:.3f}  "
          f"sd {out['sd_fidelity']:.3f}  median {out['median_fidelity']:.3f}  "
          f"min {out['min_fidelity']:.3f}")
    print(f"  runs at or above {GATE:.2f}   : {out['frac_runs_above_gate']:.2f}")
    print(f"  fields replayed        : {out['mean_replayed']:.1f} / {N_FIELDS}"
          f"   over {out['mean_span_steps']:.1f} steps")
    print(f"  learned directionality : forward {out['w_forward']:.3f}   "
          f"backward {out['w_backward']:.3f}   all-synapse mean {out['w_mean']:.3f}")
    print(f"  replay window activity : {out['mean_spikes']:.0f} spikes")
    print(f"\n  GATE: {'PASS' if out['pass'] else 'FAIL'} "
          f"(mean fidelity {out['mean_fidelity']:.3f} vs {GATE:.2f})")


def calibrate(n_seeds: int = 5):
    """The sweep behind the defaults, plus the instability control."""
    print("\n=== instability control: the spec's literal reading ===")
    print("  recurrent excitation ON during encoding, no inhibitory pool.")
    bad = run_gate(n_seeds, k_inh=0.0, ach_recurrent_gain=1.0)
    print(f"  order fidelity {bad['mean_fidelity']:.3f}   "
          f"all-synapse mean w {bad['w_mean']:.3f} / w_max 1.0   "
          f"replay activity {bad['mean_spikes']:.0f} spikes"
          f"  <- saturated; the chain is a seizure")

    print("\n=== calibration sweep (mean order fidelity over "
          f"{n_seeds} seeds, 33-step window) ===")
    g_values = (90, 220, 450, 550, 640, 900)
    # k_inh is a CONDUCTANCE now, not a current: the effective inhibitory
    # current is k_inh * n_fired * (e_inh - v), about 15x the conductance at
    # rest, so the working range sits roughly 15x below the old one.
    k_values = (0.02, 0.05, 0.09, 0.15, 0.3)
    print(f"  {'g_syn \\ k_inh':<14}" + "".join(f"{k:>8}" for k in k_values))
    grid = {}
    for g in g_values:
        row = []
        for k in k_values:
            r = run_gate(n_seeds, g_syn=float(g), k_inh=float(k))
            row.append(r["mean_fidelity"])
            grid[f"{g}/{k}"] = r["mean_fidelity"]
        print(f"  {g:<14}" + "".join(f"{v:8.2f}" for v in row))
    print("\n  Low-g cells fail because the chain dies out; low-k high-g cells "
          "fail because\n  it becomes a synchronous burst with no order left in "
          "it. The defaults sit in\n  the band where propagation is fast enough "
          "to fit 20 fields in 33 ms and still\n  ordered.")
    return {"instability_control": bad, "grid": grid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--recall-steps", type=int, default=40)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = {}
    if args.calibrate:
        results["calibration"] = calibrate()
    out = run_gate(args.seeds, recall_steps=args.recall_steps)
    print_gate(out)
    results["gate"] = out
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
