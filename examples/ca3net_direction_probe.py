# -*- coding: utf8 -*-
"""Phase 2 gate for the CA3 arm: does the learned chain replay FORWARD?

Why this decides the port
-------------------------
ca3net's shipped configuration uses a *symmetric* STDP kernel (Mishra et al. 2016).
A symmetric kernel potentiates both pre->post and post->pre, so the learned matrix
is (near) symmetric and the network replays in either direction. Measured on
upstream's own place-cell setup (2026-08-28): 6/7 reverse spontaneously, 8/10
reverse when cued.

Under MemVal's `predict_next` a reverse sweep is a *wrong answer* for reasons that
have nothing to do with memory, so ~2/3 of events would score as failures. `stdp.py`
also ships an `asym` mode; this probe measures the forward fraction under both.

GATE: asym yields >= 80% forward sweeps. If it does not, the arm can only be scored
with a direction-agnostic metric -- a change to the suite, not to the arm.

Design
------
The stimulus is a synthetic ordered chain, not a `PlaceCellEncoder` trajectory:
item k is a Gaussian bump centred on feature dimension k*step. That makes "forward"
unambiguous (chain order increases with cell index) and isolates the direction
question from encoder geometry. The chain is cued at its START, so forward and
reverse have comparable room to run.

Learning and recall run in SEPARATE processes: `stdp.learning()` wants Brian2's
cpp_standalone device, which is a global one-way switch.

Usage
-----
    python ca3net_direction_probe.py --stage learn  --mode sym
    python ca3net_direction_probe.py --stage recall --mode sym
    (repeat for --mode asym)
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memval.models.baselines._ca3net_io import Ca3netIO  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "scratch", "ca3net_phase2")

DIM = 400
N_CELLS = 8000
N_ITEMS = 16                 # chain length
BUMP_SIGMA = 4.0             # in feature-dimension units
CHAIN_LO, CHAIN_HI = 20, 380  # leave margins at both ends of the population
N_TRAVERSALS = 150            # repetitions of the chain during learning
INTER_TRAVERSAL_MS = 300.0    # silence between traversals -- see stage_learn
EVENT_MS = 100.0

# STDP parameters, verbatim from upstream stdp.py __main__
STDP_PARAMS = {
    "sym":  dict(taup_ms=62.5, taum_ms=62.5, Ap=4e-3,  Am=4e-3,  wmax=2e-8, scale=0.62),
    "asym": dict(taup_ms=20.0, taum_ms=20.0, Ap=0.01,  Am=-0.01, wmax=4e-8, scale=1.27),
}


def build_chain():
    """N_ITEMS Gaussian bumps marching from CHAIN_LO to CHAIN_HI."""
    centres = np.linspace(CHAIN_LO, CHAIN_HI, N_ITEMS)
    x = np.arange(DIM)[None, :]
    return np.exp(-((x - centres[:, None]) ** 2) / (2 * BUMP_SIGMA ** 2)), centres


def cue_block_for(centre):
    """Contiguous PC range covering the bump at `centre` (cells per dim = N/DIM)."""
    per_dim = N_CELLS // DIM
    lo = int(max(0, (centre - 2 * BUMP_SIGMA)) * per_dim)
    hi = int(min(DIM, (centre + 2 * BUMP_SIGMA)) * per_dim)
    return lo, hi


# ------------------------------------------------------------------ learning
def stage_learn(mode):
    from brian2 import ms
    import memval.vendor.ca3net as ca3net

    ca3net.use_standalone(directory=os.path.join(OUT, "standalone_%s" % mode))
    from memval.vendor.ca3net import stdp as vstdp
    from memval.vendor.ca3net.helper import save_wmx

    chain, _ = build_chain()
    io = Ca3netIO(dim=DIM, n_cells=N_CELLS, event_ms=EVENT_MS, seed=11)
    reps = np.tile(chain, (N_TRAVERSALS, 1))
    # A blank gap after each traversal. Without it `np.tile` puts item N back to
    # back with item 1, so STDP learns the wrap-around link and the chain becomes a
    # closed LOOP: once cued, activity cycles forever and the network shows a single
    # run-long event instead of discrete SWR-like ones. Measured 2026-09-04: with no
    # gap there is no discrete-event regime at any weight scale, in either STDP mode.
    gaps = np.zeros(len(reps))
    gaps[N_ITEMS - 1::N_ITEMS] = INTER_TRAVERSAL_MS
    st, ni = io.encode_events(reps, intervals_ms=gaps)
    dur_s = (len(reps) * EVENT_MS + N_TRAVERSALS * INTER_TRAVERSAL_MS) * 1e-3
    print("chain %d items x %d traversals -> %.0f s, %d spikes"
          % (N_ITEMS, N_TRAVERSALS, dur_s, len(st)), flush=True)

    p = STDP_PARAMS[mode]
    t0 = time.time()
    # upstream scales the amplitudes by wmax before calling learning()
    wmx = vstdp.learning(ni, st * 1e-3, p["taup_ms"] * ms, p["taum_ms"] * ms,
                         p["Ap"] * p["wmax"], p["Am"] * p["wmax"],
                         p["wmax"], 1e-10)
    wmx *= p["scale"]
    nz = wmx[wmx > 0]
    print("learned in %.0fs | %d nonzero | mean %.3f nS | max %.3f nS"
          % (time.time() - t0, nz.size, nz.mean(), wmx.max()), flush=True)

    # asymmetry index: how much more forward (j>i) mass than backward
    fwd = np.triu(wmx, 1).sum(); bwd = np.tril(wmx, -1).sum()
    print("weight mass forward %.1f / backward %.1f -> asymmetry %.3f"
          % (fwd, bwd, (fwd - bwd) / (fwd + bwd + 1e-12)), flush=True)

    os.makedirs(OUT, exist_ok=True)
    save_wmx(wmx, os.path.join(OUT, "wmx_%s.npz" % mode))
    print("saved wmx_%s.npz" % mode, flush=True)


# -------------------------------------------------------------------- recall
def _events_from_rate(rate_ms, th=2.0, min_len_ms=260, bin_ms=20):
    """Minimal high-activity detector. Upstream's `slice_high_activity` is not
    vendored (it depends on the len_sim landmine); this reimplements the same
    rule -- contiguous bins above `th` Hz lasting at least `min_len_ms` -- with
    the time axis taken from the actual array length rather than a constant."""
    n = len(rate_ms) // bin_ms
    binned = rate_ms[:n * bin_ms].reshape(n, bin_ms).mean(axis=1)
    hot = binned >= th
    out, i = [], 0
    while i < n:
        if hot[i]:
            j = i
            while j < n and hot[j]:
                j += 1
            if (j - i) * bin_ms >= min_len_ms:
                out.append((i * bin_ms, j * bin_ms))
            i = j
        else:
            i += 1
    return out


def _run_once(mode, scale, seed=12345):
    """One cued run at a given global weight scale. Returns (rate, spikes)."""
    import memval.vendor.ca3net as ca3net
    from memval.vendor.ca3net import spw_network as spw
    from memval.vendor.ca3net.helper import load_wmx, preprocess_monitors

    _, centres = build_chain()
    spw.CUE_TARGET = cue_block_for(centres[0])
    ca3net.configure(sim_duration_ms=10000)
    wmx = load_wmx(os.path.join(OUT, "wmx_%s.npz" % mode))
    wmx = wmx.multiply(scale).tocoo()
    SM, SMB, RM, RMB, sel, stm, stmb = spw.run_simulation(
        wmx, "sym", cue=True, save=False, seed=seed, verbose=False)
    return preprocess_monitors(SM, RM)


def stage_calibrate(mode):
    """The learned matrix is far stronger than upstream's (our chain is denser
    than simulated maze running), so the network saturates. Sweep a global scale
    until the dynamics are back in the discrete-event regime -- the same
    bistability window found during the 2026-08-28 evaluation: silent below,
    one run-long event above, discrete SWR-like events in between."""
    print("%-8s %-10s %-8s %s" % ("scale", "rate(Hz)", "events", "durations(ms)"), flush=True)
    scales = [float(x) for x in os.environ.get(
        "CA3_SCALES", "0.02,0.05,0.1,0.2,0.35,0.5").split(",")]
    for scale in scales:
        st, ni, rate, _, _ = _run_once(mode, scale)
        evs = _events_from_rate(rate)
        durs = [b - a for a, b in evs]
        print("%-8.2f %-10.2f %-8d %s"
              % (scale, np.mean(rate), len(evs), durs[:8]), flush=True)


def stage_recall(mode, scale=1.0, seeds=(12345, 23456, 34567), n_seeds=None):
    if n_seeds is not None:
        seeds = tuple(10000 + 1111 * k for k in range(n_seeds))
    import memval.vendor.ca3net as ca3net
    from memval.vendor.ca3net import spw_network as spw
    from memval.vendor.ca3net.helper import load_wmx, preprocess_monitors

    fwd = rev = 0
    print(" %-6s %-8s %-9s %-9s %-9s %s"
          % ("seed", "t0", "start_id", "end_id", "delta", "dir"))
    for seed in seeds:
        t0 = time.time()
        st, ni, rate, _, _ = _run_once(mode, scale, seed=seed)
        evs = _events_from_rate(rate)
        for a, b in evs:
            m = (st >= a) & (st < b)
            t, i = st[m], ni[m]
            if len(t) < 50:
                continue
            e = np.median(i[t < a + 80]); l = np.median(i[t > b - 80])
            d = l - e
            fwd += d > 0; rev += d <= 0
            print(" %-6d %-8d %-9.0f %-9.0f %-9.0f %s"
                  % (seed, a, e, l, d, "forward" if d > 0 else "reverse"), flush=True)
        print("   (seed %d: %d events, %.2f Hz, %.0fs)"
              % (seed, len(evs), np.mean(rate), time.time() - t0), flush=True)
        np.savez(os.path.join(OUT, "recall_%s_%d.npz" % (mode, seed)),
                 spike_times=st, spiking_neurons=ni, rate=rate)
    total = fwd + rev
    frac = fwd / total if total else float("nan")
    print("\nMODE=%s scale=%.2f  forward %d / %d = %.2f   GATE(>=0.80): %s"
          % (mode, scale, fwd, total, frac, "PASS" if frac >= 0.80 else "FAIL"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["learn", "calibrate", "recall"], required=True)
    ap.add_argument("--mode", choices=["sym", "asym"], required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="global multiplier on the learned weights (see --stage calibrate)")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="number of seeds for --stage recall (default: 3 fixed seeds)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage == "learn":
        stage_learn(a.mode)
    elif a.stage == "calibrate":
        stage_calibrate(a.mode)
    else:
        stage_recall(a.mode, scale=a.scale, n_seeds=a.n_seeds)
