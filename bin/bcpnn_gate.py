#!/usr/bin/env python
"""Correctness gate for the Tully et al. (2016) spiking BCPNN arm: the paper's own task.

Before any MemVal score from ``BCPNNSpikingNetwork`` is allowed to mean
anything, the reimplementation has to reproduce the paper's own result on the
paper's own network: 9 hypercolumns x 10 minicolumns x 30 AdEx cells, ten
orthogonal patterns trained as an IPI = 0 sequence, then **cued replay in
trained order** (Fig 4D-F, Fig 7) and spontaneous wandering (Fig 3G-I). Recall
is scored the way the paper scores it: attractors detected by Eq 10 with a
25 ms persistence criterion, the conditional response probability (CRP) over
lags -4..5, and the Levenshtein distance of each recalled run from the trained
template (Eq 13).

Modes::

    python bin/bcpnn_gate.py                 # the gate, 3 seeds
    python bin/bcpnn_gate.py --calibrate     # stated gains vs figure-matched gains
    python bin/bcpnn_gate.py --stim          # training rate vs stimulus synapse
    python bin/bcpnn_gate.py --codec         # registry substrates: readout siting

Gate: cued recall must put >= 0.8 of its attractor transitions at lag +1 and
its mean edit distance at <= 2 over the seeds. The paper's IPI = 0 network
has lag-1 CRP ~0.9 (Fig 5B, S3 Fig).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.models.baselines.bcpnn_spiking import BCPNNSpikingNetwork  # noqa: E402

N_HC, N_MC, N_PER = 9, 10, 30
N_PAT = 10
T_STIM = 100


# ------------------------------------------------------------ paper's readout
def pattern_cells(m, k):
    """Cells of pattern k: minicolumn k in every hypercolumn."""
    return np.concatenate([np.arange(h * m.n_pyr_per_hc + k * m.n_per_mc,
                                     h * m.n_pyr_per_hc + (k + 1) * m.n_per_mc)
                           for h in range(m.n_hc)])


def pattern_stimulus(m, k, rng, T=T_STIM, rate=200.0):
    S = np.zeros((m.n_neurons, T))
    c = pattern_cells(m, k)
    S[c] = rng.random((c.size, T)) < rate * 1e-3
    return S


def train(m, epochs, seed, fenced=False):
    """The paper's protocol is cyclic: IPI = 0 throughout, so the wrap-around
    transition 9 -> 0 is learned too and free replay cycles (Fig 3H, 4E).
    ``fenced`` is MemVal's ingestion (a boundary at every pass)."""
    rng = np.random.default_rng(seed)
    m.on_event_boundary()
    for _ in range(epochs):
        if fenced:
            m.on_event_boundary()
        for k in range(N_PAT):
            m.fit_event(pattern_stimulus(m, k, rng))
    if fenced:
        m.on_event_boundary()


def pattern_rates(m, raster, bin_ms=10):
    """``(N_PAT, n_bins)`` mean rate (Hz) of each pattern's cells per bin."""
    nb = raster.shape[1] // bin_ms
    out = np.zeros((N_PAT, nb))
    for k in range(N_PAT):
        c = pattern_cells(m, k)
        out[k] = raster[c, :nb * bin_ms].reshape(c.size, nb, bin_ms).sum(axis=(0, 2)) / (c.size * bin_ms * 1e-3)
    return out


def detect_attractors(rates, c=1.0, persist_bins=3):
    """Eq 10: r_a > c sigma > r_second, held for >= 25 ms. Returns (index, start_bin, n_bins) runs."""
    nb = rates.shape[1]
    active = np.full(nb, -1)
    for t in range(nb):
        r = rates[:, t]
        order = np.argsort(r)[::-1]
        a, second = order[0], order[1]
        sigma = r.std()
        if r[a] > c * sigma > r[second]:
            active[t] = a
    runs = []
    t = 0
    while t < nb:
        if active[t] < 0:
            t += 1
            continue
        s = t
        while t < nb and active[t] == active[s]:
            t += 1
        if t - s >= persist_bins:
            if runs and runs[-1][0] == int(active[s]):
                # The Eq 10 condition flickered inside one attractor; a
                # self-transition is not a transition (paper: lag 0 is
                # "effectively unattainable"), so extend the run.
                k, s0, n0 = runs[-1]
                runs[-1] = (k, s0, t - s0)
            else:
                runs.append((int(active[s]), s, t - s))
    return runs


def crp(seq, n=N_PAT):
    """Conditional response probability over lags -4..5 (paper, Methods)."""
    lags = np.arange(-4, 6)
    counts = np.zeros(len(lags))
    for p, q in zip(seq[:-1], seq[1:]):
        d = (q - p) % n
        lag = d if d <= 5 else d - n
        counts[lag + 4] += 1
    total = counts.sum()
    return lags, counts / total if total else counts


def levenshtein(a, b):
    D = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    D[:, 0] = np.arange(len(a) + 1)
    D[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(D[-1, -1])


def edit_distance(seq, template=tuple(range(N_PAT))):
    """Mean Levenshtein distance of each recalled run (split at the first element)."""
    if not seq:
        return float(N_PAT)
    runs, cur = [], []
    for k in seq:
        if k == template[0] and cur:
            runs.append(cur)
            cur = []
        cur.append(k)
    runs.append(cur)
    return float(np.mean([levenshtein(r, list(template)) for r in runs]))


def score_raster(m, raster):
    rates = pattern_rates(m, raster)
    runs = detect_attractors(rates)
    seq = [r[0] for r in runs]
    lags, c = crp(seq)
    dwell = np.mean([r[2] * 10.0 for r in runs]) if runs else float("nan")
    in_rate = np.mean([rates[r[0], r[1]:r[1] + r[2]].mean() for r in runs]) if runs else float("nan")
    return {"n_attractors": len(runs), "sequence": seq, "crp_lag1": float(c[lags == 1][0]) if runs else 0.0,
            "crp": dict(zip(lags.tolist(), np.round(c, 3).tolist())),
            "edit_distance": edit_distance(seq), "dwell_ms": float(dwell),
            "speed_per_s": float(1000.0 / dwell) if runs else 0.0,
            "attractor_rate_hz": float(in_rate), "mean_rate_hz": float(raster.mean() * 1000.0)}


def weight_summary(m):
    w = m.weights()
    c0, c1, c2 = (pattern_cells(m, k) for k in range(3))
    mask = m.mask
    def blk(W, a, b):
        sub = W[np.ix_(a, b)]
        return float(sub[mask[np.ix_(a, b)]].mean())
    others = np.setdiff1d(np.arange(m.n_neurons), c0)
    return {"ampa_in_pattern_nS": blk(w["ampa"], c0, c0),
            "ampa_other_nS": blk(w["ampa"], others, c0),
            "nmda_forward_nS": blk(w["nmda"], c0, c1), "nmda_backward_nS": blk(w["nmda"], c1, c0),
            "nmda_skip_nS": blk(w["nmda"], c0, c2),
            "I_beta_pA": float(w["I_beta"][c0].mean())}


def one_run(seed, epochs, cued_ms, free_ms, overrides=None, verbose=True, fenced=False):
    t0 = time.time()
    m = BCPNNSpikingNetwork(N_HC, N_MC, N_PER, seed=seed, n_epochs=epochs, overrides=overrides)
    train(m, epochs - 1, seed, fenced=fenced)
    # last epoch by hand, to read the training rate
    rng = np.random.default_rng(seed + 999)
    if fenced:
        m.on_event_boundary()
    rates = []
    for k in range(N_PAT):
        S = pattern_stimulus(m, k, rng)
        out = m._run(S, T_STIM, plastic=True, recurrent=False, rng=m.rng, r_bg=0.0)
        m._flush()
        rates.append(out[pattern_cells(m, k)].mean() * 1000.0)
    m.on_event_boundary()
    res = {"seed": seed, "train_rate_hz": float(np.mean(rates)), "train_wall_s": time.time() - t0}
    res.update(weight_summary(m))
    t0 = time.time()
    cue = pattern_stimulus(m, 0, np.random.default_rng(seed + 7))
    res["cued"] = score_raster(m, m.replay(cue, cued_ms, r_bg=m.p["r_bg_cued"]))
    res["free"] = score_raster(m, m.replay(None, free_ms, r_bg=m.p["r_bg_free"]))
    res["recall_wall_s"] = time.time() - t0
    if verbose:
        print(f"seed {seed}: train {res['train_rate_hz']:.1f} Hz | w AMPA in/other {res['ampa_in_pattern_nS']:.2f}/"
              f"{res['ampa_other_nS']:.2f} nS, NMDA fwd/bwd/skip {res['nmda_forward_nS']:.3f}/"
              f"{res['nmda_backward_nS']:.3f}/{res['nmda_skip_nS']:.3f} nS, I_beta {res['I_beta_pA']:.0f} pA")
        for mode in ("cued", "free"):
            r = res[mode]
            print(f"   {mode:5s}: {r['n_attractors']:3d} attractors, CRP lag1 {r['crp_lag1']:.2f}, D_L {r['edit_distance']:.2f}, "
                  f"dwell {r['dwell_ms']:.0f} ms ({r['speed_per_s']:.1f}/s), attractor rate {r['attractor_rate_hz']:.0f} Hz, "
                  f"mean rate {r['mean_rate_hz']:.1f} Hz")
            print(f"          sequence {r['sequence'][:24]}")
    return res


def gate(args):
    results = [one_run(s, args.epochs, args.cued_ms, args.free_ms, fenced=args.fenced) for s in range(args.seeds)]
    lag1 = np.mean([r["cued"]["crp_lag1"] for r in results])
    dl = np.mean([r["cued"]["edit_distance"] for r in results])
    ok = lag1 >= 0.8 and dl <= 2.0
    print(f"\nGATE {'PASS' if ok else 'FAIL'}: cued CRP lag-1 {lag1:.2f} (>= 0.80), mean D_L {dl:.2f} (<= 2.0) over {args.seeds} seeds")
    return results, ok


def calibrate(args):
    """The same trained network read out under the stated and the figure-matched gains."""
    print("Weights and recall regime under the gains as stated in the text (calib 1/1) "
          "and as matched to the paper's plotted terminal weights (0.3/0.025, the default).")
    out = {}
    for label, ov in (("stated", dict(ampa_calib=1.0, nmda_calib=1.0)), ("figure-matched", None)):
        print(f"\n== {label}")
        out[label] = [one_run(s, args.epochs, args.cued_ms, args.free_ms, overrides=ov, fenced=args.fenced) for s in range(args.seeds)]
    return out


def stim_scan(args):
    print("Training rate of stimulated cells vs the stimulus synapse (paper: 'such that neurons fired at f_max = 20 Hz').")
    out = {}
    for w in (5.0, 10.0, 15.0, 20.0):
        m = BCPNNSpikingNetwork(N_HC, N_MC, N_PER, seed=0, overrides=dict(w_stim=w))
        rng = np.random.default_rng(1)
        r = []
        for k in range(N_PAT):
            o = m._run(pattern_stimulus(m, k, rng), T_STIM, plastic=True, recurrent=False, rng=m.rng, r_bg=0.0)
            r.append(o[pattern_cells(m, k)].mean() * 1000.0)
        out[w] = float(np.mean(r))
        print(f"  w_stim {w:4.0f} nS -> {out[w]:5.1f} Hz")
    return out


# ------------------------------------------------------------- codec siting
def codec_scan(args):
    """Readout siting on the registry's substrates, through the real codec.

    For each substrate and ``n_per_feature``, train one network on a short
    list and score next-item recall for every (lag, window) of the readout,
    so the registry defaults are measured rather than guessed. Reports the
    active cells per item, which is what ``pattern_cells`` must be set to.
    """
    from memval.encoders.hierarchical import HierarchicalEncoder
    from memval.encoders.spike_codec import PopulationSpikeDecoder, PopulationSpikeEncoder
    from memval.encoders.symbolic import SymbolicEncoder

    def material(kind, n_items, seed):
        if kind == "hierarchical":
            enc = HierarchicalEncoder(branching=[2, 3], features_per_node=4, seed=seed)
            V = enc.encode(list(enc.items)[:n_items])
        elif kind == "symbolic":
            vocab = [f"item{i}" for i in range(n_items)]
            enc = SymbolicEncoder(vocab, embedding_dim=100, category_variance=0.2, seed=seed)
            V = enc.encode(vocab)
        elif kind == "orthogonal":
            V = np.eye(max(n_items, 8))[:n_items]
        else:
            raise ValueError(kind)
        return np.asarray(V, dtype=float)

    lags, wins = (0, 25, 50, 100), (50, 100, 150)
    table = {}
    for kind in args.substrates:
        V = material(kind, args.items, 0)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        D = V.shape[1]
        for npf in args.npf:
          for w_stim in args.w_stim:
            enc = PopulationSpikeEncoder(D, n_per_feature=npf, window_steps=50, seed=3)
            dec = PopulationSpikeDecoder(enc)
            active = float(np.mean([(np.abs(v) > 1e-9).sum() for v in V]) * npf)
            m = BCPNNSpikingNetwork(D, 2, npf, seed=1, pattern_cells=active,
                                    overrides=dict(w_stim=w_stim))
            t0 = time.time()
            train_rate = []
            for ep in range(args.epochs):
                m.on_event_boundary()
                for v in V:
                    S = np.tile(enc.encode(v), (1, 2))
                    out = m._run(S, S.shape[1], plastic=True, recurrent=False, rng=m.rng, r_bg=0.0)
                    m._flush()
                    if ep == args.epochs - 1:
                        drv = S.sum(axis=1) > 0
                        train_rate.append(out[drv].mean() * 1000.0)
                m.on_event_boundary()
            wall = time.time() - t0
            train_rate = float(np.mean(train_rate))
            rasters = []
            for i in range(len(V) - 1):
                cue = np.tile(enc.encode(V[i], rng=np.random.default_rng(11)), (1, 2))
                rasters.append(m.replay(cue, 100 + max(lags) + max(wins), r_bg=m.r_bg)[:, 100:])
            best = None
            row = {}
            for lag in lags:
                for w in wins:
                    hit, marg = 0, []
                    for i, r in enumerate(rasters):
                        c = V @ dec.decode(r[:, lag:lag + w])
                        hit += int(c.argmax() == i + 1)
                        marg.append(c[i + 1] - np.delete(c, i + 1).max())
                    row[(lag, w)] = (hit / len(rasters), float(np.mean(marg)))
                    if best is None or row[(lag, w)] > best[1]:
                        best = ((lag, w), row[(lag, w)])
            table[(kind, npf, w_stim)] = {"D": D, "N": m.n_neurons, "active_cells": active, "gain_scale": m.gain_scale,
                                          "w_stim": w_stim, "train_rate_hz": train_rate,
                                          "train_wall_s": wall, "best": best, "rows": row}
            print(f"{kind:12s} npf={npf:2d} w_stim={w_stim:4.0f} D={D:3d} N={m.n_neurons:5d} active/item {active:5.0f} "
                  f"gain_scale {m.gain_scale:5.2f} train {train_rate:5.1f} Hz {wall:4.0f}s | best lag/window {best[0]} "
                  f"acc {best[1][0]:.2f} margin {best[1][1]:+.2f} | "
                  + " ".join(f"{l}/{w}:{row[(l, w)][0]:.2f}" for l in lags for w in wins))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--stim", action="store_true")
    ap.add_argument("--codec", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fenced", action="store_true", help="train with a boundary at every pass (MemVal's ingestion) instead of the paper's cyclic IPI = 0 stream")
    ap.add_argument("--w-stim", nargs="+", type=float, default=[15.0], help="codec mode: stimulus synapse values to scan (nS)")
    ap.add_argument("--epochs", type=int, default=10, help="paper: 50; S7 Fig shows convergence by 10")
    ap.add_argument("--cued-ms", type=int, default=5000)
    ap.add_argument("--free-ms", type=int, default=10000)
    ap.add_argument("--substrates", nargs="+", default=["hierarchical", "symbolic", "orthogonal"])
    ap.add_argument("--npf", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.calibrate:
        out = calibrate(args)
    elif args.stim:
        out = stim_scan(args)
    elif args.codec:
        out = codec_scan(args)
        out = {f"{k[0]}/npf{k[1]}/wstim{k[2]:g}": {kk: (vv if kk != "rows" else {f"{a}/{b}": v for (a, b), v in vv.items()})
                                     for kk, vv in val.items()} for k, val in out.items()}
    else:
        out, _ = gate(args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
