#!/usr/bin/env python
"""Does the spiking BCPNN arm recover its paper's regime on a columnar code?

Standalone, fast-iteration companion to ``docs/capacity_report_bcpnn.md`` sec 1.
The arm floors on the shared population codec because that codec is a 2-interval
code per feature with magnitude-graded rates, while Tully et al. 2016 represent
an item as **one minicolumn per hypercolumn at a flat rate** -- the S1 Appendix
calls the general case "a discrete coded or interval coded continuous variable".
This script builds that code and asks how much of the paper's behaviour returns.

Two codes, both in the paper's geometry (``n_hc`` hypercolumns x ``n_mc``
minicolumns x ``n_per_mc`` cells, one minicolumn active per hypercolumn, active
cells Poisson at ``rate`` Hz):

``orthogonal``  item k -> minicolumn k in every hypercolumn. The paper's own
                training patterns; carries no semantics. The ceiling.
``interval``    a real embedding, optionally projected to ``n_hc`` dimensions,
                each dimension quantised into ``n_mc`` equal-width intervals
                over the item set's range; the interval's minicolumn is active.
                Semantically similar items share minicolumns in some
                hypercolumns, which is the overlap the paper disambiguates.

Readout is a nearest-pattern decoder in spike space: an item's score is the
share of readout spikes that fall on its minicolumns. Ranking the items by that
share gives MRR and accuracy for cue -> next item, and cue -> itself.

    python standalone_bcpnn_columnar.py                     # the default sweep
    python standalone_bcpnn_columnar.py --code interval --material symbolic --n-mc 8 --n-hc 20
"""
import argparse
import sys
import time

import numpy as np

from memval.benchmarks.symbolic_pipeline import load_vocab
from memval.encoders.hierarchical import HierarchicalEncoder
from memval.encoders.symbolic import SymbolicEncoder
from memval.models.baselines.bcpnn_spiking import BCPNNSpikingNetwork

FRUIT = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']


# ------------------------------------------------------------------ materials
def material(kind, n_items, seed=42):
    if kind == "symbolic":
        vocab = load_vocab()
        enc = SymbolicEncoder({w: c for w, c in vocab.items() if c == "fruit"},
                              embedding_dim=100, category_variance=0.2, seed=seed)
        V = enc.encode(FRUIT[:n_items])
    elif kind == "hierarchical":
        enc = HierarchicalEncoder(branching=[2, 4], features_per_node=8, seed=seed)
        V = np.asarray(enc.encode(list(enc.items)[:n_items]), dtype=float)
    elif kind == "random":
        V = np.random.default_rng(seed).normal(size=(n_items, 100))
    else:
        raise ValueError(kind)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


# --------------------------------------------------------------------- codecs
class ColumnarCodec:
    """Items -> (n_hc,) minicolumn indices -> (N, T) spike arrays, and back."""

    def __init__(self, n_hc, n_mc, n_per_mc, window_ms=100, rate=200.0, seed=0):
        self.n_hc, self.n_mc, self.n_per_mc = n_hc, n_mc, n_per_mc
        self.window_ms, self.rate = window_ms, rate
        self.n_neurons = n_hc * n_mc * n_per_mc
        self.rng = np.random.default_rng(seed)

    def cells(self, mcs):
        """Cells of the pattern with minicolumn ``mcs[h]`` active in hypercolumn h."""
        base = (np.arange(self.n_hc) * self.n_mc + np.asarray(mcs)) * self.n_per_mc
        return (base[:, None] + np.arange(self.n_per_mc)[None, :]).ravel()

    def encode(self, mcs, rng=None):
        gen = self.rng if rng is None else rng
        S = np.zeros((self.n_neurons, self.window_ms))
        c = self.cells(mcs)
        S[c] = gen.random((c.size, self.window_ms)) < self.rate * 1e-3
        return S

    def scores(self, raster, patterns):
        """Share of readout spikes on each pattern's cells; ``(n_items,)``."""
        per_cell = np.asarray(raster, dtype=float).sum(axis=1)
        total = per_cell.sum()
        if total <= 0:
            return np.zeros(len(patterns))
        return np.array([per_cell[self.cells(p)].sum() / total for p in patterns])


def orthogonal_patterns(n_items, n_hc):
    return [np.full(n_hc, k) for k in range(n_items)]


def interval_patterns(V, n_hc, n_mc, seed=0):
    """Project to ``n_hc`` dims (identity when n_hc == D) and bin each dim."""
    D = V.shape[1]
    if n_hc == D:
        Z = V
    else:
        P = np.random.default_rng(seed).normal(size=(D, n_hc)) / np.sqrt(D)
        Z = V @ P
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    width = np.where(hi > lo, hi - lo, 1.0)
    P = None if n_hc == D else P

    def binner(v):
        z = v if P is None else v @ P
        return np.clip(((z - lo) / width * n_mc).astype(int), 0, n_mc - 1)

    return [binner(V[i]) for i in range(len(V))], binner


def overlap(patterns):
    """Mean fraction of hypercolumns in which two items share a minicolumn."""
    n = len(patterns)
    return float(np.mean([(patterns[a] == patterns[b]).mean()
                          for a in range(n) for b in range(a + 1, n)]))


# ------------------------------------------------------------------------ run
def run(code, kind, n_items=7, n_hc=20, n_mc=8, n_per_mc=10, epochs=20,
        n_trials=5, readout_lag=0, recall_steps=100, seed=0, noise=0.0, verbose=True):
    """``noise``: SD of Gaussian added to the cue *vector* before binning at probe
    time -- the pipeline's cue corruption (0.05 by default there)."""
    t0 = time.time()
    binner = None
    if code == "orthogonal":
        n_mc = max(n_mc, n_items)
        patterns = orthogonal_patterns(n_items, n_hc)
    else:
        V = material(kind, n_items)
        patterns, binner = interval_patterns(V, n_hc, n_mc, seed=seed)
    codec = ColumnarCodec(n_hc, n_mc, n_per_mc, seed=seed)
    m = BCPNNSpikingNetwork(n_hc, n_mc, n_per_mc, seed=seed, n_epochs=epochs,
                            cue_repeats=1, readout_lag_ms=readout_lag,
                            recall_steps=recall_steps, n_basket_per_hc=3, substeps=2)
    for _ in range(epochs):
        m.on_event_boundary()
        for p in patterns:
            m.fit_event(codec.encode(p))
        m.on_event_boundary()
    t_fit = time.time() - t0

    # weights, for the record
    w = m.weights()
    c0, c1 = codec.cells(patterns[0]), codec.cells(patterns[1])
    blk = lambda W, a, b: float(W[np.ix_(a, b)][m.mask[np.ix_(a, b)]].mean())
    w_in, w_fwd = blk(w["ampa"], c0, c0), blk(w["nmda"], c0, c1)

    t0 = time.time()
    ranks_next, ranks_self = [], []
    for trial in range(n_trials):
        gen = np.random.default_rng(1000 + trial)
        for i in range(n_items - 1):
            cue = patterns[i]
            if noise > 0 and binner is not None:
                cue = binner(V[i] + gen.normal(0.0, noise, V.shape[1]))
            r = m.predict_next(codec.encode(cue, rng=gen))
            sc = codec.scores(r, patterns)
            order = np.argsort(-sc)
            ranks_next.append(int(np.where(order == i + 1)[0][0]) + 1)
            ranks_self.append(int(np.where(order == i)[0][0]) + 1)
    t_probe = (time.time() - t0) / len(ranks_next)
    rn, rs = np.array(ranks_next), np.array(ranks_self)
    res = dict(code=code, material=kind if code == "interval" else "-", n_hc=n_hc, n_mc=n_mc, noise=noise,
               n_per_mc=n_per_mc, N=m.n_neurons, overlap=overlap(patterns),
               acc=float(np.mean(rn == 1)), mrr=float(np.mean(1.0 / rn)),
               self_rank1=float(np.mean(rs == 1)), w_in=w_in, w_fwd=w_fwd,
               t_fit=t_fit, t_probe=t_probe)
    if verbose:
        print(f"{code:10s} {res['material']:12s} items={n_items:2d} hc={n_hc:3d} mc={n_mc:2d} per={n_per_mc:2d} N={m.n_neurons:5d} "
              f"noise {noise:.2f} overlap {res['overlap']:.2f} | next: acc {res['acc']:.2f} MRR {res['mrr']:.2f} | "
              f"cue itself rank1 {res['self_rank1']:.2f} | w_in {w_in:5.2f} nS w_fwd {w_fwd:6.3f} nS | "
              f"fit {t_fit:4.0f}s probe {t_probe:.2f}s")
    return res


def verify(n_trials=5):
    """The registry arm ``bcpnn_spiking_columnar`` on the suite's own material.

    Everything here goes through the codec tier and the vector interface the
    pipeline uses: ``CodecBCPNNNetwork(codec="columnar")`` at its registry
    defaults, cues corrupted in R^D, predictions ranked by cosine in R^D.
    """
    import sys as _sys
    _sys.path.insert(0, "bin")
    from run_benchmark import MODEL_REGISTRY
    entry = MODEL_REGISTRY["bcpnn_spiking_columnar"]

    def one(kind, n_items, noise, epochs=20, seed=0):
        V = material(kind, n_items)
        kw = dict(entry["default_kwargs"]); kw["seed"] = seed
        m = entry["class"](n_features=V.shape[1], **kw)
        enc, dec = m.encoder, m.decoder
        # codec's own round trip: does encode -> decode identify the item?
        rt = [int((V @ dec.decode(enc.encode(v, rng=np.random.default_rng(9)))).argmax() == i)
              for i, v in enumerate(V)]
        bins = np.stack([enc.bins(v) for v in V])
        ov = np.mean([(bins[a] == bins[b]).mean() for a in range(n_items) for b in range(a + 1, n_items)])
        t0 = time.time(); m.fit_sequence(V, epochs=epochs); t_fit = time.time() - t0
        rng = np.random.default_rng(1)
        ranks = []
        t0 = time.time()
        for _ in range(n_trials):
            for i in range(n_items - 1):
                y = m.predict_next(V[i] + rng.normal(0.0, noise, V.shape[1]))
                c = V @ y
                ranks.append(int(np.where(np.argsort(-c) == i + 1)[0][0]) + 1)
        t_probe = (time.time() - t0) / len(ranks)
        ranks = np.array(ranks)
        w = m.inner.weights(); c0, c1 = enc.cells(bins[0]), enc.cells(bins[1])
        blk = lambda W, a, b: float(W[np.ix_(a, b)][m.inner.mask[np.ix_(a, b)]].mean())
        print(f"  {kind:12s} items={n_items:2d} noise={noise:.2f} seed={seed} N={m.inner.n_neurons} "
              f"| codec round-trip id {np.mean(rt):.2f} overlap {ov:.2f} | next acc {np.mean(ranks == 1):.2f} "
              f"MRR {np.mean(1 / ranks):.2f} | w_in {blk(w['ampa'], c0, c0):.2f} nS w_fwd {blk(w['nmda'], c0, c1):.3f} nS "
              f"| fit {t_fit:.0f}s probe {t_probe:.2f}s")

    print("registry arm bcpnn_spiking_columnar; chance acc 1/(n-1), MRR ~0.37 at 7 items")
    print("-- clean cues")
    one("symbolic", 7, 0.0); one("hierarchical", 8, 0.0); one("random", 12, 0.0)
    print("-- the pipeline's cue noise (0.05)")
    for sd in (0, 1, 2):
        one("symbolic", 7, 0.05, seed=sd)
    one("hierarchical", 8, 0.05)
    print("-- load, clean and noisy")
    one("random", 20, 0.0); one("random", 20, 0.05)


def tau_p_sweep(taus_s=(5.0, 20.0, 80.0), noise=0.0, n_trials=3, criterion=0.9, ceiling=128,
                b_passes=32):
    """The palimpsest trade-off: how fast list A is acquired vs how much of it
    survives learning list B, as a function of the P-trace time constant.

    A is trained to criterion by staircase (acquisition cost), then B is trained
    for a FIXED ``b_passes`` -- the 32 passes ``multiple_sequences`` spent on it
    when censored, i.e. 22 s of kappa = 1 with A absent -- and A is re-probed.
    B's exposure is fixed rather than to-criterion because with clean cues B
    acquires in one pass, which would leave A untouched at any tau_p and answer
    nothing. Clean cues keep the trade-off unconfounded with the noise ceiling.
    A diagnostic of a parameter the paper itself calls a simulation
    convenience; the registered arm keeps the paper's 5 s.

    Run alone: two concurrent numpy processes oversubscribe the BLAS threads
    and this slows 10-30x.
    """
    import sys as _sys
    _sys.path.insert(0, "bin")
    from run_benchmark import MODEL_REGISTRY
    entry = MODEL_REGISTRY["bcpnn_spiking_columnar"]
    A = material("symbolic", 7)
    vocab = load_vocab()
    encB = SymbolicEncoder({w: c for w, c in vocab.items() if c == "animal"},
                           embedding_dim=100, category_variance=0.2, seed=42)
    B = encB.encode(['cat', 'dog', 'cow', 'horse', 'sheep', 'pig', 'lion'])
    B = B / np.linalg.norm(B, axis=1, keepdims=True)

    def acc(m, V):
        rng = np.random.default_rng(1)
        hits = []
        for _ in range(n_trials):
            for i in range(len(V) - 1):
                y = m.predict_next(V[i] + rng.normal(0.0, noise, V.shape[1]))
                hits.append(int((V @ y).argmax() == i + 1))
        return float(np.mean(hits))

    def to_criterion(m, V):
        """Staircase 1, 2, 4, ... passes (cumulative) until acc >= criterion."""
        done, ep = 0, 1
        while True:
            m.fit_sequence(V, epochs=ep - done)
            done = ep
            a = acc(m, V)
            if a >= criterion or done >= ceiling:
                return done, a
            ep *= 2

    b_time = b_passes * 7 * 0.1
    print(f"tau_p sweep: A = 7 fruit to criterion acc {criterion} (ceiling {ceiling}), then B = 7 animals for "
          f"{b_passes} passes ({b_time:.0f} s of kappa = 1), clean cues")
    print("tau_p    | A: passes to acquire  acc | B acc after its 32 | A after B | B-time / tau_p")
    for tau in taus_s:
        kw = dict(entry["default_kwargs"]); kw["overrides"] = dict(kw["overrides"], tau_p=tau * 1000.0)
        m = entry["class"](n_features=100, **kw)
        t0 = time.time()
        ep_a, acc_a = to_criterion(m, A)
        m.fit_sequence(B, epochs=b_passes)
        acc_b = acc(m, B)
        acc_a_after = acc(m, A)
        print(f"{tau:5.0f} s  | {ep_a:8d}              {acc_a:.2f} |       {acc_b:.2f}         |   {acc_a_after:.2f}    |   {b_time / tau:5.2f}     [{time.time() - t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", choices=["orthogonal", "interval"], default=None)
    ap.add_argument("--material", choices=["symbolic", "hierarchical", "random"], default="symbolic")
    ap.add_argument("--n-hc", type=int, default=20)
    ap.add_argument("--n-mc", type=int, default=8)
    ap.add_argument("--n-per-mc", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--items", type=int, default=7)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--robustness", action="store_true", help="cue noise, load and seeds")
    ap.add_argument("--tau-p", action="store_true",
                    help="retention vs acquisition trade-off across tau_p (diagnostic, not a re-siting)")
    ap.add_argument("--verify", action="store_true",
                    help="the codec-tier ColumnarSpikeEncoder through CodecBCPNNNetwork, scored in R^D")
    args = ap.parse_args()
    if args.verify:
        return verify()
    if args.tau_p:
        return tau_p_sweep()
    if args.code is not None:
        run(args.code, args.material, args.items, args.n_hc, args.n_mc, args.n_per_mc,
            args.epochs, readout_lag=args.lag, seed=args.seed, noise=args.noise)
        return
    if args.robustness:
        print("-- cue noise (the pipeline's 0.05 and above), symbolic 7 items, hc 20 mc 8")
        for nz in (0.05, 0.1, 0.2):
            run("interval", "symbolic", n_hc=20, n_mc=8, noise=nz)
        print("-- load: random unit-norm items, hc 20 / 50, mc 8")
        for n_items, n_hc in [(12, 20), (20, 20), (20, 50)]:
            run("interval", "random", n_items=n_items, n_hc=n_hc, n_mc=8, noise=0.05)
        print("-- hierarchical, all 8 leaves, with cue noise")
        run("interval", "hierarchical", n_items=8, n_hc=20, n_mc=8, noise=0.05)
        print("-- seeds, symbolic hc 20 mc 8, noise 0.05")
        for sd in (1, 2, 3):
            run("interval", "symbolic", n_hc=20, n_mc=8, noise=0.05, seed=sd)
        return
    print("chance: acc 1/6 = 0.17, MRR over 7 items ~0.37\n")
    print("-- ceiling: the paper's own geometry, item-identity code")
    run("orthogonal", "-", n_hc=9, n_mc=8, n_per_mc=30)          # 2160 cells, 270/item
    run("orthogonal", "-", n_hc=20, n_mc=8, n_per_mc=10)         # 1600 cells, 200/item
    print("-- interval code on the real symbolic material (7 fruit words)")
    for n_hc, n_mc in [(20, 4), (20, 8), (50, 8), (100, 8)]:
        run("interval", "symbolic", n_hc=n_hc, n_mc=n_mc, n_per_mc=10 if n_hc <= 50 else 5)
    print("-- interval code on hierarchical material (2 x 4 leaves, siblings share a parent)")
    for n_hc, n_mc in [(20, 8), (50, 8)]:
        run("interval", "hierarchical", n_hc=n_hc, n_mc=n_mc, n_per_mc=10)


if __name__ == "__main__":
    sys.exit(main())
