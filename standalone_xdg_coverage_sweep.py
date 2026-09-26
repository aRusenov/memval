"""
XdG coverage sweep (v2): does retention track the cumulative-coverage law (1-p)^n_B?

Improvements over v1 (which gave a noisy, non-monotonic behavioural curve):
  * PER-TRANSITION ONE-STEP metric instead of 5-step autoregressive recall. Each
    a-transition (x->y) is scored independently: predict_next(x) -> decode -> ==y.
    No autoregressive drift (v1's a_after_a=0.60 was ~0.9^5 drift, not weak
    learning), and pooling n_a x n_seeds binary points gives fine resolution.
  * 15 seeds (was 3) -> small standard error per config.
  * Process-parallel across (p, seed), single-threaded BLAS per worker.

Design unchanged: per-item capacity held fixed (constant gate_k), p varied by
n_hidden = gate_k / p, so retention changes reflect DISJOINTNESS not capacity.
"""

import os

# single-threaded BLAS per worker (set before numpy import) so process-level
# parallelism across seeds doesn't oversubscribe the cores.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import string
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from memval.models.baselines.dg_xdg_eqprop import DGXdGEqPropSequenceNetwork


# ----------------------------- config -----------------------------
LIST_LEN = 6
DIM = 64
N_DG = 128
DG_SPARSITY = 0.05
GATE_K = 20                              # active hidden units per item — FIXED
P_SWEEP = [0.20, 0.10, 0.05, 0.033]     # n_hidden = GATE_K / p  (<=~606)
N_EPOCHS = 300
N_SETTLE = 30
LR = 0.1
BETA = 0.5
N_SEEDS = 15
ACT_EPS = 1e-6


class ToyEncoder:
    def __init__(self, symbols, dim, seed):
        rng = np.random.default_rng(seed)
        raw = np.abs(rng.standard_normal((len(symbols), dim)))
        self.codes = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self.symbols = list(symbols)
        self.idx = {s: i for i, s in enumerate(self.symbols)}

    def encode(self, seq):
        return np.stack([self.codes[self.idx[s]] for s in seq])

    def decode(self, vec):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        return self.symbols[int(np.argmax(self.codes @ v))]


def transitions_of(seq):
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def one_step(model, enc, transitions):
    """Per-transition one-step decode accuracy + cosine fidelity (no drift)."""
    accs, fids = [], []
    for x, y in transitions:
        pred = model.predict_next(enc.encode([x])[0])
        accs.append(1.0 if enc.decode(pred) == y else 0.0)
        pv = pred / (np.linalg.norm(pred) + 1e-12)
        tv = enc.encode([y])[0]
        tv = tv / (np.linalg.norm(tv) + 1e-12)
        fids.append(float(pv @ tv))
    return accs, fids


def hidden_act(model, x):
    s_h, _ = model._settle(model._separate(x), target=None, beta=0.0)
    return s_h


def active_sets(vectors):
    return [frozenset(np.nonzero(v > ACT_EPS)[0].tolist()) for v in vectors]


def mean_pairwise_jaccard(sets_a, sets_b):
    vals = [len(A & B) / len(A | B) if (A | B) else 0.0
            for A in sets_a for B in sets_b]
    return float(np.mean(vals))


def wo_block_cos(model, trans, enc, a_idx, b_idx):
    sigs = []
    for x, y in trans:
        _, d_ho, _, _ = model._transition_deltas(enc.encode([x])[0], enc.encode([y])[0])
        sigs.append(d_ho.ravel() / (np.linalg.norm(d_ho) + 1e-12))
    G = np.vstack(sigs)
    J = G @ G.T
    return float(np.mean(np.abs(J[np.ix_(a_idx, b_idx)])))


def run_task(args):
    """One (p, seed) run. Returns per-transition lists so seeds pool cleanly."""
    p, seed = args
    letters = list(string.ascii_uppercase)
    list_a = letters[:LIST_LEN]
    list_b = letters[LIST_LEN:2 * LIST_LEN]
    trans_a, trans_b = transitions_of(list_a), transitions_of(list_b)
    trans = trans_a + trans_b
    na = len(trans_a)
    a_idx, b_idx = list(range(na)), list(range(na, len(trans)))

    n_hidden = int(round(GATE_K / p))
    enc = ToyEncoder(list_a + list_b, dim=DIM, seed=seed)
    model = DGXdGEqPropSequenceNetwork(
        n_features=DIM, n_hidden=n_hidden, n_dg=N_DG,
        dg_target_sparsity=DG_SPARSITY, gate_sparsity=p,
        learning_rate=LR, beta=BETA, n_settle_steps=N_SETTLE, n_epochs=N_EPOCHS,
        seed=seed, dg_seed=seed, gate_seed=seed)

    # mechanism metrics at init
    hid_a = [hidden_act(model, x) for x in enc.encode(list_a)]
    hid_b = [hidden_act(model, x) for x in enc.encode(list_b)]
    hidden_jacc = mean_pairwise_jaccard(active_sets(hid_a), active_sets(hid_b))
    wo_cos = wo_block_cos(model, trans, enc, a_idx, b_idx)

    model.fit_sequence(enc.encode(list_a))
    base_acc, _ = one_step(model, enc, trans_a)          # list a, before b
    model.fit_sequence(enc.encode(list_b))
    ret_acc, ret_fid = one_step(model, enc, trans_a)     # list a, after b
    b_acc, _ = one_step(model, enc, trans_b)             # list b, after b

    return {"p": p, "n_hidden": n_hidden, "hidden_jacc": hidden_jacc, "wo_cos": wo_cos,
            "base_acc": base_acc, "ret_acc": ret_acc, "ret_fid": ret_fid, "b_acc": b_acc}


def main():
    tasks = [(p, s) for p in P_SWEEP for s in range(N_SEEDS)]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        results = list(ex.map(run_task, tasks))

    n_B = LIST_LEN - 1
    print(f"gate_k={GATE_K} (fixed), n_B={n_B}, list_len={LIST_LEN}, n_dg={N_DG}, "
          f"seeds={N_SEEDS}, one-step per-transition metric")
    print(f"\n{'p':>6} {'n_hid':>6} {'(1-p)^nB':>9} | {'base_a':>7} "
          f"{'retain_a':>8} {'±se':>5} {'ret_fid':>7} {'b_learn':>7} | "
          f"{'hidJacc':>8} {'W_ho cos':>8}")
    print("-" * 92)

    for p in P_SWEEP:
        rows = [r for r in results if r["p"] == p]
        pred = (1.0 - p) ** n_B
        base = np.mean([a for r in rows for a in r["base_acc"]])
        ret = np.array([a for r in rows for a in r["ret_acc"]])
        retm, se = ret.mean(), ret.std() / np.sqrt(len(ret))
        rfid = np.mean([f for r in rows for f in r["ret_fid"]])
        blearn = np.mean([a for r in rows for a in r["b_acc"]])
        hj = np.mean([r["hidden_jacc"] for r in rows])
        wc = np.mean([r["wo_cos"] for r in rows])
        print(f"{p:>6.3f} {rows[0]['n_hidden']:>6d} {pred:>9.3f} | {base:>7.2f} "
              f"{retm:>8.2f} {se:>5.2f} {rfid:>7.2f} {blearn:>7.2f} | "
              f"{hj:>8.3f} {wc:>8.3f}")


if __name__ == "__main__":
    main()
