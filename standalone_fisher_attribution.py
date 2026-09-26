"""
Per-parameter-group Fisher attribution: WHERE does the residual forgetting live?

For each model (Original / DG / DG+XdG) we estimate how much training list b raises
list a's loss, per parameter group, using the EWC second-order estimate:

    dL_A  ~=  1/2 * sum_i  F_A[i] * (theta_B[i] - theta_A[i])^2

  * F_A[i]           = diagonal Fisher of list a (importance of weight i to a),
                       computed at the POST-a weights as mean over a's transitions
                       of the squared EP two-phase gradient (model._transition_deltas).
  * theta_A          = weights right after training a.
  * theta_B          = weights right after training b (no reset).
  * (theta_B-theta_A)= the ACTUAL displacement training b caused.

The per-group share of that sum is the forgetting attribution. Split into
W_ih, W_ho, b_h, b_o — the biases are what the W_ih/W_ho cosine split cannot see.

Prediction: gating (XdG) crushes the W_ih and W_ho contributions (disjoint
supports -> F_A and the displacement no longer co-locate), leaving the ungated
shared b_o as the dominant residual leak — the ceiling the coverage plateau hit.

Decomposition reported per group:
  * contrib%   : share of total F-weighted displacement  (the forgetting attribution)
  * Fmass%     : share of A's total Fisher mass           (where a's importance lives)
  * disp%      : share of b's total squared displacement  (where b moved weights)
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import string
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork
from memval.models.baselines.dg_original_eqprop import DGOriginalEqPropSequenceNetwork
from memval.models.baselines.dg_xdg_eqprop import DGXdGEqPropSequenceNetwork


# ----------------------------- config -----------------------------
LIST_LEN = 6
DIM = 64
N_HIDDEN = 128
N_DG = 128
DG_SPARSITY = 0.05
GATE_SPARSITY = 0.1
N_EPOCHS = 300
LR = 0.1
BETA = 0.5
N_SEEDS = 5
GROUPS = ("W_ih", "W_ho", "b_h", "b_o")


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


def one_step_acc(model, enc, transitions):
    return float(np.mean([
        enc.decode(model.predict_next(enc.encode([x])[0])) == y
        for x, y in transitions]))


def fisher(model, codes):
    """Diagonal Fisher per group at current weights = mean squared EP gradient."""
    F = {g: np.zeros_like(getattr(model, g)) for g in GROUPS}
    n = codes.shape[0] - 1
    for t in range(n):
        d_ih, d_ho, d_bh, d_bo = model._transition_deltas(codes[t], codes[t + 1])
        F["W_ih"] += d_ih ** 2
        F["W_ho"] += d_ho ** 2
        F["b_h"] += d_bh ** 2
        F["b_o"] += d_bo ** 2
    return {g: F[g] / max(n, 1) for g in GROUPS}


def snapshot(model):
    return {g: getattr(model, g).copy() for g in GROUPS}


def build(model_type, seed):
    common = dict(n_features=DIM, n_hidden=N_HIDDEN, learning_rate=LR, beta=BETA,
                  n_settle_steps=50, n_epochs=N_EPOCHS, seed=seed)
    if model_type == "original":
        return OriginalEqPropSequenceNetwork(**common)
    if model_type == "dg":
        return DGOriginalEqPropSequenceNetwork(
            n_dg=N_DG, dg_target_sparsity=DG_SPARSITY, dg_seed=seed, **common)
    if model_type == "xdg":
        return DGXdGEqPropSequenceNetwork(
            n_dg=N_DG, dg_target_sparsity=DG_SPARSITY, gate_sparsity=GATE_SPARSITY,
            dg_seed=seed, gate_seed=seed, **common)
    raise ValueError(model_type)


def run_task(args):
    model_type, seed = args
    letters = list(string.ascii_uppercase)
    list_a, list_b = letters[:LIST_LEN], letters[LIST_LEN:2 * LIST_LEN]
    enc = ToyEncoder(list_a + list_b, dim=DIM, seed=seed)
    a_codes, b_codes = enc.encode(list_a), enc.encode(list_b)

    model = build(model_type, seed)
    model.fit_sequence(a_codes)
    theta_A = snapshot(model)
    F_A = fisher(model, a_codes)                 # importance of a, at post-a weights
    base = one_step_acc(model, enc, transitions_of(list_a))

    model.fit_sequence(b_codes)
    theta_B = snapshot(model)
    retain = one_step_acc(model, enc, transitions_of(list_a))

    # per-group second-order forgetting terms
    contrib, fmass, disp = {}, {}, {}
    for g in GROUPS:
        d2 = (theta_B[g] - theta_A[g]) ** 2
        contrib[g] = float((F_A[g] * d2).sum())
        fmass[g] = float(F_A[g].sum())
        disp[g] = float(d2.sum())
    return {"model": model_type, "base": base, "retain": retain,
            "contrib": contrib, "fmass": fmass, "disp": disp}


def main():
    tasks = [(mt, s) for mt in ("original", "dg", "xdg") for s in range(N_SEEDS)]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        results = list(ex.map(run_task, tasks))

    print(f"list_len={LIST_LEN}, n_hidden={N_HIDDEN}, n_dg={N_DG}, "
          f"gate_sparsity={GATE_SPARSITY}, seeds={N_SEEDS}")
    print("forgetting attribution: contrib% = share of  sum_i F_A[i]*(dtheta_B[i])^2\n")

    for mt in ("original", "dg", "xdg"):
        rows = [r for r in results if r["model"] == mt]
        base = np.mean([r["base"] for r in rows])
        ret = np.mean([r["retain"] for r in rows])

        def share(field):
            # average the per-seed normalised shares (each seed sums to 1)
            out = {g: [] for g in GROUPS}
            for r in rows:
                tot = sum(r[field][g] for g in GROUPS) or 1.0
                for g in GROUPS:
                    out[g].append(r[field][g] / tot)
            return {g: 100 * np.mean(out[g]) for g in GROUPS}

        c, f, d = share("contrib"), share("fmass"), share("disp")
        print(f"{mt.upper():>8}   base_a={base:.2f}  retain_a={ret:.2f}")
        print(f"           {'':>8} " + "".join(f"{g:>9}" for g in GROUPS))
        print(f"           contrib% " + "".join(f"{c[g]:>9.1f}" for g in GROUPS))
        print(f"           Fmass%   " + "".join(f"{f[g]:>9.1f}" for g in GROUPS))
        print(f"           disp%    " + "".join(f"{d[g]:>9.1f}" for g in GROUPS))
        print()


if __name__ == "__main__":
    main()
