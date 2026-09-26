"""
Standalone: does a DG expansion layer fix the weight-overlap that causes CF?

Compares `OriginalEqPropSequenceNetwork` vs `DGOriginalEqPropSequenceNetwork` on
the same A/B forgetting task (train list a, then list b on the same weights), and
instruments the mechanism at three levels so we can see *where* separation is
achieved and *where* it leaks:

  1. INPUT-layer Jaccard  (A-items vs B-items): overlap of the code fed to W_ih.
       original = raw encoding (dense) ; DG = sparse separated code.
       -> "did the separator disjoin the inputs?"
  2. HIDDEN-layer Jaccard (A vs B): overlap of active hidden units.
       -> "did that separation SURVIVE into the hidden representation?"
  3. Weight-update cosine (a<->b block), split into W_ih and W_ho:
       -> "which weight matrix actually collides?"  W_ho is the readout, where
          forgetting is decided.

Predicted signature of the known plain-DG failure mode:
  input Jaccard: original ~1  ->  DG ~floor          (DG separates the input)
  hidden Jaccard: original ~1  ->  DG still ~1        (did NOT propagate)
  W_ih a<->b cos: original high ->  DG ~0             (tracks input Jaccard)
  W_ho a<->b cos: high for BOTH                       (tracks hidden Jaccard)
  behaviour:      both forget                         (readout untouched by DG)

All representational + interference metrics are read at INIT (codes are input-
driven; a learned transition's update shrinks to ~0 and its direction becomes
meaningless, so init is the honest reference). Behaviour is measured after
training. Non-negative unit codes are used so targets are representable under the
[0,1] state clip.
"""

import string

import numpy as np

from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork
from memval.models.baselines.dg_original_eqprop import DGOriginalEqPropSequenceNetwork
from memval.models.baselines.dg_xdg_eqprop import DGXdGEqPropSequenceNetwork


# ----------------------------- config -----------------------------
LIST_LEN = 6
DIM = 64
N_HIDDEN = 128        # wider so each gated subnetwork still has capacity
N_DG = 256
DG_SPARSITY = 0.05
GATE_SPARSITY = 0.1   # XdG: fraction of hidden units active per item
N_EPOCHS = 350
LR = 0.1
BETA = 0.5
SEEDS = range(4)
ACT_EPS = 1e-6        # "active unit" threshold


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


def recall_accuracy(model, enc, seq):
    codes = enc.encode(seq)
    recalled = model.recall(codes[0:1], length=len(seq) - 1)
    decoded = [enc.decode(r) for r in recalled]
    return float(np.mean([d == t for d, t in zip(decoded, seq[1:])]))


# ---- representational hooks (model-aware) ----
def input_code(model, x):
    """Code actually fed to W_ih: DG-separated for the DG net, raw otherwise."""
    return model._separate(x) if hasattr(model, "_separate") else np.asarray(x)


def hidden_act(model, x):
    """Free-phase hidden activation for raw input x."""
    s_h, _ = model._settle(input_code(model, x), target=None, beta=0.0)
    return s_h


# ---- set / Jaccard helpers ----
def active_sets(vectors):
    return [frozenset(np.nonzero(v > ACT_EPS)[0].tolist()) for v in vectors]


def mean_pairwise_jaccard(sets_a, sets_b):
    vals = []
    for A in sets_a:
        for B in sets_b:
            u = len(A | B)
            vals.append(len(A & B) / u if u else 0.0)
    return float(np.mean(vals))


def mean_sparsity(vectors, n):
    return float(np.mean([np.count_nonzero(v > ACT_EPS) for v in vectors]) / n)


def jaccard_floor(pa, pb):
    """Expected Jaccard of two random active sets with densities pa, pb."""
    denom = pa + pb - pa * pb
    return (pa * pb) / denom if denom > 0 else 0.0


# ---- weight-update signatures, split by matrix, at current weights ----
def update_sigs(model, transitions, enc):
    """Per-transition (dW_ih, dW_ho) via the model's own _transition_deltas
    (DG-aware). Returns two lists of flattened, L2-normalised direction vectors."""
    ih, ho = [], []
    for x, y in transitions:
        d_ih, d_ho, _, _ = model._transition_deltas(enc.encode([x])[0], enc.encode([y])[0])
        ih.append(d_ih.ravel() / (np.linalg.norm(d_ih) + 1e-12))
        ho.append(d_ho.ravel() / (np.linalg.norm(d_ho) + 1e-12))
    return np.vstack(ih), np.vstack(ho)


def block_mean_abs_cos(G, a_idx, b_idx):
    J = G @ G.T
    return float(np.mean(np.abs(J[np.ix_(a_idx, b_idx)])))


def run_model(make_model, enc, list_a, list_b, seed):
    model = make_model(seed)
    trans = transitions_of(list_a) + transitions_of(list_b)
    na = len(list_a) - 1
    a_idx, b_idx = list(range(na)), list(range(na, len(trans)))

    a_codes = enc.encode(list_a)
    b_codes = enc.encode(list_b)

    # --- representational Jaccard at init (input & hidden) ---
    in_a = [input_code(model, x) for x in a_codes]
    in_b = [input_code(model, x) for x in b_codes]
    n_in = len(in_a[0])
    hid_a = [hidden_act(model, x) for x in a_codes]
    hid_b = [hidden_act(model, x) for x in b_codes]

    p_in = mean_sparsity(in_a + in_b, n_in)
    p_hid = mean_sparsity(hid_a + hid_b, N_HIDDEN)
    jacc = {
        "input": mean_pairwise_jaccard(active_sets(in_a), active_sets(in_b)),
        "input_floor": jaccard_floor(p_in, p_in),
        "input_density": p_in,
        "hidden": mean_pairwise_jaccard(active_sets(hid_a), active_sets(hid_b)),
        "hidden_floor": jaccard_floor(p_hid, p_hid),
        "hidden_density": p_hid,
    }

    # --- weight-update interference at init, split by matrix ---
    G_ih, G_ho = update_sigs(model, trans, enc)
    cos = {
        "W_ih": block_mean_abs_cos(G_ih, a_idx, b_idx),
        "W_ho": block_mean_abs_cos(G_ho, a_idx, b_idx),
    }

    # --- behaviour: train a, then b (no reset) ---
    model.fit_sequence(a_codes)
    acc_a1 = recall_accuracy(model, enc, list_a)
    model.fit_sequence(b_codes)
    acc_a2 = recall_accuracy(model, enc, list_a)
    acc_b = recall_accuracy(model, enc, list_b)

    return {"jacc": jacc, "cos": cos,
            "acc_a1": acc_a1, "acc_a2": acc_a2, "acc_b": acc_b}


def summarize(name, rows):
    def m(path):
        vals = [r for r in rows]
        for k in path:
            vals = [v[k] for v in vals]
        return np.mean(vals)

    print(f"\n{'='*72}\n{name}\n{'='*72}")
    print("  behaviour (autoregressive recall accuracy)")
    print(f"    list a  after a : {m(['acc_a1']):.2f}")
    print(f"    list a  after b : {m(['acc_a2']):.2f}   <-- retention of a")
    print(f"    list b  after b : {m(['acc_b']):.2f}")
    print("  representational Jaccard  (A-items vs B-items, at init)")
    print(f"    INPUT  layer : {m(['jacc','input']):.3f}   "
          f"(density {m(['jacc','input_density']):.2f}, floor {m(['jacc','input_floor']):.3f})")
    print(f"    HIDDEN layer : {m(['jacc','hidden']):.3f}   "
          f"(density {m(['jacc','hidden_density']):.2f}, floor {m(['jacc','hidden_floor']):.3f})")
    print("  weight-update a<->b overlap  (mean |cos|, at init)")
    print(f"    W_ih (input->hidden) : {m(['cos','W_ih']):.3f}")
    print(f"    W_ho (hidden->output): {m(['cos','W_ho']):.3f}   <-- the readout")


def main():
    letters = list(string.ascii_uppercase)
    list_a = letters[:LIST_LEN]
    list_b = letters[LIST_LEN:2 * LIST_LEN]

    def make_original(seed):
        return OriginalEqPropSequenceNetwork(
            n_features=DIM, n_hidden=N_HIDDEN, learning_rate=LR, beta=BETA,
            n_settle_steps=50, n_epochs=N_EPOCHS, seed=seed)

    def make_dg(seed):
        return DGOriginalEqPropSequenceNetwork(
            n_features=DIM, n_hidden=N_HIDDEN, n_dg=N_DG,
            dg_target_sparsity=DG_SPARSITY, learning_rate=LR, beta=BETA,
            n_settle_steps=50, n_epochs=N_EPOCHS, seed=seed, dg_seed=seed)

    def make_xdg(seed):
        return DGXdGEqPropSequenceNetwork(
            n_features=DIM, n_hidden=N_HIDDEN, n_dg=N_DG,
            dg_target_sparsity=DG_SPARSITY, gate_sparsity=GATE_SPARSITY,
            learning_rate=LR, beta=BETA, n_settle_steps=50, n_epochs=N_EPOCHS,
            seed=seed, dg_seed=seed, gate_seed=seed)

    print(f"list_len={LIST_LEN}, dim={DIM}, n_hidden={N_HIDDEN}, n_dg={N_DG}, "
          f"dg_sparsity={DG_SPARSITY}, gate_sparsity={GATE_SPARSITY}, seeds={list(SEEDS)}")

    orig_rows, dg_rows, xdg_rows = [], [], []
    for s in SEEDS:
        enc = ToyEncoder(list_a + list_b, dim=DIM, seed=s)   # shared codes per seed
        orig_rows.append(run_model(make_original, enc, list_a, list_b, s))
        dg_rows.append(run_model(make_dg, enc, list_a, list_b, s))
        xdg_rows.append(run_model(make_xdg, enc, list_a, list_b, s))

    summarize("ORIGINAL EP", orig_rows)
    summarize("DG-ORIGINAL EP", dg_rows)
    summarize("DG + XdG EP", xdg_rows)


if __name__ == "__main__":
    main()
