"""
Scale test: 5 sequences x 10 items, retention MATRIX per model.

Trains 5 disjoint-category lists in sequence (no reset) and, after finishing
each list j, measures noise-free recall MRR of every list i<=j — the standard
continual-learning retention matrix R[j][i]. Reading DOWN column i shows how
list i decays as later lists are added; the diagonal is just-learned recall.

Compares Original / DG / XdG with use_output_bias=False (the established setting;
bias-on collapses all three via the shared b_o and hides the differences). Pushes
capacity: 50 items / 45 transitions through n_hidden=256 (XdG gate 0.1 -> ~26
units/item), so cumulative subnetwork coverage becomes the binding constraint.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork
from memval.models.baselines.dg_original_eqprop import DGOriginalEqPropSequenceNetwork
from memval.models.baselines.dg_xdg_eqprop import DGXdGEqPropSequenceNetwork
from memval.diagnostics import transitions_of, retrieval_mrr

CATS = ["alpha", "beta", "gamma", "delta", "epsilon"]
LISTS = [[f"{c}{k}" for k in range(10)] for c in CATS]
VOCAB = {item: c for c, items in zip(CATS, LISTS) for item in items}
CANDIDATES = [w for lst in LISTS for w in lst]        # full 50-word pool
DIM = 100
N_HIDDEN = 256
N_DG = 256
DG_SPARSITY = 0.05
GATE_SPARSITY = 0.1
EPOCHS = 100
USE_BIAS = False
SEEDS = [0, 1, 2]
MODELS = ("original", "dg", "xdg")
NL = len(LISTS)


class EncDec:
    def __init__(self):
        self.enc = SymbolicEncoder(VOCAB, embedding_dim=DIM, category_variance=0.2, seed=42)
        self.dec = SymbolicDecoder(self.enc)

    def encode(self, seq):
        return self.enc.encode(seq)

    def decode(self, vec):
        return self.dec.decode(vec, top_k=1)[0]


def build(model_type, seed):
    common = dict(n_features=DIM, n_hidden=N_HIDDEN, learning_rate=0.1, beta=0.5,
                  n_settle_steps=40, n_epochs=EPOCHS, seed=seed, use_output_bias=USE_BIAS)
    if model_type == "original":
        return OriginalEqPropSequenceNetwork(**common)
    if model_type == "dg":
        return DGOriginalEqPropSequenceNetwork(
            n_dg=N_DG, dg_target_sparsity=DG_SPARSITY, dg_seed=seed, **common)
    return DGXdGEqPropSequenceNetwork(
        n_dg=N_DG, dg_target_sparsity=DG_SPARSITY, gate_sparsity=GATE_SPARSITY,
        dg_seed=seed, gate_seed=seed, **common)


def run_task(args):
    model_type, seed = args
    enc = EncDec()
    model = build(model_type, seed)
    R = np.full((NL, NL), np.nan)
    for j in range(NL):
        model.fit_sequence(enc.encode(LISTS[j]), epochs=EPOCHS)
        for i in range(j + 1):
            R[j, i] = np.mean(retrieval_mrr(model, enc, transitions_of(LISTS[i]), CANDIDATES))
    return {"model": model_type, "R": R}


def main():
    tasks = [(m, s) for m in MODELS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = list(ex.map(run_task, tasks))

    print(f"SCALE: {NL} sequences x {len(LISTS[0])} items; noise-free MRR retention matrix; "
          f"n_hidden={N_HIDDEN}, n_dg={N_DG}, gate={GATE_SPARSITY}, use_output_bias={USE_BIAS}, "
          f"epochs={EPOCHS}, seeds={SEEDS}")
    print("R[j,i] = recall MRR of list i after training through list j. "
          "Read DOWN a column = decay of that list as more are added.\n")

    for mt in MODELS:
        Rs = np.stack([r["R"] for r in rows if r["model"] == mt])
        R = np.nanmean(Rs, axis=0)
        diag = np.nanmean(np.diag(R))
        below = R[np.tril_indices(NL, -1)]
        final_old = R[NL - 1, :NL - 1]                 # earlier lists after all 5 trained
        print(f"=== {mt.upper()} ===   just-learned(diag)={diag:.2f}  "
              f"forgetting(below-diag mean)={np.nanmean(below):.2f}  "
              f"final retention of L0..L3={np.nanmean(final_old):.2f}")
        header = "            " + "".join(f"  L{i:<4}" for i in range(NL))
        print(header)
        for j in range(NL):
            cells = "".join((f"  {R[j, i]:.2f} " if not np.isnan(R[j, i]) else "       ")
                            for i in range(NL))
            print(f"  after L{j} {cells}")
        print()


if __name__ == "__main__":
    main()
