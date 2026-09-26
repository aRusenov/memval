"""
Catastrophic forgetting under the NOISE-FREE graded metrics.

Train list A, then list B (no reset), and score list A's retention with the
noise-free cosine margin (primary) + reciprocal rank — so we see forgetting in
margin space, not just top-1. Compares Original vs DG on the real SymbolicEncoder.

Questions:
  * Is forgetting GRACEFUL (margins shrink toward 0) or CATASTROPHIC (margins go
    strongly negative)?
  * Does DG's input separation protect list A's margins better than the plain net,
    or does the shared readout/b_o wipe them out anyway?
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
from memval.diagnostics import transitions_of, retrieval_margin, retrieval_mrr, one_step_accuracy

LIST_A = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']
LIST_B = ['cat', 'dog', 'cow', 'horse', 'sheep', 'pig', 'lion']
VOCAB = {w: 'fruit' for w in LIST_A} | {w: 'animal' for w in LIST_B}
CANDIDATES = LIST_A + LIST_B
DIM = 100
N_HIDDEN = 256
N_DG = 256
DG_SPARSITY = 0.05
GATE_SPARSITY = 0.1
EPOCHS = 150
SEEDS = [0, 1, 2, 3, 4]
MODELS = ("original", "dg", "xdg")


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
                  n_settle_steps=40, n_epochs=EPOCHS, seed=seed)
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
    ta, tb = transitions_of(LIST_A), transitions_of(LIST_B)
    model = build(model_type, seed)

    model.fit_sequence(enc.encode(LIST_A), epochs=EPOCHS)
    base_margin = retrieval_margin(model, enc, ta, CANDIDATES)
    base_mrr = retrieval_mrr(model, enc, ta, CANDIDATES)

    model.fit_sequence(enc.encode(LIST_B), epochs=EPOCHS)
    ret_margin = retrieval_margin(model, enc, ta, CANDIDATES)
    ret_mrr = retrieval_mrr(model, enc, ta, CANDIDATES)
    b_margin = retrieval_margin(model, enc, tb, CANDIDATES)
    b_top1 = one_step_accuracy(model, enc, tb)

    return {"model": model_type,
            "base_margin": base_margin, "base_mrr": base_mrr,
            "ret_margin": ret_margin, "ret_mrr": ret_mrr,
            "b_margin": b_margin, "b_top1": b_top1}


def stats(margins):
    m = np.array(margins)
    return (f"mean={m.mean():+.3f}  median={np.median(m):+.3f}  min={m.min():+.3f}  "
            f"frac_neg={np.mean(m < 0):.2f}")


def main():
    tasks = [(m, s) for m in MODELS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = list(ex.map(run_task, tasks))

    print(f"A->B forgetting, NOISE-FREE metrics; real SymbolicEncoder; "
          f"n_hidden={N_HIDDEN}, n_dg={N_DG}, epochs={EPOCHS}, seeds={SEEDS}\n")

    for mt in MODELS:
        rs = [r for r in rows if r["model"] == mt]
        bm = [m for r in rs for m in r["base_margin"]]
        rm = [m for r in rs for m in r["ret_margin"]]
        bmmrr = np.mean([m for r in rs for m in r["base_mrr"]])
        rmmrr = np.mean([m for r in rs for m in r["ret_mrr"]])
        blm = np.mean([m for r in rs for m in r["b_margin"]])
        # paired per-transition margin drop
        drop = np.array([b - r for row in rs
                         for b, r in zip(row["base_margin"], row["ret_margin"])])
        print(f"=== {mt.upper()} ===")
        print(f"  list A margin  BEFORE B : {stats(bm)}   MRR={bmmrr:.3f}")
        print(f"  list A margin  AFTER  B : {stats(rm)}   MRR={rmmrr:.3f}")
        print(f"  per-transition margin drop (before-after): "
              f"mean={drop.mean():+.3f}  median={np.median(drop):+.3f}")
        print(f"  list B margin (learned) : mean={blm:+.3f}   top1={np.mean([r['b_top1'] for r in rs]):.3f}")
        print()


if __name__ == "__main__":
    main()
