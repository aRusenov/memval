"""
Output-bias ablation via the new `use_output_bias` flag, all 3 models.

b_o is expressively redundant (full W_ho maps each distinct cue->target) and is
the dominant shared forgetting leak on the real benchmark. This runs Original /
DG / XdG with use_output_bias True vs False and scores list-A retention with the
noise-free cosine margin. Question: how does DG (which INVERTS list A with the
bias on) fare with the bias dropped?
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
from memval.diagnostics import transitions_of, retrieval_margin, retrieval_mrr

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


def build(model_type, seed, use_output_bias):
    common = dict(n_features=DIM, n_hidden=N_HIDDEN, learning_rate=0.1, beta=0.5,
                  n_settle_steps=40, n_epochs=EPOCHS, seed=seed,
                  use_output_bias=use_output_bias)
    if model_type == "original":
        return OriginalEqPropSequenceNetwork(**common)
    if model_type == "dg":
        return DGOriginalEqPropSequenceNetwork(
            n_dg=N_DG, dg_target_sparsity=DG_SPARSITY, dg_seed=seed, **common)
    return DGXdGEqPropSequenceNetwork(
        n_dg=N_DG, dg_target_sparsity=DG_SPARSITY, gate_sparsity=GATE_SPARSITY,
        dg_seed=seed, gate_seed=seed, **common)


def run_task(args):
    model_type, bo_on, seed = args
    enc = EncDec()
    ta, tb = transitions_of(LIST_A), transitions_of(LIST_B)
    model = build(model_type, seed, use_output_bias=bo_on)

    model.fit_sequence(enc.encode(LIST_A), epochs=EPOCHS)
    base = retrieval_margin(model, enc, ta, CANDIDATES)
    base_mrr = np.mean(retrieval_mrr(model, enc, ta, CANDIDATES))
    model.fit_sequence(enc.encode(LIST_B), epochs=EPOCHS)
    ret = retrieval_margin(model, enc, ta, CANDIDATES)
    ret_mrr = np.mean(retrieval_mrr(model, enc, ta, CANDIDATES))
    b_learn = np.mean(retrieval_mrr(model, enc, tb, CANDIDATES))
    # sanity: b_o must stay exactly 0 when the bias is off
    bo_norm = float(np.linalg.norm(model.b_o))
    return {"model": model_type, "bo_on": bo_on,
            "base": base, "base_mrr": base_mrr,
            "ret": ret, "ret_mrr": ret_mrr, "b_learn": b_learn, "bo_norm": bo_norm}


def stats(margins):
    m = np.array(margins)
    return f"mean={m.mean():+.3f} median={np.median(m):+.3f} frac_neg={np.mean(m<0):.2f}"


def main():
    tasks = [(m, bo, s) for m in MODELS for bo in (True, False) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = list(ex.map(run_task, tasks))

    print(f"use_output_bias ABLATION, noise-free margins; real SymbolicEncoder; "
          f"n_hidden={N_HIDDEN}, n_dg={N_DG}, gate={GATE_SPARSITY}, epochs={EPOCHS}, seeds={SEEDS}\n")
    for mt in MODELS:
        for bo in (True, False):
            rs = [r for r in rows if r["model"] == mt and r["bo_on"] == bo]
            base = [x for r in rs for x in r["base"]]
            ret = [x for r in rs for x in r["ret"]]
            tag = "b_o ON " if bo else "b_o OFF"
            check = "" if bo else f"  [b_o norm={np.max([r['bo_norm'] for r in rs]):.1e}]"
            print(f"=== {mt.upper():8} {tag} ==={check}")
            print(f"  A before B: {stats(base)}  MRR={np.mean([r['base_mrr'] for r in rs]):.3f}")
            print(f"  A after  B: {stats(ret)}  MRR={np.mean([r['ret_mrr'] for r in rs]):.3f}")
            print(f"  B learned : MRR={np.mean([r['b_learn'] for r in rs]):.3f}")
            print()


if __name__ == "__main__":
    main()
