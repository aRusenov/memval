"""
Isolate the DG's NOISE-FREE recall cost: cosine margin (primary) + proper MRR.

The benchmark's MRR gets its gradation from injected cue noise (multi-trial Monte
Carlo). Set noise=0 and that collapses to a coarse deterministic top-1. To grade
recall WITHOUT noise we instead use a graded scoring rule on a single
deterministic pass:
  * cosine margin = cos(pred, true) - max_{distractor} cos(pred, distractor)
      sign = correct/wrong, magnitude = confidence  (PRIMARY)
  * reciprocal rank = 1 / rank of true successor among the vocab              (2nd)

Variability comes from SEEDS (model init / DG projection), not cue noise. Trains
list A only (no interference) on the real SymbolicEncoder; candidates = full
14-word vocab (as the benchmark decoder ranks against).

Question: is the DG's noise-free top-1 deficit (0.889 vs 1.0) GRACEFUL (all
transitions correct with smaller margin) or CATASTROPHIC (a few negative margins
= associations that did not store)?
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
from memval.diagnostics import (transitions_of, retrieval_margin, retrieval_mrr,
                                one_step_accuracy)

LIST_A = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']
LIST_B = ['cat', 'dog', 'cow', 'horse', 'sheep', 'pig', 'lion']
VOCAB = {w: 'fruit' for w in LIST_A} | {w: 'animal' for w in LIST_B}
CANDIDATES = LIST_A + LIST_B
DIM = 100
N_HIDDEN = 256
N_DG = 256
DG_SPARSITY = 0.05
EPOCHS = 150
SEEDS = [0, 1, 2, 3, 4]


def build(model_type, seed):
    common = dict(n_features=DIM, n_hidden=N_HIDDEN, learning_rate=0.1, beta=0.5,
                  n_settle_steps=40, n_epochs=EPOCHS, seed=seed)
    if model_type == "original":
        return OriginalEqPropSequenceNetwork(**common)
    return DGOriginalEqPropSequenceNetwork(
        n_dg=N_DG, dg_target_sparsity=DG_SPARSITY, dg_seed=seed, **common)


class EncDec:
    """Adapter giving the diagnostics both .encode (encoder) and .decode (decoder)."""
    def __init__(self):
        self.enc = SymbolicEncoder(VOCAB, embedding_dim=DIM, category_variance=0.2, seed=42)
        self.dec = SymbolicDecoder(self.enc)

    def encode(self, seq):
        return self.enc.encode(seq)

    def decode(self, vec):
        return self.dec.decode(vec, top_k=1)[0]


def run_task(args):
    model_type, seed = args
    enc = EncDec()
    trans = transitions_of(LIST_A)
    model = build(model_type, seed)
    model.fit_sequence(enc.encode(LIST_A), epochs=EPOCHS)
    return {
        "model": model_type,
        "margins": retrieval_margin(model, enc, trans, CANDIDATES),
        "mrr": retrieval_mrr(model, enc, trans, CANDIDATES),
        "top1": one_step_accuracy(model, enc, trans),
    }


def main():
    tasks = [(m, s) for m in ("original", "dg") for s in SEEDS]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = list(ex.map(run_task, tasks))

    print(f"NOISE-FREE base recall (train list A only), real SymbolicEncoder; "
          f"n_hidden={N_HIDDEN}, n_dg={N_DG}, epochs={EPOCHS}, seeds={SEEDS}")
    print(f"candidates = {len(CANDIDATES)}-word vocab; margins/MRR per transition, pooled\n")

    for mt in ("original", "dg"):
        rs = [r for r in rows if r["model"] == mt]
        margins = np.array([m for r in rs for m in r["margins"]])
        mrr = np.array([m for r in rs for m in r["mrr"]])
        top1 = np.mean([r["top1"] for r in rs])
        frac_neg = float(np.mean(margins < 0))
        print(f"=== {mt.upper()} ===")
        print(f"  cosine margin (PRIMARY): mean={margins.mean():+.3f}  "
              f"min={margins.min():+.3f}  median={np.median(margins):+.3f}  "
              f"frac_negative={frac_neg:.2f}")
        print(f"  reciprocal rank (MRR)  : mean={mrr.mean():.3f}")
        print(f"  top-1 accuracy         : {top1:.3f}")
        # margin percentiles to see the distribution shape
        pct = np.percentile(margins, [10, 25, 50, 75, 90])
        print(f"  margin pctiles [10/25/50/75/90]: "
              + " ".join(f"{p:+.3f}" for p in pct))
        print()


if __name__ == "__main__":
    main()
