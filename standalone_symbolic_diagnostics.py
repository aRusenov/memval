"""
Run the memval.diagnostics tiers on the REAL symbolic multiple-sequences task.

Uses the benchmark's own setup — fruits (list A) then animals (list B), the
100-dim zero-centred SymbolicEncoder (seed 42, category_variance 0.2), and the
benchmark's MRR (measure_recall_associative -> mean_recall_rate) — so behaviour
is directly comparable to the recorded delta_mrr (original -0.561, dg -0.806,
dg_xdg -0.356 at its 2048/0.02 winner). Here we use a MODERATE config for speed,
so absolute delta_mrr will be milder than the winner; the point is the MECHANISM
on real (zero-centred) embeddings vs the non-negative toy:
  * does DG's input Jaccard now fall to ~floor (toy was stuck ~0.23)?
  * does DG still fail / XdG still help behaviourally?
  * does the b_o Fisher share shrink (toy 75% was largely a non-negative-code artifact)?
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.benchmarks.symbolic_pipeline import measure_recall_associative, mean_recall_rate
from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork
from memval.models.baselines.dg_original_eqprop import DGOriginalEqPropSequenceNetwork
from memval.models.baselines.dg_xdg_eqprop import DGXdGEqPropSequenceNetwork
from memval.diagnostics import (representation_overlap, update_interference,
                                fisher_diagonal, snapshot_parameters, fisher_attribution)

LIST_A = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']
LIST_B = ['cat', 'dog', 'cow', 'horse', 'sheep', 'pig', 'lion']
VOCAB = {w: 'fruit' for w in LIST_A} | {w: 'animal' for w in LIST_B}
DIM = 100
N_HIDDEN = 256
N_DG = 256
DG_SPARSITY = 0.05
GATE_SPARSITY = 0.1
EPOCHS = 150
N_TRIALS = 20
SEEDS = [0, 1]


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


def mrr(model, enc, dec):
    model.reset_context()
    return mean_recall_rate(
        measure_recall_associative(model, LIST_A, enc, dec, n_trials=N_TRIALS, noise_scale=0.05))


def run_task(args):
    model_type, seed = args
    enc = SymbolicEncoder(VOCAB, embedding_dim=DIM, category_variance=0.2, seed=42)
    dec = SymbolicDecoder(enc)
    emb_A, emb_B = enc.encode(LIST_A), enc.encode(LIST_B)
    tv_A = [(emb_A[i], emb_A[i + 1]) for i in range(len(emb_A) - 1)]
    tv_B = [(emb_B[i], emb_B[i + 1]) for i in range(len(emb_B) - 1)]

    model = build(model_type, seed)

    # init-time geometry + interference
    geom = representation_overlap(model, list(emb_A), list(emb_B))
    interf = update_interference(model, tv_A, tv_B)

    # train A -> importance; train B -> forgetting
    model.fit_sequence(emb_A, epochs=EPOCHS)
    mrr_before = mrr(model, enc, dec)
    theta_A = snapshot_parameters(model)
    F_A = fisher_diagonal(model, tv_A)

    model.fit_sequence(emb_B, epochs=EPOCHS)
    mrr_after = mrr(model, enc, dec)
    attr = fisher_attribution(F_A, theta_A, snapshot_parameters(model))

    return {"model": model_type, "seed": seed,
            "mrr_before": mrr_before, "mrr_after": mrr_after,
            "delta_mrr": mrr_after - mrr_before, "geom": geom,
            "interf": interf, "attr": attr}


def main():
    tasks = [(m, s) for m in ("original", "dg", "xdg") for s in SEEDS]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = list(ex.map(run_task, tasks))

    print(f"REAL SymbolicEncoder (dim={DIM}, cat_var=0.2, seed42); fruits->animals; "
          f"n_hidden={N_HIDDEN}, n_dg={N_DG}, dg_sp={DG_SPARSITY}, gate={GATE_SPARSITY}, "
          f"epochs={EPOCHS}, seeds={SEEDS}  (MODERATE config, not the 2048/0.02 winner)\n")

    for mt in ("original", "dg", "xdg"):
        rs = [r for r in rows if r["model"] == mt]
        mb = np.mean([r["mrr_before"] for r in rs])
        ma = np.mean([r["mrr_after"] for r in rs])
        dm = np.mean([r["delta_mrr"] for r in rs])
        print(f"=== {mt.upper()} ===")
        print(f"  MRR: before={mb:.3f}  after={ma:.3f}  delta_mrr={dm:+.3f}")

        # geometry: input + hidden Jaccard (A vs B) vs floor
        g0 = rs[0]["geom"]
        for layer in ("input", "hidden"):
            if layer in g0:
                jab = np.mean([r["geom"][layer]["jaccard_ab"] for r in rs])
                flr = np.mean([r["geom"][layer]["floor"] for r in rs])
                den = np.mean([r["geom"][layer]["density_a"] for r in rs])
                print(f"  {layer:>6} Jaccard(A,B)={jab:.3f}  (density {den:.2f}, floor {flr:.3f})")

        # readout collision + forgetting attribution
        who = np.mean([r["interf"]["W_ho"]["ab"] for r in rs])
        print(f"  W_ho update-cos A<->B = {who:.3f}")
        cs = {g: np.mean([r["attr"]["contrib_share"][g] for r in rs])
              for g in rs[0]["attr"]["contrib_share"]}
        print("  forgetting attribution: " +
              "  ".join(f"{g}={100*s:.1f}%" for g, s in cs.items()))
        print()


if __name__ == "__main__":
    main()
