"""
Why does adding the DG expansion sharply drop recall on the real encoder?

Hypothesis: it's the DG's noise-AMPLIFICATION, exposed by the benchmark's noisy
cue (measure_recall_associative adds Gaussian noise; the toy's one-step metric
did not). The DG's hard top-k/threshold is non-Lipschitz: a small cue
perturbation flips which units win -> the noisy cue separates to a DIFFERENT
sparse code than the trained one -> the association misfires. The plain net (no
threshold) degrades smoothly.

Decisive test: train each model on list A only (no interference), then sweep the
recall cue-noise. If Original and DG are CLOSE at noise=0 but DG collapses much
faster as noise rises, the degradation is noise-brittleness, not learnability.
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

LIST_A = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']
LIST_B = ['cat', 'dog', 'cow', 'horse', 'sheep', 'pig', 'lion']
VOCAB = {w: 'fruit' for w in LIST_A} | {w: 'animal' for w in LIST_B}
DIM = 100
N_HIDDEN = 256
N_DG = 256
DG_SPARSITY = 0.05
EPOCHS = 150
N_TRIALS = 40
NOISES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
SEEDS = [0, 1, 2]


def build(model_type, seed):
    common = dict(n_features=DIM, n_hidden=N_HIDDEN, learning_rate=0.1, beta=0.5,
                  n_settle_steps=40, n_epochs=EPOCHS, seed=seed)
    if model_type == "original":
        return OriginalEqPropSequenceNetwork(**common)
    return DGOriginalEqPropSequenceNetwork(
        n_dg=N_DG, dg_target_sparsity=DG_SPARSITY, dg_seed=seed, **common)


def dg_code_shift(model, enc, noise, seed):
    """How much the DG code moves under cue noise (Jaccard distance of active sets,
    clean vs noisy), averaged over list-A cues. 0 for the plain net (no _separate)."""
    if not hasattr(model, "_separate"):
        return float("nan")
    rng = np.random.default_rng(1000 + seed)
    dists = []
    for w in LIST_A:
        clean = enc.encode([w])[0]
        c = set(np.nonzero(model._separate(clean) > 1e-6)[0])
        for _ in range(20):
            n = clean + rng.normal(0, noise, DIM)
            d = set(np.nonzero(model._separate(n) > 1e-6)[0])
            u = len(c | d)
            dists.append(1.0 - (len(c & d) / u if u else 1.0))
    return float(np.mean(dists))


def run_task(args):
    model_type, seed = args
    enc = SymbolicEncoder(VOCAB, embedding_dim=DIM, category_variance=0.2, seed=42)
    dec = SymbolicDecoder(enc)
    model = build(model_type, seed)
    model.fit_sequence(enc.encode(LIST_A), epochs=EPOCHS)

    mrr = {}
    code_shift = {}
    for ns in NOISES:
        model.reset_context()
        mrr[ns] = mean_recall_rate(
            measure_recall_associative(model, LIST_A, enc, dec, n_trials=N_TRIALS, noise_scale=ns))
        code_shift[ns] = dg_code_shift(model, enc, ns, seed)
    return {"model": model_type, "mrr": mrr, "code_shift": code_shift}


def main():
    tasks = [(m, s) for m in ("original", "dg") for s in SEEDS]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = list(ex.map(run_task, tasks))

    print(f"REAL SymbolicEncoder; train list A only; recall cue-noise sweep. "
          f"n_hidden={N_HIDDEN}, n_dg={N_DG}, epochs={EPOCHS}, seeds={SEEDS}\n")
    header = "  noise  " + "".join(f"{ns:>7}" for ns in NOISES)
    for mt in ("original", "dg"):
        rs = [r for r in rows if r["model"] == mt]
        print(f"=== {mt.upper()} ===")
        print(header)
        mrrs = [np.mean([r["mrr"][ns] for r in rs]) for ns in NOISES]
        print("  MRR    " + "".join(f"{v:>7.3f}" for v in mrrs))
        if mt == "dg":
            shifts = [np.mean([r["code_shift"][ns] for r in rs]) for ns in NOISES]
            print("  code Δ " + "".join(f"{v:>7.3f}" for v in shifts)
                  + "   (DG-code Jaccard distance clean vs noisy)")
        print()


if __name__ == "__main__":
    main()
