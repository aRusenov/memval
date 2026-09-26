#!/usr/bin/env python
"""
Continual-learning CHAIN experiment (isolated & fast).

Trains a model on a chain of N orthogonal sequences one after another
(seq0 -> seq1 -> ... ), and after each training phase measures cued-recall
retention of every sequence learned so far. This yields the standard continual-
learning retention matrix R, where

    R[j, i] = MRR on sequence i, measured after training through sequence j   (j >= i)

Rows are training stages, columns evaluated sequences, matching
`memval.metrics.retention`, which owns the matrix and its summary statistics so
that other benchmarks report the same numbers.

From R we report, per model, averaged over random seeds (bootstrap), both axes
of the stability-plasticity trade-off rather than the stability half alone:

  stability   final_acc   mean of the final row      (how much survives at the end)
              forgetting  mean drop from peak to final
  plasticity  learning    mean of the DIAGONAL       (acquisition under load)
              intransig.  R[0,0] - mean of the rest of the diagonal
  joint       ratio       chance-corrected ACC / LA  (of what got in, what stayed)
              SPI         harmonic mean of learning and ratio

The plasticity columns exist because every stability column is maximised by an
arm that stops learning after task 0: it has a perfect final row, zero
forgetting and zero backward transfer. The diagonal is that arm's tell, and SPI
refuses to reward either half alone. `--plane` writes the two coordinates as a
scatter so the failure *direction* is visible, which no scalar shows.

Design choices matching the symbolic benchmark:
  * SymbolicEncoder(embedding_dim=100, category_variance=0.2)
  * Each sequence is ONE category, so sequences are ~orthogonal across, similar
    within (well-orthogonalized, as requested).
  * Retention scored with the benchmark's own measure_recall_associative /
    mean_recall_rate, so numbers are directly comparable.

Only this experiment runs (not the full symbolic suite), so it is fast.

Usage:
    python bin/continual_chain_experiment.py                        # defaults
    python bin/continual_chain_experiment.py --models ewc_dg,ewc_original,original
    python bin/continual_chain_experiment.py --seeds 5 --epochs 300
"""
import argparse
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.benchmarks.symbolic_pipeline import measure_recall_associative, mean_recall_rate
from memval.benchmarks.continual_chain import ContinualChainBenchmark
from memval.metrics.retention import (
    retention_summary,
    plot_retention_matrix,
    plot_stability_plasticity_plane,
)
from memval.models.baselines import (
    OriginalEqPropSequenceNetwork,
    EWCOriginalEqPropSequenceNetwork,
    DGOriginalEqPropSequenceNetwork,
    DGXdGEqPropSequenceNetwork,
    EWCDGXdGEqPropSequenceNetwork,
    TemporalPCNetwork,
    MultilayerTemporalPCNetwork,
    AsymmetricHopfieldNetwork,
    ThetaPhaseSequenceNetwork,
)

# ----------------------------------------------------------------------
# The chain: 4 sequences x 5 elements, one distinct category per sequence
# (distinct categories => ~orthogonal across sequences in the encoder).
# ----------------------------------------------------------------------
SEQUENCES = {
    "fruit":   ["apple", "banana", "orange", "grape", "pear"],
    "animal":  ["cat", "dog", "cow", "horse", "sheep"],
    "vehicle": ["car", "truck", "bus", "train", "plane"],
    "tool":    ["hammer", "wrench", "saw", "drill", "pliers"],
}

EMBED_DIM = 100
CATEGORY_VARIANCE = 0.2
NOISE_SCALE = 0.05
EVAL_NOISE_BASE = 10_000   # fixed eval-noise seeds so retention comparisons are low-variance


# ----------------------------------------------------------------------
# Model registry: name -> (class, kwargs-factory(seed, epochs))
# ----------------------------------------------------------------------
def _common(seed, epochs):
    return dict(n_hidden=128, learning_rate=0.1, n_epochs=epochs, beta=0.5, seed=seed)


def _dg_common(seed, epochs):
    return dict(_common(seed, epochs), n_dg=1000, dg_target_sparsity=0.05,
                dg_inhibition=1.0, dg_seed=seed)


MODELS = {
    "original":     lambda s, e: (OriginalEqPropSequenceNetwork, _common(s, e)),
    "ewc_original": lambda s, e: (EWCOriginalEqPropSequenceNetwork, dict(_common(s, e), ewc_lambda=3e4)),
    "dg":           lambda s, e: (DGOriginalEqPropSequenceNetwork, _dg_common(s, e)),
    "dg_xdg":       lambda s, e: (DGXdGEqPropSequenceNetwork, dict(_dg_common(s, e), gate_sparsity=0.2, gate_seed=s)),
    "ewc_dg":       lambda s, e: (EWCDGXdGEqPropSequenceNetwork,
                                  dict(_dg_common(s, e), gate_sparsity=1.0, gate_seed=s, ewc_lambda=3e4)),
    # --- non-EP arms: temporal predictive coding (Tang, Barron & Bogacz 2023) ---
    # tpc1 is the AHN-with-whitening single-layer model, so `ahn` is its
    # controlled comparison (same architecture, no whitening).
    "tpc1":         lambda s, e: (TemporalPCNetwork,
                                  dict(learning_rate=0.01, n_epochs=e, seed=s)),
    "tpc2":         lambda s, e: (MultilayerTemporalPCNetwork,
                                  dict(n_hidden=128, learning_rate=0.05, n_epochs=e, seed=s)),
    "ahn":          lambda s, e: (AsymmetricHopfieldNetwork, dict(learning_rate=0.1)),
    # --- theta-phase encode/retrieve (Hasselmo, Bodelon & Wyble 2002) ---
    # At ltp_phase_offset=0 this takes the same step as `ahn`, so the pair is
    # controlled: same rule, error derived by phase separation instead of by
    # explicit subtraction.
    "theta":        lambda s, e: (ThetaPhaseSequenceNetwork,
                                  dict(learning_rate=0.1, n_epochs=e,
                                       modulation_depth=1.0)),
    # X=0 is the paper's fornix-lesion control: gates go constant, the
    # cycle integral vanishes, and prior associations persist unopposed.
    "theta_lesion": lambda s, e: (ThetaPhaseSequenceNetwork,
                                  dict(learning_rate=0.1, n_epochs=e,
                                       modulation_depth=0.0)),
}


def cross_sequence_cosine(enc, cats):
    """Mean pairwise cosine between elements of different sequences (orthogonality check)."""
    embs = {c: enc.encode(SEQUENCES[c]) for c in cats}
    vals = []
    for a in range(len(cats)):
        for b in range(a + 1, len(cats)):
            for u in embs[cats[a]]:
                for v in embs[cats[b]]:
                    vals.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))
    return float(np.mean(vals))


def run_chain(model_class, kw, seed, epochs, n_trials, rehearse=False):
    """Train the chain on one model and return the benchmark's full output.

    The protocol, the matrix and every statistic come from
    ``ContinualChainBenchmark`` / ``memval.metrics.retention``, so this script
    and the ``continual_chain`` section of the symbolic suite report the
    identical quantities on the identical definitions. What stays here is only
    the chain's *content* -- four categories rather than the suite's six, which
    is what makes this variant fast.
    """
    cats = list(SEQUENCES.keys())
    vocab = {w: c for c in cats for w in SEQUENCES[c]}
    enc = SymbolicEncoder(vocab, embedding_dim=EMBED_DIM,
                          category_variance=CATEGORY_VARIANCE, seed=seed)
    dec = SymbolicDecoder(enc)
    embs = {c: enc.encode(SEQUENCES[c]) for c in cats}

    # ONE model; weights persist across sequences -- this is what makes it continual.
    model = model_class(n_features=EMBED_DIM, **kw)

    def score_task(i):
        model.reset_context()                 # transient state only, NOT weights
        return mean_recall_rate(
            measure_recall_associative(model, SEQUENCES[cats[i]], enc, dec,
                                       n_trials=n_trials, noise_scale=NOISE_SCALE))

    return ContinualChainBenchmark(
        n_trials=n_trials,
        noise_scale=NOISE_SCALE,
        eval_noise_base=EVAL_NOISE_BASE,
        rehearse=rehearse,
    ).evaluate(
        model=model,
        datasets={"sequences": SEQUENCES, "embeddings": embs, "score_fn": score_task},
        epochs=epochs,
        chance_level=1.0 / len(vocab),
    )


#: Columns printed per model: (label, metric key, format).
REPORT_COLUMNS = (
    ("final_acc  ", "chain_avg_accuracy", "{:.3f}"),
    ("forgetting ", "chain_avg_forgetting", "{:.3f}"),
    ("learning   ", "chain_avg_learning", "{:.3f}"),
    ("intransig. ", "chain_intransigence", "{:+.3f}"),
    ("ret_ratio  ", "chain_retention_ratio", "{:.3f}"),
    ("SPI        ", "chain_stability_plasticity_index", "{:.3f}"),
)

SELECTIVITY_COLUMNS = (
    ("selectivity", "select_selectivity", "{:+.3f}"),
    ("  rehearsed gain", "select_rehearsed_gain", "{:+.3f}"),
    ("  other drift  ", "select_unrehearsed_drift", "{:+.3f}"),
)


def main():
    p = argparse.ArgumentParser(description="4-sequence continual-learning chain experiment.")
    p.add_argument("--models", type=str, default="ewc_dg,ewc_original",
                   help="Comma-separated model names. Options: " + ", ".join(MODELS))
    p.add_argument("--seeds", type=int, default=3, help="Number of random seeds to bootstrap over.")
    p.add_argument("--epochs", type=int, default=300, help="Training epochs per sequence.")
    p.add_argument("--n-trials", type=int, default=30, help="Recall trials per measurement.")
    p.add_argument("--rehearse", action="store_true",
                   help="Run the selective-retention phase: re-present the oldest "
                        "half of the chain, then re-score everything. Only "
                        "interpretable under capacity pressure -- read "
                        "select_under_pressure before quoting selectivity.")
    p.add_argument("--plane", type=str, default=None, metavar="PNG",
                   help="Write the stability-plasticity plane across the "
                        "requested models to this path.")
    p.add_argument("--matrix-dir", type=str, default=None, metavar="DIR",
                   help="Write each model's mean retention matrix heatmap here.")
    args = p.parse_args()

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    for nm in names:
        if nm not in MODELS:
            p.error(f"unknown model '{nm}'. Options: {', '.join(MODELS)}")

    cats = list(SEQUENCES.keys())
    seeds = list(range(args.seeds))

    # orthogonality check (seed 0 instance)
    enc0 = SymbolicEncoder({w: c for c in cats for w in SEQUENCES[c]},
                           embedding_dim=EMBED_DIM, category_variance=CATEGORY_VARIANCE, seed=0)
    print(f"Chain: {len(cats)} sequences x {len(SEQUENCES[cats[0]])} elements "
          f"({' -> '.join(cats)})")
    print(f"Mean cross-sequence cosine (orthogonality): {cross_sequence_cosine(enc0, cats):.3f}  (0 = orthogonal)")
    print(f"Seeds: {seeds}   epochs/seq: {args.epochs}   trials: {args.n_trials}\n")

    if args.matrix_dir:
        os.makedirs(args.matrix_dir, exist_ok=True)
    plane_points = []

    for nm in names:
        runs = []
        for s in seeds:
            cls, kw = MODELS[nm](s, args.epochs)
            runs.append(run_chain(cls, kw, s, args.epochs, args.n_trials,
                                  rehearse=args.rehearse))
        Rs = np.stack([np.array(r["series"]["retention_matrix"], dtype=float)
                       for r in runs])
        Rmean = np.nanmean(Rs, axis=0)

        print(f"===== {nm} =====")
        print("  Mean retention matrix R[j,i] = MRR on seq i after training thru seq j:")
        header = "        " + "".join(f"{c[:6]:>8}" for c in cats)
        print(header)
        for j, cj in enumerate(cats):
            row = "".join((f"{Rmean[j, i]:8.3f}" if not np.isnan(Rmean[j, i]) else f"{'-':>8}")
                          for i in range(len(cats)))
            print(f"  {cj[:6]:>6}{row}")

        def spread(key):
            vals = [r["metrics"][key] for r in runs]
            return float(np.mean(vals)), float(np.std(vals))

        for label, key, fmt in REPORT_COLUMNS:
            mu, sd = spread(key)
            print(f"  {label} = {fmt.format(mu)} +/- {fmt.format(sd).lstrip('+')}")

        # The forgetting gradient: what the scalar `forgetting` averages away.
        gaps = np.nanmean(np.stack([r["series"]["forgetting_by_gap"] for r in runs]), axis=0)
        print("  retained fraction by interposed tasks: "
              + "  ".join(f"g={g}:{v:.3f}" for g, v in enumerate(gaps, start=1)))

        if args.rehearse:
            pressure, _ = spread("select_pressure")
            under = all(r["metrics"]["select_under_pressure"] for r in runs)
            for label, key, fmt in SELECTIVITY_COLUMNS:
                mu, sd = spread(key)
                print(f"  {label} = {fmt.format(mu)} +/- {fmt.format(sd).lstrip('+')}")
            print(f"  (capacity pressure = {pressure:.3f}, under_pressure = {under}"
                  + ("" if under else "  -- selectivity NOT interpretable: nothing "
                                      "had to be discarded") + ")")
        print()

        # Plane coordinates from the mean matrix, so the point matches the
        # matrix printed above rather than averaging two ratios.
        s_mean = retention_summary(Rmean, chance_level=runs[0]["metrics"]["chain_chance_level"])
        plane_points.append({"label": nm, "summary": s_mean})

        if args.matrix_dir:
            out = os.path.join(args.matrix_dir, f"retention_{nm}.png")
            plot_retention_matrix(Rmean, out, title=f"Retention — {nm}",
                                  value_label="cued-recall MRR", task_labels=cats)
            print(f"  wrote {out}")

    if args.plane:
        plot_stability_plasticity_plane(
            plane_points, args.plane,
            title=f"Stability–plasticity plane ({len(cats)}-task chain)")
        print(f"Wrote stability–plasticity plane to {args.plane}")


if __name__ == "__main__":
    main()
