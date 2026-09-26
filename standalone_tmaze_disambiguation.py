"""
Standalone T-Maze Odour Disambiguation benchmark.

Extracted from ``memval/benchmarks/spatial_pipeline.py`` (section 3, "T-Maze
Odour Disambiguation") into a self-contained, runnable script so the
disambiguation task can be studied in isolation.

Two T-maze routes share an identical central stem (same place-cell input) and
diverge into left/right arms. On the stem each route carries a distinct,
orthogonal odour (LEC) one-hot. A correct model must use the odour to decide
which arm to replay.

The place (MEC) block has ~14 active cells while the odour (LEC) block is a
single unit, so the two modalities enter the model at very different effective
magnitudes. This script rescales both blocks to a MATCHED per-timestep norm so
neither modality dominates by raw scale, and the model is left to decide how to
weight them. ``odour_scale`` further tunes the odour magnitude relative to the
(unit-normed) place block: 1.0 == exactly matched.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from memval.generators.tmaze_disambiguation import TMazeDisambiguationGenerator
from memval.benchmarks.spatial_disambiguation import SpatialDisambiguationBenchmark
from memval.models.baselines.asymmetric_hopfield import AsymmetricHopfieldNetwork


def balance_route_pair(route_pair, odour_scale: float = 1.0):
    """Rescale the place (MEC) and odour (LEC) blocks to matched magnitude.

    For every timestep, the place block is normalised to unit L2 norm and the
    odour block is normalised to ``odour_scale``. Zero blocks (e.g. odour off
    the stem) are left untouched. The place-cell decoder is a scale-invariant
    centre-of-mass, so per-timestep place normalisation does not distort
    decoded coordinates.

    Args:
        route_pair: dict from TMazeDisambiguationGenerator.generate().
        odour_scale: target L2 norm of the odour block where it is active.
                     1.0 == exactly matched to the (unit-normed) place block.

    Returns:
        A shallow copy of ``route_pair`` with rescaled ``input_A``/``input_B``.
    """
    rp = dict(route_pair)
    n_pc = rp["encoder"].n_cells

    def rescale(X):
        X = X.copy()
        pc, od = X[:, :n_pc], X[:, n_pc:]
        pc_norm = np.linalg.norm(pc, axis=1, keepdims=True)
        od_norm = np.linalg.norm(od, axis=1, keepdims=True)
        pc_norm[pc_norm == 0] = 1.0
        od_norm[od_norm == 0] = 1.0
        X[:, :n_pc] = pc / pc_norm
        X[:, n_pc:] = od / od_norm * odour_scale
        return X

    rp["input_A"] = rescale(rp["input_A"])
    rp["input_B"] = rescale(rp["input_B"])
    return rp


def evaluate_disambiguation(route_pair, fit_epochs=100, n_trials=20, learning_rate=0.1):
    """Train a fresh AsymmetricHopfield on A then B and evaluate both conditions."""
    n_features = route_pair["input_A"].shape[-1]
    model = AsymmetricHopfieldNetwork(n_features=n_features, learning_rate=learning_rate)
    model.reset_context()
    model.fit_sequence(route_pair["input_A"], epochs=fit_epochs)
    model.fit_sequence(route_pair["input_B"], epochs=fit_epochs)

    if not np.all(np.isfinite(model.W)):
        return None  # LMS diverged (input magnitude too large for this lr)

    bench = SpatialDisambiguationBenchmark()
    full = bench.evaluate(model=model, route_pair=route_pair, condition="full",
                          n_trials=n_trials, fit_epochs=fit_epochs)
    mec = bench.evaluate(model=model, route_pair=route_pair, condition="mec_only",
                         n_trials=n_trials, fit_epochs=fit_epochs)
    return {"full": full, "mec_only": mec}


def run(sequence_length=30, stem_fraction=0.50, n_place_cells=400, n_odour_dims=20,
        odour_scale=1.0, fit_epochs=100, n_trials=20, seed=42):
    print("=" * 66)
    print(" T-Maze Odour Disambiguation  (matched-magnitude modalities)")
    print("=" * 66)

    gen = TMazeDisambiguationGenerator(seed=seed)
    raw = gen.generate(sequence_length=sequence_length, stem_fraction=stem_fraction,
                       n_place_cells=n_place_cells, n_odour_dims=n_odour_dims, seed=seed,
                       odour_on_arms=True)
    n_pc = raw["encoder"].n_cells
    se = raw["shared_end"]

    # Diagnostic: raw per-modality magnitude at the divergent (last-stem) step
    pre = raw["input_A"][se - 1]
    print(f"seq_len={sequence_length}  stem_end={se}  n_place={n_pc}  n_odour={n_odour_dims}")
    print(f"raw magnitudes  ||place||={np.linalg.norm(pre[:n_pc]):.3f}  "
          f"||odour||={np.linalg.norm(pre[n_pc:]):.3f}  "
          f"(place spread over {(pre[:n_pc] > 1e-3).sum()} cells)\n")

    # --- Sweep odour_scale to expose the modality-weight dependence ---
    print(f"{'condition':>16} | {'full acc':>9} | {'mec acc':>9} | {'route A':>7} | {'route B':>7}")
    print("-" * 66)

    def report(tag, res):
        if res is None:
            print(f"{tag:>16} | {'DIVERGED (LMS NaN — lr too high for this scale)':>0}")
            return
        f, m = res["full"], res["mec_only"]
        print(f"{tag:>16} | {f['branch_accuracy']:>9.3f} | {m['branch_accuracy']:>9.3f} | "
              f"{f['route_A_branch_accuracy']:>7.2f} | {f['route_B_branch_accuracy']:>7.2f}")

    # Baseline: original unbalanced encoding
    report("raw/unbalanced", evaluate_disambiguation(raw, fit_epochs, n_trials))

    # Matched magnitude at the requested odour_scale (default 1.0 == equal norm)
    balanced = balance_route_pair(raw, odour_scale=odour_scale)
    report(f"matched x{odour_scale:g}", evaluate_disambiguation(balanced, fit_epochs, n_trials))

    # Small sweep so the dependence on relative odour weight is visible
    for os_ in (2.0, 4.0):
        rp = balance_route_pair(raw, odour_scale=os_)
        report(f"matched x{os_:g}", evaluate_disambiguation(rp, fit_epochs, n_trials))
    print("-" * 66)
    print("full > mec  ⇒  the odour is actually being used to route.\n")

    # --- Plot the recalled arms at the requested odour_scale ---
    _plot(balanced, odour_scale, fit_epochs)


def _plot(route_pair, odour_scale, fit_epochs):
    encoder = route_pair["encoder"]
    n_pc = encoder.n_cells
    n_od = route_pair["input_A"].shape[-1] - n_pc
    se = route_pair["shared_end"]
    A, B = route_pair["input_A"], route_pair["input_B"]
    traj_A, traj_B = encoder.decode(A[:, :n_pc]), encoder.decode(B[:, :n_pc])

    model = AsymmetricHopfieldNetwork(n_features=A.shape[-1])
    model.reset_context()
    model.fit_sequence(A, epochs=fit_epochs)
    model.fit_sequence(B, epochs=fit_epochs)

    def recall(seq):
        model.reset_context()
        for t in range(se - 1):
            model.predict_next(seq[t])
        curr = seq[se - 1].copy()
        out = []
        for t in range(len(seq) - se):
            pred = model.predict_next(curr)
            out.append(pred[:n_pc])
            # place is auto-regressive; the odour is exogenous and clamped from
            # the input sequence, so it persists through the arm when present.
            nxt = se + t
            od = seq[nxt, n_pc:] if nxt < len(seq) else np.zeros(n_od)
            curr = np.concatenate([pred[:n_pc], od])
        return encoder.decode(np.array(out))

    rec_A, rec_B = recall(A), recall(B)

    plt.figure(figsize=(6, 6))
    plt.plot(traj_A[:, 0], traj_A[:, 1], '--', color='blue', alpha=0.3, label='Target Left (A)')
    plt.plot(traj_B[:, 0], traj_B[:, 1], '--', color='red', alpha=0.3, label='Target Right (B)')
    plt.plot(traj_A[:se, 0], traj_A[:se, 1], '-', color='green', linewidth=3, label='Shared stem (odour)')
    plt.plot(rec_A[:, 0], rec_A[:, 1], '-b^', label='Recall A (left odour)')
    plt.plot(rec_B[:, 0], rec_B[:, 1], '-rx', label='Recall B (right odour)')
    plt.title(f'T-Maze Odour Disambiguation (odour_scale={odour_scale:g})')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.tight_layout()
    os.makedirs("./results", exist_ok=True)
    out = "./results/tmaze_disambiguation_balanced.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Standalone T-Maze odour disambiguation with matched-magnitude modalities.")
    p.add_argument("--odour-scale", type=float, default=1.0,
                   help="Odour block L2 norm relative to unit-normed place block (default 1.0 = matched).")
    p.add_argument("--fit-epochs", type=int, default=100)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=30)
    args = p.parse_args()
    run(sequence_length=args.seq_len, odour_scale=args.odour_scale,
        fit_epochs=args.fit_epochs, n_trials=args.n_trials)
