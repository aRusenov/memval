"""
Standalone T-Maze Odour Disambiguation benchmark -- Original Equilibrium
Propagation network.

Sibling of ``standalone_tmaze_disambiguation.py`` (which uses the flat linear
AsymmetricHopfieldNetwork). Same task: two T-maze routes share an identical
central stem and diverge into left/right arms, distinguished only by a
persistent orthogonal odour (LEC) cue (``odour_on_arms=True``).

Why a separate script: OriginalEqPropSequenceNetwork is an energy-based net
with a *hidden layer* whose settling dynamics clip the output to [0, 1] (not
tanh [-1, 1]). Two consequences:

  1. The hidden layer can form conjunctive place-x-odour codes, so the shared
     stem is no longer a single linear resource -- EP is expected to
     disambiguate at raw modality scale, without the magnitude over-weighting
     the linear associator needed.
  2. SpatialDisambiguationBenchmark rescales predictions with ``0.5*p + 0.5``
     (correct only for tanh models). That drags EP's [0, 1] output toward the
     grid centre and poisons the autoregressive feedback, so we use a clean,
     rescale-free rollout here.

The odour is treated as exogenous: place cells are auto-regressive (fed back
from the model), the odour is clamped from the input sequence at each step.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from memval.generators.tmaze_disambiguation import TMazeDisambiguationGenerator
from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork

# Reuse the matched-magnitude modality scaling from the linear-associator script.
from standalone_tmaze_disambiguation import balance_route_pair


def _branch_accuracy(model, route_pair, condition):
    """Clean (rescale-free) autoregressive rollout + branch classification.

    Ingest the shared stem, then roll out the arm one step at a time: the place
    block is fed back from the model's own [0, 1] output, the odour block is
    clamped exogenously from the (optionally ablated) input sequence. Each
    decoded suffix position is classified to whichever route's ground-truth
    coordinate it is nearest. Returns (mean_acc, acc_A, acc_B).
    """
    enc = route_pair["encoder"]
    npc = enc.n_cells
    A, B = route_pair["input_A"], route_pair["input_B"]
    nod = A.shape[-1] - npc
    se, L = route_pair["shared_end"], len(A)
    coords_A, coords_B = enc.decode(A[:, :npc]), enc.decode(B[:, :npc])

    def rollout(test_seq, target):
        model.reset_context()
        for t in range(se - 1):
            model.predict_next(test_seq[t])
        curr = test_seq[se - 1].copy()
        preds = []
        for t in range(L - se):
            pred = model.predict_next(curr)          # EP output already in [0, 1]
            preds.append(pred[:npc])
            nxt = se + t
            od = test_seq[nxt, npc:] if nxt < L else np.zeros(nod)
            curr = np.concatenate([pred[:npc], od])  # place auto-regressive, odour clamped
        decoded = enc.decode(np.array(preds))
        correct = 0
        for i in range(L - se):
            g = se + i
            dA = np.linalg.norm(decoded[i] - coords_A[g])
            dB = np.linalg.norm(decoded[i] - coords_B[g])
            correct += (dA < dB) if target == "A" else (dB < dA)
        return correct / (L - se)

    tA, tB = A.copy(), B.copy()
    if condition == "mec_only":          # ablate the odour -> should collapse to chance
        tA[:, npc:] = 0.0
        tB[:, npc:] = 0.0
    accA, accB = rollout(tA, "A"), rollout(tB, "B")
    return 0.5 * (accA + accB), accA, accB


def _fresh_model(n_features, n_hidden, n_epochs, seed):
    return OriginalEqPropSequenceNetwork(
        n_features=n_features, n_hidden=n_hidden, n_epochs=n_epochs,
        learning_rate=0.1, beta=0.5, seed=seed,
    )


def evaluate(route_pair, n_hidden, n_epochs, seed):
    model = _fresh_model(route_pair["input_A"].shape[-1], n_hidden, n_epochs, seed)
    model.reset_context()
    model.fit_sequence(route_pair["input_A"])
    model.fit_sequence(route_pair["input_B"])
    return {
        "full": _branch_accuracy(model, route_pair, "full"),
        "mec_only": _branch_accuracy(model, route_pair, "mec_only"),
    }


def run(sequence_length=15, stem_fraction=0.50, n_place_cells=100, n_odour_dims=20,
        n_hidden=256, n_epochs=40, seed=42):
    print("=" * 66)
    print(" T-Maze Odour Disambiguation  --  Original Equilibrium Propagation")
    print("=" * 66)

    gen = TMazeDisambiguationGenerator(seed=seed)
    raw = gen.generate(sequence_length=sequence_length, stem_fraction=stem_fraction,
                       n_place_cells=n_place_cells, n_odour_dims=n_odour_dims, seed=seed,
                       sigma_scale=1.0, odour_on_arms=True)
    npc = raw["encoder"].n_cells
    se = raw["shared_end"]
    pre = raw["input_A"][se - 1]
    print(f"seq_len={sequence_length}  stem_end={se}  n_place={npc}  n_odour={n_odour_dims}  "
          f"n_hidden={n_hidden}  n_epochs={n_epochs}")
    print(f"raw magnitudes  ||place||={np.linalg.norm(pre[:npc]):.3f}  "
          f"||odour||={np.linalg.norm(pre[npc:]):.3f}\n")

    print(f"{'encoding':>16} | {'full acc':>9} | {'mec acc':>9} | {'route A':>7} | {'route B':>7}")
    print("-" * 66)

    def report(tag, res):
        f, m = res["full"], res["mec_only"]
        print(f"{tag:>16} | {f[0]:>9.3f} | {m[0]:>9.3f} | {f[1]:>7.2f} | {f[2]:>7.2f}")

    # EP has a nonlinearity, so it should not need magnitude over-weighting.
    # Show raw and matched (unit-norm) encodings; both are kept <= 1 to respect
    # EP's [0, 1] output range.
    report("raw", evaluate(raw, n_hidden, n_epochs, seed))
    report("matched x1", evaluate(balance_route_pair(raw, odour_scale=1.0), n_hidden, n_epochs, seed))
    print("-" * 66)
    print("full >> mec  ⇒  the hidden layer routes on the odour (conjunctive code).\n")

    _plot(raw, n_hidden, n_epochs, seed)


def _plot(route_pair, n_hidden, n_epochs, seed):
    enc = route_pair["encoder"]
    npc = enc.n_cells
    A, B = route_pair["input_A"], route_pair["input_B"]
    nod = A.shape[-1] - npc
    se, L = route_pair["shared_end"], len(A)
    traj_A, traj_B = enc.decode(A[:, :npc]), enc.decode(B[:, :npc])

    model = _fresh_model(A.shape[-1], n_hidden, n_epochs, seed)
    model.reset_context()
    model.fit_sequence(A)
    model.fit_sequence(B)

    def recall(seq):
        model.reset_context()
        for t in range(se - 1):
            model.predict_next(seq[t])
        curr = seq[se - 1].copy()
        out = []
        for t in range(L - se):
            pred = model.predict_next(curr)
            out.append(pred[:npc])
            nxt = se + t
            od = seq[nxt, npc:] if nxt < L else np.zeros(nod)
            curr = np.concatenate([pred[:npc], od])
        return enc.decode(np.array(out))

    rec_A, rec_B = recall(A), recall(B)

    plt.figure(figsize=(6, 6))
    plt.plot(traj_A[:, 0], traj_A[:, 1], '--', color='blue', alpha=0.3, label='Target Left (A)')
    plt.plot(traj_B[:, 0], traj_B[:, 1], '--', color='red', alpha=0.3, label='Target Right (B)')
    plt.plot(traj_A[:se, 0], traj_A[:se, 1], '-', color='green', linewidth=3, label='Shared stem (odour)')
    plt.plot(rec_A[:, 0], rec_A[:, 1], '-b^', label='Recall A (left odour)')
    plt.plot(rec_B[:, 0], rec_B[:, 1], '-rx', label='Recall B (right odour)')
    plt.title('T-Maze Odour Disambiguation -- Original EqProp')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.tight_layout()
    os.makedirs("./results", exist_ok=True)
    out = "./results/tmaze_disambiguation_ep.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Standalone T-Maze odour disambiguation with Original EqProp.")
    p.add_argument("--n-hidden", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=40)
    p.add_argument("--n-place-cells", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=15)
    args = p.parse_args()
    run(sequence_length=args.seq_len, n_place_cells=args.n_place_cells,
        n_hidden=args.n_hidden, n_epochs=args.n_epochs)
