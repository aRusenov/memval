"""
Standalone experiment: does a two-weight CA3 layer (symmetric attractors +
asymmetric STDP transitions + adaptation) produce an autonomous A->B->C sweep?

Tests
-----
  1. Auto-association: W_sym cleans a noisy cue back to the correct attractor.
  2. Sweep: cue the first state, run free-phase dynamics, and check the network
     visits the stored states in order (a compressed theta-sequence-like sweep).

Run:  .venv/bin/python standalone_two_weight_ca3.py
Saves an overlap-vs-time heatmap to results/theta_two_weight/sweep.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from memval.models.baselines.theta_two_weight_ca3 import ThetaTwoWeightCA3

OUT = "results/theta_two_weight"
os.makedirs(OUT, exist_ok=True)

N_STATES = 6
SEED = 0


def build(**kw):
    net = ThetaTwoWeightCA3(n_units=400, sparsity=0.05, seed=SEED, **kw)
    pats = net.make_patterns(N_STATES)
    net.learn_sequence(pats, n_presentations=5, stdp_lr=1.0)
    return net, pats


# ----------------------------------------------------------------------
# Test 1 — auto-association (attractor cleanup)
# ----------------------------------------------------------------------
def _test_autoassoc(net, pats, flip_frac=0.4):
    rng = np.random.default_rng(1)
    recovered = 0
    for i, xi in enumerate(pats):
        cue = xi.copy()
        # corrupt: turn off some active units, turn on some inactive ones
        on = np.where(xi > 0)[0]; off = np.where(xi == 0)[0]
        n_flip = int(flip_frac * len(on))
        cue[rng.choice(on, n_flip, replace=False)] = 0.0
        cue[rng.choice(off, n_flip, replace=False)] = 1.0
        s = net.settle_cue(cue, steps=40)
        # decode
        Pc = pats - net.a
        ov = (Pc @ (s - net.a)) / (np.sum(Pc * Pc, axis=1) + 1e-9)
        recovered += int(np.argmax(ov) == i)
    return recovered, len(pats)


# ----------------------------------------------------------------------
# Test 2 — the sweep
# ----------------------------------------------------------------------
def _test_sweep(net, pats, n_steps=30, settle_substeps=20):
    states, overlaps = net.recall(pats[0], n_steps=n_steps, settle_substeps=settle_substeps)
    decoded = net.decode_overlaps(overlaps)
    # the ordered set of distinct states visited (collapsing dwell repeats)
    visited = []
    for d in decoded:
        if not visited or visited[-1] != d:
            visited.append(d)
    # how much of the target order 0,1,2,...,K-1 is a prefix of `visited`?
    target = list(range(len(pats)))
    correct_prefix = 0
    for a, b in zip(visited, target):
        if a == b:
            correct_prefix += 1
        else:
            break
    return decoded, visited, overlaps, correct_prefix


def sweep_report(overlaps, decoded, visited, correct_prefix, tag):
    print(f"\n[{tag}] decoded-state timeline (collapsed): {visited}")
    print(f"[{tag}] ordered A->...-> prefix correct: {correct_prefix}/{N_STATES}  "
          f"peak overlaps/state: {[round(float(overlaps[:,i].max()),2) for i in range(N_STATES)]}")


def main():
    net, pats = build()

    rec, tot = _test_autoassoc(net, pats)
    print(f"[auto-assoc] recovered {rec}/{tot} states from 40%-corrupted cues")

    # working regime (found by instrumenting the dynamics; see git history):
    #   dwell needs support_A > push, so lambda < 1; adaptation must build past
    #   (support - push) to trigger the hand-off → moderate a_gain, dt_a.
    LAM, A_GAIN, DT_A = 0.5, 1.0, 0.1
    print(f"\n[sweep] cue = state A, free-phase dynamics "
          f"(lambda={LAM}, a_gain={A_GAIN}, dt_a={DT_A}):")
    net, pats = build(lam=LAM, a_gain=A_GAIN, dt_a=DT_A)
    decoded, visited, overlaps, cp = _test_sweep(net, pats, n_steps=30, settle_substeps=20)
    sweep_report(overlaps, decoded, visited, cp, "sweep")

    # overlap-vs-time staircase
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(overlaps.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("recall timestep"); ax.set_ylabel("stored state")
    ax.set_yticks(range(N_STATES)); ax.set_yticklabels([chr(65 + i) for i in range(N_STATES)])
    ax.set_title(f"Two-weight CA3 sweep — overlap(state, s) over time\n"
                 f"cue=A, visited={visited}  (lam={LAM}, a_gain={A_GAIN}, dt_a={DT_A})")
    fig.colorbar(im, label="overlap")
    fig.tight_layout()
    path = os.path.join(OUT, "sweep.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved {path}")

    ok = (rec >= tot - 1) and (cp >= N_STATES - 1)
    print(f"\nRESULT: auto-assoc {'OK' if rec>=tot-1 else 'WEAK'}; "
          f"sweep prefix {cp}/{N_STATES} -> "
          f"{'CLEAN SWEEP EMERGES' if cp >= N_STATES-1 else 'PARTIAL/NO SWEEP'}")
    return ok


if __name__ == "__main__":
    main()
