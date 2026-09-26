"""Splitter index demo (docs/disambiguation_design.md sec 5.4).

What internal-state divergence shows that branch accuracy cannot.

    splitter_index(t) = 1 - cos(h_A(t), h_B(t))   for t in [shared_start, shared_end)

h_X(t) is the arm's carried state after observing route X's prefix up to t.
Right of the odour zone the two routes' INPUTS are identical, so any separation
measured there was carried, not sensed.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memval.generators.bifurcating_route import BifurcatingRouteGenerator
from memval.models.baselines import (MultilayerTemporalPCNetwork,
                                     AsymmetricHopfieldNetwork)

N_PC, N_OD = 120, 20
TOTAL, SHARED_FRAC = 16, 0.5
EPOCHS_TPC = 30


def make_route(zone_fraction):
    gen = BifurcatingRouteGenerator(seed=42)
    return gen.generate(
        total_length=TOTAL, shared_fraction=SHARED_FRAC, shared_position=0.0,
        zone_fraction=zone_fraction, zone_offset=0.0,
        n_place_cells=N_PC, n_encounter_dims=N_OD,
        balance_modalities=True, odour_scale=2.0,
        odour_on_suffix=False, seed=42,
    )


def tpc_state_profile(route):
    """tPC carries a latent z across steps and declares StatePrimeable, so its
    prefix can be delivered through `observe` (state moves, weights do not)."""
    n_feat = route["input_A"].shape[1]
    m = MultilayerTemporalPCNetwork(n_features=n_feat, n_hidden=128,
                                    n_epochs=EPOCHS_TPC, inf_iters=50, seed=42)
    m.reset_context()
    for _ in range(EPOCHS_TPC):
        m.fit_sequence(route["input_A"])
        m.fit_sequence(route["input_B"])

    def traj(seq):
        m.reset_context()
        out = []
        for t in range(len(seq)):
            m.observe(seq[t])
            out.append(np.asarray(m._prev_z, dtype=float).copy())
        return np.array(out)

    return traj(route["input_A"]), traj(route["input_B"])


def hopfield_state_profile(route):
    """A memoryless associator has no h(t) that depends on history: its only
    per-step quantity is its own prediction, a pure function of the current
    input. Shown to make explicit what the metric reads when nothing is carried."""
    n_feat = route["input_A"].shape[1]
    m = AsymmetricHopfieldNetwork(n_features=n_feat, n_epochs=1,
                                  activation="relu", seed=42)
    m.reset_context()
    m.fit_sequence(route["input_A"])
    m.fit_sequence(route["input_B"])

    def traj(seq):
        m.reset_context()
        out = []
        for t in range(len(seq)):
            out.append(np.asarray(m.predict_next(seq[t]), dtype=float).copy())
        return np.array(out)

    return traj(route["input_A"]), traj(route["input_B"])


def splitter_index(hA, hB):
    num = (hA * hB).sum(axis=1)
    den = np.linalg.norm(hA, axis=1) * np.linalg.norm(hB, axis=1)
    cos = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0)
    return 1.0 - cos


ZFS = [1.0, 0.5, 0.25]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
rows = []

for ax, zf in zip(axes, ZFS):
    route = make_route(zf)
    ss, se = route["shared_start"], route["shared_end"]
    zs, ze = route["zone_start"], route["zone_end"]
    delay = se - ze

    hA, hB = tpc_state_profile(route)
    si_tpc = splitter_index(hA, hB)
    gA, gB = hopfield_state_profile(route)
    si_hop = splitter_index(gA, gB)

    t = np.arange(len(si_tpc))
    ax.axvspan(ss - 0.5, se - 0.5, color="0.92", zorder=0,
               label="shared corridor")
    if ze > zs:
        ax.axvspan(zs - 0.5, ze - 0.5, color="#cfe8cf", zorder=1,
                   label="odour available")
    ax.plot(t, si_tpc, "-o", ms=4, color="tab:purple", zorder=3,
            label="tPC (carried latent z)")
    ax.plot(t, si_hop, "-s", ms=4, color="tab:orange", zorder=3,
            label="Hopfield (no carried state)")
    ax.axvline(se - 0.5, color="k", ls="--", lw=1.2, zorder=2)
    ax.annotate("divergence", xy=(se - 0.5, 0.98), xycoords=("data", "axes fraction"),
                fontsize=7, rotation=90, va="top", ha="right")
    ax.set_title(f"zone_fraction={zf}  →  cue-free delay = {delay} steps",
                 fontsize=10)
    ax.set_xlabel("step t")
    ax.set_ylabel("splitter index   1 − cos(h_A, h_B)")
    ax.legend(fontsize=7, loc="upper left")

    if se > ze:
        m_tpc = float(np.mean(si_tpc[ze:se]))
        m_hop = float(np.mean(si_hop[ze:se]))
    else:
        m_tpc = m_hop = float("nan")   # fully-cued rung: no cue-free steps
    rows.append((zf, delay, ss, se, zs, ze, m_tpc, m_hop,
                 float(si_tpc[se - 1]), float(si_hop[se - 1])))

print(f"{'zf':>6} {'delay':>6} {'shared':>9} {'zone':>9} "
      f"{'tPC cue-free':>13} {'Hop cue-free':>13} {'tPC@div':>9} {'Hop@div':>9}")
for zf, delay, ss, se, zs, ze, mt, mh, dt_, dh in rows:
    print(f"{zf:>6} {delay:>6} {f'[{ss},{se})':>9} {f'[{zs},{ze})':>9} "
          f"{mt:>13.4f} {mh:>13.4f} {dt_:>9.4f} {dh:>9.4f}")

fig.suptitle("Splitter index: internal separation across the shared corridor "
             "(inputs are IDENTICAL right of the green band)", fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "splitter_index_demo.png")
fig.savefig(out, dpi=150)
print("saved", out)


# ---------------------------------------------------------------------------
# What the run shows (seed 42, the numbers printed above):
#
#   zone_fraction  delay   tPC cue-free mean   Hopfield cue-free mean
#          1.0        0            n/a                    n/a
#          0.5        4         0.0492                -0.0000
#         0.25        6         0.0600                -0.0000
#
# Three readings that branch accuracy collapses into the single number 0.5:
#
#  1. Hopfield is EXACTLY zero across the cue-free stretch. Its per-step state
#     is a pure function of the current input, and the two routes' inputs are
#     identical there, so no separation can exist. "Architecturally incapable."
#  2. tPC is positive but decays fast: ~0.05-0.06 averaged over the cue-free
#     stretch, but only 0.001-0.002 by the divergence step. "Carries a trace
#     and loses it before the decision" -- a different failure from (1), and
#     invisible to any behavioural readout.
#  3. At delay 0 both arms separate strongly (0.42 / 0.48). The odour is in the
#     input there, so that separation is SENSED, not carried. This is why the
#     metric is summarised over [zone_end, shared_end) and not over the whole
#     shared corridor.
#
# Right of the divergence step both curves rise for a trivial reason: the
# inputs genuinely differ there. That region is outside the metric's window.
