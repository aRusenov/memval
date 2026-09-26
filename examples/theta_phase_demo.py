"""
Minimal bootstrap for the theta-phase encode/retrieve arm
(Hasselmo, Bodelon & Wyble 2002).

Four checks:

1. **Sanity** --- the arm learns a chain online, one theta cycle per event, and
   recalls it.
2. **The controlled pair** --- at the paper's optimal phases the cycle integral
   reduces exactly to the delta rule, so this arm and `AsymmetricHopfieldNetwork`
   take *identical* steps. Same rule, two ways of deriving the error: explicit
   subtraction vs oscillatory phase separation.
3. **Figure 4** --- the paper's performance measure (eq 2.14) over the
   two-dimensional phase surface, computed analytically. Maximal when EC is in
   phase with LTP and CA3 is 180 degrees out, which is the model's central
   claim and the reason the defaults sit where they do.
4. **The fornix lesion** --- modulation depth X is the paper's lesion control.
   At X=0 the gates go constant, the zero-mean plasticity profile integrates
   them away, and nothing is learned, so an association acquired earlier
   persists unopposed.

Run:  python examples/theta_phase_demo.py
"""

import numpy as np

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.models.baselines import AsymmetricHopfieldNetwork, ThetaPhaseSequenceNetwork

EMBED_DIM = 64
SEED = 0
N_REPS = 60
LR = 0.1

SEQUENCES = {
    "fruit":  ["apple", "banana", "orange", "grape", "pear"],
    "animal": ["cat", "dog", "cow", "horse", "sheep"],
}
VOCAB = [w for seq in SEQUENCES.values() for w in seq]


def stream_online(model, sequences, n_reps):
    """Ingest every sequence event-by-event, interleaved across reps."""
    for _ in range(n_reps):
        for events in sequences:
            model.on_event_boundary()
            for ev in events:
                model.fit_event(ev)
    model.on_event_boundary()


def next_item_accuracy(model, sequences, decoder, words):
    """Fraction of within-sequence transitions recovered from a one-item cue."""
    hits = total = 0
    for events, labels in zip(sequences, words):
        for t in range(len(labels) - 1):
            pred = model.predict_next(events[t])
            hits += decoder.decode_with_score(pred)[0] == labels[t + 1]
            total += 1
    return hits / total


def main():
    encoder = SymbolicEncoder(VOCAB, embedding_dim=EMBED_DIM, seed=SEED)
    decoder = SymbolicDecoder(encoder)
    words = list(SEQUENCES.values())
    sequences = [encoder.encode(w) for w in words]

    print("=" * 72)
    print("1. Sanity: online learning through theta cycles")
    print("=" * 72)
    model = ThetaPhaseSequenceNetwork(n_features=EMBED_DIM, learning_rate=LR)
    stream_online(model, sequences, N_REPS)
    print(f"  next-item accuracy after {N_REPS} online reps : "
          f"{next_item_accuracy(model, sequences, decoder, words):.3f}")
    rolled = decoder.decode(model.recall(sequences[0][0], length=4))
    print(f"  autoregressive recall from '{words[0][0]}'     : {' -> '.join(rolled)}")
    print(f"  expected                                  : {' -> '.join(words[0][1:])}")

    print()
    print("=" * 72)
    print("2. Controlled pair: theta at the paper's optimum == AHN")
    print("=" * 72)
    theta = ThetaPhaseSequenceNetwork(n_features=EMBED_DIM, learning_rate=LR)
    ahn = AsymmetricHopfieldNetwork(n_features=EMBED_DIM, learning_rate=LR)
    for events in sequences:
        theta.fit_sequence(events)
        ahn.fit_sequence(events)
    c = theta.phase_coefficients()
    print(f"  max |W_theta - W_ahn|                     : "
          f"{float(np.max(np.abs(theta.W - ahn.W))):.3e}")
    print(f"  encode / retrieve / residual              : "
          f"{c['encode']:+.4f} / {c['retrieve']:+.4f} / {c['residual']:+.4f}")

    print()
    print("=" * 72)
    print("3. Figure 4: performance measure M over the phase surface (eq 2.14)")
    print("=" * 72)
    steps = [0.0, 0.5, 1.0, 1.5]
    print("            phi_LTP - phi_CA3 ->")
    print("  d_EC  " + "".join(f"{s:>8.1f}pi" for s in steps))
    for d_ec in steps:
        row = []
        for d_ca3 in steps:
            m = ThetaPhaseSequenceNetwork(
                n_features=EMBED_DIM,
                phase_ec=-d_ec * np.pi,
                phase_ca3=-d_ca3 * np.pi,
            )
            row.append(m.M())
        best = max(row) == max(
            ThetaPhaseSequenceNetwork(
                n_features=EMBED_DIM, phase_ec=-a * np.pi, phase_ca3=-b * np.pi
            ).M()
            for a in steps
            for b in steps
        )
        print(f"  {d_ec:>3.1f}pi " + "".join(f"{v:>10.3f}" for v in row)
              + ("   <- max" if best else ""))
    print("  Max at (d_EC=0, d_CA3=1.0pi): EC in phase with LTP, CA3 antiphase.")

    print()
    print("=" * 72)
    print("4. Fornix lesion: modulation depth X")
    print("=" * 72)
    for x in (1.0, 0.5, 0.0):
        m = ThetaPhaseSequenceNetwork(
            n_features=EMBED_DIM, learning_rate=LR, modulation_depth=x
        )
        stream_online(m, sequences, N_REPS)
        c = m.phase_coefficients()
        label = "intact" if x == 1.0 else ("lesion" if x == 0.0 else "partial")
        print(f"  X={x:<4} ({label})  encode {c['encode']:+.3f}  "
              f"retrieve {c['retrieve']:+.3f}  |W| {np.linalg.norm(m.W):8.4f}  "
              f"acc {next_item_accuracy(m, sequences, decoder, words):.3f}")
    print("  X=0: gates constant, cycle integral vanishes, nothing is learned --")
    print("  so a previously acquired association would persist unopposed.")


if __name__ == "__main__":
    main()
