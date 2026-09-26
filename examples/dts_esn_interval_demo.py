"""
Minimal check that the DTS-ESN honours an explicit inter-event interval.

The design makes the interval *load-bearing*: the same word sequence is followed
by two different continuations, and the only thing distinguishing them is how
much time elapsed. If the arm reproduces the right continuation for the right
gap, elapsed time has genuinely been absorbed into the reservoir state — an
arm that merely memorised "blue -> something" cannot pass.

    stream L :  red -> green -> blue --[ 8.0 s ]--> dog -> cat -> bird
    stream S :  red -> green -> blue --[ 1.0 s ]--> one -> two -> three

Both streams are ingested online, interleaved, via `fit_event`. At test time we
cue with the shared prefix and vary only `elapsed`.

`AsymmetricHopfieldNetwork` is run on the identical streams as a control: it is
clocked by event ordinal, so it sees two contradictory transitions out of
"blue" and has no way to tell them apart.

Run:  python examples/dts_esn_interval_demo.py
"""

import numpy as np

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.models.baselines import AsymmetricHopfieldNetwork, DTSESNSequenceNetwork

EMBED_DIM = 64
SPACING = 1.0      # within-sequence inter-item interval
LONG_GAP = 8.0     # gap that should lead to continuation B
SHORT_GAP = 1.0    # gap that should lead to continuation C
N_REPS = 40
SEED = 0

PREFIX = ["red", "green", "blue"]
CONT_LONG = ["dog", "cat", "bird"]
CONT_SHORT = ["one", "two", "three"]
VOCAB = PREFIX + CONT_LONG + CONT_SHORT


def build_streams(encoder):
    """Return (events, intervals) for both streams. intervals[t] is the gap
    *preceding* event t."""
    seq_long = PREFIX + CONT_LONG
    seq_short = PREFIX + CONT_SHORT
    # gap before each item: 0 for the first, SPACING within a run, GAP at the seam
    iv_long = [0.0, SPACING, SPACING, LONG_GAP, SPACING, SPACING]
    iv_short = [0.0, SPACING, SPACING, SHORT_GAP, SPACING, SPACING]
    return (
        (encoder.encode(seq_long), iv_long),
        (encoder.encode(seq_short), iv_short),
    )


def main():
    encoder = SymbolicEncoder(VOCAB, embedding_dim=EMBED_DIM, seed=SEED)
    decoder = SymbolicDecoder(encoder)
    (ev_long, iv_long), (ev_short, iv_short) = build_streams(encoder)

    model = DTSESNSequenceNetwork(
        n_features=EMBED_DIM,
        n_units=400,
        tau_min=0.1,
        tau_max=20.0,
        spectral_radius=0.9,
        dt=0.05,
        seed=SEED,
    )

    # --- online, interleaved ingestion --------------------------------------
    for _ in range(N_REPS):
        model.fit_sequence(ev_long, intervals=iv_long)
        model.fit_sequence(ev_short, intervals=iv_short)

    prompt = encoder.encode(PREFIX)
    prompt_iv = [0.0, SPACING, SPACING]

    print("=" * 68)
    print("1. Within-sequence recall (sanity: ordinary sequence learning works)")
    print("=" * 68)
    for cue_len in (1, 2):
        cue = prompt[:cue_len]
        pred = model.predict_next(
            cue, prompt_intervals=prompt_iv[:cue_len], elapsed=SPACING
        )
        word, score = decoder.decode_with_score(pred)
        print(f"  cue {PREFIX[:cue_len]!s:<28} + {SPACING}s -> {word:<8} ({score:.3f})"
              f"   expected {PREFIX[cue_len]}")

    print()
    print("=" * 68)
    print("2. THE TEST: same prefix, interval is the only disambiguator")
    print("=" * 68)
    results = {}
    for label, gap, expected in (
        ("LONG ", LONG_GAP, CONT_LONG[0]),
        ("SHORT", SHORT_GAP, CONT_SHORT[0]),
    ):
        pred = model.predict_next(prompt, prompt_intervals=prompt_iv, elapsed=gap)
        word, score = decoder.decode_with_score(pred)
        results[label] = word
        mark = "PASS" if word == expected else "FAIL"
        print(f"  {label} gap={gap:>5.1f}s -> {word:<8} ({score:.3f})"
              f"   expected {expected:<6} [{mark}]")

    print()
    print("=" * 68)
    print("3. Interval sweep: where does the readout switch continuation?")
    print("=" * 68)
    for gap in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0]:
        pred = model.predict_next(prompt, prompt_intervals=prompt_iv, elapsed=gap)
        word, score = decoder.decode_with_score(pred)
        print(f"  gap={gap:>5.1f}s -> {word:<8} ({score:.3f})")

    print()
    print("=" * 68)
    print("4. Full rollout of each branch (3 steps after the gap)")
    print("=" * 68)
    for label, gap, expected in (
        ("LONG ", LONG_GAP, CONT_LONG),
        ("SHORT", SHORT_GAP, CONT_SHORT),
    ):
        rolled = model.recall(
            prompt, length=3,
            prompt_intervals=prompt_iv,
            intervals=[gap, SPACING, SPACING],
        )
        words = decoder.decode(rolled)
        print(f"  {label} gap={gap:>5.1f}s -> {words}   expected {expected}")

    print()
    print("=" * 68)
    print("5. CONTROL: ordinal-clocked arm (AHN) on the identical streams")
    print("=" * 68)
    ahn = AsymmetricHopfieldNetwork(n_features=EMBED_DIM, learning_rate=0.1)
    for _ in range(N_REPS):
        ahn.fit_sequence(ev_long)
        ahn.fit_sequence(ev_short)
    blue = encoder.encode(["blue"])[0]
    ahn_word, ahn_score = decoder.decode_with_score(ahn.predict_next(blue))
    print(f"  AHN after 'blue' -> {ahn_word:<8} ({ahn_score:.3f})")
    print("  (one answer only: AHN receives no interval, so both continuations")
    print("   collapse onto the same transition regardless of the gap)")

    print()
    verdict = (results.get("LONG ") == CONT_LONG[0]
               and results.get("SHORT") == CONT_SHORT[0])
    print(f"VERDICT: interval {'IS' if verdict else 'IS NOT'} honoured by DTS-ESN")


if __name__ == "__main__":
    main()
