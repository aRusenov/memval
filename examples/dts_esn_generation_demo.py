"""
Autonomous generation with the DTS-ESN timing head: the model produces both
*what* comes next and *when*.

This needs the opposite stimulus design from `dts_esn_interval_demo.py`. There,
the gap had to be *unpredictable* from the items so it could carry information.
Here it must be *determined* by the history so the timing head can learn it —
you cannot test both capacities on one stream.

  test 1 (tempo)  : two sequences, distinguishable by their items, ingested at
                    different tempos. Generation should reproduce each tempo.
  test 2 (rhythm) : one sequence with an internal pause. Generation should put
                    the pause back in the right position.

Run:  python examples/dts_esn_generation_demo.py
"""

import numpy as np

from memval.encoders.symbolic import SymbolicDecoder, SymbolicEncoder
from memval.models.baselines import DTSESNSequenceNetwork

EMBED_DIM = 64
N_REPS = 40
SEED = 0

FAST = ["do", "re", "mi", "fa"]
SLOW = ["ka", "ki", "ko", "ku"]
RHYTHM = ["r1", "r2", "r3", "r4", "r5", "r6"]

FAST_TEMPO, SLOW_TEMPO = 0.5, 4.0
RHYTHM_GAPS = [0.0, 0.5, 0.5, 3.0, 0.5, 0.5]      # pause between r3 and r4


def make_model(encoder, **kw):
    model = DTSESNSequenceNetwork(
        n_features=EMBED_DIM, n_units=400, tau_min=0.1, tau_max=20.0,
        spectral_radius=0.9, dt=0.05, predict_timing=True, seed=SEED, **kw
    )
    # codebook cleanup on the feedback path, as measure_recall_autoregressive does
    emb = encoder.embeddings

    def cleanup(y):
        n = np.linalg.norm(y) or 1.0
        return emb[int(np.argmax(emb @ (y / n)))]

    model.decode_prediction = cleanup
    return model


def main():
    # ---------------- test 1: tempo ---------------------------------------
    vocab = FAST + SLOW
    enc = SymbolicEncoder(vocab, embedding_dim=EMBED_DIM, seed=SEED)
    dec = SymbolicDecoder(enc)
    model = make_model(enc)

    ev_fast, ev_slow = enc.encode(FAST), enc.encode(SLOW)
    iv_fast = [0.0] + [FAST_TEMPO] * (len(FAST) - 1)
    iv_slow = [0.0] + [SLOW_TEMPO] * (len(SLOW) - 1)

    for _ in range(N_REPS):
        model.fit_sequence(ev_fast, intervals=iv_fast)
        model.fit_sequence(ev_slow, intervals=iv_slow)

    print("=" * 72)
    print("1. TEMPO: cue with the first item, model generates items AND timing")
    print("=" * 72)
    for label, seq, ev, tempo in (
        ("fast", FAST, ev_fast, FAST_TEMPO),
        ("slow", SLOW, ev_slow, SLOW_TEMPO),
    ):
        events, gaps = model.generate(ev[0], length=3)
        words = dec.decode(events)
        print(f"  cue {seq[0]!r:>6} -> items {words}   expected {seq[1:]}")
        print(f"  {'':>13} gaps  {np.round(gaps, 2).tolist()}"
              f"   true tempo {tempo}")
    print()

    # ---------------- test 2: rhythm --------------------------------------
    enc2 = SymbolicEncoder(RHYTHM, embedding_dim=EMBED_DIM, seed=SEED + 1)
    dec2 = SymbolicDecoder(enc2)
    model2 = make_model(enc2)
    ev_r = enc2.encode(RHYTHM)

    for _ in range(N_REPS):
        model2.fit_sequence(ev_r, intervals=RHYTHM_GAPS)

    print("=" * 72)
    print("2. RHYTHM: one sequence with an internal pause (r3 -> r4 is 3.0s)")
    print("=" * 72)
    events, gaps = model2.generate(ev_r[0], length=5)
    words = dec2.decode(events)
    true_gaps = RHYTHM_GAPS[1:]
    print(f"  items     {words}")
    print(f"  expected  {RHYTHM[1:]}")
    print(f"  gaps      {np.round(gaps, 2).tolist()}")
    print(f"  true gaps {true_gaps}")
    err = np.abs(np.array(gaps) - np.array(true_gaps))
    print(f"  abs error {np.round(err, 3).tolist()}   mean {err.mean():.3f}s")
    print()

    pause_idx = int(np.argmax(gaps))
    true_idx = int(np.argmax(true_gaps))
    print(f"  longest generated gap at step {pause_idx} "
          f"(true pause at step {true_idx}) -> "
          f"{'PASS' if pause_idx == true_idx else 'FAIL'}")

    # ---------------- the negative control --------------------------------
    print()
    print("=" * 72)
    print("3. CONTROL: timing head on the ambiguous interval-as-cue stream")
    print("=" * 72)
    prefix, cl, cs = ["red", "green", "blue"], ["dog"], ["one"]
    enc3 = SymbolicEncoder(prefix + cl + cs, embedding_dim=EMBED_DIM, seed=SEED)
    model3 = make_model(enc3)
    evL, evS = enc3.encode(prefix + cl), enc3.encode(prefix + cs)
    for _ in range(N_REPS):
        model3.fit_sequence(evL, intervals=[0.0, 1.0, 1.0, 8.0])
        model3.fit_sequence(evS, intervals=[0.0, 1.0, 1.0, 1.0])
    prompt = enc3.encode(prefix)
    predicted = model3.predict_time_to_next(prompt, prompt_intervals=[0.0, 1.0, 1.0])
    print(f"  gap predicted after 'blue': {predicted:.2f}s")
    print(f"  the two trained gaps were 8.0s and 1.0s (mean 4.5s)")
    print("  -> as expected, an ambiguous gap collapses toward the mean.")
    print("     Interval-as-cue and interval-as-output need different streams.")


if __name__ == "__main__":
    main()
