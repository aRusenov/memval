"""Serial order, metric time: the two sections that ask about *when*.

`docs/capacity_questions.md` gives Serial order five questions. Three of them —
establishment, binding, unrolling — are ordinal: they ask what order events came
in. The remaining two, **5.4 and 5.5, carry half the capacity's weight between
them**, and both ask about elapsed time rather than event count:

    5.4  interval_retention   Does the memory retain *how long* separated two
                              events -- enough that the interval alone can decide
                              what comes next?
    5.5  interval_generation  Can it reproduce a sequence with its original tempo
                              and pauses, not just its order?

Until now neither was a wired section, so every arm reported Serial-order
coverage of 30% and the half of the capacity that needs a clock was unmeasured —
including on `DTSESNSequenceNetwork`, the one arm in the taxonomy that declares
the capability to answer it.

Why these are two sections and not one
--------------------------------------
They need **opposite stimulus designs**, and this is the reason the audit records
them as separate dimensions rather than two read-outs of one sweep:

* For 5.4 the gap must be *unpredictable from the items*, or it carries no
  information and cannot disambiguate anything.
* For 5.5 the gap must be *determined by the history*, or the timing head has
  nothing learnable to reproduce.

A single stream cannot satisfy both. `examples/dts_esn_generation_demo.py`
demonstrates the failure directly: run the timing head on 5.4's ambiguous stream
and the predicted gap collapses toward the mean of the two trained gaps. That
control is reproduced here as `interval_ambiguous_gap_collapse`.

Capability gating
-----------------
Both sections are gated on a **declared** capability, never on duck-typing:
5.4 on ``TemporallyClocked``, 5.5 on the strictly stronger ``TimingPredictive``.
An arm that declares neither is not run and is scored *not applicable* — its
`fit_sequence` would accept ``intervals=`` through ``**kwargs`` and silently drop
them, which is exactly the failure mode that would report a capability gap as a
performance one.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from memval.encoders.symbolic import SymbolicDecoder, SymbolicEncoder
from memval.models.capabilities import predicts_timing, supports_intervals

# --------------------------------------------------------------------------
# 5.4 interval-as-cue
# --------------------------------------------------------------------------
#: Shared prefix, then two continuations that ONLY the elapsed gap distinguishes.
#: Kept identical to examples/dts_esn_interval_demo.py so the section and the
#: demo cannot drift apart.
PREFIX = ["red", "green", "blue"]
CONT_LONG = ["dog", "cat", "bird"]
CONT_SHORT = ["one", "two", "three"]

SPACING = 1.0     #: within-run inter-item gap
LONG_GAP = 8.0    #: the gap that should lead to CONT_LONG
SHORT_GAP = 1.0   #: the gap that should lead to CONT_SHORT

#: Log-spaced, and spanning both trained gaps with room either side. The sweep is
#: what `interval_switch_sharpness` is measured on, so it must be dense enough
#: for "gradual" and "abrupt" to be distinguishable and wide enough to contain
#: the crossing.
GAP_SWEEP = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0)

# --------------------------------------------------------------------------
# 5.5 interval-as-output
# --------------------------------------------------------------------------
FAST = ["do", "re", "mi", "fa"]
SLOW = ["ka", "ki", "ko", "ku"]
RHYTHM = ["r1", "r2", "r3", "r4", "r5", "r6"]

FAST_TEMPO, SLOW_TEMPO = 0.5, 4.0
#: One internal pause, between r3 and r4. `rhythm_pause_position_acc` asks
#: whether autonomous generation puts it back in that position.
RHYTHM_GAPS = (0.0, 0.5, 0.5, 3.0, 0.5, 0.5)


def _decode(decoder: SymbolicDecoder, vec: np.ndarray) -> str:
    return decoder.decode(np.asarray(vec), top_k=1)[0]


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# ==========================================================================
# 5.4  interval_retention
# ==========================================================================
def run_interval_retention(
    model_factory: Callable[[], Any],
    *,
    n_reps: int = 40,
    embedding_dim: int = 64,
    n_trials: int = 1,
    seed: int = 42,
    noise_scale: float = 0.0,
) -> Dict[str, Any]:
    """Can the elapsed gap alone decide what comes next?

    One prefix, two continuations, and nothing but the interval telling them
    apart. An ordinal-clocked arm sees two contradictory transitions out of the
    same item and can only pick one; an arm that has absorbed elapsed time into
    its state returns whichever continuation matches the gap it is given.

    `model_factory` builds a fresh arm, so the ordinal control can be trained on
    the identical stream through the identical code path.
    """
    rng = np.random.default_rng(seed)
    vocab = PREFIX + CONT_LONG + CONT_SHORT
    enc = SymbolicEncoder(vocab, embedding_dim=embedding_dim, seed=seed)
    dec = SymbolicDecoder(enc)

    ev_long = enc.encode(PREFIX + CONT_LONG)
    ev_short = enc.encode(PREFIX + CONT_SHORT)
    iv_long = [0.0, SPACING, SPACING, LONG_GAP, SPACING, SPACING]
    iv_short = [0.0, SPACING, SPACING, SHORT_GAP, SPACING, SPACING]

    prompt = enc.encode(PREFIX)
    prompt_iv = [0.0, SPACING, SPACING]

    def train(with_intervals: bool) -> Any:
        """Interleaved, so neither branch is merely the more recent one."""
        m = model_factory()
        for _ in range(n_reps):
            if with_intervals:
                m.fit_sequence(ev_long, intervals=iv_long)
                m.fit_sequence(ev_short, intervals=iv_short)
            else:
                m.fit_sequence(ev_long)
                m.fit_sequence(ev_short)
        return m

    model = train(with_intervals=True)

    def branch_at(m: Any, gap: float, cue: np.ndarray,
                  cue_iv: Optional[Sequence[float]]) -> str:
        kw: Dict[str, Any] = {"elapsed": gap}
        if cue_iv is not None:
            kw["prompt_intervals"] = list(cue_iv)
        return _decode(dec, m.predict_next(cue, **kw))

    # ---- the test: right continuation for the right gap ------------------
    # Scored over noisy repeats of the cue so a single lucky decode cannot carry
    # the metric, matching how every other section probes.
    hits, n = 0, 0
    for _ in range(n_trials):
        cue = prompt + rng.normal(0.0, noise_scale, prompt.shape) if noise_scale else prompt
        for gap, expected in ((LONG_GAP, CONT_LONG[0]), (SHORT_GAP, CONT_SHORT[0])):
            hits += int(branch_at(model, gap, cue, prompt_iv) == expected)
            n += 1
    discrimination = hits / n if n else float("nan")

    # ---- the sweep: where, and how abruptly, does the branch switch? -----
    sweep_words = [branch_at(model, g, prompt, prompt_iv) for g in GAP_SWEEP]
    sweep_branch = [1 if w in CONT_LONG else (0 if w in CONT_SHORT else -1)
                    for w in sweep_words]

    # `interval_encoded` is the guard-style question -- did the gap change the
    # answer AT ALL -- so it is deliberately blind to whether the answer is
    # right. 0.0 = one answer for every gap (the ordinal-clocked signature);
    # 1.0 = the two branches are chosen equally often across the sweep.
    known = [b for b in sweep_branch if b >= 0]
    if known:
        p_long = sum(known) / len(known)
        interval_encoded = _clip01(2.0 * min(p_long, 1.0 - p_long))
    else:
        interval_encoded = 0.0

    # Sharpness: a model that honours the interval crosses ONCE, near the
    # trained gap. One that has absorbed some slow drift wanders back and forth.
    # 1.0 = a single clean crossing; 0.0 = never switches, or switches at every
    # step. Undefined-by-construction cases return NaN rather than 0, so the
    # scorer excludes them instead of reading "no switch" as "bad switch".
    transitions = sum(1 for a, b in zip(sweep_branch, sweep_branch[1:])
                      if a >= 0 and b >= 0 and a != b)
    n_pairs = sum(1 for a, b in zip(sweep_branch, sweep_branch[1:])
                  if a >= 0 and b >= 0)
    if n_pairs == 0:
        sharpness = float("nan")
    elif transitions == 0:
        sharpness = 0.0                      # never switched: no dependence
    else:
        sharpness = _clip01(1.0 - (transitions - 1) / max(n_pairs - 1, 1))

    # ---- the control the demo insists on ---------------------------------
    # The SAME stream with the gaps removed, through the SAME arm. If this also
    # discriminates, the design leaked something other than time.
    control = train(with_intervals=False)
    c_hits, c_n = 0, 0
    for gap, expected in ((LONG_GAP, CONT_LONG[0]), (SHORT_GAP, CONT_SHORT[0])):
        c_hits += int(branch_at(control, gap, prompt, None) == expected)
        c_n += 1
    control_acc = c_hits / c_n if c_n else float("nan")

    crossing = float("nan")
    for (g0, b0), (g1, b1) in zip(zip(GAP_SWEEP, sweep_branch),
                                  zip(GAP_SWEEP[1:], sweep_branch[1:])):
        if b0 >= 0 and b1 >= 0 and b0 != b1:
            crossing = float(math.sqrt(g0 * g1))   # geometric midpoint, log axis
            break

    return {
        "metrics": {
            "interval_discrimination_acc": float(discrimination),
            "interval_switch_sharpness": float(sharpness),
            "interval_encoded": float(interval_encoded),
            "interval_control_acc": float(control_acc),
            "interval_crossing_gap": crossing,
            "interval_n_reps": int(n_reps),
        },
        "series": {
            "interval_sweep": {
                "gaps": list(GAP_SWEEP),
                "branch": sweep_branch,
                "words": sweep_words,
                "trained_gaps": [SHORT_GAP, LONG_GAP],
            }
        },
    }


# ==========================================================================
# 5.5  interval_generation
# ==========================================================================
def run_interval_generation(
    model_factory: Callable[[], Any],
    *,
    n_reps: int = 40,
    embedding_dim: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    """Does autonomous rollout reproduce the tempo and the rhythm?

    Two stimuli, because tempo and rhythm are different claims: a model can hold
    one global rate and still not know *where* the pause goes.

    `model_factory` must build an arm declaring ``TimingPredictive``; the section
    drives `generate()` (items **and** gaps) and `predict_time_to_next()`.
    """
    # ---- test 1: tempo ---------------------------------------------------
    vocab = FAST + SLOW
    enc = SymbolicEncoder(vocab, embedding_dim=embedding_dim, seed=seed)
    dec = SymbolicDecoder(enc)
    ev_fast, ev_slow = enc.encode(FAST), enc.encode(SLOW)
    iv_fast = [0.0] + [FAST_TEMPO] * (len(FAST) - 1)
    iv_slow = [0.0] + [SLOW_TEMPO] * (len(SLOW) - 1)

    model = model_factory()
    _attach_codebook_cleanup(model, enc)
    for _ in range(n_reps):
        model.fit_sequence(ev_fast, intervals=iv_fast)
        model.fit_sequence(ev_slow, intervals=iv_slow)

    tempo_rows: List[Dict[str, Any]] = []
    rel_errors: List[float] = []
    for label, seq, ev, tempo in (("fast", FAST, ev_fast, FAST_TEMPO),
                                  ("slow", SLOW, ev_slow, SLOW_TEMPO)):
        events, gaps = model.generate(ev[0], length=len(seq) - 1)
        gaps = np.asarray(gaps, dtype=float).ravel()
        words = [_decode(dec, e) for e in np.asarray(events)]
        # Normalised by the trained gap so fast and slow contribute equally --
        # an absolute error would let the slow tempo dominate by construction.
        rel = np.abs(gaps - tempo) / tempo
        rel_errors.extend(rel.tolist())
        tempo_rows.append({"label": label, "true_tempo": tempo,
                           "gaps": gaps.tolist(), "words": words,
                           "expected": seq[1:],
                           "item_acc": float(np.mean([w == t for w, t
                                                      in zip(words, seq[1:])]))})
    tempo_error = float(np.mean(rel_errors)) if rel_errors else float("nan")

    # Weber: does timing variability scale with the interval? Diagnostic of the
    # mechanism, not of quality -- scalar timing predicts a positive slope.
    weber = float("nan")
    spreads = [(r["true_tempo"], float(np.std(r["gaps"]))) for r in tempo_rows]
    if len(spreads) >= 2 and spreads[1][0] != spreads[0][0]:
        weber = float((spreads[1][1] - spreads[0][1])
                      / (spreads[1][0] - spreads[0][0]))

    # ---- test 2: rhythm --------------------------------------------------
    enc2 = SymbolicEncoder(list(RHYTHM), embedding_dim=embedding_dim, seed=seed + 1)
    dec2 = SymbolicDecoder(enc2)
    ev_r = enc2.encode(list(RHYTHM))
    model2 = model_factory()
    _attach_codebook_cleanup(model2, enc2)
    for _ in range(n_reps):
        model2.fit_sequence(ev_r, intervals=list(RHYTHM_GAPS))

    events, gaps = model2.generate(ev_r[0], length=len(RHYTHM) - 1)
    gaps = np.asarray(gaps, dtype=float).ravel()
    true_gaps = np.asarray(RHYTHM_GAPS[1:], dtype=float)
    rhythm_words = [_decode(dec2, e) for e in np.asarray(events)]
    pause_acc = float(int(np.argmax(gaps) == np.argmax(true_gaps))) if gaps.size else 0.0

    # ---- peak time: the time-cell read-out, teacher-forced ---------------
    # Not from the rollout: this asks whether the arm can say WHEN the next event
    # is due given the true history, which is upstream of reproducing it
    # autonomously and fails for different reasons.
    peak_errs: List[float] = []
    for t in range(1, len(RHYTHM) - 1):
        true_gap = float(RHYTHM_GAPS[t + 1])
        if true_gap <= 0:
            continue
        pred = float(model2.predict_time_to_next(
            ev_r[:t + 1], prompt_intervals=list(RHYTHM_GAPS[:t + 1])))
        peak_errs.append(abs(pred - true_gap) / true_gap)
    peak_error = float(np.mean(peak_errs)) if peak_errs else float("nan")

    # ---- the negative control, from the generation demo -------------------
    # The timing head on 5.4's AMBIGUOUS stream, where the gap is not determined
    # by the items. It should collapse toward the mean of the two trained gaps,
    # and that is the evidence that 5.4 and 5.5 need different streams rather
    # than being two read-outs of one.
    enc3 = SymbolicEncoder(PREFIX + CONT_LONG[:1] + CONT_SHORT[:1],
                           embedding_dim=embedding_dim, seed=seed)
    model3 = model_factory()
    _attach_codebook_cleanup(model3, enc3)
    ev_l = enc3.encode(PREFIX + CONT_LONG[:1])
    ev_s = enc3.encode(PREFIX + CONT_SHORT[:1])
    for _ in range(n_reps):
        model3.fit_sequence(ev_l, intervals=[0.0, SPACING, SPACING, LONG_GAP])
        model3.fit_sequence(ev_s, intervals=[0.0, SPACING, SPACING, SHORT_GAP])
    ambiguous = float(model3.predict_time_to_next(
        enc3.encode(PREFIX), prompt_intervals=[0.0, SPACING, SPACING]))
    mean_gap = 0.5 * (LONG_GAP + SHORT_GAP)
    collapse = _clip01(1.0 - abs(ambiguous - mean_gap)
                       / max(abs(LONG_GAP - mean_gap), 1e-9))

    return {
        "metrics": {
            "tempo_reproduction_error": tempo_error,
            "rhythm_pause_position_acc": pause_acc,
            "peak_time_error": peak_error,
            "weber_slope": weber,
            "interval_ambiguous_gap_collapse": collapse,
            "generation_n_reps": int(n_reps),
        },
        "series": {
            "interval_generation": {
                "tempo": tempo_rows,
                "rhythm_gaps": gaps.tolist(),
                "rhythm_true_gaps": true_gaps.tolist(),
                "rhythm_words": rhythm_words,
                "rhythm_expected": list(RHYTHM[1:]),
                "ambiguous_predicted_gap": ambiguous,
                "ambiguous_trained_gaps": [SHORT_GAP, LONG_GAP],
            }
        },
    }


def _attach_codebook_cleanup(model: Any, encoder: SymbolicEncoder) -> None:
    """Snap the rollout's feedback to the nearest codebook entry.

    The same projection `measure_recall_autoregressive(feedback_mode="quantized")`
    applies, and for the same reason: without it an autonomous rollout drifts off
    the encoder's manifold within a step or two and the TIMING read-out is then
    being scored on a state the items never visit. It is the drift-free upper
    bound, and it is the protocol `examples/dts_esn_generation_demo.py` uses --
    stated here because it is a choice, not a detail.
    """
    emb = encoder.embeddings

    def cleanup(y: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(y)) or 1.0
        return emb[int(np.argmax(emb @ (np.asarray(y) / n)))]

    model.decode_prediction = cleanup


def interval_sections_applicable(model_class: Any) -> Dict[str, bool]:
    """Which of the two metric-time sections this arm can be asked at all.

    Nominal, per `memval/models/capabilities`: `TimingPredictive` is strictly
    stronger than `TemporallyClocked` -- absorbing a supplied interval is not the
    same as emitting one -- so an arm can be eligible for 5.4 and not 5.5.
    """
    return {
        "interval_retention": supports_intervals(model_class),
        "interval_generation": predicts_timing(model_class),
    }
