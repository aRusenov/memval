"""The two metric-time sections, and the capability gate in front of them.

`interval_retention` (5.4) and `interval_generation` (5.5) carry half of Serial
order's weight. They are the only sections gated on a declared capability, and
the gate is the part most worth testing: an arm whose `fit_sequence` takes
`**kwargs` will happily accept `intervals=[...]` and drop them, so a duck-typed
check would run the section on an ordinal-clocked arm and score the resulting
floor as a model failure rather than an abstention.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin"))

from memval.benchmarks.interval_timing import (  # noqa: E402
    CONT_LONG,
    CONT_SHORT,
    LONG_GAP,
    PREFIX,
    SHORT_GAP,
    interval_sections_applicable,
    run_interval_generation,
    run_interval_retention,
)
from memval.models.baselines import (  # noqa: E402
    AsymmetricHopfieldNetwork,
    DTSESNSequenceNetwork,
    MultilayerTemporalPCNetwork,
)

DIM = 64

#: Exactly the keys the scorer's SPEC expects from each section. A section that
#: stops emitting one of these silently drops a scored metric; one that emits
#: something new fails `test_pipelines.py`'s classification check.
RETENTION_SCORED = {"interval_discrimination_acc", "interval_switch_sharpness",
                    "interval_encoded"}
GENERATION_SCORED = {"tempo_reproduction_error", "rhythm_pause_position_acc",
                     "peak_time_error", "weber_slope"}


def _dts():
    return DTSESNSequenceNetwork(
        n_features=DIM, n_units=400, tau_min=0.1, tau_max=20.0,
        spectral_radius=0.9, dt=0.05, predict_timing=True, seed=42)


# ---------------------------------------------------------------- gating ---
@pytest.mark.parametrize("cls,expected", [
    (DTSESNSequenceNetwork, {"interval_retention": True, "interval_generation": True}),
    (AsymmetricHopfieldNetwork, {"interval_retention": False, "interval_generation": False}),
    (MultilayerTemporalPCNetwork, {"interval_retention": False, "interval_generation": False}),
])
def test_capability_gate_is_nominal_not_duck_typed(cls, expected):
    """Only a declared capability opens these sections.

    `AsymmetricHopfieldNetwork.fit_sequence` accepts `**kwargs`, so it would take
    `intervals=` without error and stay clocked by ordinal. That is precisely why
    the gate reads the declaration rather than the signature.
    """
    assert interval_sections_applicable(cls) == expected


def test_an_ordinal_arm_would_silently_swallow_intervals():
    """The failure the gate prevents, demonstrated rather than asserted in prose."""
    m = AsymmetricHopfieldNetwork(n_features=DIM, learning_rate=0.1)
    x = np.random.default_rng(0).normal(size=(4, DIM))
    m.fit_sequence(x, intervals=[0.0, 1.0, 8.0, 1.0])   # accepted...
    a = m.predict_next(x[0])
    m2 = AsymmetricHopfieldNetwork(n_features=DIM, learning_rate=0.1)
    m2.fit_sequence(x)                                   # ...and identical without
    assert np.allclose(a, m2.predict_next(x[0])), (
        "AsymmetricHopfieldNetwork appears to honour `intervals=`; if that is now "
        "true it should declare TemporallyClocked, and this test should change.")


# ------------------------------------------------------------ 5.4 shape ----
def test_interval_retention_emits_its_scored_keys():
    out = run_interval_retention(_dts, n_reps=10, embedding_dim=DIM, n_trials=4)
    assert RETENTION_SCORED <= set(out["metrics"])
    for k in RETENTION_SCORED:
        v = out["metrics"][k]
        assert v != v or 0.0 <= v <= 1.0, f"{k}={v} outside [0,1]"
    sweep = out["series"]["interval_sweep"]
    assert len(sweep["gaps"]) == len(sweep["branch"]) == len(sweep["words"])


def test_interval_retention_runs_the_gaps_removed_control():
    """`interval_control_acc` must come from a twin trained WITHOUT intervals.

    Without this control the discrimination score is uninterpretable: any
    accuracy above chance could come from the two continuations differing in
    something other than their timing. The control shares the arm, the stream and
    the code path, and differs only in whether the gaps are passed.
    """
    out = run_interval_retention(_dts, n_reps=10, embedding_dim=DIM, n_trials=4)
    assert "interval_control_acc" in out["metrics"]
    ctrl = out["metrics"]["interval_control_acc"]
    assert 0.0 <= ctrl <= 1.0
    # The ordinal twin sees two contradictory transitions out of the same prefix,
    # so it cannot beat chance on a two-branch choice other than by luck.
    assert ctrl <= 0.5 + 1e-9, (
        f"the gaps-removed control discriminated at {ctrl:.2f}; the stream is "
        f"leaking something other than elapsed time, and 5.4's score cannot be "
        f"read as an interval result until that is found.")


def test_a_trained_dts_esn_actually_uses_the_interval():
    """The capacity claim itself, at the smallest exposure that shows it."""
    out = run_interval_retention(_dts, n_reps=40, embedding_dim=DIM, n_trials=4)
    assert out["metrics"]["interval_discrimination_acc"] > 0.5
    assert out["metrics"]["interval_encoded"] > 0.0, (
        "the read-out gave one answer for every gap, i.e. the interval changed "
        "nothing")
    xg = out["metrics"]["interval_crossing_gap"]
    assert SHORT_GAP <= xg <= LONG_GAP, (
        f"branch crossing at {xg}s falls outside the two trained gaps "
        f"({SHORT_GAP}, {LONG_GAP})")


# ------------------------------------------------------------ 5.5 shape ----
def test_interval_generation_emits_its_scored_keys():
    out = run_interval_generation(_dts, n_reps=10, embedding_dim=DIM)
    assert GENERATION_SCORED <= set(out["metrics"])
    assert 0.0 <= out["metrics"]["rhythm_pause_position_acc"] <= 1.0
    assert out["metrics"]["tempo_reproduction_error"] >= 0.0
    ser = out["series"]["interval_generation"]
    assert len(ser["rhythm_gaps"]) == len(ser["rhythm_true_gaps"])


def test_generation_reproduces_tempo_and_rhythm():
    out = run_interval_generation(_dts, n_reps=40, embedding_dim=DIM)
    assert out["metrics"]["tempo_reproduction_error"] < 0.5, (
        "generated gaps are more than 50% off the trained tempo")
    assert out["metrics"]["rhythm_pause_position_acc"] == 1.0, (
        "the longest generated gap did not fall where the trained sequence paused")


def test_the_two_sections_need_different_streams():
    """5.5's negative control: the timing head on 5.4's ambiguous stream.

    A gap that carries information cannot also be predictable from the items. Run
    the timing head on interval-as-cue's stream and the prediction should collapse
    toward the mean of the two trained gaps — which is why these are two sections
    rather than two read-outs of one, and why the audit lists them as separate
    dimensions.
    """
    out = run_interval_generation(_dts, n_reps=40, embedding_dim=DIM)
    collapse = out["metrics"]["interval_ambiguous_gap_collapse"]
    assert collapse > 0.5, (
        f"collapse={collapse:.2f}: the timing head resolved a gap it should not "
        f"be able to resolve, so the two streams are not as independent as the "
        f"section design assumes.")
