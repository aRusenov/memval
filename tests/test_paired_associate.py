"""Tests for the AB/AC paired-associate section.

What is pinned here is the protocol and the response coding: that pairs are
trained independently rather than as a list, that the control condition trains
disjoint cues for an equal budget, and that the three response buckets partition
every probe. The interference *result* is a model property and belongs in a run,
not in a test.
"""
import numpy as np
import pytest

from memval.benchmarks.paired_associate import (
    PairedAssociateBenchmark,
    RESPONSE_BUCKETS,
)
from memval.encoders.symbolic import SymbolicDecoder, SymbolicEncoder

DIM = 32
N_PAIRS = 3

CUES_A = ["apple", "banana", "orange"]
TARGETS_B = ["cat", "dog", "cow"]
TARGETS_C = ["car", "truck", "bus"]
CUES_D = ["red", "blue", "green"]
TARGETS_E = ["one", "two", "three"]

VOCAB = dict(
    [(w, "fruit") for w in CUES_A] + [(w, "animal") for w in TARGETS_B]
    + [(w, "vehicle") for w in TARGETS_C] + [(w, "color") for w in CUES_D]
    + [(w, "number") for w in TARGETS_E]
)


@pytest.fixture(scope="module")
def datasets():
    enc = SymbolicEncoder(VOCAB, embedding_dim=DIM, category_variance=0.2, seed=3)
    return {
        "cues_A": CUES_A, "targets_B": TARGETS_B, "targets_C": TARGETS_C,
        "cues_D": CUES_D, "targets_E": TARGETS_E,
        "encoder": enc, "decoder": SymbolicDecoder(enc),
    }


class RecordingModel:
    """A perfect, instantly-overwriting pairwise associator.

    Stores one target per cue, keyed by the cue's exact embedding, and returns
    it for the nearest stored cue. That makes it the *overwriter* corner by
    construction, which is the behaviour the section is built to detect.
    """

    def __init__(self):
        self.pairs = []          # (cue_vec, target_vec) as trained
        self.fit_calls = []
        self.current_t = 0
        self.memory = {}

    def fit_sequence(self, sequence_data, epochs=None, **kw):
        self.fit_calls.append((np.array(sequence_data), epochs))
        for a, b in zip(sequence_data[:-1], sequence_data[1:]):
            self.memory[tuple(np.round(a, 6))] = np.array(b)

    def predict_next(self, current_event, current_context=None, **kw):
        if not self.memory:
            return np.zeros_like(current_event)
        keys = np.array(list(self.memory.keys()))
        d = np.linalg.norm(keys - np.asarray(current_event), axis=1)
        return self.memory[tuple(keys[int(np.argmin(d))])]

    def reset_context(self):
        pass


def run(datasets, condition, **kw):
    bench = PairedAssociateBenchmark(n_trials=2)
    model = RecordingModel()
    out = bench.evaluate(model=model, datasets=datasets, condition=condition,
                         ab_trials=2, phase2_trials=3, epochs=5, **kw)
    return model, out


# ------------------------------------------------------------------- protocol

def test_pairs_are_trained_independently_not_as_a_list(datasets):
    """Concatenating the list would additionally teach b_i -> a_{i+1}."""
    model, _ = run(datasets, "abac")
    assert all(seq.shape[0] == 2 for seq, _ in model.fit_calls), \
        "every fit_sequence call must be a single two-element pair"
    # 2 AB passes + 3 phase-2 passes, each over 3 pairs.
    assert len(model.fit_calls) == (2 + 3) * N_PAIRS


def test_control_trains_disjoint_cues_for_an_equal_budget(datasets):
    """The control must differ from AB/AC in the material, not the quantity."""
    m_abac, _ = run(datasets, "abac")
    m_ctrl, _ = run(datasets, "control")
    assert len(m_abac.fit_calls) == len(m_ctrl.fit_calls)
    assert {e for _, e in m_abac.fit_calls} == {e for _, e in m_ctrl.fit_calls}

    enc = datasets["encoder"]
    # Under the control, phase 2 must never present a cue from list A.
    cue_A_vecs = {tuple(np.round(v, 6)) for v in enc.encode(CUES_A)}
    phase2 = m_ctrl.fit_calls[2 * N_PAIRS:]
    assert not any(tuple(np.round(seq[0], 6)) in cue_A_vecs for seq, _ in phase2)


def test_rejects_an_unknown_condition(datasets):
    with pytest.raises(ValueError, match="unknown condition"):
        PairedAssociateBenchmark().evaluate(
            model=RecordingModel(), datasets=datasets, condition="ABAC")


# ------------------------------------------------------------ response coding

def test_response_buckets_partition_every_probe(datasets):
    _, out = run(datasets, "abac")
    s = out["series"]
    for t in range(len(s["trials"])):
        total = sum(s[f"{b}_rate"][t] for b in RESPONSE_BUCKETS)
        assert total == pytest.approx(1.0), f"buckets do not sum to 1 at trial {t}"


def test_curves_start_before_phase_two(datasets):
    """Index 0 of every curve is the pre-phase-2 measurement."""
    _, out = run(datasets, "abac")
    s, m = out["series"], out["metrics"]
    assert s["trials"] == [0, 1, 2, 3]
    assert s["old_rate"][0] == pytest.approx(m["pa_abac_ab_recall_baseline"])
    assert s["old_rate"][-1] == pytest.approx(m["pa_abac_ab_recall_final"])


def test_shared_cue_overwrites_and_disjoint_cue_does_not(datasets):
    """The contrast the section exists for, on a model whose behaviour is known.

    RecordingModel replaces the target stored against a cue, so AB/AC must
    destroy the old association and the disjoint-cue control must leave it
    untouched. Anything else means the protocol, not the model, is producing
    the effect.
    """
    _, abac = run(datasets, "abac")
    _, ctrl = run(datasets, "control")

    assert abac["metrics"]["pa_abac_ab_acquired"] is True
    assert abac["metrics"]["pa_abac_ab_recall_final"] == pytest.approx(0.0)
    assert abac["metrics"]["pa_abac_ac_recall_final"] == pytest.approx(1.0)
    # A clean replacement, not a collapse.
    assert abac["metrics"]["pa_abac_other_rate_final"] == pytest.approx(0.0)

    assert ctrl["metrics"]["pa_control_ab_recall_final"] == pytest.approx(1.0)
    # Floor check: the C words are never trained under the control.
    assert ctrl["metrics"]["pa_control_ac_recall_final"] == pytest.approx(0.0)


def test_phase_one_is_identical_across_conditions(datasets):
    """The baselines must match, or the two runs were not matched."""
    _, abac = run(datasets, "abac")
    _, ctrl = run(datasets, "control")
    assert (abac["metrics"]["pa_abac_ab_recall_baseline"]
            == pytest.approx(ctrl["metrics"]["pa_control_ab_recall_baseline"]))


def test_trials_to_criterion_counts_phase_two_passes(datasets):
    """One-based over phase-2 trials, ignoring the pre-phase-2 curve point."""
    _, out = run(datasets, "abac")
    assert out["metrics"]["pa_abac_ac_criterion_reached"] is True
    assert out["metrics"]["pa_abac_ac_trials_to_criterion"] == 1.0


def test_trials_to_criterion_is_nan_when_never_reached(datasets):
    """The control never trains C, so its criterion is unreachable by design."""
    _, ctrl = run(datasets, "control")
    assert ctrl["metrics"]["pa_control_ac_criterion_reached"] is False
    assert np.isnan(ctrl["metrics"]["pa_control_ac_trials_to_criterion"])


def test_probe_noise_is_fixed_across_measurements(datasets):
    """A change across phase-2 trials must be the model, not the probe."""
    bench = PairedAssociateBenchmark(n_trials=2)
    model = RecordingModel()
    model.fit_sequence(datasets["encoder"].encode([CUES_A[0], TARGETS_B[0]]))
    a = bench._probe(model, CUES_A, TARGETS_B, TARGETS_C,
                     datasets["encoder"], datasets["decoder"])
    b = bench._probe(model, CUES_A, TARGETS_B, TARGETS_C,
                     datasets["encoder"], datasets["decoder"])
    assert a == b
