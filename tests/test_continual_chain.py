"""Tests for the continual-chain section.

The protocol is what is pinned here — training order, no weight reset, the
selective-retention difference-in-differences and its guards. The matrix and
its statistics belong to ``memval.metrics.retention`` and are pinned in
``test_retention_metric.py``.
"""
import numpy as np
import pytest

from memval.benchmarks.continual_chain import (
    ContinualChainBenchmark,
    consolidate_if_supported,
)


class StubModel:
    """Records the calls the protocol makes on it. Learns nothing."""

    def __init__(self):
        self.fit_calls = []
        self.resets = 0
        self.consolidations = 0

    def fit_sequence(self, sequence_data, epochs=None, **kw):
        self.fit_calls.append((len(sequence_data), epochs))

    def reset_context(self):
        self.resets += 1


class ConsolidatingStub(StubModel):
    def consolidate(self):
        self.consolidations += 1


class SequenceAwareStub(StubModel):
    def consolidate(self, sequence_data, context_data=None):
        self.consolidations += 1
        assert context_data is not None and len(context_data) == len(sequence_data)


def make_datasets(scores, n_tasks=4, seq_len=3):
    """A chain of ``n_tasks`` toy tasks whose score is dictated by ``scores``.

    ``scores`` is called with the task index and the number of training stages
    completed so far, so a test can script any retention profile it likes.
    """
    names = [f"task{i}" for i in range(n_tasks)]
    state = {"stage": 0}

    def score_fn(i):
        return scores(i, state["stage"])

    return names, state, {
        "sequences": {n: [f"{n}_w{k}" for k in range(seq_len)] for n in names},
        "embeddings": {n: np.zeros((seq_len, 4)) for n in names},
        "score_fn": score_fn,
    }


def run(bench, scores, model=None, n_tasks=4, **kw):
    names, state, datasets = make_datasets(scores, n_tasks=n_tasks)
    model = model or StubModel()

    # Count stages by wrapping fit_sequence: score_fn needs to know how far the
    # chain has got, which is exactly what a retention matrix varies.
    inner = model.fit_sequence

    def counting_fit(sequence_data, epochs=None, **kwargs):
        inner(sequence_data, epochs=epochs, **kwargs)
        state["stage"] += 1

    model.fit_sequence = counting_fit
    return model, bench.evaluate(model=model, datasets=datasets, **kw)


# ------------------------------------------------------------------- protocol

def test_each_task_is_trained_once_in_order_without_reset():
    # criterion=None pins a fixed budget, which is what makes "one fit_sequence
    # per task" a meaningful assertion. The criterion-referenced default trains
    # in steps and is covered separately below.
    model, out = run(ContinualChainBenchmark(rehearse=False, criterion=None),
                     lambda i, stage: 1.0, epochs=7)
    # One fit_sequence per task, at the requested budget, and no more.
    assert len(model.fit_calls) == 4
    assert {e for _, e in model.fit_calls} == {7}
    assert out["metrics"]["chain_n_tasks"] == 4
    # Nothing in the protocol may reset weights; only score_fn touches context,
    # and this stub's score_fn does not.
    assert model.resets == 0


def test_consolidation_hook_is_called_with_either_signature():
    for stub in (ConsolidatingStub(), SequenceAwareStub()):
        m, _ = run(ContinualChainBenchmark(rehearse=False, criterion=None),
                   lambda i, stage: 1.0, model=stub)
        assert stub.consolidations == 4


def test_evaluation_noise_is_fixed_per_task_across_stages():
    """A change down a column must be the model moving, not the probe."""
    draws = {}

    def scores(i, stage):
        draws.setdefault(i, []).append(float(np.random.rand()))
        return 1.0

    run(ContinualChainBenchmark(rehearse=False, criterion=None), scores)
    for i, vals in draws.items():
        assert len(set(vals)) == 1, f"task {i} was probed with different noise"
    # ...and different tasks get *different* noise, so the seeding is per task.
    assert len({v[0] for v in draws.values()}) == len(draws)


def test_rejects_a_chain_too_short_to_be_continual():
    with pytest.raises(ValueError, match="at least 2 tasks"):
        run(ContinualChainBenchmark(), lambda i, stage: 1.0, n_tasks=1)


# ------------------------------------------------------- plasticity read-outs

def test_frozen_arm_is_caught_by_the_chain_metrics():
    """Learns task 0, then nothing. Perfect stability, no plasticity."""
    _, out = run(ContinualChainBenchmark(rehearse=False),
                 lambda i, stage: 1.0 if i == 0 else 0.0)
    m = out["metrics"]
    assert m["chain_avg_forgetting"] == pytest.approx(0.0)
    assert m["chain_intransigence"] == pytest.approx(1.0)
    assert m["chain_stability_plasticity_index"] < 0.5


def test_acquired_guard_fires_when_the_diagonal_is_at_chance():
    _, out = run(ContinualChainBenchmark(rehearse=False),
                 lambda i, stage: 0.02, chance_level=1 / 30)
    assert out["metrics"]["chain_acquired"] is False

    _, out = run(ContinualChainBenchmark(rehearse=False),
                 lambda i, stage: 0.9, chance_level=1 / 30)
    assert out["metrics"]["chain_acquired"] is True


def test_series_carries_the_matrix_and_the_load_curve():
    _, out = run(ContinualChainBenchmark(rehearse=False), lambda i, stage: 1.0)
    s = out["series"]
    assert np.array(s["retention_matrix"]).shape == (4, 4)
    assert s["gaps"] == [1, 2, 3]
    assert len(s["forgetting_by_gap"]) == 3
    assert len(s["task_labels"]) == 4


# ------------------------------------------------------- selective retention

def test_rehearsed_set_is_the_oldest_half():
    """Deliberate: most decayed (headroom for a gain) and recency-disadvantaged."""
    b = ContinualChainBenchmark()
    assert b._rehearsed_indices(6) == [0, 1, 2]
    assert b._rehearsed_indices(4) == [0, 1]
    # Both buckets stay non-empty however the request is clamped.
    assert b._rehearsed_indices(2) == [0]
    assert ContinualChainBenchmark(n_rehearsed=99)._rehearsed_indices(4) == [0, 1, 2]
    assert ContinualChainBenchmark(n_rehearsed=0)._rehearsed_indices(4) == [0]


def test_selectivity_is_a_difference_in_differences():
    """Rehearsed tasks recover, the rest drift down: the intended profile."""
    # `stage` counts completed fit_sequence calls, so the chain's own final row
    # is scored at stage 4 and the rehearsal presentations push it to 6.
    # Before rehearsal every task sits at 0.2; afterwards the rehearsed half is
    # restored to 0.9 and the rest falls to 0.1.
    def scores(i, stage):
        if stage <= 4:
            return 0.2
        return 0.9 if i in (0, 1) else 0.1

    _, out = run(ContinualChainBenchmark(rehearse=True, criterion=None), scores)
    m = out["metrics"]
    assert m["select_rehearsed_gain"] == pytest.approx(0.7)
    assert m["select_unrehearsed_drift"] == pytest.approx(-0.1)
    assert m["select_selectivity"] == pytest.approx(0.8)
    assert m["select_n_rehearsed"] == 2


def test_a_uniform_lift_is_not_selectivity():
    """Everything improving equally is rehearsal leaking, not selection."""
    def scores(i, stage):
        return 0.2 if stage <= 4 else 0.9

    _, out = run(ContinualChainBenchmark(rehearse=True, criterion=None), scores)
    assert out["metrics"]["select_selectivity"] == pytest.approx(0.0)
    # ...and it is distinguishable from "nothing happened" by the component.
    assert out["metrics"]["select_rehearsed_gain"] == pytest.approx(0.7)


def test_pressure_guard_marks_an_unsaturated_chain_uninterpretable():
    """No forgetting means nothing had to be discarded: the question was not posed."""
    _, out = run(ContinualChainBenchmark(rehearse=True), lambda i, stage: 1.0)
    m = out["metrics"]
    assert m["select_pressure"] == pytest.approx(0.0)
    assert m["select_under_pressure"] is False
    assert m["select_rehearsal_effective"] is False
    # The selectivity value still exists -- it is guarded, not suppressed.
    assert m["select_selectivity"] == pytest.approx(0.0)


def test_rehearsal_trains_only_the_relevant_subset():
    model, out = run(ContinualChainBenchmark(rehearse=True),
                     lambda i, stage: 1.0 if stage <= 4 else 0.5)
    # 4 chain stages + 2 rehearsal presentations (the oldest half of 4).
    assert len(model.fit_calls) == 6
    assert out["series"]["selective_retention"]["rehearsed"] == [0, 1]
    assert out["series"]["selective_retention"]["unrehearsed"] == [2, 3]


def test_rehearsal_can_be_switched_off():
    _, out = run(ContinualChainBenchmark(rehearse=False), lambda i, stage: 1.0)
    assert not any(k.startswith("select_") for k in out["metrics"])
    assert "selective_retention" not in out["series"]


def test_consolidate_if_supported_skips_arms_without_the_hook():
    plain = StubModel()
    consolidate_if_supported(plain, np.zeros((3, 4)))   # must not raise
    assert not hasattr(plain, "consolidations") or plain.consolidations == 0


# ------------------------------------------------- criterion-referenced exposure

def test_criterion_mode_trains_until_the_task_is_acquired():
    """Exposure is a result: each task trains until score_fn reaches criterion."""
    need = {0: 1, 1: 3, 2: 2, 3: 5}
    seen = {i: 0 for i in need}

    def scores(i, stage):
        seen[i] += 1
        return 1.0 if seen[i] >= need[i] else 0.0

    model, out = run(ContinualChainBenchmark(rehearse=False, criterion=0.95,
                                             max_epochs=32),
                     scores, n_tasks=4)
    assert out["metrics"]["chain_exposure_mode"] == "criterion"
    assert out["metrics"]["chain_criterion_reached"] is True
    # More epochs on the tasks that needed more, and never a fixed budget.
    per_task = [e["epochs"] for e in out["series"]["exposures"]]
    assert per_task[3] > per_task[0], "a slower task must cost more exposure"


def test_criterion_mode_censors_instead_of_looping_forever():
    _, out = run(ContinualChainBenchmark(rehearse=False, criterion=0.95,
                                         max_epochs=8),
                 lambda i, stage: 0.0, n_tasks=3)
    assert out["metrics"]["chain_criterion_reached"] is False
    assert all(e["epochs"] == 8 for e in out["series"]["exposures"]), \
        "the budget is reported as a lower bound"


def test_fixed_mode_still_available_for_reproducing_old_numbers():
    _, out = run(ContinualChainBenchmark(rehearse=False, criterion=None),
                 lambda i, stage: 1.0, epochs=7, n_tasks=3)
    assert out["metrics"]["chain_exposure_mode"] == "fixed"
    assert all(e["epochs"] == 7 for e in out["series"]["exposures"])
