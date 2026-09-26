"""Criterion-referenced exposure helper."""
import pytest

from memval.benchmarks.exposure import epochs_to_criterion, _ladder


class Toy:
    """Reaches `score = epochs / target` — crosses 1.0 at exactly `target`."""
    def __init__(self, target):
        self.target, self.trained = target, 0


def _harness(target, **kw):
    fits = {"n": 0}

    def make():
        return Toy(target)

    def fit(m, epochs):
        fits["n"] += 1
        m.trained += epochs

    def score(m):
        return m.trained / m.target

    return epochs_to_criterion(make, fit, score, criterion=1.0, **kw), fits


def test_ladder_is_geometric_and_ends_at_the_budget():
    assert _ladder(16) == [1, 2, 4, 8, 16]
    assert _ladder(20)[-1] == 20


@pytest.mark.parametrize("target", [1, 2, 3, 7, 13, 20, 32])
def test_finds_the_exact_minimum_not_the_checkpoint(target):
    res, _ = _harness(target, max_epochs=64)
    assert res["reached"] is True
    assert res["epochs"] == target, "refinement must beat the doubling grid"


def test_refine_off_returns_the_bracketing_checkpoint():
    res, _ = _harness(13, max_epochs=64, refine=False)
    assert res["reached"] is True
    assert res["epochs"] == 16              # the checkpoint, not the true 13


def test_censors_rather_than_lying_when_unreachable():
    res, _ = _harness(1000, max_epochs=32)
    assert res["reached"] is False
    assert res["epochs"] == 32, "the budget is reported as a lower bound"
    assert res["score"] < 1.0


def test_incremental_matches_from_scratch():
    a, fits_a = _harness(13, max_epochs=64, incremental=False)
    b, fits_b = _harness(13, max_epochs=64, incremental=True)
    assert a["epochs"] == b["epochs"] == 13


def test_returns_a_model_trained_to_the_reported_exposure():
    res, _ = _harness(7, max_epochs=64)
    assert res["model"].trained == 7, "caller must not have to retrain"


def test_history_records_every_checkpoint_evaluated():
    res, _ = _harness(7, max_epochs=64)
    assert [h["epochs"] for h in res["history"]][:4] == [1, 2, 4, 8]
    assert all("score" in h for h in res["history"])


# ---------------------------------------------------------------------------
# Fixed budgets from the registry (2026-09-05): control sections only
# ---------------------------------------------------------------------------

def test_control_sections_are_disjoint_from_result_sections():
    from memval.benchmarks.exposure import CONTROL_SECTIONS
    result_sections = {"semantic_similarity", "schema_consistency", "presentation_duration",
                       "continual_chain", "paired_associate", "interval_retention", "interval_generation"}
    assert not (CONTROL_SECTIONS & result_sections)


def test_registry_budget_applies_to_control_sections_only():
    from memval.benchmarks.exposure import resolve_exposure
    def arg(_ba, _s, _k, default=None): return default
    ctl = resolve_exposure(None, "sequence_length", arg, 300, model_fixed_epochs=86)
    assert ctl["mode"] == "fixed" and ctl["epochs"] == 86 and ctl["source"] == "registry"
    res = resolve_exposure(None, "semantic_similarity", arg, 300, model_fixed_epochs=86)
    assert res["mode"] == "criterion" and res["source"] is None


def test_exposure_precedence_cli_section_over_global_over_registry():
    from memval.benchmarks.exposure import resolve_exposure
    def arg(ba, s, k, default=None): return (ba or {}).get(s, {}).get(k, default)
    r = resolve_exposure({"sequence_length": {"epochs": 7}}, "sequence_length", arg, 300,
                         global_epochs=50, model_fixed_epochs=86)
    assert (r["epochs"], r["source"]) == (7, "cli:section")
    r = resolve_exposure(None, "sequence_length", arg, 300, global_epochs=50, model_fixed_epochs=86)
    assert (r["epochs"], r["source"]) == (50, "cli:global")


def test_force_criterion_keeps_the_ladder_for_a_result_use_of_a_control_section():
    from memval.benchmarks.exposure import resolve_exposure
    def arg(_ba, _s, _k, default=None): return default
    r = resolve_exposure(None, "multiple_sequences", arg, 300, model_fixed_epochs=86, force_criterion=True)
    assert r["mode"] == "criterion"
