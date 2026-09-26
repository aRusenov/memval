"""Tests for the shared continual-retention metric.

Extracted from the removed online_continual pipeline (audit C4/D8), so the
orientation and the summary statistics are pinned here rather than inside any
one suite.
"""
import numpy as np
import pytest

from memval.metrics.retention import (
    retention_matrix,
    retention_summary,
    plot_retention_matrix,
)


def test_matrix_is_lower_triangular_and_ordered():
    """R[j, i]: rows are training stages, columns evaluated tasks."""
    trained = []
    scored = []

    def train(j):
        trained.append(j)

    def score(i, j):
        scored.append((i, j))
        return 1.0

    R = retention_matrix(4, train, score)

    assert trained == [0, 1, 2, 3], "tasks must be trained once each, in order"
    # A task is never scored before it has been seen.
    assert all(i <= j for i, j in scored)
    assert np.all(np.isnan(np.triu(R, k=1)[np.triu_indices(4, k=1)]))
    assert not np.any(np.isnan(np.tril(R)))


def test_summary_detects_forgetting():
    # Each task decays by 0.1 per subsequent task; every task is learned perfectly.
    R = retention_matrix(3, lambda j: None, lambda i, j: 1.0 - 0.1 * (j - i))
    s = retention_summary(R)

    assert s["per_task_learned"] == [1.0, 1.0, 1.0]
    assert s["per_task_final"] == pytest.approx([0.8, 0.9, 1.0])
    assert s["avg_accuracy"] == pytest.approx(0.9)
    # BWT is negative when later tasks damage earlier ones.
    assert s["backward_transfer"] == pytest.approx(-0.15)
    assert s["avg_forgetting"] == pytest.approx(0.15)


def test_summary_detects_positive_backward_transfer():
    """A later task that *helps* an earlier one gives BWT > 0.

    `avg_forgetting` goes negative in the same situation, and that is correct
    rather than a floor violation: following Chaudhry et al., the peak is taken
    over training stages *before* the final one, so a task whose score improves
    at the last stage has a negative drop from that peak. Read the sign — it
    carries the same news as BWT, from the peak rather than the diagonal.
    """
    R = retention_matrix(2, lambda j: None, lambda i, j: 0.5 if i == j else 0.9)
    s = retention_summary(R)
    assert s["backward_transfer"] == pytest.approx(0.4)
    assert s["avg_forgetting"] == pytest.approx(-0.4)


def test_forgetting_uses_peak_not_diagonal():
    """A task that improves before decaying is scored against its peak."""
    R = np.array([[0.5, np.nan, np.nan],
                  [0.9, 0.5, np.nan],
                  [0.4, 0.4, 0.5]])
    s = retention_summary(R)
    # Task 0 peaked at 0.9 (row 1), not at its diagonal 0.5.
    assert s["per_task_forgetting"][0] == pytest.approx(0.5)
    # BWT, which uses the diagonal, disagrees -- that is the point of reporting both.
    assert s["backward_transfer"] == pytest.approx(-0.1)


def test_single_task_chain_is_defined():
    R = retention_matrix(1, lambda j: None, lambda i, j: 0.7)
    s = retention_summary(R)
    assert s["avg_accuracy"] == pytest.approx(0.7)
    assert s["backward_transfer"] == 0.0
    assert s["avg_forgetting"] == 0.0


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        retention_matrix(0, lambda j: None, lambda i, j: 1.0)
    with pytest.raises(ValueError):
        retention_summary(np.zeros((2, 3)))


def test_plot_writes_a_file(tmp_path):
    R = retention_matrix(3, lambda j: None, lambda i, j: 1.0 - 0.1 * (j - i))
    out = tmp_path / "retention.png"
    plot_retention_matrix(R, str(out), title="test", task_labels=["a", "b", "c"])
    assert out.exists() and out.stat().st_size > 0


# ------------------------------------------------- plasticity axis / joint index

def _matrix(fn, T=4):
    return retention_matrix(T, lambda j: None, fn)


def test_frozen_model_is_caught_by_the_diagonal():
    """The gaming case the stability statistics alone cannot see.

    An arm that learns task 0 and then stops has a *perfect* stability profile:
    nothing it holds ever decays, so avg_forgetting and BWT are both 0. Only the
    diagonal shows that three of the four tasks never went in at all.
    """
    R = _matrix(lambda i, j: 1.0 if i == 0 else 0.0)
    s = retention_summary(R)

    # Stability: spotless, and meaningless.
    assert s["avg_forgetting"] == pytest.approx(0.0)
    assert s["backward_transfer"] == pytest.approx(0.0)
    # Plasticity: caught. R[0,0]=1, every later diagonal entry 0.
    assert s["avg_learning"] == pytest.approx(0.25)
    assert s["intransigence"] == pytest.approx(1.0)
    assert s["learning_slope"] < 0
    # The joint index refuses it despite the perfect forgetting score.
    assert s["stability_plasticity_index"] < 0.5


def test_overwriter_and_frozen_score_alike_and_the_ideal_does_not():
    """The two failure modes are symmetric, and neither is a pass.

    An arithmetic mean of the two coordinates would hand both of these 0.5 and
    call them half-good; the harmonic mean is what keeps a single maxed axis
    from carrying an arm.
    """
    frozen = retention_summary(_matrix(lambda i, j: 1.0 if i == 0 else 0.0))
    overwriter = retention_summary(_matrix(lambda i, j: 1.0 if i == j else 0.0))
    ideal = retention_summary(_matrix(lambda i, j: 1.0))

    assert frozen["stability_plasticity_index"] == pytest.approx(
        overwriter["stability_plasticity_index"])
    assert ideal["stability_plasticity_index"] == pytest.approx(1.0)
    assert frozen["stability_plasticity_index"] < 0.5

    # ...but they are told apart by *which* coordinate collapsed.
    assert frozen["avg_learning"] < overwriter["avg_learning"]
    assert frozen["retention_ratio"] > overwriter["retention_ratio"]


def test_retention_ratio_gives_no_credit_for_unlearned_material():
    """avg_forgetting rewards never learning; retention_ratio does not."""
    frozen = retention_summary(_matrix(lambda i, j: 1.0 if i == 0 else 0.0))
    # Perfect on the forgetting column...
    assert frozen["avg_forgetting"] == pytest.approx(0.0)
    # ...because only one task was ever acquired, and that one did survive.
    assert frozen["retention_ratio"] == pytest.approx(1.0)
    assert frozen["avg_learning"] == pytest.approx(0.25)
    # Which is exactly why the ratio must be read with LA, not instead of it.


def test_forward_transfer_gives_negative_intransigence():
    """Acquisition that *improves* with load reads negative, not clipped."""
    R = _matrix(lambda i, j: 0.5 + 0.1 * i if i == j else 0.5)
    s = retention_summary(R)
    assert s["intransigence"] < 0
    assert s["learning_slope"] > 0


def test_forgetting_by_gap_is_the_decay_gradient():
    """Retained fraction at each interposition gap, indexed g-1."""
    # Every task keeps 90% per interposed task, compounding.
    R = _matrix(lambda i, j: 0.9 ** (j - i))
    s = retention_summary(R)
    assert len(s["forgetting_by_gap"]) == 3
    assert s["forgetting_by_gap"] == pytest.approx([0.9, 0.81, 0.729])


def test_forgetting_by_gap_skips_tasks_that_were_never_learned():
    """A zero diagonal entry has no trace to decay, so it cannot be divided by."""
    R = np.array([[0.0, np.nan], [0.0, 1.0]])
    s = retention_summary(R)
    # Task 0 was never learned -> the single gap-1 pair is dropped -> nan.
    assert np.isnan(s["forgetting_by_gap"][0])


def test_chance_correction_applies_to_the_joint_index_only():
    """A chance floor inflates ratios; the differences must stay raw."""
    R = _matrix(lambda i, j: 0.5)
    raw = retention_summary(R, chance_level=0.0)
    corrected = retention_summary(R, chance_level=0.5)

    # Differences are untouched.
    assert raw["avg_learning"] == corrected["avg_learning"] == pytest.approx(0.5)
    assert raw["avg_forgetting"] == corrected["avg_forgetting"]
    # At chance, the index is 0: the arm demonstrated nothing.
    assert corrected["stability_plasticity_index"] == pytest.approx(0.0)
    assert raw["stability_plasticity_index"] > 0.0


def test_nothing_learned_gives_zero_not_nan():
    """A chain at the floor must lose the comparison, not drop out of it."""
    s = retention_summary(_matrix(lambda i, j: 0.0), chance_level=0.0)
    assert s["retention_ratio"] == 0.0
    assert s["stability_plasticity_index"] == 0.0


def test_single_task_chain_defines_the_plasticity_columns_too():
    R = retention_matrix(1, lambda j: None, lambda i, j: 0.7)
    s = retention_summary(R)
    assert s["intransigence"] == 0.0
    assert s["learning_slope"] == 0.0
    assert s["forgetting_by_gap"] == []


def test_plane_plot_writes_a_file(tmp_path):
    from memval.metrics.retention import plot_stability_plasticity_plane
    pts = [
        {"label": "frozen", "summary": retention_summary(
            _matrix(lambda i, j: 1.0 if i == 0 else 0.0))},
        {"label": "overwriter", "summary": retention_summary(
            _matrix(lambda i, j: 1.0 if i == j else 0.0))},
    ]
    out = tmp_path / "plane.png"
    plot_stability_plasticity_plane(pts, str(out))
    assert out.exists() and out.stat().st_size > 0
