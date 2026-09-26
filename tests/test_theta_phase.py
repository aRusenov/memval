"""Theta-phase arm: interface conformance, the delta-rule reduction, the
controlled pair against AHN, the paper's two-dimensional phase surface, and the
fornix-lesion control."""

import numpy as np
import pytest

from memval.models.baselines import AsymmetricHopfieldNetwork, ThetaPhaseSequenceNetwork

DIM = 16
LR = 0.1


def _seq(seed=1, n=6):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n, DIM))
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def make(**over):
    kw = dict(n_features=DIM, learning_rate=LR)
    kw.update(over)
    return ThetaPhaseSequenceNetwork(**kw)


# ----------------------------------------------------------------- interface
def test_predict_next_shape_and_recall_shape():
    m = make()
    seq = _seq()
    m.fit_sequence(seq)
    assert m.predict_next(seq[0]).shape == (DIM,)
    assert m.recall(seq[0], length=4).shape == (4, DIM)
    # 2D prompt: seeds from the last row
    assert m.recall(seq[:3], length=2).shape == (2, DIM)


def test_predict_next_accepts_and_ignores_context():
    m = make()
    m.fit_sequence(_seq())
    x = _seq()[0]
    assert np.allclose(m.predict_next(x), m.predict_next(x, current_context=np.array([1.0])))


def test_short_sequence_is_a_noop_not_a_crash():
    m = make()
    before = m.W.copy()
    m.fit_sequence(np.zeros((1, DIM)))
    assert np.array_equal(m.W, before)


def test_predict_next_is_pure():
    """`measure_recall_associative` probes positions independently, so
    predict_next must not leak state between probes."""
    m = make()
    seq = _seq()
    m.fit_sequence(seq)
    m.fit_event(seq[0])
    prev, w = m._prev_event.copy(), m.W.copy()
    a = m.predict_next(seq[2])
    _ = m.predict_next(seq[4])
    b = m.predict_next(seq[2])
    assert np.allclose(a, b)
    assert np.array_equal(m._prev_event, prev)
    assert np.array_equal(m.W, w)


# ------------------------------------------------------------- the reduction
def test_default_phases_give_the_papers_optimum():
    c = make().phase_coefficients()
    assert c["encode"] == pytest.approx(1.0)
    assert c["retrieve"] == pytest.approx(-1.0)
    assert c["residual"] == pytest.approx(0.0, abs=1e-12)


def test_cycle_integral_reduces_to_the_delta_rule():
    """One theta cycle == (x_next - W x_cur) x_cur^T at the paper's config."""
    m = make()
    seq = _seq()
    m.fit_sequence(seq)  # get W away from zero so the retrieval term bites
    x, y = seq[0], seq[1]
    expected = np.outer(y - m.W @ x, x)
    assert np.allclose(m._cycle_delta(x, y), expected)


def test_theta_at_optimum_matches_ahn_weights():
    """The controlled pair: same step, error derived two different ways."""
    theta, ahn = make(), AsymmetricHopfieldNetwork(n_features=DIM, learning_rate=LR)
    seq = _seq()
    theta.fit_sequence(seq)
    ahn.fit_sequence(seq)
    assert np.allclose(theta.W, ahn.W)


def test_rectify_output_is_what_matches_ahn_on_predictions():
    """Identical weights are not identical predictions: AHN rectifies its
    readout and eq 2.4 does not, so the pair is only exact with the flag."""
    seq = _seq()
    ahn = AsymmetricHopfieldNetwork(n_features=DIM, learning_rate=LR)
    linear, rectified = make(), make(rectify_output=True)
    for m in (ahn, linear, rectified):
        m.fit_sequence(seq)
    assert np.allclose(rectified.predict_next(seq[0]), ahn.predict_next(seq[0]))
    assert not np.allclose(linear.predict_next(seq[0]), ahn.predict_next(seq[0]))
    assert np.all(rectified.predict_next(seq[0]) >= 0.0)


def test_online_and_batch_schedules_agree_for_one_epoch():
    batch, online = make(), make()
    seq = _seq()
    batch.fit_sequence(seq)
    for ev in seq:
        online.fit_event(ev)
    assert np.allclose(batch.W, online.W)


def test_event_boundary_blocks_the_cross_seam_transition():
    seamed, joined = make(), make()
    a, b = _seq(seed=2, n=3), _seq(seed=3, n=3)
    for ev in a:
        seamed.fit_event(ev)
    seamed.on_event_boundary()
    for ev in b:
        seamed.fit_event(ev)
    for ev in np.vstack([a, b]):
        joined.fit_event(ev)
    assert not np.allclose(seamed.W, joined.W)


# ------------------------------------------------- the paper's phase surface
def test_phases_are_independent_so_the_surface_is_not_degenerate():
    """Locking CA3 to the antiphase of EC would force residual==0 everywhere and
    make the paper's claim untestable. Moving phase_ca3 alone must break it."""
    residuals = [
        make(phase_ca3=p).phase_coefficients()["residual"]
        for p in (np.pi, 0.75 * np.pi, 0.5 * np.pi, 0.0)
    ]
    assert residuals[0] == pytest.approx(0.0, abs=1e-12)
    assert all(abs(r) > 0.1 for r in residuals[1:])
    # in-phase CA3 and EC means retrieval reinforces instead of subtracting
    assert make(phase_ca3=0.0).phase_coefficients()["retrieve"] > 0


@pytest.mark.parametrize(
    "d_ec, d_ca3", [(0.0, np.pi), (0.0, 0.5 * np.pi), (0.5 * np.pi, np.pi)]
)
def test_coefficients_match_the_closed_form(d_ec, d_ca3):
    """coef = (X/4) cos(phi_LTP - phi_gate), normalised to 1/4."""
    x = 0.7
    m = make(phase_ec=-d_ec, phase_ca3=-d_ca3, phase_ltp=0.0, modulation_depth=x)
    c = m.phase_coefficients()
    assert c["encode"] == pytest.approx(x * np.cos(d_ec), abs=1e-9)
    assert c["retrieve"] == pytest.approx(x * np.cos(d_ca3), abs=1e-9)


def test_M_reproduces_figure_4_optimum():
    """Eq 2.14 is maximal at phi_LTP-phi_EC = 0 and phi_LTP-phi_CA3 = pi."""
    best = make().M()
    grid = [
        make(phase_ec=-a, phase_ca3=-b).M()
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)
        for b in np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ]
    assert best == pytest.approx(max(grid))
    assert best == pytest.approx(np.pi - 1.0)  # X = K = 1


# ------------------------------------------------- the fornix-lesion control
def test_zero_modulation_depth_abolishes_learning():
    """X=0: gates go constant, the zero-mean LTP profile integrates them away,
    so nothing is learned and a prior association would persist unopposed."""
    m = make(modulation_depth=0.0)
    seq = _seq()
    for _ in range(10):
        m.fit_sequence(seq)
    assert np.allclose(m.W, 0.0)
    assert m.phase_coefficients()["encode"] == pytest.approx(0.0, abs=1e-12)


def test_modulation_depth_scales_the_update_linearly():
    seq = _seq()
    full, half = make(), make(modulation_depth=0.5)
    full.fit_sequence(seq[:2])
    half.fit_sequence(seq[:2])
    assert np.allclose(half.W, 0.5 * full.W)


# ------------------------------------------------------------- unlearning
def test_absent_target_gives_pure_depotentiation():
    """The paper's error trial is a_EC = 0. What remains must shrink the
    retrieved association, in proportion to how strongly it is retrieved."""
    m = make()
    x, y = _seq()[0], _seq()[1]
    for _ in range(30):
        m.fit_sequence(np.stack([x, y]))
    retrieved_before = np.linalg.norm(m.predict_next(x))
    for _ in range(5):
        m.W += m.learning_rate * m._cycle_delta(x, np.zeros(DIM))
    retrieved_after = np.linalg.norm(m.predict_next(x))
    assert retrieved_after < retrieved_before


# ------------------------------------------------- standalone reversal task
# Hasselmo, Bodelon & Wyble 2002 section 2.3: L -> F_L, then L -> 0 (error
# trials, a_EC = 0), then R -> F_R. Place and food occupy orthogonal
# subspaces, and L is orthogonal to R, so learning R -> F_R cannot disturb
# L -> F_L: extinction is the only route to removing the stale association.
RDIM = 8
_L, _R, _FL, _FR = (np.eye(RDIM)[i] for i in range(4))
N_ACQ, N_ERR, N_REV = 20, 10, 20


def _rmodel(**over):
    kw = dict(n_features=RDIM, learning_rate=0.1)
    kw.update(over)
    return ThetaPhaseSequenceNetwork(**kw)


def _acquire(m, n=N_ACQ):
    for _ in range(n):
        m.fit_sequence(np.stack([_L, _FL]))
    return float(m.predict_next(_L) @ _FL)


def _error_trials(m, n=N_ERR):
    for _ in range(n):
        m.fit_sequence(np.stack([_L, np.zeros(RDIM)]))
    return float(m.predict_next(_L) @ _FL)


def _reverse(m, n=N_REV):
    for _ in range(n):
        m.fit_sequence(np.stack([_R, _FR]))
    return float(m.predict_next(_R) @ _FR)


def test_reversal_acquisition_follows_the_delta_rule_fixed_point():
    """Cued recall converges to 1-(1-lr)^n, not to 1."""
    m = _rmodel()
    assert _acquire(m) == pytest.approx(1.0 - 0.9 ** N_ACQ, abs=1e-9)


def test_error_trials_extinguish_geometrically():
    """a_EC = 0 decays the stale association by (1-lr) per trial -- extinction
    rate and learning rate are the same constant."""
    m = _rmodel()
    after_acq = _acquire(m)
    after_err = _error_trials(m)
    assert after_err == pytest.approx(after_acq * 0.9 ** N_ERR, abs=1e-9)
    assert after_err < after_acq


def test_reversal_completes_and_flips_the_retrieval_choice():
    """M = correct - stale must end positive: the new association dominates."""
    m = _rmodel()
    _acquire(m)
    _error_trials(m)
    correct = _reverse(m)
    stale = float(m.predict_next(_L) @ _FL)
    assert correct > stale
    assert correct - stale > 0.25


def test_orthogonal_arms_mean_reversal_alone_cannot_extinguish():
    """Without error trials the stale association is untouched by learning the
    new one -- which is why the paper's error trials carry the argument."""
    m = _rmodel()
    after_acq = _acquire(m)
    _reverse(m)
    assert float(m.predict_next(_L) @ _FL) == pytest.approx(after_acq, abs=1e-12)


def test_lesion_after_acquisition_blocks_extinction_and_reacquisition():
    """X=0 from the first error trial: the equations' lesion freezes the model."""
    m = _rmodel()
    after_acq = _acquire(m)
    m.modulation_depth = 0.0
    m._build_phase_profiles()
    assert _error_trials(m) == pytest.approx(after_acq, abs=1e-12)
    assert _reverse(m) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "frac, expected",
    [(1.0, "extinguish"), (0.5, "untouched"), (0.0, "reinforce")],
)
def test_detuning_ca3_phase_breaks_extinction(frac, expected):
    """The paper's window: the old association decays only for
    pi/2 < phi_LTP - phi_CA3 < 3pi/2. Outside it, error trials stop working
    and then actively reinforce what they were meant to erase."""
    m = _rmodel(phase_ca3=frac * np.pi)
    after_acq = _acquire(m)
    ratio = _error_trials(m) / after_acq
    if expected == "extinguish":
        assert ratio < 0.5
    elif expected == "untouched":
        assert ratio == pytest.approx(1.0, abs=1e-12)
    else:
        assert ratio > 2.0
