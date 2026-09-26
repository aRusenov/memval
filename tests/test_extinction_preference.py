"""The cheese-odour T-maze and the adaptive extinction-preference protocol."""
import numpy as np
import pytest

from memval.benchmarks.extinction_preference import ExtinctionPreferenceBenchmark, other
from memval.generators.t_maze_extinction import TMazeExtinctionGenerator as G


@pytest.fixture
def task():
    return G(seed=3).generate(seed=3, n_place_cells=100)


def test_stem_is_identical_across_arms_and_variants(task):
    se = task["stem_end"]
    L = G.trajectory(task, "left", 1.0, "on")
    R = G.trajectory(task, "right", 0.3, "noise", np.random.default_rng(0))
    assert np.array_equal(L[:se], R[:se])


def test_odour_is_ambient_in_stem_and_ramps_to_goal(task):
    p = task["odour_profile"]; se, gs = task["stem_end"], task["goal_start"]
    assert np.allclose(p[:se], p[0])
    assert np.all(np.diff(p[se:gs + 1]) >= 0) and p[gs] == pytest.approx(1.0)
    assert p[0] < p[gs]


def test_extinction_fades_arm_odour_only_and_noises_the_reward(task):
    n_pc, n_od, se, gs = task["n_place_cells"], task["n_odour_dims"], task["stem_end"], task["goal_start"]
    X = G.trajectory(task, "left", 0.5, "noise", np.random.default_rng(1))
    Y = G.trajectory(task, "left", 1.0, "on")
    od_x, od_y = X[:, n_pc:n_pc + n_od] @ task["odour_vec"], Y[:, n_pc:n_pc + n_od] @ task["odour_vec"]
    assert np.allclose(od_x[:se], od_y[:se])
    assert np.allclose(od_x[se:], 0.5 * od_y[se:])
    rew = X[:, n_pc + n_od:]
    assert np.allclose(rew[:gs], 0.0)
    # Noise energy matches the reward it replaces: expected row norm = gain.
    norms = np.linalg.norm(rew[gs:], axis=1)
    assert 0.5 < norms.mean() < 1.6
    assert abs(rew[gs:] @ task["reward_vec_left"]).mean() < 0.6      # carries no goal identity
    assert np.allclose(Y[gs:, n_pc + n_od:] @ task["reward_vec_left"], task["reward_gain"])


def test_noise_norm_matches_reward_norm_in_expectation(task):
    n_pc, n_od, gs = task["n_place_cells"], task["n_odour_dims"], task["goal_start"]
    rng = np.random.default_rng(7)
    norms = [np.linalg.norm(G.trajectory(task, "right", 1.0, "noise", rng)[gs:, n_pc + n_od:], axis=1).mean()
             for _ in range(50)]
    assert np.mean(norms) == pytest.approx(task["reward_gain"], abs=0.1)


class FixedArm:
    """Runs a fixed arm perfectly (nearest-step lookup), ignores training,
    except that after `flip_after` extinction fits it switches arms."""

    def __init__(self, task, arm, flip_after=None):
        self.task, self.arm, self.flip_after, self.n_ext = task, arm, flip_after, 0

    def reset_context(self):
        pass

    def fit_sequence(self, X, epochs=None, **kw):
        n_pc, n_od, gs = self.task["n_place_cells"], self.task["n_odour_dims"], self.task["goal_start"]
        vec = self.task["reward_vec_left"] if X[gs, n_pc] == self.task["place"]["left"][gs, 0] else None
        rew = X[gs:, n_pc + n_od:]
        if not (np.allclose(rew @ self.task["reward_vec_left"], 1.0) or
                np.allclose(rew @ self.task["reward_vec_right"], 1.0)):
            self.n_ext += 1
            if self.flip_after is not None and self.n_ext == self.flip_after:   # flip once
                self.arm = other(self.arm)

    def predict_next(self, x, current_context=None):
        Xa = G.trajectory(self.task, self.arm, 1.0, "on")
        n_pc = self.task["n_place_cells"]
        t = int(np.argmin(np.linalg.norm(Xa[:, :n_pc] - x[:n_pc], axis=1)))
        return Xa[min(t + 1, len(Xa) - 1)]


def test_protocol_extinguishes_the_preferred_arm_and_scores_zero_for_an_unmoved_model(task):
    m = FixedArm(task, "left")
    out = ExtinctionPreferenceBenchmark(acquisition_pairs=2, extinction_trials=3, nudge_points=5,
                                        measure_savings=False).evaluate(m, task, epochs=1, end_on="right")
    mt = out["metrics"]
    assert mt["acquisition_preferred_arm"] == "left"
    assert mt["extinction_1_arm"] == "left"
    assert mt["extinction_1_delta_pref_ext"] == pytest.approx(0.0)
    assert mt["extinction_1_chose_ext_start"] == 1.0 and mt["extinction_1_chose_ext_end"] == 1.0
    assert mt["extinction_2_arm"] == "left"            # still preferred -> adaptive picks it again
    assert mt["extinction_1_ext_reward_pred_end"] == pytest.approx(1.0)   # the stub keeps predicting cheese


def test_a_model_that_flees_extinction_scores_negative_and_stage_two_follows_the_new_preference(task):
    m = FixedArm(task, "right", flip_after=2)
    out = ExtinctionPreferenceBenchmark(acquisition_pairs=2, extinction_trials=3, nudge_points=5,
                                        measure_savings=False).evaluate(m, task, epochs=1, end_on="left")
    mt = out["metrics"]
    assert mt["extinction_1_arm"] == "right"
    assert mt["extinction_1_delta_pref_ext"] < 0                 # fled the extinguished arm
    assert mt["extinction_1_chose_ext_end"] == 0.0
    assert mt["extinction_2_arm"] == "left"                      # adaptive: the NEW preferred arm


def test_second_stage_other_targets_the_unextinguished_arm(task):
    m = FixedArm(task, "left")
    out = ExtinctionPreferenceBenchmark(acquisition_pairs=1, extinction_trials=2, nudge_points=3,
                                        second_stage="other", measure_savings=False).evaluate(m, task, end_on="right")
    assert out["metrics"]["extinction_2_arm"] == "right"


def test_psychometric_midpoint_saturates_for_an_immovable_model(task):
    m = FixedArm(task, "left")
    b = ExtinctionPreferenceBenchmark(nudge_points=5)
    _, prefs, mid = b._psychometric(m, task)
    assert np.all(prefs < 0) and mid == b.nudge_max


# --------------------------------------------------------------------------
# The choice-point margin and the leak-free reward readout
# --------------------------------------------------------------------------

from memval.benchmarks.extinction_preference import (  # noqa: E402
    choice_margin, choice_margin_ceiling)


def test_margin_is_antisymmetric_in_both_modes(task):
    m = FixedArm(task, "left")
    for mode in ("rollout", "forced"):
        a = choice_margin(m, task, "left", mode)
        b = choice_margin(m, task, "right", mode)
        assert a == pytest.approx(-b, abs=1e-9)


def test_margin_is_positive_for_the_arm_the_model_runs(task):
    for arm in ("left", "right"):
        m = FixedArm(task, arm)
        assert choice_margin(m, task, arm, "forced") > 0
        assert choice_margin(m, task, other(arm), "forced") < 0


def test_rollout_mode_cues_only_the_entrance(task):
    """`forced` walks the true stem in; `rollout` sees the entrance, then only
    the model's own output. Uses a constant-output model so a fed-back
    prediction is distinguishable from a true stem state (a perfect predictor's
    rollout would legitimately revisit them)."""

    class Constant:
        def __init__(self, n):
            self.out = np.full(n, 0.017)
            self.seen = []

        def reset_context(self):
            pass

        def predict_next(self, x, current_context=None):
            self.seen.append(x.copy())
            return self.out.copy()

    se = task["stem_end"]
    stem = _true_stem(task)
    for mode, expected_true in (("forced", se), ("rollout", 1)):
        m = Constant(task["n_features"])
        choice_margin(m, task, "left", mode)
        assert len(m.seen) == se
        n_true = sum(any(np.allclose(x, s) for s in stem) for x in m.seen)
        assert n_true == expected_true, mode
        if mode == "rollout":
            assert np.allclose(m.seen[0], stem[0])
            assert all(np.allclose(x, m.out) for x in m.seen[1:])


def _true_stem(task):
    from memval.generators.t_maze_extinction import TMazeExtinctionGenerator as Gen
    return list(Gen.trajectory(task, "left", 1.0, "on")[: task["stem_end"]])


def test_ceiling_is_the_candidate_separation(task):
    se = task["stem_end"]
    a, b = task["place"]["left"][se], task["place"]["right"][se]
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    assert choice_margin_ceiling(task) == pytest.approx(1 - cos)
    m = FixedArm(task, "left")
    assert choice_margin(m, task, "left", "forced") <= choice_margin_ceiling(task) + 1e-9


def test_reward_entry_is_the_one_step_that_cannot_copy_its_input(task):
    """A copy-the-input model scores ~0 on entry and high on the zone average."""

    class Copier:
        def reset_context(self):
            pass

        def fit_sequence(self, X, epochs=None, **kw):
            pass

        def predict_next(self, x, current_context=None):
            return x.copy()

    c = ExtinctionPreferenceBenchmark()._content(Copier(), task, "left")
    assert c["reward_entry"] == pytest.approx(0.0, abs=1e-9)
    assert c["reward_pred"] > 0.5


# --------------------------------------------------------------------------
# Trace quality from the fork: reward-free, direction-imposed
# --------------------------------------------------------------------------

from memval.benchmarks.extinction_preference import arm_rollout  # noqa: E402


def test_a_perfect_runner_has_near_zero_trace_error_on_its_own_arm(task):
    for arm in ("left", "right"):
        r = arm_rollout(FixedArm(task, arm), task, arm)
        assert r["trace_error"] < 1e-6
        assert r["trace_fidelity"] == pytest.approx(1.0, abs=1e-6)


def test_running_the_wrong_arm_scores_fidelity_zero(task):
    """'As wrong as the other arm' is exactly fidelity 0 by definition.

    Uses an index-exact stub (emits the other arm's state for the step being
    predicted) rather than FixedArm, whose nearest-step lookup does not land
    cleanly on the other arm when cued from this one's entry.
    """

    class OtherArmAtIndex:
        def __init__(self, task, wrong_arm):
            from memval.generators.t_maze_extinction import TMazeExtinctionGenerator as Gen
            self.X = Gen.trajectory(task, wrong_arm, 1.0, "on")
            self.i = -1

        def reset_context(self):
            self.i = -1

        def predict_next(self, x, current_context=None):
            self.i += 1
            return self.X[min(self.i + 1, len(self.X) - 1)].copy()

    r = arm_rollout(OtherArmAtIndex(task, "right"), task, "left")
    assert r["trace_error_norm"] == pytest.approx(1.0, abs=1e-9)
    assert r["trace_fidelity"] == pytest.approx(0.0, abs=1e-9)
    assert r["route"] == pytest.approx(0.0)


def test_trace_error_is_independent_of_the_reward_channel(task):
    """The readout must not move when only the reward block changes."""

    class RewardBlind(FixedArm):
        pass

    m = RewardBlind(task, "left")
    a = arm_rollout(m, task, "left")
    task2 = dict(task); task2["reward_gain"] = 5.0          # only the outcome scale
    b = arm_rollout(RewardBlind(task2, "left"), task2, "left")
    assert a["trace_error"] == pytest.approx(b["trace_error"], abs=1e-9)


def test_trace_error_by_step_has_one_entry_per_rolled_out_step(task):
    r = arm_rollout(FixedArm(task, "left"), task, "left")
    assert len(r["trace_error_by_step"]) == task["sequence_length"] - task["stem_end"] - 1
    assert r["trace_error"] == pytest.approx(np.mean(r["trace_error_by_step"]))


def test_a_negative_output_model_still_decodes_off_the_grid_centre(task):
    """The affine remap collapses a small-amplitude negative-output arm onto the
    grid centre; clipping must leave the spatial structure readable.

    The amplitudes here match the real failure: `temporal_pc` produces a place
    block in roughly [-0.02, +0.05], which the ``0.5x + 0.5`` remap buries under
    a 0.5 pedestal across all cells.
    """
    from memval.benchmarks.extinction_preference import _decodable

    se = task["stem_end"]
    true = task["place"]["left"][se + 1:]
    signal = 0.05 * true / true.max() - 0.01          # tPC-like scale, goes negative
    assert signal.min() < 0.0 and signal.max() < 0.06

    gt = task["gt_coords"]["left"][se + 1:]
    dec_clip = task["encoder"].decode(_decodable(signal))
    dec_affine = task["encoder"].decode(0.5 * signal + 0.5)

    err_clip = np.linalg.norm(dec_clip - gt, axis=1).mean()
    err_affine = np.linalg.norm(dec_affine - gt, axis=1).mean()
    assert err_clip < err_affine

    # The route has real extent; the affine decode loses nearly all of it.
    extent_true = float(np.ptp(gt, axis=0).max())
    assert np.ptp(dec_affine, axis=0).max() < 0.05 * extent_true
    assert np.ptp(dec_clip, axis=0).max() > 0.5 * extent_true
