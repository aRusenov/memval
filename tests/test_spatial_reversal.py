import numpy as np
import pytest

from memval.generators.t_maze_reversal import TMazeReversalGenerator
from memval.benchmarks.spatial_reversal import SpatialReversalBenchmark, PROBE_METRICS
from memval.models.baselines.temporal_pc import TemporalPCNetwork


N_PC = 100
N_REWARD = 4
SEQ_LEN = 16


def _task(**overrides):
    gen = TMazeReversalGenerator(seed=42)
    kwargs = dict(
        sequence_length=SEQ_LEN,
        stem_fraction=0.5,
        n_place_cells=N_PC,
        n_reward_dims=N_REWARD,
        reward_gain=1.0,
        reward_zone_fraction=0.4,
        sigma_scale=1.0,
        balance_modalities=True,
        seed=42,
    )
    kwargs.update(overrides)
    return gen, gen.generate(**kwargs)


def _model(**overrides):
    kwargs = dict(n_features=N_PC + N_REWARD, learning_rate=0.05, n_epochs=10, seed=0)
    kwargs.update(overrides)
    return TemporalPCNetwork(**kwargs)


# ----------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------

def test_generator_shapes_and_geometry():
    _, task = _task()
    n_feat = N_PC + N_REWARD
    for key in ("left_rewarded", "left_unrewarded", "right_rewarded", "right_unrewarded"):
        assert task[key].shape == (SEQ_LEN, n_feat)

    stem_end, goal_start = task["stem_end"], task["goal_start"]
    assert 0 < stem_end < goal_start < SEQ_LEN

    encoder = task["encoder"]
    left_xy = encoder.decode(task["left_rewarded"][:, :N_PC])
    right_xy = encoder.decode(task["right_rewarded"][:, :N_PC])
    # The stem is the perfectly overlapping context; the arms must diverge.
    assert np.allclose(left_xy[:stem_end], right_xy[:stem_end])
    assert not np.allclose(left_xy[stem_end:], right_xy[stem_end:])


def test_reward_block_is_a_stationary_box_at_the_goal():
    _, task = _task()
    goal_start = task["goal_start"]
    reward = task["left_rewarded"][:, N_PC:]

    assert np.allclose(reward[:goal_start], 0.0)
    # Constant inside the zone: the signal never creeps earlier with experience.
    assert np.allclose(reward[goal_start:], task["reward_vec_left"] * task["reward_gain"])
    for t in range(goal_start, SEQ_LEN):
        assert np.allclose(reward[t], reward[goal_start])


def test_rewarded_and_unrewarded_differ_only_in_the_reward_block():
    _, task = _task()
    for arm in ("left", "right"):
        rewarded = task[f"{arm}_rewarded"]
        unrewarded = task[f"{arm}_unrewarded"]
        assert np.allclose(rewarded[:, :N_PC], unrewarded[:, :N_PC])
        assert np.allclose(unrewarded[:, N_PC:], 0.0)
        assert not np.allclose(rewarded[:, N_PC:], unrewarded[:, N_PC:])


def test_goal_representations_are_orthogonal():
    _, task = _task()
    assert np.isclose(task["reward_vec_left"] @ task["reward_vec_right"], 0.0)


def test_dead_channel_control_zeroes_the_signal():
    """reward_gain=0 leaves the block present but always zero, so the control
    isolates added dimensionality from the reward signal itself."""
    _, task = _task(reward_gain=0.0)
    assert task["left_rewarded"].shape[1] == N_PC + N_REWARD
    assert np.allclose(task["left_rewarded"], task["left_unrewarded"])


def test_generator_rejects_degenerate_reward_block():
    gen = TMazeReversalGenerator(seed=0)
    with pytest.raises(ValueError):
        gen.generate(n_reward_dims=1)


def test_neutral_trajectory_matches_the_feature_space():
    gen, task = _task()
    neutral = gen.neutral_trajectory(task)
    assert neutral.shape == task["left_rewarded"].shape
    assert np.allclose(neutral[:, N_PC:], 0.0)
    # Away from the maze, and not a re-run of either arm.
    encoder = task["encoder"]
    assert not np.allclose(
        encoder.decode(neutral[:, :N_PC]),
        encoder.decode(task["left_rewarded"][:, :N_PC]),
    )


def test_neutral_trajectory_does_not_mutate_the_task_encoder():
    """The encoder infers sigma on first encode; encoding the interference path
    through it would silently change the feature space the task was built in."""
    gen, task = _task()
    sigma_before = task["encoder"].sigma
    gen.neutral_trajectory(task)
    assert task["encoder"].sigma == sigma_before


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------

def test_probe_does_not_train():
    """Probes must be pure measurement -- otherwise the learning curve counts
    exposures it never advertised."""
    _, task = _task()
    bench = SpatialReversalBenchmark()
    model = _model()
    model.fit_sequence(task["left_rewarded"])

    before = model.W_r.copy()
    gt = {
        "left": task["encoder"].decode(task["left_rewarded"][:, :N_PC]),
        "right": task["encoder"].decode(task["right_rewarded"][:, :N_PC]),
    }
    bench._probe(model, task, "left", task["reward_vec_left"], gt)
    assert np.allclose(model.W_r, before)


def test_evaluate_returns_full_curves_per_stage():
    gen, task = _task()
    bench = SpatialReversalBenchmark()
    out = bench.evaluate(
        model=_model(), task=task, condition="extinction",
        max_trials=4, interference_trials=1, generator=gen,
    )

    assert set(out["curves"]) == {"acquisition", "extinction", "reversal"}
    for stage in out["curves"].values():
        assert set(stage) == set(PROBE_METRICS)
        # Every stage runs all max_trials, so curves are complete and
        # equal-length across arms -- no early stopping.
        for series in stage.values():
            assert len(series) == 4

    for stage in ("acquisition", "extinction", "reversal"):
        assert f"{stage}_trials_to_criterion" in out
        assert f"{stage}_final_arm_accuracy" in out
    assert "recovery_after_interference_arm_accuracy" in out


def test_direct_condition_skips_the_extinction_stage():
    _, task = _task()
    out = SpatialReversalBenchmark().evaluate(
        model=_model(), task=task, condition="direct",
        max_trials=3, interference_trials=0,
    )
    assert set(out["curves"]) == {"acquisition", "reversal"}
    assert "extinction_reward_drop" not in out


def test_savings_stage_is_opt_in():
    _, task = _task()
    out = SpatialReversalBenchmark().evaluate(
        model=_model(), task=task, condition="direct", max_trials=3,
        measure_savings=True, interference_trials=0,
    )
    assert "rereversal" in out["curves"]
    assert "savings" in out


def test_unknown_condition_fails_loudly():
    _, task = _task()
    with pytest.raises(ValueError):
        SpatialReversalBenchmark().evaluate(model=_model(), task=task, condition="nope")


def test_trials_to_criterion_needs_a_consecutive_window():
    bench = SpatialReversalBenchmark(criterion=0.75, criterion_window=2)
    assert bench._trials_to_criterion([0.2, 0.9, 0.3, 0.9, 0.9]) == (4.0, True)
    # A single lucky trial does not count.
    assert bench._trials_to_criterion([0.2, 0.9, 0.3]) == (3.0, False)
    assert bench._trials_to_criterion([0.9, 0.9, 0.9]) == (1.0, True)


def test_anticipation_lead_is_zero_for_an_untrained_model():
    """A blank memory has no reward representation to smear backward, so the
    lead must start at zero -- it is a learned signature, not a free gift of the
    stimulus (which never moves)."""
    _, task = _task()
    bench = SpatialReversalBenchmark()
    gt = {
        "left": task["encoder"].decode(task["left_rewarded"][:, :N_PC]),
        "right": task["encoder"].decode(task["right_rewarded"][:, :N_PC]),
    }
    probe = bench._probe(_model(), task, "left", task["reward_vec_left"], gt)
    assert probe["anticipation_lead"] == 0.0
    assert abs(probe["reward_pre_goal"]) < 0.05


def test_extinction_decays_the_reward_prediction_not_the_branch():
    """The rat still runs the same arm during extinction, so the branch readout
    should hold while the reward prediction decays."""
    gen, task = _task()
    out = SpatialReversalBenchmark().evaluate(
        model=_model(n_epochs=20, learning_rate=0.1), task=task,
        condition="extinction", max_trials=6, interference_trials=0, generator=gen,
    )
    ext = out["curves"]["extinction"]
    assert ext["reward_pred"][-1] < ext["reward_pred"][0]
    assert out["extinction_reward_drop"] > 0
    assert ext["arm_accuracy"][-1] >= 0.75


def test_dead_channel_control_reports_undefined_anticipation():
    """With no reward signal, a threshold on the reward block is meaningless
    rather than trivially satisfied."""
    _, task = _task(reward_gain=0.0)
    out = SpatialReversalBenchmark().evaluate(
        model=_model(), task=task, condition="direct",
        max_trials=2, interference_trials=0,
    )
    assert np.all(np.isnan(out["curves"]["acquisition"]["anticipation_lead"]))


def test_task_is_learnable_and_reversible():
    """Sanity check on the task itself: a delta-rule arm should acquire the
    first association and then follow the reversal. If this fails the task is
    mis-specified, not the model."""
    gen, task = _task()
    out = SpatialReversalBenchmark(criterion=0.75, criterion_window=1).evaluate(
        model=_model(n_epochs=40, learning_rate=0.1), task=task,
        condition="extinction", max_trials=8, interference_trials=0, generator=gen,
    )
    assert out["acquisition_criterion_reached"], out["curves"]["acquisition"]["arm_accuracy"]
    assert out["reversal_criterion_reached"], out["curves"]["reversal"]["arm_accuracy"]
