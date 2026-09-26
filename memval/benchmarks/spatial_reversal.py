"""T-maze reversal benchmark.

Scores the opposite failure mode from the retention matrix: how fast a model
*overwrites* an association that has become invalid, rather than how well it
protects an old one. Motivated by Hasselmo, Bodelon & Wyble (2002); see
docs/spatial_reversal_design.md for the full design rationale, including why the
reward channel is scoped to this section and why anticipation is measured in the
model's predictions rather than injected into the stimulus.
"""
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..models.base import HippocampalModel
from .base import Benchmark
from .spatial_disambiguation import consolidate_if_supported


# Stage plans per protocol condition. Each stage is
# (name, training-sequence key, arm being run, goal vector key).
# The extinction stage runs the *same* arm with the reward block zeroed -- the
# paper's a_EC = 0 error trial -- so what should change is the reward
# prediction, not the branch.
_CONDITIONS: Dict[str, List[Tuple[str, str, str, str]]] = {
    "direct": [
        ("acquisition", "left_rewarded", "left", "reward_vec_left"),
        ("reversal", "right_rewarded", "right", "reward_vec_right"),
    ],
    "extinction": [
        ("acquisition", "left_rewarded", "left", "reward_vec_left"),
        ("extinction", "left_unrewarded", "left", "reward_vec_left"),
        ("reversal", "right_rewarded", "right", "reward_vec_right"),
    ],
    "extinction_only": [
        ("acquisition", "left_rewarded", "left", "reward_vec_left"),
        ("extinction", "left_unrewarded", "left", "reward_vec_left"),
    ],
}

# Appended when measure_savings=True: a second reversal back to the original arm.
_SAVINGS_STAGE = ("rereversal", "left_rewarded", "left", "reward_vec_left")

# Per-trial probe metrics, in the order they are reported.
PROBE_METRICS = (
    "arm_accuracy",
    "perseveration",
    "reward_pred",
    "goal_identity_margin",
    "reward_pre_goal",
    "anticipation_lead",
)


class SpatialReversalBenchmark(Benchmark):
    """
    Evaluates how fast a model extinguishes a learned place->reward association
    and acquires its opposite, on a T-maze whose stem is a perfectly overlapping
    context.

    A **trial is one presentation** -- a single ``fit_sequence`` call. The
    epochs-per-presentation is whatever the model was constructed with, held
    equal across arms; ``OriginalEqPropSequenceNetwork.fit_sequence`` ignores a
    call-level ``epochs=`` kwarg, so per-call overrides are not a portable way to
    fix exposure. Every stage runs a fixed ``max_trials`` presentations rather
    than stopping at criterion, so learning curves are complete and equal-length
    across arms and trials-to-criterion is derived post hoc.

    Probes never train, and are **fully autoregressive on both blocks** -- unlike
    the disambiguation benchmark, which clamps its odour cue throughout the
    rollout. An odour is a cue available in advance; a reward is an outcome.
    Clamping it would hand the model the goal and destroy the anticipation
    measurement.
    """

    def __init__(
        self,
        criterion: float = 0.75,
        criterion_window: int = 2,
        anticipation_threshold: float = 0.1,
    ):
        """
        Args:
            criterion: ``arm_accuracy`` a trial must reach to count as learned.
            criterion_window: How many consecutive trials must hold the criterion.
            anticipation_threshold: Fraction of ``reward_gain`` the predicted
                reward projection must cross *before* the goal zone to count as
                an anticipatory signal. Deliberately low: anticipation is a
                sub-threshold ramp ahead of the true onset, so a high threshold
                measures nothing but the onset itself. See the note on
                ``anticipation_lead`` in ``_probe``.
        """
        self.criterion = criterion
        self.criterion_window = criterion_window
        self.anticipation_threshold = anticipation_threshold

    # ------------------------------------------------------------------
    # Probing (no learning)
    # ------------------------------------------------------------------

    def _rollout(self, model: HippocampalModel, task: Dict[str, Any]) -> np.ndarray:
        """Cue the model with the stem and roll the arm out autoregressively.

        The stem is identical across all four task variants (shared corridor,
        reward block zero throughout), so which variant seeds the cue is
        immaterial -- that perfect overlap is the point of the task.

        Returns the ``(arm_steps, n_features)`` raw predictions, where row ``i``
        is the model's prediction for global timestep ``stem_end + i``.
        """
        seq = task["left_rewarded"]
        stem_end = task["stem_end"]
        L = len(seq)

        model.reset_context()
        for t in range(stem_end - 1):
            model.predict_next(seq[t])

        current = seq[stem_end - 1].copy()
        preds = []
        for _ in range(L - stem_end):
            pred = np.asarray(model.predict_next(current), dtype=float).ravel()
            preds.append(pred)
            current = pred  # both blocks fed back; nothing is clamped
        return np.array(preds)

    def _probe(
        self,
        model: HippocampalModel,
        task: Dict[str, Any],
        target_arm: str,
        goal_vec: np.ndarray,
        gt_coords: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """One probe rollout, scored. See PROBE_METRICS."""
        n_pc = task["n_place_cells"]
        stem_end = task["stem_end"]
        goal_start = task["goal_start"]
        reward_gain = task["reward_gain"]
        encoder = task["encoder"]

        preds = self._rollout(model, task)
        if len(preds) == 0:
            return {k: float("nan") for k in PROBE_METRICS}

        # --- branch readout, place block only ---
        place = preds[:, :n_pc].copy()
        # The place-cell decoder is a centre of mass over non-negative
        # activations, so a tanh-range model must be remapped to [0, 1] first.
        # Models already in [0, 1] or [0, inf) are used as-is; applying the remap
        # unconditionally would lift every cell above 0.5 and collapse the decode
        # to the grid centre.
        if place.min() < 0.0:
            place = 0.5 * place + 0.5
        decoded = encoder.decode(place)

        idx = np.arange(stem_end, stem_end + len(preds))
        d_left = np.linalg.norm(decoded - gt_coords["left"][idx], axis=1)
        d_right = np.linalg.norm(decoded - gt_coords["right"][idx], axis=1)
        closer_left = d_left < d_right
        closer_right = d_right < d_left
        frac_left = float(np.mean(closer_left))
        frac_right = float(np.mean(closer_right))

        if target_arm == "left":
            arm_accuracy, perseveration = frac_left, frac_right
        else:
            arm_accuracy, perseveration = frac_right, frac_left

        # --- reward readout, reward block only ---
        # Projections onto the stage's goal direction and onto the opposite
        # goal. Both blocks are kept separate from the spatial score: mixing
        # them would make arms with strong reward prediction look spatially
        # better.
        unit = goal_vec / np.linalg.norm(goal_vec)
        other_vec = (task["reward_vec_right"] if np.allclose(goal_vec, task["reward_vec_left"])
                     else task["reward_vec_left"])
        other_unit = other_vec / np.linalg.norm(other_vec)

        rewards = preds[:, n_pc:]
        proj = rewards @ unit
        proj_other = rewards @ other_unit

        in_goal = slice(max(0, goal_start - stem_end), len(preds))
        pre_goal = slice(0, max(0, goal_start - stem_end))
        scale = reward_gain if reward_gain > 0 else 1.0

        reward_pred = float(np.mean(proj[in_goal]) / scale)
        goal_identity_margin = float(np.mean(proj[in_goal] - proj_other[in_goal]) / scale)
        pre = proj[pre_goal]
        reward_pre_goal = float(np.mean(pre) / scale) if len(pre) else float("nan")

        if reward_gain > 0:
            # anticipation_lead is a *bio-signature*, not a performance score.
            # A model that predicts the next state exactly scores 0 by
            # construction, because the reward genuinely is not there yet. A
            # positive lead means the reward representation has smeared backward
            # along the arm -- the hippocampal signature (Mehta et al. 1997/2000)
            # that motivated measuring anticipation instead of injecting it.
            crossings = np.where(pre >= self.anticipation_threshold * reward_gain)[0]
            if len(crossings):
                first_global = stem_end + int(crossings[0])
                anticipation_lead = float(max(0, goal_start - first_global))
            else:
                anticipation_lead = 0.0
        else:
            # Dead-channel control: the goal zone carries no signal, so a
            # threshold on it is meaningless rather than trivially satisfied.
            anticipation_lead = float("nan")

        return {
            "arm_accuracy": arm_accuracy,
            "perseveration": perseveration,
            "reward_pred": reward_pred,
            "goal_identity_margin": goal_identity_margin,
            "reward_pre_goal": reward_pre_goal,
            "anticipation_lead": anticipation_lead,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _trials_to_criterion(self, curve: List[float]) -> Tuple[float, bool]:
        """First trial (1-based) after which the criterion holds for
        ``criterion_window`` consecutive trials. Returns ``(trials, reached)``;
        an unmet criterion returns ``len(curve)`` so the scalar stays plottable,
        with ``reached=False`` carrying the distinction."""
        w = self.criterion_window
        for i in range(len(curve) - w + 1):
            window = curve[i:i + w]
            if all((not np.isnan(v)) and v >= self.criterion for v in window):
                return float(i + 1), True
        return float(len(curve)), False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: HippocampalModel,
        task: Dict[str, Any],
        condition: str = "extinction",
        max_trials: int = 12,
        measure_savings: bool = False,
        interference_trials: int = 3,
        generator: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run one protocol condition end to end.

        Args:
            model: A freshly constructed model. It is trained in place.
            task: Output of ``TMazeReversalGenerator.generate()``.
            condition: ``"direct"`` | ``"extinction"`` | ``"extinction_only"``.
                Whether the extinction stage helps is a measured manipulation,
                not a design assumption -- hence the conditions.
            max_trials: Presentations per stage. Every stage runs all of them.
            measure_savings: Append a second reversal back to the original arm
                and report re-acquisition speed against the first acquisition.
                Costs one extra stage.
            interference_trials: Presentations of an unrelated trajectory run
                after the final stage, before the recovery probe. Zero disables
                the recovery probe.
            generator: The ``TMazeReversalGenerator`` that produced ``task``,
                needed to build the interference trajectory. Optional; if absent
                the recovery probe is skipped.

        Returns:
            Dict of scalar metrics, plus a ``"curves"`` entry holding the
            per-trial learning curves (a nested dict of lists -- this benchmark
            deviates from the flat ``Dict[str, float]`` of the base class because
            trials-to-criterion is meaningless without the curve behind it).
        """
        if condition not in _CONDITIONS:
            raise ValueError(
                f"Unknown reversal condition '{condition}'. "
                f"Available: {sorted(_CONDITIONS)}"
            )

        stages = list(_CONDITIONS[condition])
        if measure_savings:
            stages.append(_SAVINGS_STAGE)

        n_pc = task["n_place_cells"]
        encoder = task["encoder"]
        gt_coords = {
            "left": encoder.decode(task["left_rewarded"][:, :n_pc]),
            "right": encoder.decode(task["right_rewarded"][:, :n_pc]),
        }

        metrics: Dict[str, Any] = {}
        curves: Dict[str, Dict[str, List[float]]] = {}

        for stage_name, seq_key, target_arm, goal_key in stages:
            train_seq = task[seq_key]
            goal_vec = task[goal_key]
            stage_curve: Dict[str, List[float]] = {k: [] for k in PROBE_METRICS}

            for _trial in range(max_trials):
                model.reset_context()
                model.fit_sequence(train_seq)
                probe = self._probe(model, task, target_arm, goal_vec, gt_coords)
                for k in PROBE_METRICS:
                    stage_curve[k].append(probe[k])

            # Consolidation belongs at stage boundaries, matching the chain
            # harness. Consolidating every trial would re-anchor mid-stage and
            # change the mechanism under test.
            consolidate_if_supported(model, train_seq)

            curves[stage_name] = stage_curve
            ttc, reached = self._trials_to_criterion(stage_curve["arm_accuracy"])
            metrics[f"{stage_name}_trials_to_criterion"] = ttc
            metrics[f"{stage_name}_criterion_reached"] = bool(reached)
            for key in PROBE_METRICS:
                metrics[f"{stage_name}_final_{key}"] = float(stage_curve[key][-1])
            leads = stage_curve["anticipation_lead"]
            metrics[f"{stage_name}_max_anticipation_lead"] = float(
                np.nanmax(leads) if not np.all(np.isnan(leads)) else np.nan
            )

        # Extinction is scored on the reward channel, not the branch: the rat
        # still runs the same arm, so what should change is the predicted reward.
        if "extinction" in curves:
            ext = curves["extinction"]["reward_pred"]
            metrics["extinction_reward_drop"] = float(ext[0] - ext[-1])

        # Savings: faster re-acquisition of the original association indicates a
        # trace that survived extinction (extinction is not erasure).
        if measure_savings and "rereversal" in curves:
            re_acq = metrics["rereversal_trials_to_criterion"]
            metrics["savings"] = float(metrics["acquisition_trials_to_criterion"] - re_acq)
            # Acquisition from a blank memory has no competitor and saturates on
            # trial 1, so `savings` against it is floored. The comparison that
            # actually carries information is against the *first* reversal, which
            # faced a competing association just as the re-reversal does.
            if "reversal_trials_to_criterion" in metrics:
                metrics["savings_vs_reversal"] = float(
                    metrics["reversal_trials_to_criterion"] - re_acq
                )

        # Recovery after retroactive interference. Deliberately NOT called
        # spontaneous recovery: a deterministic model with no state drift would
        # return an identical probe after mere elapsed time, so the delay is
        # filled with unrelated experience instead.
        last_stage, _, last_arm, last_goal_key = stages[-1]
        if interference_trials > 0 and generator is not None:
            neutral = generator.neutral_trajectory(task)
            for _ in range(interference_trials):
                model.reset_context()
                model.fit_sequence(neutral)
            post = self._probe(model, task, last_arm, task[last_goal_key], gt_coords)
            metrics["recovery_after_interference_arm_accuracy"] = post["arm_accuracy"]
            metrics["recovery_after_interference_perseveration"] = post["perseveration"]
            metrics["recovery_after_interference_delta"] = float(
                post["perseveration"] - curves[last_stage]["perseveration"][-1]
            )

        metrics["condition"] = condition
        metrics["max_trials"] = int(max_trials)
        metrics["curves"] = curves
        return metrics
