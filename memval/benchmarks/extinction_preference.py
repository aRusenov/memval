"""Adaptive extinction-preference protocol on the cheese-odour T-maze.

Designed from the behavioural expectation, not from any arm's mechanics:

    acquire both arms with cheese  ->  probe which arm is preferred
    extinguish THE PREFERRED ARM   ->  preference should move AWAY from it
    extinguish the (new) preferred ->  and away again

"Extinguish" = run the arm with noise where the reward was and the cheese
odour fading along that arm over trials. Making the protocol adaptive removes
the question of *why* an arm was preferred at the start (side bias, recency, a
coin flip): the test is only whether extinction repels preference from the arm
it is applied to. The stage score is therefore **signed relative to the
extinguished arm**, with three interpretable outcomes:

    negative   preference flees the extinguished arm   (the naturalistic profile)
    positive   preference follows it (running a route reinforces it more than
               losing its reward weakens it)
    ~0         preference does not move

Three readouts from one run:

  preference   free rollout from the stem, which arm it commits to (binary per
               seed), plus a nudge sweep at the choice point giving the
               psychometric midpoint -- the graded bias, which moves before a
               commit flips.
  content      teacher-forced along the extinguished arm's ORIGINAL trajectory:
               predicted reward in the goal zone and predicted odour along the
               arm, relative to the original. What the trace has become.
  savings      (optional) re-reward the first-extinguished arm and count trials
               to recover the reward prediction against acquisition's count.
               Whether the trace was suppressed or erased.

Exposure note: extinction trials are presentations, so the extinguished arm
ends with more exposure than the other. That asymmetry is part of the
phenomenon being measured, not a confound to remove; state it on the figure.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..generators.t_maze_extinction import TMazeExtinctionGenerator as _G
from .base import Benchmark

ARMS = ("left", "right")


def _decodable(place: np.ndarray) -> np.ndarray:
    """Project a place-block prediction onto what the place decoder expects.

    ``PlaceCellEncoder.decode`` is a centre of mass over NON-NEGATIVE
    activations, so an arm whose output goes negative has to be mapped first.
    We clip at zero rather than applying the affine ``0.5x + 0.5`` remap used
    elsewhere in the spatial suite, because that remap lifts every one of the
    ~400 cells above zero and the centre of mass then collapses onto the grid
    centre for ANY input -- measured here as a constant 0.042 trace fidelity
    for `temporal_pc` and `dts_esn` at every exposure from 3 to 60 passes,
    identical to their untrained value. Under clipping the same models decode
    to sensible positions (fidelity 0.49 and 0.42). Clipping is also the
    principled choice: place codes are non-negative by construction, so
    clipping is a projection onto the encoder's own manifold, while the affine
    map moves every arm off it.

    NOTE: `memval/benchmarks/spatial_reversal.py` still uses the affine remap
    and has the same failure mode; changing it would move already-published
    numbers, so it is flagged rather than edited here.
    """
    place = np.asarray(place, dtype=float)
    return np.clip(place, 0.0, None) if place.min() < 0.0 else place


def other(arm: str) -> str:
    return "right" if arm == "left" else "left"


class ExtinctionPreferenceBenchmark(Benchmark):
    def __init__(
        self,
        acquisition_pairs: int = 12,
        extinction_trials: int = 12,
        second_stage: str = "preferred",
        nudge_max: float = 1.0,
        nudge_points: int = 21,
        reward_noise: float = 1.0,
        commit_threshold: float = 0.5,
        measure_savings: bool = True,
        savings_max_trials: int = 24,
        savings_criterion: float = 0.8,
        seed: int = 0,
    ):
        """
        Args:
            acquisition_pairs: interleaved (left, right) presentation pairs.
            extinction_trials: presentations per extinction stage; the odour
                on the extinguished arm fades linearly to 0 across them.
            second_stage: ``"preferred"`` extinguishes whichever arm is
                preferred after stage 1 (adaptive, the default);
                ``"other"`` extinguishes the arm not extinguished in stage 1.
            nudge_max / nudge_points: the psychometric sweep at the choice
                point, in units of the unit-norm place block.
            reward_noise: sd of the goal-zone noise, in units of reward_gain.
            commit_threshold: |preference| needed to count a rollout as
                committed to an arm.
            measure_savings: re-reward the first-extinguished arm at the end.
            savings_criterion: FRACTION of the arm's own end-of-acquisition
                reward prediction that counts as "re-acquired". Relative, so
                it is meaningful for an arm that plateaus at 0.6 as much as
                one that reaches 1.0; acquisition trials are counted against
                the same level.
        """
        if second_stage not in ("preferred", "other"):
            raise ValueError("second_stage must be 'preferred' or 'other'")
        self.acquisition_pairs = int(acquisition_pairs)
        self.extinction_trials = int(extinction_trials)
        self.second_stage = second_stage
        self.nudge_max = float(nudge_max)
        self.nudge_points = int(nudge_points)
        self.reward_noise = float(reward_noise)
        self.commit_threshold = float(commit_threshold)
        self.measure_savings = bool(measure_savings)
        self.savings_max_trials = int(savings_max_trials)
        self.savings_criterion = float(savings_criterion)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ probes
    def _rollout(self, model, task, nudge: float = 0.0) -> np.ndarray:
        """Free autoregressive rollout of the arm from the (shared) stem.

        ``nudge`` adds ``nudge * (right_next - left_next)`` on the place dims
        of the choice-point cue, unit direction, so positive pushes right."""
        seq = _G.trajectory(task, "left", 1.0, "on")       # stem is identical for both
        se, L, n_pc = task["stem_end"], task["sequence_length"], task["n_place_cells"]
        model.reset_context()
        for t in range(se - 1):
            model.predict_next(seq[t])
        cue = seq[se - 1].copy()
        if nudge != 0.0:
            d = task["place"]["right"][se] - task["place"]["left"][se]
            d = d / (np.linalg.norm(d) + 1e-12)
            cue[:n_pc] += nudge * d
        preds, cur = [], cue
        for _ in range(L - se):
            p = np.asarray(model.predict_next(cur), dtype=float).ravel()
            preds.append(p)
            cur = p
        return np.array(preds)

    def _preference(self, preds: np.ndarray, task) -> Tuple[float, float, float]:
        """(pref, frac_left, frac_right); pref = frac_right - frac_left in [-1, 1]."""
        n_pc, se = task["n_place_cells"], task["stem_end"]
        place = _decodable(preds[:, :n_pc])
        dec = task["encoder"].decode(place)
        idx = np.arange(se, se + len(preds))
        dl = np.linalg.norm(dec - task["gt_coords"]["left"][idx], axis=1)
        dr = np.linalg.norm(dec - task["gt_coords"]["right"][idx], axis=1)
        fl, fr = float(np.mean(dl < dr)), float(np.mean(dr < dl))
        return fr - fl, fl, fr

    def _choice(self, pref: float) -> str:
        if pref >= self.commit_threshold:
            return "right"
        if pref <= -self.commit_threshold:
            return "left"
        return "none"

    def _psychometric(self, model, task) -> Tuple[np.ndarray, np.ndarray, float]:
        eps = np.linspace(-self.nudge_max, self.nudge_max, self.nudge_points)
        prefs = np.array([self._preference(self._rollout(model, task, e), task)[0] for e in eps])
        # midpoint: first zero crossing, interpolated; saturated -> +-nudge_max
        if np.all(prefs > 0):
            mid = -self.nudge_max
        elif np.all(prefs < 0):
            mid = self.nudge_max
        else:
            s = np.sign(prefs); s[s == 0] = 1
            k = int(np.argmax(s[1:] != s[:-1]))
            y0, y1, x0, x1 = prefs[k], prefs[k + 1], eps[k], eps[k + 1]
            mid = float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)
        return eps, prefs, float(mid)

    def _content(self, model, task, arm: str) -> Dict[str, float]:
        """Teacher-forced one-step predictions along the arm's ORIGINAL run."""
        X = _G.trajectory(task, arm, 1.0, "on")
        n_pc, n_od = task["n_place_cells"], task["n_odour_dims"]
        se, gs, L = task["stem_end"], task["goal_start"], task["sequence_length"]
        vec = task["reward_vec_left"] if arm == "left" else task["reward_vec_right"]
        model.reset_context()
        rew, od, entry = [], [], float("nan")
        for t in range(L - 1):
            p = np.asarray(model.predict_next(X[t]), dtype=float).ravel()
            if t + 1 >= se:
                od.append(float(p[n_pc:n_pc + n_od] @ task["odour_vec"]))
            if t + 1 >= gs:
                r = float(p[n_pc + n_od:] @ vec)
                rew.append(r)
                if t + 1 == gs:
                    # The ONE leak-free step: predicting INTO the goal zone from
                    # the last position outside it, whose input reward block is
                    # still zero. Every later goal-zone step is teacher-forced
                    # with the reward already in the input, so it can be part
                    # copy. `reward_entry` is the anticipation measurement;
                    # `reward_pred` averages the zone and is kept for continuity.
                    entry = r
        orig_od = float(np.mean(task["odour_profile"][se:]))
        return {
            "reward_pred": float(np.mean(rew)) / task["reward_gain"],
            "reward_entry": entry / task["reward_gain"],
            "odour_retained": float(np.mean(od)) / orig_od,
        }

    def _probe(self, model, task, ext_arm: Optional[str]) -> Dict[str, Any]:
        pref, fl, fr = self._preference(self._rollout(model, task), task)
        _, prefs, mid = self._psychometric(model, task)
        bias_right = -mid                         # goes right without a push -> positive
        rec = {
            "pref": pref, "frac_left": fl, "frac_right": fr, "choice": self._choice(pref),
            "midpoint": mid, "bias_right": bias_right,
            "psychometric": prefs.tolist(),
            "content": {arm: self._content(model, task, arm) for arm in ARMS},
        }
        if ext_arm is not None:
            sgn = 1.0 if ext_arm == "right" else -1.0
            rec["pref_ext"] = sgn * pref
            rec["bias_ext"] = sgn * bias_right
            rec["chose_ext"] = float(rec["choice"] == ext_arm)
        return rec

    def _preferred(self, rec: Dict[str, Any]) -> str:
        if rec["choice"] != "none":
            return rec["choice"]
        return "right" if rec["bias_right"] >= 0 else "left"

    # ---------------------------------------------------------------- protocol
    def evaluate(self, model, task, epochs: int = 3, end_on: str = "right", **kw) -> Dict[str, Any]:
        """Run acquisition -> extinction 1 -> extinction 2 [-> savings]."""
        fit = lambda X: model.fit_sequence(X, epochs=epochs)
        curves: Dict[str, List[Dict[str, Any]]] = {}
        metrics: Dict[str, Any] = {}

        # --- stage 1: acquisition, interleaved, ending on `end_on` ----------
        first = other(end_on)
        acq: List[Dict[str, Any]] = []
        acq_reached = {a: None for a in ARMS}
        for k in range(self.acquisition_pairs):
            fit(_G.trajectory(task, first, 1.0, "on"))
            fit(_G.trajectory(task, end_on, 1.0, "on"))
            rec = self._probe(model, task, None); rec["trial"] = k + 1
            acq.append(rec)
        curves["acquisition"] = acq
        # Criterion level per arm: a fraction of what THIS arm reached.
        crit = {a: self.savings_criterion * acq[-1]["content"][a]["reward_pred"] for a in ARMS}
        for a in ARMS:
            hits = [r["trial"] for r in acq if r["content"][a]["reward_pred"] >= crit[a]]
            acq_reached[a] = hits[0] if hits else None
            metrics[f"acquisition_{a}_criterion_level"] = crit[a]
        pref_arm = self._preferred(acq[-1])
        metrics["acquisition_preferred_arm"] = pref_arm
        metrics["acquisition_end_on"] = end_on
        metrics["acquisition_final_pref"] = acq[-1]["pref"]
        metrics["acquisition_final_bias_right"] = acq[-1]["bias_right"]
        for a in ARMS:
            metrics[f"acquisition_{a}_reward_pred"] = acq[-1]["content"][a]["reward_pred"]
            metrics[f"acquisition_{a}_trials_to_reward_criterion"] = acq_reached[a]

        # --- stages 2, 3: extinction of the preferred arm ------------------
        def extinguish(stage: str, arm: str, baseline: Dict[str, Any]):
            recs = []
            base = dict(baseline)
            sgn = 1.0 if arm == "right" else -1.0
            base["pref_ext"] = sgn * base["pref"]; base["bias_ext"] = sgn * base["bias_right"]
            base["chose_ext"] = float(base["choice"] == arm); base["trial"] = 0
            recs.append(base)
            for k in range(self.extinction_trials):
                fade = 1.0 - (k + 1) / self.extinction_trials
                fit(_G.trajectory(task, arm, fade, "noise", self.rng, self.reward_noise))
                rec = self._probe(model, task, arm); rec["trial"] = k + 1; rec["odour_scale"] = fade
                recs.append(rec)
            curves[stage] = recs
            e0, e1 = recs[0], recs[-1]
            metrics[f"{stage}_arm"] = arm
            metrics[f"{stage}_pref_ext_start"] = e0["pref_ext"]
            metrics[f"{stage}_pref_ext_end"] = e1["pref_ext"]
            metrics[f"{stage}_delta_pref_ext"] = e1["pref_ext"] - e0["pref_ext"]
            metrics[f"{stage}_bias_ext_start"] = e0["bias_ext"]
            metrics[f"{stage}_bias_ext_end"] = e1["bias_ext"]
            metrics[f"{stage}_delta_bias_ext"] = e1["bias_ext"] - e0["bias_ext"]
            metrics[f"{stage}_chose_ext_start"] = e0["chose_ext"]
            metrics[f"{stage}_chose_ext_end"] = e1["chose_ext"]
            for a in ARMS:
                tag = "ext" if a == arm else "other"
                metrics[f"{stage}_{tag}_reward_pred_start"] = e0["content"][a]["reward_pred"]
                metrics[f"{stage}_{tag}_reward_pred_end"] = e1["content"][a]["reward_pred"]
                metrics[f"{stage}_{tag}_odour_retained_start"] = e0["content"][a]["odour_retained"]
                metrics[f"{stage}_{tag}_odour_retained_end"] = e1["content"][a]["odour_retained"]
            return recs

        ext1 = extinguish("extinction_1", pref_arm, acq[-1])
        arm2 = self._preferred(ext1[-1]) if self.second_stage == "preferred" else other(pref_arm)
        ext2 = extinguish("extinction_2", arm2, ext1[-1])

        # --- savings: re-reward the first-extinguished arm -----------------
        if self.measure_savings:
            sav, reached = [], None
            for k in range(self.savings_max_trials):
                fit(_G.trajectory(task, pref_arm, 1.0, "on"))
                rec = self._probe(model, task, pref_arm); rec["trial"] = k + 1
                sav.append(rec)
                if rec["content"][pref_arm]["reward_pred"] >= crit[pref_arm]:
                    reached = k + 1
                    break
            curves["savings"] = sav
            a0 = acq_reached[pref_arm]
            metrics["savings_arm"] = pref_arm
            metrics["savings_trials_to_reacquire"] = reached
            metrics["savings_acquisition_trials"] = a0
            metrics["savings_ratio"] = (float(reached) / a0 if (reached and a0) else float("nan"))
            metrics["savings_reached"] = reached is not None

        # Guard: a diverging arm produces huge reward/odour predictions and a
        # rollout that decodes to nothing meaningful. Flag it rather than let
        # it pass as a preference result.
        peak = max(abs(rec["content"][a][f]) for recs in curves.values() for rec in recs
                   for a in ARMS for f in ("reward_pred", "odour_retained"))
        metrics["max_abs_content"] = float(peak)
        metrics["diverged"] = bool(peak > 3.0)
        return {"metrics": metrics, "curves": curves}


# ---------------------------------------------------------------------------
# Per-arm probes for the timeline view (train A -> train B -> extinguish one)
# ---------------------------------------------------------------------------

def choice_margin(model, task, arm: str, mode: str = "rollout") -> float:
    """Decision margin on *what comes after the choice point*, for ``arm``.

    The model is asked for one thing -- the state that follows the last stem
    position -- and there are exactly two possible answers, the first step of
    the left arm and the first step of the right arm. Scored on the place dims
    only (odour and reward excluded, so a strong outcome prediction cannot
    inflate a spatial choice):

        margin(arm) = cos(pred, arm's first arm-step)
                      - cos(pred, other arm's first arm-step)

    Same definition as the suite's symbolic ``measure_recall_margin`` -- cosine
    to the target minus cosine to the best competitor -- with a competitor set
    of one, because a T-maze has two continuations. Zero is a genuine decision
    boundary, and the readout is antisymmetric (``margin(A) == -margin(B)``),
    so one arm's line carries all of it.

    ``mode``:
      ``"rollout"`` (default) -- cue with the FIRST stem position only and
        unroll autoregressively to the scored step, so the model navigates the
        stem on its own predictions. Drift through the stem is included, and a
        latent-state arm builds its state the way it would in use. This is what
        the figures report.
      ``"forced"`` -- walk the true stem in and take one step from its last
        position. No drift; isolates the association from stem navigation.
        Useful as the control that says whether a low rollout margin is a
        choice failure or a stem-drift failure.

    The margin's ceiling is set by the stimulus, not the model: it is
    ``1 - cos(left_first_step, right_first_step)``, because the two candidates
    are the arms' first divergent step and are not yet orthogonal. Report it
    beside the number (``choice_margin_ceiling``).
    """
    if mode not in ("rollout", "forced"):
        raise ValueError(f"mode must be 'rollout' or 'forced', got {mode!r}")
    seq = _G.trajectory(task, "left", 1.0, "on")     # stem identical across variants
    se, n_pc = task["stem_end"], task["n_place_cells"]
    model.reset_context()
    if mode == "forced":
        for t in range(se - 1):
            model.predict_next(seq[t])
        p = np.asarray(model.predict_next(seq[se - 1]), dtype=float).ravel()
    else:
        cur = seq[0].copy()
        for _ in range(se):                          # se steps: index 0 -> index se
            p = np.asarray(model.predict_next(cur), dtype=float).ravel()
            cur = p
    p = p[:n_pc]
    pn = np.linalg.norm(p)
    if pn <= 1e-12:
        return -1.0
    p = p / pn
    a = task["place"][arm][se]; b = task["place"][other(arm)][se]
    return float(p @ a / np.linalg.norm(a) - p @ b / np.linalg.norm(b))


def choice_margin_ceiling(task) -> float:
    """Largest margin the stimulus allows: ``1 - cos`` of the two candidates."""
    se = task["stem_end"]
    a, b = task["place"]["left"][se], task["place"]["right"][se]
    return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def arm_rollout(model, task, arm: str) -> Dict[str, float]:
    """Roll out from ``arm``'s entry step (teacher-forced past the choice).

    The direction is IMPOSED -- the stem is walked in and the model is placed on
    the arm's first step -- and from there it runs free, feeding its own
    predictions back, so drift accumulates the way it would in use. This
    dissociates *which way it would go* from *whether the route is still there*:
    an arm that no longer prefers this branch may still run it perfectly, and
    an arm that prefers it may no longer be able to.

    Returns:
      ``trace_error``       mean Euclidean distance, in arena units, between the
                            decoded predicted position and the true position,
                            over every step from the fork to the cheese. The
                            trace-quality readout, and it needs no reward
                            channel -- it is purely "does it still know the way".
      ``trace_error_norm``  the same divided by the mean distance between the
                            two arms over those steps, so 1.0 means the
                            prediction is as far from the truth as the other
                            arm is. Scale-free, comparable across mazes.
      ``trace_fidelity``    ``max(0, 1 - trace_error_norm)``: 1.0 perfect,
                            0.0 no better than pointing at the other arm.
      ``route``             fraction of rolled-out steps decoded closer to this
                            arm than to the other (the coarse, saturating form).
      ``reward_in_rollout`` reward predicted where the goal zone falls in THIS
                            rollout. Drift-dominated; kept for continuity.
    """
    X = _G.trajectory(task, arm, 1.0, "on")
    se, gs, L = task["stem_end"], task["goal_start"], task["sequence_length"]
    n_pc, n_od = task["n_place_cells"], task["n_odour_dims"]
    vec = task["reward_vec_left"] if arm == "left" else task["reward_vec_right"]
    model.reset_context()
    for t in range(se):
        model.predict_next(X[t])
    cur, preds = X[se].copy(), []
    for _ in range(L - se - 1):
        p = np.asarray(model.predict_next(cur), dtype=float).ravel()
        preds.append(p); cur = p
    P = np.array(preds)
    place = _decodable(P[:, :n_pc])
    dec = task["encoder"].decode(place)
    idx = np.arange(se + 1, L)
    dm = np.linalg.norm(dec - task["gt_coords"][arm][idx], axis=1)
    do = np.linalg.norm(dec - task["gt_coords"][other(arm)][idx], axis=1)
    goal = idx >= gs
    rew = P[goal, n_pc + n_od:] @ vec / task["reward_gain"] if goal.any() else np.array([np.nan])
    # Positional error against the arm's own ground truth, scaled by how far
    # apart the two arms are over the same steps (the natural unit: 1.0 = as
    # wrong as naming the other arm).
    sep = float(np.mean(np.linalg.norm(
        task["gt_coords"][arm][idx] - task["gt_coords"][other(arm)][idx], axis=1)))
    err = float(np.mean(dm))
    err_norm = err / sep if sep > 1e-12 else float("nan")
    return {"route": float(np.mean(dm < do)),
            "reward_in_rollout": float(np.mean(rew)),
            "trace_error": err,
            "trace_error_norm": err_norm,
            "trace_fidelity": float(max(0.0, 1.0 - err_norm)),
            "trace_error_by_step": np.linalg.norm(
                dec - task["gt_coords"][arm][idx], axis=1).tolist()}
