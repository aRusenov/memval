"""T-maze with a cheese odour, for the adaptive extinction-preference protocol.

Input vector per step:

    [ place cells (n_place_cells) | odour (n_odour_dims) | reward (n_reward_dims) ]

The odour is the *cheese odour*: a distal cue of the reward, faint and constant
in the stem (ambient), ramping up along each arm to full strength in the goal
zone, on both arms. That gives it a reason to be there and a reason to fade --
when the cheese is removed the odour along that arm decays over trials, which
is how extinction is presented here (``odour_scale`` on the arm steps only; the
stem is never touched, so the stem stays byte-identical across every variant
and the probe cue is unambiguous). The reward block carries an orthogonal goal
vector per arm in the goal zone during acquisition and Gaussian *noise* there
during extinction, so the extinguished outcome is uninformative rather than
zero.

The stem of the right trajectory is copied from the left one before encoding,
so the shared corridor is identical by construction, not by luck.
"""
from typing import Any, Dict, Optional

import numpy as np

from .base import SequenceGenerator
from .t_maze import TMazeGenerator
from ..encoders.spatial import PlaceCellEncoder


class TMazeExtinctionGenerator(SequenceGenerator):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def generate(
        self,
        sequence_length: int = 16,
        stem_fraction: float = 0.5,
        n_place_cells: int = 400,
        n_odour_dims: int = 5,
        n_reward_dims: int = 20,
        reward_gain: float = 1.0,
        reward_zone_fraction: float = 0.4,
        odour_stem: float = 0.2,
        odour_goal: float = 1.0,
        noise_std: float = 0.0,
        sigma_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        seed = self.seed if seed is None else seed
        if n_reward_dims < 2:
            raise ValueError("n_reward_dims must be >= 2 for orthogonal goal vectors")
        if n_odour_dims < 1:
            raise ValueError("n_odour_dims must be >= 1")

        stem_len, arm_len = stem_fraction, 1.0 - stem_fraction
        gen = TMazeGenerator(stem_length=stem_len, arm_length=arm_len, seed=seed)
        stem_end = max(2, int(sequence_length * (stem_len / (stem_len + arm_len))))
        arm_steps = sequence_length - stem_end
        zone_steps = max(1, int(round(arm_steps * reward_zone_fraction)))
        goal_start = sequence_length - zone_steps
        if goal_start <= stem_end:
            raise ValueError("goal zone must start after the choice point")

        traj_left = gen.generate(n_sequences=1, sequence_length=sequence_length,
                                 turn_direction="left", noise_std=noise_std)[0]
        traj_right = gen.generate(n_sequences=1, sequence_length=sequence_length,
                                  turn_direction="right", noise_std=noise_std)[0]
        traj_right = traj_right.copy()
        traj_right[:stem_end] = traj_left[:stem_end]          # shared stem, by construction

        n_per_dim = int(np.round(np.sqrt(n_place_cells)))
        encoder = PlaceCellEncoder(n_cells_per_dim=n_per_dim, env_bounds=1.0,
                                   seed=seed, sigma_scale=sigma_scale)
        place = {}
        for arm, traj in (("left", traj_left), ("right", traj_right)):
            pc = encoder.encode(traj)
            norm = np.linalg.norm(pc, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            place[arm] = pc / norm                            # unit-norm place block
        n_pc = place["left"].shape[1]

        odour_vec = np.zeros(n_odour_dims); odour_vec[0] = 1.0
        reward_vec_left = np.zeros(n_reward_dims); reward_vec_left[0] = 1.0
        reward_vec_right = np.zeros(n_reward_dims); reward_vec_right[1] = 1.0

        # Ambient in the stem, ramping along the arm, full in the goal zone.
        profile = np.full(sequence_length, odour_stem)
        ramp = np.linspace(odour_stem, odour_goal, goal_start - stem_end + 1)
        profile[stem_end:goal_start + 1] = ramp
        profile[goal_start:] = odour_goal

        return {
            "place": place,
            "odour_profile": profile,
            "odour_vec": odour_vec,
            "reward_vec_left": reward_vec_left,
            "reward_vec_right": reward_vec_right,
            "stem_end": stem_end,
            "goal_start": goal_start,
            "sequence_length": sequence_length,
            "n_place_cells": n_pc,
            "n_odour_dims": n_odour_dims,
            "n_reward_dims": n_reward_dims,
            "n_features": n_pc + n_odour_dims + n_reward_dims,
            "reward_gain": reward_gain,
            "encoder": encoder,
            "gt_coords": {arm: encoder.decode(place[arm]) for arm in ("left", "right")},
        }

    @staticmethod
    def trajectory(task: Dict[str, Any], arm: str, odour_scale: float = 1.0,
                   reward: str = "on", rng: Optional[np.random.Generator] = None,
                   noise_scale: float = 1.0) -> np.ndarray:
        """Assemble one run down ``arm``.

        ``odour_scale`` multiplies the odour on the ARM steps only (stem
        ambient odour is untouched). ``reward`` is ``"on"`` (goal vector in the
        goal zone), ``"noise"`` or ``"off"`` (zeros).

        ``"noise"`` is Gaussian in every reward dim, fresh per call, with its
        sd scaled by ``1/sqrt(n_reward_dims)`` so that its *expected norm*
        equals ``noise_scale * reward_gain`` -- the same energy as the reward
        it replaces, carrying no information. Without that scaling a 20-dim
        noise block has ~4.5x the norm of the unit reward vector, which takes a
        delta-rule arm past its stability bound (lr * ||x||^2 < 2) and the
        reward prediction diverges. That is an artefact of the stimulus, not a
        property of the arm, so it is prevented here.
        """
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        L, se, gs = task["sequence_length"], task["stem_end"], task["goal_start"]
        prof = task["odour_profile"].copy()
        prof[se:] *= float(odour_scale)
        odour = prof[:, None] * task["odour_vec"][None, :]
        rew = np.zeros((L, task["n_reward_dims"]))
        vec = task["reward_vec_left"] if arm == "left" else task["reward_vec_right"]
        if reward == "on":
            rew[gs:] = vec * task["reward_gain"]
        elif reward == "noise":
            rng = np.random.default_rng() if rng is None else rng
            sd = noise_scale * task["reward_gain"] / np.sqrt(task["n_reward_dims"])
            rew[gs:] = rng.normal(0.0, sd, size=(L - gs, task["n_reward_dims"]))
        elif reward != "off":
            raise ValueError(f"reward must be 'on', 'noise' or 'off', got {reward!r}")
        return np.concatenate([task["place"][arm], odour, rew], axis=1)
