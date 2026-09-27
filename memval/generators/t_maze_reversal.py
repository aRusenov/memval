import numpy as np
from typing import Optional, Dict, Any

from .base import SequenceGenerator
from .t_maze import TMazeGenerator
from ..encoders.spatial import PlaceCellEncoder


class TMazeReversalGenerator(SequenceGenerator):
    """
    Wraps TMazeGenerator to produce the stimulus set for a T-maze reversal task
    (Hasselmo, Bodelon & Wyble 2002): learn stem->left with food reward, then
    extinguish it and acquire stem->right.

    Each trajectory is emitted in two variants that differ *only* in the reward
    block: a rewarded one (food present at the goal) and an unrewarded one (the
    same run with the reward block left at zero). The unrewarded variant is this
    benchmark's translation of the paper's ``a_EC = 0`` error trial -- the rat
    runs the arm and finds no food.

    Emitted vectors concatenate two blocks::

        [ place cells (n_place_cells) | reward (n_reward_dims) ]

    The reward block is zero everywhere except a box over the final
    ``reward_zone_fraction`` of the arm, where it carries an arm-specific unit
    direction scaled by ``reward_gain``. Left and right goal vectors are
    orthogonal (e_0 and e_1), matching the paper's assumption that the entorhinal
    patterns for left-arm food and right-arm food are orthogonal, and making
    *which* goal a model predicts readable by projection.

    Note that the reward signal is stationary across trials by design: it never
    creeps earlier with experience. Anticipation is measured in the model's
    predictions (see ``SpatialReversalBenchmark``), not injected into the
    stimulus -- see docs/sections/spatial_reversal_design.md section 4.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def generate(
        self,
        sequence_length: int = 16,
        stem_fraction: float = 0.5,
        n_place_cells: int = 400,
        n_reward_dims: int = 20,
        reward_gain: float = 1.0,
        reward_zone_fraction: float = 0.4,
        noise_std: float = 0.0,
        sigma_scale: float = 1.0,
        balance_modalities: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build the rewarded/unrewarded trajectory pair for each arm.

        Args:
            sequence_length: Timesteps per run through the maze.
            stem_fraction: Fraction of the maze length taken by the shared stem.
            n_place_cells: Size of the place-cell block (rounded to a square grid).
            n_reward_dims: Size of the reward block. Must be >= 2 so the left and
                right goal representations can be orthogonal.
            reward_gain: Magnitude of the reward block inside the goal zone. This
                is the teaching-signal strength and is a *choice*: sweep it and
                report the plateau. ``reward_gain=0.0`` yields a present-but-dead
                channel, which is the control isolating the added dimensionality
                from the reward signal itself.
            reward_zone_fraction: Fraction of the *arm* (measured back from the
                goal) over which the reward box is active. A single-timestep
                spike is fragile under rollout decoding; a box is robust.
            noise_std: Gaussian noise on the spatial positions.
            sigma_scale: Place-field width relative to the mean step size.
            balance_modalities: Normalise the place block to unit norm per
                timestep so ``reward_gain`` is an explicit signal strength rather
                than an accident of the place-cell tiling. Place-cell decoding is
                a scale-invariant centre of mass, so this does not distort the
                decoded coordinates.
            seed: Random seed.

        Returns:
            Dict containing:
                - left_rewarded / left_unrewarded: (L, n_place_cells + n_reward_dims)
                - right_rewarded / right_unrewarded: same, for the right arm
                - stem_end: index of the first arm timestep (the choice point)
                - goal_start: index of the first timestep inside the reward zone
                - reward_vec_left / reward_vec_right: the orthogonal goal directions
                - n_place_cells, n_reward_dims, reward_gain
                - encoder: the PlaceCellEncoder (for decoding predictions)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if n_reward_dims < 2:
            raise ValueError(
                "n_reward_dims must be at least 2 so the left and right goal "
                "representations can be orthogonal."
            )
        if not 0.0 < reward_zone_fraction <= 1.0:
            raise ValueError("reward_zone_fraction must be in (0, 1].")

        stem_len_val = stem_fraction
        arm_len_val = 1.0 - stem_fraction

        tmaze_gen = TMazeGenerator(
            stem_length=stem_len_val,
            arm_length=arm_len_val,
            seed=seed,
        )

        # TMazeGenerator splits the sequence between stem and arm by length ratio;
        # mirror that split here so stem_end lines up with the actual choice point.
        total_len = stem_len_val + arm_len_val
        stem_end = max(2, int(sequence_length * (stem_len_val / total_len)))
        arm_steps = sequence_length - stem_end

        traj_left = tmaze_gen.generate(
            n_sequences=1, sequence_length=sequence_length,
            turn_direction="left", noise_std=noise_std,
        )[0]
        traj_right = tmaze_gen.generate(
            n_sequences=1, sequence_length=sequence_length,
            turn_direction="right", noise_std=noise_std,
        )[0]

        n_cells_per_dim = int(np.round(np.sqrt(n_place_cells)))
        encoder = PlaceCellEncoder(
            n_cells_per_dim=n_cells_per_dim,
            env_bounds=1.0,
            seed=seed,
            sigma_scale=sigma_scale,
        )
        pc_left = encoder.encode(traj_left)
        pc_right = encoder.encode(traj_right)

        # Orthogonal goal representations (paper's assumption on the EC patterns).
        reward_vec_left = np.zeros(n_reward_dims)
        reward_vec_left[0] = 1.0
        reward_vec_right = np.zeros(n_reward_dims)
        reward_vec_right[1] = 1.0

        # Reward box: the last reward_zone_fraction of the arm, so goal_start is
        # at least one step past the choice point whenever the arm has room.
        zone_steps = max(1, int(round(arm_steps * reward_zone_fraction)))
        goal_start = sequence_length - zone_steps

        def _reward_block(vec: np.ndarray, rewarded: bool) -> np.ndarray:
            block = np.zeros((sequence_length, n_reward_dims))
            if rewarded:
                block[goal_start:, :] = vec * reward_gain
            return block

        def _assemble(pc: np.ndarray, vec: np.ndarray, rewarded: bool) -> np.ndarray:
            X = np.concatenate([pc, _reward_block(vec, rewarded)], axis=-1)
            if balance_modalities:
                place = X[:, :n_place_cells]
                place_norm = np.linalg.norm(place, axis=1, keepdims=True)
                place_norm[place_norm == 0] = 1.0
                X[:, :n_place_cells] = place / place_norm
            return X

        return {
            "left_rewarded": _assemble(pc_left, reward_vec_left, True),
            "left_unrewarded": _assemble(pc_left, reward_vec_left, False),
            "right_rewarded": _assemble(pc_right, reward_vec_right, True),
            "right_unrewarded": _assemble(pc_right, reward_vec_right, False),
            "stem_end": stem_end,
            "goal_start": goal_start,
            "reward_vec_left": reward_vec_left,
            "reward_vec_right": reward_vec_right,
            "n_place_cells": n_place_cells,
            "n_reward_dims": n_reward_dims,
            "reward_gain": reward_gain,
            "encoder": encoder,
        }

    def neutral_trajectory(
        self,
        task: Dict[str, Any],
        sequence_length: Optional[int] = None,
    ) -> np.ndarray:
        """Build an unrelated trajectory for the interference delay.

        A short run across a different part of the arena (left-to-right along the
        bottom edge, well clear of the maze), encoded in the same feature space
        with an all-zero reward block. Used by the recovery probe: "time" for a
        deterministic model has to be operationalised as intervening experience,
        so the delay is filled with retroactive interference rather than left
        empty (see docs/sections/spatial_reversal_design.md section 8).
        """
        encoder = task["encoder"]
        n_place_cells = task["n_place_cells"]
        n_reward_dims = task["n_reward_dims"]
        L = sequence_length or len(task["left_rewarded"])

        xs = np.linspace(-0.9, 0.9, L)
        ys = np.full(L, -0.8)
        traj = np.column_stack([xs, ys])

        # Encode through a twin encoder with sigma pinned to the maze's inferred
        # value: same grid of centres (they depend only on n_cells_per_dim and
        # env_bounds), same field width, and no mutation of the task encoder --
        # calling encode() on it again would re-infer sigma from this path's step
        # size and silently change the feature space the task was built in.
        twin = PlaceCellEncoder(
            n_cells_per_dim=encoder.n_cells_per_dim,
            env_bounds=encoder.env_bounds,
            sigma=encoder.sigma,
            threshold=encoder.threshold,
        )
        pc = twin.encode(traj)
        place_norm = np.linalg.norm(pc, axis=1, keepdims=True)
        place_norm[place_norm == 0] = 1.0
        pc = pc / place_norm

        return np.concatenate([pc, np.zeros((L, n_reward_dims))], axis=-1)
