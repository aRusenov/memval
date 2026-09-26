from typing import Optional

import numpy as np

from .base import SequenceGenerator


class TMazeGenerator(SequenceGenerator):
    """
    Generates deterministic spatial trajectories representing a standard T-maze
    learning environment (central stem followed by a discrete left or right turn).
    This serves as an excellent benchmark for sequence disambiguation since 
    the central stem acts as a perfectly overlapping context.
    """

    def __init__(self, stem_length: float = 1.0, arm_length: float = 1.0, seed: Optional[int] = None):
        """
        Initialize the T-Maze sequence generator.

        Args:
            stem_length (float): Length of the initial central hallway.
            arm_length (float): Length of the decision arms (left and right).
            seed (int, optional): Random seed.
        """
        super().__init__(seed)
        self.stem_length = stem_length
        self.arm_length = arm_length

    def generate(self, n_sequences: int, sequence_length: int, 
                 turn_direction: str = "both", noise_std: float = 0.0) -> np.ndarray:
        """
        Generate continuous spatial sequences tracking linear progress along the T-maze paths.

        Args:
            n_sequences (int): Batch size.
            sequence_length (int): Total number of discrete points to sample along the spatial continuous path.
            turn_direction (str): "left", "right", or "both" (samples randomly 50/50).
            noise_std (float): Gaussian noise variance on the spatial positions.

        Returns:
            np.ndarray: Trajectories of shape (n_sequences, sequence_length, 2).
        """
        total_length = self.stem_length + self.arm_length
        # Determine how many points of the sequence belong to the stem vs the arm based on length
        stem_points_count = max(2, int(sequence_length * (self.stem_length / total_length)))
        arm_points_count = sequence_length - stem_points_count

        # Base template for the stem: y goes from 0 to stem_length, x remains 0
        stem_y = np.linspace(0, self.stem_length, stem_points_count)
        stem_x = np.zeros_like(stem_y)
        stem = np.column_stack([stem_x, stem_y])

        # Left arm: x goes from 0 to -arm_length, y = stem_length (skip first point to avoid duplicate with stem end)
        left_x = np.linspace(0, -self.arm_length, arm_points_count + 1)[1:]
        left_y = np.full_like(left_x, self.stem_length)
        left_arm = np.column_stack([left_x, left_y])

        # Right arm: x goes from 0 to arm_length, y = stem_length
        right_x = np.linspace(0, self.arm_length, arm_points_count + 1)[1:]
        right_y = np.full_like(right_x, self.stem_length)
        right_arm = np.column_stack([right_x, right_y])

        left_path = np.vstack([stem, left_arm])
        right_path = np.vstack([stem, right_arm])

        trajectories = np.zeros((n_sequences, sequence_length, 2))

        for i in range(n_sequences):
            if turn_direction == "left":
                trajectories[i] = left_path
            elif turn_direction == "right":
                trajectories[i] = right_path
            else:
                # randomly pick a side
                if self.rng.random() < 0.5:
                    trajectories[i] = left_path
                else:
                    trajectories[i] = right_path

        if noise_std > 0:
            trajectories = self.add_sensory_noise(trajectories, noise_std)

        return trajectories
