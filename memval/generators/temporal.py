import numpy as np
from typing import Optional

from .base import SequenceGenerator


class PureToneSequenceGenerator(SequenceGenerator):
    """
    Generates deterministic pure tone temporal trajectories representing
    a sequence of frequency transitions (e.g. an ascending musical scale or arpeggio).
    This serves as a starting benchmark for the audio modality.
    """

    def __init__(self, base_frequencies: list = [261.63, 293.66, 329.63, 349.23, 392.00], seed: Optional[int] = None):
        """
        Initialize the Pure Tone sequence generator.

        Args:
            base_frequencies (list): Target frequencies to sequence through.
            seed (int, optional): Random seed.
        """
        super().__init__(seed)
        self.base_frequencies = base_frequencies

    def generate(self, n_sequences: int, sequence_length: int, 
                 noise_std: float = 0.0) -> np.ndarray:
        """
        Generate continuous frequency sequences interpolating between base frequencies.

        Args:
            n_sequences (int): Batch size.
            sequence_length (int): Total number of discrete points to sample.
            noise_std (float): Gaussian noise variance on the frequencies.

        Returns:
            np.ndarray: Trajectories of shape (n_sequences, sequence_length, 1).
        """
        # Determine how many points per transition
        n_targets = len(self.base_frequencies)
        if n_targets < 2:
            raise ValueError("Need at least 2 base frequencies to create a sequence.")

        # Account for the final exact target point
        actual_seq_len = sequence_length - 1
        points_per_segment = actual_seq_len // (n_targets - 1)
        remaining_points = actual_seq_len % (n_targets - 1)

        # Generate a continuous glissando/transition between targets
        segments = []
        for i in range(n_targets - 1):
            n_points = points_per_segment + (1 if i < remaining_points else 0)
            if n_points > 0:
                # Avoid duplicate points at the edges of segments
                seg = np.linspace(self.base_frequencies[i], self.base_frequencies[i+1], n_points + 1)[:-1]
                segments.append(seg)
        
        # Ensure we hit the exact final target
        segments.append(np.array([self.base_frequencies[-1]]))

        single_path = np.concatenate(segments)
        
        # Duplicate for n_sequences
        trajectories = np.tile(single_path, (n_sequences, 1))[..., np.newaxis]

        if noise_std > 0:
            trajectories = self.add_sensory_noise(trajectories, noise_std)

        return trajectories
