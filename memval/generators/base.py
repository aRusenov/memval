from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class SequenceGenerator(ABC):
    """
    Abstract base class for all sequence generators in MemVal.
    
    A SequenceGenerator is responsible for creating synthetic episodic
    memory sequences with configurable structural properties such as
    length, inter-trial intervals, and specific noise profiles.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the sequence generator.

        Args:
            seed (int, optional): Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self, n_sequences: int, sequence_length: int, **kwargs) -> Any:
        """
        Generate a batch of sequences.

        Args:
            n_sequences (int): Number of independent sequences to generate.
            sequence_length (int): Length (number of events or time steps) per sequence.
            **kwargs: Modality-specific configuration parameters (e.g., noise_variance, overlap).

        Returns:
            Any: The generated data representation. Usually a numpy array, dict, or list
                 of sequence data.
        """
        pass

    def add_sensory_noise(self, data: np.ndarray, std: float) -> np.ndarray:
        """
        Inject Gaussian sensory noise into the generated sequence data.

        Args:
            data (np.ndarray): The clean sequence data.
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: The noisy data.
        """
        if std <= 0:
            return data
        noise = self.rng.normal(loc=0.0, scale=std, size=data.shape)
        return data + noise
