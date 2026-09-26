import numpy as np
from typing import Optional

class PureToneEncoder:
    """
    Sensory encoder that projects low-dimensional (1D) pure tone frequencies 
    into a sparse, high-dimensional representation (Frequency Bins).
    
    This simulates an early auditory cortical representation mapping continuous audio
    frequencies into a set of discrete neuronal channels before reaching the hippocampus.
    """

    def __init__(self, n_bins: int = 50, min_freq: float = 200.0, max_freq: float = 600.0, 
                 sigma: float = 15.0, threshold: float = 0.01, seed: Optional[int] = None):
        """
        Initialize the Pure Tone Encoder.

        Args:
            n_bins (int): Resolution of the frequency tiling.
            min_freq (float): Minimum frequency bound.
            max_freq (float): Maximum frequency bound.
            sigma (float): Width (radius) of each frequency bin receptive field.
            threshold (float): Activations below this value are clipped to zero.
            seed (int, optional): Random seed.
        """
        self.n_cells = n_bins
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sigma = sigma
        self.threshold = threshold
        
        # Initialize frequency centers evenly across the frequency space
        self.cell_centers = np.linspace(min_freq, max_freq, n_bins)
        self.rng = np.random.default_rng(seed)

    def encode(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Translates a 1D frequency trajectory into high-dimensional sparse activations.

        Args:
            trajectory: Array of shape (seq_len, 1) or (batch, seq_len, 1).
        
        Returns:
            np.ndarray: Sparse activations of shape (..., n_cells).
        """
        # Distances from each point in trajectory to every frequency center
        # Reshape for broadcasting
        points = trajectory[..., np.newaxis] # (..., 1, 1)
        centers = self.cell_centers # (n_cells)
        
        # Calculate squared differences (..., n_cells)
        dist_sq = (points[..., 0] - centers)**2
        
        # Gaussian activation
        activations = np.exp(-dist_sq / (2 * self.sigma**2))
        
        # Enforce sparsity
        activations[activations < self.threshold] = 0.0
        
        return activations

    def decode(self, activations: np.ndarray) -> np.ndarray:
        """
        Decodes high-dimensional activations back to 1D frequency 
        using a weighted center-of-mass approach.

        Args:
            activations: Array of shape (..., n_cells).
        
        Returns:
            np.ndarray: Reconstructed 1D frequency trajectory (..., 1).
        """
        total_activation = np.sum(activations, axis=-1, keepdims=True)
        # Avoid division by zero
        total_activation[total_activation == 0] = 1.0
        
        weighted_sum = np.sum(activations * self.cell_centers, axis=-1, keepdims=True)
        
        return weighted_sum / total_activation
