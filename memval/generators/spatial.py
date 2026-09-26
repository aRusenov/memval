from typing import Optional, Union

import numpy as np

from .base import SequenceGenerator


class SpatialSequenceGenerator(SequenceGenerator):
    """
    Generates multi-dimensional continuous spatial trajectories to simulate
    physical navigation environments where hippocampal place cells form.
    """

    def __init__(self, n_dimensions: int = 2, env_bounds: float = 1.0, seed: Optional[int] = None):
        """
        Initialize the spatial sequence generator.

        Args:
            n_dimensions (int): Dimensionality of the spatial arena (default 2D).
            env_bounds (float): Boundaries of the environment [-bounds, +bounds].
            seed (int, optional): Random seed.
        """
        super().__init__(seed)
        self.n_dimensions = n_dimensions
        self.env_bounds = env_bounds

    def generate(self, n_sequences: int, sequence_length: int, 
                 step_size: float = 0.1, momentum: float = 0.8,
                 noise_std: float = 0.0,
                 drift: Optional[Union[float, np.ndarray]] = None,
                 start_pos: Optional[Union[np.ndarray, str]] = None) -> np.ndarray:
        """
        Produce batch trajectories simulating smooth movement over time.

        It uses a momentum-based random walk to create somewhat realistic 
        continuous trajectories rather than completely disjointed points.

        Args:
            n_sequences (int): Batch size of walks to simulate.
            sequence_length (int): Timesteps per walk ($L$).
            step_size (float): Maximum allowed shift per timestep.
            momentum (float): Factor [0-1] controlling the smoothness of the turn.
            noise_std (float): Gaussian noise applied to the final trajectory outputs.
            drift (float or np.ndarray, optional): Constant directional bias (drift) 
                added at each step to increase dispersion and prevent the trajectory 
                from folding back on itself. If a float is passed, it represents the 
                magnitude of a random persistent direction vector. If a numpy array 
                is passed, it specifies the exact drift direction.
            start_pos (np.ndarray or str, optional): Starting position for the trajectories.
                If 'center', all sequences start at the origin (all zeros). If a numpy 
                array is passed, it specifies the exact starting coordinates. If None 
                (default), random start positions within the environment boundaries are generated.

        Returns:
            np.ndarray: Trajectory batch of shape (n_sequences, sequence_length, n_dimensions).
        """
        trajectories = np.zeros((n_sequences, sequence_length, self.n_dimensions))
        
        # Start at specified or random positions within the environment
        if start_pos is not None:
            if isinstance(start_pos, str) and start_pos == "center":
                current_pos = np.zeros((n_sequences, self.n_dimensions))
            else:
                current_pos = np.array(start_pos, dtype=float).copy()
                if current_pos.ndim == 1:
                    current_pos = np.tile(current_pos, (n_sequences, 1))
        else:
            current_pos = self.rng.uniform(-self.env_bounds, self.env_bounds, size=(n_sequences, self.n_dimensions))
        
        trajectories[:, 0, :] = current_pos
        
        # Initial velocities
        velocities = np.zeros((n_sequences, self.n_dimensions))

        # Handle drift direction (persistent bias)
        if drift is not None:
            if isinstance(drift, (int, float)):
                if self.n_dimensions == 2:
                    angles = self.rng.uniform(0, 2 * np.pi, size=n_sequences)
                    drift_vectors = np.column_stack([np.cos(angles), np.sin(angles)]) * drift
                else:
                    # Generic unit vector generation for d dimensions
                    raw = self.rng.normal(size=(n_sequences, self.n_dimensions))
                    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
                    norms[norms == 0] = 1.0
                    drift_vectors = (raw / norms) * drift
            else:
                drift_vectors = np.atleast_2d(drift)
        else:
            drift_vectors = np.zeros((n_sequences, self.n_dimensions))

        for t in range(1, sequence_length):
            # Update velocities stochastically but preserve momentum and apply drift bias
            random_forces = self.rng.uniform(-step_size, step_size, size=(n_sequences, self.n_dimensions))
            velocities = momentum * velocities + (1 - momentum) * (random_forces + drift_vectors)
            
            # Apply velocities
            current_pos += velocities
            
            # Bound handling (elastic bounce off walls)
            out_of_bounds = np.abs(current_pos) > self.env_bounds
            velocities[out_of_bounds] *= -1.0 # Bounce back
            current_pos = np.clip(current_pos, -self.env_bounds, self.env_bounds)
            
            trajectories[:, t, :] = current_pos

        if noise_std > 0:
            trajectories = self.add_sensory_noise(trajectories, noise_std)

        return trajectories
