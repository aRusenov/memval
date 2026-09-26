from typing import List, Optional

import numpy as np

from .base import SequenceGenerator


class ObjectArenaGenerator(SequenceGenerator):
    """
    Generates trajectories where an agent navigates a 2D arena and 
    encounters specific objects at defined locations.
    The state vector emitted is [x, y, obj_1, obj_2, ..., obj_k] 
    where the object dimensions are one-hot encoded when interacting.
    """

    def __init__(self, n_objects: int = 3, n_locations: int = 2, 
                 env_bounds: float = 1.0, seed: Optional[int] = None):
        """
        Initialize the object arena.

        Args:
            n_objects (int): Total number of unique objects available in the dictionary.
            n_locations (int): Number of interaction waypoints in the arena.
            env_bounds (float): Spatial boundaries of the environment.
            seed (int, optional): Random seed.
        """
        super().__init__(seed)
        self.n_objects = n_objects
        self.n_locations = n_locations
        self.env_bounds = env_bounds
        
        # Pre-define fixed evenly-spaced locations along a path (e.g. x-axis)
        self.locations = []
        for i in range(n_locations):
            x = -0.5 * env_bounds + (env_bounds * i / max(1, n_locations - 1))
            self.locations.append(np.array([x, 0.0]))
        self.locations = np.array(self.locations)

    def generate(self, n_sequences: int, sequence_length: int, 
                 object_mapping: List[Optional[int]], 
                 interaction_radius: float = 0.2,
                 step_size: float = 0.05,
                 noise_std: float = 0.0) -> np.ndarray:
        """
        Generates navigation paths through the predefined locations, 
        activating object features when near a populated location.

        Args:
            n_sequences (int): Number of sequences.
            sequence_length (int): Timesteps per sequence.
            object_mapping (List[Optional[int]]): Maps location index to an object ID (0 to n_objects-1).
            interaction_radius (float): Distance threshold to activate an object.
            step_size (float): Movement speed per timestep.
            noise_std (float): Gaussian noise applied to state.

        Returns:
            np.ndarray: Trajectory batches of shape (n_sequences, sequence_length, 2 + n_objects).
        """
        if len(object_mapping) != self.n_locations:
            raise ValueError(f"object_mapping must have length {self.n_locations}")

        features_dim = 2 + self.n_objects
        trajectories = np.zeros((n_sequences, sequence_length, features_dim))
        
        for i in range(n_sequences):
            # Start at left edge
            current_pos = np.array([-self.env_bounds, 0.0])
            current_target_idx = 0
            
            for t in range(sequence_length):
                # Simple pathing: move towards current target location, then next
                if current_target_idx < self.n_locations:
                    target = self.locations[current_target_idx]
                    direction = target - current_pos
                    dist = np.linalg.norm(direction)
                    
                    if dist < step_size:
                        # Snap to target and switch to next
                        current_pos = target.copy()
                        current_target_idx += 1
                    else:
                        current_pos += (direction / dist) * step_size
                else:
                    # Once all locations visited, just keep moving right
                    current_pos += np.array([step_size, 0.0])
                
                # Add behavioral jitter to pathing
                pos_jitter = current_pos + self.rng.normal(0, step_size * 0.2, size=2)
                trajectories[i, t, 0:2] = pos_jitter
                
                # Check object interactions
                one_hot_objects = np.zeros(self.n_objects)
                for loc_idx, loc in enumerate(self.locations):
                    if np.linalg.norm(current_pos - loc) <= interaction_radius:
                        obj_id = object_mapping[loc_idx]
                        if obj_id is not None:
                            one_hot_objects[obj_id] = 1.0
                
                trajectories[i, t, 2:] = one_hot_objects
                
        if noise_std > 0:
            trajectories = self.add_sensory_noise(trajectories, noise_std)
            
        return trajectories
