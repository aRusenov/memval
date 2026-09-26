import numpy as np
from typing import Optional, Dict, Any
from .base import SequenceGenerator
from .t_maze import TMazeGenerator
from ..encoders.spatial import PlaceCellEncoder

class TMazeDisambiguationGenerator(SequenceGenerator):
    """
    Wraps TMazeGenerator to produce a multi-modal route_pair dictionary
    expected by SpatialDisambiguationBenchmark.
    
    The central stem of the T-maze is treated as the shared corridor, where the
    odour signal is active. The arms are the unique suffixes.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def generate(
        self,
        sequence_length: int = 30,
        stem_fraction: float = 0.50,
        n_place_cells: int = 400,
        n_odour_dims: int = 20,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
        sigma_scale: float = 2.0,
        odour_on_arms: bool = False,
        balance_modalities: bool = False,
        odour_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generate T-maze trajectories and concatenate them with orthogonal odour features.
        
        Returns:
            Dict containing:
                - input_A: np.ndarray (L, n_place_cells + n_odour_dims)
                - input_B: np.ndarray (L, n_place_cells + n_odour_dims)
                - shared_start: 0
                - shared_end: stem_end
                - zone_start: 0
                - zone_end: stem_end
                - prefix_end: 0
                - suffix_start: stem_end
                - encoder: PlaceCellEncoder
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if n_odour_dims < 2:
            raise ValueError("n_odour_dims must be at least 2 to support orthogonal odour representations.")

        # Compute lengths of stem and decision arms
        stem_len_val = stem_fraction
        arm_len_val = 1.0 - stem_fraction
        
        # Instantiate TMazeGenerator to generate trajectories
        tmaze_gen = TMazeGenerator(
            stem_length=stem_len_val,
            arm_length=arm_len_val,
            seed=seed
        )
        
        # Determine exact stem points count
        total_len = stem_len_val + arm_len_val
        stem_end = max(2, int(sequence_length * (stem_len_val / total_len)))

        # Route A -> Left Turn
        traj_A = tmaze_gen.generate(n_sequences=1, sequence_length=sequence_length, turn_direction="left", noise_std=noise_std)[0]
        # Route B -> Right Turn
        traj_B = tmaze_gen.generate(n_sequences=1, sequence_length=sequence_length, turn_direction="right", noise_std=noise_std)[0]

        # Place cell encoder
        n_cells_per_dim = int(np.round(np.sqrt(n_place_cells)))
        # Determine max bound based on stem/arm coords (stem goes 0 to stem_length, arms to ±arm_length)
        # Using 1.0 as standard env_bounds
        encoder = PlaceCellEncoder(
            n_cells_per_dim=n_cells_per_dim,
            env_bounds=1.0,
            seed=seed,
            sigma_scale=sigma_scale
        )
        
        # Encode coordinates to place cell activations
        pc_A = encoder.encode(traj_A)
        pc_B = encoder.encode(traj_B)

        # Odour signals (LEC)
        odour_X = np.zeros(n_odour_dims)
        odour_X[0] = 1.0
        
        odour_Y = np.zeros(n_odour_dims)
        odour_Y[1] = 1.0

        lec_A = np.zeros((sequence_length, n_odour_dims))
        lec_B = np.zeros((sequence_length, n_odour_dims))

        # Odour extent. By default the odour is active only on the stem
        # (0 to stem_end). With odour_on_arms=True it stays on for the entire
        # sequence, so the discriminative cue persists through the arm rollout
        # rather than switching off at the decision point.
        odour_end = sequence_length if odour_on_arms else stem_end
        lec_A[0:odour_end, :] = odour_X
        lec_B[0:odour_end, :] = odour_Y

        # Concatenate MEC + LEC
        input_A = np.concatenate([pc_A, lec_A], axis=-1)
        input_B = np.concatenate([pc_B, lec_B], axis=-1)

        # Optional matched-magnitude scaling. The place (MEC) block has many
        # active cells while the odour (LEC) block is a single unit, so the two
        # modalities enter a model at very different effective magnitudes. When
        # balance_modalities=True, each block is normalised per-timestep to a
        # common norm (place -> unit norm, odour -> odour_scale) so neither
        # dominates by raw scale. The place-cell decoder is a scale-invariant
        # centre-of-mass, so per-timestep place normalisation does not distort
        # decoded coordinates.
        if balance_modalities:
            for X in (input_A, input_B):
                pc, od = X[:, :n_place_cells], X[:, n_place_cells:]
                pc_norm = np.linalg.norm(pc, axis=1, keepdims=True)
                od_norm = np.linalg.norm(od, axis=1, keepdims=True)
                pc_norm[pc_norm == 0] = 1.0
                od_norm[od_norm == 0] = 1.0
                X[:, :n_place_cells] = pc / pc_norm
                X[:, n_place_cells:] = od / od_norm * odour_scale

        return {
            "input_A": input_A,
            "input_B": input_B,
            "shared_start": 0,
            "shared_end": stem_end,
            "zone_start": 0,
            "zone_end": stem_end,
            "prefix_end": 0,
            "suffix_start": stem_end,
            "encoder": encoder
        }
