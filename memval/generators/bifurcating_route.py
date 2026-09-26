import numpy as np
from typing import Optional, Dict, Any
from .base import SequenceGenerator
from ..encoders.spatial import PlaceCellEncoder
from ..benchmarks.overlap_config import OverlapConfig

class BifurcatingRouteGenerator(SequenceGenerator):
    """
    Generates a pair of 2D spatial trajectories that share a configurable common
    corridor, paired with a multi-modal encounter vector (representing odour or objects).
    Both spatial (MEC) and encounter (LEC) features are concatenated into a single
    input vector at each timestep.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def generate(
        self,
        total_length: int = 30,
        shared_fraction: float = 0.40,
        shared_position: float = 0.33,
        zone_fraction: float = 0.50,
        zone_offset: float = 0.25,
        encounter_similarity: float = 0.0,
        n_place_cells: int = 400,
        n_encounter_dims: int = 50,
        arena_bounds: float = 1.0,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
        balance_modalities: bool = False,
        odour_scale: float = 1.0,
        odour_on_suffix: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate two bifurcating routes A and B.
        
        Returns:
            Dict containing:
                - input_A: np.ndarray of shape (L, n_place_cells + n_encounter_dims)
                - input_B: np.ndarray of shape (L, n_place_cells + n_encounter_dims)
                - shared_start: int
                - shared_end: int
                - zone_start: int
                - zone_end: int
                - prefix_end: int
                - suffix_start: int
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        if n_encounter_dims < 2:
            raise ValueError("n_encounter_dims must be at least 2 to support similarity rotation.")

        # Resolve config geometries
        config = OverlapConfig(
            total_length=total_length,
            shared_fraction=shared_fraction,
            shared_position=shared_position,
            zone_fraction=zone_fraction,
            zone_offset=zone_offset,
            encounter_similarity=encounter_similarity
        )
        geom = config.resolve()
        ss = geom["shared_start"]
        se = geom["shared_end"]
        zs = geom["zone_start"]
        ze = geom["zone_end"]

        # Calculate lengths
        prefix_len = ss
        shared_len = se - ss
        suffix_len = total_length - se

        # Define 2D coordinates for the routes
        # Shared corridor runs vertically along y-axis in the center
        entry_point = np.array([0.0, -0.2 * arena_bounds])
        exit_point = np.array([0.0, 0.2 * arena_bounds])
        
        # Start points of prefixes
        start_A = np.array([-0.5 * arena_bounds, -0.7 * arena_bounds])
        start_B = np.array([0.5 * arena_bounds, -0.7 * arena_bounds])
        
        # End points of suffixes
        end_A = np.array([-0.5 * arena_bounds, 0.7 * arena_bounds])
        end_B = np.array([0.5 * arena_bounds, 0.7 * arena_bounds])

        # Interpolate Prefix paths
        if prefix_len > 0:
            prefix_A = np.linspace(start_A, entry_point, prefix_len, endpoint=False)
            prefix_B = np.linspace(start_B, entry_point, prefix_len, endpoint=False)
        else:
            prefix_A = np.zeros((0, 2))
            prefix_B = np.zeros((0, 2))

        # Interpolate Shared corridor path
        if shared_len > 0:
            shared = np.linspace(entry_point, exit_point, shared_len, endpoint=False)
        else:
            shared = np.zeros((0, 2))

        # Interpolate Suffix paths
        if suffix_len > 0:
            suffix_A = np.linspace(exit_point, end_A, suffix_len + 1)[1:]
            suffix_B = np.linspace(exit_point, end_B, suffix_len + 1)[1:]
        else:
            suffix_A = np.zeros((0, 2))
            suffix_B = np.zeros((0, 2))

        # Concatenate paths
        path_A = np.vstack([prefix_A, shared, suffix_A])
        path_B = np.vstack([prefix_B, shared, suffix_B])

        # Apply noise if requested
        if noise_std > 0:
            path_A = self.add_sensory_noise(path_A, noise_std)
            path_B = self.add_sensory_noise(path_B, noise_std)

        # Encode coordinates to place cells
        n_cells_per_dim = int(np.round(np.sqrt(n_place_cells)))
        # Adjust n_place_cells if not a perfect square
        encoder = PlaceCellEncoder(
            n_cells_per_dim=n_cells_per_dim,
            env_bounds=arena_bounds,
            seed=seed
        )
        
        pc_A = encoder.encode(path_A)
        pc_B = encoder.encode(path_B)

        # Build encounter/odour vectors (LEC)
        # Cosine similarity rotation
        rho = min(1.0, max(0.0, encounter_similarity))
        theta = np.arccos(rho)
        
        enc_vec_A = np.zeros(n_encounter_dims)
        enc_vec_A[0] = 1.0 # Unit vector along dimension 0
        
        enc_vec_B = np.zeros(n_encounter_dims)
        enc_vec_B[0] = np.cos(theta)
        enc_vec_B[1] = np.sin(theta)

        # Create timeseries of encounters
        lec_A = np.zeros((total_length, n_encounter_dims))
        lec_B = np.zeros((total_length, n_encounter_dims))
        
        # Populate odour zone
        if ze > zs:
            lec_A[zs:ze, :] = enc_vec_A
            lec_B[zs:ze, :] = enc_vec_B

        # odour_on_suffix carries the discriminator through the arms as well as
        # the zone. This is the *degenerate* case and exists to reproduce the
        # shipped T-maze corner as a reference: with the odour present at and
        # after the branch step, the probe re-supplies the discriminating cue at
        # the moment of the decision, so the task is solved by concurrent
        # binding and needs no memory of the corridor at all. Leave it False for
        # any row that is meant to test carrying the cue across a delay.
        if odour_on_suffix:
            lec_A[se:, :] = enc_vec_A
            lec_B[se:, :] = enc_vec_B

        # Concatenate MEC + LEC to get model inputs
        input_A = np.concatenate([pc_A, lec_A], axis=-1)
        input_B = np.concatenate([pc_B, lec_B], axis=-1)

        # Optional matched-magnitude scaling, mirroring
        # TMazeDisambiguationGenerator. Unbalanced, the place block carries a
        # per-step norm of ~2.1 against the odour's 1.0, so the discriminating
        # cue enters at under half the weight of the signal it has to override
        # -- an arm can then fail the task for a scaling reason rather than a
        # memory one. With balance_modalities=True each block is normalised
        # per-timestep to a common norm (place -> 1.0, odour -> odour_scale).
        # The place-cell decoder is a scale-invariant centre of mass, so
        # per-timestep place normalisation does not distort decoded
        # coordinates. Default False to leave existing callers untouched.
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
            "shared_start": ss,
            "shared_end": se,
            "zone_start": zs,
            "zone_end": ze,
            "prefix_end": ss,
            "suffix_start": se,
            "encoder": encoder
        }
