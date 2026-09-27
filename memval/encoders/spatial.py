import numpy as np
from typing import Optional, Union

class PlaceCellEncoder:
    """
    Sensory encoder that projects low-dimensional (2D) spatial coordinates 
    into a high-dimensional sparse representation (Place Cells).
    
    This simulates the dimensional expansion of the medial entorhinal cortex 
    and dentate gyrus.
    """

    def __init__(self, n_cells_per_dim: int = 20, env_bounds: float = 1.0, 
                 sigma: Optional[float] = None, sigma_scale: float = 2.0,
                 threshold: float = 0.01, seed: Optional[int] = None):
        """
        Initialize the Place Cell Encoder.

        Args:
            n_cells_per_dim (int): Resolution of the tiling (n x n grid).
            env_bounds (float): Spatial boundaries [-bounds, +bounds].
            sigma (float, optional): Width (radius) of each place cell field.
                If None (default), sigma is inferred automatically on the first
                call to encode() as ``sigma_scale * mean_step_size``, so that
                the field width tracks the actual temporal resolution of the
                trajectory being encoded.  Pass an explicit value to override
                this behaviour and keep sigma fixed across all encode calls.
            sigma_scale (float): Multiplier applied to the mean step size when
                sigma is inferred automatically (default 2.0).  The resulting
                sigma satisfies step/sigma ≈ 1/sigma_scale, giving enough
                overlap between consecutive place-cell patterns for the model
                to learn smooth transitions while still discriminating adjacent
                positions.  Values in [1.5, 3.0] are typically well-behaved.
            threshold (float): Activations below this value are clipped to zero.
            seed (int, optional): Random seed.
        """
        self.n_cells_per_dim = n_cells_per_dim
        self.n_cells = n_cells_per_dim ** 2
        self.env_bounds = env_bounds
        self._sigma_fixed = sigma   # None means auto-compute on first encode
        self.sigma = sigma          # May be updated by _infer_sigma()
        self.sigma_scale = sigma_scale
        self.threshold = threshold
        
        # Initialize cell centers evenly across the 2D space
        x_centers = np.linspace(-env_bounds, env_bounds, n_cells_per_dim)
        y_centers = np.linspace(-env_bounds, env_bounds, n_cells_per_dim)
        
        XX, YY = np.meshgrid(x_centers, y_centers)
        self.cell_centers = np.column_stack([XX.ravel(), YY.ravel()])
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_sigma(self, trajectory: np.ndarray) -> None:
        """Compute and store sigma from the mean Euclidean step size.

        This is called automatically inside encode() when no fixed sigma was
        provided.  The trajectory's step-size distribution sets the natural
        scale, and sigma_scale controls the overlap ratio between adjacent
        place-cell activation patterns.

        Args:
            trajectory: Array of shape (..., seq_len, n_dim).
        """
        # Compute step vectors along the time axis (second-to-last dim)
        steps = np.diff(trajectory, axis=-2)               # (..., seq_len-1, n_dim)
        step_norms = np.sqrt(np.sum(steps ** 2, axis=-1))  # (..., seq_len-1)
        mean_step = float(step_norms.mean())

        if mean_step > 0:
            self.sigma = self.sigma_scale * mean_step
        else:
            # Degenerate case (static trajectory): fall back to 5 % of env extent
            self.sigma = 0.05 * self.env_bounds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Translates a 2D trajectory into high-dimensional place cell activations.

        If sigma was not set at construction time, it is inferred from the
        mean step size of the provided trajectory before encoding.

        Args:
            trajectory: Array of shape (seq_len, 2) or (batch, seq_len, 2).
        
        Returns:
            np.ndarray: Sparse activations of shape (..., n_cells).
        """
        if self._sigma_fixed is None:
            self._infer_sigma(trajectory)

        # Distances from each point in trajectory to every cell center
        # Reshape for broadcasting
        points = trajectory[..., np.newaxis, :] # (..., 1, 2)
        centers = self.cell_centers # (n_cells, 2)
        
        # Calculate squared Euclidean distances (..., n_cells)
        dist_sq = np.sum((points - centers)**2, axis=-1)
        
        # Gaussian activation
        activations = np.exp(-dist_sq / (2 * self.sigma**2))
        
        # Enforce sparsity
        activations[activations < self.threshold] = 0.0
        
        return activations

    def decode(self, activations: np.ndarray, power: float = 1.0) -> np.ndarray:
        """
        Decodes high-dimensional activations back to 2D coordinates 
        using a weighted center-of-mass approach.

        Args:
            activations: Array of shape (..., n_cells).
            power: Exponent to sharpen activations (suppresses background noise).
        
        Returns:
            np.ndarray: Reconstructed 2D trajectory (..., 2).
        """
        if power != 1.0:
            activations = np.power(activations, power)

        # (..., n_cells, 1) * (1, n_cells, 2) sum over n_cells
        total_activation = np.sum(activations, axis=-1, keepdims=True)
        # Avoid division by zero
        total_activation[total_activation == 0] = 1.0
        
        weighted_sum = np.sum(activations[..., np.newaxis] * self.cell_centers, axis=-2)
        
        return weighted_sum / total_activation


def route_step(trajectory: np.ndarray) -> float:
    """Mean distance between consecutive positions on a route."""
    t = np.asarray(trajectory, dtype=float)
    return float(np.mean(np.linalg.norm(np.diff(t, axis=0), axis=1))) if len(t) > 1 else 0.0


def decode_floor(encoder, trajectory: np.ndarray) -> np.ndarray:
    """Per-position CLEAN round-trip error of this encoder/decoder pair.

    The tolerance a spatial criterion grants has to be read against this. On the
    15-point T-maze with ``n_cells_per_dim=10, sigma_scale=1.0`` the floor is
    ~0.000 along the stem (which lies on the cell grid's own axis) and ~0.05 on
    the arm, rising to 0.072 at the last position -- roughly HALF of an absolute
    ``eps=0.15`` is spent before a model contributes any error. Measured
    2026-09-23; see docs/models/model_table.md.
    """
    t = np.asarray(trajectory, dtype=float)
    return np.linalg.norm(encoder.decode(encoder.encode(t)) - t, axis=1)


def relative_eps(trajectory: np.ndarray, k: float = 1.05, encoder=None,
                 floor_margin: float = 0.0) -> float:
    """Position tolerance as a MULTIPLE OF THE ROUTE'S STEP, not an absolute.

    ``eps`` was fixed at 0.15 while the T-maze keeps a constant physical extent
    whatever its point count, so the step scales as 1/L: 0.143 at 15 points
    (eps ~= one step) but 0.200 at 11 points (eps = 0.75 of a step, i.e. the
    criterion silently demands sub-step precision). Any change of route length
    then reads as a change in capability. ``k=1.05`` reproduces the historical
    0.15 at 15 points.

    With ``encoder`` given, the pair's own clean-decode floor is added on top
    (scaled by ``floor_margin``), so the criterion can be stated as "the model
    adds no more than k steps beyond what the decoder itself loses".
    """
    eps = float(k) * route_step(trajectory)
    if encoder is not None and floor_margin:
        eps += float(floor_margin) * float(np.max(decode_floor(encoder, trajectory)))
    return eps
