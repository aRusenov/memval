"""
Classic Symmetric Hopfield Network.

A faithful implementation of the original Hopfield (1982) associative memory
model, extended with per-neuron firing thresholds θ as discussed in the energy
function literature.

Architecture
------------
N fully-connected binary neurons with symmetric weights (W = Wᵀ, diag = 0).
No hidden layers — every unit is both a storage unit and an output unit.

Energy Function
---------------
The network is governed by a Lyapunov / energy function:

    E(s) = -½ Σᵢ≠ⱼ Wᵢⱼ sᵢ sⱼ  -  Σᵢ θᵢ sᵢ

where:
    s ∈ {-1, +1}ᴺ   — state vector (bipolar neuron activations)
    W                — symmetric weight matrix, zero diagonal
    θ ∈ ℝᴺ          — per-neuron firing threshold (bias)

Because W is symmetric, every asynchronous state update is guaranteed to
either lower E or leave it unchanged (see _update_neuron for the proof sketch).
The network therefore always converges to a fixed point (local energy minimum).

Learning Rule
-------------
Patterns are stored via one-shot Hebbian learning (outer product rule):

    W ← W  +  (1/N) · p · pᵀ       (with diagonal forced to zero)

The (1/N) normalisation keeps weights from growing unboundedly as M increases.
Stored patterns become local minima of E because W = (1/N) Σ_μ p^μ (p^μ)ᵀ
implies  W p^a ≈ p^a  when patterns are approximately orthogonal.

Recall (State Update)
---------------------
Starting from a cue s₀ (possibly noisy or partial), neurons update one at a
time (asynchronously) using the rule:

    sᵢ ← sign( Σⱼ Wᵢⱼ sⱼ  +  θᵢ )

Each update satisfies ΔE ≤ 0 (proof: see _update_neuron docstring).
Iteration continues until no neuron changes state (convergence = fixed point).

Capacity
--------
Reliable recall holds for M < 0.138 · N stored patterns (Amit et al., 1985).
Beyond this, interference from overlapping patterns causes retrieval failure.

References
----------
Hopfield, J.J. (1982). Neural networks and physical systems with emergent
collective computational abilities. PNAS, 79(8), 2554–2558.

Amit, D.J., Gutfreund, H., Sompolinsky, H. (1985). Storing infinite numbers
of patterns in a spin-glass model of neural networks. PRL, 55(14), 1530.
"""

from typing import Optional

import numpy as np


class ClassicHopfieldNetwork:
    """
    Symmetric Hopfield associative memory with per-neuron firing thresholds.

    Neurons are bipolar: s ∈ {-1, +1}.
    External data in [0, 1] is converted to bipolar on store/recall and
    converted back on return so the interface stays consistent with the rest
    of the codebase.

    Parameters
    ----------
    n_neurons : int
        Number of neurons N.  All stored patterns must have this length.
    theta : float or array of shape (N,)
        Firing threshold(s).  A positive θᵢ makes neuron i harder to activate
        (it requires a stronger net input to flip to +1).
        Scalar → same threshold for every neuron.
    max_iter : int
        Maximum asynchronous update sweeps before declaring non-convergence.
    seed : int, optional
        Random seed for the asynchronous update order.
    """

    def __init__(
        self,
        n_neurons: int,
        theta: float | np.ndarray = 0.0,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ):
        self.n_neurons = n_neurons
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        # ── Weight matrix ─────────────────────────────────────────────────
        # W ∈ ℝᴺˣᴺ,  symmetric,  zero diagonal (no self-connections)
        # Initialised to zero; patterns are added via store().
        self.W: np.ndarray = np.zeros((n_neurons, n_neurons))

        # ── Firing thresholds θ ───────────────────────────────────────────
        # Appears in the energy as  -Σᵢ θᵢ sᵢ
        # Positive θᵢ  →  neuron i prefers the OFF (-1) state at rest.
        # Negative θᵢ  →  neuron i prefers the ON  (+1) state at rest.
        if np.isscalar(theta):
            self.theta = np.full(n_neurons, float(theta))
        else:
            self.theta = np.asarray(theta, dtype=float)
            assert self.theta.shape == (n_neurons,), \
                f"theta must be scalar or shape ({n_neurons},)"

        # Book-keeping
        self._n_stored: int = 0          # number of patterns stored so far
        self._patterns: list[np.ndarray] = []  # kept for diagnostics

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_bipolar(x: np.ndarray) -> np.ndarray:
        """Convert [0, 1] binary pattern to bipolar {-1, +1}."""
        return np.where(x > 0.5, 1.0, -1.0)

    @staticmethod
    def _to_binary(s: np.ndarray) -> np.ndarray:
        """Convert bipolar {-1, +1} back to binary [0, 1]."""
        return np.where(s > 0, 1.0, 0.0)

    # ──────────────────────────────────────────────────────────────────────
    # Energy
    # ──────────────────────────────────────────────────────────────────────

    def energy(self, s: np.ndarray) -> float:
        """
        Compute the Hopfield energy for state s.

        Formula
        -------
            E(s) = -½ sᵀ W s  -  θᵀ s

        The first term is the pairwise interaction energy (the ½ and the
        zero diagonal of W together ensure we don't double-count pairs and
        don't count self-interactions).

        The second term is the threshold/bias energy: a neuron with a large
        positive θᵢ lowers E by being in the ON state (+1), so it is
        energetically favourable for that neuron to fire.

        Parameters
        ----------
        s : (N,) bipolar state vector.

        Returns
        -------
        float : scalar energy value.  Lower is more stable.
        """
        interaction_term = -0.5 * s @ self.W @ s   # -½ sᵀ W s
        threshold_term   = -self.theta @ s          # -θᵀ s
        return float(interaction_term + threshold_term)

    # ──────────────────────────────────────────────────────────────────────
    # Learning
    # ──────────────────────────────────────────────────────────────────────

    def store(self, pattern: np.ndarray) -> None:
        """
        Store a single pattern using the Hebbian outer-product rule.

        Learning rule (for one pattern p ∈ {-1, +1}ᴺ):

            W ← W  +  (1/N) · p · pᵀ

        then the diagonal is zeroed:

            Wᵢᵢ ← 0  for all i           (no self-connections)

        Why (1/N)?
        ----------
        Without normalisation, weights grow as O(M) and the signal-to-noise
        ratio stays constant.  Dividing by N keeps the scale bounded.

        Why zero diagonal?
        ------------------
        Self-connections Wᵢᵢ would mean a neuron's state feeds back into
        itself directly, breaking the energy-decrease guarantee.

        Parameters
        ----------
        pattern : (N,) array, values in {0, 1} or {-1, +1}.
        """
        p = self._to_bipolar(pattern)       # ensure bipolar

        # Outer product:  p · pᵀ  has shape (N, N)
        # Entry [i, j] = pᵢ · pⱼ — large positive if both neurons fire
        # together, large negative if they fire opposite.
        self.W += (1.0 / self.n_neurons) * np.outer(p, p)

        # Zero the diagonal — no neuron reinforces itself
        np.fill_diagonal(self.W, 0.0)

        self._n_stored += 1
        self._patterns.append(p.copy())

    def store_all(self, patterns: np.ndarray) -> None:
        """
        Convenience wrapper: store every row of `patterns` as one pattern.

        Parameters
        ----------
        patterns : (M, N) array of M patterns.
        """
        for p in patterns:
            self.store(p)

    # ──────────────────────────────────────────────────────────────────────
    # Recall (iterative energy minimisation)
    # ──────────────────────────────────────────────────────────────────────

    def _update_neuron(self, s: np.ndarray, i: int) -> np.ndarray:
        """
        Asynchronously update neuron i and return the new state.

        Update rule:

            sᵢ ← sign( Σⱼ Wᵢⱼ sⱼ  +  θᵢ )

        Proof that ΔE ≤ 0
        -----------------
        Let hᵢ = Σⱼ Wᵢⱼ sⱼ + θᵢ  (the 'local field' seen by neuron i).

        Before the update: contribution of neuron i to E is  -sᵢ hᵢ
                                  (ignoring terms that don't involve i).
        After the update:  contribution is                   -sᵢ' hᵢ

        Since sᵢ' = sign(hᵢ), we have sᵢ' · hᵢ = |hᵢ| ≥ sᵢ · hᵢ.
        Therefore  ΔE = -(sᵢ' - sᵢ) hᵢ ≤ 0.  □

        The proof requires W to be symmetric (so hᵢ does not change when
        sᵢ changes — only neuron i is updated at a time).

        Parameters
        ----------
        s : current bipolar state vector.
        i : index of the neuron to update.
        """
        # Local field: net input to neuron i from all others, plus threshold
        #   hᵢ = Σⱼ Wᵢⱼ sⱼ  +  θᵢ
        h_i = self.W[i] @ s + self.theta[i]

        s_new = s.copy()
        s_new[i] = 1.0 if h_i >= 0 else -1.0   # sign(hᵢ)
        return s_new

    def recall(
        self,
        cue: np.ndarray,
        return_energy_trace: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[float]]:
        """
        Retrieve the stored pattern closest to `cue` via iterative settling.

        Starting from the cue, neurons are updated one at a time in a random
        order (asynchronous updates) until no neuron changes state.  Each
        update is guaranteed to lower (or preserve) the energy, so the
        process always terminates at a local energy minimum.

        Parameters
        ----------
        cue : (N,) array — the query, possibly noisy or partial.
        return_energy_trace : bool
            If True, also return a list of energy values after each full sweep
            (useful for verifying the energy-decrease property).

        Returns
        -------
        recalled : (N,) binary array in [0, 1] — the retrieved pattern.
        energy_trace : list of float (only if return_energy_trace=True).
        """
        s = self._to_bipolar(cue)
        energy_trace = [self.energy(s)] if return_energy_trace else []

        for _ in range(self.max_iter):
            s_prev = s.copy()

            # Asynchronous: visit every neuron once in a random order
            update_order = self.rng.permutation(self.n_neurons)
            for i in update_order:
                s = self._update_neuron(s, i)

            if return_energy_trace:
                energy_trace.append(self.energy(s))

            # Convergence: no neuron changed state during this sweep
            if np.array_equal(s, s_prev):
                break

        recalled = self._to_binary(s)

        if return_energy_trace:
            return recalled, energy_trace
        return recalled

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def overlap(self, s: np.ndarray, pattern_idx: int) -> float:
        """
        Compute the overlap m between state s and stored pattern μ.

        Definition:

            m^μ(s) = (1/N) Σᵢ p^μᵢ · sᵢ

        Interpretation
        --------------
        m = +1 : s is identical to pattern μ
        m =  0 : s is orthogonal to pattern μ (no correlation)
        m = -1 : s is the bitwise inverse of pattern μ

        The overlap is the order parameter used in statistical physics
        analyses of Hopfield networks to measure retrieval quality.
        """
        p = self._patterns[pattern_idx]
        s_bipolar = self._to_bipolar(s)
        return float(np.dot(p, s_bipolar) / self.n_neurons)

    @property
    def n_stored(self) -> int:
        """Number of patterns stored so far."""
        return self._n_stored

    @property
    def capacity_fraction(self) -> float:
        """M / (0.138 · N) — fraction of theoretical capacity used."""
        return self._n_stored / (0.138 * self.n_neurons)
