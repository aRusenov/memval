"""
Two-Weight CA3 prototype — auto-associative attractors + hetero-associative
directed transitions in one recurrent layer.

This is an *exploratory* prototype for the EP + theta-phase direction
(see docs/ep_theta_phase_exploration.md, §"two-weight system"). It is NOT yet a
HippocampalModel — it exists to test one question: do a symmetric attractor
matrix, an asymmetric transition matrix, and spike-frequency adaptation together
produce a clean autonomous A->B->C sweep?

Two weight matrices, two roles, two rules
-----------------------------------------
  W_sym  (symmetric)  — auto-associative attractors ("place fields"). Stored by a
                        one-shot Hebbian (covariance) imprint per state — the fast,
                        BTSP-style path. EP-contrastive refinement is a future hook
                        (the imprint is the free/slow-weight learning target).
  W_asym (asymmetric) — hetero-associative directed links A->B. Learned by an STDP
                        / ordered-pair rule from the sequence order.

Recall dynamics (the "free phase" sweep)
----------------------------------------
    rec_i  = Σ_j W_sym[i,j] s_j          # attractor cleanup (auto-association)
           + λ Σ_j W_asym[i,j] tr_j      # directional push from the trace (hetero)
    s     <- clip(s + dt(-s + rec - θ - adapt), 0, 1)   # θ = soft-kWTA inhibition
    adapt <- adapt + dt_a (a_gain·s - adapt)            # fatigue → release current state
    tr    <- tr + dt_tr (s - tr)                        # eligibility trace

The asymmetric term enters as an external *field* (via the trace), not a symmetric
coupling, so the symmetric energy — and hence EP's validity for W_sym — is
preserved. Adaptation is what makes the trajectory *advance* through attractors
(a heteroclinic tour) rather than lock into the cue; W_sym cleanup is what keeps
each step on a stored pattern instead of drifting off-manifold (the learned
version of the "quantized feedback" cleanup used in measure_recall_autoregressive).
"""

from typing import List, Optional, Tuple

import numpy as np


class ThetaTwoWeightCA3:
    def __init__(
        self,
        n_units: int = 400,
        sparsity: float = 0.05,
        dt: float = 0.5,
        lam: float = 1.0,          # strength of the asymmetric (directional) push
        a_gain: float = 1.0,       # adaptation gain (how strongly activity fatigues)
        dt_a: float = 0.06,        # adaptation timescale (slow → releases states gradually)
        dt_tr: float = 0.5,        # eligibility-trace timescale
        seed: Optional[int] = 0,
    ):
        self.N = n_units
        self.a = sparsity
        self.k = max(1, int(round(sparsity * n_units)))  # target #active units
        self.dt = dt
        self.lam = lam
        self.a_gain = a_gain
        self.dt_a = dt_a
        self.dt_tr = dt_tr
        self.rng = np.random.default_rng(seed)

        self.W_sym = np.zeros((n_units, n_units))
        self.W_asym = np.zeros((n_units, n_units))
        self.patterns: List[np.ndarray] = []   # stored state prototypes (binary)

    # ------------------------------------------------------------------
    # Pattern bank
    # ------------------------------------------------------------------
    def make_patterns(self, n_states: int) -> np.ndarray:
        """Random sparse binary patterns (DG-code-like), one per state."""
        pats = []
        for _ in range(n_states):
            idx = self.rng.choice(self.N, size=self.k, replace=False)
            p = np.zeros(self.N)
            p[idx] = 1.0
            pats.append(p)
        self.patterns = pats
        return np.array(pats)

    # ------------------------------------------------------------------
    # Learning: W_sym (auto-assoc, one-shot Hebbian) + W_asym (STDP, ordered pairs)
    # ------------------------------------------------------------------
    def imprint_state(self, xi: np.ndarray) -> None:
        """One-shot covariance (sparse-Hopfield) imprint of an attractor into W_sym."""
        c = xi - self.a                      # center by sparsity (covariance rule)
        self.W_sym += np.outer(c, c)
        np.fill_diagonal(self.W_sym, 0.0)

    def learn_sequence(self, seq_patterns: np.ndarray, n_presentations: int = 5,
                       stdp_lr: float = 1.0) -> None:
        """Store each state as an attractor (W_sym, once) and imprint the directed
        transitions into W_asym via an ordered-pair (STDP) rule, repeated over
        presentations (theta cycles)."""
        # W_sym: one-shot per distinct state (attractors don't need repetition)
        for xi in seq_patterns:
            self.imprint_state(xi)
        # normalize so a matched cue produces an O(1) recurrent drive on its active
        # units: W_sym @ xi on an active unit ~ k*(1-a)^2, so divide by k.
        self.W_sym /= self.k

        # W_asym: pre-before-post directed imprint, accumulated over presentations
        for _ in range(n_presentations):
            for t in range(len(seq_patterns) - 1):
                pre = seq_patterns[t] - self.a
                post = seq_patterns[t + 1] - self.a
                self.W_asym += stdp_lr * np.outer(post, pre)   # post <- pre (directed)
        np.fill_diagonal(self.W_asym, 0.0)
        # same O(1) normalization: W_asym @ (state) pushes its successor at ~unit gain
        self.W_asym /= (n_presentations * self.k)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _kwta_threshold(self, drive: np.ndarray) -> float:
        """Soft k-WTA: threshold at the k-th strongest drive → keep ~k units active."""
        if self.k >= self.N:
            return 0.0
        return np.partition(drive, -self.k)[-self.k]

    def _kwta_target(self, drive: np.ndarray) -> np.ndarray:
        """Hard k-WTA: the top-k units by drive are the active set (binary target).
        Relaxing s toward this keeps attractor states crisp (amplitude ~1) instead
        of collapsing to the small residual left by threshold subtraction."""
        theta = self._kwta_threshold(drive)
        return (drive >= theta).astype(float)

    def settle_cue(self, cue: np.ndarray, steps: int = 30) -> np.ndarray:
        """Auto-association only (W_sym, no push, no adaptation): clean a noisy cue
        toward the nearest stored attractor."""
        s = np.clip(cue.astype(float).copy(), 0.0, 1.0)
        for _ in range(steps):
            target = self._kwta_target(self.W_sym @ s)
            s = s + self.dt * (target - s)
        return s

    def recall(self, cue: np.ndarray, n_steps: int = 60,
               settle_substeps: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Free-phase sweep with TIMESCALE SEPARATION. Each macro-step: (i) settle
        for `settle_substeps` fast steps so the state cleans up onto an attractor
        given the current adaptation + directional push, then (ii) apply the slow
        adaptation and trace updates. Fast cleanup keeps every step on a stored
        pattern; slow adaptation is what advances the trajectory to the successor.
        Returns (states [n_steps, N], overlaps [n_steps, n_stored])."""
        s = np.clip(cue.astype(float).copy(), 0.0, 1.0)
        adapt = np.zeros(self.N)
        tr = s.copy()
        P = np.array(self.patterns)
        Pc = P - self.a                        # centered prototypes for overlap
        norms = np.sum(Pc * Pc, axis=1) + 1e-9

        states = np.zeros((n_steps, self.N))
        overlaps = np.zeros((n_steps, len(self.patterns)))
        for t in range(n_steps):
            # (i) fast settle onto the nearest attractor under push + adaptation
            for _ in range(settle_substeps):
                drive = self.W_sym @ s + self.lam * (self.W_asym @ tr) - adapt
                target = self._kwta_target(drive)
                s = s + self.dt * (target - s)
            # (ii) slow updates: fatigue the current state, advance the trace
            adapt = adapt + self.dt_a * (self.a_gain * s - adapt)
            tr = tr + self.dt_tr * (s - tr)

            states[t] = s
            overlaps[t] = (Pc @ (s - self.a)) / norms
        return states, overlaps

    def decode_overlaps(self, overlaps: np.ndarray) -> List[int]:
        """Per-timestep nearest stored state (argmax overlap)."""
        return list(np.argmax(overlaps, axis=1))
