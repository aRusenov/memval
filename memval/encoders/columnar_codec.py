"""Columnar (interval-coded) spike codec: R^D <-> one minicolumn per hypercolumn.

The population codec (``spike_codec.py``) is a 2-interval code per feature --
ON or OFF -- with a magnitude-graded rate. Columnar attractor models (Tully et
al. 2016 and the BCPNN family) represent a continuous variable the other way
round: the S1 Appendix of that paper calls it "a discrete coded or interval
coded continuous variable" -- one hypercolumn per variable, one minicolumn per
*interval* of its value, exactly one minicolumn active, at a flat rate. This
codec is that code, so such an arm can be run on the suite's own material in
the format its paper prescribes.

Why it exists is measured, not argued: on the population codec the spiking
BCPNN arm learns in-item weights of 0.36 nS against its paper's ~4 nS and
scores at chance; on this code it learns the paper's weights and scores 1.00
on the same seven words (``docs/capacity_report_bcpnn.md`` sec 1-1b).

The setting is **fixed from the paper's side and never re-sited per section**
(the distinction in that report between a fair presentation and a fitted
one):

* ``n_hc = D``: one hypercolumn per embedding dimension. No projection, so
  the round trip loses only the quantisation.
* ``n_mc = 10``: the paper's minicolumns per hypercolumn, hence its chance
  overlap of 1/10 between unrelated items.
* ``n_per_mc = 3``: an item drives ``3 D = 300`` cells at ``D = 100``; the
  paper's pattern is 270.
* ``rate = 200 Hz`` for ``window_steps = 100`` ms: the paper's stimulus.
* bins: equal width over ``+- z_range`` standard deviations of a coordinate
  of an isotropic unit-norm vector (``1/sqrt(D)``), clipped at the ends. A
  fixed reference frame, not the item set's range -- items that happen to
  cluster share bins, and that is their overlap, honestly represented.

Layout ``(d, mc, k) -> (d * n_mc + mc) * n_per_mc + k`` is the arm's own
``(hypercolumn, minicolumn, cell)`` order, so the codec's neuron ``i`` is the
network's pyramidal cell ``i`` with no permutation.

The decoder is the inverse of the code, not a learned readout: per
hypercolumn, the spike-count-weighted mean of the active minicolumns' bin
centres, mapped back to value units and L2-normalised. It never returns a
symbol (spec S2.1); the benchmark's own decoders rank the vector.

Invariants shared with the population codec: explicit seeding and a private
generator (no global RNG), sign preserved, byte-identical output for the same
seed.
"""

from typing import Optional

import numpy as np


class ColumnarSpikeEncoder:
    """R^D -> ``(n_neurons, window_steps)`` boolean spike array."""

    def __init__(self, embedding_dim: int, n_mc: int = 10, n_per_mc: int = 3,
                 window_steps: int = 100, dt_ms: float = 1.0, rate: float = 200.0,
                 z_range: float = 2.5, seed: Optional[int] = None):
        if embedding_dim < 1 or n_mc < 2 or n_per_mc < 1 or window_steps < 1:
            raise ValueError("embedding_dim >= 1, n_mc >= 2, n_per_mc >= 1, window_steps >= 1.")
        if z_range <= 0 or rate <= 0:
            raise ValueError("z_range and rate must be > 0.")
        self.embedding_dim = int(embedding_dim)
        self.n_hc = self.embedding_dim
        self.n_mc = int(n_mc)
        self.n_per_mc = int(n_per_mc)
        self.window_steps = int(window_steps)
        self.dt_ms = float(dt_ms)
        self.rate = float(rate)
        self.z_range = float(z_range)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        #: A coordinate of an isotropic unit-norm vector has SD 1/sqrt(D); bins
        #: are laid out in those units so the code does not depend on D.
        self.scale = float(np.sqrt(self.embedding_dim))
        #: Bin centres in value units, ``(n_mc,)``.
        self.centers = ((np.arange(self.n_mc) + 0.5) / self.n_mc * 2.0 * self.z_range
                        - self.z_range) / self.scale

    # ------------------------------------------------------------------ meta
    @property
    def n_neurons(self) -> int:
        return self.n_hc * self.n_mc * self.n_per_mc

    @property
    def spikes_per_unit(self) -> float:
        """Expected spikes from one active minicolumn over the window."""
        return self.n_per_mc * self.window_steps * self.rate * self.dt_ms / 1000.0

    @property
    def params(self) -> dict:
        return {"codec": "columnar", "embedding_dim": self.embedding_dim,
                "n_hc": self.n_hc, "n_mc": self.n_mc, "n_per_mc": self.n_per_mc,
                "window_steps": self.window_steps, "dt_ms": self.dt_ms,
                "rate": self.rate, "z_range": self.z_range,
                "n_neurons": self.n_neurons, "seed": self.seed}

    def describe(self) -> str:
        return (f"ColumnarSpikeEncoder(D={self.embedding_dim}, n_mc={self.n_mc}, "
                f"n_per_mc={self.n_per_mc}, T={self.window_steps}, rate={self.rate:g}Hz, "
                f"z_range={self.z_range:g}, N={self.n_neurons}, seed={self.seed})")

    def reset_rng(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ code
    def bins(self, vector: np.ndarray) -> np.ndarray:
        """``(D,)`` -> ``(D,)`` active minicolumn per hypercolumn."""
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.shape[0] != self.embedding_dim:
            raise ValueError(f"expected {self.embedding_dim} features, got {v.shape[0]}.")
        z = v * self.scale
        b = np.floor((z + self.z_range) / (2.0 * self.z_range) * self.n_mc)
        return np.clip(b, 0, self.n_mc - 1).astype(int)

    def cells(self, bins: np.ndarray) -> np.ndarray:
        """Cell indices of the pattern with minicolumn ``bins[d]`` active in hypercolumn d."""
        base = (np.arange(self.n_hc) * self.n_mc + np.asarray(bins)) * self.n_per_mc
        return (base[:, None] + np.arange(self.n_per_mc)[None, :]).ravel()

    def encode(self, vector: np.ndarray,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        gen = self.rng if rng is None else rng
        S = np.zeros((self.n_neurons, self.window_steps), dtype=bool)
        c = self.cells(self.bins(vector))
        p = min(1.0, self.rate * self.dt_ms / 1000.0)
        S[c] = gen.random((c.size, self.window_steps)) < p
        return S

    def encode_stack(self, vectors: np.ndarray,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """``(L, D)`` -> ``(L, n_neurons, window_steps)``."""
        X = np.atleast_2d(np.asarray(vectors, dtype=float))
        return np.stack([self.encode(x, rng=rng) for x in X])


class ColumnarSpikeDecoder:
    """Spike array -> R^D, unit-norm; the inverse of the code."""

    def __init__(self, encoder: ColumnarSpikeEncoder):
        self.encoder = encoder

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    def counts(self, spikes: np.ndarray) -> np.ndarray:
        """``(n_neurons, T')`` -> ``(D, n_mc)`` spike totals per minicolumn."""
        S = np.asarray(spikes)
        if S.ndim == 1:
            S = S[:, None]
        e = self.encoder
        if S.shape[0] != e.n_neurons:
            raise ValueError(f"expected {e.n_neurons} neurons, got {S.shape[0]}.")
        per_cell = S.astype(float).sum(axis=1)
        return per_cell.reshape(e.n_hc, e.n_mc, e.n_per_mc).sum(axis=2)

    def decode(self, spikes: np.ndarray) -> np.ndarray:
        c = self.counts(spikes)                                   # (D, n_mc)
        total = c.sum(axis=1)
        v = np.zeros(self.embedding_dim)
        active = total > 0
        v[active] = (c[active] @ self.encoder.centers) / total[active]
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def decode_stack(self, spikes: np.ndarray) -> np.ndarray:
        S = np.asarray(spikes)
        return np.stack([self.decode(S[i]) for i in range(S.shape[0])])


def make_columnar_codec(embedding_dim: int, seed: int, **kwargs):
    enc = ColumnarSpikeEncoder(embedding_dim, seed=seed, **kwargs)
    return enc, ColumnarSpikeDecoder(enc)
