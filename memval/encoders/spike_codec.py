"""Population spike codec: R^D vectors <-> spike trains.

Implements ``docs/spike_codec_spec.md`` S3. The codec is the transport layer
that lets a spiking arm be scored on the *existing* symbolic sections without
changing a single metric, and -- equally important -- it is the thing whose own
cost is measurable, via the transport control in
``memval.models.codec_wrapper``.

Where it sits
-------------
::

    SymbolicEncoder / HierarchicalEncoder
            |  R^D, unit-norm
            v
       [ PROBE CORRUPTION HAPPENS HERE ]     <- noise, cue masking, partiality
            |
            v
       PopulationSpikeEncoder     (this module)
            |  (N neurons, T steps) bool
            v
       spiking model
            |  (N neurons, T steps) bool
            v
       PopulationSpikeDecoder     (this module)
            |  R^D, unit-norm
            v
       SymbolicDecoder            (unchanged)

Probe corruption stays **upstream** of the codec, in vector space, so every arm
-- rate and spiking -- sees the identical corrupted vector on the identical
probe grid. Corrupting downstream would quantise the probe grid to
``1 / n_per_feature`` and make spiking arms incomparable with rate arms. What is
at risk under upstream corruption is *sensitivity*, not definability: the codec
can wash a small corruption out before the model sees it. That is measured, not
assumed -- see ``bin/spike_codec_phase0.py``.

Constraints this module is written against (spec S2)
----------------------------------------------------
1. ``PopulationSpikeDecoder.decode`` returns a vector in R^D, never a symbol,
   so ``SymbolicDecoder`` and every metric behind it are untouched.
2. Output lands on the unit-norm manifold (the decoder L2-normalises last).
3. Sign is preserved, via separate ON and OFF populations. Spike rates cannot be
   negative and embeddings are signed; collapsing that is how
   ``divergence_margin`` got compressed for negative-output arms.
4. Explicit seeding. The codec owns a ``np.random.Generator`` and never touches
   the global RNG, so two runs with the same seed are byte-identical.
5. The codec is a stated condition, never an invisible default. Its parameters
   are exposed as ``params`` for figure captions and run records.

Substrate
---------
Prefer ``HierarchicalEncoder`` over ``SymbolicEncoder``. Its rows are sparse and
binary before normalisation, so most dimensions decode to *exactly* zero with no
Poisson variance at all, and the magnitude that is present sits in few
dimensions where it is large. ``SymbolicEncoder``'s dense unit-norm rows spread
the same norm over every dimension, giving each one a per-step spike probability
near the noise floor. The measured consequence is large: see the Phase 0 report.
"""

from typing import Optional, Sequence, Union

import numpy as np

#: Encoding modes. ``rate`` is the default and the only one whose spike count is
#: linear in feature magnitude; ``latency`` is secondary (spec S3.4).
MODES = ("rate", "latency")


class PopulationSpikeEncoder:
    """Encode an R^D vector as Poisson (or latency-coded) population spikes.

    Neuron layout is ``(D, 2, n_per_feature)`` flattened in C order, i.e.
    neuron index ``(d * 2 + polarity) * n_per_feature + k`` with
    ``polarity`` 0 = ON, 1 = OFF. ``PopulationSpikeDecoder`` reshapes on the
    same convention; nothing else should depend on the layout.

    Args:
        embedding_dim: ``D``, the dimensionality of the vectors being encoded.
        n_per_feature: Neurons per embedding dimension **per polarity**.
        window_steps: ``T``, simulation steps in one item presentation.
        dt_ms: Duration of one step, in ms.
        r_max: Firing rate (Hz) at unit feature magnitude.
        mode: ``"rate"`` (default) or ``"latency"``.
        seed: RNG seed. Required in practice -- ``None`` gives a nondeterministic
            generator, which violates spec constraint 4 for any recorded run.
    """

    def __init__(self, embedding_dim: int, n_per_feature: int = 5,
                 window_steps: int = 50, dt_ms: float = 1.0,
                 r_max: float = 200.0, mode: str = "rate",
                 seed: Optional[int] = None):
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1.")
        if n_per_feature < 1:
            raise ValueError("n_per_feature must be >= 1.")
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1.")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")

        self.embedding_dim = int(embedding_dim)
        self.n_per_feature = int(n_per_feature)
        self.window_steps = int(window_steps)
        self.dt_ms = float(dt_ms)
        self.r_max = float(r_max)
        self.mode = mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ meta
    @property
    def n_neurons(self) -> int:
        """``2 * D * n_per_feature`` -- factor 2 is ON/OFF."""
        return 2 * self.embedding_dim * self.n_per_feature

    @property
    def spikes_per_unit(self) -> float:
        """Expected spikes from one ON/OFF population at unit feature magnitude.

        ``n_per_feature * T * r_max * dt / 1000`` -- the decoder's normaliser,
        and the quantity that sets round-trip fidelity: it is the number of
        Poisson events available to represent one dimension.
        """
        return (self.n_per_feature * self.window_steps
                * self.r_max * self.dt_ms / 1000.0)

    @property
    def params(self) -> dict:
        """Codec condition, for figure captions and run records (spec S2.5)."""
        return {
            "codec": "PopulationSpikeCodec",
            "embedding_dim": self.embedding_dim,
            "n_per_feature": self.n_per_feature,
            "window_steps": self.window_steps,
            "dt_ms": self.dt_ms,
            "r_max": self.r_max,
            "mode": self.mode,
            "seed": self.seed,
            "n_neurons": self.n_neurons,
        }

    def reset_rng(self) -> None:
        """Rewind the generator to its seed, so a re-run repeats byte for byte."""
        self.rng = np.random.default_rng(self.seed)

    def describe(self) -> str:
        return (f"PopulationSpikeCodec(mode={self.mode}, "
                f"n_per_feature={self.n_per_feature}, T={self.window_steps}, "
                f"r_max={self.r_max:g}Hz, dt={self.dt_ms:g}ms, "
                f"N={self.n_neurons}, seed={self.seed})")

    # -------------------------------------------------------------- encoding
    def _split_polarity(self, vector: np.ndarray) -> np.ndarray:
        """(D,) signed -> (D, 2) non-negative magnitudes [ON, OFF]."""
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.shape[0] != self.embedding_dim:
            raise ValueError(
                f"expected a vector of length {self.embedding_dim}, got {v.shape[0]}.")
        return np.stack([np.maximum(v, 0.0), np.maximum(-v, 0.0)], axis=1)

    def encode(self, vector: np.ndarray,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """``(D,)`` -> ``(n_neurons, window_steps)`` boolean spike array.

        Args:
            rng: Generator to draw from, overriding ``self.rng`` for this call.
                Passing a freshly seeded generator makes the encoding a
                deterministic function of the vector, which is what
                ``CodecWrappedModel`` needs to keep ``predict_next`` pure.
        """
        mag = self._split_polarity(vector)                    # (D, 2)
        gen = self.rng if rng is None else rng
        if self.mode == "rate":
            return self._encode_rate(mag, gen)
        return self._encode_latency(mag, gen)

    def _encode_rate(self, mag: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Independent Poisson trains, p linear in feature magnitude.

        Linearity is the reason ``rate`` is the default: a corruption of size
        eps upstream produces a proportional change in expected spike count,
        which is what preserves the probe grid through transport.
        """
        p = np.clip(mag * self.r_max * self.dt_ms / 1000.0, 0.0, 1.0)  # (D, 2)
        # (D, 2) -> (D, 2, n_per_feature, T); every neuron draws independently.
        p_full = np.repeat(p[:, :, None], self.n_per_feature, axis=2)
        draws = rng.random((self.embedding_dim, 2, self.n_per_feature,
                            self.window_steps))
        spikes = draws < p_full[..., None]
        return spikes.reshape(self.n_neurons, self.window_steps)

    def _encode_latency(self, mag: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One spike per neuron; strong feature -> early spike.

        Secondary mode (spec S3.4). It exists because STDP is a function of
        spike *order* and a Poisson rate code carries little reliable order.
        It does **not** preserve graded corruption linearly, and any section run
        under it must say so.
        """
        T = self.window_steps
        base = np.rint((1.0 - np.clip(mag, 0.0, 1.0)) * (T - 1)).astype(int)
        jitter = rng.integers(-1, 2, size=(self.embedding_dim, 2,
                                           self.n_per_feature))
        t = np.clip(base[:, :, None] + jitter, 0, T - 1)
        spikes = np.zeros((self.embedding_dim, 2, self.n_per_feature, T), dtype=bool)
        active = mag > 0.0                                    # zero magnitude -> silent
        idx = np.nonzero(np.repeat(active[:, :, None], self.n_per_feature, axis=2))
        spikes[idx[0], idx[1], idx[2], t[idx]] = True
        return spikes.reshape(self.n_neurons, self.window_steps)

    def encode_sequence(self, vectors: np.ndarray,
                        gap_steps: Optional[Union[int, Sequence[int]]] = None,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """``(L, D)`` -> ``(n_neurons, L * window_steps + gaps)``.

        Items in order, one window each, no gap by default.

        Args:
            gap_steps: Silent steps inserted **after** each item. An int applies
                to every item; a sequence gives one gap per item. This is the
                natural place to inject inter-event intervals when the interval
                section is built (spec S3.7); nothing in the codec depends on it.
        """
        V = np.atleast_2d(np.asarray(vectors, dtype=float))
        gaps = self._gap_vector(V.shape[0], gap_steps)
        blocks = []
        for i, v in enumerate(V):
            blocks.append(self.encode(v, rng=rng))
            if gaps[i] > 0:
                blocks.append(np.zeros((self.n_neurons, gaps[i]), dtype=bool))
        return np.concatenate(blocks, axis=1)

    def encode_stack(self, vectors: np.ndarray,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """``(L, D)`` -> ``(L, n_neurons, window_steps)``, one window per item.

        The per-item form, for arms whose ``fit_sequence`` iterates events.
        ``encode_sequence`` is the concatenated form for arms that consume one
        continuous train.
        """
        V = np.atleast_2d(np.asarray(vectors, dtype=float))
        return np.stack([self.encode(v, rng=rng) for v in V], axis=0)

    def _gap_vector(self, n_items: int,
                    gap_steps: Optional[Union[int, Sequence[int]]]) -> np.ndarray:
        if gap_steps is None:
            return np.zeros(n_items, dtype=int)
        arr = np.asarray(gap_steps, dtype=int)
        if arr.ndim == 0:
            return np.full(n_items, int(arr))
        if arr.shape[0] != n_items:
            raise ValueError(
                f"gap_steps has {arr.shape[0]} entries for {n_items} items.")
        if np.any(arr < 0):
            raise ValueError("gap_steps must be non-negative.")
        return arr


class PopulationSpikeDecoder:
    """Spike trains -> R^D, unit-norm. Never returns a symbol (spec S2.1).

    Holds a reference to the encoder, mirroring
    ``SymbolicDecoder.__init__(encoder)``.
    """

    def __init__(self, encoder: PopulationSpikeEncoder):
        self.encoder = encoder

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    def counts(self, spikes: np.ndarray) -> np.ndarray:
        """``(n_neurons, T')`` -> ``(D, 2)`` total spikes per ON/OFF population.

        Exposed because the population counts, not the decoded vector, are what
        a diagnostic wants when asking *which* populations a model activated.
        """
        S = np.asarray(spikes)
        if S.ndim == 1:
            S = S[:, None]
        if S.shape[0] != self.encoder.n_neurons:
            raise ValueError(
                f"expected {self.encoder.n_neurons} neurons, got {S.shape[0]}.")
        per_neuron = S.astype(float).sum(axis=1)
        return per_neuron.reshape(self.embedding_dim, 2,
                                  self.encoder.n_per_feature).sum(axis=2)

    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """``(n_neurons, T')`` -> ``(D,)``, L2-normalised.

        The window length ``T'`` need not equal the encoder's ``window_steps``
        -- a spiking arm reads out over its own recall window -- so the
        normaliser below only sets the scale, which L2-normalisation then
        removes anyway. Returns zeros when nothing fired.
        """
        c = self.counts(spikes)
        v_hat = (c[:, 0] - c[:, 1]) / self.encoder.spikes_per_unit
        norm = np.linalg.norm(v_hat)
        if norm == 0.0:
            return np.zeros(self.embedding_dim)
        return v_hat / norm

    def decode_sequence(self, spikes: np.ndarray,
                        gap_steps: Optional[Union[int, Sequence[int]]] = None,
                        n_items: Optional[int] = None) -> np.ndarray:
        """``(n_neurons, L * window_steps + gaps)`` -> ``(L, D)``.

        Args:
            gap_steps: Must match what ``encode_sequence`` was given, so the
                windows line up.
            n_items: Required when ``gap_steps`` is a scalar and the trailing
                gap makes ``L`` ambiguous; inferred otherwise.
        """
        S = np.asarray(spikes)
        T = self.encoder.window_steps
        if gap_steps is None:
            if S.shape[1] % T != 0:
                raise ValueError(
                    f"{S.shape[1]} steps is not a whole number of {T}-step windows.")
            L = S.shape[1] // T
            return np.stack([self.decode(S[:, i * T:(i + 1) * T]) for i in range(L)])

        if n_items is None:
            arr = np.asarray(gap_steps, dtype=int)
            if arr.ndim == 0:
                raise ValueError("pass n_items with a scalar gap_steps.")
            n_items = arr.shape[0]
        gaps = self.encoder._gap_vector(n_items, gap_steps)
        out, cursor = [], 0
        for i in range(n_items):
            out.append(self.decode(S[:, cursor:cursor + T]))
            cursor += T + int(gaps[i])
        return np.stack(out)

    def decode_stack(self, spikes: np.ndarray) -> np.ndarray:
        """``(L, n_neurons, T')`` -> ``(L, D)``."""
        S = np.asarray(spikes)
        return np.stack([self.decode(S[i]) for i in range(S.shape[0])])


def make_codec(embedding_dim: int, seed: int, **kwargs):
    """Build a matched ``(encoder, decoder)`` pair. ``seed`` is not optional."""
    enc = PopulationSpikeEncoder(embedding_dim, seed=seed, **kwargs)
    return enc, PopulationSpikeDecoder(enc)
