# -*- coding: utf8 -*-
"""Encoder / decoder between MemVal event vectors and ca3net spike trains.

This is the adapter that makes the CA3 arm modality-agnostic. Upstream ca3net
treats its 8000 pyramidal cells as *place* cells on a 1-D track: it generates
input by simulating maze running (`generate_spike_train.py`) and decodes recall
by inferring a scalar position from per-neuron tuning curves
(`bayesian_decoding.py`). Porting that would tie the arm to 1-D spatial tasks
forever.

Instead we treat the PC population as a **generic assembly substrate**:

    * partition the N cells into D disjoint contiguous blocks, one per feature
      dimension of the MemVal encoder;
    * ENCODE event vector ``v`` by driving block ``j`` as a Poisson source whose
      rate is an affine function of ``v[j]``;
    * DECODE by counting spikes per block over a window and normalising back to
      a D-vector.

Nothing here imports Brian2 -- it is pure NumPy so it can be unit-tested on its
own, which is the Phase 1 gate in ``docs/ca3net_port_plan.md``.

Rate defaults follow upstream's place-cell statistics: 0.1 Hz out of field,
20 Hz in field (`generate_spike_train.py`).
"""

from typing import Optional, Tuple

import numpy as np

N_CELLS_DEFAULT = 8000
OUTFIELD_RATE_HZ = 0.1
INFIELD_RATE_HZ = 20.0

# Upstream presents inputs every 25 ms (gamma), but that is its *stimulus* rhythm,
# not a fidelity requirement. Our encode->decode fidelity is set by how many spikes
# a window contains, and 25 ms does not clear the Phase 1 gate. Measured round-trip
# cosine on 400-d PlaceCellEncoder items (mean over 8 events):
#
#     event_ms   25     50     100    200    400
#     cosine     0.919  0.957  0.978  0.989  0.994
#
# 100 ms is the default: comfortably above the 0.95 gate with room for the noise a
# real network adds, without inflating simulation cost. Note the RECALL side is not
# bound by this -- there the window is set by the replay sweep, which traverses a
# whole sequence in 380-780 ms.
EVENT_MS_DEFAULT = 100.0

# Brian2's SpikeGeneratorGroup REFUSES a train in which a neuron spikes twice
# inside one time step (100 us default), and independent Poisson draws do that
# routinely at this population size. Upstream applies the same 5 ms cleanup
# (`helper.refractoriness`) to its generated trains.
REFRACTORY_MS_DEFAULT = 5.0


def block_assignment(n_cells: int, dim: int) -> np.ndarray:
    """Map each cell index to a feature-dimension index.

    Contiguous blocks of ``n_cells // dim``; the first ``n_cells % dim`` blocks
    take one extra cell so every cell is used and block sizes differ by at most
    one.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if dim > n_cells:
        raise ValueError("dim (%d) exceeds n_cells (%d): a block would be empty"
                         % (dim, n_cells))
    base, extra = divmod(n_cells, dim)
    sizes = np.full(dim, base, dtype=int)
    sizes[:extra] += 1
    return np.repeat(np.arange(dim), sizes)


def _enforce_refractory(times: np.ndarray, ids: np.ndarray,
                        ref_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """Drop spikes that fall within ``ref_ms`` of the same neuron's previous one.

    Required, not cosmetic: Brian2's ``SpikeGeneratorGroup`` raises outright if a
    neuron spikes twice inside one time step (100 us by default), and independent
    Poisson draws do that regularly once the population is large. Upstream hits
    the same problem and solves it the same way (``helper.refractoriness``,
    5 ms, applied in ``generate_spike_train.py``).

    Iterates because a single pass can leave a violation behind when three or
    more spikes cluster; converges in one or two rounds in practice.
    """
    if ref_ms <= 0 or times.size == 0:
        return times, ids
    order = np.lexsort((times, ids))
    t, i = times[order], ids[order]
    while True:
        same = np.zeros(t.size, dtype=bool)
        same[1:] = i[1:] == i[:-1]
        gap = np.full(t.size, np.inf)
        gap[1:] = t[1:] - t[:-1]
        bad = same & (gap < ref_ms)
        if not bad.any():
            return t, i
        t, i = t[~bad], i[~bad]


class Ca3netIO:
    """Bidirectional adapter for one (n_cells, dim) pairing.

    Parameters
    ----------
    dim : feature dimensionality of the MemVal encoder (e.g. `PlaceCellEncoder`
        with ``n_cells_per_dim=20`` gives 400).
    n_cells : size of the ca3net PC population.
    event_ms : duration of the presentation window for one event.
    infield_rate / outfield_rate : Hz at ``v=1`` and ``v=0`` respectively.
    refractory_ms : minimum inter-spike interval per neuron. Must stay above the
        Brian2 time step; see REFRACTORY_MS_DEFAULT.
    seed : RNG seed. Encoding is stochastic (Poisson), so this is load-bearing
        for reproducibility.
    """

    def __init__(self, dim: int, n_cells: int = N_CELLS_DEFAULT,
                 event_ms: float = EVENT_MS_DEFAULT,
                 infield_rate: float = INFIELD_RATE_HZ,
                 outfield_rate: float = OUTFIELD_RATE_HZ,
                 refractory_ms: float = REFRACTORY_MS_DEFAULT,
                 seed: Optional[int] = 42):
        self.dim = int(dim)
        self.n_cells = int(n_cells)
        self.event_ms = float(event_ms)
        self.infield_rate = float(infield_rate)
        self.outfield_rate = float(outfield_rate)
        self.refractory_ms = float(refractory_ms)
        self.blocks = block_assignment(self.n_cells, self.dim)
        self._cells_per_block = np.bincount(self.blocks, minlength=self.dim)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- encode
    def _rates(self, v: np.ndarray) -> np.ndarray:
        """Per-cell firing rate (Hz) for one event vector."""
        v = np.asarray(v, dtype=float).ravel()
        if v.size != self.dim:
            raise ValueError("event has %d features, adapter built for %d"
                             % (v.size, self.dim))
        # scale to [0, 1] by the event's own max so that encoding is invariant to
        # the encoder's absolute activation scale; an all-zero event stays at
        # baseline rather than dividing by zero.
        peak = float(np.max(v))
        vn = v / peak if peak > 0 else v
        vn = np.clip(vn, 0.0, 1.0)
        per_dim = self.outfield_rate + vn * (self.infield_rate - self.outfield_rate)
        return per_dim[self.blocks]

    def encode_events(self, events: np.ndarray,
                      intervals_ms: Optional[np.ndarray] = None,
                      t0_ms: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a sequence of event vectors as a spike train.

        Parameters
        ----------
        events : (T, dim) array.
        intervals_ms : optional (T,) or (T-1,) array of *blank* gaps inserted
            AFTER each event, during which every cell fires at ``outfield_rate``.
            This is the hook for metric-time work: an inter-event gap is endured
            by the network rather than counted (docs/interval_encoding_design.md).
        t0_ms : time offset of the first event.

        Returns
        -------
        (spike_times_ms, neuron_ids), both 1-D and sorted by time.
        """
        events = np.atleast_2d(np.asarray(events, dtype=float))
        n = len(events)
        gaps = np.zeros(n) if intervals_ms is None else np.asarray(intervals_ms, float)
        if gaps.size == n - 1:
            gaps = np.append(gaps, 0.0)
        if gaps.size != n:
            raise ValueError("intervals_ms must have length T or T-1")

        times, ids, t = [], [], float(t0_ms)
        for i, v in enumerate(events):
            ts, ns = self._poisson_window(self._rates(v), t, self.event_ms)
            times.append(ts); ids.append(ns)
            t += self.event_ms
            if gaps[i] > 0:
                base = np.full(self.n_cells, self.outfield_rate)
                ts, ns = self._poisson_window(base, t, float(gaps[i]))
                times.append(ts); ids.append(ns)
                t += float(gaps[i])

        if not times:
            return np.empty(0), np.empty(0, dtype=int)
        spike_times = np.concatenate(times)
        neuron_ids = np.concatenate(ids).astype(int)
        spike_times, neuron_ids = _enforce_refractory(
            spike_times, neuron_ids, self.refractory_ms)
        order = np.argsort(spike_times, kind="mergesort")
        return spike_times[order], neuron_ids[order]

    def _poisson_window(self, rates_hz: np.ndarray, t_start: float,
                        width_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        """Homogeneous Poisson per cell within one window (rates are constant
        across the window, so this is exact, not an approximation)."""
        lam = rates_hz * (width_ms * 1e-3)
        counts = self.rng.poisson(lam)
        total = int(counts.sum())
        if total == 0:
            return np.empty(0), np.empty(0, dtype=int)
        ids = np.repeat(np.arange(self.n_cells), counts)
        times = t_start + self.rng.random(total) * width_ms
        return times, ids

    # ---------------------------------------------------------------- decode
    def decode_window(self, spike_times: np.ndarray, neuron_ids: np.ndarray,
                      t0: float, t1: float,
                      subtract_baseline: bool = True) -> np.ndarray:
        """Spike counts per block over ``[t0, t1)`` -> normalised D-vector.

        ``subtract_baseline`` removes the expected out-of-field count so that a
        silent block reads ~0 rather than a small positive floor; this matters
        because cosine similarity against a sparse target is otherwise inflated
        by the baseline common-mode.
        """
        spike_times = np.asarray(spike_times, float)
        neuron_ids = np.asarray(neuron_ids, int)
        sel = (spike_times >= t0) & (spike_times < t1)
        counts = np.bincount(self.blocks[neuron_ids[sel]], minlength=self.dim).astype(float)

        width_s = max(t1 - t0, 1e-9) * 1e-3
        if subtract_baseline:
            counts -= self._cells_per_block * self.outfield_rate * width_s
            counts = np.clip(counts, 0.0, None)
        # per-cell rate, then normalise to the in-field maximum so the output
        # lives on the same [0, 1] scale as a MemVal activation vector
        rate = counts / (self._cells_per_block * width_s)
        return rate / self.infield_rate

    # ------------------------------------------------------------- utilities
    def round_trip(self, events: np.ndarray) -> np.ndarray:
        """Encode then decode with no network in between. The Phase 1 gate."""
        events = np.atleast_2d(np.asarray(events, dtype=float))
        st, ni = self.encode_events(events)
        out = np.zeros_like(events, dtype=float)
        for i in range(len(events)):
            t0 = i * self.event_ms
            out[i] = self.decode_window(st, ni, t0, t0 + self.event_ms)
        return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
