"""Tully, Linden, Hennig & Lansner (2016) -- spike-based BCPNN sequence learning.

*Spike-Based Bayesian-Hebbian Learning of Temporal Sequences*, PLOS
Computational Biology 12(5):e1004954.

Provenance, in the roster's three-way split (``docs/paper_models_roster.md``):
the **learning rule is reimplemented against a reference** -- the authors' NEST
2.2 synapse module ``bcpnn_connection.h/.cpp`` (Tully & Kaplan 2011-12, in
``Florian-Fiebig/BCPNN-for-NEST222-MPI``), transliterated in
``tests/test_bcpnn_spiking.py`` and used there as the oracle for this module's
trace arithmetic -- while the **network around it is built from the published
equations**, because the paper's network scripts were never released ("all
data within the paper"; the Fiebig & Lansner 2017 scripts are "available upon
request"). So the rule is held to code, the circuit to the Methods and S1
Appendix. The reference module has no LICENSE file; nothing from it is
vendored.

Why this arm exists
-------------------
It is the roster's spiking x online arm with a **real membrane**. The Vieth
arm learns online but its units are memoryless stochastic thresholds; the Bush
arm has Izhikevich membranes but ships no code and had to be built from the
Methods alone; spiking EP is a spiking substrate under a rate interface. This
one has AdEx pyramidal cells with spike-frequency adaptation, conductance
synapses with AMPA/NMDA/GABA kinetics, and a local, per-spike, three-factor
plasticity rule whose reference implementation is readable C++.

The rule is also **natively hetero-associative and natively directional**:
the presynaptic NMDA trace is slow (``tau_zi_nmda = 150 ms``) and the
postsynaptic trace fast (``tau_zj = 5 ms``), so the co-activation estimate
``P_ij`` is a temporally *shifted* window. A pattern that follows another gets
positive NMDA weights from it; the reverse pair does not. Swapping the two time
constants reverses recall (paper Fig 4G). Direction is a parameter, not a
task-geometry accident -- contrast ``ca3net``'s symmetric kernel.

The model (paper Eqs 1-9, S1 Appendix)
--------------------------------------
Traces, per neuron, Euler-integrated at ``dt_ms`` (reference module, at its
resolution)::

    tau_z  dZ/dt = S / (f_max dt) - Z + eps        # Z_i (AMPA 5 ms, NMDA 150 ms), Z_j (5 ms)
    tau_p  dP/dt = kappa (Z - P)                   # P_i, P_j
    tau_p dP_ij/dt = kappa (Z_i Z_j - P_ij)        # per synapse, per receptor type

``1 / (f_max dt)`` normalises so a neuron firing at ``f_max = 20 Hz`` has
``Z ~ 1``; ``eps = 0.01`` is the smallest probability, so every log below is
defined. ``kappa`` is the print-now signal: 1 while a stimulus is on, 0
otherwise, so **the P traces are the memory and they are frozen outside
training**. Weights and biases are read off the traces, not stored::

    w_ij  = w_gain^syn log( P_ij / (P_i P_j) )     # AMPA 6.02 nS, NMDA 1.22 nS
    I_b_j = beta_gain  log( P_j )                  # 50 pA

(with the two gains calibrated to the paper's plotted weights -- see
``PAPER["ampa_calib"]`` and the port note; the stated values overshoot the
figures by 3x and 40x and the replay regime with them.)

A negative ``w_ij`` is delivered through the inhibitory reversal potential
(paper Fig 1B, Eq 7). Pyramidal cells are AdEx with spike-triggered adaptation
only (``b = 150 pA``, ``tau_w = 150 ms``), reset at ``V_t = -55 mV``; basket
cells are the same without adaptation or bias. Positive recurrent synapses
carry Tsodyks-Markram depression (``U = 0.25``, ``tau_rec = 800 ms``).

**Training and recall are two regimes of the same network** (paper, Results):
during training ``kappa = 1`` and *both gains are zero* -- the recurrent
BCPNN synapses and the bias current are switched off, so the network is
feedforward stimulus plus local WTA and the traces simply record what was
driven. During recall ``kappa = 0`` and the gains are on. ``fit_event`` is the
first regime, ``predict_next`` the second.

What "online" means here, precisely
-----------------------------------
Every trace is updated on every step from that step's spikes, one presentation
at a time, with nothing held back: ``fit_sequence`` is a ``fit_event`` loop.
The one implementation liberty is **how** ``P_ij`` is accumulated. Its update
is linear in the outer product ``Z_i Z_j^T``, so over an item of ``T`` steps
with ``kappa`` constant the ``T`` per-step outer products are one matrix
product with closed-form per-step weights -- the same numbers, to floating
point, as stepping it (``test_lazy_accumulation_equals_stepping``). That is an
efficiency choice with no dynamical role, in exactly the sense
``docs/paper_models_roster.md`` uses for EP's minibatch; the P traces are
never *read* during training because the gains are zero, so nothing observes
the difference.

Where the codec meets the model
-------------------------------
The arm consumes and emits **spike arrays in codec-neuron space** behind
``memval.models.codec_wrapper``, like Bush. The mapping is structural, not a
readout convention:

* a **hypercolumn is one feature dimension**, and its **two minicolumns are the
  ON and OFF polarities**. The paper's within-hypercolumn WTA -- pyramidal
  cells drive shared basket cells that inhibit the whole hypercolumn -- is then
  competition between a feature's sign populations, which is what the codec's
  differential read (``counts[:,0] - counts[:,1]``) assumes;
* codec neuron ``k`` *is* pyramidal cell ``k``, stimulated through the paper's
  5 nS AMPA input synapse; the codec's default ``r_max = 200 Hz`` is the
  paper's own stimulus rate;
* the readout is the pyramidal spike raster over the recall window, and the
  decoder reduces it. No learned readout, no argmax, no threshold in the arm.

Deviations from the paper, stated rather than absorbed
------------------------------------------------------
* **No spatial layout.** The paper draws axonal delays from hypercolumn
  distance on a 3x3 grid (Eq 5); features have no geometry, so each
  presynaptic cell draws one delay from ``N(delay_ms, 0.1 delay_ms)``, rounded
  to whole ms. ``delay_ms`` defaults to the paper's grid average.
* **Time steps.** Traces and spike delivery run on a 1 ms grid (the paper's
  spike duration ``dt``), membranes and conductances on ``dt_ms / substeps``.
  NEST integrated at finer resolution; the trace Euler decay at 1 ms differs
  from the exact exponential by ~2% for the 5 ms traces.
* **No E trace.** The reference module has a third trace level (``tau_e``)
  between Z and P; the paper's Eq 3 has two, and this is the paper's form.
  The oracle in the tests carries the E level so the transliteration is
  faithful, and is compared to this module with that level disabled.
* **Stimulus length.** A codec window is ``window_steps`` ms; the paper's
  ``t_stim`` is 100 ms. ``cue_repeats`` tiles the window to reach it.
* **Size normalisation.** Cell counts are set by the codec, not chosen, and a
  conductance-based network's operating point moves with how many presynaptic
  partners a cell has. The paper's pattern is one minicolumn per hypercolumn,
  9 x 30 = 270 cells; a basket cell sees 0.7 x 30 cells of an active
  minicolumn and each pyramidal cell sees 0.7 x 30 baskets. The three gains
  are therefore scaled so that **a full pattern delivers the paper's drive
  whatever the geometry**: recurrent gains by ``270 / pattern_cells``
  (default ``n_hc n_per_mc``, one minicolumn per hypercolumn),
  pyr->basket by ``30 / n_per_mc``, basket->pyr by ``30 / n_basket_per_hc``.
  All three are exactly 1 at the paper's 9 x 10 x 30 with 30 baskets, so the
  gate runs the paper unscaled. Per-cell quantities (stimulus, background,
  bias) are not scaled. Without this the codec-sized network sits in the
  ground state and never hands over (measured: 60-cell patterns, 4 Hz, no
  transitions).
"""

from typing import Optional

import numpy as np

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode
from ..codec_wrapper import OnlineCodecWrappedModel
from ...encoders.spike_codec import PopulationSpikeEncoder
from ...encoders.columnar_codec import ColumnarSpikeDecoder, ColumnarSpikeEncoder

#: Paper values (S1 Appendix tables B-E), verbatim. Overridable per instance
#: through ``overrides=``; kept as one dict so a diff against the paper is one
#: comparison.
PAPER = dict(
    # AdEx (S1 Appendix B; Eq 6)
    C_m=280.0, g_L=14.0, E_L=-70.0, V_t=-55.0, Delta_T=3.0, V_r=-70.0,
    b=150.0, tau_w=150.0,
    # Absolute refractory period. NOT in the paper (NEST's aeif default is 0);
    # without one a strongly driven cell fires on every membrane substep. 2 ms
    # is the conventional value and caps rates at 500 Hz.
    t_ref=2.0,
    # synapses (Eq 7, 9; Methods)
    E_exc=0.0, E_inh=-75.0, tau_ampa=5.0, tau_nmda=150.0, tau_gaba=5.0,
    w_pb=6.65, w_bp=33.3, p_local=0.7, p_conn=0.25,
    # Background input synapse (Methods: 5 nS Poisson processes).
    w_bg=5.0,
    # Stimulus synapse. The paper states 5 nS and that patterns were driven
    # "such that neurons in active patterns fired at f_max"; in this
    # implementation 5 nS at 200 Hz gives 7 Hz against the WTA and adaptation,
    # and 15 nS gives 20 Hz (bin/bcpnn_gate.py --calibrate). The rate, not
    # the conductance, is the paper's stated criterion, so 15 nS is the default.
    w_stim=15.0,
    # short-term depression (Eq 8)
    U=0.25, tau_rec=800.0,
    # BCPNN (Eq 2-4; S1 Appendix C-D)
    tau_zi_ampa=5.0, tau_zi_nmda=150.0, tau_zj=5.0, tau_p=5000.0,
    f_max=20.0, eps=0.01, gain_ampa=6.02, gain_nmda=1.22, beta_gain=50.0,
    # Weight calibration -- READ THIS BEFORE CHANGING THE GAINS ABOVE.
    # Eq 4 with the stated gains and the stated protocol gives terminal
    # in-pattern weights of ~17 nS AMPA and ~1.3 nS NMDA here; the paper's own
    # figures show ~4 nS and ~0.03 nS (Fig 3E/F, 4B) while its negative AMPA
    # weights (~-8 nS) match. With the stated gains this implementation
    # replays sequences at 80-150 Hz and ~10 attractors/s; with the gains
    # multiplied by these two factors it replays at 20-40 Hz and 5-7/s, which
    # is the paper's Fig 3H/I and 4E/F. The factors are therefore matched to
    # the paper's plotted weights and validated on its plotted dynamics; the
    # source of the discrepancy in the text is not identified
    # (docs/bcpnn_spiking_port.md). ``ampa_calib=nmda_calib=1`` is the
    # equations-as-stated ablation; bin/bcpnn_gate.py --calibrate runs both.
    ampa_calib=0.3, nmda_calib=0.025,
    # recall drive (Methods, "Stimulation Paradigm")
    r_bg_cued=150.0, r_bg_free=350.0,
)


def _paper_mean_delay_ms(d_norm: float = 0.75, velocity: float = 0.2) -> float:
    """Eq 5 averaged over all hypercolumn pairs of the paper's 3x3 grid."""
    pts = np.array([(m, n) for m in range(3) for n in range(3)], dtype=float)
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return float(d_norm * dist.mean() / velocity + 1.0)


class BCPNNSpikingNetwork(HippocampalModel, OnlineTrainable):
    """Columnar AdEx network with spike-based BCPNN plasticity.

    Args:
        n_hc: Hypercolumns. Behind the codec this is ``D``.
        n_mc: Minicolumns per hypercolumn; the WTA groups. 2 behind the codec.
        n_per_mc: Pyramidal cells per minicolumn; ``n_per_feature`` behind the
            codec. The paper uses 9 x 10 x 30.
        n_basket_per_hc: Basket cells per hypercolumn. Default 30, the
            paper's; the basket -> pyramidal gain is normalised by it (module
            docstring, "Size normalisation").
        pattern_cells: Pyramidal cells an item is expected to activate, the
            quantity the recurrent gains are normalised by
            (``gain_scale = 270 / pattern_cells``). Default ``n_hc * n_per_mc``,
            one minicolumn per hypercolumn -- the paper's pattern and what a
            dense substrate produces. **A sparse substrate must set it**: on
            the hierarchical encoder ~4% of cells are active per item, so the
            default under-drives every attractor ~15x and the arm floors on a
            gain setting rather than on the material. It is a codec condition
            in the sense of spec S2.5 and belongs in the caption.
        cue_repeats: Times the cue window is tiled in ``predict_next`` so a
            ``window_steps``-ms codec window reaches the paper's 100 ms
            stimulus. 1 presents it once.
        readout_lag_ms: Silent (background-only) ms between the end of the cue
            and the start of the readout window. **Not the paper's** -- the
            paper reads a continuous replay; MemVal probes one step, so the
            window has to start after the cued attractor has handed over.
            ``bin/bcpnn_gate.py --codec`` measures where that is.
        recall_steps: Length of the readout window in ms.
        r_bg: Background Poisson rate (Hz) into every pyramidal cell during
            recall, through the 5 nS stimulus synapse. Paper: 150 Hz cued,
            350 Hz free-running.
        delay_ms: Mean recurrent axonal delay; see the module docstring.
        dt_ms: Trace / delivery grid. 1 ms is the paper's spike duration and
            the trace normaliser assumes it.
        substeps: Membrane substeps per ``dt_ms``.
        n_epochs: Passes over the material in ``fit_sequence``. Paper: 50.
        overrides: Any key of :data:`PAPER`, replacing the paper's value.
        seed: Instance RNG. Probes use a derived, fixed generator.
    """

    #: ``recall`` feeds the arm's own output spikes back as the next cue --
    #: the same protocol the codec wrapper runs in R^D. The network's native
    #: free-running replay is :meth:`replay`, kept for the correctness gate.
    rollout_mode = RolloutMode.OBSERVATION
    prompt_conditioned = False
    #: ``fit_sequence`` is exactly ``fit_event`` over the items, repeated.
    online_equivalent = True

    def __init__(self, n_hc: int, n_mc: int, n_per_mc: int,
                 n_basket_per_hc: Optional[int] = None,
                 pattern_cells: Optional[float] = None,
                 cue_repeats: int = 2,
                 readout_lag_ms: int = 50,
                 recall_steps: int = 200,
                 r_bg: Optional[float] = None,
                 delay_ms: Optional[float] = None,
                 dt_ms: float = 1.0,
                 substeps: int = 5,
                 n_epochs: int = 50,
                 overrides: Optional[dict] = None,
                 seed: Optional[int] = None, **kwargs):
        if min(n_hc, n_mc, n_per_mc) < 1:
            raise ValueError("n_hc, n_mc and n_per_mc must all be >= 1.")
        if cue_repeats < 1 or recall_steps < 1 or readout_lag_ms < 0:
            raise ValueError("cue_repeats, recall_steps >= 1; readout_lag_ms >= 0.")
        if substeps < 1:
            raise ValueError("substeps must be >= 1.")
        self.p = dict(PAPER)
        for k, v in (overrides or {}).items():
            if k not in PAPER:
                raise KeyError(f"unknown parameter {k!r}; see bcpnn_spiking.PAPER")
            self.p[k] = float(v)

        self.n_hc, self.n_mc, self.n_per_mc = int(n_hc), int(n_mc), int(n_per_mc)
        self.n_pyr_per_hc = self.n_mc * self.n_per_mc
        self.n_neurons = self.n_hc * self.n_pyr_per_hc
        #: This arm's "features" are its pyramidal cells, as for every codec arm.
        self.n_features = self.n_neurons
        self.n_basket_per_hc = int(30 if n_basket_per_hc is None else n_basket_per_hc)
        self.pattern_cells = float(self.n_hc * self.n_per_mc if pattern_cells is None
                                   else pattern_cells)
        if self.pattern_cells <= 0:
            raise ValueError("pattern_cells must be > 0.")
        self.n_basket = self.n_hc * self.n_basket_per_hc
        self.cue_repeats = int(cue_repeats)
        self.readout_lag_ms = int(readout_lag_ms)
        self.recall_steps = int(recall_steps)
        self.r_bg = float(self.p["r_bg_cued"] if r_bg is None else r_bg)
        self.delay_ms = float(_paper_mean_delay_ms() if delay_ms is None else delay_ms)
        self.dt_ms = float(dt_ms)
        self.substeps = int(substeps)
        self.n_epochs = int(n_epochs)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._build_circuit()
        self._init_traces()
        self.reset_context()
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ build
    def _build_circuit(self) -> None:
        p, N, M = self.p, self.n_neurons, self.n_basket
        rng = self.rng
        # Hypercolumn index of every pyramidal and basket cell.
        self.hc_of_pyr = np.repeat(np.arange(self.n_hc), self.n_pyr_per_hc)
        self.hc_of_basket = np.repeat(np.arange(self.n_hc), self.n_basket_per_hc)
        same_hc = self.hc_of_pyr[:, None] == self.hc_of_basket[None, :]   # (N, M)

        # Static local WTA (Methods): pyr -> basket AMPA, basket -> pyr GABA,
        # both p = 0.7 within the hypercolumn, weights N(mean, 10%).
        # W_pb[i, m]: pyramidal i -> basket m. W_bp[m, j]: basket m -> pyr j.
        self.W_pb = (same_hc & (rng.random((N, M)) < p["p_local"])) * \
            rng.normal(p["w_pb"], 0.1 * p["w_pb"], (N, M))
        self.W_bp = (same_hc.T & (rng.random((M, N)) < p["p_local"])) * \
            rng.normal(p["w_bp"], 0.1 * p["w_bp"], (M, N))
        self.W_pb = np.clip(self.W_pb, 0.0, None)
        self.W_bp = np.clip(self.W_bp, 0.0, None)

        # Plastic BCPNN connectivity among all pyramidal cells, p = 0.25, no
        # autapses. mask[i, j]: presynaptic i -> postsynaptic j.
        self.mask = rng.random((N, N)) < p["p_conn"]
        np.fill_diagonal(self.mask, False)

        # One delay per presynaptic cell (see module docstring), whole ms >= 1.
        d = rng.normal(self.delay_ms, 0.1 * self.delay_ms, N)
        self.delays = np.maximum(1, np.rint(d / self.dt_ms)).astype(int)
        self.max_delay = int(self.delays.max())

        # Size normalisation (module docstring): the paper's pattern is 270
        # cells, its active minicolumn 30, its basket pool 30.
        self.gain_scale = 270.0 / self.pattern_cells
        self.W_pb *= 30.0 / self.n_per_mc
        self.W_bp *= 30.0 / self.n_basket_per_hc

        # Per-cell constants for the joint (pyramidal + basket) membrane arrays.
        self.b_cell = np.concatenate([np.full(N, p["b"]), np.zeros(M)])
        self.is_pyr = np.concatenate([np.ones(N, bool), np.zeros(M, bool)])

    def _init_traces(self) -> None:
        """Traces at rest: Z = eps, P_i = P_j = eps, P_ij = eps^2 -- weight 0."""
        p, N = self.p, self.n_neurons
        eps = p["eps"]
        self.Z_i_ampa = np.full(N, eps)
        self.Z_i_nmda = np.full(N, eps)
        self.Z_j = np.full(N, eps)
        self.P_i_ampa = np.full(N, eps)
        self.P_i_nmda = np.full(N, eps)
        self.P_j = np.full(N, eps)
        self.P_ij_ampa = np.full((N, N), eps * eps)
        self.P_ij_nmda = np.full((N, N), eps * eps)
        self._pending = []          # buffered (Z_i_ampa, Z_i_nmda, Z_j) rows
        self._weights_dirty = True
        self._W_ampa = self._W_nmda = self._I_beta = None

    @property
    def params(self) -> dict:
        d = {
            "model": "BCPNNSpikingNetwork",
            "n_hc": self.n_hc, "n_mc": self.n_mc, "n_per_mc": self.n_per_mc,
            "n_neurons": self.n_neurons, "n_basket": self.n_basket,
            "cue_repeats": self.cue_repeats, "readout_lag_ms": self.readout_lag_ms,
            "recall_steps": self.recall_steps, "r_bg": self.r_bg,
            "n_basket_per_hc": self.n_basket_per_hc,
            "pattern_cells": self.pattern_cells, "gain_scale": self.gain_scale,
            "delay_ms": self.delay_ms, "dt_ms": self.dt_ms,
            "substeps": self.substeps, "n_epochs": self.n_epochs, "seed": self.seed,
        }
        d.update({k: v for k, v in self.p.items() if v != PAPER[k]})
        return d

    def named_parameters(self) -> dict:
        self._flush()
        return {"P_ij_ampa": self.P_ij_ampa, "P_ij_nmda": self.P_ij_nmda,
                "P_i_ampa": self.P_i_ampa, "P_i_nmda": self.P_i_nmda, "P_j": self.P_j}

    # ------------------------------------------------------------------ state
    def _blank_dynamics(self, rng: np.random.Generator) -> dict:
        p, N, M = self.p, self.n_neurons, self.n_basket
        return {
            # V initialised uniformly in [V_r, V_t] (Methods).
            "V": rng.uniform(p["V_r"], p["V_t"], N + M),
            "I_w": np.zeros(N + M),
            "refr": np.zeros(N + M),
            # conductances: exc/inh by receptor kinetics; GABA from baskets
            "gA_e": np.zeros(N + M), "gA_i": np.zeros(N + M),
            "gN_e": np.zeros(N + M), "gN_i": np.zeros(N + M),
            "gG": np.zeros(N + M),
            "x": np.ones(N),                                   # STD resources
            "ring": np.zeros((self.max_delay + 1, N)),         # pending pyr spikes
            "basket_last": np.zeros(M),                        # basket spikes, 1 ms delay
            "t": 0,
        }

    def reset_context(self):
        """Clear membranes, conductances, delay lines and the Z traces.

        The P traces are the memory and stay. Z is set to its resting value
        ``eps`` -- what a long silent gap with ``kappa = 0`` leaves it at
        (paper: kappa clamped to 0 between stimuli) -- so no transition can
        form across a seam.
        """
        self._flush()
        self._state = self._blank_dynamics(self.rng)
        eps = self.p["eps"]
        self.Z_i_ampa.fill(eps)
        self.Z_i_nmda.fill(eps)
        self.Z_j.fill(eps)

    def on_event_boundary(self, boundary_context: Optional[np.ndarray] = None) -> None:
        self.reset_context()

    def get_latent_state(self) -> dict:
        self._flush()
        return {"P_ij_ampa": self.P_ij_ampa.copy(), "P_ij_nmda": self.P_ij_nmda.copy(),
                "P_i_ampa": self.P_i_ampa.copy(), "P_i_nmda": self.P_i_nmda.copy(),
                "P_j": self.P_j.copy()}

    def _save_state(self) -> dict:
        return {"dyn": {k: (v.copy() if isinstance(v, np.ndarray) else v)
                        for k, v in self._state.items()},
                "Z": (self.Z_i_ampa.copy(), self.Z_i_nmda.copy(), self.Z_j.copy())}

    def _restore_state(self, saved: dict) -> None:
        self._state = saved["dyn"]
        self.Z_i_ampa, self.Z_i_nmda, self.Z_j = saved["Z"]

    # ----------------------------------------------------------------- traces
    def _update_traces(self, spikes: np.ndarray) -> None:
        """One ``dt_ms`` step of Eqs 2-3 with ``kappa = 1``.

        ``spikes`` are this step's pyramidal spike counts. Order is the reference
        module's: Z first (spike kick, decay, ``+eps``), then P from the updated
        Z. ``P_ij`` is buffered (see :meth:`_flush`).
        """
        p, dt = self.p, self.dt_ms
        kick = spikes / (p["f_max"] * dt * 1e-3)           # S / (f_max dt), dt in s
        eps = p["eps"]
        self.Z_i_ampa += (kick - self.Z_i_ampa + eps) * dt / p["tau_zi_ampa"]
        self.Z_i_nmda += (kick - self.Z_i_nmda + eps) * dt / p["tau_zi_nmda"]
        self.Z_j += (kick - self.Z_j + eps) * dt / p["tau_zj"]
        a = dt / p["tau_p"]
        self.P_i_ampa += a * (self.Z_i_ampa - self.P_i_ampa)
        self.P_i_nmda += a * (self.Z_i_nmda - self.P_i_nmda)
        self.P_j += a * (self.Z_j - self.P_j)
        self._pending.append((self.Z_i_ampa.copy(), self.Z_i_nmda.copy(), self.Z_j.copy()))
        self._weights_dirty = True

    def _flush(self) -> None:
        """Apply the buffered ``P_ij`` steps as one matrix product.

        Stepping ``P <- (1-a) P + a Z_i Z_j^T`` for ``T`` steps gives
        ``P(T) = (1-a)^T P(0) + sum_t a (1-a)^(T-1-t) Z_i(t) Z_j(t)^T``, so the
        sum is ``(A Z_i) @ Z_j^T`` with the per-step weights on the rows. Exact
        to floating point (tested), and the P traces are not read during
        training, so this is purely how the same numbers are computed.
        """
        if not self._pending:
            return
        a = self.dt_ms / self.p["tau_p"]
        T = len(self._pending)
        wts = a * (1.0 - a) ** np.arange(T - 1, -1, -1)              # (T,)
        Zia = np.stack([z[0] for z in self._pending], axis=1)        # (N, T)
        Zin = np.stack([z[1] for z in self._pending], axis=1)
        Zj = np.stack([z[2] for z in self._pending], axis=1)
        decay = (1.0 - a) ** T
        self.P_ij_ampa *= decay
        self.P_ij_ampa += (Zia * wts) @ Zj.T
        self.P_ij_nmda *= decay
        self.P_ij_nmda += (Zin * wts) @ Zj.T
        self._pending = []

    def _compute_weights(self) -> None:
        """Eq 4: log-ratio weights and biases from the (flushed) P traces."""
        self._flush()
        if not self._weights_dirty and self._W_ampa is not None:
            return
        p = self.p
        outer_a = self.P_i_ampa[:, None] * self.P_j[None, :]
        outer_n = self.P_i_nmda[:, None] * self.P_j[None, :]
        ga = p["gain_ampa"] * p["ampa_calib"] * self.gain_scale
        gn = p["gain_nmda"] * p["nmda_calib"] * self.gain_scale
        self._W_ampa = ga * np.log(self.P_ij_ampa / outer_a) * self.mask
        self._W_nmda = gn * np.log(self.P_ij_nmda / outer_n) * self.mask
        self._I_beta = p["beta_gain"] * np.log(self.P_j)
        self._weights_dirty = False

    def weights(self) -> dict:
        """The current effective weights, ``W[pre, post]`` in nS, and biases in pA."""
        self._compute_weights()
        return {"ampa": self._W_ampa.copy(), "nmda": self._W_nmda.copy(),
                "I_beta": self._I_beta.copy()}

    # ----------------------------------------------------------------- engine
    def _step_ms(self, ext: np.ndarray, plastic: bool, recurrent: bool,
                 rng: np.random.Generator, r_bg: float) -> np.ndarray:
        """Advance one ``dt_ms``: deliver spikes, integrate membranes, learn.

        Args:
            ext: External stimulus spike counts per pyramidal cell this ms,
                delivered through the ``w_stim`` AMPA synapse.
            plastic: ``kappa = 1`` -- update the traces.
            recurrent: BCPNN synapses and bias current on (recall regime).
            rng: Background Poisson draws.
            r_bg: Background rate (Hz) into every pyramidal cell; 0 = none.

        Returns this ms's pyramidal spike counts.
        """
        s, p, N, M = self._state, self.p, self.n_neurons, self.n_basket
        dt = self.dt_ms

        # ---- deliveries at the start of the step ---------------------------
        s["gA_e"][:N] += p["w_stim"] * ext
        if r_bg > 0.0:
            # Poisson background as a Bernoulli arrival per ms (r dt << 1).
            s["gA_e"][:N] += p["w_bg"] * (rng.random(N) < r_bg * dt * 1e-3)

        slot = s["t"] % (self.max_delay + 1)
        arriving = s["ring"][slot].copy()        # copy: the row is cleared next
        s["ring"][slot] = 0.0
        if arriving.any():
            pre = np.flatnonzero(arriving)
            cnt = arriving[pre]
            # pyr -> basket, static AMPA
            s["gA_e"][N:] += cnt @ self.W_pb[pre]
            if recurrent:
                # Tsodyks-Markram: release U x of the resource, then deplete.
                rel = cnt * self.p["U"] * s["x"][pre]
                s["x"][pre] -= rel                                  # per spike, x <- x - U x
                # paper Eq 9: conductance kick x_dep * w; the STD factor is
                # applied to positive weights only. rel/U == x * count.
                scale = rel / self.p["U"]
                Wa, Wn = self._W_ampa[pre], self._W_nmda[pre]        # rows: pre -> all post
                s["gA_e"][:N] += scale @ np.clip(Wa, 0.0, None)
                s["gA_i"][:N] += cnt @ np.clip(-Wa, 0.0, None)
                s["gN_e"][:N] += scale @ np.clip(Wn, 0.0, None)
                s["gN_i"][:N] += cnt @ np.clip(-Wn, 0.0, None)
        if s["basket_last"].any():
            s["gG"][:N] += s["basket_last"] @ self.W_bp
        # STD recovery over the step (exact).
        s["x"] = 1.0 - (1.0 - s["x"]) * np.exp(-dt / p["tau_rec"])

        # ---- membranes, `substeps` Euler steps -----------------------------
        h = dt / self.substeps
        dA, dN, dG = (np.exp(-h / p["tau_ampa"]), np.exp(-h / p["tau_nmda"]),
                      np.exp(-h / p["tau_gaba"]))
        dW = np.exp(-h / p["tau_w"])
        I_b = np.zeros(N + M)
        if recurrent:
            I_b[:N] = self._I_beta
        fired = np.zeros(N + M)
        V, Iw = s["V"], s["I_w"]
        hC = h / p["C_m"]
        for _ in range(self.substeps):
            g_e = s["gA_e"] + s["gN_e"]
            g_i = s["gA_i"] + s["gN_i"] + s["gG"]
            expo = p["g_L"] * p["Delta_T"] * np.exp(
                np.minimum((V - p["V_t"]) / p["Delta_T"], 5.0))
            # Semi-implicit step: every conductance term is taken at the new
            # voltage, so the update is stable however large the total
            # conductance gets (explicit Euler overshoots past the reversal
            # potentials once g_tot h / C_m > 2 and the network seizes).
            # Only the exponential term is explicit; it is bounded because V
            # is reset at V_t.
            num = V + hC * (p["g_L"] * p["E_L"] + g_e * p["E_exc"] + g_i * p["E_inh"]
                            + expo - Iw + I_b)
            V_new = num / (1.0 + hC * (p["g_L"] + g_e + g_i))
            active = s["refr"] <= 0.0
            V[active] = V_new[active]
            s["refr"] -= h
            hot = V >= p["V_t"]
            if hot.any():
                fired[hot] += 1.0
                V[hot] = p["V_r"]
                Iw[hot] += self.b_cell[hot]
                s["refr"][hot] = p["t_ref"]
            Iw *= dW
            s["gA_e"] *= dA; s["gA_i"] *= dA
            s["gN_e"] *= dN; s["gN_i"] *= dN
            s["gG"] *= dG

        pyr_spk, basket_spk = fired[:N], fired[N:]
        if pyr_spk.any():
            j = np.flatnonzero(pyr_spk)
            slots = (s["t"] + self.delays[j]) % (self.max_delay + 1)
            s["ring"][slots, j] += pyr_spk[j]
        s["basket_last"] = basket_spk
        s["t"] += 1
        if plastic:
            self._update_traces(pyr_spk)
        return pyr_spk

    def _run(self, ext: Optional[np.ndarray], n_ms: int, plastic: bool,
             recurrent: bool, rng: np.random.Generator, r_bg: float) -> np.ndarray:
        """Run ``n_ms`` steps; ``ext`` is ``(N, n_ms)`` or None. Returns the raster."""
        N = self.n_neurons
        out = np.zeros((N, n_ms))
        zero = np.zeros(N)
        for t in range(n_ms):
            e = zero if ext is None else ext[:, t]
            out[:, t] = self._step_ms(e, plastic, recurrent, rng, r_bg)
        return out

    # -------------------------------------------------------------------- i/o
    def _as_spikes(self, event: np.ndarray) -> np.ndarray:
        S = np.asarray(event)
        if S.ndim == 1:
            S = S[:, None]
        if S.shape[0] != self.n_neurons:
            raise ValueError(
                f"expected {self.n_neurons} neurons in the spike array, got {S.shape[0]}. "
                "This arm consumes spike trains, not vectors -- wrap it with "
                "memval.models.codec_wrapper.wrap_with_codec.")
        return S.astype(float)

    # --------------------------------------------------------------- training
    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None,
                  **kwargs):
        """Present one item's spike train: ``kappa = 1``, gains off (paper).

        Membranes and Z traces persist across calls -- the paper's IPI = 0
        sequence protocol -- which is where the shifted NMDA co-activation
        between consecutive items comes from. Call ``on_event_boundary`` at a
        seam so no transition forms across it.
        """
        S = self._as_spikes(event)
        self._run(S, S.shape[1], plastic=True, recurrent=False, rng=self.rng, r_bg=0.0)
        self._flush()

    def fit_sequence(self, sequence_data: np.ndarray,
                     context_data: Optional[np.ndarray] = None,
                     epochs: Optional[int] = None, **kwargs):
        """Present a sequence ``n_epochs`` times, fenced at both seams.

        Args:
            sequence_data: ``(L, n_neurons, T)`` (``encode_stack``'s form) or
                ``(n_neurons, T)`` for one continuous train.
            epochs: Passes for this call, overriding ``n_epochs``.
        """
        X = np.asarray(sequence_data)
        items = [X] if X.ndim == 2 else list(X)
        for _ in range(self.n_epochs if epochs is None else int(epochs)):
            self.on_event_boundary()
            for item in items:
                self.fit_event(item)
            self.on_event_boundary()

    # ---------------------------------------------------------------- readout
    def _probe_seed(self) -> int:
        return 0 if self.seed is None else int(self.seed) + 104729

    def predict_next(self, current_event: np.ndarray,
                     current_context: Optional[np.ndarray] = None,
                     **kwargs) -> np.ndarray:
        """Cue with the paper's recall protocol and return the readout raster.

        ``kappa = 0``, gains on, background at ``r_bg``. The cue window is
        tiled ``cue_repeats`` times as the stimulus, then ``readout_lag_ms`` of
        background alone, then ``recall_steps`` ms whose pyramidal spikes are
        returned as ``(n_neurons, recall_steps)`` boolean -- **not** reduced to
        a vector; that is the decoder's job (spec S2.4).

        Pure function of the cue: the traces are frozen, the background draws
        come from a fixed generator, and all transient state is saved and
        restored around the call.
        """
        cue = self._as_spikes(current_event)
        self._compute_weights()
        saved = self._save_state()
        try:
            gen = np.random.default_rng(self._probe_seed())
            self._state = self._blank_dynamics(gen)
            stim = np.tile(cue, (1, self.cue_repeats))
            self._run(stim, stim.shape[1], plastic=False, recurrent=True,
                      rng=gen, r_bg=self.r_bg)
            self._run(None, self.readout_lag_ms, plastic=False, recurrent=True,
                      rng=gen, r_bg=self.r_bg)
            out = self._run(None, self.recall_steps, plastic=False, recurrent=True,
                            rng=gen, r_bg=self.r_bg)
            return out > 0
        finally:
            self._restore_state(saved)

    def replay(self, cue: Optional[np.ndarray], duration_ms: int,
               r_bg: Optional[float] = None, seed: Optional[int] = None) -> np.ndarray:
        """The paper's own readout: a continuous raster ``(n_neurons, duration)``.

        Optionally cued (``cue`` is ``(N, T)``, presented once at the start),
        under background ``r_bg`` (default ``r_bg_free`` when uncued, ``r_bg``
        when cued). Pure like ``predict_next``. This is what
        ``bin/bcpnn_gate.py`` scores with the paper's attractor detector and
        CRP; benchmarks never call it.
        """
        self._compute_weights()
        saved = self._save_state()
        try:
            gen = np.random.default_rng(self._probe_seed() if seed is None else seed)
            self._state = self._blank_dynamics(gen)
            if r_bg is None:
                r_bg = self.r_bg if cue is not None else self.p["r_bg_free"]
            ext = None
            if cue is not None:
                c = self._as_spikes(cue)
                ext = np.zeros((self.n_neurons, duration_ms))
                ext[:, :min(duration_ms, c.shape[1])] = c[:, :duration_ms]
            return self._run(ext, duration_ms, plastic=False, recurrent=True,
                             rng=gen, r_bg=float(r_bg))
        finally:
            self._restore_state(saved)

    def recall(self, prompt_event: np.ndarray, length: int,
               prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Autoregressive rollout in spike space, ``(length, N, recall_steps)``.

        Provided so the arm is usable standalone; benchmarks roll out through
        ``CodecWrappedModel.recall`` in R^D, as for the Bush arm.
        """
        prompt = np.asarray(prompt_event)
        cue = prompt[-1] if prompt.ndim > 2 else prompt
        out = np.zeros((length, self.n_neurons, self.recall_steps), dtype=bool)
        for t in range(length):
            cue = self.predict_next(cue)
            out[t] = cue
        return out


class CodecBCPNNNetwork(OnlineCodecWrappedModel):
    """Registry-facing form: :class:`BCPNNSpikingNetwork` behind a codec.

    Two codecs, selected by ``codec=``:

    ``"population"``  the shared population spike codec (``spike_codec.py``):
        ``n_hc = D``, ``n_mc = 2`` (ON/OFF), ``n_per_mc = n_per_feature``.
        The arm floors on it for the structural reasons in the module
        docstring; it is kept as the condition every spiking arm shares.
    ``"columnar"``  the interval-coded columnar codec
        (``columnar_codec.py``): ``n_hc = D``, ``n_mc`` intervals per
        dimension, ``n_per_mc`` cells each, one minicolumn active at a flat
        rate -- the format the paper's S1 Appendix prescribes for a continuous
        variable. Its defaults are fixed from the paper's side (10
        minicolumns, ~300 cells per item, 200 Hz for 100 ms) and are not
        re-sited per section.

    Trained through ``fit_event`` whichever regime the caller asks for, like
    the other online arms.
    """

    rollout_mode = RolloutMode.OBSERVATION
    prompt_conditioned = False
    online_equivalent = True

    def __init__(self, n_features: int,
                 codec: str = "population",
                 n_per_feature: int = 5,
                 window_steps: Optional[int] = None,
                 dt_ms: float = 1.0,
                 r_max: float = 200.0,
                 codec_mode: str = "rate",
                 n_mc: int = 10,
                 n_per_mc: int = 3,
                 z_range: float = 2.5,
                 probe_seed: int = 0,
                 seed: Optional[int] = 42,
                 **kwargs):
        if codec == "population":
            encoder = PopulationSpikeEncoder(
                n_features, n_per_feature=n_per_feature,
                window_steps=50 if window_steps is None else window_steps,
                dt_ms=dt_ms, r_max=r_max, mode=codec_mode, seed=seed)
            decoder = None
            inner = BCPNNSpikingNetwork(n_hc=n_features, n_mc=2, n_per_mc=n_per_feature,
                                        dt_ms=dt_ms, seed=seed, **kwargs)
        elif codec == "columnar":
            encoder = ColumnarSpikeEncoder(
                n_features, n_mc=n_mc, n_per_mc=n_per_mc,
                window_steps=100 if window_steps is None else window_steps,
                dt_ms=dt_ms, rate=r_max, z_range=z_range, seed=seed)
            decoder = ColumnarSpikeDecoder(encoder)
            # The window is already the paper's 100 ms stimulus.
            kwargs.setdefault("cue_repeats", 1)
            inner = BCPNNSpikingNetwork(n_hc=n_features, n_mc=n_mc, n_per_mc=n_per_mc,
                                        dt_ms=dt_ms, seed=seed, **kwargs)
        else:
            raise ValueError("codec must be 'population' or 'columnar'.")
        super().__init__(inner, encoder, decoder, inner_domain="spikes",
                         probe_seed=probe_seed)
