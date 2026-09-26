"""Bush, Philippides, Husbands & O'Shea (2010) -- STDP in a spiking recurrent
hippocampal network.

*Dual Coding with STDP in a Spiking Recurrent Neural Network Model of the
Hippocampus*, PLOS Computational Biology 6(7):e1000839.

This is a **reimplementation from the published equations**, not a port: the
paper ships no code, which is also why it was chosen over ``ca3net`` and
Cutsuridis in a pipeline already carrying six unlicensed repos. It is the
roster's **Hebbian contrast case** -- purely local, no error signal beyond what
STDP's temporal asymmetry provides -- against AHN, EP, tPC and the theta arm,
all of which are error-correcting.

The arm consumes and emits **spike arrays**, not vectors. Transport to and from
R^D is the job of ``memval.encoders.spike_codec`` via
``memval.models.codec_wrapper.wrap_with_codec``; see ``docs/spike_codec_spec.md``.
Nothing in this module knows about embeddings, and the decode choice is
deliberately *not* baked in: ``predict_next`` returns the whole
``(n_neurons, recall_steps)`` array and lets the decoder reduce it.

Dynamics (paper S "Methods")
----------------------------
Izhikevich point neurons with excitatory parameters ``a=0.02, b=0.2, c=-65,
d=6``::

    v' = 0.04 v^2 + 5 v + 140 - u + I
    u' = a (b v - u)
    if v >= 30:   v <- c ;  u <- u + d

Integrated with two 0.5 ms substeps per 1 ms step (Izhikevich's standard
scheme); the single-step form is unstable at these parameters. Recurrent, no
self-connections, weights in ``[0, w_max]`` initialised at ``0.01 w_max``.
Fully connected by default -- the paper reports no significant difference at 15
random presynaptic connections per neuron, so sparsity is an optimisation, not a
correctness question (``n_presynaptic`` is provided for that).

Plasticity: additive pair-based STDP in online trace form, all-to-all pairing::

    x_pre  <- x_pre  * exp(-dt/tau_plus)  + spike_pre       # tau_plus  = 20 ms
    y_post <- y_post * exp(-dt/tau_minus) + spike_post      # tau_minus = 50 ms

    on post spike:  w += Phi * A_plus  * x_pre              # A_plus  = +0.02 w_max
    on pre  spike:  w += Phi * A_minus * y_post             # A_minus = -0.01 w_max

    w <- clip(w, 0, w_max)

``Phi`` is the ACh gate: **1 during ``fit_event``, 0 during ``predict_next``**.
That is the encode/retrieve separation, and it is what makes ``predict_next``
non-mutating.

The default is the pair-based **BCM** variant, in which the depression window is
the wider one (``A_minus tau_minus = 0.5 > A_plus tau_plus = 0.4``): symmetric,
non-causal spike pairings -- which is what within-item pairs are -- net-depress,
while causal across-item pairings potentiate. The asymmetry of the learned
weight matrix is produced by that imbalance and nothing else. The non-BCM
variant (``A_minus = -0.021``, ``tau_plus = tau_minus = 20``) is available as an
ablation via ``variant="non_bcm"``; the triplet rule is not implemented.

Fidelity to the paper's Methods (revised 2026-09-04)
----------------------------------------------------
An earlier version of this file was built from ``docs/spike_codec_spec.md`` S2
rather than from the paper, and the spec is wrong on several load-bearing
points. The Methods were then read directly (PLOS Comput Biol 6(7):e1000839,
also PMC2895637) and this implementation now follows them. What the paper
actually specifies, and what it cost to have guessed instead:

* **Inhibition is a theta-modulated global input, and it is bounded.**
  "Inhibitory input to every simulated neuron at each millisecond time step is
  randomly sampled from a Gaussian distribution with mean ``Iinh = -15*theta``
  and standard deviation ``sigma_inh = 2``", where theta "oscillates
  sinusoidally in the range [0 : 1] at a rate of 8Hz". During recall "theta
  frequency inhibitory input to the network is ceased".

  This is an *external drive*, not feedback: it cannot run away, because it is
  not proportional to activity. The spec deferred it as "not needed for a first
  scored arm"; that deferral is what forced the previous version to invent an
  activity-proportional pool, which produced rebound seizures. The paper's own
  mechanism has no such failure mode, and the invented one is gone.

* **Synaptic transmission is current-based and instantaneous within the 1 ms
  step.** Weights are in current units: "a single synaptic current of
  ``I = 16.5`` is required" to generate an action potential, with ``w_max = 1``.
  There is no exponential PSP; the previous ``tau_syn`` was an invention.

* **Axonal delays are per-neuron and distributed.** "Each simulated neuron has
  an axonal delay (Di) randomly assigned from a uniform distribution in the
  range [1ms : Dms] with ``D = 5``". The previous fixed one-step delay for every
  neuron was a simplification that removed the paper's own spread.

* **Background noise.** "Neural noise ... generated in the network by the
  constant application of excitatory current, randomly sampled from a uniform
  distribution in the range [0 : Inoise] where ``Inoise = 0.8``", giving ~0.1 Hz
  spontaneous firing. Previously absent.

* **Learning protocol.** "active place cells fire stochastically in each theta
  phase window for a period of 1s" at a "mean in-field firing rate of ~15Hz",
  over "ten traversals of a route".

* **Recall window.** Hetero-associative sequence recall is scored "over a period
  of ~400ms"; the 20 ms figure is the *auto-associative* pattern-completion
  window. The spec's "~33 ms sharp-wave-ripple window" appears nowhere in the
  paper, and the previous version's reported 33-vs-40-step "deviation" was
  measured against a number the spec invented.

* **``I_cue = 30`` for a single 1 ms step** -- the one recall parameter the spec
  had right.

Still deviating, and stated rather than absorbed
------------------------------------------------
* **Place fields do not overlap here.** The paper's rat traverses overlapping
  fields, which is what produces phase precession and hence the *temporal* half
  of the dual code. Fields are presented as successive non-overlapping windows,
  so the asymmetric weights come from field order alone. This is the largest
  remaining gap.
* **Plasticity is not theta-phase modulated** (potentiation at peak, depression
  at trough). Deferred, as in the spec.
* Spike detection is per 0.5 ms substep, so a neuron emits at most one spike per
  1 ms step; Izhikevich's two-substep scheme is used because the single-step
  form is unstable at these parameters.

Known limitations (spec S2.6) -- recorded up front
--------------------------------------------------
* **Not one-shot.** Needs ~10 presentations. The rapid one-shot section should
  be expected to fail; that is a result about a Hebbian arm, not a bug.
* Native readout is temporal order fidelity over a sharp-wave-ripple window,
  which is span-*like* but not identical to ``max_memory_span``. Do not silently
  equate them.
"""

from typing import Optional

import numpy as np

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode
from ..codec_wrapper import OnlineCodecWrappedModel
from ...encoders.spike_codec import PopulationSpikeEncoder

#: STDP parameter sets. ``bcm`` is the default (spec S2.2); ``non_bcm`` is the
#: paper's ablation, kept because "which pairing rule" is the first thing to
#: vary if the correctness gate fails.
STDP_VARIANTS = {
    "bcm":     dict(tau_plus=20.0, tau_minus=50.0, a_plus=0.02, a_minus=-0.01),
    "non_bcm": dict(tau_plus=20.0, tau_minus=20.0, a_plus=0.02, a_minus=-0.021),
}


class BushSTDPSpikingNetwork(HippocampalModel, OnlineTrainable):
    """Spiking recurrent network trained by pair-based STDP.

    Args:
        n_neurons: Network size. **Set by the codec, not chosen**:
            ``2 * D * n_per_feature``. Recurrent connectivity is O(N^2), so this
            is a runtime parameter as much as a capacity one.
        n_presentations: Repeats of a sequence in ``fit_sequence``. **This is
            the arm's epoch**: one presentation is one pass over the material,
            the same unit every rate arm's ``n_epochs`` denotes, so the arm goes
            on a criterion-referenced staircase unchanged. ``n_epochs`` is an
            alias for it -- readable, settable, and accepted as a constructor
            kwarg -- and ``fit_sequence`` honours a call-level ``epochs=``. Both
            paths are wired because the suite drives them both: ``_fit`` helpers
            set the attribute *and* pass the kwarg, and silently swallowing
            either is the defect ``tests/test_epochs_kwarg_contract.py`` exists
            to catch.
        recall_steps: Length of the readout window, in steps. **Defaults to 40,
            not the paper's ~33.** This reimplementation replays the full
            20-field sequence in correct order but takes about **36 steps** to
            do it, so a 33-step window truncates the last two or three fields.
            Scored over 50 seeds: 0.821 at 33 steps (17.0 of 20 fields), 0.958
            at 40 (19.6 of 20). The deviation is in the *speed* of replay, not
            its order, and it is stated rather than absorbed -- a section
            comparing against the paper's compression ratio must use 33 and
            report 0.82.
        variant: ``"bcm"`` (default) or ``"non_bcm"``.
        cue_drive: ``"binary"`` injects ``i_cue`` into every population that
            spikes at all in the cue (the paper's protocol). ``"rate"`` scales
            the injection by each population's spike count, which is the knob to
            reach for if graded upstream corruption turns out to be invisible to
            the arm -- binary cueing discards magnitude by construction.
        n_presynaptic: If set, each neuron keeps this many random presynaptic
            partners instead of being fully connected.
        g_syn: Scale on the weight-to-current conversion. Defaults to 1.0
            because the paper's weights **are** currents: ``w_max = 1`` and
            "a single synaptic current of I = 16.5 is required" to fire a
            neuron, so the anchor is the paper's, not a fitted gain.
        i_noise: Background excitatory current, uniform ``[0, i_noise]`` per
            neuron per step. Paper: 0.8, giving ~0.1 Hz spontaneous firing.
        i_inh: Amplitude of the theta-modulated inhibitory drive; the mean is
            ``-i_inh * theta`` with ``theta`` sinusoidal in [0, 1]. Paper: 15.
            **Bounded by construction** -- it is an external drive, not
            feedback, so no level of network activity can make it run away.
        sigma_inh: SD of that Gaussian. Paper: 2.
        theta_hz: Frequency of the theta oscillation. Paper: 8 Hz.
        axonal_delay_max: ``D``. Each neuron gets a delay drawn once, uniform in
            ``[1, D]`` ms. Paper: 5.
    """

    #: `recall` feeds the arm's own output spikes back as the next cue. There is
    #: no vector-space cleanup inside the arm; when it runs behind
    #: ``CodecWrappedModel`` the rollout is that wrapper's, in R^D.
    rollout_mode = RolloutMode.OBSERVATION

    #: ``predict_next`` is a pure function of the cue and carries nothing
    #: between probes, so a longer prompt establishes no extra state.
    prompt_conditioned = False

    #: ``fit_sequence`` is exactly ``fit_event`` over the items, repeated.
    online_equivalent = True

    def __init__(self, n_neurons: int,
                 n_presentations: int = 10,
                 recall_steps: int = 400,
                 w_max: float = 1.0,
                 w_init_frac: float = 0.01,
                 variant: str = "bcm",
                 dt_ms: float = 1.0,
                 substeps: int = 2,
                 g_syn: float = 1.0,
                 i_noise: float = 0.8,
                 i_inh: float = 15.0,
                 sigma_inh: float = 2.0,
                 theta_hz: float = 8.0,
                 i_drive: float = 30.0,
                 i_cue: float = 30.0,
                 cue_steps: int = 1,
                 cue_drive: str = "binary",
                 axonal_delay_max: int = 5,
                 n_presynaptic: Optional[int] = None,
                 izh_a: float = 0.02, izh_b: float = 0.2,
                 izh_c: float = -65.0, izh_d: float = 6.0,
                 seed: Optional[int] = None, **kwargs):
        if variant not in STDP_VARIANTS:
            raise ValueError(f"variant must be one of {tuple(STDP_VARIANTS)}.")
        if cue_drive not in ("binary", "rate"):
            raise ValueError("cue_drive must be 'binary' or 'rate'.")
        if axonal_delay_max < 1:
            raise ValueError("axonal_delay_max must be >= 1 (a zero-delay chain "
                             "cannot order itself in time).")

        self.n_neurons = int(n_neurons)
        #: Alias so benchmark code that reads ``n_features`` sees the spike
        #: dimensionality. This arm's "features" ARE its neurons.
        self.n_features = self.n_neurons
        # MODEL_REGISTRY supplies ``n_epochs`` to every arm's constructor; take
        # it as the alias it is rather than dropping it into **kwargs.
        self.n_presentations = int(kwargs.pop("n_epochs", n_presentations))
        self.recall_steps = int(recall_steps)
        self.w_max = float(w_max)
        self.variant = variant
        self.dt_ms = float(dt_ms)
        self.substeps = int(substeps)
        self.g_syn = float(g_syn)
        self.i_noise = float(i_noise)
        self.i_inh = float(i_inh)
        self.sigma_inh = float(sigma_inh)
        self.theta_hz = float(theta_hz)
        self.i_drive = float(i_drive)
        self.i_cue = float(i_cue)
        self.cue_steps = int(cue_steps)
        self.cue_drive = cue_drive
        self.axonal_delay_max = int(axonal_delay_max)
        self.a, self.b, self.c, self.d = izh_a, izh_b, izh_c, izh_d
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        p = STDP_VARIANTS[variant]
        self.tau_plus, self.tau_minus = p["tau_plus"], p["tau_minus"]
        self.a_plus = p["a_plus"] * self.w_max
        self.a_minus = p["a_minus"] * self.w_max
        self.decay_plus = float(np.exp(-self.dt_ms / self.tau_plus))
        self.decay_minus = float(np.exp(-self.dt_ms / self.tau_minus))

        N = self.n_neurons
        # W[i, j]: weight of the connection FROM j (pre) TO i (post).
        self.W = np.full((N, N), w_init_frac * self.w_max)
        np.fill_diagonal(self.W, 0.0)
        self.mask = np.ones((N, N))
        np.fill_diagonal(self.mask, 0.0)
        if n_presynaptic is not None:
            if not 1 <= n_presynaptic < N:
                raise ValueError("n_presynaptic must be in [1, n_neurons).")
            self.mask = np.zeros((N, N))
            for i in range(N):
                choices = np.delete(np.arange(N), i)
                self.mask[i, self.rng.choice(choices, n_presynaptic, replace=False)] = 1.0
        self.n_presynaptic = n_presynaptic
        self.W *= self.mask

        # Per-neuron axonal delay, uniform in [1, D] ms (paper's Methods).
        self.delays = self.rng.integers(1, self.axonal_delay_max + 1, size=N)

        self.reset_context()
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ meta
    @property
    def n_epochs(self) -> int:
        """Alias for :attr:`n_presentations` -- one pass over the material.

        Exposure is the suite's common unit, and this arm's unit is the same
        one: ``epochs_to_criterion`` documents ``fit(model, epochs)`` as
        "trains for exactly ``epochs`` passes", and a presentation is a pass.
        What differs from a rate arm is how much learning a pass *contains* --
        one plasticity event per spike here versus one per transition there,
        about 55x more on an 8-item list -- and that multiplier is set by the
        codec's ``window_steps`` and ``r_max``, not by the sequence. It is
        therefore a stated codec condition (spec S2.5), not an exposure unit.
        """
        return self.n_presentations

    @n_epochs.setter
    def n_epochs(self, value: int) -> None:
        self.n_presentations = int(value)

    @property
    def params(self) -> dict:
        return {
            "model": "BushSTDPSpikingNetwork",
            "n_neurons": self.n_neurons,
            "n_presentations": self.n_presentations,
            "recall_steps": self.recall_steps,
            "variant": self.variant,
            "g_syn": self.g_syn, "i_noise": self.i_noise,
            "i_inh": self.i_inh, "sigma_inh": self.sigma_inh,
            "theta_hz": self.theta_hz,
            "axonal_delay_max": self.axonal_delay_max,
            "i_drive": self.i_drive, "i_cue": self.i_cue,
            "cue_drive": self.cue_drive,
            "n_presynaptic": self.n_presynaptic,
            "seed": self.seed,
        }

    def named_parameters(self) -> dict:
        return {"W": self.W}

    # ---------------------------------------------------------------- engine
    def _blank_dynamics(self) -> dict:
        N = self.n_neurons
        return {
            "v": np.full(N, self.c),
            "u": np.full(N, self.b * self.c),
            "x_pre": np.zeros(N),
            "y_post": np.zeros(N),
            # Ring buffer of pending presynaptic spikes, one slot per ms of
            # possible delay. Neuron j's spike is deposited in slot
            # (t + delays[j]); the slot due now is read and cleared each step.
            "delay": np.zeros((self.axonal_delay_max + 1, N)),
            "t": np.array(0.0),          # ms since reset, drives the theta clock
        }

    def _save_dynamics(self) -> dict:
        return {k: v.copy() for k, v in self._state.items()}

    def theta(self, t_ms: float) -> float:
        """LFP proxy: sinusoidal in [0, 1] at ``theta_hz`` (paper's variable)."""
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * self.theta_hz * t_ms / 1000.0))

    def _step(self, i_ext: np.ndarray, plastic: bool,
              rng: Optional[np.random.Generator] = None,
              inhibition: bool = True) -> np.ndarray:
        """Advance one ``dt_ms`` step. Returns the boolean spike vector.

        Args:
            i_ext: External drive this step.
            plastic: The ACh gate ``Phi`` -- 1 during encoding, 0 during recall.
            rng: Source for the noise and inhibition draws. ``predict_next``
                passes a freshly seeded generator so probes stay pure.
            inhibition: Theta-modulated inhibitory drive. The paper ceases it
                during recall, so ``predict_next`` passes False.
        """
        s = self._state
        gen = self.rng if rng is None else rng

        # Synaptic current: current-based and instantaneous within the step.
        # Spikes are 0/1, so the matvec is a column sum over who is arriving --
        # O(N * n_arriving) rather than O(N^2).
        slot = int(s["t"]) % (self.axonal_delay_max + 1)
        arriving = s["delay"][slot]
        i_syn = np.zeros(self.n_neurons)
        if arriving.any():
            pre = np.flatnonzero(arriving)
            i_syn = self.g_syn * self.W[:, pre].sum(axis=1)
        s["delay"][slot] = 0.0

        I = i_ext + i_syn + gen.uniform(0.0, self.i_noise, self.n_neurons)
        if inhibition:
            # Bounded external drive, NOT feedback: the mean is -i_inh*theta
            # with theta in [0, 1], so it can never exceed -i_inh however
            # active the network is. That is why the paper's model does not
            # need an activity-proportional pool, and why it cannot seize.
            I = I + gen.normal(-self.i_inh * self.theta(float(s["t"])),
                               self.sigma_inh, self.n_neurons)

        s["x_pre"] *= self.decay_plus
        s["y_post"] *= self.decay_minus

        fired = np.zeros(self.n_neurons, dtype=bool)
        h = self.dt_ms / self.substeps
        for _ in range(self.substeps):
            v, u = s["v"], s["u"]
            v += h * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            hot = v >= 30.0
            if hot.any():
                fired |= hot
                v[hot] = self.c
                u[hot] += self.d
        s["u"] += self.dt_ms * self.a * (self.b * s["v"] - s["u"])

        if plastic and fired.any():
            idx = np.flatnonzero(fired)
            # LTP: these neurons just spiked as POST; pair with the pre traces
            # standing before this step's own spikes are added.
            self.W[idx, :] = np.clip(self.W[idx, :] + self.a_plus * s["x_pre"][None, :],
                                     0.0, self.w_max) * self.mask[idx, :]
            # LTD: the same neurons spiked as PRE; pair with the post traces.
            self.W[:, idx] = np.clip(self.W[:, idx] + self.a_minus * s["y_post"][:, None],
                                     0.0, self.w_max) * self.mask[:, idx]

        s["x_pre"][fired] += 1.0
        s["y_post"][fired] += 1.0
        if fired.any():
            j_fired = np.flatnonzero(fired)
            slots = (int(s["t"]) + self.delays[j_fired]) % (self.axonal_delay_max + 1)
            s["delay"][slots, j_fired] += 1.0
        s["t"] += self.dt_ms
        return fired

    def _run_window(self, drive: np.ndarray, plastic: bool,
                    rng: Optional[np.random.Generator] = None,
                    inhibition: bool = True) -> np.ndarray:
        """Run ``drive`` -- ``(n_neurons, T)`` of input currents -- for T steps."""
        T = drive.shape[1]
        out = np.zeros((self.n_neurons, T), dtype=bool)
        for t in range(T):
            out[:, t] = self._step(drive[:, t], plastic, rng=rng,
                                   inhibition=inhibition)
        return out

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

    # -------------------------------------------------------------- training
    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None,
                  **kwargs):
        """Present one item's spike train with the ACh gate open (``Phi = 1``).

        Membrane state and STDP traces persist across calls, which is exactly
        where the across-item pairings that build the sequence come from. Call
        ``on_event_boundary`` at a seam so no transition forms across it.
        """
        self._run_window(self.i_drive * self._as_spikes(event), plastic=True)

    def fit_sequence(self, sequence_data: np.ndarray,
                     context_data: Optional[np.ndarray] = None,
                     epochs: Optional[int] = None, **kwargs):
        """Present a sequence ``n_presentations`` times.

        Args:
            sequence_data: ``(L, n_neurons, T)`` -- one spike window per item,
                the form ``PopulationSpikeEncoder.encode_stack`` produces -- or
                ``(n_neurons, T_total)``, one continuous train presented whole.
                The rank disambiguates; there is no window length to guess.
            epochs: Passes for this call, overriding ``n_presentations``. One
                pass is one presentation -- see :attr:`n_epochs`.
        """
        X = np.asarray(sequence_data)
        items = [X] if X.ndim == 2 else list(X)
        n_passes = self.n_presentations if epochs is None else int(epochs)
        for _ in range(n_passes):
            # Reset between passes so the last item does not bind to the first.
            self.reset_context()
            for item in items:
                self.fit_event(item)

    # --------------------------------------------------------------- readout
    def _cue_current(self, cue: np.ndarray) -> np.ndarray:
        counts = self._as_spikes(cue).sum(axis=1)
        if self.cue_drive == "binary":
            return self.i_cue * (counts > 0).astype(float)
        peak = counts.max()
        if peak <= 0:
            return np.zeros(self.n_neurons)
        return self.i_cue * (counts / peak)

    def predict_next(self, current_event: np.ndarray,
                     current_context: Optional[np.ndarray] = None,
                     **kwargs) -> np.ndarray:
        """Cue the network and return the recall window's spikes.

        Injects ``i_cue`` for ``cue_steps`` into the cue's active populations,
        then runs ``recall_steps`` with the ACh gate **shut** (``Phi = 0``) and
        the theta inhibitory drive ceased, both per the paper's recall protocol.

        Pure function of the cue, by construction: the gate keeps ``W`` fixed,
        and membrane state, traces and the axonal delay line are saved before
        the call and restored after it. Without that, membrane state would leak
        between the independent probes ``measure_recall_associative`` assumes
        are independent -- the defect ``MultilayerTemporalPCNetwork`` had to
        avoid.

        Returns:
            ``(n_neurons, recall_steps)`` boolean. Deliberately **not** reduced
            to a vector: that decode choice belongs to
            ``PopulationSpikeDecoder``, not to the arm.
        """
        saved = self._save_dynamics()
        try:
            self._state = self._blank_dynamics()
            drive = np.zeros((self.n_neurons, self.recall_steps))
            drive[:, :self.cue_steps] = self._cue_current(current_event)[:, None]
            # Theta inhibition ceases during recall (paper's Methods). Noise
            # does not, so the probe draws from a freshly seeded generator --
            # common random numbers across probes, which is what keeps
            # predict_next a pure function of the cue.
            return self._run_window(drive, plastic=False,
                                    rng=np.random.default_rng(self._probe_seed()),
                                    inhibition=False)
        finally:
            self._state = saved

    def _probe_seed(self) -> int:
        return 0 if self.seed is None else int(self.seed) + 104729

    def recall(self, prompt_event: np.ndarray, length: int,
               prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Autoregressive rollout **in spike space**: ``(length, N, recall_steps)``.

        Each step's output spikes become the next cue. The shape does not match
        the ``(length, Features)`` contract of the base class because this arm's
        features are spike trains; benchmarks should roll out through
        ``CodecWrappedModel.recall``, which does it in R^D where every other arm
        is scored. Provided so the arm is usable standalone.
        """
        prompt = np.asarray(prompt_event)
        cue = prompt[-1] if prompt.ndim > 2 else prompt
        out = np.zeros((length, self.n_neurons, self.recall_steps), dtype=bool)
        for t in range(length):
            cue = self.predict_next(cue)
            out[t] = cue
        return out

    # ----------------------------------------------------------------- state
    def get_latent_state(self) -> dict:
        return {"W": self.W.copy(),
                "v": self._state["v"].copy(),
                "u": self._state["u"].copy(),
                "x_pre": self._state["x_pre"].copy(),
                "y_post": self._state["y_post"].copy()}

    def reset_context(self):
        """Clear membrane state, traces and the delay line. ``W`` is untouched."""
        self._state = self._blank_dynamics()


# --------------------------------------------------------------------------
# The paper's own task and readout.
#
# These are NOT a MemVal section and must not be confused with one: no encoder,
# no vocabulary, no cosine, no MRR. They exist so the reimplementation can be
# held to the paper's own result -- >90% recall fidelity on the dual-coded
# configuration -- **before** its scores on MemVal sections are allowed to mean
# anything. ``bin/bush_stdp_gate.py`` runs the gate.
# --------------------------------------------------------------------------

def dual_coded_stimulus(n_fields: int = 20, n_per_field: int = 5,
                        field_ms: int = 1000, step_ms: int = 250,
                        rate_hz: float = 15.0, theta_hz: float = 8.0,
                        phase_kappa: float = 4.0, dt_ms: float = 1.0,
                        seed: Optional[int] = 0) -> np.ndarray:
    """The paper's dual-coded configuration: 20 fields x 5 neurons, theta-coded.

    Fields **overlap**, and cells phase-precess. Both matter, and an earlier
    version of this fixture had neither -- it presented successive fields as
    disjoint 50 ms windows, and the resulting weight matrix had within-field
    weights three times the forward ones (0.064 vs 0.022) because successive
    fields were a whole second apart and essentially never paired inside the
    20 ms STDP window. Non-overlapping fields do not simplify this model; they
    delete the mechanism it is named for.

    What the dual code does here: the animal traverses a route at constant
    speed, so several fields are active at once (``field_ms / step_ms`` of
    them). Within each ~125 ms theta cycle every active cell fires near a
    preferred phase set by its progress through its own field -- late on entry,
    early on exit, i.e. classical phase precession. A theta cycle therefore
    sweeps out the *upcoming sequence of fields in order, compressed to a few
    milliseconds per field*, and repeats it 8 times a second for the whole
    traversal. That is what puts successive fields inside ``tau_plus`` of each
    other, over and over, and it is what builds the asymmetric weights.

    Rate coding rides on top: firing is enveloped by progress through the field,
    giving the "mean in-field firing rate of ~15Hz" the paper reports.

    Args:
        field_ms: How long one field stays active. Paper: 1 s.
        step_ms: Spacing between successive field onsets. ``field_ms/step_ms``
            fields are co-active; 4 at the defaults.
        rate_hz: Mean in-field firing rate. Paper: ~15 Hz.
        phase_kappa: Concentration of the phase preference. Larger is a tighter
            phase-locked packet and a sharper sequence sweep.

    Returns:
        ``(n_fields, n_fields * n_per_field, T)`` boolean, one presentation
        window per field for ``fit_sequence`` -- each window is that field's
        onset step, so the arm sees the traversal in order and the overlap
        appears as the tail of earlier fields still firing.
    """
    rng = np.random.default_rng(seed)
    N = n_fields * n_per_field
    total = (n_fields - 1) * step_ms + field_ms
    t = np.arange(0, total, dtype=float)
    theta_phase = (t * theta_hz / 1000.0) % 1.0                    # [0, 1)

    rates = np.zeros((n_fields, len(t)))
    for k in range(n_fields):
        onset = k * step_ms
        prog = (t - onset) / field_ms                              # in-field progress
        active = (prog >= 0.0) & (prog < 1.0)
        envelope = np.where(active, np.sin(np.pi * np.clip(prog, 0, 1)), 0.0)
        preferred = 1.0 - np.clip(prog, 0.0, 1.0)                  # precesses late -> early
        bump = np.exp(phase_kappa *
                      (np.cos(2 * np.pi * (theta_phase - preferred)) - 1.0))
        rates[k] = envelope * bump

    # Scale so the mean rate *while a field is active* is rate_hz.
    active_mask = rates > 0
    if active_mask.any():
        rates *= rate_hz / (rates[active_mask].mean() * 1000.0 / dt_ms) * 1000.0 / dt_ms
    p = np.clip(rates * dt_ms / 1000.0, 0.0, 1.0)

    spikes = np.zeros((N, len(t)), dtype=bool)
    for k in range(n_fields):
        rows = slice(k * n_per_field, (k + 1) * n_per_field)
        spikes[rows] = rng.random((n_per_field, len(t))) < p[k]

    # One window per field onset, so fit_sequence presents the traversal in
    # order; the overlap shows up as earlier fields still firing in later
    # windows, which is exactly the co-activity the phase code needs.
    out = np.zeros((n_fields, N, step_ms), dtype=bool)
    for k in range(n_fields):
        seg = spikes[:, k * step_ms:(k + 1) * step_ms]
        out[k, :, :seg.shape[1]] = seg
    return out


def dual_coded_cue(field: int, n_fields: int = 20, n_per_field: int = 5,
                   steps: int = 1) -> np.ndarray:
    """A clean activation of one field's population, for cueing recall.

    The paper cues recall with "superthreshold excitation ... to a small number
    of randomly selected neurons" *of the pattern*, not with a slice of the
    traversal. That distinction is load-bearing here: a traversal window taken
    at a field's onset contains almost none of that field's own spikes, because
    the rate envelope peaks mid-field. Cueing with window 0 of
    ``dual_coded_stimulus`` therefore cues nothing at all.

    Returns:
        ``(n_fields * n_per_field, steps)`` boolean.
    """
    N = n_fields * n_per_field
    cue = np.zeros((N, steps), dtype=bool)
    cue[field * n_per_field:(field + 1) * n_per_field] = True
    return cue


def order_fidelity(spikes: np.ndarray, n_fields: int = 20, n_per_field: int = 5
                   ) -> dict:
    """Temporal order fidelity of a replay window -- the paper's readout.

    A field's replay time is the **first** step at which any of its neurons
    fires. Fidelity is the fraction of consecutive pairs ``(k, k+1)`` that both
    replayed and did so in the right order. A field that never fires inside the
    window counts as a **failure**, not as a skipped pair: a replay that stops
    after five fields has not recalled the sequence, and averaging only over
    what did fire would hide exactly that.

    This is span-*like* but is **not** ``max_memory_span`` -- do not equate them.

    Returns:
        ``{"fidelity", "n_replayed", "span_steps", "first_spike"}``.
    """
    S = np.asarray(spikes)
    t = np.full(n_fields, np.nan)
    for k in range(n_fields):
        block = S[k * n_per_field:(k + 1) * n_per_field]
        idx = np.nonzero(block.any(axis=0))[0]
        if idx.size:
            t[k] = idx[0]
    replayed = ~np.isnan(t)
    correct = sum(1 for k in range(n_fields - 1)
                  if replayed[k] and replayed[k + 1] and t[k] < t[k + 1])
    span = float(np.nanmax(t) - np.nanmin(t)) if replayed.sum() > 1 else float("nan")
    return {"fidelity": correct / (n_fields - 1),
            "n_replayed": int(replayed.sum()),
            "span_steps": span,
            "first_spike": t}


class CodecBushNetwork(OnlineCodecWrappedModel):
    """Registry-facing form: :class:`BushSTDPSpikingNetwork` behind the codec.

    ``MODEL_REGISTRY`` constructs every arm as ``cls(n_features=D, **kwargs)``,
    but this arm consumes spikes and its network size is *derived*
    (``N = 2 * D * n_per_feature``), so it cannot be registered directly. This
    class is the adapter: it builds the codec, sizes the network from it, and
    presents the ordinary vector interface.

    Ingestion is streamed (``fit_event``); the arm is registered under
    ``online_symbolic``. ``resample_per_pass`` makes that byte-identical to the
    batch path, so the regime is a statement of protocol rather than a change of
    result.

    **The defaults will not transfer.** ``g_syn`` and ``k_inh`` sit on a
    stability ridge whose coordinates move with the number of co-active neurons,
    so every new configuration needs re-siting with
    ``bin/bush_stdp_siting_sweep.py``. The values here are the best known point
    at comparable network size (N=1560), not a fit to the registry's own
    substrate -- on which, see the class note below.

    What to expect on ``online_symbolic``
    -------------------------------------
    A floor, and an attributable one. That pipeline's substrate is
    ``SymbolicEncoder(embedding_dim=100, category_variance=0.2)``, whose rows are
    **dense**: every dimension carries magnitude, so about 25% of the network
    fires for *every* item and any two items share ~21% of their active neurons
    (population Jaccard). Measured against the alternatives:

    ==========================  ==========  =============  =============
    material                    mean |cos|  pop. Jaccard   active frac
    ==========================  ==========  =============  =============
    ``SymbolicEncoder`` D=100   0.091       **0.213**      **0.247**
    ``HierarchicalEncoder``     0.214       0.144          0.037
    orthogonal blocks           0.000       0.000          0.036
    ==========================  ==========  =============  =============

    Note that cosine and population overlap **rank these materials
    differently**: by cosine the registry's substrate is the easiest of the
    three, by population overlap it is the hardest. This arm tracks the second.
    That is mechanical rather than mysterious -- STDP builds a chain out of
    item-specific *populations*, and a code in which every item drives a quarter
    of the network has no item-specific populations to chain. Cosine is the
    wrong overlap axis for a spiking arm, and reporting one without the other
    will mislead.

    ``HierarchicalEncoder`` is the substrate the spec names as the reference
    (S3.1) and the one on which the codec clears its own fidelity gate; a
    section that wants this arm to be doing anything other than flooring should
    run there.
    """

    #: Same as the wrapped arm's: the rollout feeds the arm's own output back.
    rollout_mode = RolloutMode.OBSERVATION
    prompt_conditioned = False
    online_equivalent = True

    def __init__(self, n_features: int,
                 n_per_feature: int = 10,
                 window_steps: int = 50,
                 dt_ms: float = 1.0,
                 r_max: float = 200.0,
                 codec_mode: str = "rate",
                 probe_seed: int = 0,
                 seed: Optional[int] = 42,
                 **kwargs):
        encoder = PopulationSpikeEncoder(
            n_features, n_per_feature=n_per_feature, window_steps=window_steps,
            dt_ms=dt_ms, r_max=r_max, mode=codec_mode, seed=seed)
        inner = BushSTDPSpikingNetwork(n_neurons=encoder.n_neurons, seed=seed,
                                       **kwargs)
        super().__init__(inner, encoder, inner_domain="spikes",
                         probe_seed=probe_seed)
