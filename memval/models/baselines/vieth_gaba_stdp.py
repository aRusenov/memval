"""Vieth & Triesch -- GABA-modulated STDP sequence learning.

*Stabilizing sequence learning in stochastic spiking networks with GABA-Modulated STDP*
(Neural Networks 183, 2025). Upstream code: ``gitmv/GABA_Modulated_STDP_Paper``
(MIT), running on PymoNNto.

This is a **reimplementation against a reference**, the same provenance class as
EP, tPC and DTS-ESN: the upstream repo is MIT and runnable, but PymoNNto pulls
PyQt5, paramiko and scp, so depending on it to run one arm is not proportionate.
The upstream network is instead cloned to ``scratch/vieth_repo`` and driven by
``scratch/vieth_ref.py``, and ``tests/test_vieth_gaba_stdp.py`` holds this port
to it **componentwise and exactly** -- the plasticity, the normalisation, the
intrinsic-plasticity and the spike rule are each checked against a recorded
upstream trace, so "matches upstream" is a measurement here and not a claim.

Why this arm exists
-------------------
It fills the roster's spiking x online cell. Every other spiking arm we have
either ships no code (Bush), does not learn online in our regime, or is a
spiking substrate under a rate interface (spiking EP). This one learns by a
local, one-step, spike-pair rule applied on **every** network iteration, and its
native task is exactly ``predict_next``: upstream trains on a character stream
and then free-runs to generate text.

What the model is
-----------------
Three populations and four weight matrices, all as upstream:

======  =====================  =========================================
``ES``  input -> excitatory    "SOMA". Binds an item to its assembly, and
                               is also the **readout**: the reconstruction
                               is ``ES^T . exc_spike``.
``EE``  excitatory -> exc.     "DISTAL". The sequence memory proper; this
                               is the x_t -> x_{t+1} matrix.
``IE``  excitatory -> inhib.
``EI``  inhibitory -> exc.     Delayed one step, and it gates plasticity.
======  =====================  =========================================

Excitatory units are **memoryless and stochastic-threshold**, not leaky
integrators::

    spike = clip(voltage * mul, 0)^exp  >  U(0, 1)      # then voltage <- 0

There is no membrane time constant, no reset, no refractory period: voltage is
accumulated within an iteration from the three synaptic inputs plus the
intrinsic-plasticity bias, used, and zeroed. Whether that clears a given
reader's bar for "spiking" is a judgement, so it is stated here rather than
buried: the units emit binary events and the learning rule is a spike-pair rule,
but the substrate carries no sub-threshold dynamics. Compare
``BushSTDPSpikingNetwork`` (Izhikevich, real membrane state) and
``SpikingEqPropSequenceNetwork`` (spikes only on the inter-layer channel).

Plasticity -- the paper's contribution
--------------------------------------
A one-step causal STDP window, gated by momentary inhibition::

    li_stdp_mul = clip((1 + input_GABA / avg_inh) * gaba_strength, min, max)
    W[pre_active(t-2), post_active(t-1)] += eta * li_stdp_mul[post_active]
    W = clip(W, 0, None)

``input_GABA`` is negative, so a strongly inhibited postsynaptic cell has its
potentiation reduced and -- because ``min`` is **-0.15**, not 0 -- can be pushed
into net depression. That sign flip is the paper's mechanism; setting
``gaba_min=0.0`` is the ablation that removes it.

Two facts about the rule that decide how material must be presented:

1. **The window is exactly one iteration.** There is no eligibility trace. So
   the only pairing that crosses an item boundary is (last frame of item t,
   first frame of item t+1); everything else is a within-item self-pair.
2. **Autapses are not removed** (upstream leaves ``remove_autapses`` commented
   out), so a unit active in consecutive frames potentiates its own recurrent
   weight.

Together these are why ``presentation_steps`` defaults to **1**. Upstream
presents one symbol per network iteration (``iterations_per_char=1``); at that
setting the cross-item pairing is 1 of 1. Presenting a codec window's full 50
steps instead makes it 1 of 50, and the self-pairs win. The parameter is
exposed, not hidden, because "how many frames per item" is a protocol choice
that changes the result and therefore belongs in the record.

Where the codec meets the model
-------------------------------
The arm consumes and emits **spike arrays in codec-neuron space**, so it runs
behind ``memval.models.codec_wrapper.wrap_with_codec`` like the Bush arm; see
``docs/spike_codec_spec.md``.

Upstream's input layer is a ``Grid(width=10, height=n_chars)`` -- ten duplicate
neurons per symbol -- and its readout sums the reconstruction over ``width`` to
get one activation per symbol. The codec's layout is the same shape: ``D``
features x 2 polarities x ``n_per_feature`` duplicates, and
``PopulationSpikeDecoder.counts`` sums over exactly that duplicate axis. So the
codec substitutes for upstream's input grid **without changing the readout** --
the arm returns ``ES^T . exc_spike`` per recall step and the decoder reduces it,
which is upstream's ``TextReconstructor`` with its argmax removed (a decode
choice that, per spec S2.4, must not live in the arm).

One consequence to state plainly: unlike Bush, whose output is literally spikes
in codec space, **this arm's output passes through a learned linear map**
(``ES``). That is upstream's own readout and not our addition, but it means the
arm carries a trained decoder, so a section that compares it against a linear
reference should say so -- the same caveat that applies to AHN's pseudoinverse
fit.

Two measured facts that decide how to read any score
----------------------------------------------------
**1. The arm is bimodal across seeds, and that is the model, not the port.**
On upstream's own character task at ``n_exc=600``, upstream itself regenerates
its training text on 5 of 10 seeds and emits noise at the correct firing rate on
the other 5 -- text scores 3.99-4.13 against 1.72-2.07, with nothing in between.
This port lands 3 of 8, with matching score ranges. So a single seed is a coin
flip: **report a distribution over seeds, never a mean and never one run.**
``bin/vieth_stdp_gate.py`` is built around the distribution for that reason.

**2. Free-running needs a settled homeostat, not just learned weights.**
``ES`` supplies most of the excitatory drive while input is present, so a
network whose intrinsic-plasticity bias was tuned *with* the input sits far
below threshold once the input stops. Upstream's recovery phase is where that
bias is re-tuned for the input-free regime, and it is not optional. Diagnosing a
silent recall as "the chain did not learn" is the trap here: transplanting
upstream's final state into this port reproduced upstream's generation exactly
(rate 0.06633 vs 0.06636, score 4.125 vs 4.127) while the port's *own* training
run, with structurally identical weights, would not free-run -- the whole
difference was a bias of -0.64 against ~0. See ``docs/vieth_stdp_port.md``.

Deviations from upstream, stated rather than absorbed
-----------------------------------------------------
* **Input is the codec's population code, not a one-hot grid.** Upstream's
  symbols are disjoint by construction; codec items overlap. This is the axis
  the Bush arm floors on and the one to check first.
* **No text-specific machinery.** ``TextGenerator``, ``TextReconstructor``'s
  argmax, the token/bar experiments and the evolutionary scoring are not ported;
  they are the harness upstream scores with, and MemVal has its own.
* **Weight initialisation** uses this arm's ``rng`` rather than the global NumPy
  RNG upstream's PymoNNto backend draws from, so runs are seedable per instance.
  The Sinkhorn-style ``CreateWeights`` normalisation is reproduced exactly.
* Hyperparameters are upstream's evolved floats, carried verbatim. They were
  evolved against a **three-sentence** text, and they do not transfer to a
  shorter one by making the network bigger: at upstream's published
  ``n_exc=2400`` / 60000 steps, upstream and this port both fail the
  one-sentence task (1.81 and 1.81-1.86). ``bin/vieth_stdp_gate.py`` re-sites
  them for any other configuration.
"""

from typing import Optional

import numpy as np

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode
from ..codec_wrapper import OnlineCodecWrappedModel
from ...encoders.spike_codec import PopulationSpikeEncoder

#: Upstream's evolved parameters (``Experiments/Experiment_char.py``), verbatim.
#: They are floats from a parameter search, not round numbers, and are kept at
#: full precision so a diff against upstream is meaningful.
UPSTREAM = dict(
    ip_strength=0.008735764741458582,
    gaba_strength=6.450234496564654,
    avg_inh=0.3427857658747104,
    gaba_min=-0.15,
    gaba_max=1.0,
    stdp_strength=0.0030597477411211885,
    exc_exp=0.7378726012049153,
    exc_mul=2.353594052973287,
    inh_duration=2.0,
    norm_every=200,
)


class ViethGabaSTDPNetwork(HippocampalModel, OnlineTrainable):
    """Recurrent stochastic-spiking network trained by GABA-modulated STDP.

    Args:
        n_neurons: Size of the **input** population. Set by the codec, not
            chosen: ``2 * D * n_per_feature``.
        n_exc: Excitatory units. Upstream uses 2400 for a 10-symbol alphabet.
        n_inh: Inhibitory units. Defaults to ``n_exc // 10``, upstream's ratio.
        target_activity: Homeostatic set-point, the fraction of excitatory units
            that should fire per iteration. Upstream sets it to
            ``1 / n_chars(grammar)`` -- one over the length of the training
            text, i.e. "one item's worth of the network". The MemVal analogue is
            ``1 / sequence_length``, so it is a **constructor argument with no
            safe default**; it is set to ``1 / 15`` here and
            :meth:`set_target_activity` exists so a pipeline can align it with
            the material it is about to present.
        presentation_steps: Network iterations per item. **1 is upstream's
            protocol**; see the module docstring for why more is not obviously
            better.
        recall_steps: Free-running iterations in ``predict_next``. The
            reconstruction is read at every one of them and returned as the
            readout window.
        cue_lag: **The network's pipeline depth**, which has two consequences
            rather than one. Input at ``t`` drives excitatory spikes at ``t+1``,
            whose recurrent effect appears at ``t+2``. So (a) the *prediction*
            starts at lag 2 and reading at lag 1 returns the cue's own item, and
            (b) a sequence seam must be preceded by this many draining
            iterations or the last items are never learned -- see
            :meth:`on_event_boundary`. One physical fact, so one parameter;
            changing it therefore changes training as well as readout.
        n_epochs: Passes over the material in ``fit_sequence``.
        gaba_min: Lower clip on the plasticity multiplier. **-0.15 is the
            paper's mechanism**; 0.0 is the ablation that removes it.
    """

    #: **Latent, not observation.** ``recall`` carries the recurrent excitatory
    #: state forward and reads the reconstruction out at each step without ever
    #: feeding it back. That is upstream's own generation protocol -- switch the
    #: input off and let the network free-run -- and it is also the only correct
    #: choice here: the readout is a real-valued weighted sum, not a spike
    #: pattern, so feeding it back through the input would require inventing a
    #: threshold and would drive every input neuron with a nonzero activation.
    rollout_mode = RolloutMode.LATENT

    #: ``recall`` presents **every** event in the prompt before free-running, so
    #: a longer prompt does establish more state. (``predict_next`` remains pure
    #: and unconditioned; this declaration is about ``recall``.)
    prompt_conditioned = True

    #: ``fit_sequence`` is exactly ``fit_event`` over the items, repeated.
    online_equivalent = True

    def __init__(self, n_neurons: int,
                 n_exc: int = 400,
                 n_inh: Optional[int] = None,
                 target_activity: float = 1.0 / 15.0,
                 presentation_steps: int = 1,
                 recall_steps: int = 1,
                 cue_lag: int = 2,
                 n_epochs: int = 20,
                 ip_strength: float = UPSTREAM["ip_strength"],
                 gaba_strength: float = UPSTREAM["gaba_strength"],
                 avg_inh: float = UPSTREAM["avg_inh"],
                 gaba_min: float = UPSTREAM["gaba_min"],
                 gaba_max: float = UPSTREAM["gaba_max"],
                 stdp_strength: float = UPSTREAM["stdp_strength"],
                 exc_exp: float = UPSTREAM["exc_exp"],
                 exc_mul: float = UPSTREAM["exc_mul"],
                 inh_duration: float = UPSTREAM["inh_duration"],
                 norm_every: int = UPSTREAM["norm_every"],
                 input_mode: str = "any",
                 seed: Optional[int] = None, **kwargs):
        if input_mode not in ("any", "frames"):
            raise ValueError("input_mode must be 'any' or 'frames'.")
        if presentation_steps < 1:
            raise ValueError("presentation_steps must be >= 1.")
        if cue_lag < 0:
            raise ValueError("cue_lag must be >= 0.")

        self.n_neurons = int(n_neurons)
        #: This arm's "features" are its input neurons, as for every codec arm.
        self.n_features = self.n_neurons
        self.n_exc = int(n_exc)
        self.n_inh = int(n_inh) if n_inh is not None else max(1, self.n_exc // 10)
        self.target_activity = float(target_activity)
        self.presentation_steps = int(presentation_steps)
        self.recall_steps = int(recall_steps)
        self.cue_lag = int(cue_lag)
        self.n_epochs = int(n_epochs)

        self.ip_strength = float(ip_strength)
        self.gaba_strength = float(gaba_strength)
        self.avg_inh = float(avg_inh)
        self.gaba_min = float(gaba_min)
        self.gaba_max = float(gaba_max)
        self.stdp_strength = float(stdp_strength)
        self.exc_exp = float(exc_exp)
        self.exc_mul = float(exc_mul)
        self.inh_duration = float(inh_duration)
        self.norm_every = int(norm_every)
        self.input_mode = input_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # W[src, dst] throughout -- upstream's orientation, so a matrix dumped
        # from either side can be compared without a transpose.
        self.W_es = self._create_weights((self.n_neurons, self.n_exc), normalize=True)
        self.W_ee = self._create_weights((self.n_exc, self.n_exc), normalize=False)
        self.W_ie = self._create_weights((self.n_exc, self.n_inh), normalize=True)
        self.W_ei = self._create_weights((self.n_inh, self.n_exc), normalize=True)

        self.iteration = 0
        self.reset_context()
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ setup
    def _create_weights(self, shape, normalize: bool) -> np.ndarray:
        """Upstream ``CreateWeights``: uniform, then Sinkhorn-style balancing.

        The ten alternating row/column divisions are upstream's, not a
        convergence criterion; reproducing the count matters because the result
        is not a fixed point.
        """
        W = self.rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        if normalize:
            for _ in range(10):
                W /= W.sum(axis=1)[:, None]
                W /= W.sum(axis=0)
        return W.astype(np.float64)

    def set_target_activity(self, sequence_length: int) -> None:
        """Align the homeostatic set-point with the material, as upstream does.

        Upstream's ``target_act = 1 / n_chars(grammar)`` is one over the number
        of symbol presentations in the training text. Call this with the
        sequence length before fitting; it is not done automatically because
        ``fit_event`` cannot see the sequence it belongs to.
        """
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")
        self.target_activity = 1.0 / float(sequence_length)

    def calibrate_target_activity(self, events, rounds: int = 4,
                                  passes: int = 200) -> float:
        """Re-site ``target_activity`` to the rate the material actually drives.

        The driven rate is a **fixed point**, not a free measurement. Intrinsic
        plasticity can only lower a unit's bias toward the target; it cannot
        shrink an assembly below what STDP binds, so when the target sits under
        the rate the material forces, the bias falls without bound and the
        network is silent as soon as input stops (``docs/vieth_stdp_port.md``,
        mechanism 2). Measuring the rate with the bias switched off is the wrong
        read -- that regime is sparse and reports 0.02-0.06 on one-hot lists,
        which then starves acquisition. So: train normally for ``passes``, set
        the target to the steady-state rate, and repeat for ``rounds``. On a
        6-item one-hot list this converges to ``1/L`` from above and from below;
        on the dense symbolic substrate to the active fraction (~0.25). Weights
        learned during calibration are kept -- it is ordinary early training.

        Returns the target it settled on.
        """
        events = list(events)
        if not events or rounds < 1 or passes < 1:
            raise ValueError("calibration needs events, >= 1 round and >= 1 pass.")
        n_last = len(events) * max(1, passes // 4)
        for _ in range(rounds):
            rates = []
            for k in range(passes):
                for e in events:
                    for frame in self._frames(e):
                        rates.append(self._iterate(frame, plastic=True).mean())
            self.target_activity = float(max(np.mean(rates[-n_last:]), 1.0 / self.n_exc))
        return self.target_activity

    @property
    def params(self) -> dict:
        return {
            "model": "ViethGabaSTDPNetwork",
            "n_neurons": self.n_neurons, "n_exc": self.n_exc, "n_inh": self.n_inh,
            "target_activity": self.target_activity,
            "presentation_steps": self.presentation_steps,
            "recall_steps": self.recall_steps, "cue_lag": self.cue_lag,
            "n_epochs": self.n_epochs,
            "gaba_min": self.gaba_min, "gaba_strength": self.gaba_strength,
            "stdp_strength": self.stdp_strength, "ip_strength": self.ip_strength,
            "norm_every": self.norm_every, "input_mode": self.input_mode,
            "seed": self.seed,
        }

    def named_parameters(self) -> dict:
        return {"W_es": self.W_es, "W_ee": self.W_ee,
                "W_ie": self.W_ie, "W_ei": self.W_ei}

    # ------------------------------------------------------------------ state
    def _blank_state(self) -> dict:
        return {
            "inp_spike": np.zeros(self.n_neurons, dtype=bool),
            "inp_spike_old": np.zeros(self.n_neurons, dtype=bool),
            "exc_spike": np.zeros(self.n_exc, dtype=bool),
            "exc_spike_old": np.zeros(self.n_exc, dtype=bool),
            "inh_spike": np.zeros(self.n_inh, dtype=bool),
            "inh_avg_act": np.zeros(self.n_inh),
            "li_stdp_mul": np.ones(self.n_exc),
        }

    def on_event_boundary(self, boundary_context: Optional[np.ndarray] = None) -> None:
        """Drain the pipeline, then clear the spike history **but not the
        inhibitory state**.

        This override is not bookkeeping; without it the arm scores at chance
        under MemVal's own ingestion path while scoring 1.00 under upstream's
        continuous one. Two separate causes, both structural, both measured on a
        6-item sequence (``bin/vieth_stdp_sanity.py``):

        **1. The pipeline has to drain (0.00 -> 0.40).** The network is
        ``cue_lag`` iterations deep, so when the last item of a sequence is
        presented none of its consequences have happened yet: its assembly has
        not fired, its input binding has not been learned, and the transition
        into it has not formed. Resetting at the seam wipes all of that before
        it exists. Running ``cue_lag`` silent but **plastic** iterations first
        reproduces what upstream gets for free by never resetting, and stops
        exactly there -- a third would start pairing free-running activity
        instead of the sequence.

        **2. Inhibition must survive the seam (0.40 -> 1.00).** The inhibitory
        population is a running average, and it is what enforces competition, so
        it needs a few iterations to ramp. Zeroing it at every seam means the
        first items of every pass are learned with **no competition at all**:
        measured assembly sizes went to ``[117, 0, 2, 46, 74, 61]`` on six
        items -- item 1 got no assembly whatsoever and item 0 took a third of
        the network -- against ``[48, 43, 53, 57, 52, 47]`` when it is kept.
        A boundary means "no transition forms across this seam", which requires
        clearing the **spike history**; the inhibitory average is homeostatic
        state, like ``sensitivity``, and clearing it is not part of that.

        ``reset_context`` remains the full transient reset and is what the
        pipelines call between trials.
        """
        for _ in range(self.cue_lag):
            self._iterate(np.zeros(self.n_neurons, dtype=bool), plastic=True)
        keep = self._state["inh_avg_act"].copy()
        self.reset_context()
        self._state["inh_avg_act"] = keep

    def reset_context(self):
        """Clear transient activity. Weights and ``sensitivity`` are untouched.

        ``sensitivity`` is deliberately preserved: intrinsic plasticity is a
        slow homeostatic *learned* bias, not transient state, and upstream never
        clears it. Resetting it at every sequence seam would restart homeostasis
        on each pass and the network would never settle at its set-point.
        """
        self._state = self._blank_state()
        if not hasattr(self, "sensitivity"):
            self.sensitivity = np.zeros(self.n_exc)

    def get_latent_state(self) -> dict:
        return {"W_ee": self.W_ee.copy(), "W_es": self.W_es.copy(),
                "sensitivity": self.sensitivity.copy(),
                "exc_spike": self._state["exc_spike"].astype(float)}

    def _save_state(self) -> dict:
        return {"state": {k: v.copy() for k, v in self._state.items()},
                "sensitivity": self.sensitivity.copy(),
                "iteration": self.iteration}

    def _restore_state(self, saved: dict) -> None:
        self._state = saved["state"]
        self.sensitivity = saved["sensitivity"]
        self.iteration = saved["iteration"]

    # ----------------------------------------------------------------- engine
    def _iterate(self, inp_spike: np.ndarray, plastic: bool,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """One network iteration, in upstream's behaviour-key order.

        The order is not cosmetic. PymoNNto sorts behaviours by key across *all*
        groups, giving 3, 3.1, 12, 20, 30, 40, 41, 50, 51, 60, 70 -- so
        plasticity (41) runs **before** the new spikes are computed (50, 51),
        and therefore pairs ``inp_spike(t-2) -> exc_spike(t-1)`` on ``ES`` and
        ``exc_spike(t-2) -> exc_spike(t-1)`` on ``EE``. Reordering it to use the
        current step's spikes silently changes which pair is learned.

        Args:
            inp_spike: Boolean input-population activity for this iteration.
            plastic: Whether STDP and normalisation run.
            rng: Source of the threshold draws; ``predict_next`` passes a
                freshly seeded one so probes stay pure.
        """
        s = self._state
        gen = self.rng if rng is None else rng
        self.iteration += 1

        # --- key 3 / 3.1: normalisation, every norm_every iterations ---------
        if plastic and (self.iteration - 1) % self.norm_every == 0:
            self._normalize()

        # --- key 12: excitatory drive (GLU), from the PREVIOUS iteration -----
        voltage = (self.W_es[s["inp_spike"]].sum(axis=0)
                   + self.W_ee[s["exc_spike"]].sum(axis=0))

        # --- key 20: inhibitory drive (GABA), delayed one iteration ----------
        input_gaba = -self.W_ei[s["inh_spike"]].sum(axis=0)
        voltage = voltage + input_gaba

        # --- key 30: intrinsic plasticity ------------------------------------
        self.sensitivity -= (s["exc_spike"].astype(float) - self.target_activity) * self.ip_strength
        voltage = voltage + self.sensitivity

        # --- key 40: GABA modulation of the plasticity multiplier ------------
        # input_gaba is <= 0, so more inhibition lowers the multiplier and, once
        # gaba_min < 0, flips potentiation into depression.
        s["li_stdp_mul"] = np.clip((1.0 + input_gaba / self.avg_inh) * self.gaba_strength,
                                   self.gaba_min, self.gaba_max)

        # --- key 41: STDP, on the pairs standing before this step's spikes ---
        if plastic:
            self._stdp(self.W_es, s["inp_spike_old"], s["exc_spike"], s["li_stdp_mul"])
            self._stdp(self.W_ee, s["exc_spike_old"], s["exc_spike"], s["li_stdp_mul"])

        # --- key 50: input group emits ---------------------------------------
        s["inp_spike_old"] = s["inp_spike"]
        s["inp_spike"] = np.asarray(inp_spike, dtype=bool)

        # --- key 51: excitatory group emits, then voltage is discarded -------
        p = np.power(np.clip(voltage * self.exc_mul, 0.0, None), self.exc_exp)
        s["exc_spike_old"] = s["exc_spike"]
        s["exc_spike"] = p > gen.uniform(0.0, 1.0, self.n_exc)

        # --- key 60/70: inhibitory group, driven by THIS step's exc spikes ---
        inh_voltage = self.W_ie[s["exc_spike"]].sum(axis=0)
        s["inh_avg_act"] = ((s["inh_avg_act"] * self.inh_duration + inh_voltage)
                            / (self.inh_duration + 1.0))
        s["inh_spike"] = ((s["inh_avg_act"] * self.avg_inh / self.target_activity)
                          > gen.uniform(0.0, 1.0, self.n_inh))
        return s["exc_spike"]

    def _stdp(self, W: np.ndarray, pre: np.ndarray, post: np.ndarray,
              li_stdp_mul: np.ndarray) -> None:
        """Upstream ``STDP``: one-step causal window, inhibition-gated.

        The increment depends only on the *post*synaptic unit, broadcast across
        every active presynaptic partner -- upstream's ``np.tile`` -- so this is
        not a pair-symmetric rule. Clipping at zero is upstream's, and it is
        one-sided: weights can be driven to zero but never negative, so the
        depression that ``gaba_min < 0`` provides is bounded by the weight.
        """
        if not pre.any() or not post.any():
            return
        idx = np.ix_(pre, post)
        dw = li_stdp_mul[post][None, :] * self.stdp_strength
        W[idx] = np.clip(W[idx] + dw, 0.0, None)

    def _normalize(self) -> None:
        """Upstream ``Normalization``: afferent+efferent on EE, afferent on ES.

        Applied in upstream's order -- EE afferent, then EE efferent, then ES
        afferent -- because the three are sequential divisions, not a joint
        solve, and the result depends on the order.
        """
        ws = self.W_ee.sum(axis=0)
        self.W_ee /= np.where(ws == 0.0, 1.0, ws)
        ws = self.W_ee.sum(axis=1)
        self.W_ee /= np.where(ws == 0.0, 1.0, ws)[:, None]
        ws = self.W_es.sum(axis=0)
        self.W_es /= np.where(ws == 0.0, 1.0, ws)

    # ------------------------------------------------------------------- i/o
    def _as_spikes(self, event: np.ndarray) -> np.ndarray:
        S = np.asarray(event)
        if S.ndim == 1:
            S = S[:, None]
        if S.shape[0] != self.n_neurons:
            raise ValueError(
                f"expected {self.n_neurons} input neurons, got {S.shape[0]}. "
                "This arm consumes spike trains, not vectors -- wrap it with "
                "memval.models.codec_wrapper.wrap_with_codec.")
        return S

    def _frames(self, event: np.ndarray) -> np.ndarray:
        """``(N, T)`` codec window -> ``(presentation_steps, N)`` input frames.

        ``any`` (default) ORs each chunk of the window: a neuron is active in a
        frame if it fired at all within it. That is the reading upstream's own
        input has -- a symbol's neurons are simply on -- and with
        ``presentation_steps=1`` it reduces the whole window to one active set.

        The cost is stated: at the codec's default ``r_max=200 Hz`` over 50
        steps, a unit-magnitude feature fires with probability ~1, so ``any``
        **saturates** and the active set becomes "every neuron with nonzero
        magnitude". Graded magnitude survives only for weak features. Use
        ``frames`` with ``presentation_steps == window_steps`` to keep the raw
        train, and read the module docstring on what that does to the pairing.
        """
        S = self._as_spikes(event).astype(bool)
        T = S.shape[1]
        k = self.presentation_steps
        if self.input_mode == "frames":
            # Raw train, one codec step per iteration. Truncated or zero-padded
            # to k frames, so presentation_steps stays the number of iterations
            # an item costs whichever mode is in force.
            if k <= T:
                return S[:, :k].T
            return np.vstack([S.T, np.zeros((k - T, self.n_neurons), dtype=bool)])
        edges = np.linspace(0, T, k + 1).astype(int)
        return np.stack([S[:, a:b].any(axis=1) for a, b in zip(edges[:-1], edges[1:])])

    # -------------------------------------------------------------- training
    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None,
                  **kwargs):
        """Present one item for ``presentation_steps`` iterations, plastic.

        Activity carries across calls, which is where the cross-item pairing
        that builds ``EE`` comes from. Call ``on_event_boundary`` at a seam.
        """
        for frame in self._frames(event):
            self._iterate(frame, plastic=True)

    def fit_sequence(self, sequence_data: np.ndarray,
                     context_data: Optional[np.ndarray] = None,
                     epochs: Optional[int] = None, **kwargs):
        """Present a sequence ``n_epochs`` times.

        Args:
            sequence_data: ``(L, n_neurons, T)`` -- ``encode_stack``'s form --
                or ``(n_neurons, T)`` for a single continuous train.
            epochs: Passes for this call, overriding ``n_epochs``.
        """
        X = np.asarray(sequence_data)
        items = [X] if X.ndim == 2 else list(X)
        for _ in range(self.n_epochs if epochs is None else int(epochs)):
            self.on_event_boundary()
            for item in items:
                self.fit_event(item)
            self.on_event_boundary()

    # --------------------------------------------------------------- readout
    def _reconstruct(self) -> np.ndarray:
        """Upstream ``TextReconstructor``, argmax removed: ``ES^T . exc_spike``.

        The argmax stays out of the arm on purpose (spec S2.4): reducing the
        per-neuron activation to a symbol is the decoder's decision, and the
        codec's ``counts()`` performs exactly upstream's sum over the duplicate
        axis before making it.
        """
        return self.W_es @ self._state["exc_spike"].astype(float)

    def predict_next(self, current_event: np.ndarray,
                     current_context: Optional[np.ndarray] = None,
                     **kwargs) -> np.ndarray:
        """Cue the network, free-run, and return the reconstruction window.

        Pure function of the cue: plasticity is off, the threshold draws come
        from a freshly seeded generator, and all transient state -- including
        ``sensitivity``, which intrinsic plasticity mutates on every iteration
        -- is saved and restored around the call.

        Returns:
            ``(n_neurons, recall_steps)`` of reconstruction activations. Real
            valued rather than boolean because the readout is a weighted sum;
            ``PopulationSpikeDecoder.decode`` sums over the window either way.
        """
        saved = self._save_state()
        try:
            self._state = self._blank_state()
            gen = np.random.default_rng(self._probe_seed())
            for frame in self._frames(current_event):
                self._iterate(frame, plastic=False, rng=gen)
            silence = np.zeros(self.n_neurons, dtype=bool)
            for _ in range(self.cue_lag):
                self._iterate(silence, plastic=False, rng=gen)
            out = np.zeros((self.n_neurons, self.recall_steps))
            for t in range(self.recall_steps):
                out[:, t] = self._reconstruct()
                self._iterate(silence, plastic=False, rng=gen)
            return out
        finally:
            self._restore_state(saved)

    def _probe_seed(self) -> int:
        return 0 if self.seed is None else int(self.seed) + 104729

    def recall(self, prompt_event: np.ndarray, length: int,
               prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Free-running rollout: present the prompt, then let the network run.

        This is upstream's generation protocol. The prompt is presented item by
        item, then the input is switched off and the recurrent excitatory
        population is left to advance on its own, with the reconstruction read
        out at every step. Nothing is fed back through the input pathway, which
        is what :attr:`rollout_mode` = ``LATENT`` declares.

        The alternative -- feeding the readout back as the next cue -- is not
        available to this arm and should not be added: the readout is a
        real-valued weighted sum over input neurons, so it would have to be
        thresholded back into a spike pattern, and any threshold is an invention
        that would be doing part of the memory's work.

        Pure, like ``predict_next``: state is saved and restored, and plasticity
        is off throughout.

        Returns:
            ``(length, n_neurons, recall_steps)``. Benchmarks should roll out
            through ``CodecWrappedModel.recall``, which works in R^D where every
            other arm is scored -- note that is a *different* protocol, observation
            feedback rather than free-running, and the two are not comparable.
        """
        prompt = np.asarray(prompt_event)
        events = list(prompt) if prompt.ndim > 2 else [prompt]
        saved = self._save_state()
        try:
            self._state = self._blank_state()
            gen = np.random.default_rng(self._probe_seed())
            for event in events:
                for frame in self._frames(event):
                    self._iterate(frame, plastic=False, rng=gen)
            silence = np.zeros(self.n_neurons, dtype=bool)
            for _ in range(self.cue_lag):
                self._iterate(silence, plastic=False, rng=gen)
            out = np.zeros((length, self.n_neurons, self.recall_steps))
            for t in range(length):
                for k in range(self.recall_steps):
                    out[t, :, k] = self._reconstruct()
                    self._iterate(silence, plastic=False, rng=gen)
            return out
        finally:
            self._restore_state(saved)


class CodecViethNetwork(OnlineCodecWrappedModel):
    """Registry-facing form: :class:`ViethGabaSTDPNetwork` behind the codec.

    ``MODEL_REGISTRY`` builds every arm as ``cls(n_features=D, **kwargs)``, but
    this arm consumes spikes and its input population is *derived*
    (``N = 2 * D * n_per_feature``), so it cannot be registered directly.

    ``n_exc`` defaults to a multiple of the input population rather than
    upstream's flat 2400: upstream sized its network against a 10-symbol
    alphabet, and the quantity that has to scale is units-per-item, not units.

    What to expect
    --------------
    The same overlap-bound behaviour the Bush arm shows, and for a sharper
    reason. STDP builds a chain out of item-specific assemblies; a dense
    substrate on which every item drives a quarter of the network has no
    item-specific assemblies to chain, and this arm's window is one iteration
    wide, so it has even less room than Bush's 20-50 ms traces. Run it on the
    ``HierarchicalEncoder`` substrate the spec names as the reference (S3.1)
    before reading anything into a floor.
    """

    #: **Deliberately not the inner arm's.** ``CodecWrappedModel.recall`` runs
    #: the rollout itself in R^D, decoding and re-encoding at each step, and
    #: does not call the inner arm's ``recall`` at all. So behind the codec this
    #: arm's rollout is observation feedback on the last prompt event, not the
    #: free-running latent protocol the inner arm declares. Two different
    #: protocols, declared separately, because ``rollout_modes_comparable``
    #: depends on the declaration being about what actually ran.
    rollout_mode = RolloutMode.OBSERVATION
    prompt_conditioned = False
    online_equivalent = True

    def __init__(self, n_features: int,
                 n_per_feature: int = 10,
                 window_steps: int = 50,
                 dt_ms: float = 1.0,
                 r_max: float = 200.0,
                 codec_mode: str = "rate",
                 exc_per_input: float = 0.5,
                 n_exc: Optional[int] = None,
                 probe_seed: int = 0,
                 seed: Optional[int] = 42,
                 **kwargs):
        encoder = PopulationSpikeEncoder(
            n_features, n_per_feature=n_per_feature, window_steps=window_steps,
            dt_ms=dt_ms, r_max=r_max, mode=codec_mode, seed=seed)
        if n_exc is None:
            n_exc = max(20, int(round(exc_per_input * encoder.n_neurons)))
        inner = ViethGabaSTDPNetwork(n_neurons=encoder.n_neurons, n_exc=n_exc,
                                     seed=seed, **kwargs)
        super().__init__(inner, encoder, inner_domain="spikes",
                         probe_seed=probe_seed)
        # CodecWrappedModel copies these off the inner arm; restore the values
        # that describe what THIS class actually does (see the note above).
        self.rollout_mode = RolloutMode.OBSERVATION
        self.prompt_conditioned = False
