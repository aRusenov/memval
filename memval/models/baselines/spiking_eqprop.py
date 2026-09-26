"""Spiking Equilibrium Propagation (O'Connor, Gavves & Welling, AISTATS 2019).

A MemVal arm around the authors' own reference implementation, vendored at
``memval.vendor.spiking_eqprop`` (upstream: ``QUVA-Lab/spiking-eqprop``).

What makes it "spiking"
-----------------------
The learning rule is unchanged Equilibrium Propagation: settle to a free
equilibrium, settle again with the output nudged toward the target, and update
each weight by the difference of the two correlation terms. What the paper
changes is the *communication channel between neurons*. In ordinary EP a
neuron sends its real-valued rate ``rho(s)`` to its neighbours. Here each
neuron instead emits a **binary spike train**: a sigma-delta quantizer turns
``rho(s)`` into 0/1 events whose running average tracks the rate, and the
receiving neuron reconstructs an estimate with a leaky (predictive) decoder.
Inter-layer traffic is therefore one bit per neuron per tick, and the
equilibrium the network settles into is only an approximation of the
real-valued one -- the paper's result is that EP still trains through it.

Setting ``quantizer=None`` swaps ``EncodingDecodingNeuronLayer`` for upstream's
real-valued ``SimpleLayerController``, giving the exact same EP arm without
spikes. That is the control this arm should be read against: any MemVal
difference between the two is attributable to the spiking channel alone.

Adaptation for sequences
------------------------
Identical in spirit to ``OriginalEqPropSequenceNetwork``: the network is a
one-step transition model. For every consecutive pair ``(x_t, x_{t+1})`` the
input layer is clamped to ``x_t`` and the output layer is nudged toward
``x_{t+1}``. ``predict_next`` is a free-phase settle from a clamped event, so
the arm holds no state across events (as with the other EP arms, the only
cross-call state online training needs is the previous event).

Unit-interval domain, and why the gain is not 1
----------------------------------------------
Upstream neurons live in ``[0, 1]`` (``rho`` clips there, and a spike rate
cannot be negative), while MemVal encoders emit signed vectors. The arm
therefore maps events through ``u = clip(offset + gain*x, 0, 1)`` on the way
in and inverts it on the way out.

**The gain must scale with the embedding dimension, and getting this wrong
silently floors the arm.** Upstream's inputs are MNIST pixels, which already
fill ``[0, 1]`` (and are close to binary), so upstream needs no gain at all.
MemVal's ``SymbolicEncoder`` emits **unit-norm** vectors, whose per-component
magnitude therefore shrinks as ``1/sqrt(n_features)``. At ``gain=1`` a 100-dim
embedding lands in ``u in [0.15, 0.79]`` -- every neuron sees a near-constant
0.5, nothing clips, and no amount of training helps:

    n_features=100, hidden=100, 7-item list, criterion MRR 0.95
      gain    clipped   MRR @ 64 updates
         1       0.0%   0.167   (chance; also 0.167 at 1..512 updates)
         4      21.6%   0.500
         6      40.9%   1.000
        10      63.1%   1.000     <- sqrt(100), the default
        30      85.9%   0.992

So ``input_gain`` defaults to ``sqrt(n_features)``, which makes the mapped
signal *dimension-invariant*: a unit-norm vector has per-component sigma of
about ``1/sqrt(n_features)``, so ``gain * sigma ~ 1`` at every embedding size,
and the arm behaves the same at dim 32 as at dim 100 rather than silently
degrading as the encoder widens. It sits mid-plateau, not on an edge -- the
criterion is reached anywhere from gain 6 to 30 at dim 100.

The resulting input saturates (roughly 60% of components clip at 0 or 1), and
that is faithful rather than pathological: it is the same regime upstream
trains in, since MNIST pixel intensities are themselves mostly at the rails.

Pass an explicit ``input_gain`` to override -- notably ``input_offset=0.0,
input_gain=1.0`` for data already in ``[0, 1]``, which makes the mapping the
identity.
"""

from typing import Optional, Sequence

import numpy as np
import torch

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode
from ...vendor.spiking_eqprop import (
    EncodingDecodingNeuronLayer,
    SimpleLayerController,
    initialize_params,
    initialize_states,
    run_eqprop_training_update,
    run_inference,
)


class SpikingEqPropSequenceNetwork(HippocampalModel, OnlineTrainable):
    """Sequence-prediction arm driven by the vendored spiking-EP reference code.

    Parameters
    ----------
    n_features : int
        Dimensionality of an event vector; also the size of the input and
        output layers.
    hidden_sizes : sequence of int
        Hidden layer widths. The paper's MNIST results use one 500-unit layer;
        MemVal sequences are much smaller, hence the smaller default.
    quantizer : {'sigma_delta', 'stochastic', 'threshold'} or None
        Spike code used between neurons. ``None`` disables spiking entirely and
        runs upstream's real-valued layer (the control). Upstream also offers
        ``'second_order_sd'``, but that class is dead code there and is broken
        as written (its ``__call__`` reads ``self.phi_1``/``self.phi_2`` while
        the fields are named ``phi1``/``phi2``), so this arm rejects it rather
        than letting it fail mid-settle.
    epsilons, lambdas : float or str
        Upstream step-size schedules, passed to ``create_step_sizer``: a number
        is a constant, a string is an expression in ``t`` (e.g.
        ``'0.354/t**0.5'``) evaluated once per settling tick. ``epsilons`` is
        the Euler step of the neuron's potential; ``lambdas`` is the time
        constant shared by the predictive encoder and decoder. ``lambdas=None``
        drops the predictive coder and quantizes ``rho(s)`` directly. Defaults
        are the paper's parameter-search values for the 1-hidden-layer net.
    beta : float
        Nudge strength in the positive phase.
    random_flip_beta : bool
        Randomly flip the sign of beta per update. This halves the first-order
        bias of the finite-beta gradient estimator, and upstream defaults it on
        -- but upstream trains MNIST with ~3000 minibatch updates per epoch, so
        the sign noise averages out. MemVal sequences give one full-batch update
        per epoch over a handful of transitions, and it does not average out:
        on the demo chain, mean next-item accuracy over 5 seeds is 0.24 with the
        flip on versus 0.88 with it off (the real-valued control degrades the
        same way, 0.60 vs 0.72, so this is a regime mismatch and not something
        the spike channel causes). Hence the default here is False, unlike
        upstream. Set True to reproduce upstream's configuration exactly.
    n_negative_steps, n_positive_steps : int
        Settling ticks in the free and nudged phases. Spiking needs far more
        than real-valued EP because the rate estimate has to average out;
        upstream's "longer" setting, and the default here, is 100/50. When
        running the real-valued control, keep this budget rather than dropping
        to upstream's vanilla 20/4 -- otherwise a comparison confounds the
        spike channel with how close each arm settled to equilibrium.
    learning_rate : float
        Step size of the EP weight update.
    n_epochs : int
        Passes over a sequence in ``fit_sequence``.
    bidirectional : bool or 'full'
        Upstream's choice of contrastive term (see ``eqprop_update``).
    splitstream : bool
        Re-run the free phase from the nudged state and use *that* as the
        negative term, rather than the free state the nudged phase grew out of.
        Upstream reports it "helps a LOT"; on the small MemVal chain it made no
        measurable difference, so it is off by default.
    l2_loss : float, optional
        Multiplicative shrinkage applied to the gradients.
    input_offset : float
        Additive centre of the affine map into the ``[0, 1]`` neuron domain.
    input_gain : float, optional
        Multiplicative scale of that map. **Defaults to ``sqrt(n_features)``**,
        not to 1: MemVal encoders emit unit-norm vectors whose per-component
        magnitude falls as ``1/sqrt(n_features)``, so a fixed gain makes the
        arm's effective input shrink as the encoder widens, and at gain 1 a
        100-dim embedding is squeezed into ``[0.15, 0.79]`` and the arm never
        learns at any budget. See the module docstring for the measured sweep.
        Pass a number to override (e.g. ``1.0`` for data already in [0, 1]).
    seed : int, optional
        Seeds torch, which drives weight init and the stochastic quantizers.
    """

    #: `recall` feeds the raw prediction back via `self.current_state`.
    rollout_mode = RolloutMode.OBSERVATION

    #: ``fit_sequence`` batches all transitions into one gradient-averaged EP
    #: update per epoch; ``fit_event`` applies one update per transition.
    online_equivalent = False

    def __init__(
        self,
        n_features: int,
        hidden_sizes: Sequence[int] = (100,),
        quantizer: Optional[str] = "sigma_delta",
        epsilons="0.354/t**0.0",
        lambdas="0.503/t**0.292",
        beta: float = 0.5,
        random_flip_beta: bool = False,
        n_negative_steps: int = 100,
        n_positive_steps: int = 50,
        learning_rate: float = 0.05,
        n_epochs: int = 100,
        bidirectional=True,
        splitstream: bool = False,
        l2_loss: Optional[float] = None,
        initial_weight_scale: float = 1.0,
        input_offset: float = 0.5,
        input_gain: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.hidden_sizes = tuple(hidden_sizes)
        self.quantizer = quantizer
        self.epsilons = epsilons
        self.lambdas = lambdas
        self.beta = beta
        self.random_flip_beta = random_flip_beta
        self.n_negative_steps = n_negative_steps
        self.n_positive_steps = n_positive_steps
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.bidirectional = bidirectional
        self.splitstream = splitstream
        self.l2_loss = l2_loss
        self.input_offset = input_offset
        # sqrt(n_features), not 1.0 -- see the module docstring. Unit-norm
        # embeddings have per-component sigma ~ 1/sqrt(n_features), so this
        # keeps gain*sigma ~ 1 at every embedding width. At gain=1 the arm sees
        # a near-constant 0.5 and floors at chance for any number of updates,
        # which reads as "spiking EP cannot do this task" when it is really
        # "the encoder was never mapped into the neurons' range".
        self.input_gain = (float(np.sqrt(n_features)) if input_gain is None
                           else float(input_gain))
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)

        self.layer_sizes = [n_features, *self.hidden_sizes, n_features]
        self.params = list(
            initialize_params(self.layer_sizes, initial_weight_scale=initial_weight_scale)
        )
        self.layer_constructor = self._build_layer_constructor()

        self.current_state = np.zeros(n_features)
        self._prev_event: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_layer_constructor(self):
        """Pick the spiking layer or upstream's real-valued control layer."""
        if self.quantizer == "second_order_sd":
            raise ValueError(
                "quantizer='second_order_sd' is broken in the upstream reference "
                "code (SecondOrderSigmaDeltaQuantizer.__call__ reads self.phi_1/"
                "self.phi_2, but the dataclass fields are phi1/phi2). It is dead "
                "code upstream -- none of the paper's experiments select it. Use "
                "'sigma_delta', 'stochastic' or 'threshold'."
            )
        if self.quantizer is None:
            epsilon = self.epsilons
            if not isinstance(epsilon, (int, float)):
                raise ValueError(
                    "The real-valued control layer (quantizer=None) takes a constant "
                    f"epsilon, not the schedule {epsilon!r}."
                )
            return SimpleLayerController.get_partial_constructor(epsilon=epsilon)
        return EncodingDecodingNeuronLayer.get_simple_constructor(
            epsilons=self.epsilons, quantizer=self.quantizer, lambdas=self.lambdas
        )

    def _fresh_states(self, n_samples: int):
        """New layer states on the current weights.

        Deliberately rebuilt for every call: a layer state carries the
        quantizer's accumulated sigma-delta charge and the step-sizer's tick
        counter, and upstream restarts both per minibatch (``renew_activations``).
        Reusing them would leak one event's spike phase into the next.
        """
        return initialize_states(
            layer_constructor=self.layer_constructor, n_samples=n_samples, params=self.params
        )

    # ------------------------------------------------------------------
    # Domain mapping
    # ------------------------------------------------------------------
    def _to_unit(self, x: np.ndarray) -> torch.Tensor:
        u = np.clip(self.input_offset + self.input_gain * np.asarray(x, dtype=float), 0.0, 1.0)
        return torch.tensor(np.atleast_2d(u), dtype=torch.float32)

    def _from_unit(self, u: torch.Tensor) -> np.ndarray:
        return (u.detach().numpy().astype(float) - self.input_offset) / self.input_gain

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _apply_update(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        """One EP update (free phase, nudged phase, contrastive step) on a batch
        of transitions, writing the new weights back into ``self.params``."""
        new_states = run_eqprop_training_update(
            x_data=x_batch,
            y_data=y_batch,
            layer_states=self._fresh_states(n_samples=x_batch.shape[0]),
            beta=self.beta,
            random_flip_beta=self.random_flip_beta,
            learning_rate=self.learning_rate,
            n_negative_steps=self.n_negative_steps,
            n_positive_steps=self.n_positive_steps,
            layer_constructor=self.layer_constructor,
            bidirectional=self.bidirectional,
            splitstream=self.splitstream,
            l2_loss=self.l2_loss,
            renew_activations=True,
        )
        self.params = [s.params for s in new_states]

    def fit_sequence(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, **kwargs):
        """Learn every ``(x_t -> x_{t+1})`` transition in one sequence.

        All transitions of the sequence form a single batch, and one EP update
        is applied per epoch — upstream's ``eqprop_update`` averages the
        gradient over the batch, so this matches the per-epoch averaged update
        the other EP arms in MemVal use. ``context_data`` is accepted for
        interface parity and unused.
        """
        seq = np.asarray(sequence_data, dtype=float)
        if seq.shape[0] < 2:
            return
        x_batch = self._to_unit(seq[:-1])
        y_batch = self._to_unit(seq[1:])
        # NOTE THE KEY. This read ``kwargs["n_epochs"]`` while every caller in
        # the suite passes ``epochs=`` -- so the lookup always missed and the
        # loop always ran ``self.n_epochs``. On continual_chain, which steps
        # one epoch at a time, that made every step 100 epochs and the section
        # ran 40 minutes instead of ~3.
        for _epoch in range(int(kwargs.get('epochs', self.n_epochs))):
            self._apply_update(x_batch, y_batch)

    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None, **kwargs):
        """Consume one streamed event, applying a single EP update for the
        transition out of the previously buffered event.

        The first event of a stream only primes the buffer. Call
        ``on_event_boundary`` (or ``reset_context``) between sequences so no
        transition is formed across the seam.
        """
        x = np.asarray(event, dtype=float).ravel()
        if self._prev_event is not None:
            self._apply_update(self._to_unit(self._prev_event), self._to_unit(x))
        self._prev_event = x

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(self, current_event: np.ndarray, current_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Clamp the event, settle freely, read the output layer's potential."""
        x = np.asarray(current_event, dtype=float).ravel()
        out = run_inference(
            x_data=self._to_unit(x),
            states=self._fresh_states(n_samples=1),
            n_steps=self.n_negative_steps,
        )
        self.current_state = self._from_unit(out)[0]
        return self.current_state.copy()

    #: Declares the batched probe path in `benchmarks/symbolic_pipeline.py`.
    #: See `predict_next_batch`.
    batched_probe = True

    def predict_next_batch(self, cues: np.ndarray,
                           current_context: Optional[np.ndarray] = None,
                           **kwargs) -> np.ndarray:
        """Settle many independent cues in ONE pass. Shape (N, F) -> (N, F).

        Exactly `predict_next` applied row-wise -- the rows never interact,
        because a free-phase settle is per-sample and `_fresh_states` builds one
        independent layer state per sample. What changes is only that the 100
        settling ticks are paid once for the whole batch instead of once per
        cue.

        That is the difference between this arm being runnable and not. Every
        tick is a handful of torch ops whose cost is dominated by per-call
        overhead, not by the (N, F) tensor width, so a 300-cue probe set costs
        about the same as a 1-cue one:

            300 separate predict_next calls   4.12 s
            one batched settle, N = 300       0.04 s      ~103x

        The probe path is where this arm spends almost all of its time -- a
        20-trial cued-recall measurement on an 11-item list took 4.2 s against
        0.19 s for real-valued EP -- so the suites were projected at multiple
        hours and are minutes with this.

        `current_context` is accepted for interface parity and unused, as in
        `predict_next`.
        """
        cues = np.atleast_2d(np.asarray(cues, dtype=float))
        out = run_inference(
            x_data=self._to_unit(cues),
            states=self._fresh_states(n_samples=cues.shape[0]),
            n_steps=self.n_negative_steps,
        )
        preds = self._from_unit(out)
        # Match `predict_next`'s side effect on the last row, so a caller that
        # mixes the two paths sees the same trailing state.
        self.current_state = preds[-1].copy()
        return preds

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Auto-regressive recall: seed with the prompt's last event and feed
        each prediction back in."""
        prompt = np.asarray(prompt_event, dtype=float)
        current = prompt[-1].copy() if prompt.ndim > 1 else prompt.copy()

        self.current_state = current
        recalled = np.zeros((length, self.n_features))
        for t in range(length):
            recalled[t] = self.predict_next(self.current_state)
            self.current_state = recalled[t]
        return recalled

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_latent_state(self) -> dict:
        """Feed-forward weights and biases, layer by layer, as numpy arrays."""
        state = {}
        for i, p in enumerate(self.params[1:], start=1):
            state[f"W_{i-1}{i}"] = p.w_aft.detach().numpy().copy()
            state[f"b_{i}"] = p.b.detach().numpy().copy()
        return state

    def named_parameters(self) -> dict:
        """Alias of ``get_latent_state`` for the diagnostics tier."""
        return self.get_latent_state()

    def reset_context(self):
        """Clear transient state (last prediction, online previous-event buffer)
        without touching the learned weights."""
        self.current_state = np.zeros(self.n_features)
        self._prev_event = None
