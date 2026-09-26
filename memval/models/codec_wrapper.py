"""Run any arm behind the population spike codec.

Two uses, one class (``docs/spike_codec_spec.md`` S4-S5):

**Transport (``inner_domain="spikes"``).** The inner model consumes and emits
spike arrays. The wrapper encodes on the way in and decodes on the way out, so
a spiking arm is scored by the existing vector-space metrics with nothing else
changed.

**The transport control (``inner_domain="vectors"``).** The inner model is an
ordinary rate arm, unchanged, but every vector it sees has round-tripped
``vector -> spikes -> vector`` on input *and* output. The difference between
this and the native arm is the codec's own contribution, **measured**. Without
it, any deficit a spiking arm shows is unattributable: a model that cannot
exploit graded input and a codec that quantised the gradation away look
identical from the outside.

Purity
------
``predict_next`` must remain a pure function of the cue (the contract
``measure_recall_associative`` depends on, and the one
``tests/test_temporal_pc.py`` pins). The codec is stochastic, so the wrapper
encodes probe cues from a **freshly seeded generator** (``probe_seed``) rather
than from the encoder's running stream: the same cue therefore always produces
the same spike train, and no state leaks between probes. That is common random
numbers across probes, not a hidden dependence on the cue's value.

Training encodes from the encoder's running stream, so repeated presentations of
the same item are independent Poisson draws -- which is what a spiking arm needs
in order to see anything other than one frozen sample.

Online arms always train through ``fit_event``
----------------------------------------------
``OnlineCodecWrappedModel.fit_sequence`` overrides the delegating path and
trains event by event, so an arm that declares ``OnlineTrainable`` sees the same
code path whichever ingestion regime the *caller* asked for and the batch path
cannot drift from the streamed one unnoticed. It is byte-identical to
delegating, for an inner arm whose own ``fit_sequence`` is already a
``fit_event`` loop.

Batch and streamed must draw the same noise
-------------------------------------------
``resample_per_pass`` (default True) makes ``fit_sequence`` own the pass loop and
re-encode the material on every pass, instead of encoding once and handing the
inner arm a frozen sample to replay. Without it the two ingestion regimes are not
the same protocol: streaming re-encodes each item on each pass by construction,
so ``fit_sequence`` would train on one Poisson realisation repeated N times while
the streamed path trained on N independent ones. Measured on a 5-item list behind
the codec, that came to ``max|dW| = 1.3e-1`` on a ``w_max = 1`` matrix -- not a
rounding difference, and it would have been reported as an ingestion-regime
effect. With resampling on, the two paths are **byte-identical**, which is what
``online_equivalent`` asserts.

Set it False only for an inner arm whose ``fit_sequence`` does not accumulate --
``HopfieldSequenceNetwork`` under ``fit_method="projection"`` replaces ``W`` with
a least-squares solution, so splitting one N-pass call into N one-pass calls
changes the rule rather than the noise.
"""

from typing import Any, Optional

import numpy as np

from .base import HippocampalModel
from .capabilities import OnlineTrainable, RolloutMode, supports_online
from ..encoders.spike_codec import PopulationSpikeDecoder, PopulationSpikeEncoder


class CodecWrappedModel(HippocampalModel):
    """Wrap ``inner`` so the outside world only ever sees R^D vectors.

    Args:
        inner: The wrapped arm. Its ``fit_*``/``predict_next`` consume spike
            arrays when ``inner_domain="spikes"``, vectors otherwise.
        encoder: A ``PopulationSpikeEncoder``.
        decoder: Its matched ``PopulationSpikeDecoder``. Built from the encoder
            when omitted.
        inner_domain: ``"spikes"`` for a spiking arm, ``"vectors"`` for the
            transport control.
        probe_seed: Seed for the per-probe generator that keeps ``predict_next``
            pure.
        resample_per_pass: Re-encode the material on every training pass, so
            batch ingestion sees the same independent Poisson draws streaming
            does. See the module docstring; leave True unless the inner arm's
            ``fit_sequence`` is non-accumulating. **Applies to the batch path
            only**: an arm declaring ``OnlineTrainable`` trains through
            ``fit_event`` whatever the caller asked for, which resamples per
            pass by construction.
    """

    def __init__(self, inner: HippocampalModel,
                 encoder: PopulationSpikeEncoder,
                 decoder: Optional[PopulationSpikeDecoder] = None,
                 inner_domain: str = "spikes",
                 probe_seed: int = 0, resample_per_pass: bool = True, **kwargs):
        super().__init__(**kwargs)
        if inner_domain not in ("spikes", "vectors"):
            raise ValueError("inner_domain must be 'spikes' or 'vectors'.")
        self.inner = inner
        self.encoder = encoder
        self.decoder = decoder if decoder is not None else PopulationSpikeDecoder(encoder)
        self.inner_domain = inner_domain
        self.probe_seed = int(probe_seed)
        self.resample_per_pass = bool(resample_per_pass)
        self.n_features = encoder.embedding_dim

        # Declared per-arm facts belong to the wrapped arm, not to the wrapper.
        self.rollout_mode = getattr(inner, "rollout_mode", RolloutMode.OBSERVATION)
        self.prompt_conditioned = bool(getattr(inner, "prompt_conditioned", False))

    # ------------------------------------------------------------------ meta
    @property
    def n_epochs(self) -> int:
        """Exposure, proxied to the inner arm.

        The suite's ``_fit`` helpers set ``model.n_epochs = epochs`` *and* pass
        ``epochs=``; without this property the assignment would land on the
        wrapper and the inner arm would keep training at its constructor value
        -- the silent-discard defect ``tests/test_epochs_kwarg_contract.py``
        exists to catch, one level up.
        """
        return int(getattr(self.inner, "n_epochs", 1))

    @n_epochs.setter
    def n_epochs(self, value: int) -> None:
        if hasattr(type(self.inner), "n_epochs") or hasattr(self.inner, "n_epochs"):
            self.inner.n_epochs = int(value)

    @property
    def params(self) -> dict:
        """Codec condition + inner arm, for figure captions (spec S2.5)."""
        p = dict(self.encoder.params)
        p["inner"] = type(self.inner).__name__
        p["inner_domain"] = self.inner_domain
        p["probe_seed"] = self.probe_seed
        p["resample_per_pass"] = self.resample_per_pass
        return p

    def describe(self) -> str:
        return (f"{type(self.inner).__name__} via {self.encoder.describe()} "
                f"[inner_domain={self.inner_domain}]")

    # ------------------------------------------------------------- transport
    def _probe_rng(self) -> np.random.Generator:
        """Fresh generator, identical on every call -- see the purity note."""
        return np.random.default_rng(self.probe_seed)

    def _to_inner(self, vector: np.ndarray,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Vector -> whatever the inner arm eats."""
        spikes = self.encoder.encode(vector, rng=rng)
        if self.inner_domain == "spikes":
            return spikes
        return self.decoder.decode(spikes)

    def _from_inner(self, out: np.ndarray,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Whatever the inner arm emits -> R^D, unit-norm."""
        if self.inner_domain == "spikes":
            return self.decoder.decode(out)
        # Transport control: the arm's own output round-trips too. It is
        # L2-normalised first because a rate arm's output scale is
        # unconstrained (AsymmetricHopfieldNetwork's ReLU output is not on the
        # unit-norm manifold), and an unnormalised vector would clip the Poisson
        # probability at 1 and destroy direction -- a codec artefact rather than
        # the transport cost we are trying to measure. The decode step
        # L2-normalises regardless, so this only fixes the scale going in.
        out = np.asarray(out, dtype=float).reshape(-1)
        norm = np.linalg.norm(out)
        if norm == 0.0:
            return np.zeros(self.n_features)
        return self.decoder.decode(self.encoder.encode(out / norm, rng=rng))

    # -------------------------------------------------------------- training
    def _encode_all(self, X: np.ndarray) -> np.ndarray:
        """One fresh draw of the whole sequence, in the inner arm's domain."""
        if self.inner_domain == "spikes":
            return self.encoder.encode_stack(X)               # (L, N, T)
        return np.stack([self._to_inner(v) for v in X])

    def fit_sequence(self, sequence_data: np.ndarray,
                     context_data: Optional[np.ndarray] = None,
                     epochs: Optional[int] = None, **kwargs):
        X = np.atleast_2d(np.asarray(sequence_data, dtype=float))
        if not self.resample_per_pass:
            return self.inner.fit_sequence(self._encode_all(X),
                                           context_data=context_data,
                                           epochs=epochs, **kwargs)
        n_passes = int(epochs) if epochs is not None else self.n_epochs
        prev = getattr(self.inner, "n_epochs", None)
        try:
            # Pin the instance as well as passing the kwarg, the way the suite's
            # own _fit helpers do: an inner arm that reads self.n_epochs and
            # ignores epochs= would otherwise multiply every pass by its own
            # default.
            if prev is not None:
                self.inner.n_epochs = 1
            for _ in range(n_passes):
                self.inner.fit_sequence(self._encode_all(X),
                                        context_data=context_data,
                                        epochs=1, **kwargs)
        finally:
            if prev is not None:
                self.inner.n_epochs = prev

    def predict_next(self, current_event: np.ndarray,
                     current_context: Optional[np.ndarray] = None,
                     **kwargs) -> np.ndarray:
        rng = self._probe_rng()
        cue = self._to_inner(current_event, rng=rng)
        out = self.inner.predict_next(cue, current_context=current_context, **kwargs)
        return self._from_inner(out, rng=rng)

    def recall(self, prompt_event: np.ndarray, length: int,
               prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Autoregressive rollout **in vector space**.

        Each step decodes to R^D and re-encodes, so the rollout protocol is the
        one the symbolic suite already sweeps rather than a spike-domain variant
        nothing else uses. The inner arm's own ``recall`` is deliberately not
        called: for a spiking arm it would have to invent a decode, and spec
        S2.4 says the decode choice must not be baked into the arm.
        """
        prompt = np.asarray(prompt_event, dtype=float)
        current = prompt[-1] if prompt.ndim > 1 else prompt
        out = np.zeros((length, self.n_features))
        for t in range(length):
            c_t = prompt_context[t] if prompt_context is not None else None
            current = self.predict_next(current, current_context=c_t)
            out[t] = current
        return out

    # ----------------------------------------------------------------- state
    def get_latent_state(self) -> dict:
        """The inner arm's latent state, numeric arrays only.

        The codec condition is deliberately **not** mixed in here: this dict is
        for RSA and for ``test_epochs_kwarg_contract``, which flattens every
        value into one vector, so a parameter dict among the arrays would break
        it. Read the codec condition off :attr:`params`.
        """
        return dict(self.inner.get_latent_state())

    def reset_context(self):
        self.inner.reset_context()

    def decode_prediction(self, raw_prediction: np.ndarray) -> np.ndarray:
        return raw_prediction


class OnlineCodecWrappedModel(CodecWrappedModel, OnlineTrainable):
    """``CodecWrappedModel`` for an inner arm that declares ``OnlineTrainable``.

    Two classes rather than one because the capability is **nominal** (see
    ``memval.models.capabilities``): ``isinstance(arm, OnlineTrainable)`` has to
    report what the author asserted, so a wrapper cannot claim the capability on
    behalf of an inner arm that lacks it. Use :func:`wrap_with_codec`, which
    picks the right class.
    """

    def __init__(self, inner: HippocampalModel, *args, **kwargs):
        if not supports_online(inner):
            raise TypeError(
                f"{type(inner).__name__} does not declare OnlineTrainable; "
                "use CodecWrappedModel or wrap_with_codec().")
        super().__init__(inner, *args, **kwargs)
        # Rule identity is unchanged by transport: the codec sits outside the
        # learning rule, so whatever the inner arm asserts still holds.
        self.online_equivalent = bool(getattr(inner, "online_equivalent", False))

    def fit_sequence(self, sequence_data: np.ndarray,
                     context_data: Optional[np.ndarray] = None,
                     epochs: Optional[int] = None, **kwargs):
        """Train an online arm through ``fit_event``, always.

        The base class delegates to the inner arm's ``fit_sequence``; this
        overrides that, so an arm that declares ``OnlineTrainable`` is trained
        one event at a time whichever ingestion regime the *caller* asked for.
        A batch suite and a streamed suite then exercise literally the same code
        path on this arm, and the batch path cannot drift away from the streamed
        one unnoticed.

        It is byte-identical to the delegating path for an arm whose
        ``fit_sequence`` is already a ``fit_event`` loop -- the codec is drawn
        from the same running stream in the same order either way -- and
        ``test_online_arms_train_through_fit_event`` pins that.

        Passes are fenced with ``on_event_boundary`` at both seams, so no
        transition forms across the sequence edge or between passes.
        """
        X = np.atleast_2d(np.asarray(sequence_data, dtype=float))
        n_passes = int(epochs) if epochs is not None else self.n_epochs
        for _ in range(n_passes):
            self.on_event_boundary()
            for t, event in enumerate(X):
                self.fit_event(
                    event,
                    context=None if context_data is None else context_data[t],
                    **kwargs)
            self.on_event_boundary()

    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None,
                  **kwargs):
        return self.inner.fit_event(self._to_inner(event), context=context, **kwargs)

    def on_event_boundary(self, boundary_context: Optional[np.ndarray] = None) -> None:
        self.inner.on_event_boundary(boundary_context)


def wrap_with_codec(inner: HippocampalModel, encoder: PopulationSpikeEncoder,
                    decoder: Optional[PopulationSpikeDecoder] = None,
                    **kwargs) -> CodecWrappedModel:
    """Wrap ``inner``, preserving whether it declares ``OnlineTrainable``."""
    cls = OnlineCodecWrappedModel if supports_online(inner) else CodecWrappedModel
    return cls(inner, encoder, decoder, **kwargs)
