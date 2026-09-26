"""``CodecWrappedModel``: the transport seam and the transport control.

Two uses of one class (``docs/spike_codec_spec.md`` S4-S5). The control -- a
rate arm whose vectors round-trip through the codec -- is the point of the
exercise: without it, any deficit a spiking arm shows is unattributable.
"""
import numpy as np
import pytest

from memval.encoders.hierarchical import HierarchicalEncoder
from memval.encoders.spike_codec import PopulationSpikeEncoder, make_codec
from memval.models.baselines import AsymmetricHopfieldNetwork, TemporalPCNetwork
from memval.models.baselines.bush_stdp import BushSTDPSpikingNetwork
from memval.models.capabilities import supports_online
from memval.models.codec_wrapper import (
    CodecWrappedModel, OnlineCodecWrappedModel, wrap_with_codec,
)


def _seq():
    return HierarchicalEncoder(branching=(2, 2, 2), features_per_node=4).embeddings


def _control(seq, seed=0, **codec_kw):
    """AHN behind the codec: the S5 transport control."""
    enc, _ = make_codec(seq.shape[1], seed=seed, **codec_kw)
    inner = AsymmetricHopfieldNetwork(n_features=seq.shape[1], n_epochs=30)
    return wrap_with_codec(inner, enc, inner_domain="vectors")


# ------------------------------------------------------------------ shapes
def test_wrapper_keeps_the_vector_interface():
    seq = _seq()
    m = _control(seq)
    m.fit_sequence(seq)
    out = m.predict_next(seq[0])
    assert out.shape == (seq.shape[1],)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert m.recall(seq[0], length=3).shape == (3, seq.shape[1])
    assert m.recall(seq[:2], length=2).shape == (2, seq.shape[1])


def test_predict_next_is_pure_despite_a_stochastic_codec():
    """The contract ``measure_recall_associative`` depends on. The codec is
    Poisson, so purity is bought by encoding probes from a freshly seeded
    generator rather than from the encoder's running stream."""
    seq = _seq()
    m = _control(seq)
    m.fit_sequence(seq)
    a = m.predict_next(seq[0])
    m.predict_next(seq[3])
    m.predict_next(seq[5])
    assert np.array_equal(a, m.predict_next(seq[0])), "codec leaked state across probes"


def test_training_draws_fresh_noise_but_probes_do_not():
    """Repeated presentations must be independent Poisson draws -- otherwise a
    spiking arm sees one frozen sample -- while probes must not be."""
    enc = PopulationSpikeEncoder(8, seed=0)
    v = np.ones(8) / np.sqrt(8)
    assert not np.array_equal(enc.encode(v), enc.encode(v))

    inner = AsymmetricHopfieldNetwork(n_features=8)
    m = wrap_with_codec(inner, enc, inner_domain="vectors")
    assert np.array_equal(m._to_inner(v, rng=m._probe_rng()),
                          m._to_inner(v, rng=m._probe_rng()))


def test_online_capability_is_preserved_and_never_invented():
    """Nominal, not duck-typed: the wrapper cannot claim a capability its inner
    arm does not declare."""
    seq = _seq()
    enc, _ = make_codec(seq.shape[1], seed=0)

    online = wrap_with_codec(TemporalPCNetwork(n_features=seq.shape[1], seed=0),
                             enc, inner_domain="vectors")
    assert isinstance(online, OnlineCodecWrappedModel) and supports_online(online)
    online.fit_event(seq[0])
    online.on_event_boundary()

    offline = wrap_with_codec(AsymmetricHopfieldNetwork(n_features=seq.shape[1]),
                              enc, inner_domain="vectors")
    assert type(offline) is CodecWrappedModel and not supports_online(offline)
    with pytest.raises(TypeError):
        OnlineCodecWrappedModel(AsymmetricHopfieldNetwork(n_features=seq.shape[1]), enc)


def test_declared_facts_come_from_the_inner_arm():
    seq = _seq()
    enc, _ = make_codec(seq.shape[1], seed=0)
    inner = AsymmetricHopfieldNetwork(n_features=seq.shape[1])
    m = wrap_with_codec(inner, enc, inner_domain="vectors")
    assert m.rollout_mode is inner.rollout_mode
    assert m.params["inner"] == "AsymmetricHopfieldNetwork"
    assert m.params["n_per_feature"] == enc.n_per_feature      # spec constraint 5
    # Numeric only -- test_epochs_kwarg_contract flattens every value.
    assert all(isinstance(v, np.ndarray) for v in m.get_latent_state().values())


def test_bad_inner_domain_is_rejected():
    enc, _ = make_codec(4, seed=0)
    with pytest.raises(ValueError):
        CodecWrappedModel(AsymmetricHopfieldNetwork(n_features=4), enc,
                          inner_domain="currents")


# ------------------------------------------------- ingestion-regime identity
def test_batch_and_streamed_ingestion_are_the_same_protocol():
    """``online_equivalent`` asserts the two regimes differ only in ingestion.

    Behind a *stochastic* codec that is not free: encoding once and replaying
    the sample would train batch on one Poisson realisation repeated N times
    while streaming saw N independent ones, and the suite would report the
    difference as an ingestion-regime effect. ``resample_per_pass`` is what
    makes the claim true; this pins it.
    """
    from memval.benchmarks.ingest import ingest

    seq = _seq()
    enc_kw = dict(n_per_feature=2, window_steps=20)

    def build():
        enc, _ = make_codec(seq.shape[1], seed=0, **enc_kw)
        return wrap_with_codec(
            BushSTDPSpikingNetwork(n_neurons=enc.n_neurons, n_presentations=3,
                                   recall_steps=10, seed=0), enc)

    batched = build()
    batched.fit_sequence(seq)
    streamed = build()
    for _ in range(3):
        ingest(streamed, seq, regime="streamed")
    assert np.array_equal(batched.inner.W, streamed.inner.W)


def test_freezing_the_sample_would_have_changed_the_answer():
    """The measurement behind ``resample_per_pass``: not a rounding difference."""
    from memval.benchmarks.ingest import ingest

    seq = _seq()

    def build(resample):
        enc, _ = make_codec(seq.shape[1], seed=0, n_per_feature=2, window_steps=20)
        inner = BushSTDPSpikingNetwork(n_neurons=enc.n_neurons, n_presentations=3,
                                       recall_steps=10, seed=0)
        if resample:
            return wrap_with_codec(inner, enc)
        # CodecWrappedModel directly, not the online subclass: an online arm now
        # always trains through fit_event, which resamples per pass by
        # construction, so the frozen behaviour only exists on the base path.
        return CodecWrappedModel(inner, enc, resample_per_pass=False)

    frozen = build(False)
    frozen.fit_sequence(seq)
    streamed = build(True)
    for _ in range(3):
        ingest(streamed, seq, regime="streamed")
    assert np.abs(frozen.inner.W - streamed.inner.W).max() > 1e-3


def test_online_arms_train_through_fit_event():
    """An arm that declares OnlineTrainable is trained event-by-event whichever
    regime the caller asked for, so a batch suite and a streamed suite exercise
    the same code path on it.

    Verified two ways: every training call must land on ``fit_event``, and the
    resulting weights must equal the streamed path byte for byte.
    """
    from memval.benchmarks.ingest import ingest

    seq = _seq()
    enc_kw = dict(n_per_feature=2, window_steps=20)

    def build():
        enc, _ = make_codec(seq.shape[1], seed=0, **enc_kw)
        return wrap_with_codec(
            BushSTDPSpikingNetwork(n_neurons=enc.n_neurons, n_presentations=3,
                                   recall_steps=10, seed=0), enc)

    batched = build()
    calls = {"fit_event": 0, "inner_fit_sequence": 0}
    real_fit_event = batched.fit_event
    batched.fit_event = lambda *a, **k: (calls.__setitem__("fit_event",
                                         calls["fit_event"] + 1),
                                         real_fit_event(*a, **k))[1]
    batched.inner.fit_sequence = lambda *a, **k: calls.__setitem__(
        "inner_fit_sequence", calls["inner_fit_sequence"] + 1)
    batched.fit_sequence(seq)
    assert calls["fit_event"] == 3 * len(seq)
    assert calls["inner_fit_sequence"] == 0, "batch path bypassed fit_event"

    plain = build()
    plain.fit_sequence(seq)
    streamed = build()
    for _ in range(3):
        ingest(streamed, seq, regime="streamed")
    assert np.array_equal(plain.inner.W, streamed.inner.W)


def test_exposure_assignment_reaches_the_inner_arm():
    """The suite's ``_fit`` sets ``model.n_epochs`` and passes ``epochs=``; both
    have to land on the arm, not on the wrapper."""
    seq = _seq()
    enc, _ = make_codec(seq.shape[1], seed=0, n_per_feature=2, window_steps=20)
    inner = BushSTDPSpikingNetwork(n_neurons=enc.n_neurons, n_presentations=1,
                                   recall_steps=10, seed=0)
    m = wrap_with_codec(inner, enc)
    m.n_epochs = 4
    assert inner.n_presentations == 4 and m.n_epochs == 4

    by_attr = wrap_with_codec(
        BushSTDPSpikingNetwork(n_neurons=enc.n_neurons, n_presentations=4,
                               recall_steps=10, seed=0),
        *make_codec(seq.shape[1], seed=0, n_per_feature=2, window_steps=20)[0:1])
    by_attr.fit_sequence(seq)
    by_kwarg = wrap_with_codec(
        BushSTDPSpikingNetwork(n_neurons=enc.n_neurons, n_presentations=1,
                               recall_steps=10, seed=0),
        *make_codec(seq.shape[1], seed=0, n_per_feature=2, window_steps=20)[0:1])
    by_kwarg.fit_sequence(seq, epochs=4)
    assert np.array_equal(by_attr.inner.W, by_kwarg.inner.W)


# ------------------------------------------------------- transport control
def test_transport_costs_something_but_not_everything():
    """Spec S5. The control must be *degraded* relative to native -- otherwise
    it is not measuring transport -- and must stay well above chance, otherwise
    it cannot serve as a reference for a spiking arm."""
    seq = _seq()
    native = AsymmetricHopfieldNetwork(n_features=seq.shape[1], n_epochs=30)
    native.fit_sequence(seq)
    wrapped = _control(seq)
    wrapped.fit_sequence(seq)

    def mean_cos(m):
        vals = []
        for t in range(len(seq) - 1):
            p = m.predict_next(seq[t])
            n = np.linalg.norm(p)
            vals.append(float(p @ seq[t + 1] / n) if n else 0.0)
        return float(np.mean(vals))

    n_cos, w_cos = mean_cos(native), mean_cos(wrapped)
    assert w_cos < n_cos, "transport that costs nothing is not measuring anything"
    assert w_cos > 0.5 * n_cos, f"transport ate the arm: {w_cos:.3f} vs {n_cos:.3f}"


def test_more_neurons_buy_back_the_transport_cost():
    """The gap has to be attributable to the codec, so it must close as the
    codec is given more Poisson events per dimension."""
    seq = _seq()
    gaps = []
    for n_per in (1, 5, 20):
        m = _control(seq, n_per_feature=n_per)
        m.fit_sequence(seq)
        gaps.append(np.mean([float(m.predict_next(seq[t]) @ seq[t + 1])
                             for t in range(len(seq) - 1)]))
    assert gaps[0] < gaps[1] < gaps[2], gaps


def test_codec_identity_bounds_what_transport_alone_can_score():
    """Spec S5's second control: a ``predict_next`` that returns the cue
    unchanged, run through the codec. Cheap, and it says what a score means
    before any learning happens."""
    seq = _seq()
    enc, dec = make_codec(seq.shape[1], seed=0)
    identity = np.array([dec.decode(enc.encode(v)) for v in seq])
    self_cos = np.mean([float(identity[i] @ seq[i]) for i in range(len(seq))])
    next_cos = np.mean([float(identity[i] @ seq[i + 1]) for i in range(len(seq) - 1)])
    assert self_cos > 0.95
    assert next_cos < self_cos


# -------------------------------------------------------- spiking transport
def test_a_spiking_arm_scores_through_the_same_seam():
    seq = _seq()
    enc, _ = make_codec(seq.shape[1], seed=0, n_per_feature=2, window_steps=20)
    inner = BushSTDPSpikingNetwork(n_neurons=enc.n_neurons, n_presentations=3,
                                   recall_steps=25, seed=0)
    m = wrap_with_codec(inner, enc)
    assert isinstance(m, OnlineCodecWrappedModel)
    m.fit_sequence(seq)
    out = m.predict_next(seq[0])
    assert out.shape == (seq.shape[1],)
    assert np.array_equal(out, m.predict_next(seq[0])), "spiking probe is not pure"
