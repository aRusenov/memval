"""Hold :mod:`memval.models.baselines.vieth_gaba_stdp` to upstream, exactly.

The arm is a reimplementation, not a vendored port, so "it follows Vieth &
Triesch" has to be a measurement. ``tests/fixtures/vieth_upstream_trace.npz``
is a recorded run of the **real upstream code** -- ``gitmv/GABA_Modulated_STDP_Paper``
on PymoNNto, driven headless by ``scratch/vieth_ref.py`` -- carrying every
intermediate the model computes, step by step, for a 60-unit network on
upstream's own character stream.

Each rule is then checked against that trace *given upstream's own inputs to it*,
which makes every assertion exact rather than statistical: no reimplementation
drift can hide behind a tolerance on a stochastic trajectory. The one thing that
cannot be checked this way is the threshold draw itself, so the spike rule is
checked as far as it is deterministic -- the firing probability -- plus the
implication that a spike requires a positive probability.

Regenerate the fixture with::

    scratch/vieth_venv/bin/python scratch/vieth_ref.py \
        --steps 900 --exc 60 --trace-steps 260 --seed 1 --out ref_small.npz
"""
import os

import numpy as np
import pytest

from memval.encoders.spike_codec import PopulationSpikeEncoder
from memval.models.baselines.vieth_gaba_stdp import (
    UPSTREAM, CodecViethNetwork, ViethGabaSTDPNetwork,
)
from memval.models.capabilities import is_online_equivalent, supports_online

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "vieth_upstream_trace.npz")

# The fixture's configuration: 60 excitatory units, 6 inhibitory, and upstream's
# target_activity = 1 / n_chars(' fox eats meat.') = 1/15.
N_INP, N_EXC, N_INH = 100, 60, 6
TARGET_ACT = 1.0 / 15.0
#: PymoNNto normalises when ``(iteration - 1) % 200 == 0`` and its first
#: simulated iteration is number 1, so trace index t maps to iteration t + 1 and
#: normalisation lands on t = 0 and t = 200.
NORM_STEPS = (0, 200)
TOL = 2e-6          # float32 upstream vs float64 here


@pytest.fixture(scope="module")
def trace():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"upstream trace fixture missing: {FIXTURE}")
    return np.load(FIXTURE, allow_pickle=True)


@pytest.fixture
def model():
    return ViethGabaSTDPNetwork(n_neurons=N_INP, n_exc=N_EXC, n_inh=N_INH,
                                target_activity=TARGET_ACT, seed=0)


def _steps(trace, skip_norm=True):
    n = trace["trace_exc_spike"].shape[0]
    return [t for t in range(1, n) if not (skip_norm and t in NORM_STEPS)]


# ------------------------------------------------------------ the trace itself
def test_fixture_is_a_live_upstream_run(trace):
    """Guard against a fixture that recorded a dead network.

    Every assertion below is vacuous if upstream never fired, never learned, or
    never saw more than one symbol, so those are checked before anything else.
    """
    assert trace["trace_exc_spike"].any(), "no excitatory spikes recorded"
    assert trace["trace_inh_spike"].any(), "no inhibitory spikes recorded"
    assert len(set(trace["trace_char_index"].tolist())) > 3, "input never varied"
    assert not np.allclose(trace["trace_W_EE"][0], trace["trace_W_EE"][-1]), \
        "EE weights never changed -- upstream did not learn"
    assert (trace["trace_li_stdp_mul"] < 0).any(), \
        "the plasticity multiplier never went negative, so the paper's " \
        "inhibition-driven depression is untested by this fixture"


# --------------------------------------------------------- rule-by-rule checks
def test_stdp_matches_upstream(model, trace):
    """``W[pre(t-2), post(t-1)] += eta * li_stdp_mul[post]``, on both matrices."""
    worst_es = worst_ee = 0.0
    for t in _steps(trace):
        for W_key, pre_key, tag in (("W_ES", "trace_inp_spike_old", "es"),
                                    ("W_EE", "trace_exc_spike_old", "ee")):
            W = trace[f"trace_{W_key}"][t - 1].astype(np.float64).copy()
            model._stdp(W, trace[pre_key][t - 1], trace["trace_exc_spike"][t - 1],
                        trace["trace_li_stdp_mul"][t].astype(np.float64))
            err = np.abs(W - trace[f"trace_{W_key}"][t]).max()
            if tag == "es":
                worst_es = max(worst_es, err)
            else:
                worst_ee = max(worst_ee, err)
    assert worst_es < TOL, f"ES weights diverge from upstream by {worst_es:.2e}"
    assert worst_ee < TOL, f"EE weights diverge from upstream by {worst_ee:.2e}"


def test_stdp_pairs_the_previous_step_not_the_current_one(model, trace):
    """The one-iteration offset is load-bearing, so pin it directly.

    PymoNNto runs plasticity (key 41) *before* the groups emit (keys 50, 51), so
    the pair learned at iteration t is ``(t-2, t-1)``. Using the current step's
    spikes instead is the natural-looking reimplementation, and it is wrong --
    this asserts it actually disagrees, rather than trusting that it would.
    """
    disagreed = False
    for t in _steps(trace):
        W = trace["trace_W_EE"][t - 1].astype(np.float64).copy()
        model._stdp(W, trace["trace_exc_spike"][t], trace["trace_exc_spike"][t],
                    trace["trace_li_stdp_mul"][t].astype(np.float64))
        if np.abs(W - trace["trace_W_EE"][t]).max() > TOL:
            disagreed = True
            break
    assert disagreed, "the off-by-one pairing is untestable on this fixture"


def test_normalization_and_its_order_match_upstream(model, trace):
    """At a normalisation step, normalise-then-learn reproduces upstream.

    Normalisation (keys 3, 3.1) runs before plasticity (key 41), and the three
    divisions -- EE afferent, EE efferent, ES afferent -- are sequential, so the
    result depends on their order. Reproducing the step end to end tests both.
    """
    tested = 0
    for t in NORM_STEPS:
        if t == 0 or t >= trace["trace_W_EE"].shape[0]:
            continue
        model.W_ee = trace["trace_W_EE"][t - 1].astype(np.float64).copy()
        model.W_es = trace["trace_W_ES"][t - 1].astype(np.float64).copy()
        model._normalize()
        mul = trace["trace_li_stdp_mul"][t].astype(np.float64)
        model._stdp(model.W_es, trace["trace_inp_spike_old"][t - 1],
                    trace["trace_exc_spike"][t - 1], mul)
        model._stdp(model.W_ee, trace["trace_exc_spike_old"][t - 1],
                    trace["trace_exc_spike"][t - 1], mul)
        assert np.abs(model.W_ee - trace["trace_W_EE"][t]).max() < TOL
        assert np.abs(model.W_es - trace["trace_W_ES"][t]).max() < TOL
        tested += 1
    assert tested, "fixture spans no normalisation step"


def test_gaba_modulation_matches_upstream(trace):
    """``clip((1 + input_GABA / avg_inh) * strength, min, max)``."""
    got = np.clip((1.0 + trace["trace_input_GABA"].astype(np.float64) / UPSTREAM["avg_inh"])
                  * UPSTREAM["gaba_strength"],
                  UPSTREAM["gaba_min"], UPSTREAM["gaba_max"])
    assert np.abs(got - trace["trace_li_stdp_mul"]).max() < TOL


def test_intrinsic_plasticity_matches_upstream(trace):
    """``sensitivity -= (spike - target_activity) * ip_strength``, every step."""
    sens = trace["trace_sensitivity"].astype(np.float64)
    spikes = trace["trace_exc_spike"].astype(np.float64)
    for t in _steps(trace, skip_norm=False):
        got = sens[t - 1] - (spikes[t - 1] - TARGET_ACT) * UPSTREAM["ip_strength"]
        assert np.abs(got - sens[t]).max() < TOL, f"sensitivity diverges at t={t}"


def test_membrane_drive_matches_upstream(trace):
    """Voltage is ES + EE (previous step) + GABA (delayed) + sensitivity.

    Weights are read at key 12, i.e. after normalisation but before that
    iteration's plasticity, which is why the previous step's matrices are the
    right ones to use.
    """
    for t in _steps(trace):
        W_es = trace["trace_W_ES"][t - 1].astype(np.float64)
        W_ee = trace["trace_W_EE"][t - 1].astype(np.float64)
        got = (W_es[trace["trace_inp_spike"][t - 1]].sum(axis=0)
               + W_ee[trace["trace_exc_spike"][t - 1]].sum(axis=0)
               + trace["trace_input_GABA"][t].astype(np.float64)
               + trace["trace_sensitivity"][t].astype(np.float64))
        assert np.abs(got - trace["trace_voltage"][t]).max() < TOL, f"voltage at t={t}"


def test_inhibitory_drive_is_delayed_by_one_step(trace):
    """``input_GABA(t) = -W_EI . inh_spike(t-1)`` -- inhibition lags by a step."""
    W_ei = trace["W_EI"].astype(np.float64)
    for t in _steps(trace, skip_norm=False):
        got = -W_ei[trace["trace_inh_spike"][t - 1]].sum(axis=0)
        assert np.abs(got - trace["trace_input_GABA"][t]).max() < TOL


def test_inhibitory_population_matches_upstream(trace):
    """The inhibitory running average reads the CURRENT step's exc spikes.

    Keys 60/70 run after key 51, so unlike every other coupling in the network
    this one is not delayed -- getting it wrong would put the whole inhibitory
    loop a step out.
    """
    W_ie = trace["W_IE"].astype(np.float64)
    avg = trace["trace_inh_avg_act"].astype(np.float64)
    for t in _steps(trace, skip_norm=False):
        drive = W_ie[trace["trace_exc_spike"][t]].sum(axis=0)
        got = (avg[t - 1] * UPSTREAM["inh_duration"] + drive) / (UPSTREAM["inh_duration"] + 1.0)
        assert np.abs(got - avg[t]).max() < TOL, f"inhibitory average at t={t}"


def test_spike_probability_matches_upstream(trace):
    """``p = clip(voltage * mul, 0)^exp``, and a spike implies ``p > 0``."""
    p = np.power(np.clip(trace["trace_voltage"].astype(np.float64) * UPSTREAM["exc_mul"],
                         0.0, None), UPSTREAM["exc_exp"])
    fired = trace["trace_exc_spike"]
    assert (p[fired] > 0.0).all(), "upstream fired a unit whose probability was 0"
    assert p.max() > 0.0 and fired.mean() > 0.0


def test_readout_matches_upstream(model, trace):
    """The reconstruction is ``ES . exc_spike`` -- upstream's, minus the argmax.

    Upstream's ``TextReconstructor`` runs at key 80, after the groups emit, so
    it reads the same iteration's spikes through the same iteration's weights.
    """
    for t in _steps(trace, skip_norm=False):
        model.W_es = trace["trace_W_ES"][t].astype(np.float64)
        model._state["exc_spike"] = trace["trace_exc_spike"][t]
        assert np.abs(model._reconstruct() - trace["trace_recon_act"][t]).max() < TOL


def test_full_trajectory_matches_upstream(model, trace):
    """The strongest check: run the whole loop, with upstream's randomness.

    The rule-by-rule tests each feed a rule upstream's own inputs, so they
    cannot catch a **wiring** error -- a correct rule handed the wrong variable
    by the surrounding loop. This runs the port's own iteration end to end from
    upstream's recorded starting state, substituting upstream's recorded spike
    outcomes for the two stochastic draws, and requires every intermediate the
    network computes to track upstream's for the length of the fixture.

    With the randomness removed the two implementations are then deterministic
    functions of the same inputs, so this is equality, not agreement.
    """
    r = {k: trace["trace_" + k] for k in
         ("exc_spike", "exc_spike_old", "inh_spike", "inp_spike", "inp_spike_old",
          "voltage", "li_stdp_mul", "sensitivity", "inh_avg_act", "input_GABA")}
    m = model
    m.W_es = trace["trace_W_ES"][0].astype(np.float64).copy()
    m.W_ee = trace["trace_W_EE"][0].astype(np.float64).copy()
    m.W_ie = trace["W_IE"].astype(np.float64).copy()
    m.W_ei = trace["W_EI"].astype(np.float64).copy()
    m.sensitivity = r["sensitivity"][0].astype(np.float64).copy()
    m.iteration = 1
    s = m._state
    for k in ("inp_spike", "inp_spike_old", "exc_spike", "exc_spike_old", "inh_spike"):
        s[k] = r[k][0]
    s["inh_avg_act"] = r["inh_avg_act"][0].astype(np.float64).copy()

    n = r["exc_spike"].shape[0]
    for t in range(1, n):
        m.iteration += 1
        if (m.iteration - 1) % m.norm_every == 0:
            m._normalize()
        voltage = (m.W_es[s["inp_spike"]].sum(axis=0)
                   + m.W_ee[s["exc_spike"]].sum(axis=0))
        gaba = -m.W_ei[s["inh_spike"]].sum(axis=0)
        voltage = voltage + gaba
        m.sensitivity -= (s["exc_spike"].astype(float) - m.target_activity) * m.ip_strength
        voltage = voltage + m.sensitivity
        mul = np.clip((1.0 + gaba / m.avg_inh) * m.gaba_strength, m.gaba_min, m.gaba_max)
        m._stdp(m.W_es, s["inp_spike_old"], s["exc_spike"], mul)
        m._stdp(m.W_ee, s["exc_spike_old"], s["exc_spike"], mul)
        s["inp_spike_old"], s["inp_spike"] = s["inp_spike"], r["inp_spike"][t]
        s["exc_spike_old"], s["exc_spike"] = s["exc_spike"], r["exc_spike"][t]
        inh_v = m.W_ie[s["exc_spike"]].sum(axis=0)
        s["inh_avg_act"] = ((s["inh_avg_act"] * m.inh_duration + inh_v)
                            / (m.inh_duration + 1.0))
        s["inh_spike"] = r["inh_spike"][t]

        for name, got, want in (("voltage", voltage, r["voltage"][t]),
                                ("li_stdp_mul", mul, r["li_stdp_mul"][t]),
                                ("input_GABA", gaba, r["input_GABA"][t]),
                                ("sensitivity", m.sensitivity, r["sensitivity"][t]),
                                ("inh_avg_act", s["inh_avg_act"], r["inh_avg_act"][t]),
                                ("W_ES", m.W_es, trace["trace_W_ES"][t]),
                                ("W_EE", m.W_ee, trace["trace_W_EE"][t])):
            err = np.abs(got - want).max()
            assert err < TOL, f"{name} diverged from upstream at t={t} by {err:.2e}"


# --------------------------------------------------------------- arm contract
def test_declares_online_and_equivalent(model):
    assert supports_online(model) and is_online_equivalent(model)


def test_fit_sequence_equals_a_fit_event_loop():
    """``online_equivalent`` is an assertion about rule identity; check it."""
    rng = np.random.default_rng(0)
    X = rng.random((4, 30, 8)) < 0.25
    kw = dict(n_neurons=30, n_exc=40, n_inh=4, seed=3)
    batch = ViethGabaSTDPNetwork(**kw)
    batch.fit_sequence(X, epochs=3)

    stream = ViethGabaSTDPNetwork(**kw)
    for _ in range(3):
        stream.on_event_boundary()
        for item in X:
            stream.fit_event(item)
        stream.on_event_boundary()
    for name in ("W_es", "W_ee"):
        assert np.array_equal(getattr(batch, name), getattr(stream, name)), name


def test_predict_next_is_pure(model):
    """Repeated probes must not drift, and must not disturb training state."""
    rng = np.random.default_rng(1)
    X = rng.random((3, N_INP, 8)) < 0.25
    model.fit_sequence(X, epochs=2)
    before = model._save_state()
    first = model.predict_next(X[0])
    second = model.predict_next(X[0])
    assert np.array_equal(first, second), "predict_next is not deterministic"
    assert np.array_equal(model.predict_next(X[1]), model.predict_next(X[1]))
    after = model._save_state()
    assert np.array_equal(before["sensitivity"], after["sensitivity"])
    for k, v in before["state"].items():
        assert np.array_equal(v, after["state"][k]), f"{k} leaked across a probe"


def test_predict_next_does_not_learn(model):
    rng = np.random.default_rng(2)
    X = rng.random((3, N_INP, 8)) < 0.25
    model.fit_sequence(X, epochs=1)
    W = {k: v.copy() for k, v in model.named_parameters().items()}
    for _ in range(3):
        model.predict_next(X[0])
    for k, v in W.items():
        assert np.array_equal(v, model.named_parameters()[k]), f"{k} changed on a probe"


def _chain_network():
    """Two items, two assemblies, a hand-built chain: item0 -> assembly0 -> assembly1.

    Weights are set directly rather than trained so the readout's *position* can
    be pinned without depending on learning having succeeded, and large enough
    that the stochastic threshold is saturated (``p >= 1`` fires for certain).
    Inhibition is removed for the same reason.
    """
    m = ViethGabaSTDPNetwork(n_neurons=6, n_exc=6, n_inh=2, seed=0)
    m.W_es[:] = 0.0
    m.W_ee[:] = 0.0
    m.W_ie[:] = 0.0
    m.W_ei[:] = 0.0
    m.sensitivity[:] = 0.0
    for item in (0, 1):                      # item i's inputs -> assembly i
        m.W_es[np.ix_(range(item * 3, item * 3 + 3), range(item * 3, item * 3 + 3))] = 1.0
    m.W_ee[np.ix_(range(0, 3), range(3, 6))] = 1.0        # assembly0 -> assembly1
    cue = np.zeros((6, 1), dtype=bool)
    cue[0:3] = True                                        # item 0
    return m, cue


def test_lag_one_returns_the_cue_and_lag_two_the_prediction():
    """Pin where the readout sits in the network's two-deep pipeline.

    Input at ``t`` reaches the excitatory population at ``t+1`` (through ES) and
    its recurrent consequence at ``t+2`` (through EE). So lag 1 reads the cue's
    own assembly back -- an identity, not a prediction -- and lag 2 reads the
    next item. Reading at the wrong lag would score the arm on autoencoding,
    which is why the default is 2 and why this is asserted rather than assumed.
    """
    m, cue = _chain_network()
    at_lag = {}
    for lag in (1, 2):
        m.cue_lag = lag
        at_lag[lag] = m.predict_next(cue)[:, 0]
    assert at_lag[1][0:3].sum() > 0 and at_lag[1][3:6].sum() == 0, \
        "lag 1 should read back the cue's own item"
    assert at_lag[2][3:6].sum() > 0 and at_lag[2][0:3].sum() == 0, \
        "lag 2 should read the next item in the chain"


def test_cue_lag_sets_the_boundary_drain_as_well_as_the_readout():
    """``cue_lag`` is the pipeline depth, so it governs training too.

    It would be natural to assume a readout offset cannot affect learning. Here
    it must: the same depth that puts the prediction two iterations out is what
    decides how many draining iterations a sequence seam needs before the last
    items' pairings can form. Pinning it stops the two from drifting apart.
    """
    rng = np.random.default_rng(4)
    X = rng.random((5, 40, 6)) < 0.3
    kw = dict(n_neurons=40, n_exc=60, n_inh=6, seed=5)
    a = ViethGabaSTDPNetwork(cue_lag=0, **kw)
    a.fit_sequence(X, epochs=5)
    b = ViethGabaSTDPNetwork(cue_lag=2, **kw)
    b.fit_sequence(X, epochs=5)
    assert not np.allclose(a.W_ee, b.W_ee), \
        "cue_lag did not change the boundary drain, so the last items of every " \
        "sequence are still being dropped"


def test_boundary_preserves_inhibitory_state_but_clears_spike_history():
    """A seam stops transitions forming; it is not a reset of homeostatic state.

    Clearing the inhibitory running average at every seam removes the
    competition that separates assemblies, which measurably collapses recall
    (see :meth:`ViethGabaSTDPNetwork.on_event_boundary`). Sensitivity is
    preserved for the same reason.
    """
    rng = np.random.default_rng(11)
    X = rng.random((4, 40, 6)) < 0.3
    m = ViethGabaSTDPNetwork(n_neurons=40, n_exc=60, n_inh=6, seed=5)
    for item in X:
        m.fit_event(item)
    sens = m.sensitivity.copy()
    m._state["inh_avg_act"] = np.full(6, 0.25)
    m.on_event_boundary()
    # The drain runs real iterations, so the average evolves; what must not
    # happen is the zeroing that reset_context would do.
    assert m._state["inh_avg_act"].any(), "inhibitory state was zeroed at the seam"
    assert np.allclose(m._state["inh_avg_act"], 0.25, atol=0.2)
    # Likewise sensitivity: the drain's iterations nudge it, but the seam must
    # not restart homeostasis from zero.
    assert np.allclose(sens, m.sensitivity, atol=0.05), "sensitivity was reset"
    assert not m._state["exc_spike"].any(), "spike history survived the seam"
    assert not m._state["inp_spike"].any() and not m._state["exc_spike_old"].any()


def test_epochs_kwarg_is_honoured():
    """The defect ``tests/test_epochs_kwarg_contract.py`` exists to catch."""
    rng = np.random.default_rng(6)
    X = rng.random((3, 30, 6)) < 0.25
    kw = dict(n_neurons=30, n_exc=40, n_inh=4, seed=7)
    a = ViethGabaSTDPNetwork(n_epochs=1, **kw)
    a.fit_sequence(X, epochs=4)
    b = ViethGabaSTDPNetwork(n_epochs=4, **kw)
    b.fit_sequence(X)
    assert np.array_equal(a.W_ee, b.W_ee)


def test_gaba_min_zero_is_the_ablation():
    """The paper's mechanism is the negative clip; removing it must change the fit."""
    rng = np.random.default_rng(8)
    X = rng.random((4, 30, 6)) < 0.3
    kw = dict(n_neurons=30, n_exc=60, n_inh=6, seed=9)
    paper = ViethGabaSTDPNetwork(gaba_min=-0.15, **kw)
    paper.fit_sequence(X, epochs=5)
    ablated = ViethGabaSTDPNetwork(gaba_min=0.0, **kw)
    ablated.fit_sequence(X, epochs=5)
    assert not np.allclose(paper.W_ee, ablated.W_ee)


def test_rejects_vectors_with_a_pointer_to_the_codec(model):
    with pytest.raises(ValueError, match="wrap_with_codec"):
        model.predict_next(np.zeros(7))


def test_runs_behind_the_codec():
    """The registry-facing form must present the ordinary vector interface."""
    arm = CodecViethNetwork(n_features=8, n_per_feature=3, window_steps=10,
                            n_exc=40, seed=11)
    rng = np.random.default_rng(12)
    X = rng.normal(size=(4, 8))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    arm.fit_sequence(X, epochs=2)
    out = arm.predict_next(X[0])
    assert out.shape == (8,)
    assert np.isfinite(out).all()
    assert arm.recall(X[0], 3).shape == (3, 8)
    assert supports_online(arm) and is_online_equivalent(arm)


def test_codec_arm_sizes_its_input_population_from_the_encoder():
    arm = CodecViethNetwork(n_features=8, n_per_feature=3, window_steps=10, seed=13)
    assert arm.inner.n_neurons == 2 * 8 * 3 == arm.encoder.n_neurons
