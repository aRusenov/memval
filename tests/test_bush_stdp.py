"""Bush et al. (2010) spiking arm: the unit tests named in
``docs/spike_codec_spec.md`` S2.5, plus the correctness gate in miniature.

The spec is explicit that the paper's own result must be reproduced *before*
the arm's MemVal scores mean anything. ``bin/bush_stdp_gate.py`` runs the full
50-seed gate; ``test_correctness_gate_in_miniature`` here is the regression
guard that keeps it from silently rotting.
"""
import numpy as np
import pytest

from memval.encoders.spike_codec import make_codec
from memval.models.capabilities import RolloutMode, supports_online
from memval.models.baselines.bush_stdp import (
    BushSTDPSpikingNetwork, dual_coded_stimulus, order_fidelity,
)


def _quiet(n=4, **kw):
    """A network with recurrence and inhibition off, so a test sees only STDP."""
    kw.setdefault("g_syn", 0.0)
    kw.setdefault("k_inh", 0.0)
    kw.setdefault("w_init_frac", 0.5)
    return BushSTDPSpikingNetwork(n_neurons=n, seed=0, **kw)


def _force_spike(m, idx):
    """Run one step in which exactly ``idx`` fires.

    Parking the membrane just under threshold is the only way to place a spike
    at an exact step: an Izhikevich neuron driven from rest takes about 3 ms to
    reach threshold, which is far too coarse to test a 1 ms STDP kernel.
    """
    m._state["v"][:] = m.c
    m._state["v"][idx] = 29.9
    fired = m._step(np.zeros(m.n_neurons), plastic=True)
    assert fired[idx] and fired.sum() == 1
    return fired


def _idle(m, steps):
    for _ in range(steps):
        m._step(np.zeros(m.n_neurons), plastic=True)


# ------------------------------------------------------------ STDP kernel
@pytest.mark.parametrize("dt", [1, 3, 10, 20, 40])
def test_stdp_kernel_matches_the_analytic_exponentials_at_both_signs(dt):
    """Spec S2.5: pre-before-post potentiates by ``A_plus exp(-dt/tau_plus)``,
    post-before-pre depresses by ``A_minus exp(-dt/tau_minus)``.

    Both signs come out of one run: neuron 0 spikes, then neuron 1 spikes ``dt``
    later, so the 0->1 synapse sees a causal pairing and the 1->0 synapse sees
    the anti-causal one.
    """
    m = _quiet()
    w_causal = m.W[1, 0]
    w_anti = m.W[0, 1]

    _force_spike(m, 0)
    _idle(m, dt - 1)
    _force_spike(m, 1)

    ltp = m.W[1, 0] - w_causal
    ltd = m.W[0, 1] - w_anti
    assert ltp == pytest.approx(m.a_plus * np.exp(-dt / m.tau_plus), rel=1e-9)
    assert ltd == pytest.approx(m.a_minus * np.exp(-dt / m.tau_minus), rel=1e-9)
    assert ltp > 0 > ltd


def test_simultaneous_spikes_neither_potentiate_nor_depress():
    """dt = 0 sits at the discontinuity; the trace form must not double-count."""
    m = _quiet()
    before = m.W.copy()
    m._state["v"][:] = 29.9
    m._step(np.zeros(m.n_neurons), plastic=True)
    assert np.array_equal(m.W, before)


def test_bcm_variant_has_the_wider_depression_window():
    """The default's whole point: symmetric pairings net-depress, causal ones
    potentiate. That imbalance is what makes the learned matrix directional."""
    m = _quiet()
    assert m.a_minus * m.tau_minus < 0
    assert abs(m.a_minus * m.tau_minus) > m.a_plus * m.tau_plus
    nb = _quiet(variant="non_bcm")
    assert nb.tau_plus == nb.tau_minus
    with pytest.raises(ValueError):
        _quiet(variant="triplet")


# --------------------------------------------------------------- stability
def test_weights_stay_in_bounds_under_adversarial_spike_trains():
    """Spec S2.5. Every neuron firing at the ceiling for a long window is the
    worst case for an *additive* rule with no multiplicative soft bound."""
    m = BushSTDPSpikingNetwork(n_neurons=12, seed=0)
    drive = np.ones((12, 400), dtype=bool)
    m.fit_event(drive)
    assert m.W.min() >= 0.0 and m.W.max() <= m.w_max
    assert np.all(np.diag(m.W) == 0.0), "recurrent, but no self-connections"

    m2 = BushSTDPSpikingNetwork(n_neurons=12, seed=0)
    rng = np.random.default_rng(3)
    for _ in range(6):
        m2.fit_event(rng.random((12, 120)) < 0.9)
    assert m2.W.min() >= 0.0 and m2.W.max() <= m2.w_max
    assert np.all(np.isfinite(m2.W))


@pytest.mark.parametrize("k_inh", [0.0, 0.05, 0.15, 0.5, 1.5, 6.0, 30.0])
def test_inhibition_never_increases_firing(k_inh):
    """The inhibitory pool must be inhibitory. Regression test for a real bug.

    An earlier version subtracted an unbounded negative current, which drove the
    Izhikevich recovery variable ``u`` deeply negative and made the network
    rebound synchronously: with recurrent excitation switched off entirely,
    raising ``k_inh`` from 0.3 to 1.5 took recall-window firing from 2.5% to
    87.5%. Inhibition was *causing* spikes.

    The fix has two halves and this test needs both. Making the term
    conductance-based (shunting toward ``e_inh``) is not sufficient on its own:
    integrated explicitly, ``h * g > 1`` overshoots, throwing v hundreds of mV
    past the reversal potential so the next substep sees a large *depolarising*
    drive -- the same pathology by a different route. The semi-implicit update
    is what makes it unconditionally stable in the conductance.

    ``g_syn = 0`` removes recurrent excitation, so nothing but the cue and the
    pool can drive anything and the comparison is unambiguous.
    """
    n, steps = 200, 20
    cue = np.zeros((n, 10), dtype=bool)
    cue[:n // 2, ::2] = True

    def firing(k):
        m = BushSTDPSpikingNetwork(n_neurons=n, g_syn=0.0, k_inh=k,
                                   recall_steps=steps, seed=0)
        return m.predict_next(cue).sum() / (n * steps)

    assert firing(k_inh) <= firing(0.0) + 1e-12, (
        f"k_inh={k_inh} produced MORE firing than no inhibition at all")


def test_no_self_connections_ever_appear():
    m = BushSTDPSpikingNetwork(n_neurons=8, seed=0)
    m.fit_event(np.ones((8, 60), dtype=bool))
    assert np.all(np.diag(m.W) == 0.0)


def test_sparse_connectivity_mask_is_respected():
    m = BushSTDPSpikingNetwork(n_neurons=20, n_presynaptic=6, seed=0)
    assert np.all(m.mask.sum(axis=1) == 6)
    m.fit_event(np.ones((20, 60), dtype=bool))
    assert np.all(m.W[m.mask == 0] == 0.0)


# ---------------------------------------------------------------- purity
def test_predict_next_does_not_move_weights_or_membrane_state():
    """Spec S2.5. ``Phi = 0`` keeps ``W`` fixed; save/restore keeps membrane
    state from leaking between probes that ``measure_recall_associative``
    assumes are independent -- the defect ``MultilayerTemporalPCNetwork`` had to
    avoid."""
    X = dual_coded_stimulus(n_fields=4, n_per_field=3, field_steps=20, seed=0)
    m = BushSTDPSpikingNetwork(n_neurons=12, n_presentations=3, seed=0)
    m.fit_sequence(X)

    W_before = m.W.copy()
    state_before = {k: v.copy() for k, v in m._state.items()}
    m.predict_next(X[0])
    assert np.array_equal(m.W, W_before), "the ACh gate did not hold"
    for k, v in state_before.items():
        assert np.array_equal(m._state[k], v), f"membrane state leaked via {k}"


def test_predict_next_is_a_pure_function_of_the_cue():
    X = dual_coded_stimulus(n_fields=4, n_per_field=3, field_steps=20, seed=0)
    m = BushSTDPSpikingNetwork(n_neurons=12, n_presentations=3, seed=0)
    m.fit_sequence(X)
    a = m.predict_next(X[0])
    m.predict_next(X[2])
    m.predict_next(X[3])
    assert np.array_equal(a, m.predict_next(X[0])), "predict_next leaked state"


def test_predict_next_returns_the_window_not_a_vector():
    """Spec S2.4: the decode choice belongs to ``PopulationSpikeDecoder``."""
    m = BushSTDPSpikingNetwork(n_neurons=10, recall_steps=17, seed=0)
    out = m.predict_next(np.ones((10, 5), dtype=bool))
    assert out.shape == (10, 17) and out.dtype == bool


def test_a_vector_argument_fails_with_a_pointed_message():
    m = BushSTDPSpikingNetwork(n_neurons=10, seed=0)
    with pytest.raises(ValueError, match="wrap_with_codec"):
        m.predict_next(np.ones(4))


# --------------------------------------------------------- reproducibility
def test_identical_seed_is_byte_identical_and_global_rng_untouched():
    X = dual_coded_stimulus(n_fields=5, n_per_field=3, field_steps=20, seed=1)

    def run(seed):
        m = BushSTDPSpikingNetwork(n_neurons=15, n_presentations=3, seed=seed)
        m.fit_sequence(X)
        return m.W.copy(), m.predict_next(X[0])

    before = np.random.get_state()
    W_a, o_a = run(0)
    after = np.random.get_state()
    W_b, o_b = run(0)
    assert np.array_equal(W_a, W_b) and np.array_equal(o_a, o_b)
    assert before[0] == after[0] and np.array_equal(before[1], after[1])


def test_one_presentation_is_one_epoch_on_every_path_the_suite_drives():
    """Exposure is the suite's common unit and this arm uses the same one.

    A presentation IS a pass over the material -- exactly what ``n_epochs``
    denotes for every rate arm -- so the arm goes on a criterion-referenced
    staircase unchanged. All three paths the suite actually uses are wired,
    because ``_fit`` helpers set the attribute *and* pass the kwarg, and
    silently swallowing either is the defect
    ``tests/test_epochs_kwarg_contract.py`` exists to catch.
    """
    X = dual_coded_stimulus(n_fields=4, n_per_field=3, field_steps=20, seed=0)

    def trained(**kw):
        m = BushSTDPSpikingNetwork(n_neurons=12, seed=0, **kw)
        return m

    by_ctor = trained(n_presentations=6)
    by_alias = trained(n_epochs=6)
    by_attr = trained(n_presentations=1)
    by_attr.n_epochs = 6
    by_kwarg = trained(n_presentations=1)

    for m in (by_ctor, by_alias, by_attr):
        m.fit_sequence(X)
    by_kwarg.fit_sequence(X, epochs=6)

    for other in (by_alias, by_attr, by_kwarg):
        assert np.array_equal(by_ctor.W, other.W), "exposure paths disagree"
    assert by_ctor.n_epochs == by_ctor.n_presentations == 6


def test_more_passes_means_more_learning():
    X = dual_coded_stimulus(n_fields=4, n_per_field=3, field_steps=20, seed=0)
    few = BushSTDPSpikingNetwork(n_neurons=12, n_presentations=1, seed=0)
    many = BushSTDPSpikingNetwork(n_neurons=12, n_presentations=8, seed=0)
    few.fit_sequence(X)
    many.fit_sequence(X)
    assert many.W.sum() > few.W.sum()


def test_exposure_staircases_like_any_other_arm():
    """``epochs_to_criterion`` documents ``fit(model, epochs)`` as 'trains for
    exactly ``epochs`` passes'. That contract has to hold here for the arm's
    exposure to be comparable with the rest of the roster."""
    from memval.benchmarks.exposure import epochs_to_criterion

    X = dual_coded_stimulus(n_fields=8, n_per_field=5, seed=1000)
    out = epochs_to_criterion(
        make_model=lambda: BushSTDPSpikingNetwork(n_neurons=40, recall_steps=20,
                                                  seed=0),
        fit=lambda m, e: m.fit_sequence(X, epochs=e),
        score=lambda m: order_fidelity(m.predict_next(X[0]), 8, 5)["fidelity"],
        criterion=0.6, max_epochs=32)
    assert out["reached"], out["history"]
    assert 1 <= out["epochs"] <= 32
    # Monotone enough to bracket: the ladder must not be reading noise.
    assert out["history"][0]["score"] < out["score"]


# ------------------------------------------------------------- interfaces
def test_declares_its_capabilities():
    m = BushSTDPSpikingNetwork(n_neurons=8, seed=0)
    assert m.rollout_mode is RolloutMode.OBSERVATION
    assert supports_online(m) and m.online_equivalent
    assert m.n_features == m.n_neurons
    assert m.named_parameters()["W"].shape == (8, 8)
    assert "g_syn" in m.params and "k_inh" in m.params


def test_event_boundary_clears_state_but_not_weights():
    m = BushSTDPSpikingNetwork(n_neurons=8, seed=0)
    m.fit_event(np.ones((8, 30), dtype=bool))
    W = m.W.copy()
    m.on_event_boundary()
    assert np.array_equal(m.W, W)
    assert np.all(m._state["x_pre"] == 0) and np.all(m._state["v"] == m.c)


def test_recall_rolls_out_in_spike_space():
    X = dual_coded_stimulus(n_fields=4, n_per_field=3, field_steps=20, seed=0)
    m = BushSTDPSpikingNetwork(n_neurons=12, n_presentations=3, recall_steps=9, seed=0)
    m.fit_sequence(X)
    assert m.recall(X[0], length=3).shape == (3, 12, 9)


def test_fit_sequence_accepts_a_continuous_train_as_well_as_stacked_items():
    m = BushSTDPSpikingNetwork(n_neurons=9, n_presentations=2, seed=0)
    m.fit_sequence(np.ones((9, 40), dtype=bool))       # 2D: one continuous train
    assert m.W.max() > 0.0


# ------------------------------------------------------------------- gate
def test_correctness_gate_in_miniature():
    """The paper's result, on three seeds. ``bin/bush_stdp_gate.py`` runs 50.

    Fidelity is scored over the paper's ~33 ms sharp-wave-ripple window, and a
    field that never replays counts as a failure -- so this also pins that the
    whole 20-field sequence fits inside the window.
    """
    fids = []
    for seed in range(3):
        X = dual_coded_stimulus(seed=1000 + seed)
        m = BushSTDPSpikingNetwork(n_neurons=100, recall_steps=40, seed=seed)
        m.fit_sequence(X)
        fids.append(order_fidelity(m.predict_next(X[0]))["fidelity"])
    assert np.mean(fids) >= 0.90, f"order fidelity {np.mean(fids):.3f} < 0.90"


def test_learned_matrix_is_directional():
    """STDP's temporal asymmetry is the only thing producing this, and a
    sequence arm that replays without it would be replaying for the wrong
    reason. (The ``ca3net`` port failed exactly this check.)"""
    X = dual_coded_stimulus(n_fields=10, seed=5)
    m = BushSTDPSpikingNetwork(n_neurons=50, seed=0)
    m.fit_sequence(X)
    W = m.W.reshape(10, 5, 10, 5).mean(axis=(1, 3))
    forward = np.mean([W[k + 1, k] for k in range(9)])
    backward = np.mean([W[k, k + 1] for k in range(9)])
    assert forward > 0.05 and backward < 0.1 * forward


def test_the_literal_reading_is_unstable():
    """The measurement behind the module's inferred-parameter note.

    With recurrent excitation on during encoding and no inhibitory pool -- the
    spec read literally -- STDP pairs the network's own reverberation instead of
    the stimulus, and forward and backward weights come out **equal**: the
    direction is gone. Pinned at two gains, because the saturation signature
    varies across ``g_syn`` while the loss of directionality does not.
    """
    X = dual_coded_stimulus(seed=1000)
    for g_syn in (45.0, 160.0):
        m = BushSTDPSpikingNetwork(n_neurons=100, recall_steps=40, g_syn=g_syn,
                                   k_inh=0.0, ach_recurrent_gain=1.0, seed=0)
        m.fit_sequence(X)
        res = order_fidelity(m.predict_next(X[0]))
        W = m.W.reshape(20, 5, 20, 5).mean(axis=(1, 3))
        forward = np.mean([W[k + 1, k] for k in range(19)])
        backward = np.mean([W[k, k + 1] for k in range(19)])
        assert backward > 0.5 * forward, f"g_syn={g_syn}: direction survived?"
        assert res["fidelity"] < 0.5, f"g_syn={g_syn}: fidelity {res['fidelity']}"


# ------------------------------------------------------------ end-to-end
def test_runs_behind_the_codec_end_to_end():
    """The arm is only scoreable through the codec; this is the seam."""
    from memval.models.codec_wrapper import wrap_with_codec
    from memval.encoders.hierarchical import HierarchicalEncoder

    E = HierarchicalEncoder(branching=(2, 2), features_per_node=2).embeddings
    enc, _ = make_codec(E.shape[1], seed=0, n_per_feature=2, window_steps=20)
    inner = BushSTDPSpikingNetwork(n_neurons=enc.n_neurons, n_presentations=3,
                                   recall_steps=25, seed=0)
    model = wrap_with_codec(inner, enc)
    model.fit_sequence(E)
    out = model.predict_next(E[0])
    assert out.shape == (E.shape[1],)
    assert np.isclose(np.linalg.norm(out), 1.0) or np.allclose(out, 0.0)
