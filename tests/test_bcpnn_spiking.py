"""Hold :mod:`memval.models.baselines.bcpnn_spiking` to its reference and its contract.

The learning rule has a reference implementation -- the authors' NEST 2.2
synapse ``bcpnn_connection.h`` (Tully & Kaplan; ``Florian-Fiebig/BCPNN-for-NEST222-MPI``)
-- and the network does not. So fidelity is established in two layers:

1. :func:`reference_module_traces` is a **line-by-line transliteration of the
   module's ``send()``**: event-driven at presynaptic spikes, replaying the
   postsynaptic history since the last one, with the module's spike-height
   normalisation, ``+epsilon`` term, three trace levels (Z, E, P) and the
   print-now vector. :func:`stepwise_traces` steps the same equations on a
   fixed grid, and the two must agree at every presynaptic spike. That pins
   the *arithmetic* of the rule to the reference, including its quirks.
2. The arm's vectorised trace engine (per-neuron Z and P, lazily accumulated
   ``P_ij``) must equal :func:`stepwise_traces` with the E level disabled --
   the paper's two-level form, which the arm implements -- for every synapse
   of a small network driven by arbitrary spike counts.

Everything else is the harness contract every arm carries, plus the paper's
own result in miniature.
"""
import numpy as np
import pytest

from memval.encoders.spike_codec import PopulationSpikeEncoder
from memval.models.baselines.bcpnn_spiking import (
    PAPER, BCPNNSpikingNetwork, CodecBCPNNNetwork, _paper_mean_delay_ms,
)
from memval.models.capabilities import is_online_equivalent, supports_online


# --------------------------------------------------------------- the oracles
def reference_module_traces(pre_times, post_times, tau_i, tau_j, tau_e, tau_p,
                            fmax, eps, K=1.0, res=1.0, gain=1.0):
    """``bcpnn_connection.h::send`` transliterated (dendritic delay 0, no STP).

    Returns one record per presynaptic spike: the state right after ``send``.
    The module's quirks are kept on purpose: the presynaptic spike is applied
    at time step 0 of the *next* interval (so a spike at t = 0 is never
    applied), post spikes at exactly the current pre-spike time fall outside
    the loop and are dropped, and traces only advance inside ``send``.
    """
    spike_width = int(1.0 / res)
    spike_height = 1000.0 / fmax
    zi = zj = 0.01
    ei = ej = 0.01
    pi = pj = 0.01
    eij = ei * ej
    pij = pi * pj
    t_last = 0.0
    out = []
    for t_spike in pre_times:
        posts = [tp for tp in post_times if t_last < tp <= t_spike]
        n_iter = int((t_spike - t_last) / res)
        post_it = 0
        for step in range(n_iter):
            yi = yj = 0.0
            if step == 0 and t_last != 0.0:
                yi = spike_height * spike_width
            if post_it < len(posts) and step == int((posts[post_it] - t_last)) / res:
                yj = spike_height * spike_width
                post_it += 1
            zi += (yi - zi + eps) * res / tau_i
            zj += (yj - zj + eps) * res / tau_j
            ei += (zi - ei) * res / tau_e
            ej += (zj - ej) * res / tau_e
            eij += (zi * zj - eij) * res / tau_e
            pi += K * (ei - pi) * res / tau_p
            pj += K * (ej - pj) * res / tau_p
            pij += K * (eij - pij) * res / tau_p
        t_last = t_spike
        out.append(dict(t=t_spike, zi=zi, zj=zj, pi=pi, pj=pj, pij=pij,
                        w=gain * np.log(pij / (pi * pj)), bias=np.log(pj)))
    return out


def stepwise_traces(pre, post, tau_i, tau_j, tau_p, fmax, eps, dt=1.0,
                    tau_e=None, K=1.0, init=0.01):
    """Eqs 2-3 on a fixed grid; ``tau_e`` inserts the module's E level."""
    kick = 1.0 / (fmax * dt * 1e-3)
    zi = zj = ei = ej = pi = pj = init
    eij = pij = init * init
    hist = []
    for t in range(len(pre)):
        zi += (kick * pre[t] - zi + eps) * dt / tau_i
        zj += (kick * post[t] - zj + eps) * dt / tau_j
        if tau_e is None:
            pi += K * (zi - pi) * dt / tau_p
            pj += K * (zj - pj) * dt / tau_p
            pij += K * (zi * zj - pij) * dt / tau_p
        else:
            ei += (zi - ei) * dt / tau_e
            ej += (zj - ej) * dt / tau_e
            eij += (zi * zj - eij) * dt / tau_e
            pi += K * (ei - pi) * dt / tau_p
            pj += K * (ej - pj) * dt / tau_p
            pij += K * (eij - pij) * dt / tau_p
        hist.append((zi, zj, pi, pj, pij))
    return hist


def _trains(seed, T=400, r_pre=0.05, r_post=0.04):
    """Bernoulli trains with no spike at t = 0 and no coincident pre/post."""
    rng = np.random.default_rng(seed)
    pre = (rng.random(T) < r_pre).astype(float)
    post = (rng.random(T) < r_post).astype(float)
    pre[0] = post[0] = 0.0
    post[pre > 0] = 0.0
    return pre, post


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stepwise_engine_matches_the_reference_module(seed):
    """Layer 1: fixed-grid stepping == the module's event-driven replay."""
    pre, post = _trains(seed)
    tau_i, tau_j, tau_e, tau_p, fmax, eps = 10.0, 10.0, 100.0, 1000.0, 50.0, 0.01
    ref = reference_module_traces(np.flatnonzero(pre).tolist(),
                                  np.flatnonzero(post).tolist(),
                                  tau_i, tau_j, tau_e, tau_p, fmax, eps)
    hist = stepwise_traces(pre, post, tau_i, tau_j, tau_p, fmax, eps, tau_e=tau_e)
    assert len(ref) > 10
    for rec in ref:
        # After send(t_k) the module has integrated times 0 .. t_k - 1.
        zi, zj, pi, pj, pij = hist[int(rec["t"]) - 1]
        assert np.isclose(rec["zi"], zi, rtol=1e-12, atol=1e-15)
        assert np.isclose(rec["zj"], zj, rtol=1e-12, atol=1e-15)
        assert np.isclose(rec["pi"], pi, rtol=1e-12, atol=1e-15)
        assert np.isclose(rec["pj"], pj, rtol=1e-12, atol=1e-15)
        assert np.isclose(rec["pij"], pij, rtol=1e-12, atol=1e-15)
        assert np.isclose(rec["w"], np.log(pij / (pi * pj)))


def test_the_reference_transliteration_is_not_vacuous():
    """The module's traces must actually move: a spiked synapse potentiates."""
    pre = np.zeros(300); post = np.zeros(300)
    pre[[20, 60, 100, 140, 180, 220, 260]] = 1
    post[[22, 62, 102, 142, 182, 222, 262]] = 1
    ref = reference_module_traces(np.flatnonzero(pre).tolist(), np.flatnonzero(post).tolist(),
                                  10.0, 10.0, 100.0, 1000.0, 50.0, 0.01)
    assert ref[-1]["w"] > 0.5 and ref[-1]["pij"] > ref[-1]["pi"] * ref[-1]["pj"]


def _small(**kw):
    kw.setdefault("seed", 0)
    return BCPNNSpikingNetwork(n_hc=3, n_mc=2, n_per_mc=4, **kw)


@pytest.mark.parametrize("seed", [0, 3])
def test_arm_traces_equal_the_paper_form_synapse_by_synapse(seed):
    """Layer 2: the vectorised engine == stepwise Eqs 2-3 without the E level.

    Drives ``_update_traces`` directly with random spike counts (bypassing the
    membranes) and checks every AMPA and NMDA synapse, the per-neuron P traces
    and the lazily accumulated ``P_ij`` against the scalar oracle.
    """
    m = _small(seed=seed)
    p = m.p
    rng = np.random.default_rng(seed)
    T = 120
    counts = (rng.random((T, m.n_neurons)) < 0.05).astype(float)
    counts[rng.random((T, m.n_neurons)) < 0.01] = 2.0          # a doublet now and then
    for t in range(T):
        m._update_traces(counts[t])
        if t % 37 == 0:
            m._flush()                                            # chunked flushes must not matter
    m._flush()
    for i in range(m.n_neurons):
        for j in range(m.n_neurons):
            ha = stepwise_traces(counts[:, i], counts[:, j], p["tau_zi_ampa"], p["tau_zj"],
                                 p["tau_p"], p["f_max"], p["eps"], init=p["eps"])[-1]
            hn = stepwise_traces(counts[:, i], counts[:, j], p["tau_zi_nmda"], p["tau_zj"],
                                 p["tau_p"], p["f_max"], p["eps"], init=p["eps"])[-1]
            assert np.isclose(m.P_ij_ampa[i, j], ha[4], rtol=1e-10, atol=1e-14)
            assert np.isclose(m.P_ij_nmda[i, j], hn[4], rtol=1e-10, atol=1e-14)
            assert np.isclose(m.P_i_ampa[i], ha[2], rtol=1e-10)
            assert np.isclose(m.P_i_nmda[i], hn[2], rtol=1e-10)
            assert np.isclose(m.P_j[j], ha[3], rtol=1e-10)
            assert np.isclose(m.Z_i_ampa[i], ha[0], rtol=1e-10)


def test_lazy_accumulation_equals_stepping():
    """One matrix product per item is the same as T outer products."""
    m = _small()
    rng = np.random.default_rng(5)
    a = m.dt_ms / m.p["tau_p"]
    P = m.P_ij_ampa.copy()
    for _ in range(90):
        c = (rng.random(m.n_neurons) < 0.1).astype(float)
        m._update_traces(c)
        P += a * (np.outer(m.Z_i_ampa, m.Z_j) - P)
    m._flush()
    assert np.allclose(m.P_ij_ampa, P, rtol=1e-12, atol=1e-16)


def test_weights_are_the_log_ratio_with_the_calibrated_gain_on_the_mask():
    m = _small()
    rng = np.random.default_rng(1)
    for _ in range(50):
        m._update_traces((rng.random(m.n_neurons) < 0.2).astype(float))
    w = m.weights()
    expect = (m.p["gain_ampa"] * m.p["ampa_calib"] * m.gain_scale
              * np.log(m.P_ij_ampa / (m.P_i_ampa[:, None] * m.P_j[None, :])))
    assert np.allclose(w["ampa"], expect * m.mask)
    assert not w["ampa"][~m.mask].any() and not np.diag(w["ampa"]).any()
    assert np.allclose(w["I_beta"], m.p["beta_gain"] * np.log(m.P_j))


def test_size_normalisation_is_the_identity_at_the_papers_geometry():
    m = BCPNNSpikingNetwork(9, 10, 30, seed=0)
    assert m.gain_scale == 1.0 and m.n_basket == 270 and m.n_neurons == 2700
    nz = m.W_pb[m.W_pb > 0]
    assert abs(nz.mean() - PAPER["w_pb"]) < 0.1 * PAPER["w_pb"]
    nz = m.W_bp[m.W_bp > 0]
    assert abs(nz.mean() - PAPER["w_bp"]) < 0.1 * PAPER["w_bp"]
    assert abs(_paper_mean_delay_ms() - 6.45) < 0.05        # 0.75 mm * 1.453 grid units / 0.2 mm/ms + 1


def _pattern(m, cells, T=100, rate=200.0, seed=0):
    rng = np.random.default_rng(seed)
    S = np.zeros((m.n_neurons, T))
    S[cells] = rng.random((len(cells), T)) < rate * 1e-3
    return S


def test_stimulated_cells_fire_near_f_max_and_the_rest_are_silent():
    m = BCPNNSpikingNetwork(4, 5, 12, seed=0)
    cells = np.concatenate([np.arange(h * 60, h * 60 + 12) for h in range(4)])
    S = _pattern(m, cells, T=500, seed=1)
    out = m._run(S, 500, plastic=False, recurrent=False, rng=m.rng, r_bg=0.0)
    rate = out[cells].sum() / (len(cells) * 0.5)
    others = np.setdiff1d(np.arange(m.n_neurons), cells)
    assert 12.0 < rate < 30.0, rate
    assert out[others].sum() == 0


def test_training_regime_is_feedforward_and_recall_regime_is_recurrent():
    """Paper: gains off while kappa is on, and the reverse."""
    m = BCPNNSpikingNetwork(4, 5, 12, seed=0)
    cells = np.arange(0, 12)
    m.fit_event(_pattern(m, cells, T=200))
    assert np.all(m._state["x"] == 1.0)                     # no recurrent release
    assert not m._state["gN_e"].any()
    P0 = m.P_ij_ampa.copy()
    m.predict_next(_pattern(m, cells, T=50))
    assert np.array_equal(m.P_ij_ampa, P0)                  # kappa = 0


def _two_item_net(overrides=None, epochs=15):
    m = BCPNNSpikingNetwork(4, 5, 12, seed=0, overrides=overrides)
    A = np.concatenate([np.arange(h * 60, h * 60 + 12) for h in range(4)])
    B = A + 12
    for k in range(epochs):
        m.on_event_boundary()
        m.fit_event(_pattern(m, A, seed=2 * k))
        m.fit_event(_pattern(m, B, seed=2 * k + 1))
        m.on_event_boundary()
    w = m.weights()["nmda"]
    fwd = w[np.ix_(A, B)][m.mask[np.ix_(A, B)]].mean()
    bwd = w[np.ix_(B, A)][m.mask[np.ix_(B, A)]].mean()
    return fwd, bwd


def test_nmda_weights_are_forward_shifted_and_swapping_the_taus_reverses_them():
    """Paper Fig 4B/G: tau_zi > tau_zj gives A -> B; swapping gives B -> A."""
    fwd, bwd = _two_item_net()
    assert fwd > 0 > bwd, (fwd, bwd)
    fwd_r, bwd_r = _two_item_net(overrides=dict(tau_zi_nmda=5.0, tau_zj=150.0))
    assert bwd_r > fwd_r, (fwd_r, bwd_r)


def test_correctness_gate_in_miniature():
    """The paper's result at 4 x 5 x 12: a cue replays its successors in order.

    Five orthogonal patterns, IPI = 0, cued replay of pattern 0; the
    attractor sequence read off 25 ms pattern rates must advance forward.
    ``bin/bcpnn_gate.py`` runs the full 9 x 10 x 30 with the paper's own
    attractor detector and CRP.
    """
    m = BCPNNSpikingNetwork(4, 5, 12, seed=0, n_epochs=10)
    pats = [np.concatenate([np.arange(h * 60 + k * 12, h * 60 + (k + 1) * 12) for h in range(4)])
            for k in range(5)]
    for ep in range(10):
        m.on_event_boundary()
        for k in range(5):
            m.fit_event(_pattern(m, pats[k], seed=100 * ep + k))
        m.on_event_boundary()
    r = m.replay(_pattern(m, pats[0], seed=7), 800, r_bg=150.0)
    rates = np.stack([r[c].reshape(len(c), 32, 25).sum(axis=(0, 2)) / (len(c) * 0.025)
                      for c in pats])                                        # (5, bins)
    seq = []
    for b in range(32):
        if rates[:, b].max() > 10.0:
            k = int(rates[:, b].argmax())
            if not seq or seq[-1] != k:
                seq.append(k)
    assert seq[:3] == [0, 1, 2], seq


def test_predict_next_is_pure_and_does_not_learn():
    m = BCPNNSpikingNetwork(4, 5, 12, seed=0)
    A = np.arange(0, 12)
    m.fit_event(_pattern(m, A))
    saved = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m._state.items()}
    Z = (m.Z_i_ampa.copy(), m.Z_i_nmda.copy(), m.Z_j.copy())
    P = m.get_latent_state()
    cue = _pattern(m, A, T=50, seed=9)
    a = m.predict_next(cue)
    b = m.predict_next(cue)
    assert a.dtype == bool and a.shape == (m.n_neurons, m.recall_steps)
    assert np.array_equal(a, b)
    for k, v in saved.items():
        assert np.array_equal(m._state[k], v) if isinstance(v, np.ndarray) else m._state[k] == v
    assert all(np.array_equal(x, y) for x, y in zip(Z, (m.Z_i_ampa, m.Z_i_nmda, m.Z_j)))
    assert all(np.array_equal(P[k], v) for k, v in m.get_latent_state().items())


def test_identical_seed_is_byte_identical_and_global_rng_untouched():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    outs = []
    for _ in range(2):
        m = BCPNNSpikingNetwork(3, 2, 4, seed=11)
        m.fit_sequence(np.stack([_pattern(m, np.arange(0, 4), T=30, seed=1),
                                 _pattern(m, np.arange(4, 8), T=30, seed=2)]), epochs=2)
        outs.append(m.get_latent_state()["P_ij_nmda"])
    assert np.array_equal(outs[0], outs[1])
    assert np.array_equal(np.random.get_state()[1], before)


def test_fit_sequence_is_a_fit_event_loop_with_fenced_passes():
    items = None
    def run(fn):
        m = BCPNNSpikingNetwork(3, 2, 4, seed=4)
        nonlocal items
        items = np.stack([_pattern(m, np.arange(0, 4), T=30, seed=1),
                          _pattern(m, np.arange(4, 8), T=30, seed=2)])
        fn(m)
        return m.get_latent_state()
    a = run(lambda m: m.fit_sequence(items, epochs=3))
    def loop(m):
        for _ in range(3):
            m.on_event_boundary()
            for it in items:
                m.fit_event(it)
            m.on_event_boundary()
    b = run(loop)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_epochs_kwarg_is_honoured():
    def build(n):
        m = BCPNNSpikingNetwork(3, 2, 4, seed=4, n_epochs=n)
        return m, np.stack([_pattern(m, np.arange(0, 4), T=30, seed=1),
                            _pattern(m, np.arange(4, 8), T=30, seed=2)])
    one, x = build(100); one.fit_sequence(x, epochs=1)
    three, x = build(100); three.fit_sequence(x, epochs=3)
    pinned, x = build(3); pinned.fit_sequence(x, epochs=3)
    assert not np.allclose(one.P_ij_ampa, three.P_ij_ampa)
    assert np.array_equal(three.P_ij_ampa, pinned.P_ij_ampa)


def test_boundary_clears_transients_but_not_the_memory():
    m = BCPNNSpikingNetwork(3, 2, 4, seed=0)
    m.fit_event(_pattern(m, np.arange(0, 4), T=80))
    P = m.P_ij_nmda.copy()
    assert m.Z_i_nmda.max() > m.p["eps"] * 2
    m.on_event_boundary()
    assert np.all(m.Z_i_nmda == m.p["eps"]) and np.all(m.Z_j == m.p["eps"])
    assert not m._state["gA_e"].any() and m._state["t"] == 0
    assert np.array_equal(m.P_ij_nmda, P)


def test_declares_its_capabilities():
    m = BCPNNSpikingNetwork(3, 2, 4, seed=0)
    assert supports_online(m) and is_online_equivalent(m)
    assert m.prompt_conditioned is False


def test_rejects_vectors_with_a_pointer_to_the_codec():
    m = BCPNNSpikingNetwork(3, 2, 4, seed=0)
    with pytest.raises(ValueError, match="wrap_with_codec"):
        m.fit_event(np.ones(7))


def test_unknown_override_is_rejected():
    with pytest.raises(KeyError):
        BCPNNSpikingNetwork(3, 2, 4, overrides={"g_leak": 1.0})


def test_runs_behind_the_codec_end_to_end():
    D = 6
    m = CodecBCPNNNetwork(n_features=D, n_per_feature=2, window_steps=20,
                          n_epochs=2, recall_steps=30, seed=0)
    assert m.inner.n_neurons == 2 * D * 2 and m.inner.n_hc == D and m.inner.n_mc == 2
    X = np.random.default_rng(0).normal(size=(3, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    m.fit_sequence(X, epochs=2)
    y = m.predict_next(X[0])
    assert y.shape == (D,)
    assert np.array_equal(y, m.predict_next(X[0]))
    assert supports_online(m)
