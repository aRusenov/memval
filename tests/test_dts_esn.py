import numpy as np
import pytest

from memval.encoders.symbolic import SymbolicDecoder, SymbolicEncoder
from memval.models.baselines import DTSESNSequenceNetwork


def _fast_model(n_features=32, seed=0):
    """Small, coarse-stepped config so tests stay quick."""
    return DTSESNSequenceNetwork(
        n_features=n_features,
        n_units=150,
        tau_min=0.2,
        tau_max=8.0,
        dt=0.1,
        seed=seed,
    )


def test_rejects_unstable_timestep():
    with pytest.raises(ValueError, match="tau_min"):
        DTSESNSequenceNetwork(n_features=8, dt=1.0, tau_min=0.5)


def test_leak_rates_are_log_spaced_over_the_tau_range():
    m = _fast_model()
    assert m.taus[0] == pytest.approx(m.tau_min)
    assert m.taus[-1] == pytest.approx(m.tau_max)
    # log-spaced => constant ratio between consecutive time constants
    ratios = m.taus[1:] / m.taus[:-1]
    assert np.allclose(ratios, ratios[0])


def test_predict_next_is_pure_and_does_not_leak_stream_state():
    """The independent-probe contract: probing must not mutate the streaming
    reservoir, and repeated probes must be identical."""
    m = _fast_model()
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((5, 32))
    m.fit_sequence(seq, intervals=[0.0, 1.0, 1.0, 1.0, 1.0])

    before = m.get_latent_state()["x"].copy()
    first = m.predict_next(seq[0], elapsed=1.0)
    after = m.get_latent_state()["x"]
    second = m.predict_next(seq[0], elapsed=1.0)

    np.testing.assert_allclose(before, after)
    np.testing.assert_allclose(first, second)


def test_elapsed_time_changes_the_prediction():
    """The whole point of the arm: same cue, different gap, different state."""
    m = _fast_model()
    rng = np.random.default_rng(1)
    seq = rng.standard_normal((4, 32))
    m.fit_sequence(seq, intervals=[0.0, 0.5, 0.5, 0.5])

    short = m.predict_next(seq[0], elapsed=0.5)
    long = m.predict_next(seq[0], elapsed=6.0)
    assert not np.allclose(short, long)


def test_works_without_intervals_for_ordinal_benchmarks():
    """Drop-in compatibility: benchmarks with no time axis must still run."""
    m = _fast_model()
    rng = np.random.default_rng(2)
    seq = rng.standard_normal((4, 32))
    m.fit_sequence(seq)
    pred = m.predict_next(seq[0])
    assert pred.shape == (32,)
    assert np.all(np.isfinite(pred))
    rolled = m.recall(seq[0], length=3)
    assert rolled.shape == (3, 32)


def test_reset_context_clears_state_but_keeps_readout():
    m = _fast_model()
    rng = np.random.default_rng(3)
    seq = rng.standard_normal((4, 32))
    m.fit_sequence(seq, intervals=[0.0, 1.0, 1.0, 1.0])

    trained = m.W_out.copy()
    m.reset_context()
    np.testing.assert_allclose(m.get_latent_state()["x"], np.zeros(m.n_units))
    np.testing.assert_allclose(m.W_out, trained)


def test_interval_disambiguates_two_word_continuations():
    """End-to-end: one shared prefix, two continuations separated only by how
    much time elapsed. An ordinal-clocked model cannot pass this."""
    prefix = ["red", "green", "blue"]
    cont_long, cont_short = ["dog"], ["one"]
    vocab = prefix + cont_long + cont_short

    encoder = SymbolicEncoder(vocab, embedding_dim=32, seed=0)
    decoder = SymbolicDecoder(encoder)
    m = _fast_model(n_features=32)

    spacing, long_gap, short_gap = 0.5, 5.0, 0.5
    ev_long = encoder.encode(prefix + cont_long)
    ev_short = encoder.encode(prefix + cont_short)
    iv_long = [0.0, spacing, spacing, long_gap]
    iv_short = [0.0, spacing, spacing, short_gap]

    for _ in range(30):
        m.fit_sequence(ev_long, intervals=iv_long)
        m.fit_sequence(ev_short, intervals=iv_short)

    prompt = encoder.encode(prefix)
    prompt_iv = [0.0, spacing, spacing]

    word_long, _ = decoder.decode_with_score(
        m.predict_next(prompt, prompt_intervals=prompt_iv, elapsed=long_gap)
    )
    word_short, _ = decoder.decode_with_score(
        m.predict_next(prompt, prompt_intervals=prompt_iv, elapsed=short_gap)
    )

    assert word_long == "dog"
    assert word_short == "one"


# ----------------------------------------------------------------------
# Timing head (predict_timing=True)
# ----------------------------------------------------------------------
def _timing_model(n_features=32, seed=0):
    return DTSESNSequenceNetwork(
        n_features=n_features, n_units=150, tau_min=0.2, tau_max=8.0,
        dt=0.1, predict_timing=True, seed=seed,
    )


def test_generation_requires_the_timing_head():
    m = _fast_model()
    with pytest.raises(RuntimeError, match="predict_timing"):
        m.generate(np.zeros(32), length=2)
    with pytest.raises(RuntimeError, match="predict_timing"):
        m.predict_time_to_next(np.zeros(32))


def test_timing_head_does_not_perturb_item_predictions():
    """Regression guard: adding the head must leave the item path identical."""
    rng = np.random.default_rng(7)
    seq = rng.standard_normal((5, 32))
    iv = [0.0, 0.5, 0.5, 0.5, 0.5]

    plain, timed = _fast_model(seed=4), _timing_model(seed=4)
    for m in (plain, timed):
        for _ in range(5):
            m.fit_sequence(seq, intervals=iv)

    np.testing.assert_allclose(
        plain.predict_next(seq[0], elapsed=0.5),
        timed.predict_next(seq[0], elapsed=0.5),
    )


def test_timing_head_learns_a_constant_tempo():
    m = _timing_model()
    rng = np.random.default_rng(8)
    seq = rng.standard_normal((5, 32))
    tempo = 2.0
    for _ in range(30):
        m.fit_sequence(seq, intervals=[0.0] + [tempo] * 4)

    predicted = m.predict_time_to_next(seq[:2], prompt_intervals=[0.0, tempo])
    assert predicted == pytest.approx(tempo, abs=0.3)


def test_generate_returns_events_and_intervals():
    m = _timing_model()
    rng = np.random.default_rng(9)
    seq = rng.standard_normal((4, 32))
    for _ in range(20):
        m.fit_sequence(seq, intervals=[0.0, 1.0, 1.0, 1.0])

    events, intervals = m.generate(seq[0], length=3)
    assert events.shape == (3, 32)
    assert intervals.shape == (3,)
    assert np.all(intervals >= m.dt)      # floored so a rollout always advances
    assert np.all(np.isfinite(events))


def test_generated_rhythm_places_the_pause_correctly():
    """A sequence with one internal pause should be regenerated with the pause
    in the same position."""
    m = _timing_model()
    rng = np.random.default_rng(10)
    seq = rng.standard_normal((5, 32))
    gaps = [0.0, 0.4, 3.0, 0.4, 0.4]          # pause before item 2
    for _ in range(30):
        m.fit_sequence(seq, intervals=gaps)

    _, generated = m.generate(seq[0], length=4)
    assert int(np.argmax(generated)) == int(np.argmax(gaps[1:]))
