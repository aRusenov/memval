"""Population spike codec: the Phase 0 acceptance criteria of
``docs/spike_codec_spec.md``, as tests.

The spec's Phase 0 is an explicit go/no-go -- if round-trip fidelity has a low
ceiling, every spiking result downstream is capped. These pin the criteria so a
later parameter change cannot quietly lower the ceiling.
``bin/spike_codec_phase0.py`` produces the full characterisation; this file is
the part that must never regress.
"""
import numpy as np
import pytest

from memval.encoders.hierarchical import HierarchicalEncoder
from memval.encoders.symbolic import SymbolicEncoder
from memval.encoders.spike_codec import (
    PopulationSpikeDecoder, PopulationSpikeEncoder, make_codec,
)


def _hier():
    return HierarchicalEncoder(branching=(2, 2, 2), features_per_node=4)


def _sym(dim=100):
    return SymbolicEncoder([f"w{i}" for i in range(16)], embedding_dim=dim, seed=0)


def _cosines(E, seed=0, **kw):
    enc, dec = make_codec(E.shape[1], seed=seed, **kw)
    return np.array([float(dec.decode(enc.encode(v)) @ v) for v in E])


# ------------------------------------------------------------------- shapes
def test_neuron_count_and_shapes():
    enc = PopulationSpikeEncoder(7, n_per_feature=3, window_steps=11, seed=0)
    assert enc.n_neurons == 2 * 7 * 3
    s = enc.encode(np.ones(7) * 0.5)
    assert s.shape == (enc.n_neurons, 11) and s.dtype == bool
    assert PopulationSpikeDecoder(enc).decode(s).shape == (7,)


def test_encode_sequence_and_stack_agree():
    enc, dec = make_codec(6, seed=1, window_steps=8)
    V = np.eye(6)[:3] * 0.5
    enc.reset_rng()
    flat = enc.encode_sequence(V)
    enc.reset_rng()
    stack = enc.encode_stack(V)
    assert flat.shape == (enc.n_neurons, 3 * 8)
    assert np.array_equal(flat.reshape(enc.n_neurons, 3, 8).transpose(1, 0, 2), stack)
    assert dec.decode_sequence(flat).shape == (3, 6)


def test_gap_steps_are_silent_and_decode_realigns():
    """Spec S3.7: inter-item gaps are where Delta t will be injected."""
    enc, dec = make_codec(4, seed=2, window_steps=6)
    V = np.eye(4)[:2] * 0.8
    spikes = enc.encode_sequence(V, gap_steps=[3, 0])
    assert spikes.shape[1] == 2 * 6 + 3
    assert not spikes[:, 6:9].any(), "the gap must be silent"
    out = dec.decode_sequence(spikes, gap_steps=[3, 0])
    assert out.shape == (2, 4)
    assert np.argmax(out[0]) == 0 and np.argmax(out[1]) == 1


def test_wrong_dimension_is_an_error_not_a_silent_broadcast():
    enc, dec = make_codec(5, seed=0)
    with pytest.raises(ValueError):
        enc.encode(np.ones(4))
    with pytest.raises(ValueError):
        dec.decode(np.zeros((3, 10)))


# -------------------------------------------------------- spec constraints
def test_output_is_unit_norm_and_zero_input_gives_zeros():
    """Constraint 2: the decoder's last step is L2-normalisation."""
    enc, dec = make_codec(12, seed=0)
    v = np.zeros(12); v[3] = 1.0
    assert np.isclose(np.linalg.norm(dec.decode(enc.encode(v))), 1.0)
    assert np.allclose(dec.decode(np.zeros((enc.n_neurons, 5), dtype=bool)), 0.0)


def test_sign_is_preserved_by_the_on_off_split():
    """Constraint 3. Spike rates cannot be negative; embeddings are signed.

    The repo has already been bitten by sign handling -- ``divergence_margin``
    was compressed ~350x for negative-output arms -- so this is pinned, not
    assumed.
    """
    enc, dec = make_codec(8, seed=0, n_per_feature=20)
    v = np.array([0.5, -0.5, 0.3, -0.3, 0.0, 0.7, -0.7, 0.1])
    out = dec.decode(enc.encode(v))
    signed = v != 0
    assert np.array_equal(np.sign(out[signed]), np.sign(v[signed]))
    assert float(out @ (v / np.linalg.norm(v))) > 0.95


def test_identical_seed_is_byte_identical_and_global_rng_untouched():
    """Constraint 4, the defect class already found in ``semantic_similarity``."""
    E = _hier().embeddings
    assert np.array_equal(_cosines(E, seed=7), _cosines(E, seed=7))
    assert not np.array_equal(_cosines(E, seed=7), _cosines(E, seed=8))

    before = np.random.get_state()
    _cosines(E, seed=7)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])


def test_reset_rng_rewinds_the_stream():
    enc = PopulationSpikeEncoder(6, seed=3)
    v = np.ones(6) / np.sqrt(6)
    first = enc.encode(v)
    enc.encode(v)
    enc.reset_rng()
    assert np.array_equal(enc.encode(v), first)


def test_explicit_rng_makes_encoding_a_function_of_the_vector():
    """What keeps ``CodecWrappedModel.predict_next`` pure."""
    enc = PopulationSpikeEncoder(6, seed=0)
    v = np.ones(6) / np.sqrt(6)
    a = enc.encode(v, rng=np.random.default_rng(11))
    b = enc.encode(v, rng=np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_codec_params_are_reportable():
    """Constraint 5: the codec is a stated condition, never an invisible default."""
    enc = PopulationSpikeEncoder(4, seed=5)
    for key in ("n_per_feature", "window_steps", "r_max", "mode", "seed"):
        assert key in enc.params
    assert "PopulationSpikeCodec" in enc.describe()


# ------------------------------------------------------ Phase 0 acceptance
def test_hierarchical_clears_the_fidelity_gate_at_defaults():
    """Median round-trip cosine >= 0.95 at the defaults, on the substrate the
    spec names as the reference (S3.1)."""
    med = np.median(np.concatenate([_cosines(_hier().embeddings, seed=s)
                                    for s in range(3)]))
    assert med >= 0.95, f"median round-trip cosine {med:.3f} < 0.95"


def test_dense_symbolic_needs_more_neurons_than_the_default():
    """The measured reason the spec prefers ``HierarchicalEncoder``.

    A dense unit-norm row spreads its norm over every dimension, so each one
    gets a per-step spike probability near the noise floor; a sparse binary row
    concentrates it, and its zero dimensions decode to exactly zero with no
    Poisson variance at all. The gap is not rhetorical -- it decides whether the
    default configuration passes.
    """
    E = _sym().embeddings
    at_default = np.median(np.concatenate([_cosines(E, seed=s) for s in range(3)]))
    at_ten = np.median(np.concatenate([_cosines(E, seed=s, n_per_feature=10)
                                       for s in range(3)]))
    assert at_default < 0.95 < at_ten


@pytest.mark.parametrize("knob,values", [("n_per_feature", (1, 2, 5, 10)),
                                         ("window_steps", (10, 25, 50, 100)),
                                         ("r_max", (50, 100, 200))])
def test_fidelity_is_monotone_in_the_knobs_that_buy_spikes(knob, values):
    """Phase 0 acceptance. All three knobs buy the same thing -- Poisson events
    per dimension -- so fidelity must not fall as any of them rises."""
    E = _hier().embeddings
    meds = [np.median(np.concatenate([_cosines(E, seed=s, **{knob: v})
                                      for s in range(3)])) for v in values]
    assert all(b >= a - 1e-9 for a, b in zip(meds, meds[1:])), dict(zip(values, meds))


def test_corruption_survives_transport():
    """Spec S1: what is at risk under upstream corruption is *sensitivity*.

    The codec must not wash a corruption out before the model sees it. Measured
    as: a noisier upstream cue must stay monotonically worse downstream, and the
    downstream cosine must not be pulled back toward the clean value.
    """
    E = _hier().embeddings
    enc, dec = make_codec(E.shape[1], seed=0)
    rng = np.random.default_rng(4)
    downstream = []
    for sigma in (0.0, 0.05, 0.1, 0.2, 0.4):
        vals = []
        for v in E:
            noisy = v + rng.standard_normal(v.shape) * sigma
            noisy /= np.linalg.norm(noisy)
            vals.append(float(dec.decode(enc.encode(noisy)) @ v))
        downstream.append(np.mean(vals))
    assert all(b < a for a, b in zip(downstream, downstream[1:])), downstream


def test_decoder_scale_is_unbiased_so_the_double_l2_cannot_double_shrink():
    """Spec open question 2. Upstream re-normalises a masked cue and the decoder
    normalises again; that only composes safely if the raw estimator is already
    unbiased in scale."""
    E = _hier().embeddings
    enc, dec = make_codec(E.shape[1], seed=0, n_per_feature=20)
    ratios = []
    for v in E:
        c = dec.counts(enc.encode(v))
        raw = (c[:, 0] - c[:, 1]) / enc.spikes_per_unit
        ratios.append(np.linalg.norm(raw) / np.linalg.norm(v))
    assert 0.95 < np.mean(ratios) < 1.05, np.mean(ratios)


def test_hierarchical_block_asymmetry_survives_transport():
    """Spec open question 1: if the codec flattened the shared/identity
    distinction, the ``cue_masking`` block modes would be N/A for spiking arms.
    They are not."""
    h = _hier()
    E = h.embeddings
    leaf = sorted(c for n in h.nodes if not n["children"] for c in n["columns"])
    shared = sorted(c for n in h.nodes if n["children"] for c in n["columns"])
    enc, dec = make_codec(E.shape[1], seed=0)

    def masked_cosine(cols):
        vals = []
        for v in E:
            m = v.copy()
            m[cols] = 0.0
            if np.linalg.norm(m) == 0:
                continue
            m /= np.linalg.norm(m)
            vals.append((float(m @ v), float(dec.decode(enc.encode(m)) @ v)))
        return np.mean(vals, axis=0)

    up_s, dn_s = masked_cosine(shared)
    up_i, dn_i = masked_cosine(leaf)
    assert abs(dn_s - dn_i) > 0.5 * abs(up_s - up_i)


# ----------------------------------------------------------- latency mode
def test_latency_mode_is_one_spike_per_active_neuron_early_for_strong():
    enc = PopulationSpikeEncoder(4, n_per_feature=2, window_steps=20,
                                 mode="latency", seed=0)
    v = np.array([0.9, 0.1, 0.0, -0.5])
    s = enc.encode(v)
    per_neuron = s.sum(axis=1).reshape(4, 2, 2).sum(axis=2)   # (D, polarity)
    assert per_neuron[2].sum() == 0, "zero magnitude must be silent"
    assert per_neuron[0, 0] == 2 and per_neuron[0, 1] == 0
    assert per_neuron[3, 0] == 0 and per_neuron[3, 1] == 2, "sign selects polarity"
    t_strong = np.nonzero(s[0 * 4 + 0])[0].min()
    t_weak = np.nonzero(s[1 * 4 + 0])[0].min()
    assert t_strong < t_weak, "strong feature must spike earlier"


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError):
        PopulationSpikeEncoder(4, mode="phase", seed=0)
