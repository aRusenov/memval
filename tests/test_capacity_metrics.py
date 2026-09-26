import numpy as np
import pytest

from memval.encoders.symbolic import SymbolicDecoder, SymbolicEncoder
from memval.metrics.capacity import (
    attractor_scan,
    category_information,
    channel_mi,
    crosstalk_matrix,
    diagonal_dominance,
    ep_update_signature,
    recall_confusion,
    synaptic_interference_matrix,
)
from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork

DIM = 16


@pytest.fixture(scope="module")
def world():
    """One tiny, well-separated (cross-category) sequence trained to recall."""
    vocab = {"apple": "fruit", "cat": "animal", "car": "vehicle", "red": "color"}
    encoder = SymbolicEncoder(vocab, embedding_dim=DIM, category_variance=0.2, seed=1)
    decoder = SymbolicDecoder(encoder)
    seq = ["apple", "cat", "car", "red"]
    model = OriginalEqPropSequenceNetwork(
        n_features=DIM, n_hidden=32, learning_rate=0.1, beta=0.5,
        n_settle_steps=15, n_epochs=150, seed=1,
    )
    model.fit_sequence(encoder.encode(seq))
    transitions = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    return dict(vocab=vocab, encoder=encoder, decoder=decoder, model=model,
                seq=seq, transitions=transitions)


# ---------------------------------------------------------------- channel_mi
def test_channel_mi_identity_equals_log2_n():
    counts = np.eye(4) * 10.0            # perfect, deterministic channel
    out = channel_mi(counts)
    assert out["mi_bits"] == pytest.approx(2.0, abs=1e-9)   # log2(4)
    assert out["efficiency"] == pytest.approx(1.0, abs=1e-9)


def test_channel_mi_independent_is_zero():
    counts = np.ones((4, 4))            # X and Y independent
    assert channel_mi(counts)["mi_bits"] == pytest.approx(0.0, abs=1e-9)


def test_channel_mi_empty_is_safe():
    assert channel_mi(np.zeros((3, 3)))["mi_bits"] == 0.0


# ---------------------------------------------------------------- crosstalk
def test_crosstalk_diagonal_dominance_positive(world):
    cues = [c for c, _ in world["transitions"]]
    tgts = [t for _, t in world["transitions"]]
    S = crosstalk_matrix(world["model"], cues, tgts, world["encoder"])
    assert S.shape == (len(cues), len(tgts))
    # a trained, well-separated chain should prefer its own target
    assert diagonal_dominance(S) > 0.0


# ------------------------------------------------------- synaptic interference
def test_synaptic_interference_symmetric_unit_diag(world):
    J = synaptic_interference_matrix(world["model"], world["transitions"], world["encoder"])
    n = len(world["transitions"])
    assert J.shape == (n, n)
    assert np.allclose(np.diag(J), 1.0, atol=1e-6)
    assert np.allclose(J, J.T, atol=1e-6)
    assert J.min() >= -1.0 - 1e-6 and J.max() <= 1.0 + 1e-6


def test_ep_update_signature_shape_and_finite(world):
    m, enc = world["model"], world["encoder"]
    g = ep_update_signature(m, enc.encode(["apple"])[0], enc.encode(["cat"])[0])
    assert g.shape == (m.n_hidden * m.n_features + m.n_features * m.n_hidden,)
    assert np.all(np.isfinite(g))


def test_ep_signature_does_not_mutate_weights(world):
    m, enc = world["model"], world["encoder"]
    before = m.get_latent_state()
    ep_update_signature(m, enc.encode(["apple"])[0], enc.encode(["cat"])[0])
    after = m.get_latent_state()
    for k in before:
        assert np.array_equal(before[k], after[k]), f"{k} was mutated"


# ---------------------------------------------------------------- confusion/MI
def test_recall_confusion_and_category_info(world):
    rng = np.random.default_rng(0)
    conf = recall_confusion(world["model"], world["transitions"], world["encoder"],
                            world["decoder"], noise_scale=0.0, n_trials=5, rng=rng)
    counts = conf["counts"]
    assert counts.sum() == 5 * len(world["transitions"])
    info = category_information(conf, world["vocab"])
    for key in ("acc_item", "acc_category"):
        assert 0.0 <= info[key] <= 1.0
    assert info["mi_item_bits"] >= info["mi_within_category_bits"] - 1e-9


# ---------------------------------------------------------------- attractor scan
def test_attractor_scan_contract(world):
    rng = np.random.default_rng(0)
    seeds = np.vstack([
        world["encoder"].encode([w])[0] + rng.normal(0, 0.2, DIM)
        for w in world["seq"] for _ in range(3)
    ])
    scan = attractor_scan(world["model"], seeds, world["decoder"], world["seq"],
                          max_steps=25)
    assert len(scan["records"]) == len(seeds)
    assert 0.0 <= scan["spurious_rate"] <= 1.0
    assert 0.0 <= scan["dominant_basin_frac"] <= 1.0
    assert sum(scan["basin_counts"].values()) <= len(seeds)
    for rec in scan["records"]:
        assert rec["label"] in ("genuine", "spurious")
        assert rec["norm_trace"].ndim == 1
