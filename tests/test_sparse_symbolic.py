"""The sparse k-of-d symbolic encoder and its overlap laws.

Dense ``SymbolicEncoder`` controls overlap by cosine of Gaussian codes; this one
controls it by how many active units two items SHARE. The laws it must obey:
within-category ``category_core + (1-core)^2 k/d``, between-category ``k/d``.
"""
import numpy as np
import pytest

from memval.encoders import SparseSymbolicEncoder, SymbolicDecoder

D, K = 512, 10


def _cat_vocab(n_cats=10, per_cat=10):
    return {f"c{c}_i{i}": f"c{c}" for c in range(n_cats) for i in range(per_cat)}


def test_every_code_has_exactly_k_active_units_and_unit_norm():
    enc = SparseSymbolicEncoder(_cat_vocab(), D, K, category_core=0.5, seed=1)
    support = (enc.embeddings > 0).sum(axis=1)
    assert set(support.tolist()) == {K}
    assert np.allclose(np.linalg.norm(enc.embeddings, axis=1), 1.0)


@pytest.mark.parametrize("core", [0.0, 0.3, 0.5, 0.8])
def test_overlap_laws(core):
    vocab = _cat_vocab()
    g = SparseSymbolicEncoder(vocab, D, K, category_core=core, seed=2).geometry()
    assert g["within_cos_measured"] == pytest.approx(g["within_cos_law"], abs=0.02)
    assert g["between_cos_measured"] == pytest.approx(K / D, abs=0.02)


def test_category_core_is_shared_exactly_and_the_rest_is_private():
    enc = SparseSymbolicEncoder(_cat_vocab(3, 4), D, K, category_core=0.5, seed=3)
    for cat in enc.categories:
        words = [w for w, c in enc.vocab.items() if c == cat]
        cores = [set(enc.category_units[cat].tolist())] * len(words)
        for w, core in zip(words, cores):
            assert core <= set(enc.word_units[w].tolist())
        shared = set.intersection(*(set(enc.word_units[w].tolist()) for w in words))
        assert shared == set(enc.category_units[cat].tolist())


def test_no_category_structure_gives_independent_codes():
    enc = SparseSymbolicEncoder([f"w{i}" for i in range(100)], D, K, seed=4)
    g = enc.geometry()
    assert g["n_core_units"] == 0
    assert g["between_cos_measured"] == pytest.approx(K / D, abs=0.02)


def test_decoder_recovers_every_item_from_its_own_code():
    enc = SparseSymbolicEncoder(_cat_vocab(), D, K, category_core=0.5, seed=5)
    dec = SymbolicDecoder(enc)
    words = list(enc.idx_to_word)
    assert dec.decode(enc.encode(words)) == words


def test_seeded_construction_is_reproducible():
    a = SparseSymbolicEncoder(_cat_vocab(), D, K, category_core=0.5, seed=6)
    b = SparseSymbolicEncoder(_cat_vocab(), D, K, category_core=0.5, seed=6)
    assert np.array_equal(a.embeddings, b.embeddings)


def test_rejects_impossible_configurations():
    with pytest.raises(ValueError):
        SparseSymbolicEncoder(_cat_vocab(), embedding_dim=8, active=20)
    with pytest.raises(ValueError):
        SparseSymbolicEncoder(_cat_vocab(), D, K, category_core=1.5)
    with pytest.raises(ValueError):
        SparseSymbolicEncoder([f"w{i}" for i in range(5)], D, K, category_core=0.5)


def test_unknown_word_is_refused():
    enc = SparseSymbolicEncoder([f"w{i}" for i in range(5)], D, K, seed=7)
    with pytest.raises(ValueError):
        enc.encode(["nope"])
