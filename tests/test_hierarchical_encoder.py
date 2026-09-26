"""Structural tests for HierarchicalEncoder.

The load-bearing claim is that the branch rule makes the SVD of the item x
feature matrix *be* the hierarchy -- one singular dimension per categorical
split, strength ordered by level. Everything the schema benchmark reports rests
on that, so it is asserted here rather than assumed.
"""
import numpy as np
import pytest

from memval.encoders.hierarchical import HierarchicalEncoder
from memval.encoders.symbolic import SymbolicDecoder


PAPER_TREE = {
    "animal": {"bird": ["sparrow", "hawk"], "fish": ["salmon", "sunfish"]},
    "plant": {"tree": ["oak", "maple"], "flower": ["rose", "daisy"]},
}


@pytest.fixture
def enc():
    return HierarchicalEncoder(tree=PAPER_TREE, features_per_node=2, seed=0)


def test_branch_rule_holds(enc):
    """No feature may appear in leaves of two different branches."""
    M = enc.feature_matrix
    animals = set(np.flatnonzero(enc.node_indicator("animal")).tolist())
    plants = set(np.flatnonzero(enc.node_indicator("plant")).tolist())
    for col in range(M.shape[1]):
        active = set(np.flatnonzero(M[:, col] > 0).tolist())
        assert not (active & animals and active & plants), (
            f"column {col} crosses the animal/plant branch")


def test_one_singular_dimension_per_split(enc):
    """With binary splits throughout, each dimension aligns exactly with one
    node or one sibling contrast."""
    rows = enc.dimension_alignment()
    assert len(rows) == len(enc.items)
    for r in rows:
        assert r["alignment"] == pytest.approx(1.0, abs=1e-9), r
    assert len({r["match_node"] for r in rows}) == len(rows)


@pytest.mark.parametrize("branching", [(2, 2, 2), (2, 2, 4), (2, 2, 6)])
def test_structural_levels_stay_clean_under_nway_leaf_splits(branching):
    """An N-way split needs N-1 dimensions, and within that degenerate subspace
    the SVD basis is arbitrary (the paper notes ties make the choice arbitrary),
    so leaf-level dimensions are not individually interpretable when N > 2.

    What must survive is the structure the schema manipulation depends on: the
    superordinate and category levels are binary and stay perfectly aligned, and
    the leaf-level dimensions still *span* the within-category contrasts even
    though no single one matches a sibling pair.
    """
    e = HierarchicalEncoder(branching=branching, features_per_node=2)
    rows = e.dimension_alignment()
    leaf_level = len(branching)

    for r in rows:
        if r["level"] < leaf_level:
            assert r["alignment"] == pytest.approx(1.0, abs=1e-9), r

    U, _, _ = e.svd()
    leaf_dims = [r["dimension"] for r in rows if r["level"] == leaf_level]
    B = U[:, leaf_dims]
    contrasts = []
    for cat in e.categories:
        leaves = e.nodes[e.name_to_node[cat]]["leaves"]
        for i in range(1, len(leaves)):
            v = np.zeros(len(e.items))
            v[leaves[0]], v[leaves[i]] = 1.0, -1.0
            contrasts.append(v)
    C = np.array(contrasts).T
    residual = np.linalg.norm(C - B @ (B.T @ C)) / np.linalg.norm(C)
    assert residual < 1e-10, residual


def test_singular_values_ordered_by_level(enc):
    """Higher-level splits carry more variance, which is what makes
    superordinate distinctions learn first."""
    rows = enc.dimension_alignment()
    by_level = {}
    for r in rows:
        by_level.setdefault(r["level"], []).append(r["singular_value"])
    levels = sorted(by_level)
    means = [np.mean(by_level[l]) for l in levels]
    assert means == sorted(means, reverse=True), by_level


def test_decoder_interop(enc):
    """SymbolicDecoder must work against this encoder unchanged."""
    dec = SymbolicDecoder(enc)
    for name in enc.items:
        assert dec.decode(enc.encode([name]), top_k=1)[0] == name


def test_unit_norm_rows(enc):
    assert np.allclose(np.linalg.norm(enc.embeddings, axis=1), 1.0)


def test_common_ancestor_level(enc):
    assert enc.common_ancestor_level("sparrow", "hawk") > \
           enc.common_ancestor_level("sparrow", "salmon") > \
           enc.common_ancestor_level("sparrow", "oak")


def test_augment_leaves_base_items_untouched(enc):
    """Base embeddings must be bit-identical after augmentation, or the schema
    phase is not comparable across rungs."""
    aug = enc.augment([{"name": "sparrowhawk", "kind": "within", "parent": "bird"}],
                      seed=0)
    n = len(enc.items)
    assert np.array_equal(aug.feature_matrix[:n, :enc.n_features], enc.feature_matrix)
    assert np.allclose(aug.feature_matrix[:n, enc.n_features:], 0.0)
    assert np.allclose(aug.embeddings[:n, :enc.n_features], enc.embeddings)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_consistency_ladder_is_monotone(enc, seed):
    """The rungs must be ordered by projection onto the existing basis, or the
    benchmark's independent variable does not exist."""
    specs = [
        {"name": "r1", "kind": "duplicate", "target": "sparrow"},
        {"name": "r2", "kind": "within", "parent": "bird"},
        {"name": "r3", "kind": "across", "parent": "bird", "other": "fish"},
        {"name": "r4", "kind": "random"},
    ]
    aug = enc.augment(specs, seed=seed)
    ratios = [enc.projection_ratio(
        aug.feature_matrix[aug.word_to_idx[s["name"]], :enc.n_features])
        for s in specs]
    assert ratios == sorted(ratios, reverse=True), ratios
    assert ratios[0] == pytest.approx(1.0)


def test_within_never_reproduces_an_existing_item(enc):
    """The sparrowhawk is a recombination, not a copy -- if `within` could
    redraw one leaf's exact column set it would collapse onto `duplicate`."""
    for seed in range(25):
        aug = enc.augment([{"name": "sh", "kind": "within", "parent": "bird"}],
                          seed=seed)
        row = aug.feature_matrix[aug.word_to_idx["sh"], :enc.n_features]
        for i in range(len(enc.items)):
            assert not np.array_equal(row, enc.feature_matrix[i]), (
                f"seed {seed} reproduced item {enc.idx_to_word[i]}")


def test_density_matched_across_rungs(enc):
    """Consistency must not be confounded with sparsity or vector norm."""
    specs = [
        {"name": "r1", "kind": "duplicate", "target": "sparrow"},
        {"name": "r2", "kind": "within", "parent": "bird"},
        {"name": "r3", "kind": "across", "parent": "bird", "other": "fish"},
        {"name": "r4", "kind": "random"},
    ]
    aug = enc.augment(specs, seed=0)
    base_density = enc.feature_matrix.sum(axis=1).mean()
    for s in specs:
        row = aug.feature_matrix[aug.word_to_idx[s["name"]], :enc.n_features]
        assert abs(row.sum() - base_density) <= 1.0, s["name"]


def test_balanced_branching_shape():
    e = HierarchicalEncoder(branching=(2, 2, 6), features_per_node=2)
    assert len(e.items) == 24
    assert len(e.categories) == 4
    assert all(len(e.nodes[e.name_to_node[c]]["leaves"]) == 6 for c in e.categories)


def test_rejects_ambiguous_construction():
    with pytest.raises(ValueError):
        HierarchicalEncoder()
    with pytest.raises(ValueError):
        HierarchicalEncoder(tree=PAPER_TREE, branching=(2, 2))
