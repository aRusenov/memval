"""Tests for the cue-masking section.

The load-bearing claims are structural rather than behavioural: that the two
feature blocks really are what the section says they are, that masking removes
features rather than merely shrinking the cue, and above all that the model-free
reference curve is computed honestly. If the reference is wrong every scored
metric in the `cue_completeness` dimension is read against the wrong window, so
its edge cases (ties, empty cues, candidate set) are asserted here rather than
assumed.
"""
import numpy as np
import pytest

from memval.benchmarks.cue_masking import (
    MASK_MODES, build_hierarchy, study_list, feature_blocks, mask_order,
    masked_cue, n_masked, cue_identifiability, cued_probe, rollout_probe,
    _threshold,
)
from memval.encoders.symbolic import SymbolicDecoder


@pytest.fixture
def enc():
    return build_hierarchy()


@pytest.fixture
def words(enc):
    return study_list(enc, 8, "across")


# ---------------------------------------------------------------- blocks ----

def test_blocks_partition_the_active_features(enc):
    for item in enc.items:
        b = feature_blocks(enc, item)
        assert set(b["shared"]).isdisjoint(b["identity"])
        assert set(b["shared"]) | set(b["identity"]) == set(b["active"])


def test_default_substrate_has_equal_blocks(enc):
    """The block contrast is only valid where both masks remove the same number
    of features, so the default tree is chosen to make the blocks 1:1. A change
    that unbalances them silently shrinks the asymmetry window."""
    sizes = {(len(feature_blocks(enc, i)["shared"]),
              len(feature_blocks(enc, i)["identity"])) for i in enc.items}
    assert sizes == {(6, 6)}


def test_identity_features_are_unique_to_their_item(enc):
    """What makes `shared` masking survivable and `identity` masking fatal."""
    for item in enc.items:
        cols = feature_blocks(enc, item)["identity"]
        others = [enc.word_to_idx[o] for o in enc.items if o != item]
        assert not enc.feature_matrix[np.ix_(others, cols)].any()


def test_shared_features_are_the_category(enc):
    """Siblings share their whole shared block; non-siblings share none of it."""
    a = enc.items[0]
    sib = next(i for i in enc.items[1:] if enc.category_of(i) == enc.category_of(a))
    far = next(i for i in enc.items if enc.category_of(i) != enc.category_of(a))
    sa = set(feature_blocks(enc, a)["shared"])
    assert sa == set(feature_blocks(enc, sib)["shared"])
    assert sa.isdisjoint(feature_blocks(enc, far)["shared"])


# ----------------------------------------------------------------- masks ----

def test_block_modes_exhaust_their_own_block_first(enc):
    rng = np.random.default_rng(0)
    item = enc.items[0]
    b = feature_blocks(enc, item)
    k = len(b["shared"])
    assert set(mask_order(enc, item, "shared", rng)[:k]) == set(b["shared"])
    rng = np.random.default_rng(0)
    k = len(b["identity"])
    assert set(mask_order(enc, item, "identity", rng)[:k]) == set(b["identity"])


def test_every_mode_covers_the_full_range(enc):
    """All three modes must reach fraction 1.0, or their curves are not
    comparable point-for-point."""
    rng = np.random.default_rng(0)
    for mode in MASK_MODES:
        order = mask_order(enc, enc.items[0], mode, rng)
        assert set(order) == set(feature_blocks(enc, enc.items[0])["active"])


def test_masking_zeroes_features_rather_than_scaling(enc):
    """The manipulation is a projection onto a coordinate subspace. If it merely
    shrank the cue, cosine decoding would be invariant to it and the whole
    section would read ~flat for a scale-free arm."""
    rng = np.random.default_rng(0)
    item = enc.items[0]
    n_active = len(feature_blocks(enc, item)["active"])
    cue = masked_cue(enc, item, 0.5, "random", rng)
    assert int(np.count_nonzero(cue)) == n_active - n_masked(n_active, 0.5)
    assert float(np.linalg.norm(cue)) == pytest.approx(1.0)


def test_unrenormalised_masking_loses_norm(enc):
    rng = np.random.default_rng(0)
    cue = masked_cue(enc, enc.items[0], 0.5, "random", rng, renormalize=False)
    assert float(np.linalg.norm(cue)) < 0.99


def test_fully_masked_cue_is_empty(enc):
    rng = np.random.default_rng(0)
    for mode in MASK_MODES:
        assert not masked_cue(enc, enc.items[0], 1.0, mode, rng).any()


# ------------------------------------------------------------- reference ----

def test_reference_is_perfect_on_the_clean_cue(enc, words):
    rng = np.random.default_rng(0)
    for mode in MASK_MODES:
        assert cue_identifiability(enc, words, 0.0, mode, rng) == 1.0


def test_reference_collapses_when_identity_is_gone(enc, words):
    """Masking the whole identity block leaves the ancestors' columns, which
    every sibling shares bit for bit. Nothing can tell those cues apart."""
    rng = np.random.default_rng(0)
    n_id = len(feature_blocks(enc, words[0])["identity"])
    n_act = len(feature_blocks(enc, words[0])["active"])
    assert cue_identifiability(enc, words, n_id / n_act, "identity", rng) == 0.0


def test_reference_survives_losing_the_whole_shared_block(enc, words):
    """The complementary half: identity alone still names the item, so shared
    masking costs the reference nothing until the cue is empty."""
    rng = np.random.default_rng(0)
    n_sh = len(feature_blocks(enc, words[0])["shared"])
    n_act = len(feature_blocks(enc, words[0])["active"])
    assert cue_identifiability(enc, words, n_sh / n_act, "shared", rng) == 1.0


def test_reference_counts_ties_as_failures(enc):
    """A tie is not a hit. Identical masked cues would otherwise hand the credit
    to whichever sibling sorts first, and the reference would read ~0.5 where
    the true value is 0 -- flattering every arm measured against it."""
    sibs = [i for i in enc.items if enc.category_of(i) == enc.category_of(enc.items[0])]
    rng = np.random.default_rng(0)
    n_id = len(feature_blocks(enc, sibs[0])["identity"])
    n_act = len(feature_blocks(enc, sibs[0])["active"])
    cues = [masked_cue(enc, s, n_id / n_act, "identity", rng) for s in sibs]
    for c in cues[1:]:                       # precondition: they really are ties
        assert np.allclose(c, cues[0])
    assert cue_identifiability(enc, sibs, n_id / n_act, "identity", rng) == 0.0


def test_empty_cue_names_nothing(enc, words):
    rng = np.random.default_rng(0)
    assert cue_identifiability(enc, words, 1.0, "random", rng) == 0.0


def test_reference_ranks_against_the_studied_list_by_default(enc, words):
    """The default candidate set is the studied list, not the vocabulary."""
    rng = np.random.default_rng(3)
    a = cue_identifiability(enc, words, 0.5, "random", rng, candidates=words)
    rng = np.random.default_rng(3)
    b = cue_identifiability(enc, words, 0.5, "random", rng)
    assert a == b


# ------------------------------------------------------------------ list ----

def test_across_scope_spans_categories(enc):
    ws = study_list(enc, 8, "across")
    assert len({enc.category_of(w) for w in ws}) == len(enc.categories)


def test_within_scope_refuses_to_borrow_silently(enc):
    """Overrunning a category would confound cue completeness with category
    structure -- the same defect as sequence_length's 12-word fruit pool."""
    with pytest.raises(ValueError, match="silently borrow"):
        study_list(enc, 99, "within")


def test_bad_mode_and_scope_raise(enc):
    with pytest.raises(ValueError):
        mask_order(enc, enc.items[0], "nonsense", np.random.default_rng(0))
    with pytest.raises(ValueError):
        study_list(enc, 4, "nonsense")


# --------------------------------------------------------------- readout ----

def test_threshold_reads_the_last_point_above_level():
    f = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert _threshold(f, [1.0, 1.0, 0.9, 0.2, 0.0]) == 0.5
    assert _threshold(f, [0.1, 0.0, 0.0, 0.0, 0.0]) == 0.0
    assert _threshold(f, [1.0] * 5) == 1.0


def test_probes_run_against_a_trained_arm(enc, words):
    """End-to-end on the cheapest arm: a perfectly-trained associator recalls
    the clean cue and both probes stay inside [0, 1]."""
    from memval.models.baselines import AsymmetricHopfieldNetwork
    model = AsymmetricHopfieldNetwork(n_features=enc.embedding_dim, n_epochs=1,
                                      learning_rate=0.1, activation="relu", seed=42)
    model.fit_sequence(enc.encode(words), epochs=1)
    dec = SymbolicDecoder(enc)
    rng = np.random.default_rng(0)
    rec, mar = cued_probe(model, enc, dec, words, 0.0, "random", rng, n_draws=2)
    assert rec == pytest.approx(1.0)
    assert mar > 0
    rng = np.random.default_rng(0)
    rec, _ = cued_probe(model, enc, dec, words, 1.0, "random", rng, n_draws=2)
    assert rec == 0.0                        # an empty cue predicts nothing
    rng = np.random.default_rng(0)
    rol = rollout_probe(model, enc, dec, words, 0.0, "random", rng, n_draws=2)
    assert 0.0 <= rol <= 1.0
