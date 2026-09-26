"""Tests for the cognitive-phenomena (L0) section.

What is pinned: the semantic layout's separation property (the whole
deconfounding argument rests on it), the error taxonomy, the serial-position
band statistics, the availability normalisation of the intrusion lags, and the
contract that the section writes nothing into the scored metrics.
"""
import json
import os

import numpy as np
import pytest

from memval.benchmarks.cognitive_phenomena import (
    HUMAN_REFERENCE, PHENOMENA, SEMANTIC_CLASSES, SEMANTIC_PATTERN,
    band_means, build_semantic_material, classify_semantic, cued_probe,
    position_bands, run_cognitive_phenomena, run_pli_recency,
    validate_semantic_pattern, _percentile_rank,
)
from memval.benchmarks.symbolic_pipeline import load_vocab
from memval.models.baselines.asymmetric_hopfield import AsymmetricHopfieldNetwork

AHN_KW = dict(learning_rate=0.1, activation="relu", seed=42)
FIXED = {"mode": "fixed", "epochs": 4, "criterion": 0.95, "max_epochs": 8, "fallback_epochs": 4}


# ------------------------------------------------------------ the layout ----

def test_pattern_separates_the_four_items_for_every_cue():
    """For every cue, the target, the cue's partner, the target's partner and
    the successor of the cue's partner are four different items, none of them
    a temporal neighbour of the target. This is what lets an error be
    attributed to organisation, output confusability or input generalisation."""
    partner = validate_semantic_pattern(SEMANTIC_PATTERN)
    L = len(SEMANTIC_PATTERN)
    for i in range(L - 1):
        t = i + 1
        items = {t, partner[i], partner[t]}
        if partner[i] + 1 < L:
            items.add(partner[i] + 1)
        assert len(items) == (4 if partner[i] + 1 < L else 3)
        for s in (partner[i], partner[t]):
            assert abs(s - t) != 1
        assert SEMANTIC_PATTERN[i] != SEMANTIC_PATTERN[t]


def test_parallel_pairs_are_rejected():
    """ABCDABCD puts every partner at lag 4, so successor(partner(cue)) IS
    partner(target): the two confusability routes collapse into one item."""
    with pytest.raises(ValueError, match="not distinct"):
        validate_semantic_pattern("ABCDABCD")


def test_adjacent_same_category_is_rejected():
    with pytest.raises(ValueError, match="adjacent"):
        validate_semantic_pattern("AABCDBCD")


# ---------------------------------------------------------- the taxonomy ----

@pytest.fixture
def semantic_material():
    vocab = load_vocab()
    mat = build_semantic_material(vocab, ("fruit", "animal", "vehicle", "color"), seed=1)
    mat["encoder_vocab"] = {w: vocab[w] for w in mat["encoder"].idx_to_word}
    return mat


def test_material_has_two_words_per_category_in_pattern_order(semantic_material):
    m = semantic_material
    assert len(m["words"]) == len(SEMANTIC_PATTERN)
    assert len(set(m["words"])) == len(m["words"])
    for p, cat in enumerate(m["category_of_pos"]):
        assert m["encoder_vocab"][m["words"][p]] == cat
    assert m["n_candidates"] == 8 + 4 * 2 + 2 * 2


def test_every_class_is_reachable(semantic_material):
    m = semantic_material
    w, partner = m["words"], m["partner"]
    i = 0
    assert classify_semantic(i, w[partner[i]], m) == "cue_partner"
    assert classify_semantic(i, w[partner[i + 1]], m) == "target_partner"
    assert classify_semantic(i, w[partner[i] + 1], m) == "cue_partner_successor"
    assert classify_semantic(i, w[i], m) == "cue_repeat"
    assert classify_semantic(i, w[i + 2], m) == "temporal_neighbour"
    assert classify_semantic(i, w[3], m) == "other_in_list"
    eli_cue = next(x for x in m["eli_same"] if m["encoder_vocab"][x] == m["category_of_pos"][i])
    eli_tgt = next(x for x in m["eli_same"] if m["encoder_vocab"][x] == m["category_of_pos"][i + 1])
    eli_lst = next(x for x in m["eli_same"]
                   if m["encoder_vocab"][x] not in (m["category_of_pos"][i], m["category_of_pos"][i + 1]))
    assert classify_semantic(i, eli_cue, m) == "eli_cue_category"
    assert classify_semantic(i, eli_tgt, m) == "eli_target_category"
    assert classify_semantic(i, eli_lst, m) == "eli_list_category"
    assert classify_semantic(i, sorted(m["eli_other"])[0], m) == "eli_other_category"
    assert set(SEMANTIC_CLASSES) == {
        "cue_partner", "target_partner", "cue_partner_successor", "cue_repeat",
        "temporal_neighbour", "other_in_list", "eli_cue_category",
        "eli_target_category", "eli_list_category", "eli_other_category"}


def test_percentile_rank_is_chance_free():
    sims = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
    assert _percentile_rank(sims, 1, exclude=[0]) == pytest.approx(1.0)   # most similar
    assert _percentile_rank(sims, 3, exclude=[0]) == pytest.approx(0.0)   # least similar
    assert np.isnan(_percentile_rank(sims, 0, exclude=[0]))


# ------------------------------------------------- serial-position bands ----

def test_bands_and_indices():
    b = position_bands(7)
    assert b == {"early": [1, 2], "middle": [3, 4], "late": [5, 6]}
    curve = [np.nan, 1, 1, 0, 0, 1, 1]
    m = band_means(curve)
    assert m["primacy_index"] == pytest.approx(1.0)
    assert m["recency_index"] == pytest.approx(1.0)
    flat = band_means([np.nan] + [0.5] * 6)
    assert flat["primacy_index"] == pytest.approx(0.0)
    short = position_bands(4)
    assert short == {"early": [1], "middle": [2], "late": [3]}


def test_cued_probe_scores_positions_one_onward():
    vocab = load_vocab()
    words = [w for w, c in vocab.items() if c == "fruit"][:5]
    from memval.encoders.symbolic import SymbolicEncoder
    enc = SymbolicEncoder({w: "fruit" for w in words}, embedding_dim=32, category_variance=0.2, seed=0)
    m = AsymmetricHopfieldNetwork(n_features=32, n_epochs=8, **AHN_KW)
    m.fit_sequence(enc.encode(words), epochs=8)
    pr = cued_probe(m, words, enc)
    assert np.isnan(pr["hit"][0]) and np.isnan(pr["margin"][0])
    assert pr["decoded"][0] is None
    assert pr["sims"].shape == (4, 5)
    assert all(pr["competitor"][p] != words[p] for p in range(1, 5))


# ------------------------------------------------- intrusion availability ----

def test_pli_lags_are_availability_normalised():
    out = run_pli_recency(AsymmetricHopfieldNetwork, AHN_KW, load_vocab(), FIXED,
                          n_tasks=3, seq_len=3, between_cosines=(0.5,), embedding_dim=32)
    assert set(out["rungs"]) == {"0.50/criterion", "0.50/one_pass"}
    assert all(e["epochs"] == 1 for e in out["rungs"]["0.50/one_pass"]["exposures"])
    r = out["rungs"]["0.50/criterion"]
    assert r["lags"] == [1, 2]
    assert r["opportunities"] == [(3 - 1) * 2, (3 - 1) * 1]
    assert r["n_current_probes"] == 3 * 2
    assert len(r["records"]) == 6
    if r["pli_total"]:
        assert sum(r["pli_share_by_lag"]) == pytest.approx(1.0)
    else:
        assert all(np.isnan(v) for v in r["pli_share_by_lag"])
    assert np.array(r["retention_matrix"]).shape == (3, 3)
    assert out["human_reference"] is HUMAN_REFERENCE["pli_recency"]


# --------------------------------------------------------------- contract ----

def test_grid_replicates_are_distinct_material():
    """Two one-category encoders under one seed draw identical vectors, so every
    (category, seed) pair must be seeded apart or the replicate SD is zero by
    construction. The replicate unit is (category, encoder seed)."""
    from memval.benchmarks.cognitive_phenomena import run_position_grid, derive_serial_position
    grid = run_position_grid(AsymmetricHopfieldNetwork, AHN_KW, load_vocab(), FIXED,
                             lengths=(5,), categories=("fruit", "animal"), ladder=(1,),
                             n_seeds=3, embedding_dim=32)
    seeds = {c["encoder_seed"] for c in grid["cells"].values()}
    assert len(seeds) == 6 == grid["n_replicates"]
    sp = derive_serial_position(grid, 5)
    assert max(sp["rungs"]["1"]["cued_margin_sd"][1:]) > 0
    assert sp["rungs"]["1"]["bands_margin_ci"]["recency_index"]["n"] == 6


def test_direction_is_undetermined_when_the_interval_spans_zero():
    """A band index whose replicate interval covers zero states no direction.
    This is the gate that the n=3 run lacked, which let a sign-unstable mean be
    read as a match to the human curve."""
    from memval.benchmarks.cognitive_phenomena import _ci, _emit_ci

    spans = _ci([0.10, -0.08, 0.04, -0.06, 0.02])
    assert spans["spans_zero"] and spans["n"] == 5
    clear = _ci([0.20, 0.22, 0.19, 0.25, 0.21])
    assert not clear["spans_zero"]
    assert _ci([])["spans_zero"] and _ci([0.5])["spans_zero"]   # n<2 cannot support one

    s = {}
    _emit_ci(s, "cog_x", spans)
    assert s["cog_x_spans_zero"] is True and s["cog_x_n"] == 5
    assert s["cog_x_ci_lo"] < 0 < s["cog_x_ci_hi"]


def test_section_writes_nothing_to_scored_metrics(tmp_path):
    out = run_cognitive_phenomena(
        AsymmetricHopfieldNetwork, AHN_KW, exposure=FIXED,
        lengths=(4, 5), headline_length=5, categories=("fruit", "animal"), ladder=(1, 2),
        pli_n_tasks=3, pli_seq_len=3, pli_between_cosines=(0.5,),
        semantic_variances=(0.2, 3.0), semantic_ladder=(1,), embedding_dim=32,
        run_dir=str(tmp_path), run_name="ahn-test")
    assert out["metrics"] == {}
    s = out["series"]["cognitive_phenomena"]["summary"]
    for key in ("cog_spc_primacy_acc_first", "cog_ll_dissociation_margin_criterion",
                "cog_pr_dissociation_margin", "cog_pli_lag1_share",
                "cog_sem_s1_cue_partner_share", "cog_sem_calibrated"):
        assert key in s
    assert all(k.startswith("cog_") for k in s)
    with open(os.path.join(tmp_path, "cognitive_phenomena_metrics.json")) as f:
        payload = json.load(f)
    assert set(payload["phenomena"]) >= set(PHENOMENA)
    assert payload["human_reference"] == HUMAN_REFERENCE
    for stem in ("cognitive_serial_position", "cognitive_list_length",
                 "cognitive_presentation_rate", "cognitive_pli_recency",
                 "cognitive_semantic_clustering"):
        assert os.path.exists(os.path.join(tmp_path, "plots", f"{stem}.png"))
