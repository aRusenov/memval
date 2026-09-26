"""The probe protocol: clean single cue by default, degradation only where it
is the declared manipulation, and every section says which."""

import numpy as np
import pytest

from memval.benchmarks.probe import (
    SECTION_PROBE_PROTOCOL, ProbeProtocol, resolve_probe, probe_record,
    measure_recall_margin, mean_margin,
)
from memval.benchmarks.symbolic_pipeline import (
    SYMBOLIC_BENCHMARKS, measure_recall_associative, mean_recall_rate,
)
from memval.benchmarks.online_symbolic_pipeline import ONLINE_SYMBOLIC_BENCHMARKS
from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.models.base import HippocampalModel
from memval.models.capabilities import is_stochastic_forward
from memval.models.baselines import OriginalEqPropSequenceNetwork, ThetaPhaseSequenceNetwork

WORDS = ["apple", "banana", "orange", "grape", "pear", "peach", "plum"]


def _enc():
    e = SymbolicEncoder({w: "fruit" for w in WORDS}, embedding_dim=64,
                        category_variance=0.2, seed=42)
    return e, SymbolicDecoder(e)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_every_shipped_section_declares_a_probe_protocol():
    missing = [s for s in SYMBOLIC_BENCHMARKS + ONLINE_SYMBOLIC_BENCHMARKS
               if s not in SECTION_PROBE_PROTOCOL]
    assert not missing, missing


def test_only_degradation_sections_are_sweeps():
    sweeps = {s for s, p in SECTION_PROBE_PROTOCOL.items()
              if p in (ProbeProtocol.NOISE_SWEEP, ProbeProtocol.MASK_SWEEP)}
    assert sweeps == {"noise_invariance", "cue_masking"}


def test_no_shipped_section_uses_the_legacy_protocol():
    assert ProbeProtocol.NOISE_TRIALS not in SECTION_PROBE_PROTOCOL.values()


def test_undeclared_section_is_an_error():
    with pytest.raises(KeyError, match="no entry"):
        resolve_probe("not_a_section", ThetaPhaseSequenceNetwork, 30)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_clean_single_resolves_to_one_clean_probe_for_a_deterministic_arm():
    assert not is_stochastic_forward(ThetaPhaseSequenceNetwork)
    assert resolve_probe("multiple_sequences", ThetaPhaseSequenceNetwork, 30) == (1, 0.0)


def test_clean_single_keeps_repeats_only_for_a_stochastic_arm():
    class Noisy(ThetaPhaseSequenceNetwork):
        stochastic_forward = True

    assert resolve_probe("multiple_sequences", Noisy, 30) == (30, 0.0)


def test_noise_sweep_passes_the_level_through():
    assert resolve_probe("noise_invariance", ThetaPhaseSequenceNetwork, 30,
                         sweep_level=0.3) == (30, 0.3)
    with pytest.raises(ValueError, match="sweep_level"):
        resolve_probe("noise_invariance", ThetaPhaseSequenceNetwork, 30)


def test_probe_record_is_serialisable_and_complete():
    r = probe_record("presentation_duration", OriginalEqPropSequenceNetwork, 30)
    assert set(r) == {"protocol", "n_trials", "noise_scale", "stochastic_forward"}
    assert r["protocol"] == "clean_single" and r["n_trials"] == 1 and r["noise_scale"] == 0.0


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------

def test_thirty_noise_trials_collapsed_to_one_for_a_well_trained_arm():
    """The empirical fact this refactor rests on: at ceiling, every noise draw
    agreed, so the trials were pure cost. The clean single probe reproduces
    the number exactly."""
    enc, dec = _enc()
    m = ThetaPhaseSequenceNetwork(n_features=64)
    m.fit_sequence(enc.encode(WORDS), epochs=30)
    np.random.seed(0)
    noisy = measure_recall_associative(m, WORDS, enc, dec, n_trials=30, noise_scale=0.05)
    clean = measure_recall_associative(m, WORDS, enc, dec, n_trials=1, noise_scale=0.0)
    assert mean_recall_rate(noisy) == mean_recall_rate(clean) == 1.0


def test_margin_is_deterministic_and_resolves_below_the_accuracy_floor():
    enc, dec = _enc()
    m = OriginalEqPropSequenceNetwork(n_features=64, n_hidden=32, seed=0, n_epochs=4)
    m.fit_sequence(enc.encode(WORDS), epochs=4)
    a = measure_recall_margin(m, WORDS, enc)
    b = measure_recall_margin(m, WORDS, enc)
    assert np.isnan(a[0]) and np.allclose(a[1:], b[1:])
    acc = measure_recall_associative(m, WORDS, enc, dec, n_trials=1, noise_scale=0.0)
    # Where accuracy is binary, margin is graded: it varies across positions
    # even where accuracy does not.
    assert np.std(a[1:]) > 0.0
    assert np.isfinite(mean_margin(a))
    # Sign agreement: a positive margin is exactly a correct clean decode.
    assert np.array_equal(a[1:] > 0, acc[1:] > 0.5)


def test_margin_position_zero_is_the_cue_not_a_score():
    enc, _ = _enc()
    m = ThetaPhaseSequenceNetwork(n_features=64)
    assert np.isnan(measure_recall_margin(m, WORDS, enc)[0])


def test_base_declares_deterministic_forward_by_default():
    assert HippocampalModel.stochastic_forward is False


# ---------------------------------------------------------------------------
# Sites that carried their own copy of the legacy protocol
# ---------------------------------------------------------------------------

def test_spatial_disambiguation_runs_once_for_a_deterministic_arm():
    from memval.generators.bifurcating_route import BifurcatingRouteGenerator
    from memval.benchmarks.spatial_disambiguation import SpatialDisambiguationBenchmark

    route = BifurcatingRouteGenerator(seed=1).generate(
        total_length=8, shared_fraction=0.5, shared_position=0.0, zone_fraction=1.0,
        zone_offset=0.0, n_place_cells=36, n_encounter_dims=4,
        balance_modalities=True, odour_scale=2.0, odour_on_suffix=True, seed=1)
    m = ThetaPhaseSequenceNetwork(n_features=40)
    r = SpatialDisambiguationBenchmark().evaluate(model=m, route_pair=route,
                                                  n_trials=20, fit_epochs=1)
    assert r["probe_protocol"] == "clean_single"
    assert r["n_trials_requested"] == 20 and r["n_trials_effective"] == 1
    assert r["stochastic_forward"] is False


def test_schema_probe_defaults_are_clean():
    import inspect
    from memval.benchmarks.schema_consistency import run_schema_consistency
    p = inspect.signature(run_schema_consistency).parameters
    assert p["n_probe_trials"].default == 1 and p["noise_scale"].default == 0.0


def test_sub_benchmark_defaults_are_clean():
    import inspect
    from memval.benchmarks.continual_chain import ContinualChainBenchmark
    from memval.benchmarks.paired_associate import PairedAssociateBenchmark
    from memval.benchmarks.interval_timing import run_interval_retention
    for obj, name in [(ContinualChainBenchmark, "n_trials"), (PairedAssociateBenchmark, "n_trials")]:
        p = inspect.signature(obj.__init__).parameters
        assert p["n_trials"].default == 1 and p["noise_scale"].default == 0.0, obj
    assert inspect.signature(run_interval_retention).parameters["n_trials"].default == 1


# ---------------------------------------------------------------------------
# multiple_sequences: probability over list pairs + margin panel
# ---------------------------------------------------------------------------

def test_multiple_sequences_replicates_over_list_pairs(tmp_path):
    import json, glob, os, numpy as np
    from memval.benchmarks.symbolic_pipeline import run_symbolic_pipeline
    run_symbolic_pipeline(ThetaPhaseSequenceNetwork,
                          model_kwargs={"n_epochs": 4, "learning_rate": 0.1, "rectify_output": True,
                                        "modulation_depth": 1.0, "seed": 42},
                          output_dir=str(tmp_path), n_trials=30, benchmarks=["multiple_sequences"],
                          benchmark_args={"multiple_sequences": {"n_lists": 2, "ingestion": "blocked"}})
    d = json.load(open(glob.glob(os.path.join(str(tmp_path), "*", "symbolic", "*.json"))[0]))
    m, L = d["metrics"], d["series"]["multiple_sequences_lists"]
    assert m["multiple_seq_n_lists"] == 2 and len(L["pairs"]) == 2
    assert L["pairs"][0]["A"] == "fruit" and L["pairs"][0]["B"] == "animal"   # canonical first
    assert len(L["prob_before"]) == 7 and len(L["margin_after"]) == 7
    assert all(0.0 <= v <= 1.0 for v in L["prob_after"])
    assert np.isnan(L["margin_after"][0]) and np.all(np.isfinite(L["margin_after"][1:]))
    for k in ("multiple_seq_mrr_before", "delta_mrr_forgetting", "multiple_seq_prob_before",
              "delta_prob_forgetting", "delta_margin_forgetting", "delta_margin_forgetting_lists"):
        assert k in m
    assert os.path.exists(glob.glob(os.path.join(str(tmp_path), "*", "symbolic", "plots", "multiple_seq_forgetting.png"))[0])
    # overlap condition: model-free covariate must separate the two conditions,
    # and every per-condition key must land.
    ov = d["series"]["multiple_sequences_overlap"]
    assert ov["same_category"]["in_span"] > ov["disjoint"]["in_span"]
    for c in ("disjoint", "same_category"):
        for k in ("in_span", "prob_before", "prob_after", "delta_prob", "margin_before", "margin_after", "delta_margin"):
            assert f"multiple_seq_overlap_{c}_{k}" in m
    assert "multiple_seq_overlap_cost_margin" in m and "multiple_seq_overlap_cost_prob" in m
    assert os.path.exists(glob.glob(os.path.join(str(tmp_path), "*", "symbolic", "plots", "multiple_seq_overlap.png"))[0])
