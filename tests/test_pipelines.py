import os
import shutil
import tempfile
import json
import pytest
import numpy as np

from memval.models.baselines.hopfield import HopfieldSequenceNetwork
from memval.benchmarks.spatial_pipeline import run_spatial_pipeline
from memval.benchmarks.symbolic_pipeline import run_symbolic_pipeline
from memval.benchmarks.online_symbolic_pipeline import run_online_symbolic_pipeline

@pytest.fixture
def temp_results_dir():
    """Fixture to create and clean up a temporary directory for output files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

def test_spatial_pipeline(temp_results_dir):
    """Test that the spatial pipeline runs successfully with HopfieldSequenceNetwork."""
    model_class = HopfieldSequenceNetwork
    model_kwargs = {
        "learning_rate": 0.5,
        "activation": "linear"
    }

    # Run with small trials to keep tests fast
    metrics = run_spatial_pipeline(
        model_class=model_class,
        model_kwargs=model_kwargs,
        output_dir=temp_results_dir,
        n_trials=2
    )

    # Check that metrics dictionary is returned
    assert "tmaze_pc_mse" in metrics
    # Non-saturating completion readouts, moved here from the removed
    # spatial_sequence section (audit C1) so MSE is not the only measure.
    assert "tmaze_pc_coverage" in metrics
    assert "tmaze_pc_divergence_step" in metrics
    assert 0 <= metrics["tmaze_pc_divergence_step"] <= metrics["tmaze_pc_recall_len"]
    assert "tmaze_disamb_full_branch_acc" in metrics
    assert "tmaze_disamb_mec_only_branch_acc" in metrics
    # Graded odour availability (audit D2): a concurrent reference row plus one
    # row per cue-free delay. The withdrawn rows are expected at a protocol
    # floor under the memoryless probe, so only presence is asserted here.
    assert "tmaze_disamb_graded_concurrent_divergence_accuracy" in metrics
    assert "tmaze_disamb_graded_delay0_divergence_accuracy" in metrics

    # Check directory structure and file contents
    run_dir = os.path.join(temp_results_dir, "HopfieldSequenceNetwork", "spatial")
    assert os.path.isdir(run_dir)
    assert os.path.isdir(os.path.join(run_dir, "plots"))
    
    # Check that plots exist
    assert os.path.exists(os.path.join(run_dir, "plots", "tmaze_completion.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "tmaze_disambiguation_graded.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "tmaze_disambiguation.png"))

    # Check metrics.json contents
    json_path = os.path.join(run_dir, "metrics.json")
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
        assert data["metadata"]["model_name"] == "HopfieldSequenceNetwork"
        assert data["metadata"]["suite"] == "spatial"
        assert "metrics" in data
        assert "series" in data
        assert len(data["series"]["tmaze_true"]) > 0

def test_symbolic_pipeline(temp_results_dir):
    """Test that the symbolic pipeline runs successfully with HopfieldSequenceNetwork."""
    model_class = HopfieldSequenceNetwork
    model_kwargs = {
        "learning_rate": 0.1,
        "activation": "relu",
        "fit_method": "delta"
    }

    # Run with small trials to keep tests fast. schema_consistency retrains a
    # fresh model per rung, so it gets a reduced tree and trial counts here; the
    # section defaults live in run_symbolic_pipeline.
    metrics = run_symbolic_pipeline(
        model_class=model_class,
        model_kwargs=model_kwargs,
        output_dir=temp_results_dir,
        n_trials=2,
        benchmark_args={
            "schema_consistency": {
                "branching": (2, 2, 3),
                "schema_trials": 8,
                "new_item_trials": 4,
                "n_probe_trials": 3,
                "n_seeds": 1,          # keep the suite test fast
            },
            # cue_masking sweeps every realisable masked-feature count in three
            # modes with two probes each, so the grid is what costs time here.
            # A smaller tree shortens the grid; the section defaults live in
            # run_symbolic_pipeline.
            "cue_masking": {
                "branching": (2, 4),
                "features_per_node": 2,
                "seq_len": 4,
                "n_draws": 2,
            },
        },
    )

    # Check that metrics dictionary is returned
    assert "convergence_mrr" in metrics
    assert "convergence_span" in metrics
    assert "convergence_epochs" in metrics or metrics.get("convergence_epochs") is None
    assert "converged" in metrics
    assert "max_memory_span" in metrics
    assert "multiple_seq_mrr_before" in metrics
    assert "multiple_seq_mrr_after" in metrics
    assert "delta_mrr_forgetting" in metrics

    # Continual chain: both axes of the retention matrix, plus the load curve
    # and the selective-retention phase. The plasticity keys are the ones no
    # other section in the suite produces.
    for key in ("chain_avg_accuracy", "chain_avg_forgetting",
                "chain_avg_learning", "chain_intransigence",
                "chain_retention_ratio", "chain_stability_plasticity_index",
                "chain_acquired", "chain_n_tasks"):
        assert key in metrics, key
    for key in ("select_selectivity", "select_rehearsed_gain",
                "select_unrehearsed_drift", "select_under_pressure",
                "select_rehearsal_effective"):
        assert key in metrics, key

    # AB/AC: the experimental arm, the disjoint-cue control, and the contrast.
    for cond in ("abac", "control"):
        for tail in ("ab_recall_baseline", "ab_recall_final", "ab_retention",
                     "ac_recall_final", "ac_trials_to_criterion",
                     "other_rate_final", "ab_acquired"):
            assert f"pa_{cond}_{tail}" in metrics, f"pa_{cond}_{tail}"
    assert "pa_cue_competition_cost" in metrics
    # The control must never train the C list: that floor is what makes the
    # AB/AC overwrite attributable to the model rather than to the vocabulary.
    assert metrics["pa_control_ac_recall_final"] == 0.0
    assert "noise_tolerance_threshold" in metrics
    assert "mrr_high_similarity" in metrics
    assert "mrr_low_similarity" in metrics
    assert "similarity_effect_mrr_drop" in metrics

    # Schema consistency: the ladder is model-free, so its ordering must hold
    # regardless of how the model performs. The per-rung readouts must all be
    # present for the section to be reportable.
    assert metrics["schema_ladder_ordered"] is True
    for rung in ("duplicate", "within", "across", "random"):
        assert f"schema_{rung}_projection_ratio" in metrics
        assert f"schema_{rung}_trials_to_criterion" in metrics
        for rel in ("same", "sibling", "far"):
            assert f"schema_{rung}_interference_{rel}" in metrics
        # Acquisition is scored on the new transition alone; the sequence-mean
        # is retained as a diagnostic only.
        assert f"schema_{rung}_new_item_recall" in metrics
        assert f"schema_{rung}_new_item_auc" in metrics
        assert f"schema_{rung}_prefix_recall" in metrics
        assert f"schema_{rung}_final_mrr" in metrics
        # Unit-free cost: ttc / staircased schema-acquisition trials.
        assert f"schema_{rung}_relative_cost" in metrics
    assert "schema_acquired" in metrics
    assert "schema_resolved" in metrics
    assert "schema_auc_resolved" in metrics
    # Cue masking: the scored in-bound rates, the guard that gates the whole
    # dimension, and the model-free reference the rates are read against. The
    # reference is asserted separately because without it the in-bound window is
    # undefined and every rate silently becomes a full-grid average.
    for key in ("mask_random_recall_in_bound", "mask_shared_recall_in_bound",
                "mask_identity_recall_in_bound", "mask_random_tolerance",
                "mask_random_margin_in_bound", "mask_block_asymmetry",
                "mask_identifiability_auc", "mask_identifiable_to",
                "mask_acquired", "mask_resolved", "mask_criterion_reached"):
        assert key in metrics, f"cue_masking did not emit {key}"
    assert 0.0 <= metrics["mask_random_recall_in_bound"] <= 1.0
    assert 0.0 <= metrics["mask_identifiable_to"] <= 1.0

    # Corruption rollout: one feedback mode (raw) since 2026-09-26. Quantized
    # restated the cued curve and l2 was a harness intervention; neither key
    # may come back.
    assert "noise_rollout_auc" in metrics
    assert "noise_rollout_raw_auc" not in metrics
    assert "noise_rollout_quantized_auc" not in metrics

    assert "schema_consistency_auc_corr" in metrics
    assert "schema_criterion_below_dilution_floor" in metrics
    # Schema acquisition is staircased to criterion, capped at schema_trials.
    assert "schema_acquisition_trials" in metrics
    assert "schema_acquisition_reached" in metrics
    # `duplicate` is reported separately; the headline correlations exclude it.
    assert metrics["schema_corr_rungs"] == "within/across/random"
    assert "schema_duplicate_excess_trials" in metrics
    for rung in ("duplicate", "within", "across", "random"):
        assert 0.0 <= metrics[f"schema_{rung}_nearest_base_cosine"] <= 1.0 + 1e-9
    assert metrics["schema_duplicate_nearest_base_cosine"] >= metrics["schema_within_nearest_base_cosine"]
    assert metrics["schema_acquisition_trials"] <= 8      # the test's schema_trials

    # Check directory structure and file contents
    run_dir = os.path.join(temp_results_dir, "HopfieldSequenceNetwork", "symbolic")
    assert os.path.isdir(run_dir)
    assert os.path.isdir(os.path.join(run_dir, "plots"))
    
    # Check that plots exist
    assert os.path.exists(os.path.join(run_dir, "plots", "convergence_curve.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "length_curves.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "multiple_seq_forgetting.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "continual_chain_retention.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "continual_chain_axes.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "continual_chain_selectivity.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "paired_associate_abac.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "noise_invariance.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "cue_masking.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "semantic_similarity.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "schema_consistency.png"))
    # The section writes its own metrics file beside the suite's, not over it.
    assert os.path.exists(os.path.join(run_dir, "schema_consistency_metrics.json"))

    # Check metrics.json contents
    json_path = os.path.join(run_dir, "metrics.json")
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
        assert data["metadata"]["model_name"] == "HopfieldSequenceNetwork"
        assert data["metadata"]["suite"] == "symbolic"
        assert "metrics" in data
        assert "series" in data
        assert "convergence_curve" in data["series"]
        assert "noise_sweep" in data["series"]

def test_online_symbolic_pipeline(temp_results_dir):
    """Test that the online symbolic pipeline runs successfully with HopfieldSequenceNetwork."""
    model_class = HopfieldSequenceNetwork
    model_kwargs = {
        "learning_rate": 0.1,
        "activation": "linear",
        "fit_method": "hebbian"
    }

    # Run with small trials to keep tests fast
    metrics = run_online_symbolic_pipeline(
        model_class=model_class,
        model_kwargs=model_kwargs,
        output_dir=temp_results_dir,
        n_trials=2
    )

    # Check that metrics dictionary is returned
    assert "online_mrr_1shot" in metrics
    assert "online_mrr_5pass" in metrics
    assert "online_mrr_20pass" in metrics
    assert "online_span_1shot" in metrics
    assert "online_span_5pass" in metrics
    assert "online_span_20pass" in metrics
    assert "isi_tolerance_mrr_sweep" in metrics
    assert "isi_tolerance_span_sweep" in metrics

    # Check directory structure and file contents
    run_dir = os.path.join(temp_results_dir, "HopfieldSequenceNetwork", "online_symbolic")
    assert os.path.isdir(run_dir)
    assert os.path.isdir(os.path.join(run_dir, "plots"))

    # Check that plots exist
    assert os.path.exists(os.path.join(run_dir, "plots", "online_convergence.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "isi_tolerance.png"))

    # Check metrics.json contents
    json_path = os.path.join(run_dir, "metrics.json")
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
        assert data["metadata"]["model_name"] == "HopfieldSequenceNetwork"
        assert data["metadata"]["suite"] == "online_symbolic"
        assert "metrics" in data
        assert "series" in data
        assert "online_convergence" in data["series"]
        assert "isi_tolerance" in data["series"]


def test_schema_interference_protocol(tmp_path):
    """The protocol switch changes what is TRAINED, never what is probed.

    Under "extended" the new list is the host list plus one item, so learning it
    rehearses the host category and the `same` interference bucket is invalid;
    `schema_interference_valid` must say so. Under "focused" only the new pair is
    trained. Both must emit the same read-out keys, since the probe is identical.
    """
    from memval.benchmarks.schema_consistency import run_schema_consistency
    from memval.models.baselines.asymmetric_hopfield import AsymmetricHopfieldNetwork

    common = dict(model_class=AsymmetricHopfieldNetwork, branching=(2, 2, 3),
                  schema_trials=4, new_item_trials=2, n_probe_trials=2,
                  n_seeds=1,
                  output_dir=str(tmp_path))
    ext = run_schema_consistency(**common, interference_protocol="extended",
                                 run_name="ext")
    foc = run_schema_consistency(**common, interference_protocol="focused",
                                 run_name="foc")

    assert ext["metrics"]["schema_interference_valid"] is False
    assert foc["metrics"]["schema_interference_valid"] is True
    assert ext["metadata"]["interference_protocol"] == "extended"
    assert foc["metadata"]["interference_protocol"] == "focused"

    # Identical probe => identical key sets, so the two runs stay comparable.
    assert set(ext["metrics"]) == set(foc["metrics"])

    # No arm can be told which sequence a pair belongs to (audit item 18).
    assert ext["metrics"]["schema_sequence_identity_available"] is False


def test_schema_rejects_unknown_protocol(tmp_path):
    from memval.benchmarks.schema_consistency import run_schema_consistency
    from memval.models.baselines.asymmetric_hopfield import AsymmetricHopfieldNetwork
    with pytest.raises(ValueError, match="interference_protocol"):
        run_schema_consistency(AsymmetricHopfieldNetwork,
                               interference_protocol="interleaved",
                               output_dir=str(tmp_path))


def test_every_symbolic_metric_is_classified_by_the_scorecard(temp_results_dir):
    """`score_capacities.py` raises on an unclassified key, so a new metric that
    never reaches its SPEC breaks the scorecard rather than being dropped
    silently. Check the two continual-retention sections here, where the keys
    are newest, without paying for the whole suite.
    """
    import importlib.util

    metrics = run_symbolic_pipeline(
        model_class=HopfieldSequenceNetwork,
        model_kwargs={"learning_rate": 0.1, "activation": "relu",
                      "fit_method": "delta"},
        output_dir=temp_results_dir,
        n_trials=2,
        benchmarks=["continual_chain", "paired_associate"],
        benchmark_args={"continual_chain": {"n_tasks": 3, "epochs": 20},
                        "paired_associate": {"phase2_trials": 2, "epochs": 20}},
    )

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "score_capacities", os.path.join(root, "bin", "score_capacities.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    unclassified = sorted(k for k in metrics if k not in mod.SPEC)
    assert not unclassified, f"add these to SPEC in bin/score_capacities.py: {unclassified}"

    # Every dimension the two sections feed must also carry a weight, or its
    # metrics are computed and then silently excluded from the rollup.
    dims = {mod.SPEC[k]["dimension"] for k in metrics}
    assert dims <= set(mod.DIMENSION_WEIGHTS["Continual retention"]), dims


def test_schema_seed_replication(tmp_path):
    """Mean under the original key, <key>_sd beside it, guards all()-ed with a
    fraction; n_seeds=1 gives nan s.d. and identical keys otherwise."""
    from memval.benchmarks.schema_consistency import run_schema_consistency
    from memval.models.baselines.asymmetric_hopfield import AsymmetricHopfieldNetwork
    common = dict(model_class=AsymmetricHopfieldNetwork, branching=(2, 2, 3),
                  schema_trials=4, new_item_trials=2, n_probe_trials=2,
                  output_dir=str(tmp_path))
    two = run_schema_consistency(**common, n_seeds=2, run_name="two")
    one = run_schema_consistency(**common, n_seeds=1, run_name="one")
    m2, m1 = two["metrics"], one["metrics"]
    assert two["metadata"]["n_seeds"] == 2 and two["metadata"]["seeds"] == [0, 1]
    assert two["metadata"]["model_seeded"] is False        # AHN takes no seed kwarg
    assert "schema_duplicate_trials_to_criterion_sd" in m2
    assert "schema_acquired_frac" in m2 and 0.0 <= m2["schema_acquired_frac"] <= 1.0
    assert isinstance(m2["schema_acquired"], bool)
    assert np.isnan(m1["schema_duplicate_trials_to_criterion_sd"])
    assert set(m1) == set(m2)                              # same key set either way
    assert len(two["series"]["per_seed"]) == 2
    with pytest.raises(ValueError, match="n_seeds"):
        run_schema_consistency(**common, n_seeds=0, run_name="zero")
