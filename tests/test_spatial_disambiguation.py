import numpy as np
from memval.benchmarks.overlap_config import OverlapConfig
from memval.generators.bifurcating_route import BifurcatingRouteGenerator
from memval.benchmarks.spatial_disambiguation import SpatialDisambiguationBenchmark
from memval.models.baselines import AsymmetricHopfieldNetwork

def test_overlap_config_resolve():
    config = OverlapConfig(
        total_length=30,
        shared_fraction=0.40,
        shared_position=0.33,
        zone_fraction=0.50,
        zone_offset=0.25
    )
    geom = config.resolve()
    assert geom["shared_start"] >= 0
    assert geom["shared_end"] <= 30
    assert geom["zone_start"] >= geom["shared_start"]
    assert geom["zone_end"] <= geom["shared_end"]
    
    # Test boundary limits
    extreme_config = OverlapConfig(
        total_length=20,
        shared_fraction=1.0,
        shared_position=0.0,
        zone_fraction=1.0,
        zone_offset=0.0
    )
    geom_ext = extreme_config.resolve()
    assert geom_ext["shared_start"] == 0
    assert geom_ext["shared_end"] == 20
    assert geom_ext["zone_start"] == 0
    assert geom_ext["zone_end"] == 20

def test_bifurcating_route_generator():
    generator = BifurcatingRouteGenerator(seed=42)
    res = generator.generate(
        total_length=30,
        shared_fraction=0.40,
        shared_position=0.33,
        zone_fraction=0.50,
        zone_offset=0.25,
        encounter_similarity=0.0,
        n_place_cells=400,
        n_encounter_dims=50
    )
    
    assert res["input_A"].shape == (30, 450)
    assert res["input_B"].shape == (30, 450)
    assert "coords_A" not in res
    assert "coords_B" not in res
    
    # Verify prefix, shared and suffix segments via decoded place cell coordinates
    encoder = res["encoder"]
    n_pc = encoder.n_cells
    decoded_A = encoder.decode(res["input_A"][:, :n_pc])
    decoded_B = encoder.decode(res["input_B"][:, :n_pc])
    
    ss = res["shared_start"]
    se = res["shared_end"]
    
    # Prefix must be different (divergent paths)
    if ss > 0:
        assert not np.allclose(decoded_A[:ss], decoded_B[:ss])
    # Shared must be identical (same place cells fire for both routes)
    assert np.allclose(decoded_A[ss:se], decoded_B[ss:se])
    # Suffix must be different (divergent arms)
    if se < 30:
        assert not np.allclose(decoded_A[se:], decoded_B[se:])

def test_spatial_disambiguation_evaluate():
    # Use AHN (cheap, deterministic) as the harness arm to verify evaluate/sweep executes end-to-end
    generator = BifurcatingRouteGenerator(seed=42)
    route_pair = generator.generate(
        total_length=20,
        shared_fraction=0.50,
        shared_position=0.20,
        zone_fraction=0.50,
        zone_offset=0.0,
        encounter_similarity=0.0,
        n_place_cells=100,
        n_encounter_dims=10
    )
    
    # Instantiate the harness arm
    # n_features = 110 (100 place cells + 10 encounter dims)
    model = AsymmetricHopfieldNetwork(n_features=110)
    
    benchmark = SpatialDisambiguationBenchmark()
    
    # Test evaluation for different conditions
    for cond in ["full", "mec_only", "lec_only", "conflicting"]:
        # Instantiate fresh model for each condition
        model = AsymmetricHopfieldNetwork(n_features=110)
        metrics = benchmark.evaluate(
            model=model,
            route_pair=route_pair,
            condition=cond,
            n_trials=2,
            fit_epochs=1
        )
        assert "branch_accuracy" in metrics
        assert "confusion_rate" in metrics
        assert "suffix_mse" in metrics
        assert 0.0 <= metrics["branch_accuracy"] <= 1.0
        assert 0.0 <= metrics["confusion_rate"] <= 1.0

def test_spatial_disambiguation_sweep():
    configs = [
        OverlapConfig(total_length=20, shared_fraction=0.40, shared_position=0.20),
        OverlapConfig(total_length=20, shared_fraction=0.50, shared_position=0.30)
    ]
    conditions = ["full", "mec_only"]
    
    benchmark = SpatialDisambiguationBenchmark()
    records = benchmark.sweep(
        model_class=AsymmetricHopfieldNetwork,
        model_kwargs={"n_features": 450},
        configs=configs,
        conditions=conditions,
        n_trials=2,
        fit_epochs=1
    )
    
    # 2 configs x 2 conditions = 4 records
    assert len(records) == 4
    for record in records:
        assert "branch_accuracy" in record
        assert "condition" in record
        assert "shared_fraction" in record

def test_tmaze_disambiguation_generator():
    from memval.generators.tmaze_disambiguation import TMazeDisambiguationGenerator
    
    generator = TMazeDisambiguationGenerator(seed=42)
    res = generator.generate(
        sequence_length=20,
        stem_fraction=0.50,
        n_place_cells=100,
        n_odour_dims=10,
        seed=42
    )
    
    assert res["input_A"].shape == (20, 110)
    assert res["input_B"].shape == (20, 110)
    assert res["shared_start"] == 0
    assert res["shared_end"] > 0
    assert res["zone_start"] == 0
    assert res["zone_end"] == res["shared_end"]
    
    # Test evaluation integration
    benchmark = SpatialDisambiguationBenchmark()
    model = AsymmetricHopfieldNetwork(n_features=110)
    metrics = benchmark.evaluate(
        model=model,
        route_pair=res,
        condition="full",
        n_trials=2,
        fit_epochs=1
    )
    assert "branch_accuracy" in metrics
    assert 0.0 <= metrics["branch_accuracy"] <= 1.0
