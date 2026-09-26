"""``observe`` is delivered to every arm through the same call.

The base class provides a no-op, so a probe can walk any arm up a prefix
without checking a capability first. Two things must then hold: the no-op
really is a no-op on an arm without carried state, and every arm that
declares ``StatePrimeable`` actually overrides it (otherwise the declaration
would be satisfied by the base no-op and claim a capability that does nothing).
"""
import inspect

import numpy as np

import memval.models.baselines as baselines
from memval.models.base import HippocampalModel
from memval.models.capabilities import StatePrimeable
from memval.models.baselines import AsymmetricHopfieldNetwork
from memval.generators.bifurcating_route import BifurcatingRouteGenerator
from memval.benchmarks.spatial_disambiguation import SpatialDisambiguationBenchmark


def test_state_primeable_arms_override_observe():
    declared = [
        cls for _, cls in inspect.getmembers(baselines, inspect.isclass)
        if issubclass(cls, StatePrimeable) and issubclass(cls, HippocampalModel)
    ]
    assert declared, "no StatePrimeable arm found in memval.models.baselines"
    for cls in declared:
        assert cls.observe is not HippocampalModel.observe, (
            f"{cls.__name__} declares StatePrimeable but inherits the base no-op observe")


def test_observe_is_a_no_op_on_a_stateless_arm():
    rng = np.random.default_rng(0)
    X = rng.random((6, 8))
    m = AsymmetricHopfieldNetwork(n_features=8, n_epochs=2, learning_rate=0.1, seed=0)
    m.fit_sequence(X, epochs=2)
    W_before = m.W.copy()
    m.reset_context()
    cold = m.predict_next(X[3])
    m.reset_context()
    m.observe_sequence(X[:3])
    warm = m.predict_next(X[3])
    np.testing.assert_allclose(cold, warm)
    np.testing.assert_allclose(m.W, W_before)


def test_guidance_sweep_shapes_and_ranges():
    route = BifurcatingRouteGenerator(seed=1).generate(
        total_length=12, shared_fraction=0.5, shared_position=0.0,
        zone_fraction=1.0, zone_offset=0.0, n_place_cells=100, n_encounter_dims=4,
        balance_modalities=True, odour_scale=2.0, odour_on_suffix=True, seed=1,
    )
    se = route["shared_end"]
    bench = SpatialDisambiguationBenchmark()
    out = bench.guidance_sweep(
        lambda s: AsymmetricHopfieldNetwork(n_features=104, n_epochs=3,
                                            learning_rate=0.1, activation="relu", seed=s),
        route, fit_epochs=3, seeds=(0, 1),
    )
    recs = out["records"]
    assert out["n_seeds"] == 2
    assert out["probe_protocol"] == "observe_then_recall"
    assert [r["cue_point"] for r in recs] == list(range(se, 0, -1))
    assert [r["recall_starts_before_fork"] for r in recs] == list(range(0, se))
    for r in recs:
        assert 0.0 <= r["arm_accuracy"] <= 1.0
        assert r["positional_error"] >= 0.0
        assert np.isfinite(r["divergence_margin"])
        assert np.isfinite(r["arm_accuracy_sd"])
    # d = 0 has no free-run stem stretch, so its control is undefined.
    assert np.isnan(recs[0]["stem_error"])
    assert np.isfinite(recs[1]["stem_error"])
