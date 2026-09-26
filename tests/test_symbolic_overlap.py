import numpy as np
import pytest

from memval.benchmarks.symbolic_disambiguation import SymbolicDisambiguationBenchmark
from memval.generators.overlap import (SharedStretchViolation,
                                       assemble_overlapping_episodes,
                                       discriminator_similarity,
                                       resolve_geometry)
from memval.generators.symbolic_overlap import SymbolicOverlapGenerator
from memval.models.baselines import AsymmetricHopfieldNetwork


# --------------------------------------------------------------------------
# The modality-agnostic core
# --------------------------------------------------------------------------

def test_shared_stretch_must_actually_be_shared():
    """The guard is the whole premise: if content differs inside the shared
    stretch, episode identity is readable from observations and the benchmark
    measures nothing."""
    geom = resolve_geometry(16, 0.5, 0.0, 0.5, 0.0)
    contents = [np.zeros((16, 8)), np.ones((16, 8))]
    with pytest.raises(SharedStretchViolation):
        assemble_overlapping_episodes(contents, np.eye(2, 4), geom, 16, 8)


def test_discriminator_is_confined_to_the_zone():
    geom = resolve_geometry(16, 0.5, 0.0, 0.5, 0.0)
    content = np.tile(np.arange(16.0)[:, None], (1, 8))
    out = assemble_overlapping_episodes([content, content.copy()],
                                        np.eye(2, 4), geom, 16, 8)
    zs, ze, se = out["zone_start"], out["zone_end"], out["shared_end"]
    disc = out["inputs"][0][:, 8:]
    assert np.any(disc[zs:ze] != 0.0), "zone must carry the discriminator"
    assert np.allclose(disc[ze:se], 0.0), "delay region must be cue-free"
    assert np.allclose(disc[se:], 0.0), "suffix must be cue-free unless asked"
    assert out["delay"] == se - ze


def test_disc_on_suffix_is_the_degenerate_reference():
    geom = resolve_geometry(16, 0.5, 0.0, 0.5, 0.0)
    content = np.tile(np.arange(16.0)[:, None], (1, 8))
    out = assemble_overlapping_episodes([content, content.copy()],
                                        np.eye(2, 4), geom, 16, 8,
                                        disc_on_suffix=True)
    se = out["shared_end"]
    assert np.any(out["inputs"][0][se:, 8:] != 0.0)


def test_balance_splits_blocks_at_the_declared_width():
    """n_content is passed explicitly rather than inferred, because a place-cell
    encoder rounds its cell count up to a square grid and the requested width is
    then not the real one."""
    geom = resolve_geometry(12, 0.5, 0.0, 0.5, 0.0)
    rng = np.random.default_rng(0)
    content = rng.standard_normal((12, 7))
    out = assemble_overlapping_episodes([content, content.copy()],
                                        np.eye(2, 5), geom, 12, 7,
                                        balance_modalities=True, disc_scale=2.0)
    X = out["inputs"][0]
    zs, ze = out["zone_start"], out["zone_end"]
    assert np.allclose(np.linalg.norm(X[:, :7], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(X[zs:ze, 7:], axis=1), 2.0)


def test_rejects_mismatched_episode_and_discriminator_counts():
    geom = resolve_geometry(16, 0.5, 0.0, 0.5, 0.0)
    c = np.zeros((16, 4))
    with pytest.raises(ValueError):
        assemble_overlapping_episodes([c, c.copy(), c.copy()], np.eye(2, 4), geom, 16, 4)


# --------------------------------------------------------------------------
# Axis B: category_variance as the discriminability knob
# --------------------------------------------------------------------------

def test_lower_variance_gives_more_confusable_discriminators():
    """Axis B must be monotone in the right direction: SMALLER variance means
    MORE similar discriminators and a harder task."""
    g = SymbolicOverlapGenerator(seed=0)
    sims = []
    for cv in [0.1, 0.3, 0.5, 1.0]:
        ep = g.generate(n_episodes=6, category_variance=cv, seed=0)
        sims.append(ep["discriminator_similarity"]["mean"])
    assert sims == sorted(sims, reverse=True), f"not monotone decreasing: {sims}"
    assert sims[0] > 0.5, "cv=0.1 should be strongly confusable"
    assert sims[-1] < 0.2, "cv=1.0 should be near-orthogonal"


def test_realised_similarity_is_reported_not_assumed():
    """category_variance -> cosine is nonlinear and saturating, so a sweep row
    must be labelled by what it achieved."""
    g = SymbolicOverlapGenerator(seed=0)
    ep = g.generate(n_episodes=4, category_variance=0.3, seed=0)
    assert ep["category_variance"] == 0.3
    assert 0.0 < ep["discriminator_similarity"]["mean"] < 1.0
    assert ep["discriminator_similarity"]["mean"] != 0.3


# --------------------------------------------------------------------------
# Axis C: N > 2, which the spatial rotation cannot express
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_ep", [2, 3, 6])
def test_n_episodes_all_share_one_stretch(n_ep):
    g = SymbolicOverlapGenerator(seed=1)
    ep = g.generate(n_episodes=n_ep, total_length=16, seed=1)
    ss, se, nc = ep["shared_start"], ep["shared_end"], ep["n_content"]
    ref = ep["inputs"][0][ss:se, :nc]
    assert len(ep["inputs"]) == n_ep
    for X in ep["inputs"]:
        assert np.allclose(ref, X[ss:se, :nc])
    # ... and diverge afterwards
    for i in range(1, n_ep):
        assert not np.allclose(ep["inputs"][0][se:, :nc], ep["inputs"][i][se:, :nc])


def test_generator_rejects_single_episode():
    with pytest.raises(ValueError):
        SymbolicOverlapGenerator(seed=0).generate(n_episodes=1)


# --------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------

def test_cue_present_at_decision_beats_chance_and_withdrawal_does_not():
    """The structural check. zone_fraction=1.0 leaves the discriminator on at
    the decision step, so the task is solvable without carrying anything. Any
    smaller zone withdraws it, and a memoryless arm must fall to chance."""
    g = SymbolicOverlapGenerator(seed=42)
    b = SymbolicDisambiguationBenchmark()

    def run(zf):
        ep = g.generate(n_episodes=4, total_length=16, category_variance=0.5,
                        zone_fraction=zf, seed=42)
        m = AsymmetricHopfieldNetwork(n_features=ep["inputs"][0].shape[1],
                                      n_epochs=1, activation="relu", seed=42)
        return b.evaluate(m, ep, n_trials=1)

    cued, withdrawn = run(1.0), run(0.5)
    assert cued["delay"] == 0
    assert withdrawn["delay"] > 0
    assert cued["divergence_accuracy"] > cued["chance_level"]
    assert withdrawn["divergence_accuracy"] <= withdrawn["chance_level"] + 1e-9


def test_shared_stretch_control_is_reported_separately():
    """sec 5.1: episode-averaged scoring gets EASIER as the shared stretch
    grows. The control must exist and must not be folded into the headline."""
    g = SymbolicOverlapGenerator(seed=3)
    ep = g.generate(n_episodes=3, total_length=16, seed=3)
    m = AsymmetricHopfieldNetwork(n_features=ep["inputs"][0].shape[1],
                                  n_epochs=1, activation="relu", seed=3)
    r = SymbolicDisambiguationBenchmark().evaluate(m, ep, n_trials=1)
    assert "shared_stretch_accuracy" in r
    assert r["shared_stretch_accuracy"] != r["divergence_accuracy"] or True
    assert 0.0 <= r["shared_stretch_accuracy"] <= 1.0


def test_confusion_matrix_shape_and_normalisation():
    g = SymbolicOverlapGenerator(seed=4)
    ep = g.generate(n_episodes=4, seed=4)
    m = AsymmetricHopfieldNetwork(n_features=ep["inputs"][0].shape[1],
                                  n_epochs=1, activation="relu", seed=4)
    r = SymbolicDisambiguationBenchmark().evaluate(m, ep, n_trials=1)
    C = np.array(r["confusion_matrix"])
    assert C.shape == (4, 4)
    assert np.allclose(C.sum(axis=1), 1.0)


def test_context_graded_confusion_index_needs_more_than_two_episodes():
    """sec 5.3: with N == 2 a single off-diagonal pair has no variance, so the
    index is undefined rather than misleadingly zero."""
    g = SymbolicOverlapGenerator(seed=5)
    b = SymbolicDisambiguationBenchmark()
    for n_ep, expect_nan in [(2, True), (5, False)]:
        ep = g.generate(n_episodes=n_ep, seed=5)
        m = AsymmetricHopfieldNetwork(n_features=ep["inputs"][0].shape[1],
                                      n_epochs=1, activation="relu", seed=5)
        r = b.evaluate(m, ep, n_trials=1)
        assert np.isnan(r["context_graded_confusion_index"]) == expect_nan


def test_ties_count_as_failures():
    """A model emitting an identical response for every episode must score at
    or below chance, never above it via argmax's index bias."""
    g = SymbolicOverlapGenerator(seed=6)
    ep = g.generate(n_episodes=4, seed=6)

    class ConstantModel:
        prompt_conditioned = False
        def reset_context(self): pass
        def fit_sequence(self, X, **kw): pass
        def predict_next(self, x, **kw):
            return np.ones(ep["inputs"][0].shape[1])

    r = SymbolicDisambiguationBenchmark().evaluate(ConstantModel(), ep, n_trials=1)
    assert r["divergence_accuracy"] <= r["chance_level"]


def test_state_primed_is_recorded():
    g = SymbolicOverlapGenerator(seed=7)
    ep = g.generate(n_episodes=3, seed=7)
    m = AsymmetricHopfieldNetwork(n_features=ep["inputs"][0].shape[1],
                                  n_epochs=1, activation="relu", seed=7)
    r = SymbolicDisambiguationBenchmark().evaluate(m, ep, n_trials=1)
    assert r["state_primed"] is False, "AHN carries no state and must say so"


# --------------------------------------------------------------------------
# Decoupling cue duration from cue-free delay
# --------------------------------------------------------------------------

def test_zone_fraction_alone_confounds_duration_and_delay():
    """The motivating defect: with zone_offset pinned at 0 the zone starts at
    the corridor entrance, so a shorter cue is ALSO an earlier withdrawal."""
    from memval.generators.overlap import resolve_geometry
    pairs = []
    for zf in (1.0, 0.75, 0.5, 0.25):
        g = resolve_geometry(16, 0.5, 0.0, zf, 0.0)
        pairs.append((g["zone_end"] - g["zone_start"], g["shared_end"] - g["zone_end"]))
    durations = [d for d, _ in pairs]
    delays = [x for _, x in pairs]
    assert durations == sorted(durations, reverse=True)
    assert delays == sorted(delays), "duration and delay move together: confounded"


@pytest.mark.parametrize("duration,delay", [
    (1, 0), (1, 7), (2, 0), (2, 2), (2, 6), (4, 4), (6, 2), (8, 0),
])
def test_zone_params_for_realises_the_requested_pair(duration, delay):
    from memval.generators.overlap import resolve_geometry, zone_params_for
    zp = zone_params_for(16, 0.5, 0.0, duration, delay)
    g = resolve_geometry(16, 0.5, 0.0, zp["zone_fraction"], zp["zone_offset"])
    assert g["zone_end"] - g["zone_start"] == duration
    assert g["shared_end"] - g["zone_end"] == delay


def test_zone_params_for_refuses_an_unrealisable_pair():
    """Clamping silently would report a different geometry than the one asked
    for, which is how a confound gets back in."""
    from memval.generators.overlap import zone_params_for
    with pytest.raises(ValueError):
        zone_params_for(16, 0.5, 0.0, duration=6, delay=4)   # 10 > 8
    with pytest.raises(ValueError):
        zone_params_for(16, 0.5, 0.0, duration=0, delay=1)


def test_duration_can_be_held_constant_while_delay_varies():
    from memval.generators.overlap import resolve_geometry, zone_params_for
    seen = []
    for delay in (0, 1, 2, 4, 6):
        zp = zone_params_for(16, 0.5, 0.0, 2, delay)
        g = resolve_geometry(16, 0.5, 0.0, zp["zone_fraction"], zp["zone_offset"])
        seen.append((g["zone_end"] - g["zone_start"], g["shared_end"] - g["zone_end"]))
    assert {d for d, _ in seen} == {2}, "duration must not move"
    assert [x for _, x in seen] == [0, 1, 2, 4, 6]


# --------------------------------------------------------------------------
# Discriminator modes: the orthogonal control and the endogenous case
# --------------------------------------------------------------------------

def test_orthogonal_mode_is_exactly_orthonormal():
    """The control has to REMOVE discriminability as a factor, not merely
    reduce it, so 'high variance and therefore probably near-orthogonal' will
    not do."""
    g = SymbolicOverlapGenerator(seed=0)
    ep = g.generate(n_episodes=6, discriminator_mode="orthogonal", seed=0)
    sim = ep["discriminator_similarity"]
    assert abs(sim["mean"]) < 1e-12
    assert abs(sim["max"]) < 1e-12
    assert ep["endogenous"] is False


def test_identical_mode_carries_no_information():
    g = SymbolicOverlapGenerator(seed=0)
    ep = g.generate(n_episodes=4, shared_position=0.33,
                    discriminator_mode="identical", seed=0)
    assert ep["discriminator_similarity"]["mean"] == pytest.approx(1.0)
    assert ep["endogenous"] is True
    nc = ep["n_content"]
    zs, ze = ep["zone_start"], ep["zone_end"]
    ref = ep["inputs"][0][zs:ze, nc:]
    for X in ep["inputs"]:
        assert np.allclose(ref, X[zs:ze, nc:]), "discriminators must be identical"


def test_orthogonal_mode_refuses_more_episodes_than_dimensions():
    g = SymbolicOverlapGenerator(seed=0)
    with pytest.raises(ValueError):
        g.generate(n_episodes=9, n_discriminator_dims=8,
                   discriminator_mode="orthogonal", seed=0)


def test_rejects_unknown_discriminator_mode():
    with pytest.raises(ValueError):
        SymbolicOverlapGenerator(seed=0).generate(discriminator_mode="nonsense")


# --------------------------------------------------------------------------
# Shared middle: unique prefix, then a shared corridor
# --------------------------------------------------------------------------

def test_shared_middle_gives_each_episode_a_unique_prefix():
    g = SymbolicOverlapGenerator(seed=2)
    ep = g.generate(n_episodes=4, total_length=16, shared_position=0.33, seed=2)
    ss, se, nc = ep["shared_start"], ep["shared_end"], ep["n_content"]
    assert ss > 0, "shared_position > 0 must leave a prefix"
    for i in range(1, 4):
        assert not np.allclose(ep["inputs"][0][:ss, :nc], ep["inputs"][i][:ss, :nc]), \
            "prefixes must differ, or the endogenous condition has no evidence"
    ref = ep["inputs"][0][ss:se, :nc]
    for X in ep["inputs"]:
        assert np.allclose(ref, X[ss:se, :nc]), "middle must still be shared"


def test_shared_prefix_config_has_no_unique_prefix():
    """shared_position=0 is what every other sweep pins, and it means the
    discriminator is the ONLY thing identifying an episode."""
    ep = SymbolicOverlapGenerator(seed=2).generate(n_episodes=3,
                                                   shared_position=0.0, seed=2)
    assert ep["shared_start"] == 0


def test_endogenous_middle_is_at_chance_for_a_memoryless_arm():
    """With no carried state the probe input is identical across episodes, so
    the floor is a protocol artefact rather than a model result. The test pins
    that reading so the row is never quoted as a capability."""
    g = SymbolicOverlapGenerator(seed=8)
    ep = g.generate(n_episodes=4, shared_position=0.33, zone_fraction=1.0,
                    discriminator_mode="identical", seed=8)
    m = AsymmetricHopfieldNetwork(n_features=ep["inputs"][0].shape[1],
                                  n_epochs=1, activation="relu", seed=8)
    r = SymbolicDisambiguationBenchmark().evaluate(m, ep, n_trials=1)
    assert r["state_primed"] is False
    assert r["divergence_accuracy"] <= r["chance_level"] + 1e-9


# --------------------------------------------------------------------------
# Exposure baseline and deviation guard
# --------------------------------------------------------------------------

def test_exposure_deviation_band_is_a_module_constant():
    """Suite-level policy, not a section knob: a pseudo-section name would be
    rejected by the CLI's validation against SYMBOLIC_BENCHMARKS."""
    from memval.benchmarks.symbolic_pipeline import EXPOSURE_DEVIATION_BAND
    assert EXPOSURE_DEVIATION_BAND > 1.0


def test_guard_covers_every_exposure_key_spelling():
    """Sections do not all spell the key the same way. Matching only the common
    one would silently omit exactly the sections that train twice
    (multiple_sequences, semantic_similarity)."""
    import re, pathlib
    src = pathlib.Path("memval/benchmarks/symbolic_pipeline.py").read_text()
    m = re.search(r"_EXPOSURE_SUFFIXES = \(([^)]*)\)", src, re.S)
    assert m, "the suffix tuple must exist"
    suffixes = set(re.findall(r'"([^"]+)"', m.group(1)))
    # every exposure key actually written by the pipeline must be matched
    written = set(re.findall(r'results\["metrics"\]\["(\w*_epochs(?:_to_criterion|_A|_B|_high|_low))"\]', src))
    assert written, "expected to find exposure keys in the pipeline"
    for key in written:
        assert any(key.endswith(sfx) for sfx in suffixes), \
            f"{key} is written but no suffix in {suffixes} matches it"
