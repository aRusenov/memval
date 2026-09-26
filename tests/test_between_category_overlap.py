"""The second overlap dial: ``SymbolicEncoder(between_category_cosine=...)``.

``category_variance`` sets the within-category cosine; this sets the cosine
between category prototypes, exactly. The default (``None``) must stay
bit-identical to the construction every saved result was produced with, and
the controlled path must obey the laws in ``docs/encoder_design.md`` §6.4:
within ``rho = 1/(1 + d sigma^2)``, between-word ``rho * rho_b``.
"""
import numpy as np
import pytest

from memval.benchmarks.continual_chain import build_chain_material
from memval.encoders.symbolic import SymbolicEncoder

D = 100


def _vocab(n_cats=6, n_words=40):
    return {f"w{c}_{i}": f"cat{c}" for c in range(n_cats) for i in range(n_words)}


def _legacy_embeddings(vocab, sigma, seed):
    """The construction as it was before the argument existed."""
    rng = np.random.default_rng(seed)
    cats = sorted(set(vocab.values()))
    C = {}
    for c in cats:
        v = rng.standard_normal(D)
        C[c] = v / np.linalg.norm(v)
    E = np.zeros((len(vocab), D))
    for i, (w, c) in enumerate(vocab.items()):
        v = C[c] + rng.standard_normal(D) * sigma
        E[i] = v / np.linalg.norm(v)
    return E


def _cos_stats(enc, vocab):
    E = enc.embeddings
    cat = np.array([vocab[w] for w in enc.idx_to_word])
    G = E @ E.T
    same = cat[:, None] == cat[None, :]
    off = ~np.eye(len(cat), dtype=bool)
    return float(G[same & off].mean()), float(G[~same].mean())


def test_default_is_bit_identical_to_the_legacy_construction():
    vocab = _vocab(4, 5)
    for sigma, seed in ((0.2, 42), (0.05, 7)):
        enc = SymbolicEncoder(vocab, embedding_dim=D, category_variance=sigma, seed=seed)
        assert enc.between_category_cosine is None
        assert np.array_equal(enc.embeddings, _legacy_embeddings(vocab, sigma, seed))


@pytest.mark.parametrize("rho_b", [0.0, 0.3, 0.75, 0.95])
def test_prototype_cosine_is_exact(rho_b):
    enc = SymbolicEncoder(_vocab(6, 3), embedding_dim=D, category_variance=0.2, seed=1,
                          between_category_cosine=rho_b)
    P = np.stack([enc.category_vectors[c] for c in enc.categories])
    G = P @ P.T
    assert np.allclose(np.diag(G), 1.0, atol=1e-12)
    off = G[~np.eye(len(P), dtype=bool)]
    assert np.allclose(off, rho_b, atol=1e-10)


@pytest.mark.parametrize("sigma,rho_b", [(0.2, 0.5), (0.1, 0.5), (0.05, 0.3), (0.2, 0.0)])
def test_word_level_laws(sigma, rho_b):
    vocab = _vocab(6, 40)
    enc = SymbolicEncoder(vocab, embedding_dim=D, category_variance=sigma, seed=3,
                          between_category_cosine=rho_b)
    within, between = _cos_stats(enc, vocab)
    rho = 1.0 / (1.0 + D * sigma ** 2)
    assert within == pytest.approx(rho, abs=0.02)
    assert between == pytest.approx(rho * rho_b, abs=0.02)


def test_within_law_is_unchanged_by_the_between_dial():
    vocab = _vocab(6, 40)
    w0, _ = _cos_stats(SymbolicEncoder(vocab, D, 0.2, seed=5, between_category_cosine=0.0), vocab)
    w1, _ = _cos_stats(SymbolicEncoder(vocab, D, 0.2, seed=5, between_category_cosine=0.9), vocab)
    assert w0 == pytest.approx(w1, abs=0.02)


def test_sweep_over_rho_b_moves_only_the_prototypes():
    """Same seed, different rho_b: every word keeps its private deviation."""
    vocab = _vocab(5, 8)
    a = SymbolicEncoder(vocab, D, 0.2, seed=11, between_category_cosine=0.0)
    b = SymbolicEncoder(vocab, D, 0.2, seed=11, between_category_cosine=0.8)

    def residuals(enc):
        out = []
        for w in enc.idx_to_word:
            e = enc.embeddings[enc.word_to_idx[w]]
            c = enc.category_vectors[vocab[w]]
            r = e - (e @ c) * c
            out.append(r / np.linalg.norm(r))
        return np.array(out)

    ra, rb = residuals(a), residuals(b)
    cos = np.sum(ra * rb, axis=1)
    # The prototype direction itself moves with rho_b, so the projection is not
    # identical -- but the private deviation is the same draw for every word.
    assert cos.mean() > 0.95 and cos.min() > 0.85


def test_between_word_cosine_cannot_exceed_within():
    vocab = _vocab(6, 40)
    within, between = _cos_stats(
        SymbolicEncoder(vocab, D, 0.2, seed=2, between_category_cosine=0.95), vocab)
    assert between < within


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_rejects_cosine_outside_the_unit_interval(bad):
    with pytest.raises(ValueError):
        SymbolicEncoder(_vocab(3, 2), D, 0.2, seed=0, between_category_cosine=bad)


def test_rejects_more_categories_than_the_dimension_can_hold():
    with pytest.raises(ValueError):
        SymbolicEncoder(_vocab(5, 2), embedding_dim=5, category_variance=0.2, seed=0,
                        between_category_cosine=0.2)


def test_chain_material_defaults_reproduce_the_section():
    vocab = {f"{c}{i}": c for c in "abcdefg" for i in range(7)}
    m = build_chain_material(vocab, n_tasks=6, seq_len=5)
    assert list(m["sequences"]) == list("abcdef")
    assert m["sequences"]["a"] == ["a0", "a1", "a2", "a3", "a4"]
    ref = SymbolicEncoder(m["vocab"], embedding_dim=100, category_variance=0.2, seed=42)
    assert np.array_equal(m["encoder"].embeddings, ref.embeddings)
    g = m["geometry"]
    assert g["between_category_cosine"] is None
    assert g["within_cos_law"] == pytest.approx(0.2)
    assert abs(g["between_word_cos_measured"]) < 0.1


def test_chain_material_reports_the_controlled_geometry():
    vocab = {f"{c}{i}": c for c in "abcdefg" for i in range(7)}
    m = build_chain_material(vocab, n_tasks=6, seq_len=5, between_category_cosine=0.5)
    g = m["geometry"]
    assert g["prototype_cos_min"] == pytest.approx(0.5, abs=1e-9)
    assert g["prototype_cos_max"] == pytest.approx(0.5, abs=1e-9)
    assert g["between_word_cos_law"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        build_chain_material(vocab, n_tasks=8, seq_len=5)


# --------------------------------------------------------------------------
# The intrusion probe: errors are classified by source, not just counted
# --------------------------------------------------------------------------

from memval.benchmarks.continual_chain import probe_intrusions  # noqa: E402
from memval.encoders.symbolic import SymbolicDecoder  # noqa: E402


class _Answers:
    """Recalls whatever word the test maps each cue to (its exact embedding)."""

    def __init__(self, encoder, mapping):
        self.enc, self.mapping, self.resets = encoder, mapping, 0

    def reset_context(self):
        self.resets += 1

    def predict_next(self, cue, current_context=None):
        w = self.mapping[self.enc.decode_cue(cue)]
        return self.enc.encode([w])[0]


def _chain_fixture():
    vocab = {f"{c}{i}": c for c in "abc" for i in range(3)}
    m = build_chain_material(vocab, n_tasks=3, seq_len=3, between_category_cosine=0.0)
    enc = m["encoder"]
    dec = SymbolicDecoder(enc)
    enc.decode_cue = lambda v: dec.decode(v, top_k=1)[0]        # test-only helper
    # the true successor of every cue
    truth = {ws[t]: ws[t + 1] for ws in m["sequences"].values() for t in range(2)}
    return m, enc, dec, truth


def test_intrusion_probe_all_correct():
    m, enc, dec, truth = _chain_fixture()
    out = probe_intrusions(_Answers(enc, truth), m["sequences"], enc, dec)
    assert out["n_probes"] == 6
    assert out["outcome_counts"]["correct"] == 6
    assert sum(out["competitor_counts"].values()) == 6
    assert all(r["margin"] > 0 for r in out["records"])


def test_intrusion_probe_classifies_the_source_of_the_error():
    m, enc, dec, truth = _chain_fixture()
    ans = dict(truth)
    ans["a0"] = "c1"      # task a recalls a later task's item
    ans["c0"] = "a2"      # task c recalls an earlier task's item
    ans["b0"] = "b2"      # task b recalls its own item two ahead (offset +1 from target b1)
    out = probe_intrusions(_Answers(enc, ans), m["sequences"], enc, dec)
    oc = out["outcome_counts"]
    assert (oc["correct"], oc["within_task"], oc["cross_earlier"], oc["cross_later"]) == (3, 1, 1, 1)
    assert out["within_offsets"] == {"1": 1}
    wrong = [r for r in out["records"] if r["outcome"] != "correct"]
    assert all(r["margin"] < 0 for r in wrong), "a wrong decode must carry a negative margin"
    by_cue = {r["cue"]: r for r in out["records"]}
    assert by_cue["a0"]["competitor"] == "c1" and by_cue["a0"]["competitor_class"] == "later"
    assert by_cue["c0"]["competitor_class"] == "earlier"


def test_intrusion_probe_resets_context_once_per_task():
    m, enc, dec, truth = _chain_fixture()
    model = _Answers(enc, truth)
    probe_intrusions(model, m["sequences"], enc, dec)
    assert model.resets == 3


# --------------------------------------------------------------------------
# The usage-rate phase
# --------------------------------------------------------------------------

from memval.benchmarks.continual_chain import spearman, usage_phase  # noqa: E402


class _Recorder:
    """Records every presentation; recall margin is whatever the test dictates."""

    def __init__(self, enc, margin_of):
        self.enc, self.margin_of, self.fits = enc, margin_of, []

    def fit_sequence(self, emb, epochs=None, **kw):
        self.fits.append((int(round(float(emb[0].sum() * 1000))), epochs))

    def reset_context(self):
        pass

    def predict_next(self, cue, current_context=None):
        # Return the embedding of the true successor scaled toward or away from
        # it, so the margin is controlled per task.
        w = self.enc.idx_to_word[int(np.argmax(self.enc.embeddings @ cue))]
        task = w[0]
        nxt = self.enc.idx_to_word[(self.enc.word_to_idx[w] + 1) % len(self.enc.idx_to_word)]
        target = self.enc.encode([nxt])[0]
        return target if self.margin_of(task) > 0 else -target


def _usage_fixture():
    vocab = {f"{c}{i}": c for c in "abcd" for i in range(3)}
    m = build_chain_material(vocab, n_tasks=4, seq_len=3, between_category_cosine=0.0)
    return m


def test_spearman_is_rank_based_and_handles_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) > 0.9


def test_usage_phase_presents_by_rate_and_never_presents_rate_zero():
    m = _usage_fixture()
    model = _Recorder(m["encoder"], lambda t: 1.0)
    out = usage_phase(model, m["sequences"], m["embeddings"], m["encoder"],
                      rates=[1.0, 0.5, 0.0, 1.0], n_blocks=8, rng=np.random.default_rng(0))
    counts = out["counts"]
    assert counts[0] == 8 and counts[3] == 8 and counts[2] == 0
    assert 0 < counts[1] < 8
    assert len(out["margins"]) == 9 and len(model.fits) == sum(counts)


def test_usage_phase_explicit_schedule_is_honoured():
    m = _usage_fixture()
    sched = np.zeros((6, 4), dtype=bool)
    sched[[0, 1], 0] = True
    sched[[4, 5], 1] = True
    model = _Recorder(m["encoder"], lambda t: 1.0)
    out = usage_phase(model, m["sequences"], m["embeddings"], m["encoder"],
                      schedule=sched, rng=np.random.default_rng(0))
    assert out["counts"] == [2, 2, 0, 0]
    assert out["n_blocks"] == 6


def test_usage_phase_allocation_scores():
    """Margin dictated per task: a and b (used) positive, c and d (unused) negative
    -> retention tracks use, misallocation negative."""
    m = _usage_fixture()
    model = _Recorder(m["encoder"], lambda t: 1.0 if t in "ab" else -1.0)
    out = usage_phase(model, m["sequences"], m["embeddings"], m["encoder"],
                      rates=[1.0, 1.0, 0.0, 0.0], n_blocks=4, rng=np.random.default_rng(0))
    # counts are tied 4/4/0/0, so with 4 tasks the ceiling is 0.894
    assert out["spearman_margin_vs_use"] > 0.85
    assert out["misallocation"] < 0
    model = _Recorder(m["encoder"], lambda t: -1.0 if t in "ab" else 1.0)
    out = usage_phase(model, m["sequences"], m["embeddings"], m["encoder"],
                      rates=[1.0, 1.0, 0.0, 0.0], n_blocks=4, rng=np.random.default_rng(0))
    assert out["spearman_margin_vs_use"] < -0.85
    assert out["misallocation"] > 0


def test_usage_phase_refuses_ambiguous_or_malformed_input():
    m = _usage_fixture()
    model = _Recorder(m["encoder"], lambda t: 1.0)
    with pytest.raises(ValueError):
        usage_phase(model, m["sequences"], m["embeddings"], m["encoder"])
    with pytest.raises(ValueError):
        usage_phase(model, m["sequences"], m["embeddings"], m["encoder"], rates=[1.0, 0.5])
