"""The suite's embeddings must not depend on PYTHONHASHSEED.

`SymbolicEncoder` draws one RNG sample per word, in the vocabulary mapping's
iteration order. A seeded encoder is therefore only reproducible if the mapping
it is handed is order-stable, and `set()` is not: its iteration order depends on
`PYTHONHASHSEED`, which CPython randomises per process.

That is not hypothetical. `semantic_similarity` built its vocabulary as
`{w: vocab[w] for w in set(list_high + list_low)}` and was the one
non-reproducible section in the suite: seven runs of the *same* learning rule
gave epochs-to-criterion of 33/37/41/45/46/47/64 at the hardest rung -- a 1.9x
spread -- with the rung's own mean cosine wandering over 0.787-0.798. The metric
it feeds, `similarity_exposure_cost`, carries weight 0.40 of the Sequence
disambiguation capacity, so the noise was larger than most of the effects being
compared against it.

These tests run the real construction in fresh interpreters under different hash
seeds, because that is the only way to exercise the failure: within one process
the ordering is fixed and everything looks reproducible.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Hash seeds to compare. "0" disables randomisation entirely; the rest are
#: arbitrary and were enough to expose the original bug on the first try.
HASH_SEEDS = ("0", "1", "12345")


def _run_under_hash_seed(seed: str, snippet: str) -> str:
    """Execute `snippet` in a fresh interpreter with PYTHONHASHSEED=`seed`."""
    env = dict(os.environ, PYTHONHASHSEED=seed, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"child failed under PYTHONHASHSEED={seed}:\n{proc.stderr[-2000:]}")
    return proc.stdout.strip().splitlines()[-1]


def test_set_built_vocab_is_the_hazard_dict_fromkeys_is_not():
    """The mechanism, pinned: `set()` moves the embeddings, `dict.fromkeys` does not.

    Guards the *reason* for the fix. If a future refactor makes
    `SymbolicEncoder` order-independent this test's first assertion becomes
    false, and it should be deleted rather than worked around -- but until then,
    silently reintroducing `set()` at a call site must not look safe.
    """
    snippet = """
        import json, sys
        sys.path.insert(0, ".")
        import numpy as np
        from memval.encoders.symbolic import SymbolicEncoder
        vocab = json.load(open("data/vocab.json"))
        high = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']
        low = ['apple', 'cat', 'car', 'hammer', 'red', 'one', 'blue']

        def mean_cos(mapping):
            e = SymbolicEncoder(mapping, embedding_dim=100,
                                category_variance=0.05, seed=42).encode(high)
            n = e / np.linalg.norm(e, axis=1, keepdims=True)
            iu = np.triu_indices(len(e), k=1)
            return float((n @ n.T)[iu].mean())

        unstable = mean_cos({w: vocab[w] for w in set(high + low)})
        stable = mean_cos({w: vocab[w] for w in dict.fromkeys(high + low)})
        print(f"{unstable!r} {stable!r}")
    """
    unstable, stable = [], []
    for seed in HASH_SEEDS:
        u, s = _run_under_hash_seed(seed, snippet).split()
        unstable.append(float(u))
        stable.append(float(s))

    assert len(set(stable)) == 1, (
        f"dict.fromkeys must be hash-seed stable, got {stable}")
    assert len(set(unstable)) > 1, (
        "set()-built vocabularies were expected to vary across hash seeds but "
        f"did not ({unstable}). If SymbolicEncoder became order-independent, "
        "delete this test; do not relax it.")


@pytest.mark.parametrize("section,keys", [
    ("semantic_similarity",
     ("series.similarity_sweep.cosine_similarities",
      "series.similarity_sweep.epochs_to_criterion",
      "metrics.mrr_high_similarity")),
])
def test_symbolic_section_is_reproducible_across_hash_seeds(section, keys, tmp_path):
    """End-to-end: the section itself, twice, under different hash seeds.

    Runs the real pipeline rather than re-deriving its vocabulary here, so a
    future section that reintroduces an unordered mapping is caught by the thing
    that actually matters -- the numbers it writes out.
    """
    snippet = f"""
        import json, sys, os
        sys.path.insert(0, ".")
        import matplotlib
        matplotlib.use("Agg")
        from memval.models.baselines import AsymmetricHopfieldNetwork
        from memval.benchmarks.symbolic_pipeline import run_symbolic_pipeline
        out = {str(tmp_path)!r} + "/" + os.environ["PYTHONHASHSEED"]
        run_symbolic_pipeline(
            model_class=AsymmetricHopfieldNetwork,
            model_kwargs={{"n_epochs": 1, "learning_rate": 0.1, "seed": 42}},
            output_dir=out, n_trials=5, benchmarks=[{section!r}])
        d = json.load(open(out + "/AsymmetricHopfieldNetwork/symbolic/metrics.json"))
        picked = {{}}
        for path in {list(keys)!r}:
            node = d
            for part in path.split("."):
                node = node[part]
            picked[path] = node
        print(json.dumps(picked, sort_keys=True))
    """
    seen = {seed: _run_under_hash_seed(seed, snippet) for seed in HASH_SEEDS}
    distinct = set(seen.values())
    assert len(distinct) == 1, (
        f"{section} is not reproducible across PYTHONHASHSEED. A vocabulary "
        f"somewhere in it is built from an unordered collection -- see this "
        f"module's docstring. Per-seed output:\n"
        + "\n".join(f"  {s}: {v}" for s, v in seen.items()))
