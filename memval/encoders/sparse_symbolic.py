"""Sparse symbolic codes: k active units out of d, with optional category cores.

The counterpart to :class:`~memval.encoders.symbolic.SymbolicEncoder`, which
gives every item a *dense* Gaussian code. Here an item is a k-of-d sparse
pattern, which is the coding regime most of the hippocampal literature the
roster is drawn from actually assumes (dentate-gyrus style sparse, near-
orthogonal codes). The two encoders differ in what "overlap" means:

    dense   overlap = cosine of two Gaussian vectors, continuous, controlled by
            `category_variance` / `between_category_cosine`
    sparse  overlap = how many active units two items SHARE, quantised in units
            of 1/k, controlled by `category_core`

Construction
------------
Each category owns a **core**: ``n_core = round(active * category_core)`` units
shared by every item in it. Each item takes that core plus ``active - n_core``
private units drawn from outside the core, so every item has exactly ``active``
units on. With ``category_core = 0`` (or a category-free vocabulary) every code
is an independent k-subset.

Similarity laws (unit-normalised codes, so cosine = shared/k):

    within category   cos = category_core + (1 - category_core)^2 * k / d + O(.)
    between category  cos = k / d                              (chance overlap)

The private parts are drawn independently, so the within-category law is the
core fraction plus the chance overlap of the private remainders; at k << d the
second term is small and ``cos ~= category_core``. Both are reported measured,
beside the law, by :meth:`geometry`.

Codes are unit-normalised (value ``1/sqrt(k)`` on active units) because
``SymbolicDecoder`` treats ``embeddings`` as already normalised and every other
encoder in the suite is unit-norm. The *support* is what is sparse; scaling all
active units alike changes no cosine and no ranking.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np


class SparseSymbolicEncoder:
    """k-of-d sparse item codes, with an optional shared core per category.

    Interface-compatible with ``SymbolicEncoder`` (``encode``, ``embeddings``,
    ``word_to_idx``, ``idx_to_word``, ``embedding_dim``), so
    ``SymbolicDecoder`` and every probe in the symbolic pipeline take it
    unchanged.

    Args:
        vocab: list of words (no categories), or ``{word: category}``.
            **Iteration order is load-bearing** for reproducibility, exactly as
            in ``SymbolicEncoder``: one draw per word in this mapping's order,
            so pass an order-stable mapping (never one built from ``set()``).
        embedding_dim: number of units ``d``.
        active: units on per item, ``k``.
        category_core: fraction of an item's active units that are its
            category's shared code, in ``[0, 1]``. ``0.0`` gives independent
            sparse codes; ``0.5`` puts half of every item's units on a
            category signature. Ignored (must be 0) for a category-free vocab.
        seed: RNG seed.
    """

    def __init__(self, vocab: Union[List[str], Dict[str, str]], embedding_dim: int = 512,
                 active: int = 10, category_core: float = 0.0, seed: Optional[int] = None):
        if not (0 < active <= embedding_dim):
            raise ValueError(f"active must be in (0, {embedding_dim}], got {active}")
        if not (0.0 <= category_core <= 1.0):
            raise ValueError(f"category_core must be in [0, 1], got {category_core}")

        self.embedding_dim = int(embedding_dim)
        self.active = int(active)
        self.category_core = float(category_core)
        self.rng = np.random.default_rng(seed)

        if isinstance(vocab, list):
            if category_core > 0.0:
                raise ValueError("category_core > 0 needs a {word: category} vocabulary")
            vocab_dict = {w: w for w in vocab}
        else:
            vocab_dict = dict(vocab)
        self.vocab = vocab_dict
        self.categories = sorted(set(vocab_dict.values()))

        n_core = int(round(self.active * self.category_core))
        self.n_core = n_core
        n_private = self.active - n_core

        # One core per category, drawn first so the RNG stream is category-major.
        self.category_units: Dict[str, np.ndarray] = {}
        for cat in self.categories:
            self.category_units[cat] = (self.rng.choice(self.embedding_dim, size=n_core,
                                                        replace=False)
                                        if n_core else np.empty(0, dtype=int))

        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: List[str] = []
        self.word_units: Dict[str, np.ndarray] = {}
        self.embeddings = np.zeros((len(vocab_dict), self.embedding_dim))
        scale = 1.0 / np.sqrt(self.active)

        for idx, (word, cat) in enumerate(vocab_dict.items()):
            self.word_to_idx[word] = idx
            self.idx_to_word.append(word)
            core = self.category_units[cat]
            if n_private:
                pool = np.setdiff1d(np.arange(self.embedding_dim), core, assume_unique=False)
                private = self.rng.choice(pool, size=n_private, replace=False)
            else:
                private = np.empty(0, dtype=int)
            units = np.concatenate([core, private]).astype(int)
            self.word_units[word] = units
            self.embeddings[idx, units] = scale

    def encode(self, sequence: List[str]) -> np.ndarray:
        """Words to (seq_len, d) codes."""
        out = np.zeros((len(sequence), self.embedding_dim))
        for i, word in enumerate(sequence):
            if word not in self.word_to_idx:
                raise ValueError(f"Word '{word}' not in vocabulary.")
            out[i] = self.embeddings[self.word_to_idx[word]]
        return out

    def geometry(self) -> Dict[str, Any]:
        """Measured within / between category cosine, beside the laws."""
        E = self.embeddings
        G = E @ E.T
        cat = np.array([self.vocab[w] for w in self.idx_to_word])
        same = cat[:, None] == cat[None, :]
        off = ~np.eye(len(cat), dtype=bool)
        k, d = self.active, self.embedding_dim
        within = G[same & off]
        between = G[~same]
        core_law = self.category_core + (1.0 - self.category_core) ** 2 * k / d
        return {
            "embedding_dim": d, "active": k, "sparsity": k / d,
            "category_core": self.category_core, "n_core_units": self.n_core,
            "within_cos_law": float(core_law) if len(self.categories) < len(self.idx_to_word) else float("nan"),
            "within_cos_measured": float(within.mean()) if within.size else float("nan"),
            "between_cos_law": float(k / d),
            "between_cos_measured": float(between.mean()) if between.size else float("nan"),
            "max_offdiag_cos": float(G[off].max()) if off.any() else float("nan"),
        }
