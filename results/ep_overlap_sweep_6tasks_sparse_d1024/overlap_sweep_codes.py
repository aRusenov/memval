"""Run bin/continual_chain_overlap_sweep.py with a different CODE, same dials and protocol.

    python overlap_sweep_codes.py sparse <out>   exact k-of-d sparse codes
    python overlap_sweep_codes.py dense  <out>   dense control at the same d

The sweep's dials are dense-encoder parameters (sigma on the d=100 law, rho_b). Each
rung is translated to its TARGET cosines -- within rho = 1/(1+100 sigma^2), between
word cosine rho*rho_b -- and the code is built to hit them:

  sparse  k=K active units at 1/sqrt(K). g = round(K*between) units shared by EVERY
          item, c = round(K*rho) - g units shared within a task, the rest private.
          All blocks drawn WITHOUT replacement from one permutation of d, so cosines are
          exact: within (g+c)/K, between g/K, and no chance-overlap floor.
  dense   SymbolicEncoder at d=D with sigma rescaled by sqrt(100/D): identical rho law,
          identical rho_b, only the dimension changes from the stored d=100 sweep.

Protocol, scoring, seeds, plotting: the sweep's own, untouched. No repo file is edited.
"""
import json, os, shutil, sys
import numpy as np
ROOT = "/Users/atanas/Documents/workspace/memval"
sys.path.insert(0, ROOT + "/bin"); sys.path.insert(0, ROOT)
import continual_chain_overlap_sweep as sw
from memval.encoders import SymbolicDecoder

MODE, OUT = sys.argv[1], sys.argv[2]
D, K, REF_D = 1024, 20, 100
assert MODE in ("sparse", "dense")


class ExactSparseEncoder:
    """The minimal encoder interface the chain uses: encode / embeddings / idx_to_word."""
    def __init__(self, units_by_word, d, k):
        self.embedding_dim = d
        self.idx_to_word = list(units_by_word)
        self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}
        self.embeddings = np.zeros((len(self.idx_to_word), d))
        for i, w in enumerate(self.idx_to_word):
            self.embeddings[i, units_by_word[w]] = 1.0 / np.sqrt(k)
    def encode(self, words):
        return np.stack([self.embeddings[self.word_to_idx[w]] for w in words])


def _select(vocab, n_tasks, seq_len):
    """Task/word selection copied from build_chain_material so the material is the same words."""
    by = {}
    for w, c in vocab.items(): by.setdefault(c, []).append(w)
    cats = [c for c in sorted(by) if len(by[c]) >= seq_len][:n_tasks]
    seqs = {c: sorted(by[c])[:seq_len] for c in cats}
    return cats, seqs, {w: c for c, ws in seqs.items() for w in ws}


def sparse_material(vocab, n_tasks, seq_len, *, embedding_dim, category_variance,
                    between_category_cosine=None, seed=42):
    cats, seqs, chain_vocab = _select(vocab, n_tasks, seq_len)
    rho = 1.0 / (1.0 + REF_D * category_variance ** 2)
    between = rho * (between_category_cosine or 0.0)
    g = int(round(K * between)); c = max(0, int(round(K * rho)) - g); p = K - g - c
    need = g + len(cats) * c + len(chain_vocab) * p
    assert need <= D, f"need {need} units > d={D}"
    perm = np.random.default_rng(seed).permutation(D); pos = g
    shared = perm[:g]; core = {}
    for cat in cats: core[cat] = perm[pos:pos + c]; pos += c
    units = {}
    for w, cat in chain_vocab.items():
        units[w] = np.concatenate([shared, core[cat], perm[pos:pos + p]]).astype(int); pos += p
    enc = ExactSparseEncoder(units, D, K); dec = SymbolicDecoder(enc)
    E = enc.embeddings; G = E @ E.T
    t = np.array([chain_vocab[w] for w in enc.idx_to_word])
    same = t[:, None] == t[None, :]; off = ~np.eye(len(t), dtype=bool)
    geometry = {"code": "sparse-exact", "embedding_dim": D, "active": K,
                "n_shared": g, "n_core": c, "n_private": p,
                "category_variance": float(category_variance),
                "between_category_cosine": between_category_cosine,
                "target_within": rho, "target_between": between,
                "within_cos_law": (g + c) / K, "within_cos_measured": float(G[same & off].mean()),
                "between_word_cos_law": g / K, "between_word_cos_measured": float(G[~same].mean()),
                "between_word_cos_sd": float(G[~same].std()),
                "prototype_cos_min": float("nan"), "prototype_cos_max": float("nan")}
    return {"sequences": seqs, "vocab": chain_vocab, "encoder": enc, "decoder": dec,
            "embeddings": {cc: enc.encode(ws) for cc, ws in seqs.items()}, "geometry": geometry}


_orig = sw.build_chain_material
def dense_material(vocab, n_tasks, seq_len, *, embedding_dim, category_variance,
                   between_category_cosine=None, seed=42):
    m = _orig(vocab, n_tasks, seq_len, embedding_dim=D,
              category_variance=category_variance * np.sqrt(REF_D / D),
              between_category_cosine=between_category_cosine, seed=seed)
    m["geometry"]["code"] = "dense"
    return m


sw.build_chain_material = sparse_material if MODE == "sparse" else dense_material
TAG = (f"SPARSE codes, k={K} of d={D}, exact allocation" if MODE == "sparse"
       else f"DENSE control, d={D}")
for name in ("plot_retention_matrix_grid", "plot_retention_sweep_axes", "plot_intrusion_breakdown"):
    f = getattr(sw, name)
    setattr(sw, name, (lambda f: lambda *a, title="", **kw: f(*a, title=f"{title}\n{TAG}", **kw))(f))

if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "--dry":
        vocab = sw.load_vocab()
        for label, s, rb in ([("between", 0.2, r) for r in (0, .25, .5, .75, .95)]
                             + [("within", s, 0.0) for s in (.05, .1, .2, .5, 1.0)]):
            g = sw.build_chain_material(vocab, 6, 5, embedding_dim=100, category_variance=s,
                                        between_category_cosine=rb, seed=42)["geometry"]
            print(f"{MODE:<6} {label:<7} sigma={s:<5} rho_b={rb:<5} within {g['within_cos_measured']:.3f} "
                  f"(law {g['within_cos_law']:.3f})  between {g['between_word_cos_measured']:+.4f} "
                  f"(law {g['between_word_cos_law']:.3f})")
        sys.exit(0)
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(__file__, os.path.join(OUT, "overlap_sweep_codes.py"))
    json.dump({"mode": MODE, "embedding_dim": D, "active": K if MODE == "sparse" else None,
               "reference_dim_for_sigma_law": REF_D, "driver": "overlap_sweep_codes.py"},
              open(os.path.join(OUT, "driver_config.json"), "w"), indent=1)
    sys.argv = [sys.argv[0], "--model", "original_eqprop", "--out", OUT]
    sw.main()
