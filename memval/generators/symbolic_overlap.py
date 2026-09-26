"""Symbolic overlapping sequences: the Agster paradigm.

N item sequences that share a middle stretch and diverge afterwards, with a
context block as the discriminator. This is the symbolic arm of
`docs/disambiguation_design.md` (build order step 3), and it exists because the
spatial modality cannot express the two axes that matter most:

**Axis B (discriminability).** `BifurcatingRouteGenerator` grades discriminator
similarity by rotating a one-hot vector: A is ``e_0`` and B is
``cos(t) e_0 + sin(t) e_1``. That hits its requested cosine exactly, but it uses
two dimensions no matter how many are allocated, and it is structurally capped
at *two* discriminators. It is also the hand-tuned cosine the design note warns
against: a scalar with no generative model behind it.

`SymbolicEncoder` grades similarity generatively instead. Items in a category
are a shared base vector plus scaled noise, so ``category_variance`` produces a
whole *family* of confusable discriminators at a controlled mutual similarity.
Measured over 6 discriminators in one category (embedding_dim=64)::

    category_variance=0.1   mean pairwise cos = +0.61
    category_variance=0.3   mean pairwise cos = +0.17
    category_variance=0.5   mean pairwise cos = +0.10
    category_variance=1.0   mean pairwise cos = +0.06

The mapping saturates and is not linear in the dial, which is exactly why the
generator reports the *realised* similarity alongside the requested variance.
Label a sweep by what it achieved, not by what it asked for.

**Axis C (extent / N).** Because the discriminators come from a category rather
than a rotation, N > 2 confusable episodes cost nothing extra. That is the
"cheapest real gain" of the design note, and it is unavailable spatially.

Note what this generator does NOT do: it does not manipulate the similarity of
the *items* in the sequences. That is `semantic_similarity`, and grading it
where transitions are bijective grades nothing, because no cue ever demands two
successors. The ambiguity has to exist first -- that is what the shared stretch
is for -- and only then is there a discriminator whose similarity means
something.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from .base import SequenceGenerator
from .overlap import (assemble_overlapping_episodes, discriminator_similarity,
                      resolve_geometry)
from ..encoders.symbolic import SymbolicEncoder


class SymbolicOverlapGenerator(SequenceGenerator):
    """N symbol sequences sharing a middle stretch, discriminated by context.

    The content channel is a symbol embedding per step; the discriminator
    channel is a context vector drawn from a single category, written into the
    zone only (unless ``disc_on_suffix``). Both are delivered as input
    dimensions, never through ``context_data`` -- routing context through that
    channel would silently drop every arm that does not implement ``n_context``,
    which is most of the roster.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def generate(
        self,
        n_episodes: int = 2,
        total_length: int = 16,
        shared_fraction: float = 0.5,
        shared_position: float = 0.0,
        zone_fraction: float = 0.5,
        zone_offset: float = 0.0,
        category_variance: float = 0.3,
        discriminator_mode: str = "category",
        embedding_dim: int = 64,
        n_discriminator_dims: int = 32,
        balance_modalities: bool = True,
        disc_scale: float = 2.0,
        disc_on_suffix: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the episode set.

        Parameters
        ----------
        n_episodes
            Axis C. Every episode shares the same middle stretch and has its own
            prefix, suffix and discriminator.
        shared_position
            Where the shared stretch sits. 0.0 makes it a shared PREFIX (no
            unique prefix at all, so the discriminator is the only thing that
            identifies an episode). Above 0.0 gives each episode a unique prefix
            followed by a shared middle -- the Agster paradigm proper, and the
            only configuration in which an episode can be identified without
            modality B at all.
        category_variance
            Axis B. Smaller means the discriminators are MORE similar and the
            task is harder. Passed to `SymbolicEncoder` for the discriminator
            vocabulary only; content items are drawn near-orthogonally so that
            item similarity is not a confound. Ignored unless
            ``discriminator_mode == "category"``.
        discriminator_mode
            What occupies the discriminator slot.

            ``"category"``    N vectors from one category at ``category_variance``.
                              The graded case; this is axis B.
            ``"orthogonal"``  N exactly orthonormal vectors. The MAXIMUM-
                              separability control: discriminability is removed
                              as a limiting factor, so anything that still
                              degrades with N is ordinary capacity rather than a
                              disambiguation failure. Without this control a
                              falling load curve cannot be attributed.
            ``"identical"``   Every episode gets the SAME vector, so modality B
                              carries zero information. Combined with
                              ``shared_position > 0`` this is the ENDOGENOUS
                              case: the unique prefix is the only discriminator
                              that exists. Meaningful only for an arm that
                              declares StatePrimeable -- with no carried state
                              the probe sees an input identical across episodes
                              and is at chance by construction.
        disc_scale, balance_modalities, disc_on_suffix
            As in `generators.overlap.assemble_overlapping_episodes`.

        Returns
        -------
        dict
            Everything `assemble_overlapping_episodes` returns, plus the symbol
            sequences, both encoders, the divergence index, and
            ``discriminator_similarity`` -- the realised mutual cosine, which is
            what a sweep row should be labelled by.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if n_episodes < 2:
            raise ValueError("Need at least 2 episodes to have anything to disambiguate.")

        geom = resolve_geometry(total_length, shared_fraction, shared_position,
                                zone_fraction, zone_offset)
        ss, se = geom["shared_start"], geom["shared_end"]
        prefix_len, shared_len, suffix_len = ss, se - ss, total_length - se

        # ---- vocabulary -----------------------------------------------------
        # Content items are spread one-per-category at high variance, i.e. drawn
        # near-orthogonally. Item similarity is deliberately NOT the manipulated
        # variable here (see the module docstring); holding it near zero keeps
        # axis B about the discriminator and nothing else.
        n_shared_items = shared_len
        n_unique_items = n_episodes * (prefix_len + suffix_len)
        content_words = [f"c{i}" for i in range(n_shared_items + n_unique_items)]
        content_vocab = {w: f"cat{i}" for i, w in enumerate(content_words)}
        content_encoder = SymbolicEncoder(
            content_vocab, embedding_dim=embedding_dim,
            category_variance=1.0, seed=int(self.rng.integers(1 << 30)))

        # The discriminators. See `discriminator_mode` in the docstring.
        if discriminator_mode not in ("category", "orthogonal", "identical"):
            raise ValueError(
                f"discriminator_mode must be 'category', 'orthogonal' or "
                f"'identical', got {discriminator_mode!r}.")
        if discriminator_mode == "orthogonal" and n_episodes > n_discriminator_dims:
            raise ValueError(
                f"Cannot build {n_episodes} orthonormal discriminators in "
                f"{n_discriminator_dims} dimensions. Raise n_discriminator_dims.")

        disc_words = [f"ctx{i}" for i in range(n_episodes)]
        disc_vocab = {w: "context" for w in disc_words}
        disc_encoder = SymbolicEncoder(
            disc_vocab, embedding_dim=n_discriminator_dims,
            category_variance=category_variance,
            seed=int(self.rng.integers(1 << 30)))

        # ---- symbol sequences ----------------------------------------------
        shared_syms = content_words[:n_shared_items]
        rest = content_words[n_shared_items:]
        sequences: List[List[str]] = []
        for e in range(n_episodes):
            base = e * (prefix_len + suffix_len)
            prefix = rest[base: base + prefix_len]
            suffix = rest[base + prefix_len: base + prefix_len + suffix_len]
            sequences.append(list(prefix) + list(shared_syms) + list(suffix))

        contents = [content_encoder.encode(s) for s in sequences]
        if discriminator_mode == "category":
            discriminators = np.array([disc_encoder.embeddings[disc_words.index(w)]
                                       for w in disc_words], dtype=float)
        elif discriminator_mode == "orthogonal":
            # Exactly orthonormal, not "high variance so probably near-
            # orthogonal": the control has to remove discriminability as a
            # factor, not merely reduce it.
            discriminators = np.eye(n_episodes, n_discriminator_dims, dtype=float)
        else:  # "identical"
            one = disc_encoder.embeddings[0]
            discriminators = np.tile(np.asarray(one, dtype=float),
                                     (n_episodes, 1))

        out = assemble_overlapping_episodes(
            contents=contents,
            discriminators=discriminators,
            geom=geom,
            total_length=total_length,
            n_content=embedding_dim,
            disc_on_suffix=disc_on_suffix,
            balance_modalities=balance_modalities,
            disc_scale=disc_scale,
        )

        out.update({
            "sequences": sequences,
            "content_encoder": content_encoder,
            "discriminator_encoder": disc_encoder,
            "shared_symbols": shared_syms,
            "category_variance": float(category_variance),
            "discriminator_mode": discriminator_mode,
            # True when modality B carries no information at all, so the only
            # thing that can identify an episode is its unique prefix.
            "endogenous": bool(discriminator_mode == "identical"),
            # Realised, not requested. category_variance -> cosine is nonlinear
            # and saturating, so a sweep must be read against this.
            "discriminator_similarity": discriminator_similarity(discriminators),
            # The step whose successor first distinguishes the episodes: the
            # last shared step. Predicting FROM it lands on the divergent suffix.
            "divergence_step": se - 1,
        })
        return out
