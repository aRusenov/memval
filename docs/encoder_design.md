# Encoder design — the symbolic stimulus space

> **Status:** design note written against the shipped code
> (`memval/encoders/symbolic.py`, `memval/encoders/hierarchical.py`). Every
> numeric claim below is reproduced by `docs/figures/encoder_geometry.py`, which
> also renders the two figures. **Citations follow the placeholder convention of
> `docs/paper_intro.md` — they are anchors, not verified page references.** A
> verify-list is at the end.

Figures:
- `docs/figures/encoder_geometry_pca.png` — PCA views of the 100-D code at three
  overlap settings, plus the similarity knob and its inverse.
- `docs/figures/encoder_hierarchy_svd.png` — the deep-tree variant: the quantised
  cosine ladder and the singular spectrum whose dimensions *are* the splits.

---

## 1. What we chose

MemVal encodes symbols with a **hierarchical synthetic embedding**: an item's
vector is built from the node structure above it, so items that share ancestors
share vector components. Two encoders implement this at two depths, and they are
not redundant.

| | `SymbolicEncoder` | `HierarchicalEncoder` |
|---|---|---|
| depth | 2 levels (category → item) | arbitrary, from a tree spec |
| dimension | dense, `embedding_dim=100` | sparse binary, `features_per_node × #nodes` |
| overlap control | **continuous** scalar `category_variance` | **quantised** by tree depth |
| coordinate meaning | none (rotationally symmetric) | one column = one node's features |
| used by | the symbolic suite (sections 1–5) | schema consistency (section 6) |

`SymbolicEncoder` is the workhorse. It exists to make **overlap a continuous
independent variable** that can be dialled to a target and inverted analytically.
`HierarchicalEncoder` exists because that dial buys its continuity by making the
coordinates meaningless, which breaks any probe that manipulates individual
dimensions.

## 2. Why synthetic, and not word2vec / GloVe / BERT

The obvious alternative is a pretrained embedding table. We decline it, and the
reason is not convenience:

1. **The independent variable would become uncontrolled.** MemVal's symbolic
   sections vary item overlap on purpose — `item_similarity` is weight 0.40 of
   Sequence disambiguation. In a pretrained table, similarity is whatever the
   corpus made it; you can *measure* it but not *set* it, so a similarity
   ladder has to be assembled by cherry-picking word lists, which confounds
   overlap with frequency, polysemy, and list composition.
2. **No inverse.** We need "give me a list whose mean pairwise cosine is 0.4."
   Section 6.4 gives that in closed form. A pretrained table gives a search
   problem with no exact solution.
3. **Known geometric pathologies.** Static embeddings carry a large common mean
   vector and a few dominant directions that distort cosine [Mu & Viswanath
   2018]; contextual embeddings are strongly anisotropic, so cosine similarity
   is inflated and non-comparable across layers and words [Ethayarajh 2019].
   Either would put an uncontrolled offset in exactly the quantity we score.
4. **Frequency confounds.** Embedding norm and neighbourhood density track
   corpus frequency, which also predicts human memory performance — so a
   frequency effect would masquerade as a similarity effect.

The cost is ecological validity, and we should state it rather than argue it
away: these are **not** lexical semantic vectors and no claim in MemVal depends
on them being so. The encoder is a *stimulus generator*. What it must reproduce
is the structural property the memory literature actually leans on — that items
have a graded, hierarchically organised similarity relation — not the lexical
content of English.

## 3. Why this particular construction

### 3.1 Random high-dimensional vectors as item codes

Independent Gaussian vectors in `d` dimensions are quasi-orthogonal: their
cosine concentrates at 0 with standard deviation `1/√d` (verified: 0.0998 at
d = 100). This is the standard justification for high-dimensional distributed
codes — Johnson–Lindenstrauss, and the vector-symbolic tradition where
near-orthogonality of random codes is the *design principle* [Plate 1995,
Holographic Reduced Representations; Kanerva 2009, hyperdimensional computing].
It is also the regime in which the associative-memory capacity results the
arms inherit were derived: near-orthogonal patterns are what Hopfield-style and
Willshaw-style capacity bounds assume [Hopfield 1982; Amit, Gutfreund &
Sompolinsky 1985; Treves & Rolls 1991].

`d = 100` is a deliberate middle. It makes chance overlap `1/√d = 0.1` — small
enough that distinct categories are effectively orthogonal, large enough that
false neighbours are a real risk a weak model can fall into rather than an
impossibility.

### 3.2 Category as a shared mean, item as a private deviation

The construction `item = category prototype + noise`, renormalised, is the
**prototype-plus-distortion** paradigm of the categorization literature [Posner
& Keele 1968], and the feature-vector-with-graded-similarity construction used
throughout the formal memory-model tradition [Hintzman 1984, 1986 (MINERVA 2);
Murdock 1982 (TODAM)]. Similarity is read out as cosine, which is what the
models' own decoder uses (`SymbolicDecoder`), so the stimulus metric and the
scoring metric are the same object — no translation step where an effect can be
manufactured. Shepard's universal law [Shepard 1987] is the background claim
that generalisation should fall off smoothly with psychological distance; a
continuous knob is what lets us test at several points on that curve instead of
one.

### 3.3 Why overlap is the axis worth controlling

Interference in human memory scales with similarity between the competing
materials — the oldest quantitative result in the field [McGeoch 1932; Osgood
1949's transfer surface; Underwood 1957]. In hippocampal modelling, the entire
functional argument for the dentate gyrus is that it *reduces* overlap before
storage [Marr 1971; McNaughton & Morris 1987; Treves & Rolls 1992, 1994; Norman
& O'Reilly 2003], with the behavioural correlate studied as mnemonic
discrimination [Yassa & Stark 2011]. MemVal carries DG-expansion and XdG arms
whose whole claim is about pattern separation. A benchmark that cannot set
overlap cannot test them. That is why the knob is the encoder's primary feature
and not an afterthought.

### 3.4 Why the deep-tree variant also exists

Two reasons, one from the literature and one from our own failures.

**From the literature.** McClelland, McNaughton & Lampinen (2020) specify a
*generative rule*: a feature may be introduced at any node and appear in that
node's descendants, but never in descendants of nodes in another branch. That
rule is what makes "is this new item consistent with the schema?" a well-posed,
model-free question — which is the whole of the schema section. The rule also
produces the structure Saxe, McClelland & Ganguli (2019) analyse: the item ×
feature matrix's singular values are ordered by the depth of the split they
encode, and in a network trained by gradient descent each mode is learned on a
timescale set by `1/s_k`, giving progressive differentiation — superordinate
distinctions before subordinate ones. The right-hand panel of figure 2 is that
spectrum, measured rather than assumed, with each dimension labelled by the
split it turns out to encode. Behind both sits the older claim that human
semantic memory is organised hierarchically [Collins & Quillian 1969; Rogers &
McClelland 2004].

**From our own failures.** `SymbolicEncoder`'s noise is isotropic, so its
coordinates carry no individual meaning: any rotation of the space gives the
same benchmark. A probe that masks or ablates individual dimensions of that code
is therefore measuring the encoder's arbitrary basis, not the arm. That was a
live defect in the cue-masking section; the fix was to run cue masking on a
structured basis, which is what `HierarchicalEncoder` provides.

## 4. What the figures show

**Figure 1** (`encoder_geometry_pca.png`). Three PCA views of the same 18-word,
3-category vocabulary at σ = 0.05, 0.20, 0.50, with the projections rotated onto
a common frame so the panels are comparable. With K = 3 categories the
between-category subspace is exactly K − 1 = 2 dimensional, so the 2-D
projection discards none of the category geometry; all residual spread inside a
cluster is the noise ball's shadow.

Read it with the chance line, not without it. The explained-variance ratio of the
top two PCs of 18 points in 100-D is **0.200 ± 0.007 under pure isotropic
noise**. The σ = 0.50 panel reports 19% — it is *at* chance, and what is drawn
there is sampling noise, not faint category structure.

The bottom-left panel is the same fact as a curve: within-category cosine follows
`1/(1 + dσ²)` across four orders of magnitude while between-category cosine stays
pinned at zero inside a `±1/√d` band. **One scalar moves overlap and moves
nothing else** — norms stay at 1, dimension stays at 100, sparsity is undefined
(the code is dense throughout). The bottom-right panel inverts it.

**Figure 2** (`encoder_hierarchy_svd.png`). The deep-tree variant on the paper's
own 8-item tree. Left: cosine is *quantised* — it takes only the values ℓ/L set
by the depth of the deepest common ancestor. Right: the singular spectrum, each
bar labelled with the split `dimension_alignment()` matched it to, at alignment
1.00. Strength falls with depth, which is the ordering that produces progressive
differentiation.

## 5. Limitations we should state, not bury

1. **Coordinates are meaningless in `SymbolicEncoder`** (§3.4). Any per-dimension
   manipulation belongs on `HierarchicalEncoder`.
2. **Cosine overlap and linear separability decouple, badly.** Measured over 90
   words in 3 categories:

   | σ | within-cat. cosine | leave-one-out nearest-centroid accuracy |
   |---|---|---|
   | 0.05 | 0.80 | 1.000 |
   | 0.20 | 0.22 | 1.000 |
   | 0.30 | 0.12 | 0.978 |
   | 0.50 | 0.05 | 0.822 |
   | 1.20 | 0.02 | 0.622 |

   At σ = 0.2 pairwise cosine says the items are nearly unrelated while a linear
   readout recovers the category perfectly. Cosine dies at σ ~ 1/√d; separability
   survives to σ ~ 1. **"Similarity" is therefore not one number**, and an arm
   with a learned linear readout and an arm scored by cosine nearest-neighbour
   are looking at different quantities. Any claim of the form "we removed the
   category structure by raising σ" must say which sense it means.
3. **No graded typicality.** Every item is equidistant from its prototype in
   expectation (`cos = √ρ`, §6.3). Real categories have central and peripheral
   members; this one does not.
4. **Between-category cosine is a per-*pair* offset, not per-word noise.** All
   fruit–animal pairs share the same `c_fruit · c_animal ≈ ±1/√d` term, fixed by
   the seed. With few categories one draw can leave two categories at +0.03 and
   two others at −0.02 (on the chain's seed-42 vocabulary the prototype pairs
   span −0.15 to +0.15). Do not read a small cross-category asymmetry as
   design. Where it matters, pass `between_category_cosine` (§6.4) — `0.0`
   removes the offset exactly, a positive value sets the overlap.
5. **`category_variance` is a standard deviation, not a variance.** The code
   multiplies `standard_normal()` by it. The name is wrong and is load-bearing in
   every formula below; renaming it would break saved configs, so it is
   documented instead.
6. **Iteration order is load-bearing.** One RNG draw per word in the mapping's
   order, so build vocabularies with `dict.fromkeys`, never `set()`. This was a
   real reproducibility defect: it gave a 1.9× spread in epochs-to-criterion
   across runs of the *same* rule.
7. **The laws in §6 are asymptotic in `d`.** At d = 100 with 600+ words they hold
   to four decimals; at 18 words expect ~2% deviation (0.786 measured vs 0.800
   predicted at σ = 0.05).

---

## 6. The mathematics

### 6.1 The pipeline, end to end

Three stages with three different lifetimes: **build** runs once per encoder,
**encode** once per episode, **decode** once per recalled step. Throughout:
`N = |V|` words, `d = 100` dimensions, `T` steps, `σ = category_variance`,
`ν = noise_scale`.

#### Build — `SymbolicEncoder.__init__`, once

| # | operation | formula | shape |
|---|---|---|---|
| B1 | partition the vocabulary | `cat: V → C`, K categories | — |
| B2 | one Gaussian per category | `g_k ~ N(0, I_d)` | `(d,)` × K |
| B3 | **project onto the unit sphere** | `c_k = g_k / ‖g_k‖` | `(d,)` × K |
| B4 | one Gaussian per word | `ε_w ~ N(0, I_d)` | `(d,)` × N |
| B5 | **mix prototype with scaled noise** | `u_w = c_{cat(w)} + σ ε_w` | `(d,)` |
| B6 | **project onto the unit sphere** | `e_w = u_w / ‖u_w‖` | `(d,)` |
| B7 | stack into the codebook | `E = [e_w]_{w∈V}` | `(N, d)` |

Two projections, not one, and both matter.

- **B3** puts prototypes on the sphere so that `1` is the reference scale for
  every cosine computed downstream.
- **B6 is what makes σ a pure similarity knob.** Without it, raising σ would
  raise every embedding's norm, hence the magnitude of every activation in the
  model, and a "similarity effect" would be partly a gain effect. With it,
  `‖e_w‖ = 1` at every σ, so the *only* thing σ moves is angle.

Note the order: noise is added in the ambient space and *then* projected, so
this is not tangential noise on the sphere. B6 discards the radial component,
which is exactly why the surviving overlap is `1/(1 + dσ²)` — a function of
`dσ²` — rather than of σ alone.

#### Encode — `SymbolicEncoder.encode`, once per episode

```
    E1   word → one-hot            δ_w ∈ {0,1}^N
    E2   project through codebook  x_w = Eᵀ δ_w = e_w                  (d,)
    E3   stack the episode         X = Δ E                             (T, d),  ‖X_t‖ = 1
    E4   corrupt the probe         x̃ = x + η,   η ~ N(0, ν² I_d)      (d,)
```

E2 is implemented as a row lookup, but it is a **linear map**, and it is the
transpose of the map the decoder applies — that symmetry is the point of §6.1's
last paragraph.

E4 happens at recall time only (`measure_recall_*`, `ν = 0.05` throughout the
suite) and is **not renormalised** — a deliberate asymmetry with B6. It is
otherwise the same operation as B5 with ν in place of σ, so the same law governs
it:

```
    cos(x̃_w, e_w)  =  1 / √(1 + dν²)  =  0.894      ‖x̃‖ ≈ 1.118      (ν=0.05, d=100)
```

So the "clean" cue in every associative-recall measurement already sits **27°
off the item it names** and carries 12% excess norm. And the two noise sources
**multiply rather than add**: for a same-category competitor `w′`,

```
    cos(x̃_w, e_w′)  =  ρ_σ / √(1 + dν²)  =  ρ_σ · √ρ_ν
```

verified at d = 100 over 600 words (σ = 0.1: measured 0.4484 vs predicted
0.4472; σ = 0.2: 0.1747 vs 0.1789). Cue noise scales the whole similarity
structure down by a constant factor; it does not blur categories into each
other.

#### Decode — `SymbolicDecoder`, once per recalled step

The model returns an activation `a ∈ R^d` with unconstrained norm.

```
    D1   normalise                 â = a / ‖a‖
    D2   project onto every item   s = E â ∈ R^N                       one matvec
    D3   pick                      ŵ = argmax_w s_w,  confidence = max_w s_w
    D4   margin                    m = s_target − max_{w ≠ target} s_w
```

D2 is the transpose of E2. Because every row of `E` is unit norm, `s_w` **is**
the cosine, and `‖â − e_w‖² = 2 − 2 s_w`, so maximum-inner-product, nearest
neighbour and smallest angle give the identical ranking — there is no choice to
make here. D4 is `diagonal_dominance` in `memval/metrics/capacity.py`; positive
means the memories are separable, near-zero means interference, and it resolves
differences below the accuracy floor that top-1 accuracy cannot see.

**What overlap costs the decoder.** The strongest competitor to any target is,
by construction, a same-category item at cosine `ρ`. So the margin available
*before the model makes any error at all* is

```
    m_max  =  1 − ρ          within category
           ≈  1 − O(1/√d)    across categories
```

Moving ρ from 0.01 to 0.80 shrinks the decodable gap fivefold. That is the
mechanism the similarity sweep exercises, and because the decoder is fixed it
applies identically to every arm regardless of learning rule.

**Tied weights, no learned codec.** Encoding is `Eᵀ`, decoding is `E`, and
neither side has a trainable parameter. The pair is an exact autoencoder on
clean input — `E Eᵀ δ_w` is maximised at `w` whenever the margin is positive —
so any shortfall a benchmark measures is attributable to the model in between,
not to a lossy encoding. That is what licenses comparing arms with very
different internals on one scale.

#### The same pipeline for `HierarchicalEncoder`

B1–B7 are replaced by a deterministic construction (no RNG at all): each
non-root node claims `f` fresh binary columns, a leaf's raw row is the OR of the
columns owned by its ancestors, and B6's L2 projection is retained so the rows
land on the same unit sphere. E1–E4 and D1–D4 are then **bit-for-bit
identical** — same `encode`, same `SymbolicDecoder`. That interface parity is
deliberate: swapping the encoder swaps the stimulus geometry and changes nothing
else in the measurement path.

### 6.2 The norm

```
    ‖u_w‖² = 1 + 2σ ⟨c_k, ε_w⟩ + σ² ‖ε_w‖²
    E[‖u_w‖²] = 1 + σ² d                                          (⟨c_k,ε⟩ ~ N(0,1),  ‖ε‖² ~ χ²_d)
```

`‖ε‖²` concentrates: relative fluctuation `√(2/d)`. So `‖u_w‖² = (1 + dσ²)(1 +
O(d^{-1/2}))`, and the combination `κ ≡ d σ²` — not `d` and `σ` separately — is
what controls the geometry.

### 6.3 Signal/noise decomposition — the one identity to remember

Project `e_w` onto its prototype and the orthogonal complement. Writing
`ρ ≡ 1/(1 + dσ²)`:

```
    e_w  =  √ρ · c_k  +  √(1−ρ) · z_w ,      z_w ⊥ c_k ,  ‖z_w‖ = 1
```

The `z_w` are, to leading order, mutually orthogonal (they inherit isotropy from
`ε`, and live in a `(d−1)`-dimensional space). So:

- **loading on the category axis:** `⟨e_w, c_k⟩ = √ρ`
  (verified at d = 100, 900 words: 0.4483 measured vs `√0.2 = 0.4472` at σ = 0.2)
- **mixing angle:** `θ = arccos √ρ` — the single geometric parameter of the code.

Everything in §6.4–6.6 is a corollary of this one line.

### 6.4 Similarity laws

**Within category** (`cat(w) = cat(w′) = k`, `w ≠ w′`):

```
    ⟨e_w, e_w′⟩ = ρ ⟨c_k, c_k⟩ + (1−ρ) ⟨z_w, z_w′⟩ = ρ + O(d^{-1/2})

    ρ_within  =  1 / (1 + d σ²)
```

Directly: `⟨u_w, u_w′⟩ = 1 + σ⟨c, ε_w + ε_w′⟩ + σ²⟨ε_w, ε_w′⟩` has expectation 1,
and both norms concentrate at `√(1 + dσ²)`.

Verified at d = 100 over 600 words:

| σ | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|
| measured | 0.8010 | 0.5009 | 0.2001 | 0.0999 | 0.0383 |
| `1/(1+dσ²)` | 0.8000 | 0.5000 | 0.2000 | 0.1000 | 0.0385 |

**Between categories** (`k ≠ l`):

```
    ⟨e_w, e_w′⟩ = ρ ⟨c_k, c_l⟩ + O(d^{-1/2}) ,     ⟨c_k, c_l⟩ ~ N(0, 1/d)

    ρ_between = 0 ,     sd → 1/√d                                (0.0998 measured at d = 100)
```

Note the `ρ` prefactor on the shared term: the per-pair offset of §5.4 is itself
*damped* as σ grows.

**Between categories, controlled** (`between_category_cosine = ρ_b`, added
2026-09-06). Passing a float builds the prototypes as

```
    c_k = √ρ_b · g + √(1−ρ_b) · u_k ,      u_k orthonormal (QR of the same draws),
                                             g ⊥ every u_k, ‖g‖ = 1
    ⟨c_k, c_l⟩ = ρ_b   exactly, every pair        (needs K + 1 ≤ d)
    ⟨e_w, e_w′⟩ = ρ · ρ_b + O(d^{-1/2})   across categories
```

so the word-level between-category cosine is `ρ ρ_b` and can never exceed the
within-category `ρ`. `ρ_b = 0.0` gives *exactly* orthogonal categories, which
the default (`None`, quasi-orthogonal, unchanged RNG stream) does not. For a
fixed seed the per-word deviations `z_w` are the same draws at every `ρ_b`, so
a sweep over it moves only the prototypes. Verified on the 6×5 chain vocabulary
at d = 100:

| σ | ρ_b | within measured / law | between-word measured / law |
|---|---|---|---|
| 0.20 | 0.0 | 0.193 / 0.200 | −0.004 / 0.000 |
| 0.20 | 0.5 | 0.184 / 0.200 | 0.085 / 0.100 |
| 0.20 | 0.9 | 0.182 / 0.200 | 0.162 / 0.180 |
| 0.10 | 0.5 | 0.487 / 0.500 | 0.234 / 0.250 |
| 0.05 | 0.5 | 0.796 / 0.800 | 0.390 / 0.400 |

This is the dial the forgetting lever needs: task B disturbs task A in
proportion to `x_B · x_A`, and until now nothing in the encoder could move that
product off its seed-fixed `±1/√d`. `bin/continual_chain_overlap_sweep.py`
sweeps both dials and draws the retention matrix at every rung.

**Inverse design.** Solving `ρ = 1/(1 + dσ²)`:

```
    σ(ρ) = √( (1 − ρ) / (ρ d) )
```

This is the operational form. To build a list at mean pairwise cosine 0.4 in
d = 100: `σ = √(0.6/40) = 0.122`. The sweep in the suite,
`σ ∈ {0.05, 0.1, 0.2, 0.5, 1.0}`, is therefore the ladder
`ρ ≈ {0.80, 0.50, 0.20, 0.038, 0.010}` — geometrically spaced in overlap at the
top and saturated at the bottom, which is worth knowing when reading that sweep:
its last two rungs are nearly the same stimulus.

**Scale invariance.** Since only `κ = dσ²` appears, `(d, σ)` and `(4d, σ/2)` give
identical similarity structure; changing `d` alone moves only the chance-overlap
floor `1/√d`.

### 6.5 Spectrum and what PCA can show

With `K` equally sized categories and quasi-orthogonal prototypes, the population
covariance of the embeddings splits into

```
    Σ  ≈  ρ Σ_k (1/K) c_k c_kᵀ   +   ((1−ρ)/d) (I − P)
          └──── K−1 informative dims ────┘   └── d−K+1 noise dims ──┘
```

after centering: `K − 1` eigenvalues of size `ρ/K`, and `≈ d − K + 1` of size
`(1 − ρ)/d`. The top-`(K−1)` explained-variance ratio is

```
    EVR ≈ [ ρ (K−1)/K  +  (K−1)(1−ρ)/d ] / [ 1 − ρ/K ]
```

which for K = 3, ρ = 0.79 gives 0.72 — close to the 0.75 the 18-word sample
shows. **But this expression is only usable when `n ≫ d`.** With `n` points in
`d ≫ n` dimensions the sample covariance has rank `n − 1` and the top two PCs
capture ≈ `0.20` of the variance *with no structure present at all* (simulated,
n = 18, d = 100: 0.200 ± 0.007). Hence the chance annotation on figure 1: a
low-but-nonzero EVR in this regime is evidence of nothing.

### 6.6 The deep-tree encoder

Tree with root at level 0, leaves at level `L`, `f = features_per_node` fresh
binary columns introduced by each non-root node, inherited by all of its
descendant leaves. Raw row `M_w` has `f · L` active columns; rows are
L2-normalised.

**Quantised cosine ladder.** Let `ℓ_ij` be the level of the deepest common
ancestor of leaves `i, j`. Shared columns are exactly those owned by the
`ℓ_ij` nodes on the shared part of the path, so

```
    ⟨M_i, M_j⟩ = f · ℓ_ij ,     ‖M_i‖² = f · L

    cos(i, j) = ℓ_ij / L        ∈ { 0, 1/L, 2/L, …, 1 }
```

Verified exactly (figure 2, left) for the 8-item tree: 2/3 for siblings, 1/3
across the level-2 split, 0 across the top split. This is the complement of
`SymbolicEncoder`: similarity is **structural and discrete**, and the way to move
it is to re-shape the tree, not to turn a dial.

**Singular values.** For the dimension aligned with a node `n` with leaf set
`Λ(n)`, let `D(n)` be the nodes of the subtree rooted at `n` (including `n`):

```
    s(n)²  =  f · ( Σ_{v ∈ D(n)} |Λ(v)|² ) / |Λ(n)|
```

Verified exact (to 1e-9) for every dimension of every tree tested — branchings
(2,2,2), (3,2), (2,3,2), (2,2,2,2), (4,3), (2,2,3), with `f ∈ {1,2,3}`. For a
balanced tree of branching `b` and depth `k` below `n` this collapses to
`s² = f·|Λ(n)|·(1 − b^{−(k+1)})/(1 − 1/b)`; on the 8-item tree it gives
`√14 > √6 > √2` for levels 1, 2, 3. **Singular value decreases monotonically with
the depth of the split**, which under gradient descent maps to learning-time
ordering `τ_k ∝ 1/s_k` [Saxe, McClelland & Ganguli 2019] — coarse distinctions
first.

**Degeneracy caveat.** A `b`-way split needs `b − 1` dimensions and produces `b −
1` *tied* singular values. Inside a tied block the SVD basis is arbitrary, so
per-dimension alignment is only meaningful at binary splits; elsewhere check
alignment at the subspace level. This is already documented in
`dimension_alignment`'s docstring and tested in
`tests/test_hierarchical_encoder.py`.

**Schema consistency.** For a new item `v` and the rank-`r` right singular basis
`V_r` of the base matrix, the model-free consistency score is the fraction of
energy inside the existing feature basis:

```
    π(v) = ‖V_r v‖² / ‖v‖²   ∈ [0, 1]
```

`π = 1` means the item demands no new dimension; `π → 0` means it is orthogonal
to everything the schema knows. This scores the *stimulus*, so the consistency
ladder (`duplicate` → `within` → `across` → `random`) can be verified before any
model is run.

### 6.7 The full pipeline, end to end

Everything above is static geometry. This is the sequence of maps a single word
actually passes through, from dictionary entry to scored prediction. Symbols:
`d = 100`, `σ = category_variance`, `η = noise_scale` (0.05 in the suite),
`ρ = 1/(1 + dσ²)`.

```
  word w
    │  ① category lookup                     cat(w) = k
    ▼
  prototype  c_k = g_k/‖g_k‖ ,  g_k ~ N(0, I_d)          ‖c_k‖ = 1
    │  ② add private deviation               u_w = c_k + σ ε_w ,  ε_w ~ N(0, I_d)
    ▼
  unnormalised item  u_w                                  ‖u_w‖ ≈ √(1 + dσ²)
    │  ③ project to the sphere               e_w = u_w / ‖u_w‖
    ▼
  EMBEDDING  e_w = √ρ·c_k + √(1−ρ)·z_w                    ‖e_w‖ = 1
    │  ④ sequence lookup                     S = [e_{s_1}; … ; e_{s_T}] ∈ R^{T×d}
    ▼
  episode matrix S            ── training sees these rows CLEAN ──
    │  ⑤ probe corruption                    x_i = e_{s_i} + η ζ ,  ζ ~ N(0, I_d)
    ▼
  cue  x_i                                                ‖x_i‖ ≈ √(1 + dη²) = 1.118
    │  ⑥ model                               a = f_θ(x_i) ∈ R^d ,  ‖a‖ arbitrary
    ▼
  activation a
    │  ⑦ normalise and match                 â = a/‖a‖ ,  s = E â ∈ R^{|V|}
    ▼
  prediction  ŵ = argmax_w s_w ,   margin m = s_target − max_{w≠target} s_w
```

**② and ⑤ are the same map.** Both are "unit vector + isotropic Gaussian", one
level apart: the item is a noisy copy of its prototype, the cue is a noisy copy
of the item. So the same law governs both, and the loadings **multiply**:

```
    ⟨e_w, c_k⟩   = 1/√(1 + dσ²) = √ρ
    ⟨x̂, e_w⟩     = 1/√(1 + dη²)                    = 0.894  at η = 0.05
    ⟨x̂, c_k⟩     = 1/√( (1 + dσ²)(1 + dη²) )
```

Verified at d = 100, 600 words: measured 0.8018 vs predicted 0.8000 (σ = 0.05),
and 0.3997 vs 0.4000 (σ = 0.2). The whole chain is one product of cosines.

**③ is where unit norm is imposed — and ⑤ breaks it.** Step ③ renormalises, so
every embedding is on the sphere. Step ⑤ adds noise and does **not**
renormalise, so the cue leaves it: `‖x‖ = √(1 + dη²) = 1.118`. Models are
trained on unit-norm rows (`encoder.encode(words)`, clean) and probed with
vectors 12% longer. For a linear readout this is a gain change; for a saturating
unit it is a shift in operating point. It is not a bug — but it is an
uncontrolled 12% input-scale mismatch between train and test, and any arm whose
score is sensitive to input scale is being charged for it.

**⑦ discards ‖a‖ entirely.** The decoder normalises before matching, so a model
is scored on the *direction* of its output only. An arm that predicts the right
direction with tiny magnitude scores identically to one that predicts it with
large magnitude. This is deliberate — it makes arms with wildly different output
scales commensurable — but it means "confidence" in MemVal is the cosine and the
margin, never the norm.

**The discriminability budget.** For a *perfect* model (`a = e_target` exactly)
the margin is not `1 − ρ`; it is `1 −` the **largest** competitor cosine, which
is an extreme value over `|V| − 1` competitors rather than the mean:

| | σ = 0.2 | σ = 0.5 |
|---|---|---|
| mean competitor `ρ` | 0.200 | 0.038 |
| max competitor, \|V\| = 30 | 0.331 | 0.207 |
| max competitor, \|V\| = 120 | 0.374 | 0.275 |
| max competitor, \|V\| = 600 | 0.425 | 0.315 |

The max grows like `√(2 ln|V| / d)` (which over-predicts by ~0.04 at these
sizes). **So the headroom a model has shrinks as the vocabulary grows even at
fixed σ**, and a recall score is only comparable across sections that hold `|V|`
fixed as well as `σ`. At σ = 0.5 the mean competitor sits at 0.038 while the
worst sits at 0.315 — the floor of the task is set by the nearest false
neighbour, not by the average one.

**Autoregressive probing** replaces step ⑤ with `x_{i+1} = a_i` — the model's own
unnormalised output becomes the next cue. Norm error then compounds across
positions, which is one reason generative and cued probing diverge for arms
whose output scale drifts.

---

---

## 7. Verify-list

Anchors used above, to be checked against the reference library before
submission:

- Collins & Quillian 1969 — hierarchical organisation of semantic memory
- Rogers & McClelland 2004 — *Semantic Cognition*; the 8-item hierarchy
- McClelland, McNaughton & Lampinen 2020 — generative rule, schema consistency
- Saxe, McClelland & Ganguli 2013 / 2019 — spectrum → learning-time ordering
- Posner & Keele 1968 — prototype plus distortion
- Hintzman 1984, 1986 — MINERVA 2 feature vectors; Murdock 1982 — TODAM
- Shepard 1987 — universal law of generalisation
- McGeoch 1932; Osgood 1949; Underwood 1957 — similarity and interference
- Marr 1971; McNaughton & Morris 1987; Treves & Rolls 1992, 1994; Norman &
  O'Reilly 2003; Yassa & Stark 2011 — pattern separation
- Hopfield 1982; Amit, Gutfreund & Sompolinsky 1985; Treves & Rolls 1991 —
  capacity under near-orthogonal patterns
- Plate 1995; Kanerva 2009 — random high-D codes as a design principle
- Mu & Viswanath 2018; Ethayarajh 2019 — geometric pathologies of pretrained
  embeddings
