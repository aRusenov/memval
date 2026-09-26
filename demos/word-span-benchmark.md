# Word-Span Benchmark Specification

> Companion document to `demos/word-span-demo.ipynb`  
> Model: **DGEqProp** · Encoder: **SymbolicEncoder** · Decoder: **SymbolicDecoder**

---

## 1. Output Metrics

All six benchmark dimensions share a common **three-metric evaluation battery**.
Additional derived metrics are defined per-dimension where relevant.

---

### 1.1 Mean Recall Rate (MRR)

**What it measures:** The average probability of correctly recalling any item in the sequence, pooled across all serial positions (excluding position 0, which is always provided as a cue).

$$\text{MRR} = \frac{1}{L-1} \sum_{i=1}^{L-1} \hat{p}_i$$

where $L$ is the list length and $\hat{p}_i$ is the empirical recall probability at position $i$ over $N_{\text{trials}}$ trials.

**Range:** 0 (no recall) → 1 (perfect recall).

**Use:** Primary aggregate score — easy to compare across conditions in a single number.

**Limitation:** Treats all serial positions equally; insensitive to the *shape* of the serial position curve (primacy/recency balance).

```python
def mean_recall_rate(recall_curve):
    """recall_curve: array of per-position recall probabilities."""
    return float(np.mean(recall_curve[1:]))   # position 0 always = 1.0
```

---

### 1.2 Memory Span

**What it measures:** The longest *prefix* of items that are recalled reliably (i.e., above a threshold θ), analogous to the classical **word-span / digit-span** measure in working-memory research.

$$\text{Span} = \max \{k \mid \hat{p}_i \geq \theta \;\; \forall i \in 1 \ldots k\}$$

**Range:** 0 → L − 1 (an integer).

**Default threshold:** θ = 0.75.

**Use:** Single-number capacity estimate; directly comparable to human cognitive norms (Miller's 7 ± 2 items).

**Limitation:** Sensitive to threshold choice; ignores recency — a model that recalls later items but not early ones scores zero.

```python
def memory_span(recall_curve, threshold=0.75):
    span = 0
    for r in recall_curve[1:]:
        if r >= threshold:
            span += 1
        else:
            break
    return span
```

---

### 1.3 Recall Fidelity

**What it measures:** Average **cosine similarity** between the model's raw predicted embedding and the clean ground-truth (GT) embedding of the target item, across all positions and trials. A continuous, geometry-aware quality measure that does not require the decoder.

$$\text{RF} = \frac{1}{N_{\text{trials}}(L-1)} \sum_{\text{trials}} \sum_{i=1}^{L-1} \cos\!\bigl(\hat{e}_{i+1},\; e_{i+1}^{\text{GT}}\bigr)$$

**Range:** −1 → 1 (in practice, 0 → 1 for normalised embeddings).

**Use:** Captures *graded* recall — a prediction that is close but not identical still gets partial credit. Useful for comparing models with similar MRR but different levels of confidence.

**Limitation:** Less interpretable than MRR; depends on the embedding geometry.

```python
def recall_fidelity(network, words, encoder, n_trials=30, noise_scale=0.05):
    gt_embs = encoder.encode(words)   # clean ground-truth embeddings
    total_sim, count = 0.0, 0
    for _ in range(n_trials):
        network.current_t = 0
        for i in range(len(words) - 1):
            noisy = encoder.encode([words[i]])[0] + \
                    np.random.normal(0, noise_scale, encoder.embedding_dim)
            pred = network.predict_next(noisy, current_context=np.array([1.0]))
            total_sim += _cosine_sim(pred, gt_embs[i + 1])
            count += 1
    return total_sim / count
```

---

### 1.4 Derived Metrics (per-dimension)

| Metric | Defined in | Formula |
|---|---|---|
| **Noise Tolerance Threshold** | Noise Invariance (§3.4) | σ at which MRR first drops below 0.5 |
| **Δ-MRR** | Multiple Sequences (§3.3) | MRR(list A, after list B) − MRR(list A, before list B); negative = forgetting |
| **Forgetting Slope** | Interval (§3.6) | Linear regression slope of MRR vs. delay steps |

---

## 2. Benchmark Summary

| # | Dimension | Status | Input Manipulation | Primary Metrics |
|---|---|---|---|---|
| 1 | **Presentation Duration** | ✅ Existing | Training epochs ∈ {100, 200, 300} | MRR, serial position curve |
| 2 | **Sequence Length** | 🔲 Proposed | List length ∈ {3, 5, 7, 9, …} | Memory Span, MRR vs. length |
| 3 | **Multiple Sequences** | 🔲 Proposed | Sequential training on list A then list B (weights retained) | Δ-MRR, Recall Fidelity |
| 4 | **Noise Invariance** | 🔲 Proposed | `noise_scale` σ swept 0 → 1.0 at recall time | MRR vs. σ, Noise Tolerance Threshold |
| 5 | **Semantic Similarity** | 🔲 Proposed | `category_variance` or list composition | MRR vs. mean pairwise cosine similarity |
| 6 | **Interval / Interference** | 🔲 Proposed | Noise proxy for delay, or interfering training steps | MRR vs. delay, Forgetting Slope |

> All experiments: **DGEqProp** model, evaluated with the **MRR + Memory Span (θ = 0.75) + Recall Fidelity** battery unless noted.

---

## 3. Per-Dimension Specifications

---

### 3.1 Presentation Duration

**Cognitive motivation:** In verbal working-memory research, longer presentation times (more rehearsal opportunities) improve recall — particularly for primacy items which benefit most from additional consolidation. This dimension tests whether training epochs are a faithful proxy for presentation duration.

**Setup:**
- Fixed 7-word list (all fruits: `apple, banana, orange, grape, pear, peach, plum`)
- Train a fresh DGEqProp model for N × 100 epochs (N = duration)
- Evaluate recall immediately after training

**Input manipulation:**

| Duration | Epochs |
|---|---|
| 1 presentation | 100 |
| 2 presentations | 200 |
| 3 presentations | 300 |

**What to plot:**
- Serial position curve (per-position recall probability) for each duration level
- Overlay all three curves on one axes to show primacy/recency evolution

**Expected results:**
- More epochs → higher overall MRR; primacy advantage (early items) appears first
- At low epochs, recency dominates (short-term); at high epochs, recall is uniformly high

**Evaluation metrics:** MRR, Memory Span, serial position curve shape

---

### 3.2 Sequence Length

**Cognitive motivation:** Miller's Law states that humans can hold approximately 7 ± 2 items in working memory. This dimension tests whether DGEqProp obeys a similar capacity limit and at what list length recall degrades.

**Setup:**
- Vary the number of words drawn from the fruit vocabulary: L ∈ {3, 5, 7, 9, 11}
- Train a fresh DGEqProp model for 300 epochs at each length
- Evaluate recall

**Input manipulation:** Number of items in the list (L).

**What to plot:**
- MRR vs. list length (expect a drop-off curve)
- Memory Span vs. list length (ideally plateaus near the model's capacity)
- Serial position curves at each length (do they resemble human U-shaped curves?)

**Expected results:**
- MRR decreases as L grows beyond the model's capacity
- The Span metric should saturate rather than grow linearly, revealing a hard capacity limit

**Evaluation metrics:** MRR, Memory Span

---

### 3.3 Multiple Sequences (Catastrophic Forgetting)

**Cognitive motivation:** Proactive interference (PI) — learning a new list disrupts memory of previously learned lists. This is the direct cognitive analogue of catastrophic forgetting in neural networks, and the core challenge targeted by DGEqProp's design.

**Setup:**
1. **Phase 1:** Train on list A for 300 epochs; record `MRR_A_before`
2. **Phase 2:** Train on list B for 300 epochs (*on the same model*, weights retained)
3. **Phase 3:** Re-test list A; record `MRR_A_after`

Weights are **not reset** between phases. The degradation in recall of list A is the measure of interference / forgetting.

**Input manipulation:**
- Number of interfering lists (start with 1, extend to 2, 3, …)
- Semantic similarity between lists (same category = harder to separate)

**What to plot:**
- Bar chart: `MRR_A_before` vs. `MRR_A_after` (with Δ-MRR labelled)
- Serial position curves before and after interference (which positions degrade most?)
- Δ-MRR as a function of number of interfering lists (forgetting accumulation curve)

**Expected results:**
- Δ-MRR < 0 (forgetting occurs), but DGEqProp's pattern separation should limit the damage
- High-similarity lists should cause more interference than low-similarity lists

**Evaluation metrics:** Δ-MRR, Recall Fidelity (before and after), Memory Span (before and after)

---

### 3.4 Noise Invariance

**Cognitive motivation:** Real-world recall cues are always degraded — by noise, distraction, or partial memory. This dimension tests how gracefully the model degrades as retrieval-cue quality decreases, analogous to the phonological noise effect in human cognition.

**Setup:**
- Train on a fixed 7-word list for 300 epochs
- At recall time, add Gaussian noise to each cue embedding: `noisy = clean_cue + N(0, σ²)`
- Sweep σ from 0 (no noise) to 1.0 (very noisy)

**Input manipulation:** `noise_scale` σ ∈ `np.linspace(0, 1.0, 20)`

> **Note on noise type:** Additive Gaussian noise (*feature corruption*) is the primary approach here — every embedding dimension is slightly perturbed, modelling degraded sensory input. An alternative is *dropout-style masking* (zeroing a fraction of dimensions), which models feature *deletion* rather than corruption. Deletion is a more severe test: the information is simply absent, not just noisy. This is reserved for a future extension.

**What to plot:**
- Heatmap: σ (y-axis) × serial position (x-axis) → recall rate (colour)
- Single line: mean MRR vs. σ (noise tolerance frontier)
- Mark the Noise Tolerance Threshold (σ at MRR = 0.5) with a vertical line

**Expected results:**
- MRR degrades monotonically with σ, with a sigmoidal shape
- DGEqProp's sparse DG codes should provide robustness (flatter curve) compared to a standard Hopfield baseline
- Later serial positions degrade faster than early ones (primacy advantage persists under noise)

**Evaluation metrics:** MRR vs. σ curve, Noise Tolerance Threshold, Recall Fidelity vs. σ

```python
def noise_tolerance_threshold(mrr_curve, noise_levels, target_mrr=0.5):
    """σ at which MRR first drops below target_mrr."""
    for sigma, mrr in zip(noise_levels, mrr_curve):
        if mrr < target_mrr:
            return sigma
    return noise_levels[-1]   # never drops below target
```

---

### 3.5 Semantic Similarity

**Cognitive motivation:** The *semantic similarity effect* in verbal learning: lists composed of items from the same category are harder to recall than lists of semantically diverse items, because similar items compete at retrieval. Hippocampal pattern separation (modelled by the DG) is hypothesised to mitigate this.

**Two sub-experiments:**

#### 3.5a List Composition (Categorical)

| Condition | List | Mean pairwise cosine similarity |
|---|---|---|
| High similarity | All fruits: `apple, banana, orange, grape, pear, peach, plum` | High (~0.9 with low `category_variance`) |
| Low similarity | One item per category: `apple, cat, car, hammer, red, seven, …` | Low (~0.1) |

- Train a fresh DGEqProp model on each condition for 300 epochs
- Compare MRR and Memory Span across conditions

#### 3.5b Parametric Sweep via `category_variance`

The `SymbolicEncoder(category_variance=v)` directly controls how tightly category embeddings cluster.  
Lower `v` → tighter clusters → higher within-category cosine similarity.

```python
variances = [0.05, 0.1, 0.2, 0.5, 1.0]
for v in variances:
    enc = SymbolicEncoder(vocab, embedding_dim=100, category_variance=v, seed=42)
    # measure mean pairwise cosine similarity of the list
    # train model, measure recall
```

Plot MRR vs. measured mean pairwise cosine similarity (x-axis is the *actual* embedding geometry, not the parameter value).

**What to plot:**
- Bar chart: MRR per condition (3.5a)
- Line plot: MRR vs. mean pairwise cosine similarity (3.5b)
- t-SNE of DG activations: do the sparse codes remain well-separated even when input embeddings cluster?

**Expected results:**
- Higher within-list cosine similarity → lower MRR (semantic similarity effect)
- DGEqProp's DG should attenuate this effect relative to a baseline (smaller drop in MRR for the same similarity increase)

**Evaluation metrics:** MRR, Memory Span, Recall Fidelity; also `calibrate_theta()` to confirm DG-level separation

---

### 3.6 Interval / Interference

**Cognitive motivation:** The Brown–Peterson paradigm: recall degrades when a distractor task fills the delay between study and test. This dimension tests retention as a function of delay or interfering activity. It bridges the noise invariance test (passive decay) and the multiple-sequences test (active interference).

**Two flavours:**

#### 3.6a Passive Decay (Noise as Proxy)

Temporal decay is modelled by increasing cue noise proportionally to delay length (since the network's weights do not drift post-training):

```python
def recall_after_delay(network, words, encoder, decoder,
                        delay_steps=range(0, 20), decay_rate=0.05, n_trials=30):
    results = {}
    for d in delay_steps:
        sigma = decay_rate * d
        r = measure_recall_associative(network, words, encoder, decoder,
                                        noise_scale=sigma, n_trials=n_trials)
        results[d] = r
    return results
```

**Input manipulation:** `delay_steps` ∈ {0, 1, …, 19}; `sigma = 0.05 × delay_steps`.

#### 3.6b Active Interference (Interleaved Training)

Between study and test, run additional training steps on a noise list (random embeddings) or an irrelevant semantic list. This more directly models distractor activity that *modifies* the network's weights.

**Input manipulation:** Number of interfering training steps ∈ {0, 50, 100, 200, 500}.

**What to plot:**
- Line: MRR vs. delay steps (3.6a) or vs. interfering training steps (3.6b)
- Fit a linear regression to extract the **Forgetting Slope** (MRR per unit delay)
- Serial position curves at delay = 0, 5, 10, 20 (which positions decay fastest?)

**Expected results:**
- MRR decreases monotonically with delay / interference
- Recency items decay faster (they depend more on short-term signal), while primacy items are more stable
- Active interference (3.6b) should cause faster decay than passive noise (3.6a)

**Evaluation metrics:** MRR vs. delay curve, Forgetting Slope, Memory Span at each delay point

---

## 4. Implementation Notes

- All recall measurements use `n_trials = 30` by default (increase to 50 for publication figures)
- Ground-truth (GT) embeddings = `encoder.encode(words)` with no noise added
- θ for Memory Span is fixed at **0.75** throughout for consistency
- The `calibrate_theta()` utility can be used to set the cosine threshold for `measure_recall_threshold` in a data-driven way (midpoint between within-category and cross-category similarity)
