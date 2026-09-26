## Primary Metrics
| Metric Name          | Measures   | Protocol                    | What it tells us                                            |
| -------------------- | ---------- | --------------------------- | ----------------------------------------------------------- |
| **MRR**              | Capacity   | Associative (Cued)          | Was the sequence successfully memorized?                    |
| **Recall Fidelity**  | Accuracy   | Associative (Cued)          | How clean are the attractor traces?                         |
| **Memory Span**      | Stability  | Autoregressive (Generative) | How long can the model retrieve autonomously?               |
| **Trajectory Drift** | Drift      | Autoregressive (Generative) | Does the coordinate state drift stably or catastrophically? |
| **Error Cascade**    | Resilience | Autoregressive (Generative) | Can the model recover when it stumbles?                     |

## Categories (grouped by primary metrics)

| Group                       | Contributing metrics                                                 | Interpretation                                            |
|-----------------------------| -------------------------------------------------------------------- | --------------------------------------------------------- |
| **Memory quality**          | `convergence_mrr`, `multiple_seq_mrr_before`                         | How well does it recall at peak?                          |
| **Capacity**                | `max_memory_span`, `convergence_span`                                | How many items can it hold?                               |
| **Plasticity / efficiency** | `convergence_epochs` (inverted), `delta_mrr_forgetting` (inverted)   | How fast does it learn? How much does it forget?          |
| **Robustness**              | `noise_tolerance_threshold`, `similarity_effect_mrr_drop` (inverted) | Stability under corruption and interference               |
| **Recall fidelity**         | `recall_fidelity`                                                    | Embedding-level precision, independent of discrete decode |

## Dimensions

| Dimension                   | Status | Input Manipulation                                                                              | Evaluation Metric                                         |
| --------------------------- | ------ | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Presentation Duration**   | ✅      | `epochs` ∈ {100, 200, 300}                                                                      | Memory Span, MRR, Recall fidelity                         |
| **Sequence Length**         | ✅      | List length ∈ {5, 7, 10, …}                                                                     | Memory Span, MRR, Recall fidelity                         |
| **Multiple Sequences**      | ✅      | Presence of interfering list B (weights retained)                                               | Δ-MRR before/after list B (catastrophic forgetting proxy) |
| **Noise Invariance**        | ✅      | noise σ swept 0→1.0 at recall time                                                              | Noise tolerance threshold                                 |
| **Semantic Similarity**     | ✅      | category variance sweep (varying cosine similarity **within** a list and **between** lists)     | MRR vs. mean pairwise cosine similarity                   |
| **Interval / Interference** | ✅      | (a) noise as a proxy for distraction between training steps, and (b) interfering training steps | MRR vs. interval duration; slope of forgetting curve                     |

## Metrics Under Consideration

Candidate low-level metrics not yet in the primary set. The **Portability** column
flags how model-agnostic each one is — i.e. whether it transfers from the EqProp
settling networks to a backprop autoregressive model (e.g. GPT-2). See
`memval/metrics/capacity.py` for reference implementations.

| Candidate metric                                  | Proposed category      | Protocol / probe                          | Portability                              | What it adds that we don't have                                             |
| ------------------------------------------------- | ---------------------- | ----------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------- |
| **Crosstalk matrix** / `diagonal_dominance`       | Separation             | Cued, one-step, scored against *all* stored targets | ✅ General (any next-item predictor)      | *Which* memories collide, per-item (not just aggregate accuracy)            |
| **Synaptic interference** (ΔW / gradient overlap) | Separation             | Weight-space; no recall needed            | 🟡 Concept-general (EP impl; use ∂L/∂θ cosine for backprop nets — "gradient interference") | *Why* they collide — shared synapses; predicts collisions before testing   |
| **Channel MI** — `channel_mi`, `category_information` | Fidelity + Robustness  | Cued under noise sweep → confusion → bits | ✅ General (trivial for token outputs)    | Threshold-free, chance-corrected, graceful; splits *category* vs *item* info |
| **Attractor scan** — spurious rate, black-hole basin frac, norm drift | Dynamical integrity    | Autoregressive free-run from perturbed seeds | 🟠 Partial (loop/collapse analog for GPT-2; energy/norm specifics are EP) | Where free recall *settles* — collapse to a single state, phantom attractors |
| **RSA / CKA** of stored vs recalled geometry (discussed, not yet built) | Separation + Fidelity   | Compare RDMs of stored vs settled states  | ✅ General                                 | Relational fidelity — graceful geometry-distortion signal for forgetting    |

### Biological Fidelity (mechanism-level match — mostly *a priori*)

Unlike the performance axes (more = better), this is a **match-to-reference** axis:
how close the *architecture* is to how brains are built. It is largely a property of
the model, ~constant across training runs, and applies to **any continual-learning
model** (EWC, SI/MAS, replay/CLS, DG, XdG, EP …), not just EqProp. Two layers:

**(a) Mechanistic rubric** — score each constraint 0/1 or graded; model-agnostic:

| Constraint                                   | Biological basis                              | CL models that tend to satisfy it        |
| -------------------------------------------- | --------------------------------------------- | ---------------------------------------- |
| Local credit assignment (no weight transport)| synapses update from locally available signals| EP, Hebbian, target-prop                 |
| Dale's law (sign-constrained units)          | a neuron is excitatory **xor** inhibitory     | rare in ANNs — differentiator            |
| Sparse distributed coding                    | sparse cortical / DG activity                 | DG / k-WTA, XdG                          |
| Online, single-pass, non-i.i.d. stream       | brains learn from an ordered stream           | online/streaming EP                      |
| Bounded storage (no verbatim replay of past) | no episodic raw-data dump                      | generative replay > raw-buffer replay    |
| Consolidation / replay with a neural analog  | hippocampal replay, systems consolidation     | CLS, generative replay                   |
| Synaptic consolidation / metaplasticity      | cascade synaptic states, importance weighting | EWC, SI, MAS                             |
| Context gating / neuromodulation             | ACh/DA gating, attentional context            | XdG, context-dependent gating            |
| Structural plasticity / neurogenesis         | DG adds granule cells over life               | dynamic-architecture CL, DG expansion    |
| Energy-based / equilibrium dynamics          | attractor dynamics                            | EP, Hopfield                            |

**(b) Neural-data match (empirical)** — RSA / encoding-model comparison of hidden
representations against hippocampal or cortical recordings; emergence of place-/grid-
like tuning; whether the pattern-separation dose-response curve matches DG data. This
half is *measured*, and is the more scientifically interesting one for a hippocampal
model. Report as: **rubric score + optional neural-RSA**.

### Cognitive Fidelity (behavior-level match — empirical, two-sided)

> **Principle:** cognitive fidelity is the **distance between the model's degradation
> profile / error structure and the human one — not a performance ceiling.** Being
> superhuman costs fidelity *only if* the model also fails in non-human ways. Raw
> accuracy is not the currency; the **shape of how it breaks** is. Score each paradigm
> as the correlation / normalized distance between the model's curve and the canonical
> human curve, then aggregate. The distance is **two-sided** — overshooting human
> capacity is penalized like undershooting.

Candidate human-memory signatures (⟳ = especially continual-learning-relevant):

| Signature                                | Canonical human pattern                          | Why it's an interesting probe                                   |
| ---------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------- |
| Serial position                          | U-curve: primacy **and** recency                 | tests order memory, not just item content                       |
| Capacity limit + chunking                | plateau ~4 (Cowan) / 7±2 (Miller)                | the classic — must **emerge**, not be capped                    |
| Forgetting curve (Ebbinghaus)            | power-law / exponential decay                    | match the *shape*, not just the rate                            |
| ⟳ Graceful vs. catastrophic forgetting   | humans degrade gradually — **no cliff**          | the core CL signature: a faithful model shouldn't fall off a cliff |
| ⟳ Proactive / retroactive interference   | old impairs new & vice versa; release-from-PI on category shift | the defining CL phenomenon, with a known human profile |
| Semantic / DRM intrusions                | false recall of category-consistent lures        | we **already** measure category-consistent errors (crosstalk / category-MI) |
| Spacing effect                           | distributed practice > massed                    | probes consolidation dynamics                                   |
| Fan effect (Anderson)                    | retrieval slows as associations-per-cue grow     | an interference signature at retrieval                          |
| Transposition gradient (serial recall)   | errors are mostly adjacent-position swaps        | fine structure of order memory                                  |

**Methodology guardrails:**
- **Emergence > fit > capping.** A human-like limit that falls out of biological
  constraints with *no* fitting is the strong result; fitting 1–2 free params to match
  a *curve shape* is standard cognitive modeling; post-hoc capping a scalar (add noise
  until span = 7) is the weak end and is *evidence of low* cognitive fidelity. Report
  which regime a match came from.
- **Never put the human metric in the training loss** — that makes the evaluation
  circular (benchmark overfitting). Keep it strictly held-out.

**Unifying thesis:** biological constraints should *cause* cognitive signatures —
limited, noisy, sparse, locally-learning hardware should yield ~7±2 spans, primacy/
recency, and graceful forgetting *for free*. Demonstrating mechanism → behavior is the
project's win condition, and it links these two axes into one causal story.

### Proposed high-level dimension refinement

The current **Categories** are a good start but have two issues worth fixing before
they become radar axes:

1. **`Plasticity / efficiency` conflates two opposite things.** Learning *speed*
   (plasticity) and forgetting *resistance* (stability) are the two ends of the
   stability–plasticity trade-off — they should be separate axes, not averaged into
   one. `delta_mrr_forgetting` is a *retention* signal, not an efficiency one.
2. **No axis captures representational structure or recall dynamics** — exactly the
   gap the 4 candidate metrics fill.

Refined set (higher = better on every axis; invert metrics marked *inv*):

| Dimension (radar axis)   | High-level concept                                   | Contributing primary metrics (existing + candidate)                                   | Portable? |
| ------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------------- | --------- |
| **Fidelity**             | Correctness & precision at peak, ideal cue           | `convergence_mrr`, `recall_fidelity`, clean `I(item)`                                  | ✅        |
| **Capacity**             | How fidelity holds as *load* grows                   | `max_memory_span`, `convergence_span`, capacity curve N\*                              | ✅        |
| **Robustness**           | How fidelity holds under *cue corruption*            | `noise_tolerance_threshold`, `I(X;Y)` vs σ (σ_c), `error_cascade`                      | ✅        |
| **Completeness**         | How fidelity holds under *cue partiality* — features missing rather than degraded | `mask_random_tolerance`, `mask_*_recall_in_bound`, `mask_random_margin_in_bound`, `mask_block_asymmetry` (fingerprint, not scored) | ✅        |
| **Retention**            | Resistance to *catastrophic forgetting* from new learning | `delta_mrr_forgetting` *inv*, interference-interval slope *inv*                    | ✅        |
| **Separation**           | Distinctness of stored representations (mechanism)   | `diagonal_dominance`, synaptic interference *inv*, `similarity_effect` *inv*, RSA      | 🟡        |
| **Dynamical integrity**  | Does free-running recall stay well-behaved?          | `trajectory_drift` *inv*, spurious rate *inv*, `dominant_basin_frac` *inv*, `memory_span` | 🟠     |
| **Efficiency** (opt.)    | Cost to learn / recall                               | `convergence_epochs` *inv*, bits-per-synapse                                           | ✅        |
| **Biological fidelity**† | Mechanism match to the brain                         | mechanistic-rubric score, neural-RSA                                                   | ✅ (any CL model) |
| **Cognitive fidelity**†  | Behavioral match to human limits & errors            | aggregate paradigm-distance (serial position, span, forgetting shape, DRM, interference) | ✅        |

† **Match-to-reference axes, not monotonic.** The other axes are "more = better";
these two score *distance to a reference* (brain / human), where being *too good* is a
deviation. Mixing them onto the same spider as the performance axes can mislead — a low
Cognitive-fidelity score can mean "superhuman," which reads wrongly next to a high
Capacity score. Recommend a **separate "plausibility" radar** (Biological × Cognitive ×
sub-signatures) alongside the performance radar. That said, the *tension between them*
is exactly the informative case: high Capacity + low Cognitive fidelity = "superhuman,
not human-like."

**Validity / orthogonality notes for the radar:**
- **Separation is a *driver* of Capacity & Retention**, so expect positive
  correlation between those axes — the polygon will bulge together there. Keep it as
  its own axis anyway: it's the *mechanistic* "why," measurable independently, and
  it's the axis your DG / pattern-separation work moves most directly.
- **Dynamical integrity** partially overlaps Robustness and Capacity (via
  `memory_span`) but isolates a distinct failure mode — free-running collapse (the
  "everything → cow" black-hole) — that step-wise metrics never see.
- Recommend **6 axes** for a readable spider chart (fold Efficiency in only if
  compute cost is a research question); drop or merge axes that come out near-perfectly
  correlated once you have data across the model zoo.

### Radar (spider) chart scoring recipe

To collapse each dimension to one [0, 1] score for the plot:
1. Orient every contributing metric so **higher = better** (invert the *inv* ones).
2. **Normalize** each metric to [0, 1] — min–max across the models being compared, or
   against a fixed reference model, so axes are commensurable.
3. **Average** (optionally weighted) the normalized metrics within a dimension → axis score.
4. Plot the 6 axis scores per model; overlay models to compare at a glance.
