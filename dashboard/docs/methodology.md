# Evaluation Methodology

How MemVal scores a model's sequence memory. Every benchmark ultimately feeds a
small set of **primary metrics**, which roll up into interpretable **categories**,
and are probed by systematically varying task **dimensions**. This page defines all
three layers.

## Two evaluation protocols

Every primary metric is collected under one of two recall protocols:

- **Associative (Cued)** — the model is given a cue and asked to retrieve the
  associated item in a single step. This isolates *storage quality*: was the pair
  memorized, and how cleanly?
- **Autoregressive (Generative)** — the model free-runs, feeding its own output back
  as the next cue. This exposes *dynamical* failure modes — drift, collapse, and the
  ability to recover — that single-step probes never see.

---

## Primary Metrics

The core scores reported for every model. Each names what it *measures*, the protocol
it is collected under, and the question it answers.

| Metric | Measures | Protocol | What it tells us |
| ------ | -------- | -------- | ---------------- |
| **MRR** | Capacity | Associative (Cued) | Was the sequence successfully memorized? |
| **Recall Fidelity** | Accuracy | Associative (Cued) | How clean are the attractor traces? |
| **Memory Span** | Stability | Autoregressive (Generative) | How long can the model retrieve autonomously? |
| **Trajectory Drift** | Drift | Autoregressive (Generative) | Does the coordinate state drift stably or catastrophically? |
| **Error Cascade** | Resilience | Autoregressive (Generative) | Can the model recover when it stumbles? |

---

## Categories

Primary metrics are grouped into higher-level categories so a model can be read at a
glance. Each category aggregates one or more low-level metrics (metrics marked
*inverted* are flipped so that higher is always better).

| Category | Contributing metrics | Interpretation |
| -------- | -------------------- | -------------- |
| **Memory quality** | `convergence_mrr`, `multiple_seq_mrr_before` | How well does it recall at peak? |
| **Capacity** | `max_memory_span`, `convergence_span` | How many items can it hold? |
| **Plasticity / efficiency** | `convergence_epochs` *(inv)*, `delta_mrr_forgetting` *(inv)* | How fast does it learn? How much does it forget? |
| **Robustness** | `noise_tolerance_threshold`, `similarity_effect_mrr_drop` *(inv)* | Stability under corruption and interference |
| **Recall fidelity** | `recall_fidelity` | Embedding-level precision, independent of discrete decode |

---

## Dimensions

Each dimension is an independent axis of task difficulty that we manipulate on the
input side and observe through the evaluation metrics above. All dimensions below are
currently active (✅).

| Dimension | Input manipulation | Evaluation metric |
| --------- | ------------------ | ----------------- |
| **Presentation Duration** | `epochs` ∈ {100, 200, 300} | Memory Span, MRR, Recall fidelity |
| **Sequence Length** | List length ∈ {5, 7, 10, …} | Memory Span, MRR, Recall fidelity |
| **Multiple Sequences** | Presence of interfering list B (weights retained) | Δ-MRR before/after list B (catastrophic-forgetting proxy) |
| **Noise Invariance** | Noise σ swept 0 → 1.0 at recall time | Noise tolerance threshold |
| **Semantic Similarity** | Category-variance sweep (cosine similarity *within* and *between* lists) | MRR vs. mean pairwise cosine similarity |
| **Interval / Interference** | (a) noise as a proxy for distraction between training steps; (b) interfering training steps | MRR vs. interval duration; slope of the forgetting curve |

---

## Evaluation Regimes

A dimension is only meaningful for a model whose training regime can express it.
Two regimes matter:

- **Offline (discretized)** — the whole sequence is presented repeatedly over epochs,
  weights settle, then recall is tested. This is how the current three models train:
  **Asymmetric Hopfield**, **pure EqProp**, and **GPT-2**. Only *structural* knobs
  apply here — anything you can encode in token-space or in the training set, with no
  real clock. "Spacing between items" for these models means **filler tokens**, not
  milliseconds.
- **Online (streaming)** — items arrive one at a time on a non-i.i.d. stream and are
  learned single-pass in real time (the theta-phase EqProp direction). This unlocks a
  whole **temporal** family of dimensions that is simply undefined offline.

## Expanded Dimensions

The dimensions above are mostly *magnitude* knobs. The table below adds *structure*
and *timing* knobs and tags each with its applicable **regime** and whether it is a
**linearity-breaking** probe — one designed so that a linear, first-order associator
(what an Asymmetric Hopfield essentially is) provably fails.

| Dimension | Regime | Breaks linearity | Probes / metric |
| --------- | ------ | ---------------- | --------------- |
| Presentation Duration | Both | — | Span, MRR, Recall fidelity |
| Sequence Length | Both | — | Capacity |
| Multiple Sequences | Both | — | Retention / forgetting |
| Noise Invariance | Both | — | Robustness |
| Semantic Similarity | Both | — | Separation (embedding) |
| **Vocabulary / alphabet size** | Both | — | Capacity (alternate axis) |
| **Code sparsity / dimensionality** | Both | — | Separation (DG lever) |
| **Item frequency (Zipf)** | Both | — | Frequency-dependent recall |
| **Positional gaps (filler tokens)** | Offline | — | Order robustness, structural spacing |
| **Grammar / predictability (entropy)** | Both | ✓ *(structure)* | Chunking, capacity emergence |
| **Item overlap / shared subsequence** | Both | ✓ *(recurrence)* | Separation, disambiguation, fan effect |
| **High-order Markov (order ≥ 2)** | Both | ✓ *(recurrence)* | History-dependent recall |
| **Context-gating / XOR transitions** | Both | ✓ *(nonlinearity)* | Non-separable binding |
| **Delayed / n-back dependency** | Both | ✓ *(recurrence)* | Cross-gap working memory |
| **Cyclic / limit-cycle free-running** | Both | ✓ *(recurrence)* | Sustained recurrent dynamics |
| **Cue fraction / partial-cue completion** | Both | — | Graded pattern completion |
| **Inter-item interval / rhythm / jitter** | Online | — | Timing, tempo invariance |
| **Wall-clock retention interval** | Online | — | Ebbinghaus forgetting shape |
| **Streaming non-i.i.d. order** | Online | — | Online catastrophic forgetting |
| **Single-pass / one-shot exposure** | Online | — | One-shot encoding pressure |
| **Novelty / surprise-gated plasticity** | Online | — | Neuromodulatory write-gating |

## Linearity-Breaking Benchmarks

An Asymmetric Hopfield network with Hebbian storage is a **linear first-order
transition operator with a pointwise nonlinearity at readout**
($x_{t+1} = \varphi(W x_t)$, $W = \sum_t \xi_{t+1}\xi_t^{\top}$). On orthogonal
codes with first-order-Markov, linearly-decodable sequences — which describes most of
the current symbolic suite — that is not just adequate but optimal and cheap. This is
*why* Hopfield can outperform EqProp: **the benchmarks don't require anything EqProp's
hidden layer and settling dynamics are for.** A linear map provably fails on exactly
two families, which motivate the proposed benchmarks:

- **(a) Injected nonlinearity** — transitions that are *not linearly separable* in the
  item code, so a hidden nonlinearity is required.
  → [Context-Gated (XOR) Transitions](#benchmark/context_gating)
- **(b) Emergent recurrent dynamics** — tasks where the successor is *not a function of
  the current observable alone*, so a hidden state carrying context is required.
  → [High-Order Markov Recall](#benchmark/high_order_markov),
  [Delayed Dependency (n-Back)](#benchmark/delayed_recall),
  [Cyclic Free-Running](#benchmark/limit_cycle)

**Linear-baseline control.** Every symbolic benchmark should be run against a plain
linear next-item predictor (ridge regression / a single linear layer). If the linear
baseline matches a model, the task provably does *not* require nonlinearity — a cheap
diagnostic that exposes which existing benchmarks are secretly trivial. Report the
**linear-baseline gap** (`accuracy_model − accuracy_linear`) alongside each score.

---

*This page summarizes the primary metrics, categories, and dimensions. Additional
candidate metrics and the biological/cognitive fidelity axes are tracked in the
[full metrics reference](docs/benchmarks/_general_metrics.md).*
