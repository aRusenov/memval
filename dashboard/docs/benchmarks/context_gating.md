# Context-Gated (XOR) Transitions Benchmark

> **Status: Proposed** — a linearity-breaking benchmark in the *injected-nonlinearity*
> family. Not yet implemented.

## Biological Context
The same cue can demand opposite responses depending on context — a hallmark of
context-dependent memory and neuromodulatory gating (ACh/DA). Behaviourally this is a
*multiplicative* interaction: context does not add to the cue, it **gates** which
mapping is active. Systems that can only sum cue and context linearly cannot express it.

## Computational Assay
The successor is determined by a **non-linearly-separable combination** of the cue and
a context bit — the transition follows an XOR (or a multiplicative gate) of cue and
context features rather than a weighted sum. We sweep the number of gating contexts and
the degree of non-separability, and measure transition accuracy.

## Why it breaks linearity
XOR / multiplicative gating is **not linearly separable** in the item code: no linear
$W$ can compute it, so the model must recruit a hidden nonlinearity. This isolates the
computation an Asymmetric Hopfield's single linear transition matrix cannot perform,
and that EqProp's hidden layer should.

## Metrics Evaluated
- **Context-gating accuracy** — transition accuracy under the gated mapping.
- **Linear-baseline gap** — accuracy above a linear next-item predictor; a large,
  positive gap is the signature of genuine nonlinear computation.
- **Accuracy vs. number of gating contexts** — how the gap scales as more context-
  conditioned mappings are packed in.
