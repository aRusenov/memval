# Cyclic Free-Running Benchmark

> **Status: Proposed** — a linearity-breaking benchmark in the *requires-recurrence*
> family. Not yet implemented.

## Biological Context
Many biological sequences are **periodic** — locomotor rhythms, song motifs, rehearsed
mnemonic loops. Sustaining such a sequence during autonomous replay requires dynamics
that hold a stable **limit cycle**, returning to the same trajectory after small
perturbations rather than decaying to a fixed point or blowing up.

## Computational Assay
The model encodes a repeating-motif sequence (e.g. `A B C A B C …`). After encoding it
is released to **free-run autoregressively** — its own output becomes the next cue —
and we measure how long the cycle is sustained before it breaks phase or drifts. This
feeds directly into the autoregressive **Memory Span** and attractor-scan metrics.

## Why it breaks linearity
A linear map cannot robustly sustain a limit cycle: its iterates decay or diverge
unless the relevant eigenvalues sit *exactly* on the unit circle, which is not
achievable in practice. A stable cycle therefore requires genuine **nonlinear
recurrent dynamics** — precisely the regime an Asymmetric Hopfield's linear transition
operator cannot occupy.

## Metrics Evaluated
- **Stable-cycle span** — number of autoregressive steps before phase break or drift.
- **Cycle purity** — fraction of free-run steps that stay on the intended motif.
- **Attractor-scan spurious rate** — how often free-running settles onto a phantom
  cycle or collapses to a single state.
