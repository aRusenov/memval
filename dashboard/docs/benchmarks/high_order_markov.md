# High-Order Markov Recall Benchmark

> **Status: Proposed** — a linearity-breaking benchmark in the *requires-recurrence*
> family. Not yet implemented.

## Biological Context
The hippocampus routinely disambiguates sequences that pass through the *same* state
but demand different continuations — the classic overlapping-route problem. "Splitter"
cells fire differently for the same location depending on where the animal came from or
is going, i.e. recall is conditioned on **history**, not just the current observation.
A memory system that only maps *current item → next item* cannot reproduce this.

## Computational Assay
Sequences are generated from an **order-*k* Markov source** (sweep $k \in \{1, 2, 3\}$)
and/or seeded with **aliased states** — the same observable item appearing at multiple
points with different required successors (e.g. `A B C B D`, where `B` must be followed
by `C` the first time and `D` the second). We sweep the Markov order and the fraction
of aliased states, and measure next-item recall.

## Why it breaks linearity
A first-order associator $x_{t+1} = \varphi(W x_t)$ **cannot** represent an
order-$\geq 2$ transition: the same current item must map to different successors
depending on earlier context, which forces an internal recurrent state to carry that
context. This is the canonical test for emergent recurrent dynamics, and the symbolic
twin of the spatial disambiguation task.

## Metrics Evaluated
- **Order-*k* next-item accuracy** — recall accuracy as a function of Markov order.
- **Disambiguation accuracy** — accuracy specifically at aliased states.
- **Linear-baseline gap** — accuracy above a ridge/linear next-item predictor; near
  zero means the task is being solved linearly and is *not* exercising recurrence.
