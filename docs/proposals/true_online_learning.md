# True Online Learning — The Raw Signal & What It Requires

> **Note (2026-09-27):** the `standalone_*.py` scripts cited here were retired
> from the repo; recover any of them with `git show archive/pre-cleanup:<script>`.

> Companion to `docs/background/online_continual_benchmark.md`.
> That document describes the benchmark *with* the theta abstraction. This one
> strips the abstraction away: it specifies the **raw serial signal** as it
> actually arrives in experiment and behavior, and enumerates the mechanisms a
> model must have to learn from it directly.

The current benchmark lives **one stage downstream** of raw experience: it
consumes the theta-compressed replay (`A B C | A B C | …`) rather than the raw
stream. "True online" means moving that stage *inside* the model — feeding it the
un-abstracted signal and requiring it to do the compression/segmentation itself.

---

## 1. The raw signal

Real experience arrives as **serial dwelling**, not as a clean repeatable
sequence:

```
raw stream:   A A A A   B B   C C C C C   B B B   ...
              └─dwell─┘ └───┘ └──dwell──┘
```

Properties the abstraction hides:

1. **Dwelling.** Each state/stimulus persists for many timesteps (place-field
   occupancy; stimulus duration). The same input repeats before the next.
2. **Variable dwell times.** Occupancy is irregular — an animal lingers at a
   junction, rushes a corridor. There is no fixed `n_presentations`.
3. **One-shot, irregular transitions.** In a free trajectory (maze, random walk)
   a transition may occur once, and the trajectory is not a repeated list.
4. **No hand-cued boundaries.** Nothing external announces "event over, reset
   now." Boundaries must be *detected* from the stream.
5. **No pre-compression.** The forward theta sweep is not given; if wanted, it
   must be generated.

Canonical sources of this signal: stimulus-presentation paradigms (item shown for
a duration), and — the key case for this project — an animal **freely exploring a
maze** (`standalone_tmaze.py`, `standalone_random_walk.py`,
`generators/bifurcating_route.py`).

---

## 2. Why the current learning rule breaks on the raw signal

The EP core learns transitions from the **immediately previous event**. Fed raw
dwelling, `fit_event` sees:

```
A A A B B B  →  A→A, A→A, A→B, B→B, B→B, ...
```

i.e. it is **flooded with self-transitions** and sees each real transition
(`A→B`) only once. The rule presupposes an already-segmented, already-compressed
input. So on the raw signal, the *unbiological element is the rule*, not the
input — which is exactly what a "true online" model must fix.

---

## 3. Mechanisms required (the missing middle)

Brains receive `A A A B B B` and still learn `A→B→C`. The conversion is done by
mechanisms the current harness abstracts away. A true-online model needs some
subset:

### 3.1 Change / novelty-gated plasticity  *(highest priority)*
Gate the weight update by input change / prediction error. Stable input →
adaptation → little learning; a **change** → large error → plasticity. The
effective update then fires at **transitions**, and dwelling self-repeats are
ignored automatically. Biologically grounded (ACh/DA neuromodulation of learning
rate). For EP: scale the two-phase update by `‖x_t − x_{t-1}‖` (or a
surprise/prediction-error signal), so dwelling contributes ≈0 and boundaries
contribute the real `A→B` update. **This alone lets the raw stream be fed
directly.**

### 3.2 Eligibility trace on the presynaptic factor
Instead of "literal previous event," carry a **decaying trace of the last
distinct state**. Then `A→B` is learnable across the dwell gap and across ISI/noise
timesteps, without B having to be the immediate successor of A. Bridges variable
dwell times. Biologically: synaptic eligibility traces (seconds).

### 3.3 Event-boundary **detection** (not hand-cued)
Replace the manual `on_event_boundary()` with a boundary **detected** from the
stream (surprise / large prediction error / context-change signal). This is what
segments continuous experience into discrete events and decides when to close off
a sweep and reset.

### 3.4 Adaptation / habituation
Input-unit adaptation makes repeated A produce diminishing drive, so learning
naturally concentrates at change points. Complements 3.1.

### 3.5 Dwelling as attractor deepening (dual use of repetition)
Repetition need not be wasted as "noise." Let dwelling on A **deepen A as a stable
attractor** (denoising / working-memory-as-attractor), while the *change* learns
the `A→B` transition. Then `A A A B B B` affords two real functions — a stable
state and a transition out of it — instead of being collapsed by hand. Natural
fit for an energy-based (EP) model.

### 3.6 Theta generation (only if compression is in scope)
If the model is to *produce* the compressed forward sweep rather than assume it,
that requires phase precession / a theta–gamma sequencing mechanism. This is the
most ambitious piece and is optional: change-gating + eligibility traces already
yield correct transition learning from raw dwelling **without** explicit theta
compression.

---

## 4. Concrete first step

Prototype a **change-gated `fit_event`** on `OriginalEqPropSequenceNetwork`:

- maintain the last-distinct-state (eligibility trace) as the presynaptic factor;
- gate the EP update by input change (novelty), so dwelling is suppressed;
- detect boundaries from a drop in change / a context signal instead of an
  external cue.

Validation: feed the **raw serial stream** `A A A B B B C C C` (variable dwell)
and confirm it recovers correct transition learning — ideally matching the recall
of the theta-abstracted harness that was *handed* `A B C | A B C`. If it does, the
model stands on its own without the pre-segmented input.

---

## 5. Open questions

- **How much theta compression is really needed?** If change-gating + traces
  suffice, explicit theta generation (3.6) may be unnecessary for the symbolic
  benchmark — but may matter for long sequences / spatial trajectories.
- **Boundary detection threshold.** How to set the surprise/change threshold that
  segments events, and its interaction with `category_variance` (similar
  consecutive items produce small change signals → under-segmentation).
- **Interaction with continual learning.** Change-gated plasticity concentrates
  updates at transitions — does that change the forgetting dynamics and the EWC
  `λ` tuning relative to the abstracted harness?
- **Maze/exploration transfer.** Fixed `n_presentations` cannot represent variable
  dwell / one-shot trajectories; the true-online mechanisms are what make the
  spatial paradigms (`standalone_tmaze`, `standalone_random_walk`) run natively.

---

## 6. Relationship to the abstracted benchmark

The two are complementary, not competing:

- **Abstracted benchmark** (`docs/background/online_continual_benchmark.md`): consumes the
  theta-compressed sequence; correct and standard for STDP-timescale plasticity;
  the right tool for measuring **retention** cleanly.
- **True online** (this doc): consumes raw dwelling; tests whether the model can
  **generate** the transition signal itself; the right tool for claims about
  learning from **realistic serial experience**, and the prerequisite for the
  spatial/exploration paradigms.
