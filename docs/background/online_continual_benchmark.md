# Online-Continual Benchmark — Design, Rationale & Abstractions

> Companion to `memval/benchmarks/online_symbolic_pipeline.py`.
> Model of record: `OriginalEqPropSequenceNetwork` (+ EWC variants).
>
> **Status (2026-09-01):** `online_continual_pipeline.py` was **removed** as a
> suite (`docs/capacities/capacity_coverage_audit.md` D8). The retention matrix and its
> summary statistics were first **extracted** into
> `memval/metrics/retention.py` (`retention_matrix`, `retention_summary`,
> `plot_retention_matrix`), so any suite section can now report the same
> quantity. `continual_chain` in the symbolic suite is the live consumer; the
> standalone `bin/continual_chain_experiment.py` and the EWC variants were retired
> on 2026-09-27 (git tag `archive/pre-cleanup`).
>
> Two things did *not* survive the removal and are open: the
> streaming-vs-batch **ingestion contrast** (the pipeline was the only place it
> ran), and the retention matrix as a **suite section** — nothing in
> `--suite spatial` or `--suite symbolic` computes one yet. The protocol below
> still describes what such a section must do.

This document records **what the online-continual benchmark does, why it is
built that way, and — critically — which biological realities it abstracts
over**. It is the "with-abstraction" companion to `docs/proposals/true_online_learning.md`,
which specifies the raw, un-abstracted signal and what a fully realistic
"true online" version would have to address.

---

## 1. What the benchmark measures

A single model learns a **sequence of word lists** (tasks) one after another,
with **no weight reset between tasks**, and after each new list we re-test recall
of *every* list seen so far. This is the standard continual-learning evaluation.

- **Retention matrix** `R[j, i]` = recall (MRR) on task *i* after training through
  task *j*. Lower-triangular (a task can't be scored before it's seen).
  - Diagonal `R[i,i]` — how well task *i* was learned when fresh.
  - Bottom row `R[T-1, i]` — what survives of each task at the end.
  - Down a column — how one task decays as later tasks overwrite it.
- **Summary metrics**: Average Accuracy (ACC), Backward Transfer (BWT; negative =
  forgetting), Average Forgetting (drop from each task's best-ever score).
  All three come from `memval.metrics.retention.retention_summary`.

Orientation note: the shared module uses `R[j, i]` — rows are training stages,
columns evaluated tasks. `bin/continual_chain_experiment.py` previously used the
transpose locally and now reports this orientation.

Two orthogonal axes are in play:

| Axis | Meaning | Knob |
|---|---|---|
| **Ingestion** | one-event-at-a-time streaming vs. batch epochs | `online=True` → `fit_event`; else `fit_sequence` |
| **Data regime** | stationary single task vs. non-stationary task chain | number of task lists |

The removed `online_continual_pipeline.py` occupied the *streaming ×
non-stationary* corner — the hardest one, and the only place the ingestion axis
was ever varied. `bin/continual_chain_experiment.py` covers *batch ×
non-stationary*; nothing currently covers the streaming column, which is the
main thing D8 cost.

---

## 1b. How the regime is selected: `OnlineTrainable` and `ingest`

> The *retrieval* counterpart of this section — what each arm's `recall()`
> feeds back at each autoregressive step — is `docs/sections/rollout_protocol.md`.
> How the harness *asks* — clean single cue vs a declared degradation
> sweep — is `docs/sections/probe_protocol.md`.

The ingestion axis is a **condition inside a suite**, never a property of the
arm — an arm that supports both regimes must be runnable under both. So there is
one `HippocampalModel` base class, not separate online/offline hierarchies. What
*is* a property of the arm is which optional interfaces it implements, and that
is declared nominally in `memval/models/capabilities.py`:

```python
class ThetaPhaseSequenceNetwork(HippocampalModel, OnlineTrainable):
    online_equivalent = True
```

`fit_event` and `on_event_boundary` live on `OnlineTrainable`, **not** on
`HippocampalModel`. This is load-bearing: the base class used to carry a
concrete `fit_event` stub that raised `NotImplementedError`, which made every
structural check (`hasattr`, a `runtime_checkable` Protocol, an
`@abstractmethod`) report *true for every arm*. A concrete method anywhere in
the MRO satisfies the capability. Probe with `supports_online(arm_or_class)`.

Every suite that varies the regime goes through one seam,
`memval.benchmarks.ingest`:

```python
ingest(model, sequence, regime="streamed")   # fit_event per timestep, both seams fenced
ingest(model, sequence, regime="batch")      # one fit_sequence call
```

Streaming an arm that has not declared the capability raises `UnsupportedRegime`.
A benchmark must record that as **not applicable**, never as a score — the
online-symbolic pipeline previously caught the `NotImplementedError` per sweep
point and wrote `0.0`, so "no streaming interface" and "streamed and recalled
nothing" were the same number on disk, and `bin/score_capacities.py` had to
un-do it afterwards from a hardcoded table. The pipeline now checks the
capability up front and writes `metadata.status = "not_applicable"` with no
metrics at all.

### `online_equivalent`

Declares whether `fit_sequence` applies *exactly* the per-transition rule that
streaming applies. Read it before interpreting any batch-vs-streamed contrast:

| Arm | `online_equivalent` | Why |
|---|---|---|
| `DTSESNSequenceNetwork` | True | `fit_sequence` is a loop over `fit_event` |
| `ThetaPhaseSequenceNetwork` | True | same `_cycle_delta` per transition |
| `TemporalPCNetwork`, `MultilayerTemporalPCNetwork` | True | same `_step` / `_transition_delta` |
| `OriginalEqPropSequenceNetwork`, `SpikingEqPropSequenceNetwork` | **False** | batch averages by `1/n_transitions` and steps once per epoch; `fit_event` takes a full step per transition |
| `HopfieldSequenceNetwork` | **False** | config-dependent: `projection` matches, `hebbian` never does, `delta` matches except under `sign` |

When it is False, an online-vs-offline difference confounds ingestion with a
change of learning rule. That matters most for the EP arms, which are the ones
the suite is built around. The True claims are verified numerically in
`tests/test_online_capability.py`, not merely asserted.

---

## 2. What the model actually learns: transitions

The EP core learns **transitions** `(x_t → x_{t+1})`, not items. `fit_event(x)`
buffers the previous event and, when a previous event exists, applies one EP
update for the pair `(prev → x)`. This single fact determines the correct shape
of the training loop (§3) and the meaning of the event boundary (§5).

---

## 3. The presentation loop (`stream_sequence`)

For each of `n_presentations` repetitions, the **whole list** is streamed
item-by-item, then `on_event_boundary()` is called:

```
present 1:  A B C |
present 2:  A B C |
...
present N:  A B C |          (| = on_event_boundary → reset)
```

**Why the whole sequence is repeated (not each item):** because the unit of
learning is the transition. Repeating `A B C` re-presents the real transitions
`A→B`, `B→C` every pass. The naive alternative — `A A A B B B C C C` (each item
N times) — would instead teach mostly **self-transitions** (`A→A`, `B→B`) and see
each real transition only once. Repetition must live at the *sequence* level
because the learned object spans two items.

**Interleaved, not blocked.** Cycling the whole list interleaves the transitions,
which is the *lower*-interference schedule. Blocking (all of `A→B`, then all of
`B→C`) would maximize catastrophic interference within a single list.

---

## 4. The central abstraction: presentations ≈ theta-compressed replay

The raw stream of experience is **serial dwelling** — an animal occupies location
A for many timesteps before moving to B (`A A A … B B B … C C C`); a stimulus is
shown for a duration before the next. The benchmark does **not** feed that. It
feeds the repeated compact sweep `A B C | A B C | …`. The justification:

- Behavioral-timescale dwelling does not reach synapses at that timescale.
  During a ~1–2 s traversal there are ~8–16 **theta cycles**, each carrying a
  *compressed forward sweep* of the local sequence (phase precession → theta
  sequences; the offline analog is sharp-wave-ripple **replay**).
- STDP-like plasticity operates **within theta cycles**, so the signal that
  actually drives learning is the theta-compressed replay — precisely
  `A B C` swept repeatedly.

So the mapping is:

| Benchmark construct | Biological correlate |
|---|---|
| one **presentation** (one `A B C` sweep) | one **theta cycle** / compressed sweep |
| `n_presentations` | number of theta cycles = **encoding duration** |
| the whole training block for a list | the **stimulus-presentation / encoding period** |
| `on_event_boundary()` after each sweep | **theta phase reset** per cycle |

Under this reading the model consumes the biologically correct input for a
transition-learning rule, and the "why not raw dwelling?" objection is answered:
the benchmark lives **one stage downstream** of raw experience, at exactly the
stage STDP sees. (Precedent: theta sequences — Skaggs/McNaughton; Jensen &
Lisman; replay — Foster & Wilson.)

### Honest boundary of the abstraction

- The model **consumes** the theta-compressed sequence; it does **not generate**
  it. Phase precession, the theta–gamma code, and the segmentation that produce
  the sweeps are upstream and assumed-as-given. This is a standard, defensible
  move, but it must be stated, not hidden.
- Real theta sweeps cover a **window** around the current position, not
  necessarily the whole list. "Whole sequence per sweep" is a fine approximation
  for short (~7-item) lists and would need revisiting for long sequences.
- The path that *does* raw-dwelling → sweeps (event segmentation, novelty-gated
  plasticity, eligibility traces) is the subject of `docs/proposals/true_online_learning.md`.

---

## 5. Event-boundary / reset semantics

`on_event_boundary()` defaults to `reset_context()`, which for
`OriginalEqPropSequenceNetwork`:

1. clears the **previous-event buffer** (`_prev_event = None`), and
2. zeroes the transient recall state (`current_state`).

Its **computational job** is (a) to prevent a **wrap-around transition** from
forming across the seam (`plum → apple` between sweeps, or `plum_A → cat_B`
between two different lists), and (b) to restart the sweep. It fires after
**every presentation**, so it maps to the per-theta-cycle phase reset, not
exclusively to the between-stimulus event boundary — the true event boundary
(list A → list B) is handled by the *same* mechanism.

Note: in this particular model there is **no persistent recurrent activity across
events** (each transition settles from zero; `predict_next` is a pure function of
the clamped event). So "activity ceases at the boundary" is a fair gloss, but
literally the reset only closes off and restarts the sweep. If a recurrent
context state is ever added, "activity ceases at the boundary" becomes a
substantive claim worth modeling deliberately.

---

## 6. Online vs. offline update schedule (and the fresh-MRR ceiling)

Freshly-learned lists top out around **MRR ≈ 0.6–0.8** online, vs. **≈ 0.9**
offline with the same weights and config. This is **not** under-exposure
(`n_presentations` is 20–30, not one-shot). It is the **update schedule**:

- **Online** applies one **un-averaged** EP update per transition. Within a sweep,
  a later transition partially overwrites the shared hidden→output readout that an
  earlier one just wrote — the list fights itself, settling into a compromise
  equilibrium.
- **Offline** `fit_sequence` **averages** the gradient over all transitions and
  applies it once, so the transitions are optimized jointly → higher ceiling.

Levers to raise the fresh-MRR ceiling (each keeps the streaming interface):
gradient accumulation over a presentation (dial toward offline), better
per-step EP-gradient quality (`n_settle_steps` at equilibrium, `beta` /
symmetric two-sided nudge), and hidden-layer capacity. Worth doing: a higher
fresh baseline sharpens every downstream forgetting number.

---

## 7. Key results (as of 2026-07-08)

Config: `n_hidden=128`, `activation=tanh`, `output_activation=tanh`
(the 100-dim symbolic embeddings are ~51% negative, so the default [0,1] state
clip caps MRR ~0.36 — tanh output is required), `beta=0.3`, `lr=0.02`,
`n_presentations≈30`, 3 seeds.

- **EWC-on-EP works and scales.** 5-task streaming chain: forgetting **0.60 → 0.30**,
  old-task retention **0.08 → 0.30**. This is the headline positive result.
- **Online EWC needs `λ ≈ 1000`** — ~10× smaller than batch's 3e4, because the
  penalty fires per-event (~180×/task) not per-epoch, and it must scale *down* as
  the chain lengthens (Fisher accumulates / per-task springs sum).
- **Per-task anchors give no advantage** over a single running anchor on
  orthogonal-task chains up to length 5. The drifting-anchor pathology needs
  *interfering* (similar) tasks to bite — the untested regime.

Durable outputs: `results/online_continual_comparison/` (3-task) and
`results/online_continual_chain5/` (5-task), each with a `*_summary.json` plus
per-(config, seed) retention matrices.

---

## 8. How to run

The batch chain, which is what still ships:

```bash
python bin/continual_chain_experiment.py --models original,ewc_original --seeds 3 --epochs 300
```

To build a retention readout into any other benchmark, supply the two callbacks
and let the shared metric own the matrix:

```python
from memval.metrics.retention import retention_matrix, retention_summary

# ONE model; weights persist across tasks -- that is what makes it continual.
def train_task(j):
    model.fit_sequence(task_emb[j], epochs=epochs)   # or stream_sequence(...) for online
    consolidate_if_supported(model, task_emb[j])

def score_task(i, j):
    model.reset_context()                            # transient state only, NOT weights
    return mean_recall_rate(measure_recall_associative(model, tasks[i], enc, dec))

R = retention_matrix(len(tasks), train_task, score_task)
summary = retention_summary(R)   # ACC / BWT / avg forgetting / per-task rows
```

Swapping `fit_sequence` for `stream_sequence` inside `train_task` is all the
ingestion axis needs; it is not wired anywhere at present.

EWC variants: use `EWCOriginalEqPropSequenceNetwork` with `ewc_lambda=1000.0`
(add `ewc_per_task=True` / `ewc_normalize=True` for the anchor-strategy variants).
