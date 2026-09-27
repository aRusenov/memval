# Interval encoding — design note

**Status:** design only. Nothing built. Online-suite section (`online_symbolic`), because it
is unrunnable under the memoryless probe by construction (see §4).

**Capacity:** Serial order, binding readout — extends it from *ordinal* time (which came
first) to *metric* time (how long between). Failure condition no other section detects:
association intact, interval lost.

**Literature anchors.**
- Kishimoto, Nakazawa, Tonegawa, Kirino & Kano (2006), J Neurosci 26(5):1562–1570. Trace
  eyeblink conditioning: CA3-NR1 KO mice still acquire the CS→US association but lose the
  adaptive *timing* of the response (wild-type CRs are withheld until just before US
  arrival). The exact dissociation this section scores. Note it is **trace**, not delay,
  conditioning — hippocampus is required only when a stimulus-free gap separates the events.
- MacDonald, Lepage, Eden & Eichenbaum (2011), Neuron 71:737–749. Time cells tile the empty
  gap between paired events, and **retime** when the trained gap is lengthened — the
  interval is represented, not endured. Grounds the retiming probe (§2.3).
- Raybuck & Lattal (2014), Neurobiol Learn Mem — review bridging trace conditioning and
  time cells, single citation for the pair if needed.

## 1. Relation to `isi_tolerance`

`isi_tolerance` sweeps `isi_steps` and asks whether recall *survives* the gap: ISI as
nuisance. This section reuses the same `stream_sequence` mechanic and asks whether the gap
length is *stored*: ISI as signal. State the contrast explicitly in the paper — it turns an
existing robustness sweep into one half of a capacity dissociation at zero extra build cost.

## 2. Probes

### 2.1 Interval reproduction
Train `A -[d blank ticks]- B` (one pair per run; sweep d, e.g. 1/3/5/10). Probe: present A,
then feed blank ticks, recording the decoder's confidence for B at every tick. Score the
**peak time** of the B-confidence time course against d.

Score peak location, not thresholded emission: a respond-vs-withhold rule would need a
response threshold or a null symbol, which is harness opinion (same reasoning as the
explicit-context decision, docs/paper/results_intro_capacities.md). Peak time is the
ideal-observer readout and mirrors how CR timing is scored in the animal literature.

Metrics: `interval_peak_time_error(d)` (signed, in ticks); `interval_encoded` (is peak time
correlated with d across the sweep, against the degenerate alternative of peaking at the
first post-A tick regardless of d).

### 2.2 Interval discrimination (the aliased version)
Train `A -[d1]- B` and `A -[d2]- C`. The cue is identical; only elapsed time disambiguates.
This is the temporal analogue of the aliased symbolic chain (`pear` twice with different
successors): a memoryless map must fail it by construction, because every blank tick
presents the same input and demands a different output trajectory.

Metrics: B/C accuracy at each of d1, d2; confusion as a function of |d1 − d2| (temporal
discriminability curve).

### 2.3 Retiming
After 2.1 converges at d, retrain the same pair at d′ ≠ d and re-probe. Does the peak move
to d′, and in how many presentations? Distinguishes a stored, updatable interval from a
frozen trace. Direct analogue of MacDonald's retiming manipulation.

### 2.4 Scalar-timing check (secondary, free)
Across the d sweep of 2.1, regress |peak error| on d. Biological interval timing is scalar
(Weber-like: error grows ~linearly with the interval). Not a pass/fail criterion — a
signature worth reporting because it separates clock-like mechanisms from decay-like ones.

## 3. What is fed during the gap — open decision

Options: (a) Gaussian noise ticks, as `isi_tolerance` already does; (b) zero vectors
(closest to "stimulus-free"); (c) a dedicated gap token. (c) is the weakest test — counting
repeated identical tokens still needs state, but a distinct token gives the model an
anchor the biology doesn't get. Recommend (a) for continuity with the existing sweep, with
(b) as a robustness check. Whichever is chosen, the *probe* gap input must match the
*training* gap input, or timing errors confound with input-statistics mismatch.

## 4. The probe-protocol prerequisite (the real cost)

`predict_next` is pure by contract, so probing `predict_next(blank)` repeatedly returns the
same output at every gap tick — a memoryless probe cannot express "the same input, later."
The probe must drive the model's *online state*: tick the model through the gap and read
predictions from the evolving state. Two interface options:

1. **Plasticity-on probing** — reuse `fit_event` during the probe, accepting that probing
   itself trains. Biologically honest and consistent with the true-online philosophy
   (docs/proposals/true_online_learning.md), but scores then depend on probe count and order; report
   presentation-matched controls.
2. **A `tick(x)` contract** — state update without weight update. Clean measurement, but a
   new interface that not all arms can implement honestly (for EP-family arms, settling and
   learning are not separable phases; the split is principled for tPC-style arms only).

This is the same state-carrying-protocol prerequisite as
docs/proposals/nonlinearity_benchmark_design.md §4 (history-dependent family) — build it once, both
sections use it. Decide per-arm honesty before building; do not let the interface silently
exclude arms (cf. the `n_context` precedent).

**Run this section only on online/stateful arms.** A stateless arm scores at chance for a
protocol reason, not a model reason; reporting it in the same table as stateful arms
misattributes a harness limitation to the model (nonlinearity doc §4 caveat, verbatim).

## 5. Confounds to check at build time

- **Noise ticks as items.** With Hebbian-style rules, gap ticks may themselves be stored as
  quasi-items, turning `A -[3]- B` into `A→n→n→n→B` — which *solves* reproduction by
  chaining without any interval representation. Diagnose by probing with a gap input drawn
  from a different distribution than training (chaining breaks, genuine timing survives).
  This is not hypothetical: it is exactly how a linear associator would pass 2.1, and it
  fails 2.2 (same noise statistics for both delays), which is why 2.2 is the load-bearing
  probe and 2.1 alone must never be quoted as evidence of interval encoding.
- **Boundary leakage.** `on_event_boundary` fires between presentations; ensure it does not
  reset the state that carries the gap count mid-pair.
- **Decoder floor.** Peak-time is undefined if B never decodes above the noise floor;
  report reproduction only where final-tick MRR clears the `isi_tolerance` floor for that d.

## 6. Placement in the capacity map

One row under Serial order (binding), marked designed-not-built until §4 is resolved:

| Serial order (binding, metric time) | `interval_encoding` (online, designed) | gap length d; |d1−d2| | peak-time error; discrimination accuracy; retiming shift; Weber slope |
