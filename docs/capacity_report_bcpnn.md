# Capacity run: spiking BCPNN (`bcpnn_spiking`), section by section

Arm: `CodecBCPNNNetwork` (Tully et al. 2016 spike-based BCPNN behind the
population spike codec), port note `docs/bcpnn_spiking_port.md`. Started
2026-09-05. Run one section at a time, each reviewed before the next, so a
defect surfaces after minutes rather than after a whole-suite run.

## Conditions, stated once

| condition | value | why |
|---|---|---|
| codec | `n_per_feature=5`, `window_steps=50`, `r_max=200 Hz` | matched to the Bush and Vieth arms |
| stimulus synapse | `w_stim = 50 nS` | unit-norm items drive cells at 20-70 Hz, not the paper's 200 Hz; 50 nS brings training to ~f_max (port note) |
| recurrent-gain normalisation | `pattern_cells = 250` | the symbolic substrate's measured active count |
| circuit cost | 3 basket cells per hypercolumn, 0.5 ms substeps | probe 0.47 s -> 0.08 s at unchanged scores; paper gate re-checked (CRP lag-1 0.89, D_L 1.5) |
| readout | cue = 2 x 50 ms windows, readout the 100 ms after the cue | `bin/bcpnn_gate.py --codec` |
| exposure | criterion staircase, ceiling 32 epochs | the P traces are running averages with `tau_p = 5 s` against 0.6 s per pass; stationary well before 32 |
| trials | 30 for sections 1, C1, C2; **10 from C3 on** (user's call, 2026-09-05, for wall-clock: C2 took 47 min at 30) | probe noise is Poisson over a 100 ms window and already averaged; stated in every caption |
| ingestion | section runs, one per output dir, merged by `bin/merge_section_runs.py` | subset runs differ from a full-suite run by <= 0.03 MRR through the global-RNG probe noise; same for every section here |

Expected before any section runs: the arm reproduces its paper on the
paper's geometry and floors behind the codec on 6-item lists (next-item
accuracy 0.2-0.6, margin ~0) for the structural reasons in the port note.
Sections below are read against that expectation.

## Sections

### 1. presentation_duration -- floor, attributed (6.5 min)

Run 2026-09-05 12:36-12:43, `results/zoo_capacity_run/_bcpnn_sections/presentation_duration`.

| metric | value |
|---|---|
| convergence curve (MRR at epochs 1, 2, 4, 8, 16, 32) | 0.167, 0.189, 0.211, 0.167, 0.167, 0.167 |
| converged / convergence_epochs | False / None (censored at 32) |
| convergence_span | 0 |

Chance for this scorer (ranks over the whole fruit vocabulary) is ~0.2, so the
curve is flat at chance from the first epoch to the last. Exposure is not the
lever: the P traces are stationary by epoch 8 and the curve does not move.

**Diagnosis, on the section's own 7-word material (32 epochs):**

* Not a readout defect. The cue never ranks first in 30 probes, the target's
  rank histogram is uniform, and a readout lag of 25-150 ms changes nothing
  (MRR 0.44 -> 0.38 over 7 candidates, i.e. chance throughout).
* **The memory is empty.** Learned weights: in-item AMPA **0.36 nS** (the
  paper's is ~4 nS), item -> next item AMPA -0.24 nS, forward NMDA
  **0.002 nS**, bias -72 pA. Each item activates ~240 of 1000 cells (24%) at
  magnitude-graded rates and any two items share 25% of their cells
  (Jaccard). BCPNN weights are log co-activation *above chance*; for this
  code that is near zero, exactly the dense-substrate mechanism in the port
  note.
* No siting rescues it: stated gains (x3.3), NMDA x4, x12, and background
  50 Hz all leave the cue's own cells at 17-19% of readout spikes against
  a 24% base rate, i.e. the cued item is suppressed rather than sustained;
  the one setting that recalls the cue itself (0.3/0.025 at 50 Hz, cos(self)
  +0.22) does so with 26 spikes in 100 ms and no successor.

Reading: the symbolic suite's substrate (`SymbolicEncoder`, D=100,
category_variance 0.2) gives this arm nothing to chain. Every remaining
symbolic section uses the same substrate and should be expected to floor
the same way; the sections that can discriminate this arm are the spatial
ones (a different code) and, structurally, anything on a sparse orthogonal
code -- which the suite does not have.

### 1b. Standalone: the paper's code on our material (`standalone_bcpnn_columnar.py`)

The floor in sec 1 is the codec's, not the embeddings'. The paper's S1
Appendix represents a continuous variable as "interval coded": one
hypercolumn per variable, one minicolumn per interval, one active at a flat
rate. Our codec is the 2-interval case with magnitude-graded rates. The
standalone builds the general case -- optionally project the embedding to
`n_hc` dimensions, bin each into `n_mc` equal-width intervals over the item
set, drive that minicolumn at 200 Hz -- and scores cue -> next item with a
nearest-pattern decoder in spike space (share of readout spikes on each
item's minicolumns). 7 items, 20 epochs, 5 trials, readout = the 100 ms after
a 100 ms cue. Chance: acc 0.17, MRR ~0.37.

| code | material | hc x mc x cells | N | overlap | next acc | MRR | in-item AMPA | fwd NMDA |
|---|---|---|---|---|---|---|---|---|
| orthogonal (paper's own) | item identity | 9 x 8 x 30 | 2160 | 0.00 | **1.00** | 1.00 | 4.5 nS | 0.024 nS |
| orthogonal | item identity | 20 x 8 x 10 | 1600 | 0.00 | **1.00** | 1.00 | 5.8 | 0.032 |
| interval | symbolic fruit, 4 bins | 20 x 4 x 10 | 800 | 0.22 | 0.77 | 0.84 | 3.0 | 0.010 |
| interval | symbolic fruit, 8 bins | 20 x 8 x 10 | 1600 | 0.10 | **1.00** | 1.00 | 4.6 | 0.021 |
| interval | symbolic fruit, 8 bins | 50 x 8 x 10 | 4000 | 0.10 | **1.00** | 1.00 | 1.7 | 0.008 |
| interval | symbolic fruit, 8 bins, no projection | 100 x 8 x 5 | 4000 | 0.11 | **1.00** | 1.00 | 1.5 | 0.008 |
| interval | hierarchical 2x4, 8 bins | 20 x 8 x 10 | 1600 | 0.10 | 0.87 | 0.93 | 4.7 | 0.016 |
| interval | hierarchical 2x4, 8 bins | 50 x 8 x 10 | 4000 | 0.12 | **1.00** | 1.00 | 1.6 | 0.007 |

Against the shared codec's 0.17 on the same seven words. The learned weights
are the paper's (4-6 nS in-item AMPA, 0.02-0.03 nS forward NMDA against
0.36 / 0.002 on the shared codec), the cue never out-ranks its successor in
the readout window, and nothing about the embeddings was fitted: the same
`SymbolicEncoder` vectors, binned. Cost: 4-23 s to train, 0.04-0.28 s per
probe.

**Robustness (`--robustness`), hc 20 x mc 8 x 10 cells unless stated:**

| check | setting | next acc | MRR |
|---|---|---|---|
| cue noise (pipeline default 0.05) | symbolic, seeds 0-3 | 0.40-0.67 | 0.61-0.78 |
| cue noise 0.10 / 0.20 | symbolic | 0.47 / 0.20 | 0.65 / 0.44 |
| load, random items | 12 items | 0.42 | 0.60 |
| load, random items | 20 items, hc 20 / hc 50 | 0.25 / 0.55 | 0.42 / 0.68 |
| hierarchical, all 8 leaves, noise 0.05 | | 0.54 | 0.70 |

The noise sensitivity is the code's resolution: seven items span ~0.4 in a
projected dimension, so 8 equal-width bins are ~0.05 wide, one SD of the
pipeline's cue noise, and most hypercolumns flip to a neighbour. What is
left is the network completing a half-corrupted cue, which it does about
half the time -- the paper's pattern completion, under a harder corruption
than the paper uses. Load is the paper's own capacity axis (10 patterns in
10 minicolumns); with 8 bins holding 2.5 items each at 20 items the
transitions blur, and more hypercolumns (50) recover part of it.

Resolution sweep under noise 0.05 (symbolic, 3 seeds): more hypercolumns
buy tolerance, finer bins do not -- hc x mc = 20x4 0.42, 20x8 0.57, 20x16
0.36, 50x4 0.68, 50x8 0.67, **100x4 0.73** (MRR 0.85). Redundant coarse
hypercolumns out-vote the flips; fine bins just flip more often.

### 1c. The columnar codec as a fixed condition (`bcpnn_spiking_columnar`)

Built 2026-09-05 as `memval/encoders/columnar_codec.py` and
`CodecBCPNNNetwork(codec="columnar")`, registered as `bcpnn_spiking_columnar`.
The setting is derived from the paper and **fixed before any section runs**:

| | value | paper-side reason |
|---|---|---|
| hypercolumns | one per embedding dimension (D = 100) | S1 Appendix: one hypercolumn per variable; no projection |
| minicolumns per hypercolumn | 10 | the paper's; chance overlap 1/10 |
| cells per minicolumn | 3 (300 cells per item) | the paper's pattern is 270 |
| rate, window | 200 Hz, 100 ms | the paper's stimulus |
| bins | equal width over +-2.5 SD of an isotropic unit-norm coordinate, clipped | a fixed frame; nothing taken from the item set |
| stimulus synapse | 15 nS | the paper's f_max criterion at the paper's 200 Hz |

The earlier standalone rows used bins fitted to each item set's range and a
random projection; those were exploration and are superseded by this
condition. Verified through the codec tier and the pipeline's own vector
interface (`standalone_bcpnn_columnar.py --verify`, 20 epochs, 5 trials):

| material | items | cue noise | codec round-trip id | overlap | next acc | MRR | in-item AMPA / fwd NMDA |
|---|---|---|---|---|---|---|---|
| symbolic fruit | 7 | 0 | 1.00 | 0.15 | **1.00** | 1.00 | 2.3 / 0.011 nS |
| symbolic fruit | 7 | 0.05, seeds 0-2 | 1.00 | 0.15 | 0.63 / 0.80 / 0.63 | 0.76 / 0.85 / 0.77 | |
| random unit-norm | 12 | 0 | 1.00 | 0.14 | **1.00** | 1.00 | 2.3 / 0.012 |
| random unit-norm | 20 | 0 / 0.05 | 1.00 | 0.14 | 0.84 / 0.45 | 0.92 / 0.58 | 2.0 / 0.009 |
| hierarchical (2x4 leaves) | 8 | 0 / 0.05 | 1.00 | **0.69** | 0.00 / 0.00 | 0.21 / 0.22 | 0.7 / 0.001 |

Cost: N = 3000, 0.8 s per epoch, 0.16 s per probe.

Two things to carry into the sections. The noise result is the code's
resolution and stands as the arm's weakness under the shared protocol. The
hierarchical floor is a different mechanism: those embeddings are mostly
exact zeros, and under a fixed frame every zero falls in the same middle
bin, so any two items share 69% of their cells -- far past the paper's own
disambiguation limit of ~2 shared elements in 9. A sparse embedding has no
faithful representation in a code where every hypercolumn must hold a
value; that is a property of the paper's format, reported rather than
patched. The symbolic suite's substrate is dense, so the sections run on
this entry read against the first two rows.

## Sections on the columnar entry (`bcpnn_spiking_columnar`)

Runs go to `results/zoo_capacity_run/_bcpnn_columnar_sections/<section>`.

### C1. presentation_duration -- learns, plateaus below criterion (5 min)

Run 2026-09-05 16:06-16:11. 7 fruit words, cue noise 0.05, 30 trials, ceiling 32.

| metric | value |
|---|---|
| convergence curve (MRR at epochs 1, 2, 4, 8, 16, 32) | 0.211, 0.511, 0.778, **0.794**, 0.750, 0.717 |
| converged / convergence_epochs | False / None (criterion 0.95) |
| convergence_mrr (best) | 0.794 |
| convergence_span | 0 (defined only at criterion) |

Against the shared codec's flat 0.17-0.21 (sec 1) and chance ~0.2. Three
readings:

* **It learns, and it learns fast.** Chance at one pass, 0.78 by four,
  peak at eight. That is the P traces converging (tau_p = 5 s against 0.7 s
  per pass), the paper's S7 Fig timescale; more exposure has nothing to add
  and the curve shows it (0.75, 0.72 at 16 and 32 -- a slight decline,
  within probe noise or the negative weights sharpening as the traces
  settle; not distinguished here).
* **The plateau is the noise result, not a learning limit.** The section
  corrupts every cue with N(0, 0.05) in vector space; the standalone gave
  0.76-0.85 MRR at that noise and 1.00 without it (sec 1c). Under the
  shared protocol the arm's presentation-duration verdict is therefore
  "reaches ~0.8 MRR, never criterion", and the mechanism is the code's
  resolution -- bins about one noise SD wide.
* Exposure staircases elsewhere in the suite will censor at their ceiling
  for the same reason; read `criterion_reached: False` on this arm as
  "plateaued", with the plateau value the number that matters.

### C2. sequence_length -- the load axis: holds 3, degrades from 5, no autonomous unrolling (47 min)

Run 2026-09-05 16:30-17:17. Lists of 3, 5, 7, 9, 11 fruit words, cue noise
0.05, 30 trials, staircase ceiling 32. "MRR" in this section is the mean
per-position recall success (`mean_recall_rate`), and a span is the longest
prefix whose per-position success is >= 0.75.

| length | cued MRR | epochs to criterion | criterion reached | cued span | autoregressive span (raw / l2 / quantized) |
|---|---|---|---|---|---|
| 3 | **0.90** | 18 | yes | 2 (= all) | 2 / 2 / 2 |
| 5 | 0.67 | 32 (censored) | no | 0 | 0 / 0 / 0 |
| 7 | 0.76 | 32 (censored) | no | 0 | 0 / 0 / 0 |
| 9 | 0.54 | 32 (censored) | no | 0 | 0 / 0 / 0 |
| 11 | 0.57 | 32 (censored) | no | 0 | 0 / 0 / 0 |

Summary metrics: `max_memory_span` 2, `unrolling_gap` 0.0,
`seqlen_criterion_reached` False, `exposure_baseline_epochs` 5 (reached:
the clean-cue 7-word reference converges in 5 passes),
`exposure_within_band` True.

Readings:

* **Load degrades it from 5 items on, well before the paper's 10.** Cued
  recall is 0.90 at three items and 0.54-0.76 from five up, with the
  non-monotone 5 vs 7 within the trial noise of a 30-trial probe. This is
  the paper's own capacity axis measured under a harder condition than the
  paper's: the items are interval-coded semantic vectors sharing ~15% of
  their minicolumns (chance is 10%), not orthogonal patterns, and every cue
  is corrupted. Together with C1 the picture is consistent: clean cues on
  7 words reach 1.00; the same list under 0.05 noise reaches ~0.8; longer
  lists under noise fall to ~0.55.
* **Spans are the noise ceiling, not a serial-position effect.** A span
  counts positions until the first one below 0.75 success. From five items
  on, position 1 already sits near 0.6-0.7, so the span is 0 at every
  length despite cued MRR of 0.5-0.76. At three items every position is
  above 0.75 and the span is the whole list.
* **No autonomous unrolling beyond three items.** `unrolling_gap` is 0.0
  because the autoregressive spans equal the cued spans (2 and 0), but
  that identity is degenerate: both are 0 wherever cued recall is below
  the span threshold. What the rollout modes do show is that feeding the
  decoded vector back (raw, L2-normalised or quantised) does not recover a
  chain the cued probe already fails on. Read the arm's serial-order
  capability as "three items, cued or unrolled".
* Exposure is not the lever, again: every censored length was flat by
  epoch 8 (C1), so raising the ceiling would not change a row.

Cost note: 47 minutes for one section at 30 trials, dominated by the
11-item length (300 probes per checkpoint at 0.16 s). The remaining
symbolic sections with similar sweeps (`multiple_sequences`,
`symbolic_disambiguation`, `continual_chain`) will each take an hour or
more at this trial count.

### C3. multiple_sequences -- a palimpsest: list B erases list A (17 min, 10 trials)

Run 2026-09-05 17:45-18:02. List A = 7 fruit, list B = 7 animals (disjoint
vocabularies), cue noise 0.05, 10 trials, ceiling 32. Blocked: A to
criterion, then B to criterion, then A re-probed. Interleaved: both lists in
one stream to criterion.

| condition | A | B |
|---|---|---|
| blocked, A after its own training | 0.75 (censored at 32) | -- |
| blocked, A after B's training | **0.17** | -- |
| `delta_mrr_forgetting` | **-0.58** | |
| interleaved | 0.77 | 0.57 |
| `multiple_seq_blocking_cost` (blocked-after minus interleaved A) | **-0.60** | |
| spans | 0 throughout | |

Per-position curves: A before B `[1, .5, .9, .6, .8, .7, 1]`; A after B
`[1, 0, 1, 0, 0, 0, 0]` -- everything gone except one association
(banana -> orange at 1.0), which the disjoint vocabularies do not explain
and which is left as an unexplained single survivor.

Readings:

* **Forgetting is complete, and it is the model's own timescale, not
  interference.** The P traces are exponential running averages with
  `tau_p = 5 s`. List B's 32 passes are 22 s of training during which A is
  never presented, so A's co-activation statistics decay by `exp(-22/5)`,
  about 1%. The weights are read off those traces, so A is not overwritten
  by B in the Hopfield sense; it has simply aged out. The paper says this
  in as many words: the P traces "constitute the memory itself, which
  decays in palimpsest fashion", and `tau_p = 5000 ms` was chosen "to speed
  up simulations", with the biological range given as "seconds to months".
* **Interleaving holds both lists**, at 0.77 / 0.57 -- the same level the
  single 7-word list reaches under this noise (C1). A blocking cost of
  -0.60 against an interleaved A of 0.77 puts this arm with the EP family
  in the zoo's dissociation (ingestion order, not capacity, decides), but
  for a different reason: EP's is a full-batch learner's averaging, this
  one is a decay clock.
* **A knob the paper itself flags as arbitrary.** `tau_p` is the one
  parameter here whose stated value is a simulation convenience rather than
  a fit. Raising it would slow acquisition proportionally (the paper says
  so), which the exposure staircase would absorb, and would lengthen
  retention proportionally. Under the "model as-is" rule it stays at 5 s
  and the result stands; a `tau_p` sweep in the standalone would be a
  diagnostic of the retention/acquisition trade-off, not a re-siting, and
  is on offer if wanted.

## Status and how to resume (stopped 2026-09-05 22:30, user's call)

Done on the columnar entry: C1 `presentation_duration`, C2 `sequence_length`,
C3 `multiple_sequences`. Section 1 on the shared codec stands as the
attributed floor.

**Stopped mid-run:** C4 `continual_chain` (launched 18:21 at 10 trials,
killed 22:30 with no output; it ran 4 h against a concurrent numpy process
and was starved -- see the contention note below). Its partial output dir
`_bcpnn_columnar_sections/continual_chain` holds nothing usable; delete or
overwrite. Prediction on record before it runs: every list but the most
recent should be gone, by the palimpsest mechanism of C3.

**Not yet run:** the redesigned tau_p diagnostic
(`python standalone_bcpnn_columnar.py --tau-p`: A to criterion, B for the
section's 32 passes, A re-probed, tau_p 5 / 20 / 80 s, clean cues), and the
sections `paired_associate`, `noise_invariance`, `cue_masking`,
`semantic_similarity`, `symbolic_disambiguation`, `schema_consistency`
(+ its focused-protocol rerun), the spatial suite, `online_symbolic`, the
serial-order probe, then merge + score.

**One process at a time.** Two concurrent numpy-heavy runs oversubscribe the
BLAS threads and slow both 10-30x (the sweep's 5 s row took 32 min for ~1 min
of work). Run each section alone, in the background, with a watcher:

```bash
S=continual_chain
mkdir -p results/zoo_capacity_run/_bcpnn_columnar_sections/$S
nohup python -u bin/run_benchmark.py --model bcpnn_spiking_columnar --suite symbolic \
  --benchmarks $S --output-dir results/zoo_capacity_run/_bcpnn_columnar_sections/$S \
  --n-trials 10 --max-epochs 32 --benchmark-args "$S:max_epochs=32" \
  > results/zoo_capacity_run/_logs/bcpnn_columnar_section_$S.log 2>&1 &
```

When all symbolic sections are in:

```bash
python bin/merge_section_runs.py --class CodecBCPNNNetwork --suite symbolic \
  --sections-root results/zoo_capacity_run/_bcpnn_columnar_sections --out results/zoo_capacity_run
```

then the spatial and online suites (whole suites, no per-section split
needed at their cost), `bin/probe_serial_order.py --model bcpnn_spiking_columnar`,
`bin/score_capacities.py` and `bin/build_scorecard_page.py` as in
`bin/run_capacity_arm.sh`.

## Continual retention only (2026-09-06)

User's call: cap every staircase at 32 epochs for this arm (it is slow, and
its traces are stationary by ~8 passes) and run only the sections feeding
the **Continual retention** capacity, then score. Per `bin/score_capacities.py`
that capacity is five dimensions:

| dimension | weight | section (suite) | status |
|---|---|---|---|
| load | 0.25 | `multiple_sequences` (symbolic) | done, C3 |
| load / plasticity_under_load / relevance | 0.25 / 0.20 / 0.10 | `continual_chain` (symbolic) | chained run, cap 32 (`max_epochs` passthrough added to the section for this) |
| contingency | 0.35 | `paired_associate` (symbolic), `tmaze_reversal` (spatial) | chained run |
| schema_consistency | 0.10 | `schema_consistency` under the focused protocol (symbolic) | chained run, both protocols |

All at 10 trials, one process at a time, merged with `bin/merge_section_runs.py`,
scored with `--model bcpnn_spiking_columnar`, then the scorecard page and the
results index are rebuilt. Sections C1-C2 stay in the merged file as the
other capacities' partial coverage.

Note on the registry: `bin/run_benchmark.py` had been pruned (2026-09-06
14:57) to six arms with the Bush, Vieth and BCPNN entries commented out;
the two BCPNN entries were uncommented for this run and the others left as
found.

### C4-C7 and the scorecard (chain run 2026-09-06 12:42-20:42, 10 trials, cap 32)

Wall-clock: paired_associate 18 min, continual_chain 28 min, schema
75 min + 11 min focused, **tmaze_reversal 5.8 h** (spatial embeddings make
the columnar network large; budget for it). Merge, score, page and index in
seconds.

**Continual retention: 0.626, coverage 90%** (inclusive 0.664). Dimensions:

| dimension | weight | score | what carried it |
|---|---|---|---|
| load | 0.25 | 0.51 | C3's total forgetting (0.75 -> 0.17) pulls down; `continual_chain` holds up: 6 tasks x 5 items, criterion in **3.3 epochs**, avg accuracy 0.71, avg forgetting 0.35, retention ratio 0.70 |
| plasticity_under_load | 0.20 | **1.00** | `chain_avg_learning` 1.0, `chain_intransigence` 0.0 -- every new list is acquired at once, whatever came before |
| contingency | 0.35 | 0.58 | T-maze: acquisition 2 trials, **reversal 1 trial**, extinction never (12, censored), perseveration 0.75 at the end of extinction; paired-associate: AC learned in 1 trial, AB retention **0.0**, no cue-competition cost |
| relevance | 0.10 | 0.33 | rehearsed lists 0.58 vs unrehearsed **0.00** (drift -0.67); selectivity 0.5, rehearsed gain -0.17 |
| schema_consistency | 0.10 | unresolved | every rung reaches criterion at trial 1 (`schema_at_ceiling`), so the consistency ladder has no dynamic range; `new_item_recall` 0.0 on duplicate/within/across and 1.0 on random; interference 0.0 throughout |

Readings, section by section:

* **continual_chain (C4) is the palimpsest seen from the other side.** The
  chain acquires each 5-item list in ~3 epochs (1.5 s of kappa = 1), so
  six lists take ~10 s -- two tau_p -- and the earlier ones have decayed
  only partly (retention ratio 0.70) instead of vanishing as in C3, where
  B's censored 32 passes were 22 s. Same clock, different amount of
  subsequent learning; C3 and C4 together are the retention-vs-time curve
  the model predicts. Plasticity is perfect for the same reason: nothing
  old stands in the way of new co-activation statistics.
* **Selective retention works only as maintenance.** Rehearsed lists stay
  at 0.58 while unrehearsed ones fall to 0.00; rehearsal does not *improve*
  them (gain -0.17). Under this rule rehearsal is re-presentation keeping a
  running average alive, nothing more.
* **paired_associate (C5): complete overwrite, and a control that fails
  for a different reason.** AB is at 1.0 after phase 1; after AC training
  AB is 0.0 and AC 1.0, learned in one trial -- the shared cue now predicts
  C, exactly what a co-activation estimate must do. The disjoint-cue
  control also loses AB (0.0) and does not learn its new pairs
  (`pa_control_ac_recall_final` 0.0, `other_rate` 1.0): the control's
  phase 2 is 60 s of kappa = 1, so AB ages out regardless of cue sharing,
  and why the new disjoint pairs are not recalled is **unexplained** -- the
  one readout in this run I would look at before quoting.
* **tmaze_reversal (C6): fast reversal, no extinction.** Acquisition in 2
  trials, reversal in 1 (new contingency written over the old within one
  presentation, as in C5), but reward extinction never reaches criterion in
  12 trials and perseveration rises to 0.75: withholding reward is *absence
  of a presentation*, and an absence does not update a co-activation trace
  -- only kappa = 1 time with other material does. The same failure DTS-ESN
  shows, from an opposite mechanism (frozen reservoir vs decaying trace).
  Accuracies are low throughout (0.375 after acquisition, 0.75 after
  reversal) and the section took 5.8 h; both worth a look at the spatial
  codec geometry before this row is compared across arms.
* **schema (C7) is out of range at the ceiling**, as the scorer says: the
  base sequence reaches criterion at the first trial on every rung, so the
  consistency manipulation cannot separate anything, and the section is
  correctly left unscored (coverage 90%). The non-zero `new_item_recall`
  only on the random rung is unexplained and is recorded, not read.

Other capacities from the same merged file, partial and not the target of
this run: One-shot learning 0.310 (coverage 50%), Serial order 0.360 (20%),
Pattern completion and Sequence disambiguation unrun.

Artifacts: `results/zoo_capacity_run/CodecBCPNNNetwork/capacity_scorecard.{json,md,html}`
and `_radar.png`; `results/zoo_capacity_run/_compare/` regenerated with
"BCPNN (columnar)" as the eighth arm; `results/zoo_capacity_run/index.html`
rebuilt; `docs/capacity_report_zoo.md` arm table extended.
