# Capacity coverage audit — what the suites actually measure

**Generated:** 2026-08-31. **Decisions taken:** 2026-08-31 (§0). **Landed in code:** 2026-09-01.
**Provenance:** built from the pipeline sources, not the design docs. Section names are the literal
entries of `SPATIAL_BENCHMARKS` / `SYMBOLIC_BENCHMARKS` / `ONLINE_SYMBOLIC_BENCHMARKS`; metric names
are the keys actually written to `results["metrics"]`. Prose promises are quoted from
`paper/capacities.md`.

**How to edit:** flip a status marker, correct a question or metric name, or add a line under any
table. Anything you change here I will carry back into the pipelines / docs.

**Status legend**

| Marker | Meaning |
|---|---|
| SHIPPED | implemented, wired into a pipeline, emits metrics |
| CAVEAT | implemented but the result is not yet quotable (see note) |
| PLANNED | scope decision taken, not yet built |
| UNWIRED | code exists and is tested, but no pipeline calls it |
| DESIGNED | full design note, nothing built |
| ABSENT | no design and no code |
| BLOCKED | cannot be built under the current memoryless probe |
| REMOVED | cut from scope — see §0 |

---

## 0. Scope decisions applied

| # | Decision | Rationale | Code impact |
|---|---|---|---|
| D1 | **Remove** `spatial_sequence` (S-curve) | Duplicates `tmaze_completion` as a completion test: one route, non-self-intersecting, linearly solvable, no sweep. Its only structural role was as the open-loop control for `anchoring_few_shot`, which D3 also removes. | see ⚠️ C1 |
| D2 | **Extend** `tmaze_disambiguation` with graded odour availability — full corridor → progressively shorter → onset-only | Places the shipped section on axis A instead of at its degenerate corner. This is the `delay = shared_end − zone_end` sweep. | see ⚠️ C2 |
| D3 | **Remove** `anchoring_few_shot` | Measures drift correction — a capacity deliberately out of scope for now. Retain the design for a future drift-correction capacity, or as a sequence-disambiguation extension (anchor availability during rollout is the same "external support" axis as D2). | dead code, see C1 |
| D4 | **Remove** object arena (NOR/OLM) stub | Never implemented; emits `"TODO"` strings into `metrics.json`. | delete 2 keys |
| D5 | **Remove** `letter_noise` | Measures the vocabulary's edit-distance geometry, not the model: across 7 architectures the across-model spread is **0.000** for `insert` and `transpose`, ≤0.111 for the rest, against **0.711** for the σ sweep. No literature anchor in `docs/` or `paper/`. | dead code, see C3 |
| D6 | **Remove** interval / interference decay from the symbolic (non-online) suite | The batch suite has no time axis. The only interval it can express is count of interposed items, which is interference — already measured by `multiple_sequences` and the retention matrix. | delete 1 key |
| D7 | **Extend** online symbolic with interval duration between events | Turns `isi_tolerance` (ISI as nuisance) into half of a dissociation whose other half is ISI as signal. This is `docs/interval_encoding_design.md`. | new section |
| D8 | **Drop** `online_continual` | Not CLI-wired; overlaps `bin/continual_chain_experiment.py`. | see ⚠️ C4 |

### Consequences — all four resolved and implemented 2026-09-01

| # | Call taken | What landed |
|---|---|---|
| C1 | **Move them** | `coverage` and `divergence_step` now come from `tmaze_completion` (`tmaze_pc_coverage`, `tmaze_pc_divergence_step`, plus `tmaze_pc_recall_len` / `tmaze_pc_eps` so both are readable against their tolerance). `_recall_metrics` survives; `_build_anchor_demo`, `_catmull_rom`, `_anchored_rollout_score` deleted. |
| C2 | **Proceed** | Graded odour availability wired as section 2b. `BifurcatingRouteGenerator` gained `balance_modalities` / `odour_scale` (place:odour was 2:1 *against* the cue) and `odour_on_suffix` (the concurrent reference). Metrics §5.1 and §5.2 implemented. |
| C4 | **Extract the metric** | `memval/metrics/retention.py` — `retention_matrix`, `retention_summary`, `plot_retention_matrix` — modality-agnostic, driven by two caller-supplied callbacks. `bin/continual_chain_experiment.py` rewired onto it. `online_continual_pipeline.py` deleted. |
| C3 | **Remove** | `letter_noise` section and `measure_noise_letter_invariance` deleted. `memval/utils/text_noise.py` kept: it is a standalone tested utility (`tests/test_gpt2_wrapper.py`) and is the natural substrate if the graded cue-interpolation replacement is built. |

**First result from C2, worth reporting.** On `hopfield`, the concurrent
reference holds at divergence accuracy 1.00; withdrawing the odour holds at 1.00
for `delay = 0` and collapses to chance (0.50) from `delay = 2` onward. The
normalised margin decays monotonically — 0.730 (concurrent) → 0.419 (delay 0) →
0.042 → 0.008 → 0.001 → 0.000 — i.e. it keeps resolving across delays where
accuracy is already flat at the floor, which is precisely the discrimination
§5.2 was added to recover. The shared-stretch control rises with delay
(0.067 → 0.148), so part of the collapse is ordinary rollout drift rather than
cue loss alone; report the two together.

**C4 closed 2026-09-02 — option (a), extended.** `continual_chain` is now a
section of the symbolic suite (`memval/benchmarks/continual_chain.py`), so a
`--suite symbolic` run computes a retention matrix and `paper/capacities.md` ¶3
is supported. The section went in wider than option (a) proposed, because
reading the matrix exposed two questions the capacity had never asked:

- **The diagonal.** Every stability read-out in this capacity — ACC, BWT,
  `avg_forgetting`, `delta_mrr_forgetting` — is maximised by an arm that stops
  learning after task 0. Nothing in the suite could detect that. `R[i, i]` is
  the tell, it was already being computed, and it was being thrown away:
  `retention_summary` returned `per_task_learned` and
  `bin/continual_chain_experiment.py` dropped it. Now scored as the
  `plasticity_under_load` dimension (`chain_avg_learning`,
  `chain_intransigence`), with `chain_stability_plasticity_index` — the
  harmonic mean of the two coordinates — as the joint read-out.
- **The load curve.** `forgetting_by_gap` gives the retained fraction at every
  interposition gap 1 … T-1. `multiple_sequences` is the g = 1 point of it.

The **ingestion contrast** remains the one thing the deleted pipeline was the
sole home for; `docs/online_continual_benchmark.md` §8 still carries that recipe.
Note the chain is a *condition* of the symbolic suite, not a suite, which is what
the settled narrative asks for.

<details>
<summary>Original consequence write-up (superseded)</summary>

### ⚠️ Consequences that need a call before code changes

**C1 — D1+D3 delete the suite's only non-saturating spatial readouts.** `coverage` and
`divergence_step` are computed by `_recall_metrics` and used only by `spatial_sequence` and
`anchoring_few_shot`. `tmaze_completion` reports `tmaze_pc_mse` alone, and MSE *saturates* in the
failure regime by construction (centre-of-mass decoding pulls a lost readout to the arena middle, so
a collapsed recall plateaus rather than diverging). Removing both sections leaves spatial completion
scored only by the metric that cannot distinguish "somewhat wrong" from "totally lost".

*Recommendation:* move `coverage` and `divergence_step` onto `tmaze_completion` **before** deleting
§2/§5. `divergence_step` is additionally the continuous-space analogue of memory span and the
error-propagation readout `paper/capacities.md` ¶17 promises — currently computed and discarded.

Dead after removal: `_build_anchor_demo`, `_catmull_rom`, `_anchored_rollout_score`, and
`_recall_metrics` unless C1 is adopted.

**C2 — D2 is the axis the design note calls half-blocked.** The generator already expresses
`delay > 0` (`OverlapConfig.zone_fraction` / `zone_offset`), so the sweep is plumbing. But under the
memoryless `predict_next` probe, every arm is predicted to fall to chance the moment the odour is
withdrawn, because none carries state across the cue-free stretch. The onset-only end of the
gradient will therefore produce a floor for a **protocol** reason, not a model reason.

That is still worth shipping — it is the behavioural face of the no-persistent-state limitation, and
a graded curve from full-corridor to onset-only localises exactly where each arm breaks. It must be
reported as a protocol finding and never as a model ranking. Unblocking it as a *model* result needs
the state-carrying ingestion path (`docs/nonlinearity_benchmark_design.md` §4).

**C3 — D5 leaves `noise_invariance` as the sole completion probe**, which is fine, but note ¶17's
structural/corruption dichotomy then has only its corruption half implemented (see §6, Pattern
completion). Dead after removal: `measure_noise_letter_invariance`, `memval/utils/text_noise.py`,
and the shared-model setup branch `if "noise_invariance" in selected or "letter_noise" in selected`.

**C4 — D8 removes the retention matrix from the suite entirely.** After it, nothing in any pipeline
computes `R[j,i]`, ACC, BWT or forgetting; only `bin/continual_chain_experiment.py` does, and that is
a standalone script, not a suite section. `paper/capacities.md` ¶3 promises "Across most benchmark
suites, we report a retention matrix" and ¶7 promises pairing exposure against it. Both become
unsupported rather than merely overstated, and **continual retention is left with `multiple_sequences`
(a T=2 special case) and `tmaze_reversal` as its only instruments.**

*Three ways to take this, pick one:* (a) drop the pipeline but promote
`bin/continual_chain_experiment.py` into the symbolic suite as the retention section — keeps the
instrument, loses the streaming/batch ingestion contrast; (b) drop the pipeline and rewrite ¶3 to
claim only A→B interference and reversal; (c) keep the pipeline and wire it into `--suite symbolic`
as an ingestion *condition*, which is what the settled narrative already says it should be.

</details>

---

## 1. Suite inventory

| Suite | CLI value | Pipeline | Sections after D1–D8 |
|---|---|---|---|
| Spatial | `--suite spatial` | `spatial_pipeline.py` | 3 |
| Symbolic | `--suite symbolic` | `symbolic_pipeline.py` | 6 |
| Online symbolic | `--suite online_symbolic` | `online_symbolic_pipeline.py` | 2 shipped + 1 planned |
| ~~Online continual~~ | — | *(deleted)* | REMOVED (D8); metric extracted to `memval/metrics/retention.py` |

The code still registers streaming as its own `--suite`, which contradicts the settled narrative
(two suites by modality; ingestion regime is a *condition inside* a suite, never a suite of its own).
D8 makes this worse, not better: dropping the only streaming-continual pipeline removes the ingestion
contrast rather than relocating it.

---

## 2. Spatial suite

| # | Section | Question it poses | Swept / conditions | Metrics emitted | Status |
|---|---|---|---|---|---|
| 1 | `tmaze_completion` | Can the model recover a stereotyped route from a cue point? | none (fixed 30% prompt, place-cell code) | `tmaze_pc_mse`, `tmaze_pc_coverage`, `tmaze_pc_divergence_step`, `tmaze_pc_recall_len`, `tmaze_pc_eps` | SHIPPED (C1) |
| 2 | `tmaze_disambiguation` | With an identical stem, does an odour cue steer the model into the right arm — **and for how much of the corridor must it remain available?** | condition ∈ {`full`, `mec_only`}; **availability: concurrent, then withdrawn at delay 0/2/4/6/7** | `tmaze_disamb_<cond>_branch_acc`, `tmaze_disamb_<cond>_confusion`; `tmaze_disamb_graded_<tag>_{branch_accuracy,divergence_accuracy,divergence_margin,branch_margin,shared_stretch_error}` | SHIPPED (D2) |
| 3 | `tmaze_reversal` | How fast is an invalid place→reward association overwritten, and does explicit extinction help? | protocol ∈ {`direct`, `extinction`}; stages acquisition → (extinction) → reversal → (rereversal) | per stage: `reversal_<cond>_<stage>_trials_to_criterion`, `_criterion_reached`, `_final_{arm_accuracy,perseveration,reward_pred,goal_identity_margin,reward_pre_goal,anticipation_lead}`, `_max_anticipation_lead`; plus `extinction_reward_drop`, `savings`, `savings_vs_reversal`, `recovery_after_interference_{arm_accuracy,perseveration,delta}`, `reversal_reward_gain`, `reversal_epochs_per_trial` | SHIPPED |
| — | ~~`spatial_sequence`~~ | — | — | — | REMOVED (D1) |
| — | ~~`anchoring_few_shot`~~ | — | — | — | REMOVED (D3) |
| — | ~~Object arena (NOR/OLM)~~ | — | — | — | REMOVED (D4) |

- The legacy block (`tmaze_disamb_{full,mec_only}_*`) still ships the `delay = 0` corner
  (`odour_on_arms=True`, odour on every step) and tests concurrent cue-binding rather than
  disambiguation over a corridor. It is retained deliberately as the sweep's reference corner.
- D2 landed as section 2b; the §5.1–§5.2 metric fixes landed with it. §5.3 (confusion matrix,
  needs N > 2) and §5.5 (linear-oracle gap) are still open.
- **C2's protocol caveat is lifted for one arm only, as of 2026-09-03.** `temporal_pc`
  (`MultilayerTemporalPCNetwork`) is now in `MODEL_REGISTRY`; it is the only registered arm that
  declares `StatePrimeable`, so it is the only one for which the benchmark's `observe_sequence`
  priming path executes. Its rows carry `state_primed=True` and are model results. The other nine
  arms remain `state_primed=False` and their withdrawn rows remain protocol floors. **Read
  `state_primed` per row; a withdrawn row quoted without it is uninterpretable.**
- The primed result: tPC is at divergence accuracy 0.500 on every withdrawn rung, margin within
  ±0.0013 of zero. The floor survives priming. Registering an arm that can carry state did not
  change the behavioural outcome, which makes this a finding rather than a fix.
- **Axis A is nearly binary as sampled.** `withdrawn, delay=0` is already at chance even though the
  odour spans the whole shared corridor there; it is withheld only on the arms. So what matters is
  presence of the discriminator *at the decision step*, not the length of its absence. The rungs
  between delay 2 and 7 add little. Resample around the delay 0/1 boundary.
- **`divergence_margin` is compressed for any arm with negative outputs.** The probe applies
  `0.5p + 0.5` whenever `pred.min() < 0`, which adds a uniform pedestal across all ~400 place cells
  and pulls the centre-of-mass decode toward the grid centre. Measured on tPC at delay 2: the margin
  reads −0.00014 under the affine remap and −0.05062 under a `clip(p, 0, None)` readout, a factor of
  ~350. Accuracy is 0.500 either way, so the conclusion is unaffected, but the margin is not
  comparable across arms with differing output ranges. This blocks cross-arm margin comparison,
  which was §5.2's whole purpose.

---

## 3. Symbolic suite

| # | Section | Question it poses | Swept | Metrics emitted | Status |
|---|---|---|---|---|---|
| 1 | `presentation_duration` | How much exposure to reach a recall criterion? | epoch checkpoints (powers of 2) to `--max-epochs` | `convergence_mrr`, `convergence_span`, `convergence_epochs`, `converged` | SHIPPED |
| 2 | `sequence_length` | Where does capacity break with list length, and how much of the break is readout drift? | L ∈ {3,5,7,9,11} × feedback ∈ {raw, l2, quantized} | `max_memory_span`, `max_memory_span_l2`, `max_memory_span_quantized` | CAVEAT |
| 3 | `multiple_sequences` | Does one interposed list destroy the first? | A→B, one interposition | `multiple_seq_mrr_before`, `multiple_seq_mrr_after`, `delta_mrr_forgetting` | SHIPPED |
| 3b | `continual_chain` | Does earlier material survive later material — **and can new material still get in once the substrate is full?** Plus: is what survives the part still in use? | 6-task chain (one category each), scored after every stage; rehearsal phase re-presents the oldest half | `chain_{avg_accuracy,avg_forgetting,backward_transfer,avg_learning,intransigence,learning_slope,retention_ratio,stability_plasticity_index,acquired,n_tasks,chance_level}`; `select_{selectivity,rehearsed_gain,unrehearsed_drift,rehearsed_recall,unrehearsed_recall,pressure,under_pressure,rehearsal_effective,n_rehearsed,rehearsal_epochs}`; series `forgetting_by_gap` | SHIPPED (C4) |
| 3c | `paired_associate` | When the **same cue** is re-paired with a new target, does the new association get in and does the old one stop being emitted? | condition ∈ {`abac` (shared cue), `control` (disjoint cue, equal budget)}; per-trial over 6 phase-2 passes | per condition `pa_<cond>_{ab_recall_baseline,ab_recall_final,ab_retention,ac_recall_final,ac_trials_to_criterion,ac_criterion_reached,other_rate_final,ab_acquired,phase2_trials,ab_trials,epochs}`; plus `pa_cue_competition_cost` | SHIPPED |
| 3d | `extinction_timeline` (standalone, `bin/extinction_timeline.py`) | When a route is still presented but stops paying, does its trace survive — and is it still the route the model would take? | cheese-odour T-maze, shared stem; train A → train B → extinguish one (reward → matched-norm noise, odour fading), 5 presentations per phase, run twice so each arm is extinguished in turn; 12 seeds | per arm and phase: `trace_fidelity` / `trace_error` (direction imposed at the fork, free rollout, error scaled by the arm separation), `choice_margin` (entrance-cued rollout, scored against the two possible successors), `reward_entry` (leak-free outcome channel, arms that predict reward as an observation only) | SHIPPED (standalone, not scored) |
| 4 | `noise_invariance` | At what cue corruption does completion fail? | Gaussian σ ∈ [0,1], 11 steps | `noise_tolerance_threshold` | SHIPPED |
| 4b | `cue_masking` | At what cue **incompleteness** does completion fail — and does it matter *which* features are missing? | masked fraction f over every realisable feature count × mode ∈ {`random`, `shared`, `identity`} | `mask_<mode>_{recall_in_bound,margin_in_bound,auc,tolerance,rollout_auc,completion_advantage,identifiable_to}`; `mask_block_asymmetry`, `mask_completion_gap`, `mask_identifiability_auc`; guards `mask_{acquired,at_ceiling,at_floor,identifiable_region,margin_ranges,resolved}` | SHIPPED (2026-09-02) |
| 5 | `semantic_similarity` | Does within-category overlap cost recall? | high- vs low-similarity list; `category_variance` ∈ {0.05…1.0} | `mrr_high_similarity`, `mrr_low_similarity`, `similarity_effect_mrr_drop` | SHIPPED |
| 6 | `schema_consistency` | Is a new item acquired faster when consistent with an acquired schema, and is interference graded by relatedness? | rungs `duplicate / within / across / random` | `schema_<rung>_{trials_to_criterion,criterion_reached,final_mrr,base_mrr,criterion,projection_ratio}`, `schema_<rung>_interference_{same,sibling,far}`, `schema_consistency_speed_corr`; guards `schema_probe_{item,category}_accuracy`, `schema_acquired`, `schema_at_{ceiling,floor}`, `schema_resolved` | CAVEAT |
| — | ~~`letter_noise`~~ | — | — | — | REMOVED (D5) |
| — | ~~Interval / interference decay~~ | — | — | — | REMOVED (D6) — intent moves to online suite §3 |

- **§2 CAVEAT (new):** the grid has no dynamic range. Across 9 models, cued MRR is 1.0 at every L for
  four arms; `quantized` span is exactly L−1 (ceiling) for four arms; `raw` span is floored at 1–2 for
  every EP arm; GPT-2 is 0 throughout. Cause: `embedding_dim=100` and Hopfield-class capacity
  α_c ≈ 0.138N ≈ 14, so the grid tops out entirely below the knee. Fix is a separate decision —
  sweep α = P/N log-spaced rather than raw L, or staircase to a per-model critical L. Blocking bug
  either way: the fruit pool is 12 words and L>12 silently appends animals, confounding load with
  category structure (`data/vocab.json` is 55 words total).
- **§6 CAVEAT:** the acquisition ladder is non-monotone at the `random` rung (EP: 1.0 / 1.7 / 7.3 /
  1.3). Read `schema_acquired` and `schema_resolved` before quoting any rung comparison. The
  consistency knob itself is sound — enforced structurally in the item×feature matrix by
  `HierarchicalEncoder.augment()`, density-matched, and verified model-free by `projection_ratio`
  (1.00 / 0.83 / 0.61–0.74 / 0.32–0.45).

---

## 4. Online symbolic suite

| # | Section | Question it poses | Swept | Metrics emitted | Status |
|---|---|---|---|---|---|
| 1 | `online_convergence` | How many streamed passes before recall holds — is one-shot possible? | presentations ∈ {1,2,5,10,20} | `online_mrr_{1shot,5pass,20pass}`, `online_span_{1shot,5pass,20pass}` | SHIPPED |
| 2 | `isi_tolerance` | Does recall *survive* noise events interposed between items? — **ISI as nuisance** | `isi_steps` ∈ {0,1,3,5,10} at 5 passes | `isi_tolerance_mrr_sweep`, `isi_tolerance_span_sweep` | SHIPPED |
| 3 | `interval_encoding` | Is the *duration* separating two events stored, not merely their order? — **ISI as signal** | gap length d ∈ {1,3,5,10}; discrimination \|d1−d2\| | `interval_peak_time_error(d)`, `interval_encoded`, discrimination accuracy, retiming shift, Weber slope | PLANNED (D7) |

- §3 reuses `stream_sequence` unchanged, so §2 and §3 form one dissociation at near-zero extra build
  cost: association intact + interval lost is a failure no other section detects. Full design in
  `docs/interval_encoding_design.md`.
- §3 is runnable **only for arms with a state of time**; it is unrunnable under the memoryless probe
  by construction. `paper/capacities.md` ¶11 already hedges it correctly ("For models that support an
  online regime and state of time"). Report which arms were included.
- Both shipped sections score the current list only; neither is crossed with retention (§6, One-shot).

---

## 5. Capacity coverage — `paper/capacities.md` promises vs shipped instruments

### Continual retention

| Promise (¶) | Instrument | Status |
|---|---|---|
| "Across most benchmark suites, we report a retention matrix" (¶3) | `continual_chain` | SHIPPED (2026-09-02, C4 option a). One suite reports one, so the prose should say *the symbolic suite*, not "most benchmark suites" — a spatial chain is a second caller of the same modality-agnostic metric and is not built. |
| "pair it with a reversal protocol… scoring preservation rather than forgetting" (¶3) | `tmaze_reversal`, `paired_associate` | SHIPPED, and now in **both** modalities. The spatial metric is literally named `perseveration`; its symbolic counterpart is `pa_abac_ab_recall_final`, scored lower-is-better because after phase 2 the old response is an intrusion. |
| "more retention is not necessarily better" / "Forgetting … is functional rather than a failure" (¶1) | `continual_chain` rehearsal phase | SHIPPED (2026-09-02). Was **ABSENT**: the prose argued that a finite substrate must discard, while every section treated all stored material as worth keeping and scored only how much survived. `select_selectivity` asks whether what survives is the part still in use. Read it only under `select_under_pressure`. |
| "a fixed neural substrate can hold only so many traces before new memories begin to degrade old ones" (¶1) | `continual_chain` diagonal | SHIPPED (2026-09-02). Was **ABSENT and unnoticed**: the claim is about acquisition degrading under load, and nothing measured acquisition under load. `chain_avg_learning` / `chain_intransigence` do. First result — Original EP's diagonal is *flat* (intransigence −0.01) while its ACC collapses to 0.20, i.e. its forgetting is overwriting, not capacity exhaustion. |
| Load as a swept axis rather than a single point (¶3, implied) | `continual_chain` `forgetting_by_gap` | SHIPPED (2026-09-02). `multiple_sequences` measured T=2 only; the chain gives the retained fraction at every gap. On Original EP: 0.25 / 0.05 / 0.03 / 0.03 / 0.00 — three quarters of the trace is gone after **one** interposed task. |
| Cue competition, as distinct from substrate interference | `paired_associate` | SHIPPED (2026-09-02). Was **ABSENT**: `multiple_sequences` uses disjoint lists (fruit then animals), so nothing ever competed for a cue list A already owned. It is this section's *control condition*, run without an experimental arm. First result — AHN holds AB at 1.00 under the disjoint-cue control and at 0.00 under AB/AC, with `other_rate` 0.00: a clean replacement, invisible to `multiple_sequences`, which reports no interference for the same arm. |

### One-shot learning

| Promise (¶) | Instrument | Status |
|---|---|---|
| "as a function of the number of exposures, from a single presentation upward" (¶7) | `online_convergence`, `presentation_duration` | SHIPPED — narrowed by D3; `anchoring_few_shot` no longer contributes. |
| "pair recall score from exposure with the retention matrix measure of previous sequences" (¶7) | `continual_chain` `exposures` series, third panel of `continual_chain_axes.png` | SHIPPED (2026-09-06) as a *plot* of what the chain already recorded: epochs to criterion at every chain position (censored tasks drawn open at the budget) beside the diagonal, with the fresh-list `convergence_epochs` as reference. Re-drawn for a saved run with `bin/replot_continual_chain.py`. Not yet a scored dimension. |
| "whether consistency with prior structure influences the speed of sequence acquisition" (¶7) | `schema_consistency` | CAVEAT — ladder non-monotone at `random`; not yet quotable as a clean result. |

### Serial order

Temporal contiguity was removed as a promise on 2026-09-02. ¶10 still cites it as the
behavioural signature in the literature, but nothing in the suite claims to measure it, and
the lag-CRP is no longer a target statistic (consequence for `free_recall`: see S-O1 below).
The capacity is now exactly the four ¶11 rows, and each has a named final metric set.

| Promise (¶) | Dimension | Final metrics | Status |
|---|---|---|---|
| "order never established" (¶11) | `establishment` | `order_given_item` (v2), `list_membership_rate` (v1), `establishment_break_length` (v1) | **ABSENT → buildable** — needs S-O2 only. Replaces the `sequence_length` MRR proxy, which conflates order with association. |
| "items recalled, but out of sequence" (¶11) | `binding_ordinal` | `order_error_fraction` (v2), `transposition_locality` (v2), `transposition_asymmetry` (diagnostic), `intrusion_rate` (v1) | **ABSENT → buildable** — needs S-O2 only. `memory_span` zeroes at the first error and cannot separate a transposition from an omission; these can. |
| "associations present but autonomous unrolling collapses" (¶11) | `unrolling` | shipped: `max_memory_span` (v2), `max_memory_span_l2`, `max_memory_span_quantized`, `span_fraction_mean_raw` (v2), `span_raw_over_quantized` (v2), `tmaze_pc_divergence_fraction` (v2). New: `unrolling_gap` (v2), `cascade_recovery_rate` (v2), `cascade_conditional_ratio` (v1), `rollout_margin_at_break` (v1) | CAVEAT → the promised decomposition is now specified. `unrolling_gap` is the literal promise and is *not currently computed* although both halves exist in `length_sweep`. |
| "we probe whether the interval separating two items is retained" (¶11) | `binding_metric` | `peak_time_error`, `interval_discrimination_acc`, `retiming_shift`, `weber_slope` | PLANNED (D7) — online arms only, gated on a state of time. Unchanged. |

**Figures (2026-09-06).** The three ordinal rows had no figure in the run tree: the probe's
numbers reached the scorecard as SVG only. `bin/plot_serial_order_probe.py` now draws two PNGs
from a saved `serial_order_probe.json` into `<arm>/symbolic/plots/`: `serial_order_failures`
(establishment, outcome taxonomy, displacement histogram — all from the fixed-exposure rows, at
the arm's registry `n_epochs`) and `serial_order_unrolling` (cued vs rollout span **at cued
criterion** from the staircase, exposure to each criterion with censoring, cascade
conditionals). Both are declared as Serial order's main set in `bin/build_results_index.py`,
with `interval_retention` third; `length_curves`, `tmaze_completion` and `interval_generation`
are supplementary. Note the probe stores two exposures: `rows` are fixed-budget, `staircase`
is criterion-referenced, and `summary.unrolling_gap` is the staircase mean.

**Dimension weights** (rebalanced 2026-09-02, was `unrolling` 0.40 / `binding_ordinal` 0.35 /
`binding_metric` 0.25 over three dimensions):
`unrolling` 0.35, `binding_ordinal` 0.30, `establishment` 0.15, `binding_metric` 0.20.
`unrolling` stays heaviest — it is the only dimension shipped, and the only one with an
instrument in both modalities. `establishment` is light because it is one conditional
read-out off an existing curve, and it doubles as a **read-before guard**: `order_given_item`
at chance voids the unrolling numbers, because a span means nothing if the order it unrolls
was never established.

**S-O1 — consequence of the contiguity removal for `free_recall`.** The design note
(`docs/free_recall_design.md`) rests its case on the lag-CRP: "the only one of the three that
yields a *genuine* lag-CRP", and §3's dissociation table is a lag-CRP table. With contiguity
out of scope that argument is void. The architecture dissociation it promised survives in
cheaper form — `transposition_locality` + `transposition_asymmetry` off cued recall separate
pure chain (few transpositions until derailment) from symmetric associator (locality graded,
asymmetry ≈ 0.5) from context-carrying (graded, forward-skewed) — so `binding_ordinal` no
longer depends on free recall. **Demote it**: it drops out of the pending register's
buildable list; its unique remaining contributions are the initiation curve (basin sizes from
a neutral start) and unconstrained-order recall, neither of which any ¶11 row asks for.

**S-O2 — one code change unlocks three of the four rows.** `measure_recall_associative` and
`measure_recall_autoregressive` both already decode a word at every step
(`decoder.decode(pred, top_k=1)[0]`) and discard the identity, keeping only a boolean.
Returning the decoded identities alongside the success curve — a per-trial × per-position
array of study positions, with a sentinel for out-of-list — is the whole prerequisite for
`establishment`, all of `binding_ordinal`, and the two `cascade_*` metrics. No new model
interface, no new protocol, no new section: the length grid of `sequence_length` is already
the right substrate. This supersedes `free_recall` as the cheapest buildable serial-order
work.

Metric definitions, all over the `sequence_length` length grid (L ∈ {3,5,7,9,11}), all from
cued recall unless stated:

- `list_membership_rate` — fraction of cued predictions decoding to *any* studied item.
  Chance = (L−1)/|vocab|. The denominator for the row below.
- `order_given_item` — P(decode is the correct next item | decode is a studied item).
  Chance = 1/(L−1). High membership with chance-level `order_given_item` is
  "order never established" as distinct from "nothing stored".
- `establishment_break_length` — longest L at which `order_given_item` ≥ 0.75.
- `order_error_fraction` — of all cued failures, the fraction that are in-list
  transpositions rather than omissions or intrusions. This is precisely the separation
  `memory_span` cannot make.
- `transposition_locality` — of transposition errors, the fraction at displacement
  |d| = 1, where d = (study position of the decode) − (true target position).
- `transposition_asymmetry` — forward share, count(d=+1) / [count(d=+1) + count(d=−1)].
  **Diagnostic, never scored**: 0.5 = symmetric associator, > 0.5 = the forward bias the
  STDP argument of ¶10 predicts. It is an architectural fingerprint, not a quality.
- `intrusion_rate` — decode falls outside the studied list. Lower is better.
- `unrolling_gap` — 1 − span(rollout, raw) / span(cued), meaned over the length grid.
  The literal ¶11 clause. Requires `length_sweep` to also store the cued span; it currently
  stores cued **MRR** and rollout spans, so the contrast is one line away and unmade.
- `cascade_recovery_rate` — P(correct at step k+1 | error at step k) under free rollout.
  0 = an error is terminal; this is `error_cascade`.
- `cascade_conditional_ratio` — P(correct k+1 | correct k) / P(correct k+1 | error k).
  The magnitude of propagation, and the conditional per-step free-run accuracy ¶11 asks for.
- `rollout_margin_at_break` — target-minus-best-competitor cosine at the step where the
  span ends. Resolves below the accuracy floor, where the span is already 0 and flat.

`bin/probe_rollout_cascade.py` already computes the rollout geometry behind the last four as
a diagnostic; the work is promoting its per-step conditionals into section metrics, not
writing them from scratch.

### Pattern completion

| Promise (¶) | Instrument | Status |
|---|---|---|
| "for the corrupted cue, we increasingly inject cue noise" (¶17) | `noise_invariance` | SHIPPED — narrowed by D5. |
| "for the partial cue, completeness is manipulated by masking a swept fraction of the cue's dimensions" (¶17) | `cue_masking` | SHIPPED (2026-09-02). Built on `HierarchicalEncoder`, **not** `SymbolicEncoder`: the latter's dimensions are a random basis, so masking them shrinks cosine to every embedding at roughly the same rate and scores the vocabulary's geometry rather than the arm — the exact failure that got `letter_noise` deleted (D5). The hierarchy's columns are owned by tree nodes, so an item's features split into a **shared** (ancestor/category) and an **identity** (own leaf) block, which is what makes this a *modality-like* manipulation rather than a fraction of an arbitrary basis. |
| "probing cued recall **and autoregressive rollout**" for both manipulations (¶17) | `cue_masking` (structural half); `noise_invariance` (corruption half, 2026-09-06) | **Closed (2026-09-06).** The corruption half now runs the rollout too: `noise_invariance` sweeps sigma on the initial cue only and reports (since 2026-09-26 under **raw** feedback only: quantized restated the cued curve, and l2 coincided with raw on four of six arms while shifting DTS-ESN and EP in opposite directions) `noise_rollout_auc`, `noise_rollout_tolerance`, `noise_completion_gap` (diagnostics, unscored, for the same reason as `mask_completion_gap`); `noise_invariance.png` gained the rollout panel. Pending item 9 is closed. Earlier status: **Half closed.** The structural manipulation now runs both probes: `cued_probe` masks every position independently, `rollout_probe` masks only the initial cue and free-runs. The rollout results are reported as diagnostics rather than scored, because their *level* is set by free-running ability, which Serial order/`unrolling` already scores — what this section contributes is their shape across the masking axis. The **corruption** half is still cued-only: `measure_recall_autoregressive` runs at a hardcoded `noise_scale=0.05` and is not crossed with σ. That is pending item 9, and it is now one line: both recall helpers took a `cue_transform` hook in this change. |
| "we only manipulate the initial cue" (¶17) | `measure_recall_autoregressive` | SHIPPED — accurate; noise is applied to item 0 only. |
| Probe property *availability* (paper figure B: cue without the audio modality, or audio alone) | `cue_availability` (2026-09-06; redesigned 2026-09-26) | SHIPPED as diagnostics under `cue_completeness`. Two-block `[symbolic \| audio]` input where **every item has its own tone** (was one tone per sequence, which made an audio-only cue positionless: it could only be scored on membership and its rollout was a fixed point). Cues: both / audio removed / symbol removed, each scored on exact next-item recall, cued over every item and rolled out from each sequence's first item; single and 3-sequence conditions, 10 items per sequence; criterion-referenced on the full cue with round-robin ingestion. Not weighted: whether availability is its own dimension is a paper decision. |
| "score how an initial error propagates" — the margin under rollout (¶17) | `noise_invariance` rollout margin (2026-09-06) | SHIPPED: `noise_rollout_margin_{first,later}_{auc,slope}`; third panel of `noise_invariance.png` separates the corrupted step from the self-driven steps. Diagnostic. |
| "score how an initial error propagates" (¶17) | `mask_completion_gap`; `cascade_recovery_rate`; `divergence_step` | SHIPPED, across three instruments and none of them scored under this capacity. `mask_completion_gap` (cued AUC − rollout AUC at matched completeness) is the structural version and is diagnostic — confounded with free-running ability, so read within-arm. `cascade_recovery_rate` is `error_cascade` and is scored under Serial order/`unrolling`. `divergence_step` is the spatial analogue and is scored under `cue_point` (C1). The ¶17 clause is measured; what it does not have is a *scored* home inside Pattern completion, deliberately, to avoid counting free-running weakness twice. |

### Sequence disambiguation

| Promise (¶) | Instrument | Status |
|---|---|---|
| "sequences that share a common stretch of observations" (¶22) | `tmaze_disambiguation`, `symbolic_disambiguation` | SHIPPED — D2 landed (graded odour availability) and the symbolic section owns the modality-free fork. |
| "grading how much external support is available… from persistently available to only present in the beginning" (¶22) | `symbolic_disambiguation` axes A (withdrawal) and A' (support and delay decoupled); `tmaze_disambiguation_graded` | SHIPPED. Onset-only rows are protocol-limited for arms without `StatePrimeable` (C2). |
| "the length of the shared stretch… varied independently" (¶22) | `symbolic_disambiguation` axis D (2026-09-06): shared length 2–10 with the suffix fixed, under persistent and onset-only support | SHIPPED — `symdis_max_shared_len_persistent` (scored), `symdis_max_shared_len_onset` (protocol-limited); sixth panel of `symbolic_disambiguation.png`. |
| "the number of episodes sharing it" (¶22) | `symbolic_disambiguation` axis C, N ∈ {2,3,4,6,8}, with the orthogonal-discriminator control | SHIPPED — `symdis_max_episodes_above_chance`, `symdis_load_disambiguation_cost`. Spatial stays at N = 2. |
| "the delivery interval" (¶22) | `symbolic_disambiguation` hold-delay rows (fixed 2-step cue, gap 0–6); `tmaze_disambiguation_graded` delay | SHIPPED — `symdis_max_delay_at_fixed_support` (protocol-limited). |
| Splitter cells fire differently at the same location by trajectory (¶21) | splitter index | DESIGNED — needs a hidden-state accessor. The prose makes a within-shared-stretch claim that behaviour alone cannot support. |
| Drifting context (`<< expand on theories of drifting context? >>`, ¶21) | — | ABSENT from the suite entirely. Expanding this paragraph enlarges the gap. |

---

## 6. Pending register, ordered by cost

**Decided, buildable now** (D2, D7 — plus the removals)

1. **D2** — wire `BifurcatingRouteGenerator` + `SpatialDisambiguationBenchmark.sweep()` over an
   `OverlapConfig` grid spanning odour availability (full → onset-only) and shared-stretch length.
   Keep the current configuration as the `delay = 0` corner so the shipped number stays comparable.
2. **D2 metrics, same change** — divergence accuracy + recovery profile + shared-stretch control;
   branch margin; confusion matrix + context-graded confusion index; linear-oracle gap. Not optional:
   the current metric eases as the axis hardens.
3. **D7** — `interval_encoding` in the online suite.
4. **D1/D3/D4/D5/D6 removals** — resolve C1 (move `coverage`/`divergence_step` onto
   `tmaze_completion`) before deleting, then strip the dead helpers.

**Buildable now, no new interface, not yet scheduled**

5. ~~Cue-dimension masking sweep (Pattern completion ¶17)~~ — **DONE 2026-09-02**, as the
   `cue_masking` section of the symbolic suite. `cue_completeness` leaves `UNBUILT_DIMENSIONS`,
   which takes Pattern completion from 0.75 coverage to 1.00. Three things went in beyond the
   scoped sweep, each because building the sweep exposed a question it could not answer:

   - **A model-free reference, and the scored metrics restricted to it.** Masking is only a model
     measurement while the masked cue still *names* its item. `cue_identifiability` is a matched
     filter on the raw cue, computed from the encoder before any arm exists. Past the point where
     several studied items' cues become identical, the arm receives one input for several targets
     and a full-grid AUC scores which colliding successor it happens to favour. Every scored metric
     is therefore an in-bound rate (`mask_*_recall_in_bound`), the full-grid AUCs are kept as
     diagnostics, and `mask_resolved` gates the dimension. This is `projection_ratio`'s role in
     `schema_consistency`. **It is a reference, not a ceiling**: recall above it means the arm
     recovers a target the degraded cue no longer specifies, which is completion as against cue
     matching, and is what `mask_*_completion_advantage` reports.
   - **The block contrast**, which is the part `paper/capacities.md` ¶17 does not ask for and the
     Horner et al. (2015) holistic-retrieval evidence it cites does. Masking `shared` (category)
     features against `identity` (own-leaf) features at an equal number removed asks whether *which*
     features are missing matters, or only how many. `mask_block_asymmetry` is diagnostic — an
     architectural fingerprint like `transposition_asymmetry`, never a quality — and ~0 is the
     holistic-retrieval null: the cue is a bag of dimensions to that arm.
   - **A `cue_transform` hook** on `measure_recall_associative` / `measure_recall_autoregressive`,
     applied after the Gaussian noise so the two compose. Default `None` is bit-for-bit the old
     behaviour. Item 9 below is now one line rather than a rewrite.

   Two properties of the substrate are worth carrying forward. The tree is **two-level**
   (`branching=(2, 12)`), shallower than `schema_consistency`'s: a three-level tree gives 2:1
   blocks whose smaller half is exhausted at a third of the sweep, taking the reference to 0 with
   it and leaving the asymmetry a three-point window that reads 0.0 for want of range. And
   `alpha = P/N ~ 0.051` is held well under the Hopfield-class knee, so a substrate-capacity
   ceiling is not read as a masking result — the confound that put `sequence_length` in CAVEAT.

   **Validity, on the letter_noise test (five arms, exposure pinned at 100 epochs).** D5 deleted
   `letter_noise` for an across-model spread of 0.000–0.111 against the σ sweep's 0.711. This
   section clears that bar on every scored read-out: `mask_random_recall_in_bound` 0.998,
   `mask_shared_recall_in_bound` 1.000, `mask_identity_recall_in_bound` 1.000,
   `mask_random_tolerance` 0.917, `mask_random_margin_in_bound` 0.554. It measures the arm, not
   the encoder. `mask_block_asymmetry` is the narrowest at 0.146, which is expected for a
   fingerprint rather than a quality.

   **First results.** Hopfield completes at 0.998 in bound with `completion_advantage` +0.025 —
   it recovers targets slightly beyond what the raw cue specifies, and `mask_block_asymmetry`
   0.000 is the **holistic-retrieval null**: it does not care which block is missing. Original EP
   sits at 0.772 with advantage −0.235, i.e. it fails inside the region where the cue was still
   unambiguous, and its asymmetry is −0.086 (mildly identity-leaning). DG-EP never acquired the
   list at this budget and is correctly withheld — `mask_acquired` False, `mask_resolved` False,
   so `cue_completeness` is dropped for it rather than scored as 0.

   **Cost.** The sweep, not the training, dominates: one grid point per cue feature × 3 modes × 2
   probes. Hopfield 243s and Original EP 204s under load, the smaller arms 11–25s. Turn
   `features_per_node` or `n_draws` down before touching `epochs`.
6. **S-O2 — decoded-identity return** from `measure_recall_associative` /
   `measure_recall_autoregressive`. Now the cheapest buildable serial-order work, and it
   unlocks items 7 and 8 below plus the whole `establishment` dimension in one change.
7. `error_cascade` + conditional per-step free-run accuracy (`cascade_recovery_rate`,
   `cascade_conditional_ratio`) — unblocks ¶17 and Serial order ¶11. Depends on 6.
   `unrolling_gap` ships with it and needs only the cued span added to `length_sweep`.
8. Order-error metrics: `order_error_fraction`, `transposition_locality`,
   `transposition_asymmetry`, `intrusion_rate`. Depends on 6.
8b. ~~`free_recall`~~ — **demoted 2026-09-02**, see S-O1. Its case was the lag-CRP, which is
   no longer a promise; `binding_ordinal` is covered by 8 at a fraction of the cost. Design
   retained in `docs/free_recall_design.md` for the initiation curve, which nothing asks for
   yet.
9. ~~Autoregressive rollout crossed with the σ sweep.~~ **DONE 2026-09-06** (`noise_invariance` rollout sweep, l2 + quantized feedback).
10. `sequence_length` load axis: α = P/N log-spaced or staircase-to-criterion, plus the >12-word
    vocabulary fix.
11. N > 2 confusable episodes.
12. Linear oracle (`docs/nonlinearity_benchmark_design.md` §3) — specified, not implemented.

**Closed by the C4 decision (2026-09-02)**

13. ~~Retention matrix~~ — **DONE**, promoted into the symbolic suite as
    `continual_chain`, and extended with the diagonal, the load curve and the
    rehearsal phase (see §0). `multiple_sequences` is kept: it is the T=2 corner
    and is what every earlier result was reported against.
14. Exposure × retention crossing (One-shot ¶7) — **PLOTTED (2026-09-06)**,
    not scored. The chain trains every task to criterion and stores the per-task
    `exposures`; `continual_chain_axes.png` now draws them as a third panel
    (epochs to criterion against chain position, fresh-list reference,
    censoring marked). The streamed version — `online_convergence`'s ladder run
    *inside* a chain — is still unbuilt.

**Opened by it**

14b. **A spatial chain.** `paper/capacities.md` ¶3 says "across most benchmark
    suites"; exactly one reports a matrix. `memval/metrics/retention.py` is
    modality-agnostic and `ContinualChainBenchmark` is not — it takes a word
    list and an encoder. Either generalise the benchmark or write a spatial
    caller of the metric, or soften the prose to name the symbolic suite.
14c. **The chain saturates for the linear arms.** At the default 6×5 both AHN
    and tPC sit at ACC 1.00 / forgetting 0.00, so `select_under_pressure` is
    False and the `relevance` dimension is correctly dropped for them — the
    question is not posed rather than failed. The vocabulary caps the chain at
    6 tasks (6 categories, smallest has 6 words), so raising load past that
    needs the >12-word vocabulary fix already logged at item 10. This is the
    same ceiling recorded for the 4×5 chain; 6×5 moved it, it did not remove it.
14d. **`select_rehearsal_epochs` is not a controlled variable.** Rehearsal
    re-presents at the chain's own per-stage budget, so an arm with a large
    `n_epochs` rehearses harder than one without. Fine for a within-arm
    difference-in-differences, not for ranking `select_rehearsed_gain` across
    arms. Match total exposure before quoting it cross-arm.
14e. **Watch `pa_abac_ac_trials_to_criterion` for saturation.** It has range —
    AHN overwrites on phase-2 pass 1, Original EP takes 4 at `epochs=100` — but
    the AHN end is the floor of the scale, the same way `tmaze_reversal`
    saturates at high `epochs`. Lower `epochs` if a comparison bunches at 1; the
    recall and intrusion read-outs are unaffected either way.
14f. **`pa_abac_ab_acquired` is reported, not gating.** Original EP reaches only
    0.70 on the AB list in 3 passes, below the 0.75 criterion, so its
    interference read-outs rest on a partly-learned association. The guard is
    emitted and shown but does not drop the `contingency` dimension, because
    that dimension also carries `tmaze_reversal` and a symbolic-only guard must
    not silence the spatial instrument — the same call already taken for
    `reversal_*_acquisition_final_arm_accuracy`. Raise `ab_trials` when the
    guard trips.
14g. **`other_rate` is doing real work, and should be reported.** It separates
    replacement from destruction, which a fall in AB recall alone cannot. First
    results: under the disjoint-cue control Original EP loses AB 0.70 → 0.22
    with `other_rate` **0.78** (destroyed into noise), while under AB/AC it loses
    AB → 0.00 with `other_rate` **0.04** (cleanly replaced by C). AHN sits at
    `other_rate` 0.00 in both. Two arms, two different failures, one number
    apart.

**Needs a new model interface**

15. Splitter index — hidden-state accessor.

**Blocked on the state-carrying ingestion protocol** (`nonlinearity_benchmark_design.md` §4)

16. D2's onset-only rung as a *model* result rather than a protocol artefact.
17. Aliased chain / high-order Markov; the five proposed nonlinearity sections, none implemented
    (four exist only as `dashboard/docs/benchmarks/` stubs that read as if shipped).
18. **Sequence identity as a capacity — "continuation vs. new episode."** Given a familiar item,
    can the model tell *this continues an episode I know* from *this is a new episode that happens
    to start with a familiar item*? Raised 2026-09-02 against `schema_consistency`, where appending
    a new item to a learned list is silently ambiguous between the two readings. **The harness has
    no channel to express the distinction**: `fit_sequence` updates per adjacent pair, no arm reads
    `current_t` during training, and the probe cues one item for its successor with no state carried
    — so every current arm is a pure pairwise associator and the two readings produce the *same*
    weight update. The distinction becomes measurable only once an arm carries state across a
    sequence. Structurally this is the sequence-identity twin of Sequence disambiguation (shared
    stretch, divergent continuation), and belongs beside it rather than inside the schema section.
    Prerequisite for a *model* result: `docs/nonlinearity_benchmark_design.md` §4. An explicit
    sequence identifier clamped as context is the natural instrument, but only
    `AsymmetricHopfieldNetwork` has an `n_context` dimension today; four arms accept `context_data`
    and silently drop it, and `OriginalEqPropSequenceNetwork` does not accept it at all — so it
    needs a capability declaration (`memval/models/capabilities.py`) before it can be a suite
    section.

**Larger design work**

18. Time-to-consolidation (`schema_benchmark_design.md` §8) — trials-to-forgetting per rung.
    Named "the largest missing piece"; would also dissolve the non-monotone-ladder confound.
19. Drifting context as the disambiguation substrate.
20. Progressive differentiation — `HierarchicalEncoder`'s SVD structure predicts coarse-before-fine
    acquisition order and is asserted exactly in `test_one_singular_dimension_per_split`, but only a
    single snapshot (`category_accuracy` 0.91 vs `item_accuracy` 0.35) measures it. Unclaimed result.
21. Drift correction as a named capacity — the home for D3's `anchoring_few_shot` if it returns.
    Note the anchor-period `k` was never swept (fixed at 2), so the drift question was never asked.

---

## 7. Discrepancies to fix in prose and docs

| Where | Problem |
|---|---|
| `paper/capacities.md` ¶3 | ~~unsupported after D8~~ — **fixed 2026-09-02**: `continual_chain` reports one. But "most benchmark suites" overstates *one* suite; and ¶3 describes the matrix as scoring stability only, which is now half of what it does (see 14b). Rewritten. |
| `paper/capacities.md` ¶7 | Pairing exposure against the retention matrix — the instrument exists again, the crossing does not. See pending item 14. |
| `paper/capacities.md` ¶1 | Argues forgetting is functional and "more retention is not necessarily better", which had no instrument until the rehearsal phase. Now supported — but only under `select_under_pressure`, which is False for the linear arms at the default load (14c). Do not quote the claim as met for those arms. |
| `paper/capacities.md` ¶10, ¶21 | Inline author markers still present: `<<add few words for autonomous recall?>>`, `<< expand on theories of drifting context? >>`. |
| `paper/capacities.md` ¶22 | Once D2 lands, this paragraph is met — but must state that the onset-only end is a protocol finding, not a model ranking (C2). |
| `docs/results_intro_capacities.md` | Capacity map still keyed to **Autonomous generation**; the paper's fifth capacity is **Serial order**. Rows no longer align with the paper's headings. |
| `docs/results_intro_capacities.md` | Rows for `spatial_sequence`, `anchoring_few_shot` and `letter_noise` must be deleted (D1/D3/D5); the `online_continual` row depends on C4. |
| `docs/disambiguation_design.md` | §6 build order steps 1–2 are now decided work (D2), not proposals. |
| `run.md` | Stale suite descriptions — omits `schema_consistency`, `tmaze_reversal`; lists sections removed by D1/D3/D5. |
| `dashboard/` | Reads `results/<model>/<suite>/metrics.json`; removed keys and the dropped `online_continual` suite need handling. `dashboard/docs/benchmarks/` also documents `context_gating`, `high_order_markov`, `delayed_recall`, `limit_cycle` — none of which exist. |
| `tests/` | `tests/test_pipelines.py` and any test asserting removed sections/keys. |
