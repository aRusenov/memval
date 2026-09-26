# The five capacities, as questions

**Drafted 2026-09-03**, from the scorer's capacity → dimension → section map
(`bin/score_capacities.py`, `DIMENSION_WEIGHTS`) and what each section actually
emits. Each row is one *dimension*: the thing a section manipulates to pose the
question. Weights are the dimension's share of its capacity's score.

**Capability tags.** Not every question can be put to every model. A tagged row is
scored *not applicable* for arms lacking the capability — never zero — and its
weight is removed from that arm's coverage rather than from its score.

| tag | needs | declared by |
|---|---|---|
| 🕐 **online-only** | ingests one event at a time, no second pass | `OnlineTrainable` |
| ⏱ **time-clocked only** | elapsed time changes its state; can emit a gap | `TemporallyClocked`, `TimingPredictive` |
| ⚑ **state-carrying only** | holds a signal across steps where it is absent | `StatePrimeable` — declared by `dts_esn` and `temporal_pc`, for which these rows score; for every other arm they are *protocol-limited* under the memoryless one-step probe and excluded |

Two read-outs cut across all five capacities and are worth naming once:
**exposure** (every section trains until it reaches a criterion and reports what
that cost, so "how fast" is always measured and never assumed) and the
**margin** (target minus best competitor, which keeps resolving after accuracy
has floored or saturated).
**Reading the Benchmarks tables.** Each capacity below now carries a *Benchmarks*
subsection listing, per section: what is measured, what is written out (plots
under `<ClassName>/<suite>/plots/`, scalars into `metrics.json`), and the
conditions it was measured under. Three condition words recur and mean exactly
one thing each:

| condition | meaning |
|---|---|
| **criterion** | exposure is staircased (`epochs_to_criterion`: geometric ladder then a refine scan, rebuilt from scratch at each checkpoint) until a read-out **upstream** of the scored one reaches 0.95, and the cost is reported as `<s>_epochs_to_criterion`. The scored quantity is then read at that exposure. A section that never reaches criterion inside `max_epochs` is **censored** and flagged `<s>_criterion_reached: False`. |
| **cued** | one-step probe: each position cued independently with its own item plus N(0, 0.05²) noise, `n_trials=30`; `predict_next` is treated as pure. Recall rate excludes position 0. |
| **rollout** | autoregressive: cue the first item, feed each prediction back. The feedback projection is a stated condition: **raw** (prediction fed back as is), **l2** (unit-normalised), **quantized** (snapped to the nearest codebook item — the drift-free upper bound). `memory_span` is the longest prefix recalled at ≥ 0.75. |

Two suites, two encoders: the **symbolic** suite uses `SymbolicEncoder` (100-dim
unit-norm vectors, category structure via `category_variance`), the **spatial**
suite `PlaceCellEncoder` on T-maze routes. The **online** suite is the streamed
regime of the symbolic material (`fit_event`, one event at a time).

---

## 1. Continual retention

*Does what was learned earlier survive what is learned later — and can something
that has become wrong be unlearned?*

| # | Question | What we vary | w | Section(s) |
|---|---|---|---:|---|
| 1.1 | After other lists have been learned, does an earlier one still come back? | how many lists intervene (load) | 0.25 | `multiple_sequences`, `continual_chain` |
| 1.2 | Can a new list still be learned once the memory is already holding several? | how many lists are already stored when the new one arrives | 0.20 | `continual_chain` (the matrix diagonal) |
| 1.3 | When a learned association stops being true, how quickly is it overwritten, and does the old response linger? | the contingency: reward moved to the other arm; the same cues re-paired with new targets | 0.35 | `tmaze_reversal`, `paired_associate` |
| 1.4 | When there is not room for everything, does the memory keep what is still in use rather than what came first? | which subset is re-presented, under capacity pressure | 0.10 | `continual_chain` (selective retention) |
| 1.5 | Does learning something new damage related memories more than unrelated ones? | how related the new item is to each old one (same / sibling / far category) | 0.10 | `schema_consistency`, focused protocol |

Stability (1.1, 1.2, 1.4, 1.5) and plasticity (1.3) are scored separately on
purpose: they pull in opposite directions on how readily the substrate changes,
so no single policy can game both.


### Benchmarks

| section | measures | plot | scalars | conditions |
|---|---|---|---|---|
| `multiple_sequences` | whether list A survives learning list B (the T = 2 case of load) | `multiple_seq_forgetting.png` — A's serial-position curve before vs after B | `delta_mrr_forgetting` (= MRR_after − MRR_before; 0 is none, −1 total loss), `multiple_seq_mrr_after` | `n_lists` (default 5) pairs of 7-item lists from disjoint categories, canonical pair fruit → animal, serial-position curve = P(recalled) over pairs, per-position margin drawn alongside; plus an **overlap** condition (B from A's own category vs a disjoint-category control, both L=6, identical budgets; model-free `in_span` covariate written; `multiple_seq_overlap_cost_{prob,margin}`; `b_exposure_multiplier` for plasticity asymmetry); **each list trained to the fixed registry budget (criterion when unpinned)** on its own cued recall before the delta is read; probe **cued**; an interleaved A+B condition is also run and reported |
| `continual_chain` | retention under load: a 6 × 6 matrix R[j, i] = recall of task i after training through task j | `continual_chain_retention.png` (the matrix), `continual_chain_axes.png` (learning vs forgetting), `continual_chain_selectivity.png` | `chain_avg_accuracy`, `chain_avg_forgetting`, `chain_retention_ratio`, `chain_avg_learning`, `chain_intransigence`, `chain_backward_transfer` | 6 tasks × 5 items, one category each; **each task trained to criterion 0.95 in 1-epoch steps** (max 512) on the same model, never reset; probe **cued**, seeded per task. The diagonal is 1.2 (acquisition under load), the sub-diagonal 1.1 |
| `continual_chain` → selective retention | when the chain is under pressure, is the re-presented subset kept at the expense of the rest? | `continual_chain_selectivity.png` | `select_selectivity` (difference-in-differences: rehearsed gain − unrehearsed drift), `select_rehearsed_gain` | rehearse 3 of 6 tasks at the registry `n_epochs` after the chain; **guarded** by `select_under_pressure` (avg forgetting > threshold) — an arm that never saturates the chain cannot be asked this, and today only the two EP arms can |
| `paired_associate` | when the same cue is re-paired with a new target, is the old response overwritten or does it linger? | `paired_associate_abac.png` — old / new / other response rates through phase 2 | `pa_abac_ab_recall_final`, `pa_abac_ac_recall_final`, `pa_abac_ac_trials_to_criterion`, `pa_abac_other_rate_final`, `pa_cue_competition_cost` | 5 cue–target pairs from disjoint categories; phase 1: AB for 3 trials; phase 2: AC for up to 6 trials, criterion 0.75 on the new association; 100 epochs per trial; probe **cued**, seeded. A **disjoint-cue control** (`pa_control_*`) trains a second list whose cues do not collide, so the cost of *cue competition* specifically is the difference |
| `tmaze_reversal` | how fast a learned place → reward association is extinguished and its opposite acquired, and whether the old arm is perseverated | `tmaze_reversal.png` — arm accuracy, reward prediction and anticipation lead by stage | `reversal_*_reversal_trials_to_criterion`, `reversal_*_reversal_final_perseveration`, `reversal_*_final_arm_accuracy`, `extinction_reward_drop_fraction` | 16-step T-maze, 400 place + 20 reward dims; **one trial = one `fit_sequence` call at 3 epochs** (the constructor value — the section's exposure contract), 12 trials per stage, criterion arm accuracy ≥ 0.75 on 2 consecutive trials; two protocols, **direct** (reward switched) and **extinction** (unrewarded trials first); probe is a fully **open-loop rollout** from the shared stem with nothing clamped, because a reward is an outcome and clamping it would hand the model the goal |
| `schema_consistency` (focused) | does learning a new item damage its own category more than a sibling or a far one? | `schema_consistency.png` (interference panel) | `schema_<rung>_interference_same / _sibling / _far`, `schema_<rung>_prefix_recall` | the interference read-outs are **valid only under `interference_protocol=focused`** (the default `extended` rehearses every base transition of the host category, so its `same` bucket measures rehearsal, not damage); run as a separate invocation into `_schema_focused/` |

`delta_mrr_forgetting` is the one signed metric here: negative is forgetting.
Until 2026-09-04 the scorer normalised it as if the sign ran the other way and
awarded a catastrophic −0.61 a clipped 1.0; it now maps 0 → 1 and −1 → 0.

## 2. One-shot learning

*How much exposure does a new episode need?*

| # | Question | What we vary | w | Section(s) |
|---|---|---|---:|---|
| 2.1 | How many presentations does a fresh list need before it is fully recalled? | number of presentations | 0.50 | `presentation_duration` |
| 2.2 | Is a new item learned faster when it fits what the memory already knows? | how consistent the new item is with an acquired schema | 0.25 | `schema_consistency` |
| 2.3 | 🕐 Can an episode be acquired from a single pass, one event at a time, with no chance to revisit? | number of streamed presentations | 0.25 | `online_convergence` |

One-shot acquisition is easy to fake — a memory that overwrites indiscriminately
is one-shot by construction — which is why 1.2 pairs acquisition with what it
cost the lists already stored.


### Benchmarks

| section | measures | plot | scalars | conditions |
|---|---|---|---|---|
| `presentation_duration` | epochs until a fresh 7-item list is fully recalled | `convergence_curve.png` — MRR against epoch on a log ladder | `convergence_epochs` (None if never), `convergence_mrr`, `convergence_span`, guard `converged` | geometric ladder 1, 2, 4 … up to `max_epochs` (500; 2048 for EP-family arms); **cued** MRR ≥ 0.95 is the convergence test; `convergence_span` is read at that point from a **raw rollout**; probe seeded |
| `schema_consistency` (extended) | trials to acquire one new item as a function of how consistent it is with an already-acquired schema | `schema_consistency.png` — the ladder: trials-to-criterion per rung, plus the new-item AUC | `schema_<rung>_trials_to_criterion`, `schema_<rung>_new_item_recall`, `schema_<rung>_new_item_auc`, `schema_consistency_speed_corr`, `schema_consistency_auc_corr` | four rungs of consistency — **duplicate / within / across / random**; base schema acquired on a (2, 2, 6)-branching hierarchical vocabulary over 40 trials; the new item then presented **one epoch per trial** for up to 15 trials, criterion = 0.75 × the arm's own base MRR (self-referenced, because base MRR differs across arms); **guards** `schema_at_ceiling` (every rung acquires on trial 1 — read the AUC instead) and `schema_at_floor` (no rung reaches criterion — the two slow EP-family arms and tPC land here) void the ladder rather than score it |
| `online_convergence` 🕐 | recall after 1, 2, 5, 10, 20 **streamed** passes — one event at a time through `fit_event`, no second look at any event | `online_convergence.png` — MRR and span against passes | `online_mrr_1shot`, `online_mrr_5pass`, `online_mrr_20pass` and the matching `online_span_*` | 7-item fruit list ingested through `ingest(..., regime="streamed")`; the sequence seam is signalled with `on_event_boundary` so no transition forms across it; probe **cued**; not applicable (skipped, never zero) for an arm without `OnlineTrainable` |

`isi_tolerance` lives in the same online suite and plots beside
`online_convergence`, but it is **not** a one-shot or a timing measure: it
interleaves 0, 1, 3, 5, 10 Gaussian noise *events* between items and asks
whether recall survives interposed material. It is filed under Continual
retention as a diagnostic and enters no score.

## 3. Pattern completion

*Can a whole episode be recovered from a degraded piece of it?*

| # | Question | What we vary | w | Section(s) |
|---|---|---|---:|---|
| 3.1 | How much noise can the cue carry before the wrong item comes back? | amplitude of noise added to the cue | 0.40 | `noise_invariance` |
| 3.2 | Given a fragment of a route, does recall find the rest, and how far does it get before losing it? | the cue point along a spatial route, recalled open-loop | 0.35 | `tmaze_completion` |
| 3.3 | How much of the cue can be missing entirely — and does it matter *which* parts are missing? | fraction of features removed; whether the removed block is what the item shares with its kind or what makes it unique | 0.25 | `cue_masking` |
| 3.4 | Can the episode be completed when a whole *modality* is absent from the cue — the item without its sound, or the sound without the item? | which modality the cue carries (both / audio removed / symbol removed); every item has its own tone; one sequence vs three | — (diagnostic, unweighted; see note) | `cue_availability` |

3.4 (added 2026-09-06) is the third probe property of the paper's figure —
*availability* — and is reported under `cue_completeness` as diagnostics: whether
it earns its own weighted dimension is a paper decision not yet taken.

3.1 corrupts every feature a little; 3.3 removes some features and leaves the
rest exact. They are the two ways a cue can be incomplete, and each has its own
model-free reference — the point at which the degraded cue no longer names its
item on its own — so that recall above the reference is completion and recall
below it is not the model's failure.


### Benchmarks

| section | measures | plot | scalars | conditions |
|---|---|---|---|---|
| `noise_invariance` | how much Gaussian noise the cue carries before the wrong item comes back | `noise_invariance.png` — MRR against σ | `noise_tolerance_threshold` (the first σ at which MRR < 0.5), `noise_sweep_mean_mrr` | 7-item list trained **to criterion at σ = 0** (so the tolerance curve cannot be told from an under-trained model), then σ swept over `linspace(0, 1, 11)` added to the **cue only**; probe **cued** |
| `tmaze_completion` | given the first 30% of a route, how far does open-loop recall follow the rest before leaving it | `tmaze_completion.png` — true route vs recalled trajectory | `tmaze_pc_coverage` (fraction of the true route within ε), `tmaze_pc_divergence_step` (first step > ε away), `tmaze_pc_mse`; `tmaze_pc_divergence_fraction` is also read by Serial order's `unrolling` | 15-step T-maze on a 10 × 10 place-cell grid; ε = 0.15 arena units; **criterion on teacher-forced one-step prediction along the route** (are the transitions learned), scored on a pure **open-loop rollout** from the cue point (`anchor_period=0`, nothing re-clamped) — criterion upstream of score, so the section cannot guarantee its own result |
| `cue_availability` | completion when a whole modality is absent: the item without its sound, the sound without the item | `cue_availability.png` — per condition (single: 1 sequence x 10 items; multi: 3 disjoint 10-item sequences, fruit / animal / number), for the full / symbol-only / audio-only cue: blue = cued exact next-item recall over every item-audio pair, red = exact next-item recall along a rollout launched from each sequence's first item; margins and next-tone recovery beneath | `avail_<cond>_<cue>_recall`, `_rollout`, `_margin`, `_audio_recall` for cue in both / symbolic / audio; config `avail_<cond>_chance_tone`, `_tone_cosine_max`; guards `avail_acquired`, `avail_<cond>_criterion_reached` | two-block input `[symbolic (100) \| pure-tone code (100 bins)]`, jointly unit-normalised (`MultimodalEncoder`); **each item has its own tone** (30 evenly spaced tones, seeded shuffled assignment, sigma 4 Hz, pairwise cosine <= 0.06; until 2026-09-26 the tone was per sequence); partial cues renormalised; next item decoded from the symbolic block against the whole 55-word vocabulary whatever the cue carried; **criterion-referenced on the full cue**, round-robin ingestion across sequences |
| `cue_masking` | how much of the cue can be missing entirely, and whether it matters *which* features are missing | `cue_masking.png` — recall and target-minus-competitor margin against fraction masked, one line per mode, with the model-free reference | `mask_<mode>_tolerance`, `mask_<mode>_recall_in_bound`, `mask_<mode>_margin_in_bound`, `mask_<mode>_margin_auc` for mode ∈ {random, shared, identity} | a **hierarchical** encoder (branching (2, 12), 6 features per node) so that masked dimensions mean something: **shared** removes the block the item has in common with its category, **identity** the block that makes it unique, **random** an unstructured control; fraction masked in steps of 1/12; list of 8 trained **to criterion on the clean cue**; each mode has a **model-free reference** — the largest fraction at which the degraded cue still names its item on its own — so recall above it is completion and below it is not the model's failure; **guard** `mask_at_ceiling` demotes the recall metrics when in-bound recall saturates (the margins still range) |

3.1 and 3.3 are the two ways a cue can be incomplete and are kept apart on
purpose: noise corrupts every feature a little, masking removes some features and
leaves the rest exact. The margin (target minus best competitor) is reported
beside recall throughout, because it keeps resolving after recall has saturated
at 1.0 — which for the linear associators it does at the default `seq_len=8`.

## 4. Sequence disambiguation

*When two episodes share a stretch of observations, does the right one come back?*

| # | Question | What we vary | w | Section(s) |
|---|---|---|---:|---|
| 4.1 | Given a stretch that several episodes share, is the correct continuation chosen at the fork? | how many episodes share it; how long the shared stretch is (axis D, 2026-09-06: length 2–10 with the suffix fixed, under persistent and onset-only support) | 0.60 | `tmaze_disambiguation`, `symbolic_disambiguation` |
| 4.2 | How alike can the distinguishing signals be before they stop distinguishing? | similarity between episodes' discriminating signals; within-category overlap of the items themselves | 0.40 | `symbolic_disambiguation`, `semantic_similarity` |
| 4.3 | ⚑ Can the memory hold the distinguishing signal across a stretch where it is no longer present? | how many shared steps separate the signal's withdrawal from the fork | (in 4.1) | both disambiguation sections |
| 4.4 | When it chooses wrongly, does it fall into a neighbouring episode or a random one? | number of episodes | (in 4.1) | `symbolic_disambiguation` (confusion matrix) |

4.3 is the hard end of the axis and, for every arm that does not declare
`StatePrimeable`, a protocol finding rather than a ranking: a one-step probe cannot express holding
a signal, so these rows floor at chance for a reason that is not the model's.


### Benchmarks

| section | measures | plot | scalars | conditions |
|---|---|---|---|---|
| `tmaze_disambiguation` | with two routes sharing a stem, does an odour cue select the right arm at the fork — and does that survive the odour being withdrawn before the fork? | `tmaze_disambiguation.png` (fully-cued corner), `tmaze_disambiguation_graded.png` (branch accuracy and divergence margin against withdrawal delay, with the concurrent reference) | `tmaze_disamb_full_branch_acc`, `tmaze_disamb_full_confusion`, `tmaze_disamb_graded_<cond>_branch_accuracy / _divergence_accuracy / _divergence_margin` for concurrent and delays; control `tmaze_disamb_mec_only_*` (place only, no odour) | two 15-step routes, 20 odour dims weighted ×2 with modalities norm-balanced; odour zone starts at stem entry and is withdrawn at zone fractions 1.0 (concurrent, the reference), 0.75, 0.5, 0.25, 0.125, so **delay = shared_end − zone_end**; trained **to criterion on one-step prediction along the route**; scored on a rollout with the odour **clamped** while available (an odour is a cue available in advance, unlike a reward) |
| `symbolic_disambiguation` | the same fork, modality-free and swept along eight axes | `symbolic_disambiguation.png` — discriminability, load (with the orthogonal control), withdrawal, support vs delay decoupled, shared middle, and shared-stretch length under two support regimes | 4.1: `symdis_max_episodes_above_chance`, `symdis_load_disambiguation_cost` (vs an orthogonal-discriminator control at the same load); 4.2: `symdis_similarity_tolerance_threshold`; 4.1 (length): `symdis_max_shared_len_persistent` (scored), `symdis_max_shared_len_onset` (protocol-limited); 4.3 ⚑: `symdis_max_delay_at_fixed_support`, `symdis_support_needed`, `symdis_middle_endogenous_accuracy`; 4.4: `symdis_context_graded_confusion_index` | episodes = shared stem + discriminator + branch, items drawn near-orthogonally; axes: discriminator similarity (5 `category_variance` rungs), load (2, 3, 4, 6, 8 episodes), withdrawal zone (4 fractions), hold delay (0, 1, 2, 4, 6) and support duration (1, 2, 4, 6) **decoupled**, an orthogonal-control load sweep, and a middle condition (informative vs endogenous); **criterion on one episode's one-step prediction** (training the criterion on all N would fold the ambiguity into it); probe is **one-step cued at the fork**, primed with the true prefix through `observe` for arms declaring `StatePrimeable` — for the rest, the delay rows are protocol-limited and excluded |
| `semantic_similarity` | how alike the items themselves can be before the same rule stops separating them, and what that costs in exposure | `semantic_similarity.png` — MRR and epochs-to-criterion against within-category cosine | `similarity_exposure_cost` (epochs at the highest overlap ÷ at the lowest), `mrr_high_similarity`, `mrr_low_similarity`, `similarity_effect_mrr_drop` | one 7-item fruit list re-encoded at `category_variance` ∈ {0.05, 0.1, 0.2, 0.5, 1.0} (mean pairwise cosine ≈ 0.79 → 0.0); **each rung trained to its own criterion**, so a low MRR at high overlap is a discriminability result and not the same budget being worth less; probe **cued**, seeded per rung; a high-vs-low categorical comparison is also run |

The symbolic and spatial accuracies are not one number: the symbolic probe is
single-step, the spatial one rolls out and is confounded with drift. Compare
within a modality only. `similarity_exposure_cost` was the suite's one
non-reproducible metric until 2026-09-04 (a `set()`-built vocabulary plus
unseeded probe noise); it is now deterministic across `PYTHONHASHSEED`.

## 5. Serial order

*Is the order of an episode stored — and its timing?*

| # | Question | What we vary | w | Section(s) |
|---|---|---|---:|---|
| 5.1 | Was the order learned at all, separately from whether the items were? | list length | 0.10 | `sequence_length` (establishment) |
| 5.2 | When recall fails, are items out of order, missing, or intruding — and are order errors local swaps or long-range? | list length | 0.20 | `sequence_length` (order errors) |
| 5.3 | Given only the first item, how far can the memory unroll the sequence on its own — and how much extra exposure does that take compared with cued recall? | list length; how the model's own output is fed back | 0.20 | `sequence_length`, `tmaze_completion` |
| 5.4 | ⏱ Does the memory retain *how long* separated two events — enough that the interval alone can decide what comes next? | the interval between two events, used as a cue | 0.25 | `interval_retention` |
| 5.5 | ⏱ Can it reproduce a sequence with its original tempo and pauses, not just its order? | the interval between events, as something the model must produce | 0.25 | `interval_generation` |

Metric time (5.4, 5.5) carries half the capacity. It is the fullest form of
serial order and, today, no arm in the taxonomy except the DTS-ESN can be asked
about it. 5.4 and 5.5 need *different* streams: a gap that carries information
cannot also be predictable from the items — which is why they are two sections
rather than two read-outs of one, and the reason is checked rather than asserted
(`interval_ambiguous_gap_collapse` runs the timing head on 5.4's ambiguous
stream and confirms it collapses to the mean of the trained gaps).

**Both became wired sections on 2026-09-04** (`memval/benchmarks/interval_timing.py`,
run from the symbolic pipeline). They are the only sections gated on a declared
capability — 5.4 on `TemporallyClocked`, 5.5 on the strictly stronger
`TimingPredictive` — and are skipped, not floored, for an arm holding neither.
Each carries its own control: 5.4 trains an ordinal twin on the identical stream
with the gaps removed, and 5.5 the ambiguous-stream check above. Both are at a
**fixed** exposure rather than a criterion staircase, which is defensible only
while a single arm can run them; a second time-clocked arm makes that a
confound, exactly as it was everywhere else.


### Benchmarks

| section | measures | plot | scalars | conditions |
|---|---|---|---|---|
| `sequence_length` | how the list length an arm can hold and unroll scales, and what unrolling costs over cued recall | `length_curves.png` — MRR against L, and span against L for each feedback mode | `max_memory_span` (raw), `max_memory_span_l2`, `max_memory_span_quantized`, `span_fraction_mean_raw`, `span_raw_over_quantized`; from the probe: `unrolling_gap`, `unrolling_exposure_ratio`, `cascade_recovery_rate`, `rollout_margin_at_break` | L ∈ {3, 5, 7, 9, 11} fruit words; at each L trained **to criterion on cued recall** (so every span is read at the point where the associations demonstrably exist); the three spans are **rollouts under raw / l2 / quantized feedback**, and the gap between raw and quantized is the drift the arm's own output injects |
| `bin/probe_serial_order.py` (feeds `sequence_length`) | 5.1 was order learned separately from items; 5.2 when recall fails, is it a transposition, an omission or an intrusion, and are transpositions local; 5.3 the self-referenced cost of unrolling | (none — a JSON side-car, `serial_order_probe.json`) | 5.1: `list_membership_rate`, `order_given_item`, `establishment_break_length`; 5.2: `order_error_fraction`, `transposition_locality`, `transposition_asymmetry`, `intrusion_rate`; 5.3: `unrolling_gap`, `unrolling_exposure_ratio` | same grid; keeps every **decoded identity** rather than a boolean, so displacement d = (study position of decoded word) − (target position) can be classified; 5.3 is a **staircase on one model, one epoch at a time**, reporting E_cued and E_roll — the smallest exposures at which cued and raw-rollout recall each reach 0.95 — so `unrolling_exposure_ratio = E_roll / E_cued` is comparable across arms with 300× different learning rates; **guard** `failures_observed`: every 5.2 metric is a conditional on a cued failure, so with none observed the dimension is undefined, not zero |
| `tmaze_completion` | how much of an open-loop rollout diverges from the route | (shared with Pattern completion) | `tmaze_pc_divergence_fraction` into `unrolling` | as above |
| `interval_retention` ⏱ | 5.4 — can the elapsed gap alone decide what comes next | `interval_retention.png` — the branch chosen against gap on a log axis, trained gaps and the crossing marked, control accuracy in the title | `interval_discrimination_acc` (chance 0.5), `interval_switch_sharpness`, `interval_encoded`; control `interval_control_acc`; diagnostic `interval_crossing_gap` | one 3-item prefix followed by two 3-item continuations that **only the gap distinguishes** (8 s vs 1 s, within-run spacing 1 s), streams interleaved; probe = `predict_next(prefix, elapsed=gap)` with the prefix's true intervals as `prompt_intervals`; sweep over 11 log-spaced gaps 0.25–20 s; **control** = the same arm on the identical event stream with the intervals removed; **fixed** exposure (`n_reps=40`, a stated condition — see below); gated on `TemporallyClocked` |
| `interval_generation` ⏱ | 5.5 — does autonomous rollout reproduce the tempo and put the pause back in the right place | `interval_generation.png` — generated vs trained gap per step (tempo, two sequences; rhythm, one sequence with an internal pause) | `tempo_reproduction_error` (relative, both tempos), `rhythm_pause_position_acc`, `peak_time_error` (time-to-next, teacher-forced); diagnostic `weber_slope`; control `interval_ambiguous_gap_collapse` | two 4-item sequences at tempos 0.5 s and 4.0 s; one 6-item sequence with a 3.0 s pause among 0.5 s gaps; rollout via `generate()` with codebook cleanup on the fed-back item (the quantized protocol, stated because it is a choice); **control** = the timing head run on 5.4's ambiguous stream, which should collapse to the mean of the two trained gaps; **fixed** exposure (`n_reps=40`); gated on `TimingPredictive` |

The two ⏱ sections are the only ones at a **fixed** exposure rather than a
criterion staircase. With a single eligible arm there is no cross-arm
learning-curve confound to create; the moment a second time-clocked arm exists
this must change, for exactly the reason every other section already did.

---

## Reading the tables against a model

A model's profile is five scores and five coverages. The score is a weighted
mean over the rows it could answer; the coverage is the share of the capacity's
weight those rows carry. A high score at low coverage is a claim about a
fraction of the capacity, and the radar plots both so the two are never
confused.

Three things are excluded from every score rather than averaged in: rows the
arm lacks the capability for (tags above), rows whose section reports its
manipulation had no dynamic range for this arm (a guard failed — e.g. no order
errors to classify, or in-bound recall saturated), and rows measured at a
censored exposure (the criterion was not reached inside the budget).
