# Sequence disambiguation — design note

**Status:** audit complete (§1–§2, measured 2026-08-24); axes specified (§3–§4).
**Build-order steps 1–2 landed 2026-09-01**: axis A is wired as the graded
odour-availability sweep in `spatial_pipeline.py`'s `tmaze_disambiguation`
section, and metrics §5.1 (divergence accuracy, recovery profile, shared-stretch
control) and §5.2 (normalised branch margin) are implemented in
`SpatialDisambiguationBenchmark`. Still open from §5: the confusion matrix and
context-graded confusion index (§5.3, needs N > 2) and the linear-oracle gap
(§5.5).

**Step 5 resolved 2026-09-03 — and it was not the protocol.** The
state-carrying ingestion path (§6 step 5) was already implemented: the
benchmark calls `model.observe_sequence` over the shared zone whenever
`supports_priming(model)` holds. What was missing was an *arm*. All nine entries
in `bin/run_benchmark.py`'s `MODEL_REGISTRY` returned
`supports_priming == False`, and the two classes that implement `observe`
(`MultilayerTemporalPCNetwork`, `DTSESNSequenceNetwork`) were absent from that
registry despite `models/baselines/__init__.py` recording them as "a crucial
part of the taxonomy ... compared normally". So the priming branch had never
executed and every shipped withdrawn row was `state_primed=False`.

`temporal_pc` is now registered. First primed run (n_trials=5, exposure to
criterion 35 epochs, `state_primed=True` on every row):

| availability | delay | divergence accuracy | divergence margin |
|---|---|---|---|
| concurrent | 0 | 1.000 | +0.0063 |
| withdrawn | 0 | 0.500 | −0.0013 |
| withdrawn | 2 | 0.500 | −0.0008 |
| withdrawn | 4 | 0.500 | −0.0001 |
| withdrawn | 6 | 0.500 | +0.0001 |
| withdrawn | 7 | 0.500 | +0.0000 |

**The floor survives priming.** tPC receives the stem through a non-learning
state update and still cannot choose the branch. Two consequences:

1. The withdrawn rows are now a *model* result for this arm, so §7's "a
   `delay > 0` result under the memoryless probe is a protocol finding" trap no
   longer applies to `temporal_pc` — but it still applies to the other nine.
   Read `state_primed` per row; never quote a withdrawn row without it.
2. The interesting boundary is not the length of the delay. `withdrawn, delay=0`
   is already at chance, and that row has the odour available through the entire
   shared corridor — it is withheld only on the arms. So performance turns on
   whether the discriminator is present *at the decision step*, not on how long
   it has been absent. Axis A as currently sampled is nearly binary; the graded
   rungs between delay 2 and 7 add little. Consider resampling the axis around
   the delay 0/1 boundary instead.

The splitter index (§5.4) is now demonstrated in
`examples/splitter_index_demo.py` and explains the floor: tPC carries ~0.05–0.06
mean state separation across the cue-free stretch but only ~0.001 by the
divergence step. It carries a trace and loses it before the decision, which is a
different failure from the memoryless arms' identically-zero separation and is
invisible to any behavioural readout. Still needs wiring into the benchmark as a
scored metric.

First result: with the odour withdrawn, every arm falls to chance at
`delay >= 2` while the concurrent reference stays at 1.0 — and the normalised
margin decays monotonically across the delays where accuracy is already flat at
chance, which is exactly the resolution §5.2 was added to recover. Report it as
a protocol finding per §3.

Historical note (the original status line): **The central finding is that most of the machinery already
exists and is unwired**: `BifurcatingRouteGenerator`, `OverlapConfig` and
`SpatialDisambiguationBenchmark.sweep()` implement all three axes below and are covered by
`tests/test_spatial_disambiguation.py`, but no pipeline calls them. `spatial_pipeline.py`
ships `TMazeDisambiguationGenerator` instead, which hardcodes the degenerate corner of the
space.

**Capacity:** Sequence disambiguation — the memory returns the wrong episode although the cue
is intact, because two episodes share a stretch of observations. Division from the neighbours:
pattern completion is getting *into* the right trajectory from a degraded cue; serial order is
staying *on* it; disambiguation is choosing *which* trajectory when the observations do not
say. The cue is never degraded here — that is what separates this capacity from completion.

**Motivating observation:** neither section currently filed under this capacity tests it. §1.

**Literature anchors.** Marked ✓ if already in the Zotero library.

- **Agster, Fortin & Eichenbaum (2002)**, J Neurosci 22(13):5760–5768 — "The hippocampus and
  disambiguation of overlapping sequences." Rats learn two odour series that overlap in the
  middle; hippocampal disruption impairs the *overlapping* items selectively. This is the
  paradigm of this note, in the symbolic modality, with the shared-stretch manipulation built
  in. It fills the "no primary source for sequence disambiguation" gap recorded in the
  citation map — better than Pastalkova alone, because the manipulation is behavioural and
  parametric. **Not in the library; acquire and verify.**
- **Levy (1996)**, Hippocampus 6:579–590 — "A sequence predicting CA3 is a flexible associator
  that learns and uses context to solve hippocampal-like tasks." Origin of sequence
  disambiguation as a *computational* task, with the shared-middle-segment construction and
  context as the disambiguator. **Not in the library.**
- **Wood, Dudchenko, Robitsek & Eichenbaum (2000)** and **Frank, Brown & Wilson (2000)**, both
  Neuron — splitter cells: hippocampal firing in the shared stretch differs by journey. These
  ground the *representational* readout of §5.4, which is the metric our existing citation
  actually supports. **Neither in the library.**
- **Pastalkova, Itskov, Amarasingham & Buzsáki (2008)**, Science 321:1322 — the primary source
  behind the wheel-running result the results intro currently cites through the review.
  **Not in the library.**
- **Howard & Kahana (2002)**, J Math Psychol 46:269–299 — TCM; the drifting-context formalism
  §4 builds on. Already an anchor of `docs/proposals/free_recall_design.md`, which is the point: one
  mechanism, two sections. **Not in the library.**
- **Polyn, Norman & Kahana (2009)**, Psych Review 116:129–156 — CMR; context as a source/task
  backdrop that supports *list* discrimination rather than item discrimination.
  **Not in the library.**
- ✓ **Güler et al. (2025)** — contextual stability as the structuring variable for episodic
  memory. Supports treating context as an ambient backdrop rather than a cue.
- ✓ **Wang & Egner (2022)** — task-set switches create event boundaries. Grounds a discrete
  context *shift* as an alternative to smooth drift (§4.3).
- ✓ **Bevan & Feuer** — the role of context in episodic memory.
- ✓ **Bellmund et al. (2020)** — sequence memory in the hippocampal–entorhinal region.
- ✓ **Kumaran et al. (2016)** — pattern separation. Keep, but note it is stated over *static*
  patterns; it does not carry the sequence case on its own, which is why the citation map
  records this capacity as "assembled".

---

## 1. The audit: what the two shipped sections measure

| Section | Filed under | What it actually measures | Evidence |
|---|---|---|---|
| `tmaze_disambiguation` (spatial §3) | disambiguation | concurrent cue-conditioned association | odour written to every step; `prefix_end: 0` |
| `semantic_similarity` (symbolic §5) | disambiguation | interference under representational overlap | every item distinct → transition set is a bijection |

**`tmaze_disambiguation`.** `TMazeDisambiguationGenerator.generate` writes the odour from step
0 to `odour_end` and returns `zone_start: 0, zone_end: stem_end, prefix_end: 0`
([tmaze_disambiguation.py:102](../../memval/generators/tmaze_disambiguation.py):102). The shipped
pipeline then passes `odour_on_arms=True`, which sets `odour_end = sequence_length`, so the
odour is present on *every step of the whole trajectory*, arms included
([spatial_pipeline.py:380](../../memval/benchmarks/spatial_pipeline.py):380). The branch decision
is therefore a pure function of the final stem input, which already contains the odour. The
stem prefix is inert.

The falsifiable check: cue with only the final stem state and confirm branch accuracy is
unchanged. Run it before the section is described as disambiguation in any text.

The same pipeline call passes `odour_scale=2.0` with the comment "so the cue is decisive for a
linear map" — the configuration is tuned *toward* linear solvability. For a converged linear
map the scale is a no-op (W absorbs it); it changes only effective learning rate and
saturation. It should not be read as a task property.

**And the metric hides what conjunctive structure remains.** From the linear-oracle audit: the
task is certified non-linear (rank 16 vs 23), but branch accuracy is nearest-of-two on the
decoded suffix, so a thin correctly-signed lateral bias passes. The optimal *linear* map scores
branch accuracy 1.0 at margin +0.023; the conjunctive map's margin is +0.155 (7×). Committed
results show the inversion — `AsymmetricHopfieldNetwork` (linear, ReLU readout) 1.0,
`OriginalEqPropSequenceNetwork` (hidden layer) 0.5, i.e. `linear_oracle_gap = −0.5`.

**`semantic_similarity`.** Every item in every list is distinct, so no cue ever demands two
successors and the task is exactly linearly solvable. [symbolic_pipeline.py:19](../../memval/benchmarks/symbolic_pipeline.py):19
already labels it "semantic interference", which is the correct description. **Action: move it
out of this capacity** into continual retention / pattern completion as an interference
stressor. It becomes relevant here only as the graded-similarity axis (§3, axis B) once there
is an actual ambiguity to grade.

Net: the capacity is empty, not thin.

## 2. Why it is empty: the probe forces the degenerate corner

Under a memoryless one-step `predict_next`, the disambiguating signal is either present in the
input at the decision step — in which case the task needs no memory, only a conjunction — or it
is absent, in which case every arm is at chance for a protocol reason rather than a model
reason. There is no intermediate case the current harness can express.

That is the whole explanation for §1. The shipped section is not an oversight; it is the only
point in the space the harness can reach. It follows that **adding more disambiguation sections
under the current probe cannot help.** The temporal half of this capacity is blocked on one
piece of infrastructure — the state-carrying ingestion path of
`docs/proposals/nonlinearity_benchmark_design.md` §4 (prefix ingested through a state-updating call, not
through `predict_next`) — and that single investment unlocks all of it.

The representational half (§3, axes B and C) is *not* blocked and is buildable now.

## 3. The three axes

The capacity is a 3-D space. The shipped section sits at its trivial corner:
(concurrent, orthogonal, minimal, N=2).

| Axis | Varies | Range | Existing knob | Blocked? |
|---|---|---|---|---|
| **A — availability** | *when* the discriminator is present | concurrent → onset-only, then withdrawn → never present (endogenous) | `zone_fraction`, `zone_offset` | partly — see below |
| **B — discriminability** | *how distinct* the two discriminators are | orthogonal → graded → identical | `encounter_similarity` | no |
| **C — extent** | how much is shared, and by how many episodes | shared length; shared fraction; N confusable episodes | `shared_fraction`, `shared_position`; N not implemented | no |

**All three knobs already exist** in `OverlapConfig`
([overlap_config.py](../../memval/benchmarks/overlap_config.py)) and are consumed by
`BifurcatingRouteGenerator.generate`
([bifurcating_route.py:19](../../memval/generators/bifurcating_route.py):19). `zone_fraction` /
`zone_offset` place the encounter (odour) zone as a *sub-interval* of the shared corridor,
which is exactly axis A: the quantity that matters is

```
delay = shared_end − zone_end
```

the number of shared steps over which no discriminating cue is available. The default config
(`total_length=30, shared_fraction=0.40, shared_position=0.33, zone_fraction=0.50,
zone_offset=0.25`) resolves to `shared=[6,18)`, `zone=[8,14)`, i.e. **a 4-step cue-free delay**.
The unwired generator's default is already non-degenerate; the shipped one is at zero.

`SpatialDisambiguationBenchmark.sweep()` takes a list of `OverlapConfig` and already records
`zone_fraction`, `zone_offset` and `encounter_similarity` per row
([spatial_disambiguation.py:236](../../memval/benchmarks/spatial_disambiguation.py):236). It is
exercised by `tests/test_spatial_disambiguation.py:116` and called from no pipeline.

**Axis A is only partly blocked.** `delay > 0` is expressible by the generator today, but under
the memoryless probe every arm is predicted to fall to chance the moment `delay > 0`, because
none carries state. That prediction is itself the reportable dissociation — it is the
behavioural face of the no-persistent-state limitation, and it is worth shipping as a result
*before* the protocol exists, provided it is stated as a protocol finding and not as a model
ranking. Reading it as "these models cannot disambiguate" would be wrong.

**Axis C, the N dimension, is the cheapest real gain.** The load finding on the symbolic chain
was that load rather than overlap separates arms (tPC/AHN at 0.000 forgetting with 4 lists,
separating only past ~12). The same logic applies here: two episodes sharing one stretch is far
below any associator's capacity. N episodes sharing one stretch is the disambiguation analogue
of the load sweep, needs no new protocol, and is not implemented in either modality.

### 3.4 Guidance withdrawal (added 2026-09-08)

The withdrawn-odour sweep above moves two things at once: the training
material changes with the rung (the odour stops at the zone end during
training too) and the cue-free stretch is free-run by the model, so cue decay
and place drift are mixed. Since 2026-09-08 that rollout is the arm's own
`recall` from the zone-end cue with nothing clamped (the withdrawn material
has zero odour after the zone, so a trained arm predicts ~0 there itself).
The harness-driven `predict_next` loop it replaced could not advance a
stateful arm: `predict_next` is pure by contract, so DTS-ESN's reservoir sat
frozen at the zone end while the fed-back place drifted, and its withdrawn
margin fell from +0.53 at delay 0 to +0.01 at delay 6. Under its own hybrid
`recall` the same weights read +0.53, +0.65, +0.90, +1.00 at delays 0, 2, 4,
6: the reservoir carries the zone's odour through its slow units once it is
allowed to move. AHN is bit-identical under both drivers (both are raw
feedback for a memoryless arm). A second sweep now sits beside it in the same
figure (bottom row of `tmaze_disambiguation_graded.png`) with a single moving
part:

* **Material is fixed**: the concurrent pair (odour through stem and arms on
  both routes), blocked fit, identical on every rung.
* **The probe moves**: the harness walks the arm up the stem with
  `observe_sequence` as far as cue point `k`, then hands the rollout to the
  arm's own `recall` from `[place_k, odour]`. Nothing is clamped after that.
  Recall starts `d = shared_end - k` steps before the fork, so the x-axis
  reads "guided this far, does it recover the correct arm".
* **`observe` is delivered to every arm.** `HippocampalModel.observe` is a
  base-class no-op; arms that carry state (`StatePrimeable`) override it.
  There is no capability check at the probe, and `state_primed` is
  provenance only. `tests/test_observe_default.py` pins both halves.
* **Two read-outs, one panel each**: the step-matched margins (fork step and
  arm mean), and a twin-axis panel with *arm accuracy* (nearest arm at any
  position, ties fail: credit for the choice even when the position has
  drifted) against *positional error* (mean decoded distance to the
  step-matched target along the arm). Nearest-of-two accuracy is not
  plotted here; §5.2 explains why.
* **Five model seeds**, trained once each and probed at every rung; the band
  is one s.d. over seeds. Arms without a `seed` kwarg run once.

Read it with one caveat. Recall is unclamped, so an arm whose output
includes the odour block re-predicts the odour and feeds it back to itself.
A memoryless observation-rollout arm can therefore carry the cue around the
loop without any internal state: AHN holds arm accuracy 1.0 at every `d`
with the margin falling from 0.75 to 0.54, and DTS-ESN sits near 0.95 with a
positional error of 0.04. The sweep measures whether the arm can keep the
cue *in its loop* once guidance stops, not whether it keeps it in a latent
state. The odour-withdrawn sweep in the top row remains the latent-state
question: there the material itself has no odour after the zone, so an arm
that re-predicts its output block re-predicts zero.

## 4. Context as the discriminator

Context is not a fourth axis. It is a choice of *what occupies the discriminator slot*, and it
is a better choice than an odour.

### 4.1 Three senses, and which one MemVal lacks

| Sense | Definition | Axis-A position | In the suite? |
|---|---|---|---|
| Discrete tag | an orthogonal externally-supplied cue (odour) | concurrent | yes — the only one |
| Maintained cue | the same tag, presented then withdrawn | onset-only | generator yes, protocol no |
| Drifting context | a slowly-varying ambient state bound to items at encoding and reinstated at retrieval | spans the axis | **absent** |
| Endogenous | no external discriminator; episodes differ only in what preceded the shared stretch | never present | absent (aliased chain, §5.4 of the nonlinearity note) |

The third is what "backdrop of context" means in the retrieved-context literature, and it is
the one with no representation anywhere in the suite.

### 4.2 Why drifting context is the right substrate

Three payoffs, and the third is the argument for building it:

1. **Axis B gets a principled parameterisation.** With a drifting context, the similarity
   between two episodes' contexts is a function of their separation in the stream. Drift rate
   *is* the discriminability knob; no hand-tuned cosines, and the axis acquires a generative
   model instead of a scale factor.
2. **It is graded by construction**, so the confusability structure is continuous rather than
   the binary same/different an odour gives.
3. **The same generator serves three capacities.** A drifting context vector produces (a) list
   discrimination here, (b) the contiguity gradient and lag-CRP that the serial-order binding
   readout needs (`docs/proposals/free_recall_design.md`, whose TCM anchor is the same Howard & Kahana
   paper), and (c) an interaction with ISI, since a context clocked by time rather than by item
   count makes spacing change contextual similarity. That converts `isi_tolerance` and this
   capacity from two unrelated sweeps into one prediction.

### 4.3 Discrete shift vs smooth drift

Wang & Egner (2022, in the library) supports a discrete alternative: a context *shift* at an
event boundary rather than continuous drift. Both are worth having, and they are one knob —
drift rate plus an optional boundary impulse. Smooth drift is the harder and more informative
case because it makes confusability graded; the boundary version is the natural control, since
it should be strictly easier and any arm failing *it* has failed for a different reason.

### 4.4 Delivery rule

Deliver context as **extra input dimensions**, in the manner of the existing LEC block — *not*
through the `context_data` channel. Only `cls_dg_eqprop`, `asymmetric_hopfield`, `dg_eqprop`
and `ewc_dg_eqprop` implement `n_context`, so routing context through that channel would
silently exclude tPC, GPT-2 and `OriginalEqProp` from a capacity in which they are the
interesting comparisons. This is §5.1 of the nonlinearity note; it binds with more force here
because context would be load-bearing for a whole capacity rather than for one section.

This does **not** contradict the "context stays implicit" decision recorded in the results
intro. That decision is about not handing the model the disambiguating signal *at the decision
point*. Axis A is precisely the control that enforces it: context may be delivered as input, so
long as `delay > 0` withholds it where the decision is made.

## 5. Metrics

Four changes. The first is a correctness fix, not a refinement.

### 5.1 Score at the divergence point, not over the episode

During the shared stretch both episodes predict the same successor, so an ambiguous
representation is scored **correct** there. Averaging over an episode therefore gives away a
fraction of the score for free, and — this is the problem — that fraction *grows with
shared-stretch length*. The metric gets easier exactly as axis C makes the task harder, which
means the axis-C sweep is currently uninterpretable in the symbolic modality.

Report, separately:

- **divergence accuracy** — the step at which the episodes diverge, scored alone;
- **recovery profile** — accuracy at divergence+1, +2, … , which separates "chose the wrong
  branch" from "chose right and then fell off";
- **shared-stretch accuracy** — as a *control* that should be near ceiling for every arm, never
  folded into the headline number.

The spatial section is partly protected because `branch_accuracy` is already computed at the
branch, but `suffix_mse` is not, and neither is any symbolic MRR.

### 5.2 Margin, not nearest-of-two

Report normalised branch **margin** — (distance to wrong arm − distance to correct arm) /
arm separation — alongside accuracy, so a thin linear bias no longer scores identically to a
conjunctive solution. This is the §2 failure and it is already specified in the nonlinearity
note; it belongs to this capacity.

### 5.3 Confusion structure, not confusion rate

With N > 2 confusable episodes (axis C), *which* wrong episode the model falls into is
informative: falling into the contextually nearest neighbour is a different failure from
falling into a random one, and only the first is evidence about context. Replace the scalar
`confusion_rate` with a confusion matrix over episodes, and report a
**context-graded confusion index** — the correlation between confusion probability and
context similarity.

### 5.4 Representational separation (the splitter index)

Behavioural branch accuracy can be achieved by a system that separates only at the last moment.
The splitter-cell result, and the wheel-running result the results intro already cites, is that
the trajectory is distinct *throughout* the shared stretch. So measure the divergence of the
model's internal state across the shared stretch, not only its output at the branch:

```
splitter_index(t) = 1 − cos( h_A(t), h_B(t) )    for t in [shared_start, shared_end)
```

reported as a profile over `t` and summarised by its mean over the cue-free portion
(`t ≥ zone_end`). This is a different claim from behaviour, it is the claim the citation
actually makes, and it degrades gracefully: an arm at chance behaviourally can still show
partial separation, which is a more informative result than a floor. Requires a hidden-state
accessor, which not every arm exposes — gate it, and report which arms are included.

### 5.5 Linear-oracle gap

Mandatory, per §3 of the nonlinearity note, in both the `linear` and `conjunctive` variants.
The `tmaze_disambiguation` inversion is the standing proof that it cannot be optional here.
Use SVD-based `lstsq`, and never quote an absolute residual in the spatial suite — place-cell
codes on a smooth trajectory are rank-deficient for a single route.

## 6. Build order

Ordered by cost. Steps 1–3 need no new model interface and no protocol work.

1. **Wire the existing sweep.** Replace the `tmaze_disambiguation` pipeline call with
   `BifurcatingRouteGenerator` + `SpatialDisambiguationBenchmark.sweep()` over an
   `OverlapConfig` grid on axes A, B, C. Keep the current configuration as the grid's
   `delay = 0, similarity = 0` corner so the shipped number stays comparable. This is
   plumbing — the generator, the benchmark, the sweep and the tests already exist.
2. **Fix the metrics** (§5.1–§5.3, §5.5) in `SpatialDisambiguationBenchmark`. Do this before
   quoting any sweep row, or the axis-C rows will be read against a metric that eases as the
   axis hardens.
3. ~~**Symbolic overlapping-sequence section.**~~ **LANDED 2026-09-03** as the
   `symbolic_disambiguation` section. Built on a modality-agnostic core
   (`generators/overlap.py`) that both modalities now share: it assembles N episodes
   sharing a stretch, writes the discriminator into the zone, balances the two blocks, and
   **refuses to build an episode set whose shared stretch is not actually shared** -- that
   silent failure would leave episode identity readable from the content channel and the
   benchmark measuring nothing.

   Axis B uses `category_variance` rather than the drift rate this note originally proposed.
   Drift is the better long-run substrate but adds a temporal model on top; category variance
   is already implemented and tested, and it is what makes N > 2 possible, since a family of
   discriminators drawn from one category can sit at a controlled mutual similarity while a
   2-D rotation cannot. Rows are labelled by the **realised** mean pairwise cosine, not by the
   dial: the variance-to-cosine mapping is nonlinear and saturating.

   First result (`AsymmetricHopfieldNetwork`, criterion reached at 3 epochs, chance 0.25):

   | realised cosine | 0.95 | 0.83 | 0.57 | 0.22 | 0.10 |
   |---|---|---|---|---|---|
   | divergence accuracy | 0.25 | 0.25 | 0.25 | 0.75 | 0.75 |

   a genuine graded discriminability curve, breaking at cosine 0.567. That is the measurement
   `semantic_similarity` was meant to provide and structurally could not: with bijective
   transitions no cue ever demands two successors, so both its conditions sit at ceiling and
   the effect reads 0.000. Load (axis C) degrades from 1.000 at N=2/3 to 0.500 at N=6.
   Withdrawal (axis A) drops to chance at any delay > 0, matching the spatial section's
   finding that the axis is nearly binary.

   **Extended 2026-09-03** with three additions, all of which changed what the section can
   claim:

   *Support and delay decoupled.* Sweeping `zone_fraction` alone moves both at once, because
   the zone starts at the corridor entrance: a shorter cue is also an earlier withdrawal. Every
   "how much cue does it need" number read off such a sweep is confounded with "how long can it
   hold one". `generators/overlap.zone_params_for` solves the pair apart and refuses an
   unrealisable request rather than silently clamping. The result is a dissociation the
   confounded sweep could not show:

   | constant 2-step cue, growing gap | delay 0 | 1 | 2 | 4 | 6 |
   |---|---|---|---|---|---|
   | AHN | 0.750 | 0.250 | 0.250 | 0.250 | 0.250 |
   | tPC (primed) | 0.750 | 0.500 | 0.250 | 0.250 | 0.250 |

   tPC holds the discriminator for exactly one step; AHN loses it immediately. Meanwhile
   growing the cue from 1 to 6 steps at a constant 2-step gap leaves both at chance throughout,
   so `symdis_support_needed` is NaN for both: **more support does not help once there is any
   gap at all.** The cliff is entirely in the delay.

   *Orthogonal-discriminator control.* The load sweep repeated with exactly orthonormal
   discriminators, which removes discriminability as a limiting factor. For AHN:

   | N | 2 | 3 | 4 | 6 | 8 |
   |---|---|---|---|---|---|
   | category | 1.000 | 1.000 | 0.750 | 0.500 | 0.625 |
   | orthogonal | 1.000 | 1.000 | 1.000 | 0.833 | 0.875 |
   | gap | 0.000 | 0.000 | +0.250 | +0.333 | +0.250 |

   The control degrades too, so roughly half the fall from 1.000 to 0.500 is **ordinary capacity
   running out, not disambiguation failure**. Without the control the whole drop would have been
   attributed to the capacity under test. The gap is the attribution, and it is reported as
   `symdis_load_disambiguation_cost`. Caveat: the metric is only meaningful when at least one
   side is off the floor. For tPC both sides sit near chance and the gap reads as noise,
   including a spurious −0.333 at N=3.

   *Shared middle.* Every other sweep pins `shared_position=0`, so the sequences share a
   PREFIX and the discriminator is the only thing that identifies one. With a shared middle each
   sequence gets a unique prefix first, which makes identity recoverable without modality B at
   all. Two rungs, both with the cue spanning the whole middle so that axis A is not
   re-entangled: `informative` (discriminators differ per sequence) and `endogenous` (every
   sequence given the SAME discriminator, so only the 3-step unique prefix distinguishes them).
   The rung is named `informative` and NOT `cued`, because in this suite "cued recall" names a
   PROBE; both rungs use the identical probe and differ only in whether modality B carries
   information. AHN scores 0.750 informative and chance endogenous; tPC scores 0.500 and chance. The endogenous rung
   is the hardest in the capacity and is meaningful only for a StatePrimeable arm -- for anything
   else the probe input is identical across sequences and the floor is a protocol artefact.

   **Probe protocol, and it is NOT the spatial one.** The symbolic section takes a SINGLE
   `predict_next` at the divergence step after priming, and scores it against the N candidate
   successors. Nothing is fed back. `spatial_disambiguation.py` instead rolls forward to the end
   of the route, feeding each prediction back, so its `divergence_accuracy` is read at the end of
   a chain of self-generated inputs and is confounded with ordinary rollout drift -- which is
   what its `shared_stretch_error` control exists to isolate. **Do not compare divergence
   accuracy across the two modalities as if it were one number.** The single-step form is what
   sec 5.1 actually asks for: score at the divergence point, and keep the recovery profile
   separate rather than folded in. Each row now carries `probe: "single_step"` so this cannot be
   lost. Making the spatial section match is the obvious follow-up and has not been done.

   **Exposure.** Criterion-referenced within the section, not borrowed from
   `presentation_duration`: a geometric staircase with downward refinement on one-step next-item
   accuracy over ONE episode, criterion 0.95, cap 512. AHN settles at 3 epochs, tPC at 37, both
   reaching criterion.

   **Trap found while building it.** The exposure criterion must be settled on ONE episode.
   Settling it on all N folds the ambiguity into the criterion -- the divergence step then has
   N legal successors, so the score is capped near (L-1)/L, the staircase never converges, it
   runs to `max_epochs`, and the scored sweep is handed a one-shot associator saturated by 512
   passes. That misreported the arm as at chance on every axis.
4. **Splitter index** (§5.4) — needs a hidden-state accessor on the model interface.
5. **State-carrying ingestion protocol** (nonlinearity note §4). Unblocks `delay > 0` as a
   *model* result rather than a protocol artefact, and with it the aliased-chain / endogenous
   rung.

Steps 1–2 are the ones that change what the paper can claim; they cost roughly a day and
retire the "assembled" grounding in the citation map.

## 7. Traps

- **Do not describe the shipped section as disambiguation** until the final-stem-state check of
  §1 has been run and failed. It currently measures concurrent cue-binding.
- **Do not fold `semantic_similarity` into this capacity.** Distinct items, bijective
  transitions, no ambiguity. It is an interference section.
- **A `delay > 0` result under the memoryless probe is a protocol finding.** Every arm at
  chance means the harness cannot express the task, not that the models cannot disambiguate.
  Report it as the behavioural face of the no-persistent-state limitation.
- **Episode-averaged scoring understates difficulty as the shared stretch grows.** §5.1. This
  invalidates naive axis-C sweeps.
- **`odour_scale=2.0` is a configuration choice, not a task property.** It tunes toward linear
  solvability and is a no-op for a converged linear map.
- **Never route context through `context_data`.** §4.4 — it silently drops three arms.
- **Certification gates the task; the oracle gap gates the metric.** `tmaze_disambiguation`
  passes rank certification and still fails to discriminate. Both checks are needed.

**The affine output remap (fixed 2026-09-08).** `_run_trials` used to apply
`0.5p + 0.5` whenever any output was negative, then decode and feed the
remapped vector back. Right only for a tanh-coded bump with a -1 background;
no arm on the roster codes its output that way. A -0.03 wiggle put a 0.5
pedestal on 400 unit-norm place cells, dragged the centre-of-mass decode to
the grid centre and read a 0.005 margin for arms separating the routes at
0.8. Suffix MSE of 0.245 (= 0.5^2) is the signature. Now: clip at zero for
decoding and MSE, raw prediction fed back. Never infer an output range from
`min() < 0`; an arm with a tanh-coded output must declare it.

## 8. Reproduction

The §1 numbers come from the linear-oracle audit of 2026-08-20 (rank identity, `lstsq` residual
split by stem/branch/arm, conjunctive-feature control) and from reading the shipped call sites:
[tmaze_disambiguation.py:102–135](../../memval/generators/tmaze_disambiguation.py):102,
[spatial_pipeline.py:368–400](../../memval/benchmarks/spatial_pipeline.py):368,
[symbolic_pipeline.py:19](../../memval/benchmarks/symbolic_pipeline.py):19. The §3 claim that the
axes exist unwired was checked by grep: `BifurcatingRouteGenerator` and `.sweep(` are
referenced only from `memval/benchmarks/spatial_disambiguation.py` and
`tests/test_spatial_disambiguation.py`, never from a pipeline. The default-config delay of 4
steps is `OverlapConfig.resolve()` evaluated at the generator's defaults; it should become a
unit test when step 1 lands.
