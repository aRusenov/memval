# Results intro: grounding the five core capacities

Draft prose for the Results section opening, with the reference audit that supports it.

## Reference audit

| Capacity | Grounding | Primary references | Notes |
|---|---|---|---|
| Continual retention | Strong | McClelland et al. 1995; Kumaran et al. 2016; Kudithipudi et al. 2022; Bidaki et al. 2025 | Four independent statements spanning neuroscience → ML. |
| One-shot learning | Strong (systems level) | McClelland et al. 1995; Kumaran et al. 2016 | No cellular-level citation in the set — add Magee 2026 (BTSP, one-trial place fields). |
| Pattern completion | Strong | Kumaran et al. 2016; Kahana 2020; McClelland et al. 1995 | Kahana supplies the operational definition (partial *or noisy* cue) and the load-dependence. |
| Sequence disambiguation | Strong, but assembled | Kumaran et al. 2016 (separation); Buzsáki & Tingley 2018 (sequences) | Separation is stated over static patterns; Buzsáki carries it into the sequence domain. |
| Autonomous generation | Adequate, needs the right refs | Buzsáki & Tingley 2018; Kahana 2020 (retrieved context); Tulving 2002 | Replay (McClelland, Kumaran, Kudithipudi) is a secondary functional argument, not the primary one. |

Per-reference fit:

- **Kahana 2020** — excellent, and load-bearing three times over: the minimal-assumptions framing, the
  benchmark-phenomena precedent (a set of effects any recall model must explain), the pattern-completion
  definition, and retrieved-context theory as the cognitive-level statement of autonomous generation.
- **McClelland et al. 1995** — excellent. Carries continual retention *and* one-shot, and crucially ties
  them to each other as two horns of one dilemma.
- **Kumaran et al. 2016** — excellent. Modern restatement; pattern separation/completion, replay,
  one-shot schema learning.
- **Buzsáki & Tingley 2018** — excellent and irreplaceable: the only reference in the set whose object of
  study is a *sequence*. Carries autonomous generation and sequence-level disambiguation.
- **Kudithipudi et al. 2022** — good as a *precedent for the move* (they likewise identify a set of key
  capabilities), and for catastrophic forgetting / replay. Their six features are not our five; do not
  present them as the same list.
- **Bidaki et al. 2025** — good but narrow. Grounds the online single-pass regime and catastrophic
  forgetting. Also useful negatively: its 83 datasets are image classification / detection /
  vision-language, none episodic-sequence. Do not cite it for completion, disambiguation, or generation.
- **Tulving 2002** — weakest as *functional* grounding; it is a definitional / consciousness paper. Use it
  for what an episode is, and for K.C. as evidence that recollection dissociates from intact semantic
  knowledge. Do not use it for one-shot or retention.

Gaps to fill:

1. One-shot at the cellular level → Magee 2026 (already in the library) or Bittner et al. 2017.
2. Sequence disambiguation, primary source → cite Pastalkova et al. 2008 directly alongside the review.
3. Pattern completion, data-level → Knierim & Neunuebel 2016 (already in the library), if a mechanism
   citation is wanted rather than a review.
4. Nothing in the set grounds *why five and not six*. The text should say the set is a working minimum,
   not a closed one.

## Draft paragraphs

When defining any computational model, we start from experimental observations and try to narrow down a
minimum set of assumptions to account for the observed facts (Kahana, 2020). In the current work, we take
the same approach in defining the core capacities a continually learning system should possess. The move
has precedent on both sides of the literature we draw on. Kahana (2020) organises the modelling of recall
around a set of benchmark phenomena that any successful model is expected to reproduce, so that a model is
judged by which effects it recovers rather than by a single aggregate score. Kudithipudi et al. (2022) make
the same move for artificial systems, identifying a set of key capabilities a lifelong learning machine
would need to realise. We follow both. Each capacity below is stated together with the failure that
signals its absence, and we treat the pattern of failures across capacities as the primary result rather
than as noise around a headline number.

The first two capacities are the two horns of a single dilemma, and the argument for them is the one that
motivated complementary learning systems theory. McClelland et al. (1995) showed that a network plastic
enough to acquire an arbitrary association from one presentation will, in making the weight changes that
acquisition requires, disrupt what it already knows — catastrophic interference — while a network that
learns slowly enough to escape that fate cannot meet the ordinary demand that information be acquired and
retained from a single exposure. Their resolution was architectural, a fast hippocampal store feeding slow
interleaved cortical consolidation, but the constraint is prior to the resolution: any system learning from
an ongoing stream must be scored on retention and on single-exposure acquisition separately, because either
can be bought at the other's expense. Kumaran et al. (2016) restate the same tension in its modern form,
noting that it is precisely the ability to act on the content of one experience — to avoid the watering hole
after a single encounter with a lion — that a purely gradual system cannot provide. The pair recurs in
machine learning as the stability–plasticity dilemma, which Kudithipudi et al. (2022) place among the key
features of lifelong learning and which Bidaki et al. (2025) identify as the organising challenge of online
continual learning, where each sample is seen once in a single pass. Notably, forgetting there is
characterised not as a capacity limit but as the overwriting of the substrate that holds earlier knowledge,
which is the failure our retention measure is designed to expose.

The next two capacities concern retrieval. Pattern completion and pattern separation are the complementary
computations that complementary learning systems theory places at the centre of hippocampal function
(Kumaran et al., 2016): the dentate gyrus orthogonalises incoming patterns before auto-associative storage
in CA3, while CA3's recurrent collaterals allow a whole stored pattern to be recovered from a fragment of
it, in the manner of an attractor network. Kahana (2020) gives the same operation its cognitive-level
definition, as the process by which a partial or noisy input retrieves a more complete trace, and shows that
it follows directly from the outer-product storage rule shared by most associative memory models: cueing
with a corrupted item returns the target plus an interference term that grows with the number of patterns
stored. Pattern completion is therefore not a binary property but one that degrades jointly with cue
corruption and with load, and it is the rate of that degradation, rather than its presence, that separates
candidate memories.

Separation, however, is usually stated over static patterns, whereas the failure that concerns us is over
sequences: the cue is intact, and the memory nonetheless returns the wrong episode, because two episodes
share a stretch of observations. The clearest evidence that the hippocampus solves this problem in the
sequence domain comes from internally generated assembly sequences (Buzsáki & Tingley, 2018). With
environmental and body-derived cues held constant — a rat running in a wheel during the delay between
trials — the population trajectory is nonetheless distinct for left- and right-choice trials and predicts the
upcoming choice, errors included. Identical observations, different episode, different trajectory: this is
the property we operationalise as sequence disambiguation. It also identifies the variable that makes the
test hard, namely the length of the shared stretch over which no distinguishing cue is concurrently
available, and we vary that length rather than merely reporting whether disambiguation occurs.

The fifth capacity is the least standard and needs the most explicit defence. Recognition and cued recall can
both be supported by a memory that is only ever driven by external input, one step at a time. Remembering,
in Tulving's (2002) sense, is not that: it is the re-experiencing of an episode from within, and the
dissociation is visible in patient K.C., whose factual knowledge of his own life is preserved while his
ability to recollect any personally experienced event is not. Two computational statements make this
operational for our purposes. Retrieved context theory (Kahana, 2020) holds that recalling an item
reinstates the context in which it was encoded, and that this retrieved context becomes the cue for the next
recall, so that free recall proceeds as a self-sustaining rollout in which the system's own output supplies
its next input. Buzsáki and Tingley (2018) make the corresponding physiological claim: perpetual change is
the default dynamic of the hippocampal system, which generates ordered sequences even in sleep, in the
absence of the experience that they encode. The capacity is also a precondition for the consolidation
mechanism that both complementary learning systems papers rely on, since reinstatement of an episode in
cortex (McClelland et al., 1995), generalised to internally generated replay in artificial systems
(Kudithipudi et al., 2022), presupposes a memory that can produce an episode without being led through it.
We therefore evaluate reconstruction under free rollout, supplying only the first element of an episode.

We do not claim this set is complete. It is a working minimum: five capacities that the literature above
converges on, that are dissociable from one another, and that are each defined by a failure a benchmark can
detect. Table 1 states each capacity together with the condition under which it is judged to have failed.

---

## Revision: narrative style (no author-as-subject citations)

Preference: citations stay parenthetical at the end of a clause; never "McClelland et al. (1995)
showed that ...". The reader is not assumed to know the authors or their points.

### Continual retention

The cornerstone of any memory system is the ability to retain information in the face of ongoing
environmental pressure. New experience arrives continuously, and the same synaptic changes that encode it
are the ones already holding what came before, so a system plastic enough to absorb the new can, in the act
of absorbing it, destroy the old (McClelland et al., 1995; Kumaran et al., 2016). This tension between
incorporating new memories and protecting existing ones is the stability–plasticity dilemma (Kudithipudi
et al., 2022; Bidaki et al., 2025), and it is what makes continual retention non-trivial to evaluate,
because more retention is not necessarily better. The constraint is not that storage runs out. It is that a
fixed substrate supports only so many codes that do not overlap, and overlapping codes overwrite one
another, which is why biological memory invests heavily in keeping its representations apart (McClelland
et al., 1995; Kumaran et al., 2016). A memory operating under that constraint cannot preserve everything,
and should not: attention-gated rapid encoding, gradual consolidation, and generalisation are all ways of
deciding what survives an incessant sensory stream, and each of them discards. Not every old memory
deserves protection, and a system that protects them all is as broken as one that protects none. What the
ideal balance is between stability and plasticity is beyond the scope of this benchmark. We instead measure
both sides of it, so that where a candidate system holds and where it breaks are visible separately.

We therefore report retention as a matrix rather than a scalar: recall of every earlier sequence, measured
again after each subsequent one, which exposes the shape of the loss and not only its magnitude. Three
standard summaries follow from it — average accuracy over all sequences at the end of the stream, backward
transfer, whose sign separates interference from consolidation, and average forgetting, the drop from each
sequence's best-ever score. All three reward stability, so we pair them with the opposite measure: a
reversal protocol in which a learned association is invalidated and must be overwritten, scoring
perseveration rather than forgetting. A memory that is simply rigid scores well on the first three and
fails the fourth. It is a system's joint position on both that we report.

> **Flag.** The draft's "the capacity of memory is physiologically bound and is thus a finite resource" is
> unsupported by the seven references, and Bidaki et al. (2025) contradicts it directly: catastrophic
> forgetting is characterised there as the rewriting of the substrate holding earlier knowledge, explicitly
> *not* as insufficient memory. Rewritten above as an interference bound rather than a storage bound — better
> supported, and it connects the constraint to sparse conjunctive coding (McClelland et al., 1995: the
> hippocampus minimises representational overlap precisely so that it does not itself suffer catastrophic
> interference). If the finite-capacity framing is kept, it needs its own citation and will argue against
> the continual-learning reference.

### Bridge to one-shot learning

Retention, measured on its own, is a conditional quantity: it scores what survives, not what ever entered.
This makes it gameable in a specific and unhelpful way, because the degenerate solution to the
stability–plasticity dilemma is total stability, and a system that encodes nothing forgets nothing.
Retention numbers are only interpretable relative to how much was acquired in the first place, which is why
acquisition needs to be measured in its own right rather than assumed. The requirement is also more
demanding than it first appears. An episode occurs once. There is no second pass over a lived experience,
and a memory that needs repeated exposure to an event has already failed by the time the event matters,
which is the demand that gradual, interleaved learning cannot meet and the reason a fast-encoding system is
needed alongside it (McClelland et al., 1995; Kumaran et al., 2016). The two capacities are therefore not
merely related but traded against each other by the same underlying mechanism: the plasticity that buys
single-exposure acquisition is the plasticity that costs retention. Reporting them jointly is what makes
the trade visible, and it is why a model that appears to have solved one is scored against the other before
that claim is accepted.

---

## Capacity → benchmark → metric map

Built from the pipeline sources, not from the design docs: section names are the literal entries of
`SPATIAL_BENCHMARKS` / `SYMBOLIC_BENCHMARKS`, and metric names are the keys actually written to
`results["metrics"]`.

> **Stale as of 2026-09-01 — read `docs/capacities/capacity_coverage_audit.md` first.**
> Two mismatches with `docs/capacities/capacities.md`: this map's fifth capacity is
> *Autonomous generation*, the paper's is *Serial order* (the merge is not
> reflected here); and the sections `spatial_sequence`, `anchoring_few_shot`,
> `letter_noise` and the `online_continual` suite were removed (audit D1/D3/D5/D8).
> Rows below are annotated but not yet rewritten.

| Capacity | Benchmark section (suite) | Swept variable | Metrics |
|---|---|---|---|
| **Continual retention** | `continual_chain` (symbolic §3b) — the retention matrix as a suite section (2026-09-02; supersedes the removed `online_continual`, D8/C4) | position in a 6-task chain | retention matrix `R[j,i]`; ACC; backward transfer; average forgetting; **retained fraction at each interposition gap** (the load curve) |
| | `continual_chain`, diagonal (symbolic §3b) | position in the chain | **`chain_avg_learning`** (acquisition under load); **`chain_intransigence`**; `chain_learning_slope`; **`chain_stability_plasticity_index`** — the plasticity axis, which every other section in this capacity is blind to |
| | `continual_chain`, rehearsal phase (symbolic §3b) | relevance: the oldest half is re-presented, the rest is not | `select_selectivity` (difference-in-differences), `select_rehearsed_gain`, `select_unrehearsed_drift`; guarded on `select_under_pressure` |
| | `multiple_sequences` (symbolic §3) | one interposed sequence | MRR before / after; ΔMRR forgetting |
| | `paired_associate` (symbolic §3c) | same cue, new target (AB/AC) vs. disjoint cue (AB/DE control) | AB intrusion rate; AC recall and trials-to-criterion; collapse rate; **`pa_cue_competition_cost`** |
| | `tmaze_reversal` (spatial §4) | acquisition → extinction → reversal → re-reversal | trials-to-criterion per stage; final arm accuracy; savings; reward gain |
| | `schema_consistency` (symbolic §6, interference readout) | relatedness of the disrupted material to the new item (same / sibling / far category) | `schema_<rung>_interference_<same\|sibling\|far>` — the graded interference profile that motivates similarity-weighted interleaving |
| **One-shot learning** | ~~`anchoring_few_shot`~~ **REMOVED (D3)** — drift correction, out of scope | exposure ladder 1→128 epochs | — |
| | online convergence (symbolic, streaming) | 1 / 5 / 20 passes | MRR and memory span at each pass count |
| | `presentation_duration` (symbolic §1) | settling / epoch budget | epochs to convergence; converged flag; MRR and span at convergence |
| | `schema_consistency` (symbolic §6) | consistency of the new item with the acquired schema (duplicate / within-branch / cross-branch / unstructured) | `schema_<rung>_trials_to_criterion`; `schema_consistency_speed_corr`; `schema_<rung>_projection_ratio` (model-free ladder check); `schema_probe_*` and `schema_acquired` (did the model learn a schema at all); `schema_resolved` (ceiling/floor guard) |
| **Pattern completion** | `tmaze_completion` (spatial §1) | ~~cue prefix length~~ (sweep dropped) | MSE; **coverage**; **divergence_step** (moved here from the removed S-curve section, audit C1) |
| | ~~`spatial_sequence`~~ **REMOVED (D1)** | — | — |
| | `noise_invariance` (symbolic §4) | Gaussian σ on the retrieval cue | noise tolerance threshold (σ at which recall breaks) |
| | ~~`letter_noise`~~ **REMOVED (D5)** — scored the vocabulary's edit-distance geometry, not the model (across-model spread 0.000–0.111 vs 0.711 for the σ sweep) | — | — |
| **Sequence disambiguation** | `tmaze_disambiguation` (spatial §2) | route pair A/B sharing the stem, **× graded odour availability** (concurrent → withdrawn at delay 0/2/4/6/7) | branch accuracy; confusion rate; per-route branch accuracy; **divergence accuracy**; **normalised branch margin**; **recovery profile**; **shared-stretch control** |
| | `semantic_similarity` (symbolic §5) | within- vs across-category overlap | MRR high vs low similarity; similarity-effect MRR drop |
| **Serial order** *(paper's capacity; this map still says "Autonomous generation")* | `sequence_length` (symbolic §2) | list length × feedback mode (raw / L2 / quantized) | memory span under autoregressive rollout, per feedback mode |
| | ~~`anchoring_few_shot`~~ **REMOVED (D3)** | — | — |

Notes for the methods text:

- `schema_consistency` is the one section that appears under two capacities, and that is the
  point rather than an untidiness: the same manipulation yields an acquisition readout
  (trials-to-criterion per rung) and an interference readout (damage to prior material, split by
  relatedness). Cite it in both rows; describe the protocol once.
- **Do not quote the rung ordering as a clean result yet.** With cued recall as the dependent
  variable, an item orthogonal to the schema is *easy* to retrieve precisely because it is
  discriminable from everything else, so the acquisition ladder is non-monotone at the
  unstructured end (observed EP trials-to-criterion 1.0 / 1.7 / 7.3 / 1.3). The paper's dependent
  variable is integration, ours is retrieval, and they come apart there. See
  `docs/sections/schema_benchmark_design.md` S7 confound 5 for the three options and S8 for the readout
  that would resolve it.
- Read `schema_acquired` and `schema_resolved` before any rung comparison. If the model never
  acquired the schema, or every rung sits at ceiling or floor, the consistency manipulation is
  vacuous and the numbers below it mean nothing.
- The results intro should promise **acquisition rate** as a function of consistency, not
  consolidation time. Time-to-consolidation is designed but not built (S8).
- **Prefix length is not an informative sweep for `tmaze_completion` (decided 2026-08-21: drop
  it).** Under the pure one-step `predict_next` contract the prediction at the turn depends only
  on the cue immediately before it; earlier stem cues in the prefix change nothing, so the sweep
  varies a quantity the probe cannot see. The completion prose must therefore not lean on
  "recall as a function of prefix length" — the shipped completion readouts are cue *corruption*
  (`noise_invariance`, `letter_noise`) and trajectory recovery from a cue point (spatial
  sections, jointly with generation). Check whether the same logic voids the `spatial_sequence`
  prefix sweep before quoting it.
- **Free recall (designed, DEMOTED 2026-09-02):** generative protocol (initiate from a
  non-item cue, roll out, decode, score output order). Its case was the genuine lag-CRP, and
  temporal contiguity is no longer a promise of the capacity — see audit sec 5, S-O1. The
  architecture dissociation it claimed (chaining vs symmetric-associative vs context-carrying)
  is now carried by `transposition_locality` + `transposition_asymmetry` off cued recall, at a
  fraction of the cost. Its unique remainders are the initiation curve and unconstrained-order
  recall, which no paragraph-11 row asks for. Full design retained:
  docs/proposals/free_recall_design.md. Do not table it as if wired, and no longer describe it as the
  next thing to build — that is S-O2 (decoded-identity return).
- **Interval encoding (designed, not built):** a serial-order binding extension from ordinal to
  metric time — does the model store *how long* separated two items, not merely which came
  first. Grounded in trace-conditioning CR timing (Kishimoto et al., 2006) and time-cell
  retiming (MacDonald et al., 2011). Online arms only; shares the state-carrying probe
  prerequisite with the nonlinearity doc's history-dependent family. Full design:
  docs/proposals/interval_encoding_design.md. Do not table it as if wired.
- **Context stays implicit — deliberately, for now.** The disambiguation prose defines the
  capacity by carried context, but the harness must not *pass* context explicitly, for three
  reasons. (1) Handing the model the disambiguating signal during the shared stretch solves the
  task on the model's behalf: the metric then measures conditional association, not context
  maintenance — this is exactly the shipped T-maze's situation, with odour on every stem step.
  (2) Where context lives is a model decision (recurrent state, fast weights, an external slot);
  an explicit context interface forces one architecture's opinion on every arm, and the existing
  `context_data` channel already shows the cost — only 4 arms implement `n_context`, so routing
  anything through it silently excludes tPC, GPT-2 and `OriginalEqProp`
  (docs/proposals/nonlinearity_benchmark_design.md §5.1). (3) Testing *maintained* context requires
  presenting the disambiguating cue early and withholding it during the shared stretch, which
  the memoryless probe cannot express — it needs the state-carrying ingestion protocol of
  §4 of the same doc. The upgrade path is §5.3's delayed-conjunction T-maze (`odour_duration`
  knob): it places the shipped section on a continuum instead of replacing it. Until then, the
  prose should present the shipped section as the fully-cued easy end of a stated continuum,
  and promise nothing about delay-bridging.

- The spatial completion sections already recall open-loop (`anchor_period=0`, pure replay), so their
  numbers are joint completion-plus-generation scores. Say so, or the autonomous-generation row looks
  like it is the only place rollout is exercised.
- The three feedback modes are the cleanest generation-specific measurement in the suite: `raw` compounds
  magnitude drift, `l2` removes it, `quantized` snaps to the codebook and is the drift-free upper bound.
  The gap between them separates representational degradation from readout drift.
- `anchoring_few_shot` crosses the exposure ladder with the anchor-gain conditions, so it is the one
  section that scores one-shot learning and autonomous generation on the same grid.
- **Resolved 2026-09-01:** the three `"TODO"` string keys are gone. `interval_decay_slope`
  was removed from the symbolic suite (audit D6 — the batch suite has no time axis; the
  elapsed-time intent moves to the online suite beside `isi_tolerance`), and the two
  `object_arena_*_mse` keys were removed with the unimplemented section (D4). No pipeline
  now writes a literal `"TODO"` into `metrics.json`.
- **Disambiguation is no longer only at its degenerate corner.** The section now ships a
  graded odour-availability sweep beside the fully-cued block. Quote it as a protocol
  finding: withdrawn rows floor at chance because no arm carries state across the cue-free
  stretch, not because the arms cannot disambiguate.
