# Cognitive phenomena — design note (L0 of the correspondence ladder)

**Status:** implemented, tested, and registered in `SYMBOLIC_BENCHMARKS` as section 6b
(`cognitive_phenomena`). Verified on AHN, theta and DTS-ESN at 15 replicates (§9). **Descriptive by contract:** the section emits no
scored metric, writes its own `cognitive_phenomena_metrics.json`, and is excluded from the
capacity scorecard on purpose.

- `memval/benchmarks/cognitive_phenomena.py` — `run_cognitive_phenomena()`
- `tests/test_cognitive_phenomena.py` — layout separation, taxonomy, bands, availability, the
  no-scored-metrics contract
- Probe: `CLEAN_SINGLE` (declared in `memval/benchmarks/probe.py`); replication over material

**Literature anchors** (the reference band; all verified against the paper's text, not abstracts):

- Kahana (2020), *Computational models of memory search*, Annu. Rev. Psychol. 71:107–138, §4
  "Benchmark recall phenomena" — the phenomena "that any successful recall model should be able to
  explain", with the figures cited per phenomenon below.
- Murdock (1962) — serial position, list length and presentation rate (Kahana Fig. 2a).
- Kahana et al. (2010) — immediate serial recall curves for 7/13/19 items (Kahana Fig. 2b).
- Zaromb et al. (2006) — prior-list intrusion recency and semantic relatedness (Kahana Fig. 5a, 5c).
- Howard & Kahana (2002) — the semantic-CRP (Kahana Fig. 4b).
- Polyn, Norman & Kahana (2009) — the percentile-rank "factor" statistic used for semantic
  organisation.
- Love (2021), *Levels of biological plausibility*, Phil. Trans. R. Soc. B — why this rung is
  descriptive (§1).

---

## 1. Why an L0 section, and why it is not scored

The correspondence declaration (see the ladder discussion in the paper's methods) assigns each model
the levels at which it claims to correspond to the brain. The lowest rung, L0, is behaviour: does the
model reproduce the *shape* of recall, not only its level? Two things make L0 different from every
other rung.

First, it is the one rung where MemVal does not have to invent a standard. The mathematical-psychology
memory literature has adjudicated models against a shared set of canonical curves for fifty years —
SAM, TCM and CMR are all judged on which of them they recover — and Kahana (2020) states the list
before introducing any model. We import that list rather than write our own.

Second, following Love (2021), the section must specify the findings a model addresses rather than
issue a verdict. So every read-out is reported beside the human signature (direction and ordering
only; the quantities are not commensurable with human recall probabilities), a `cog_dir_*` flag
records whether the sign matches — or `None` when the replicate interval spans zero and no direction
was measured — and **nothing is summed, weighted or ranked**. An arm that declares
a behavioural correspondence is obliged to these read-outs; an arm that does not is merely described
by them. This is also why the section keeps out of `results["metrics"]`: the scorer classifies every
key there and would either have to score these or carry a growing list of "diagnostic" exceptions,
and either would blur the line the declaration draws.

## 2. The five phenomena and their human signatures

| phenomenon | human signature (Kahana 2020) | read as |
|---|---|---|
| **Serial position** | Immediate serial recall: strong primacy, modest recency (Fig. 2b). Immediate free recall: strong recency, smaller primacy. | `primacy_index > 0`, `recency_index > 0` on the **cued** curve. |
| **List length** | Longer lists reduce recall of early and middle positions and do not reliably change the last few (Fig. 2a). | `early_slope < 0`, `late_slope ≈ 0`; dissociation `= late − early > 0`. |
| **Presentation rate** | A slower rate increases early/middle recall and has no discernible effect on the final items. | `early_slope > 0`, `late_slope ≈ 0` against log₂(passes); dissociation `= early − late > 0`. |
| **PLI recency** | Prior-list intrusions come mostly from the last two lists and fall off with list lag (Fig. 5a). | `pli_lag1_share` high, `pli_recency_spearman < 0` on the availability-corrected rate. |
| **Semantic clustering** | Transitions and intrusions are biased toward items semantically similar to the just-recalled item, even on lists without obvious associates (Fig. 4b, 5c). | `cue_partner` share above chance and `semantic_factor_cue > 0.5` at structured rungs, both at chance at the no-structure rung. |

Our probe is serial (each transition cued by its predecessor), so the serial-recall variants are the
reference, not the free-recall ones. The free-recall phenomena that need an unordered output (lag-CRP,
probability of first recall) belong to `docs/free_recall_design.md` and are not attempted here.

## 3. Protocol

### 3.1 One grid for three phenomena

Phenomena 1–3 are three marginals of a single material grid,

```
recall[position, list length L, exposure]
```

- **Material.** One list per category, `L` first words in vocabulary order, for categories with at
  least `max(lengths)` words: fruit, animal, number. Lengths `(5, 7, 10)`, headline `L = 7` so the
  serial-position curve is on the same list every other section uses.
- **Replicates.** The unit is **(category, encoder seed)**, `n_seeds = 5` → 15. See §7 for why
  category alone is not enough.
- **Encoder.** One `SymbolicEncoder` per (category, seed) over **all** of that category's words, so the decode
  candidate set is the same at every length (chance does not move with `L`) and the first `L` words
  carry identical vectors at every length (one RNG draw per word, in vocabulary order).
- **Exposure.** Each cell is trained to criterion (0.95 cued recall) by the staircase, and *also* at a
  fixed ladder of `(1, 2, 4, 8)` passes from scratch. The criterion rung is where the suite's other
  sections read; the ladder is where the curve's shape lives in accuracy rather than only in the
  margin (§7).
- **Read-outs per cell and rung.** Cued accuracy, cued margin (target minus best competitor), and the
  autoregressive rollout curve (l2 feedback, harness-controlled).

Then: serial position = the headline-length cell, per rung, replicate-averaged; list length = band
means against `L`, per rung; presentation rate = band means against log₂(passes) over the fixed
ladder at the headline length (the criterion rung is excluded because its exposure differs per cell).

Bands over the scored positions `1 … L−1` (position 0 is the cue): two positions per end when at least
five are scored, otherwise one; the middle is the remainder.

### 3.2 Prior-list intrusion recency

A chain of six lists (one category each, five words; the `continual_chain` material via
`build_chain_material`) trained **in order on one model with no reset**. After each list `j` is
trained, that list is probed and every error is attributed to the list its decoded word belongs to
— lag `k = j − source` — and so is every runner-up (the best non-target item by cosine), which
resolves the same question above the accuracy floor.

Two exposure regimes, both reported. `criterion` trains each list to the suite's criterion on its own
cued recall, as every other section does — which also means the just-trained list is at ceiling by
construction, so an arm can only show prior-list intrusions there through its runner-ups. `one_pass`
studies each list exactly once, which is the human paradigm and where a slower arm's current-list
errors live. A lag profile needs intrusions to be a profile: below `min_pli = 5` the accuracy read-out
has not posed the question (`question_posed = False`) and the direction is read from the runner-up
profile, with `direction_source` saying so.

The between-list prototype cosine is swept `(0.0, 0.5, 0.8)`. This is the section's overlap dial and
also its null: under an *exact* between-list cosine every prior list is equidistant from the current
one, so the stimulus predicts a **flat** lag profile and any gradient is the memory's. The rung with
the most prior-list intrusions is the headline; a rung with none has not posed the question and says
so (`question_posed = False`) rather than reporting a gradient of zeros.

Availability normalisation: at stage `j` only lags `1 … j` exist, so counts are divided by
`opportunities(k) = (seq_len − 1) · (T − k)`. Kahana's share-of-all-PLIs is reported beside it; with
six lists the share is dominated by availability and the rate is the fair statistic. The retention
matrix and the final-state source matrix (every list probed, errors by source list) are kept as
context.

### 3.3 Semantic clustering

**The problem.** For the model, "semantic similarity" *is* embedding proximity — the two cannot be
separated at the input, because the model sees only vectors. What can be separated is what the
memory does with that proximity. Three routes lead from a cue to a semantically related wrong answer:

- **S1 organisation** — from the cue, retrieve its semantic associate instead of its temporal
  successor. This is the human semantic-CRP analogue (Howard & Kahana 2002): the transition itself
  follows meaning.
- **S2 output confusability** — retrieve the right neighbourhood and land on the target's sibling.
- **S3 input generalisation** — treat the cue as its sibling and follow *that* chain.

S2 and S3 are what a reviewer means by "embedding-space clustering"; S1 is the phenomenon. They are
confounded on any list where the partners sit at a uniform lag.

**The layout.** Eight slots, four categories, each appearing twice, arranged `A B C D B A D C`. For
every cue `i` with target `i+1`, the four items {target, partner(cue), partner(target),
successor(partner(cue))} are pairwise distinct and none is a temporal neighbour of the target
(`validate_semantic_pattern`; pinned in the tests, which also show that the "obvious" layout
`ABCDABCD` collapses S2 and S3 onto one item). Adjacent slots never share a category, so the temporal
successor is never a semantic associate.

**Candidates.** The decoder ranks over the eight list words plus two further words of every list
category and two words of every non-list category (20 candidates), so extra-list intrusions are
possible and classifiable by their category relation to the cue and to the target — Kahana Fig. 5c's
"intrusions are semantically related" finding.

**Taxonomy.** Each error (decoded ≠ target) and each runner-up is one of: `cue_partner` (S1),
`target_partner` (S2), `cue_partner_successor` (S3), `cue_repeat`, `temporal_neighbour`,
`other_in_list`, `eli_cue_category`, `eli_target_category`, `eli_list_category`,
`eli_other_category`.

**Statistics.**
- Share of errors and rate per probe in each class, against `chance_specific = 1/(V−2)` — the
  probability that a uniformly random error lands on one specific item.
- **Semantic factor** (after Polyn et al. 2009): the percentile rank of the produced item's cosine to
  the cue among all non-target, non-cue candidates; 0.5 is no organisation, 1 is the most cue-similar
  candidate every time. Reported cue-relative (the semantic-CRP analogue) and target-relative (the
  confusability reading), on errors at the first ladder rung and on runners-up at criterion.
- **Calibration.** The within-category cosine is swept via `category_variance` `(0.1, 0.2, 1.0,
  3.0)` — cosines ≈ 0.5, 0.2, 0.01, 0.001 by the encoder law `ρ = 1/(1 + d σ²)`. At the last rung
  there is no semantic structure, so every semantic share must fall to chance and the factor to 0.5;
  `calibrated` records whether it did. If it did not, the measure is reading something other than
  semantics.
- **Stimulus reference.** `nn_is_partner_fraction`: how often a cue's nearest candidate is its
  partner. At high within-cosine this is 1 — the codec alone would confuse partners — and S2/S3
  shares must be read against it.

Three lists rotate the category quadruples; each is trained fresh, so the replication is over
material, not over cue noise.

## 4. What the section answers about the semantic confound

The question was: how to design the manipulation so that semantic similarity (shared features) is
not confounded by clustering in embedding space, which is itself semantic proximity. The answer this
section takes is that the confound cannot be removed at the input and should not be pretended away;
it is removed at the **output**, three ways at once:

1. **Positionally** — partners are separated in time so that the temporal prediction, the cue's
   associate, the target's sibling and the sibling's successor are four different words. An error
   names which route it took.
2. **Statistically** — a specific-item chance level, a permutation-free percentile-rank factor, and
   the runner-up read-out that works at ceiling.
3. **By calibration** — the structure dial is turned to zero and the semantic read-outs are required
   to go to chance there. A measure that survives the removal of the thing it claims to measure is
   not measuring it.

S1 is then the only class that counts as semantic organisation. S2 and S3 are reported with equal
prominence because a model that shows *only* S2/S3 is doing exactly what the reviewer suspected —
nearest-neighbour confusion in the code — and the figure should say so.

## 5. Harness opinions — declared choices

1. **The serial-position curve is the cued curve, never the rollout.** A chained rollout that errs
   is off the manifold, so its curve decays monotonically by construction: primacy there is error
   propagation and recency is impossible. The rollout curve is drawn beside the cued one to make the
   contrast visible, not as a read-out. The same precision as the free-recall note's "a cued
   transposition gradient is not a lag-CRP".
2. **"Presentation rate" is passes over the list.** The batch suite has no per-item duration;
   exposure is uniform across positions, as in Murdock's manipulation. The read-out is the
   dissociation (which positions gain), not the level. The retired `presentation duration` axis of
   the manipulation taxonomy was a *probe* property; this is an encoding-side manipulation with a
   specific human signature and lives in a different slot.
3. **Two exposures are read, and the doc says which is which.** At criterion the cued accuracy is
   ≥ 0.95 by construction and the curve's shape lives in the margin alone; at one pass the shape is
   visible in accuracy. Summary keys carry `_first` or `_criterion` so the two are never conflated.
4. **Replication is over material.** Categories for the grid, lists for the semantic section — never
   cue noise, per the probe protocol. A curve averaged over three categories is a three-replicate
   mean and is reported with its SD.
5. **PLI headline rung = most intrusions**, stated in the output. The alternative — averaging
   profiles across overlap rungs — mixes rungs where the question was posed with rungs where it was
   not.
6. **Ties are failures**, as everywhere in the suite; a null prediction loses to everything.

## 6. Why it discriminates

The phenomena that separate architectures are the ones that separate **chaining** from
**positional / context** coding, because every arm in the roster is chaining-shaped
(`predict_next` over item pairs):

- A pure pairwise associator learns a position-independent transition map, so its cued curve is
  flat: no primacy, no recency. The human serial-recall curve is not flat. That dissociation is
  available with no new code and is a finding, not a null.
- List length and presentation rate cost or benefit every position equally for such a map; the human
  data protect the last few items in both manipulations. Whether an arm's late band is protected
  is a direct question about whether recency is carried by anything but the transition weights.
- PLI recency separates a memory whose interference is *ordered in time* (recent lists intrude more)
  from one where interference is set by geometry alone (flat under the exact-cosine null). The
  runner-up version asks the same of an arm that never errs.
- The semantic taxonomy separates an arm that *retrieves by meaning* (S1) from one that merely
  confuses neighbours in the code (S2/S3).

## 7. Caveats

- **Ceiling at criterion.** By construction. Read shape from the margin there, from accuracy at one
  pass; the section reports both and labels them.
- **Human word lists, no rehearsal.** Kahana's primacy is a rehearsal account (enhanced by long
  inter-item intervals, reduced by incidental encoding). No arm here rehearses, so even where the
  curve matches the explanation does not transfer. The claim is only ever "the shape of arm X's curve
  under this protocol, against the human shape as a band".
- **Replicates and power.** The replicate unit is **(category, encoder seed)**, `n_seeds = 5` over
  three categories = 15. Category alone caps at three, because only fruit/animal/number hold ten
  words — and at n = 3 *every* band index flipped sign across replicates for all three arms tested,
  so no direction was supportable. Every band index is therefore reported as mean + 95% interval,
  and `cog_dir_*` returns `None` (undetermined) whenever the interval spans zero. `None` means the
  run did not measure a direction; it does not mean "no effect".
- **Small intrusion counts.** Six lists of five give 24 current-list probes per rung. The
  availability-corrected rate is the honest statistic; shares of a handful of intrusions are not.
  The runner-up profile has 24 observations per rung regardless.
- **"Modality" collision.** Kahana's modality effect is auditory vs visual presentation; MemVal's
  modalities are spatial vs symbolic. Out of scope, and worth one sentence in the methods.
- **Interresponse times** are out of scope: no commensurable retrieval-latency analogue across arms.
  Settle-steps-to-convergence would be one for the energy-based arms only.

## 8. Placement

Not in the capacity map and not in `MAIN_FIGURES`: it is the L0 rung of the correspondence
declaration, not a capacity. Output:

- `results/<run>/<Arm>/symbolic/cognitive_phenomena_metrics.json` — everything, including the human
  reference block and the per-probe records.
- `results["series"]["cognitive_phenomena"]` — the `cog_*` summary and the reference block, so a
  cross-arm comparison can read it from `metrics.json` without the scorer ever classifying it.
- Five figures, `plots/cognitive_*.png`, captioned in `bin/build_results_index.py`.

Run it alone with

```bash
python bin/run_benchmark.py --model hopfield --suite symbolic --benchmarks cognitive_phenomena --output-dir results/cognitive_l0
```

**Never into a directory holding a full-suite `metrics.json`**: a single-section run rewrites that
file with only its own sections.

## 9. Results — AHN, theta, DTS-ESN (2026-09-07)

Run: `results/cognitive_l0/<Arm>/symbolic/`, registry defaults, criterion exposure, **15 replicates**
(3 categories x 5 encoder seeds). All three arms reach criterion in one pass on every grid cell, so
cued accuracy is 1.0 everywhere and every read-out below is from the **margin** or the **runner-up**.
A direction is claimed only where the replicate 95% interval excludes zero.

> **Correction.** The first version of this section reported AHN's list-length dissociation as
> matching the human direction. That was read off an n = 3 mean whose replicates flipped sign. At
> n = 15 the interval spans zero and the correct statement is that no direction was measured. The
> interval machinery and the `spans_zero` gate were added in response.

### 9.1 theta is AHN, to the digit

Every one of the ~50 summary read-outs is bit-identical between `theta` and `hopfield`. That is the
expected result — at the paper's optimal phases the theta cycle integral reduces exactly to the delta
rule, so the pair takes the same weight step — and it makes this section a **regression test on the
harness**: any future divergence between them is a bug or a readout effect, never a model difference.

### 9.2 Serial position, list length, presentation rate (margin at criterion, n = 15)

| read-out | AHN / theta | DTS-ESN | human |
|---|---|---|---|
| primacy index | −0.007 [−0.046, +0.033] | **+0.049 [+0.009, +0.088]** | > 0 |
| recency index | +0.011 [−0.042, +0.064] | −0.036 [−0.084, +0.012] | > 0 |
| list-length dissociation (late − early slope) | −0.005 [−0.018, +0.008] | **−0.029 [−0.040, −0.018]** | > 0 |
| presentation-rate dissociation (early − late slope) | +0.004 [−0.002, +0.010] | **+0.024 [+0.016, +0.032]** | > 0 |

**AHN / theta: all four undetermined.** No primacy, no recency, no protected late band under either
list length or study time. This is the clean null predicted for a position-independent transition
map, and it is the section's headline negative result.

**DTS-ESN: three of four measured, two matching — but see §9.7 before reading the primacy.** Primacy
is positive under the suite's probe. The capacity report in §9.7 shows that this is a property of the
*cold* probe, not of the memory: under the training-matched probe the same arm has no primacy and a
strong, monotone recency. The interpretation first written here ("the reservoir's transient gives early
items an advantage") was wrong and is withdrawn. The presentation-rate dissociation is human-directional and 6x AHN's: extra passes
of RLS help the early band far more than the late. The list-length dissociation is
**significantly opposite**: DTS-ESN's late band degrades *faster* with length, where humans protect
it. Recency is undetermined at n = 15 (a 30-replicate pilot put it at −0.052, CI excluding zero;
treat anti-recency as suggestive, not measured).

### 9.3 Prior-list intrusion recency

No arm errs on the just-trained list at any overlap rung (0/24 probes; DTS-ESN has **zero forgetting**
at every rung against AHN/theta's 0.05–0.10), so the intrusion read-out never poses its question and
the direction comes from the runner-up profile, as `direction_source` records.

| | AHN / theta | DTS-ESN |
|---|---|---|
| prior-list runner-up share (rho_b 0.8) | 0.54 | 0.46 |
| lag-1 share | 0.38 | 0.27 |
| runner-up recency (Spearman vs lag) | −0.31 | **−0.67** |

Both match the human direction, and under the exact-cosine null every prior list is equidistant, so
the gradient is the memory's, not the stimulus's. DTS-ESN's is twice as steep: its competition comes
much more sharply from the recent past despite it forgetting nothing.

### 9.4 Semantic clustering

No arm errs at any structure rung, so the accuracy read-outs are undetermined for all three and the
runner-up profile carries it. Every arm calibrates (shares fall to chance, factor to 0.5, at the
no-structure rung).

| | AHN / theta | DTS-ESN |
|---|---|---|
| S1 `cue_partner` (organisation) | **0.000** | **0.000** |
| S2 `target_partner` (output confusability) | 0.238 | 0.286 |
| cue-relative semantic factor | 0.426 | 0.252 |

**No arm shows any semantic organisation of retrieval.** S1 is exactly zero everywhere: not one probe
in any arm has the cue's own semantic associate as its runner-up. What all three show instead is
confusability among the *target's* neighbours, and both cue-relative factors sit **below** 0.5, so the
produced competitor is if anything less cue-similar than a random candidate. This is precisely the
"clustering in embedding space" the layout was built to separate out, and the separation works: the
phenomenon is absent while the artifact is present and measurable.

### 9.5 What to say in the paper

Two arms that take the identical weight step (AHN, theta) produce identical L0 profiles, and a
reservoir with a fixed random dynamics and an RLS readout (DTS-ESN) produces a different one on three
of five phenomena. So the section discriminates, and it discriminates on the axis it was built for:
DTS-ESN carries something AHN does not — a genuine primacy, an exposure profile that favours early
items, a steeper recency gradient in interference — because its state has a history that a pairwise
transition map lacks. None of the three shows semantic organisation of retrieval, and none shows
human recency.

### 9.6 Next (superseded in part by 9.7)

The arms that will exercise the sub-criterion ladder, where the *accuracy* read-outs (currently
undetermined for want of any error) start to resolve: `temporal_pc`, `original_eqprop`, `chen`, and
the spiking arms. Add `--benchmark-args cognitive_phenomena:n_seeds=10` for a 30-replicate run when a
direction sits near the interval boundary.

### 9.7 DTS-ESN capacity report — pushed to 2560 items (2026-09-09)

`bin/esn_cognitive_report.py`; figure `results/cognitive_l0/DTSESNSequenceNetwork/symbolic/plots/
esn_capacity_report.png`. Random-vector lists (no category structure) decoded over the full 2560-word
candidate set, one pass, 6 material seeds, first/middle/last **3 items** as bands. Two probes, because
the arm is stateful: the suite's **cold** probe rebuilds the reservoir from rest on the cue alone; the
**training-matched (warm)** probe primes with items 0..t−2 and predicts from item t−1, which is the
state the RLS readout was trained on.

| | warm | cold |
|---|---|---|
| recall = 1.0 up to | L = 320 | L = 40 |
| L = 640 / 1280 / 2560 | 0.95 / 0.74 / 0.45 | 0.12 / 0.03 / 0.02 |

**Forgetting appears at L = 1280** (warm). At that length the three phenomena read:

- **Serial position:** recall rises monotonically from 0.41 (first decile) to 0.95 (last). No
  primacy; strong recency. The exact **mirror** of human serial recall (primacy-dominated).
- **List length:** the last 3 items stay at 1.00 out to 2560 while the first 3 collapse (1.00 → 0.67 →
  0.06 → 0.00 across 320 → 2560) and the middle declines (1.00 → 0.45). The human dissociation
  (late protected, early/middle cost) is present, but as a cliff for early items rather than a graded
  loss, and with the late band *perfectly* protected rather than merely stable.
- **Presentation rate:** the first 3 items go 0.06 → 0.39 → 0.83 → 1.00 over 1 → 8 passes while the
  last 3 sit at 1.00 throughout. The human direction (study time helps early/middle, not late),
  sharply.

**Mechanism.** The readout is recursive least squares with no rehearsal: each new transition
overwrites the solution in the directions the newest state occupies, so the most recent transitions
are the least overwritten. The arm is a **palimpsest**, and all three curves are the same fact: a
pure recency memory. Extra passes re-present the early items and rescue them — presentation rate is
"forgetting undone", not "encoding strengthened".

**The cold-probe gap.** Under the cold probe recall falls from L = 160 and the L = 7 curve shows the
"primacy" reported in §9.2. That primacy is the mismatch: a cold cue discards the history the state
was trained with, and later positions have more history to lose. It is a probe artefact for any
stateful arm and does not appear under the matched probe. **Follow-up:** the section's `cued_probe`
uses the house cold path for every arm; the `StatePrimeable` arms (DTS-ESN, tPC, Chen) should be probed
matched, with the cold curve kept as the reported gap. This is the same point as
`docs/probe_protocol.md`'s "the criterion probe must match the training input", applied to L0.
