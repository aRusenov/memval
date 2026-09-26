# Schema benchmark — design note

**Status:** encoder implemented and tested; benchmark implemented, running, and **registered in
`SYMBOLIC_BENCHMARKS` as section 6** (`schema_consistency`). The behavioural protocol still has two
open confounds (S7) — read the caveats in S6 before quoting any rung ordering as a result.

- `memval/encoders/hierarchical.py` — `HierarchicalEncoder` (19 tests, `tests/test_hierarchical_encoder.py`)
- `memval/benchmarks/schema_consistency.py` — `run_schema_consistency()`
**Motivating paper:** McClelland, McNaughton & Lampinen (2020), *Integration of new information in
memory: new insights from a complementary learning systems perspective*, Phil. Trans. R. Soc. B
375:20190637. PDF: https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonLampinen20IntegrNewInfoCLS.pdf

---

## 1. Why a schema section

CLS theory as stated in 1995 makes cortical learning slow *per se*. The 2020 paper retracts that: it is
an oversimplification to call learning in a cortex-like network inherently fast or slow, because the rate
is prior-knowledge dependent, and rapid learning does not always produce interference. Both are matters
of degree set by consistency with what is already known.

That is a benchmark-design claim. It says a scalar one-shot number and a scalar forgetting number are
both under-specified: each should be reported **as a function of consistency with existing knowledge**.
MemVal currently reports both as scalars, so this section is the direct response.

Secondary payoff: the paper's own future-work list asks whether **sparse coding that reduces pattern
similarity enhances the advantage of similarity-weighted interleaving**, while noting excessive sparsity
destroys the similarity structure needed to generalise. We have DG-expansion and XdG arms, which are
sparsity manipulations. A sparsity x similarity-weighting sweep answers a question the original authors
posed and could not answer, reusing machinery we already have. That is a stronger contribution than
reproducing SWIL.

## 2. How the paper actually builds the hierarchy

Verified against the paper, SS2-3 and figures 4 and 7.

**Generative rule.** Items are produced by a process starting at a root node and successively branching.
A feature may be introduced at any node and occur in any of that node's children, but **cannot** occur in
children of nodes in other branches. Deviations from strict hierarchy (features that cut across branches,
e.g. gender) are acknowledged and set aside.

**Concrete dataset.** An item-property matrix: 8 items as rows, binary features as columns.
- 2 sets of 4 (animals / plants) with zero feature overlap between the sets
- within each set of 4, 2 sets of 2 (birds / fish) sharing some features, differing on others
- within each pair, differentiating features (hawk large+fierce, sparrow small+meek)
- one unique identifying feature per item, standing in for its name

Simplification from the original Rumelhart & Todd network: the relation units (ISA / IS / CAN / HAS) are
dropped, so each item maps to a single feature vector rather than 32 item-relation training pairs.
Inputs are one-hot. The paper argues this is not a limitation — the theory is unchanged for orthogonal
multi-dimensional input vectors, and extends to correlated inputs as long as patterns mapped to distinct
outputs are linearly separable.

**The hierarchy IS the SVD.** M = sum_i s_i u_i v_i^T, with u_i an item-classifying vector, v_i^T a
feature-synthesizing vector, and s_i = sqrt(variance explained). Each singular dimension is one split:

| Dimension | Encodes |
|---|---|
| 1 | mean features of the animals |
| 2 | mean features of the plants |
| 3 | birds vs fish (u3 +ve for birds, -ve for fish; v3 +ve for bird-features, -ve for fish-features) |
| 4 | flowers vs trees |
| 5-8 | per-item offsets within each mid-level category |

One dimension per pairwise split; an N-way split at the same level needs N-1. Branches carrying more
features carry more variance and therefore larger singular values. Since learning rate in a deep linear
network is set by the singular value, **progressive differentiation is a mathematical consequence, not a
design choice** — superordinate splits are learned before subordinate ones.

**Adding a new item.** The sparrowhawk is small like the sparrow, fierce like the hawk, plus a unique
feature. Re-running the SVD on the augmented dataset: dims 1 and 3 are slightly altered and slightly
*stronger* (one more animal/bird contributing statistics), dim 5 still separates sparrow from hawk, and
**one new dimension is required** to map all three birds correctly. The new item's row in U is its
consistency profile.

So: **schema-consistent means the new item lies within the span of the existing singular basis.**
Continuous and computable, never annotated.

## 3. The consistency ladder

Ordered by construction, with the paper's predictions:

| Rung | Construction | Predicted acquisition | Predicted interference |
|---|---|---|---|
| 1 | duplicate of an existing item's feature profile, new name (cardinal, trout) | near-instant; pure projection, no new dimension | "only the slightest" |
| 2 | novel within-branch recombination + unique feature (sparrowhawk) | fast; mostly projection + one weak new dimension | mild, and *local* |
| 3 | cross-branch recombination (penguin: bird ISA x fish CAN) | slower; violates the branch rule | systematic — calls animals birds, says they all swim |
| 4 | random cross-pairing across output subsets | very slow; no projection | drastic, all previously acquired items (McCloskey & Cohen regime) |

Rung 1 is worth stating carefully: the association is **arbitrary** (the network could not have known
which output units to map to) and still learned rapidly. Consistency is about structure, not
predictability. Any port must preserve that property or it measures something else.

**Interference is graded by similarity.** Sparrowhawk learning interferes most with the existing birds,
moderately with the fish, not at all with trees and flowers. This observation is what motivates SWIL; it
is not a separate finding.

## 4. SWIL, and the control that makes it a result

Similarity-weighted interleaved learning: per epoch, 1 presentation of the new sparrowhawk and each known
bird, 0.2 of each known fish, 0 of trees and flowers = 3.4 items/epoch, against 9 for full interleaving.
Result: essentially identical per-epoch learning at 38% of the presentations, i.e. 2.5x fewer
presentations for the same outcome.

The result is only meaningful because of the **matched-budget control**: uniform interleaving at the same
3.4 items/epoch (0.3 each across all eight) is far less efficient, and the slowdown tracks almost exactly
the reduced exposure to the three birds. Any SWIL work here must replicate that control or it degenerates
into "relevant data helps".

## 5. Porting to sequences — open decisions

The paper has **no temporal structure anywhere**. Carrying this into a `predict_next` sequence benchmark
is an extension, not an application; that is the novelty claim, and also where it can go wrong.

**Proposed route: swap the item-property matrix for an item-successor matrix.** Decompose the transition
statistics over the vocabulary and the apparatus transfers directly — dimensions correspond to branches
of a hierarchical transition grammar, and schema-consistency becomes the fraction of a new sequence's
successor statistics captured by the existing basis. Computable, not declared. Independently motivated
by the successor representation (Stachenfeld et al. 2017, already in the library), which is the same
linear-algebraic object and whose eigenvectors give the grid-like structure.

Open, needs a decision before spec:

1. **Vocabulary is unbalanced.** `data/vocab.json` is fruit 12, animal 7, vehicle 3, tool 2, colour 3,
   number 3. The paper's design is strictly balanced (2/2/2/2) because branch size sets singular value and
   singular value sets learning order. On the current vocab the fruit branch dominates every dimension and
   the differentiation ordering is an artefact of vocabulary construction. Either build a balanced
   vocabulary for this section, or promote imbalance to a manipulated variable (more interesting, bigger
   build).
2. **Successor matrices are not symmetric**, so SVD and eigendecomposition come apart and "one dimension
   per split" may not hold as cleanly as in the symmetric item-property case. Needs a numerical check on a
   toy grammar first.
3. **The schema must be acquired, not declared.** Consistency in the paper is relative to the basis the
   network actually learned. If we define it from generator category labels, a model that never acquired
   the schema is scored as though it had one. The section needs a schema-acquisition phase plus a probe
   confirming acquisition (generalisation to held-out category members) before the consistency conditions
   mean anything.
4. **Ceiling risk.** The 4x5 symbolic chain is already at ceiling for tPC/AHN. A schema-acquisition phase
   built on the same chain will be too. The acquisition phase needs real load.
5. **The extension is ambiguous, and the harness cannot disambiguate it.** Appending the new item to the
   host list means the model sees `... -> wren -> sparrowhawk`. Nothing tells it whether that *extends*
   the learned bird list or is a *new* two-item episode that happens to start with `wren`. There is no
   sequence identifier anywhere in the interface: `fit_sequence` updates per adjacent pair, no arm reads
   `current_t` during training, and the probe cues one item for its successor with no state carried
   across positions. Two consequences:

   - **For every current arm the distinction is behaviourally void.** They are pairwise associators, and
     `wren` is the *last* item of the host list, so it has no stored successor to compete with — both
     readings produce the identical weight update. The protocol choice is therefore a *measurement*
     decision, not a semantic one.
   - **So the extended-sequence protocol does not buy the extension semantics it appears to.** It cannot
     signal "same sequence"; it only retrains five already-known pairs. We pay the rehearsal confound
     (which nulls the `same` interference bucket — see §7) and get nothing semantic in return. Verified
     2026-09-02: focused pair-only training reproduces the paper's local-interference gradient
     (+0.84/+0.16/+0.04 same/sibling/far on Original EP) and the effect is *not* a dose artefact — it is
     unchanged at a 40x smaller update budget.

   The capacity this ambiguity points at — continuation vs. new episode — is worth building, but not
   here: it needs a state-carrying arm, and it belongs beside Sequence disambiguation. Logged as item 18
   of `docs/capacity_coverage_audit.md` §6.

## 6. Caveats to carry into the writeup

- The simulations are deep **linear** networks (following Saxe et al.), chosen for analytic tractability.
  The authors are explicit about caution in extrapolating to nonlinear networks or biological systems.
- Human work implicates **novelty** as well as schema consistency in driving replay, and some evidence
  suggests *less* consistent information is prioritised for replay. "Consistent -> fast to learn" and
  "consistent -> preferentially replayed" are separate claims and the second is contested. Do not build a
  benchmark that assumes the second follows from the first.


---

## 7. Implementation status and open confounds

### Verified

- **The branch rule reproduces the paper's structure.** On the paper's own tree
  (2 birds / 2 fish / 2 trees / 2 flowers, 2 features per node) the SVD gives
  dims 1-2 = the two superordinate groups, dims 3-4 = the mid-level splits,
  dims 5-8 = within-pair contrasts, every one at alignment 1.000, with singular
  values ordered by level (3.74 > 2.45 > 1.41). Asserted in
  `test_one_singular_dimension_per_split`.
- **N-way splits are degenerate, as the paper's N-1 rule implies.** With a 4- or
  6-way leaf split the leaf-level dimensions tie, so the SVD basis inside that
  subspace is arbitrary and no single dimension matches a sibling contrast
  (alignment 0.61-0.95). The structural levels stay exact, and the leaf
  dimensions still *span* the within-category contrasts (residual ~1e-16).
  Read per-dimension alignment only where branching is binary.
- **The consistency ladder is monotone and density-matched:** projection ratios
  1.00 / 0.83 / 0.61-0.74 / 0.32-0.45, stable across seeds, all rungs at the base
  mean of active features. This is model-free, so the independent variable is
  known to exist before any model is run.
- **The guard rails work.** Ceiling detection fired on tPC (perfect recall, every
  rung at criterion on trial 1) and floor detection on EP under an absolute
  criterion. Both caught real design errors rather than being decorative.
- **Phase B does what it was built to do.** EP after schema training shows
  category accuracy 0.91 against item accuracy 0.35 — it knows which category
  comes next but not which item. That is schema generalisation, measured.

### Fixed during implementation

1. **Substitution created mechanical interference.** Putting the new item in an
   existing slot contradicted a transition the base sequence teaches, so every
   rung took the same 1/3 MRR hit regardless of consistency. The new item now
   extends the sequence, as the sparrowhawk is added to the dataset rather than
   swapped in for the sparrow.
2. **A 20-epoch "trial" cannot resolve fast acquisition** — every rung saturated
   on trial 1. One presentation per trial.
3. **An absolute criterion is unreachable for a weak model and trivial for a
   strong one.** Criterion is now a fraction of the model's own post-schema base
   MRR, which is what let EP resolve at all.

### Open confounds — need a design decision

4. **Same-category interference is confounded with rehearsal.** The new sequence
   is the host category's items plus the new item, so training on it re-presents
   the whole host sequence. Measured "interference" on that category comes out
   *negative* (MRR improves). Either shorten the host context to a held-out
   prefix, or drop same-category from the interference readout and report only
   sibling/far.
5. **The `random` rung is easy to retrieve precisely because it is
   inconsistent.** Observed EP trials-to-criterion: 1.0 / 1.7 / 7.3 / 1.3 across
   the ladder — non-monotone, with the *least* consistent rung among the
   fastest. An item orthogonal to everything is trivially discriminable in a
   cued-recall test, so orthogonality helps retrieval even as it hurts
   integration. The paper's dependent variable is integration into a shared
   structured mapping; ours is cued recall of a new sequence, and these come
   apart at the unstructured end of the ladder.

   This is the substantive one. Options: (a) score integration rather than
   retrieval — probe whether the new item inherits its category's *other*
   properties, which is what the paper actually tests when it asks the network
   for the trout's IS and HAS features; (b) keep retrieval but report it against
   interference jointly, since the prediction is that random is fast to retrieve
   *and* maximally destructive; (c) drop rung 4 and treat the
   duplicate/within/across range as the ladder. Option (a) is closest to the
   paper and is probably right, but it needs a generalisation probe that does not
   exist in the suite yet.

### Not built

- Per-node `features_per_node` (the branch-imbalance knob of S5 decision 1).
  Balanced only, for now.
- Time-to-consolidation (S8), the largest missing piece.

### Pipeline registration — done

`schema_consistency` is section 6 of `SYMBOLIC_BENCHMARKS`. Run it alone with

```bash
python bin/run_benchmark.py --model <name> --suite symbolic --benchmarks schema_consistency
```

Two call paths, both supported:

- **Embedded** (via `run_symbolic_pipeline`): metrics merge into the suite's
  `results["metrics"]` under their existing `schema_*` prefix, the plot lands in
  `results/<model>/symbolic/plots/schema_consistency.png`, and the section's own
  record is written beside the suite's as `schema_consistency_metrics.json`.
- **Standalone** (`run_schema_consistency()` called directly): unchanged, writes
  `metrics.json` plus the plot under `results/<model>/schema_consistency/`.

Per-section overrides go through `--benchmark-args` and honour `branching`,
`features_per_node`, `schema_trials`, `new_item_trials`, `n_probe_trials`,
`criterion_frac`, `seed` and `epochs_per_trial`.

**The section deliberately ignores the global `--epochs`.** Its dependent
variable is trials-to-criterion at one presentation per trial; inheriting a
global epoch count re-creates fix 2 above, saturating every rung on trial 1 and
erasing the effect. Change `epochs_per_trial` only per-section, and knowingly.

---

### `duplicate` is reported separately — decided 2026-09-08

The headline correlations (`schema_consistency_speed_corr`, `_auc_corr`) are computed over
`within / across / random` only; `schema_corr_rungs` records this. `duplicate` is kept and
reported on its own as `schema_duplicate_excess_trials` (its trials beyond the other three's
mean) alongside `schema_<rung>_nearest_base_cosine` for every rung.

**Why.** The rung is bit-identical to a base item in the base columns (nearest cosine 0.87; the
next rung is 0.72), so a same-space `predict_next` decoder must separate two near-identical
targets. On Original EP over 5 seeds (`focused`, one epoch per trial) it is the *slowest* rung —
5.0 ± 2.0 trials against 2.2–2.6 for the other three — the only rung with any seed-to-seed spread,
and it alone flips both correlations to the wrong sign. That is a discriminability cost the
paper's architecture never pays (a localist input unit per item, attributes read out, not
identity), not a consistency effect; leaving it in lets the decoder vote on the schema question.
`schema_resolved` now follows the same three rungs, so a spread that exists only in `duplicate`
cannot mark the consistency effect resolved. Plot: dashed, hollow markers, hatched bars.

**What the three-rung result then says on EP** (5 seeds): flat — `within ≈ across ≈ random`
on acquisition, `relative_cost` 0.08–0.09, `auc_corr −0.60 ± 0.10`. Locality of interference
replicates (same > sibling > far on every rung; `random` near-global); the consistency
*ordering* does not. The remaining structured-rung question is whether `random`'s low nearest
cosine (0.29) is itself the confound — the structured-impossible rung (§7 open confounds) is the
test.

## 8. Future work — time-to-consolidation

The section currently measures how fast a schema-consistent item is **acquired**.
The stronger claim in the motivating literature is about how fast it becomes
**consolidated** — stable without further support. In Tse et al. (2007), one-trial
paired associates became hippocampus-independent within 48 hours in schema-trained
rats, against weeks for naive animals in comparable tasks. The manipulated variable
is the same (consistency with prior knowledge); the dependent variable is not.
Acquisition rate and consolidation rate are dissociable, and we currently report
only the first.

What this would add, concretely: after the new item reaches criterion, continue the
stream with *other* sequences and measure how long its trace survives, as a function
of its rung on the consistency ladder. The natural readout is **trials-to-forgetting**
— the number of interposed sequences after which recall of the new item falls below
criterion — reported per rung, so the section yields an acquisition curve and a
retention curve over the same ladder.

The prediction that makes it worth building: consistency should trade the two against
each other in a specific way. An item orthogonal to the schema is fast to *retrieve*
(it is discriminable from everything, see confound 5) but should be fastest to decay,
having no structure to be absorbed into; a schema-consistent item should be slower to
discriminate but far more durable. If that holds, it dissolves confound 5 rather than
working around it — the non-monotone acquisition ladder becomes interpretable once the
retention ladder is placed beside it, because the two axes are measuring different
halves of what "integration" means.

Prerequisites, in order:

1. Settle confound 5 (S7). The retention readout inherits whatever the acquisition
   readout is scored on, so a retrieval-vs-integration decision made here propagates.
2. A generalisation probe — asking whether the new item inherits its category's other
   transitions, not just whether it is recalled. This is option (a) of confound 5 and
   does not exist in the suite yet; it is the single component both open problems need.
3. Reuse of the interposed-sequence machinery from `multiple_sequences` and
   `online_continual` rather than a third implementation of the same stream.

Until this exists, the one-shot capacity in the results intro should promise
acquisition rate as a function of consistency, and nothing about consolidation time.
See `docs/results_intro_capacities.md`.
