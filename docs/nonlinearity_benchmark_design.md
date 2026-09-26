# Non-linearity benchmarks — design note

**Status:** audit complete (S1–S2, measured 2026-08-20); linear oracle specified but not
implemented (S3); benchmarks proposed, none implemented (S5). Four of the proposals already
exist as stubs under `dashboard/docs/benchmarks/` (`context_gating`, `high_order_markov`,
`delayed_recall`, `limit_cycle`); this note supersedes them by adding the audit that motivates
the family split and the oracle that certifies any of them.

**Motivating observation:** none of the shipped sections can dissociate a model with a learned
hidden layer from one without. The claim "EP's hidden layer buys conjunctive coding" is
currently unsupported by any benchmark in the suite.

---

## 1. The audit: is non-linearity tested anywhere?

Tested exactly rather than by inspection. A one-step linear map satisfying `W·xᵢ = yᵢ` for
every transition in a task exists **iff**

```
rank([X | Y]) == rank(X)
```

where `X` stacks the inputs `xₜ` and `Y` the targets `xₜ₊₁`. This is a rank identity, not a
regression — it is free of learning rate, epochs, regularisation and conditioning, so it
answers "is this task linearly solvable in principle" rather than "did this model solve it".

Run against the real generators:

| Task | rank(X) | rank([X\|Y]) | Verdict |
|---|---|---|---|
| `multiple_sequences`: 2 lists × 7 distinct words | 12 | 12 | **Linearly solvable** (LSQ residual 1e-30) |
| Same lists with one aliased item (`pear` twice, different successors) | 4 | 5 | Not linearly solvable |
| `tmaze_disambiguation`, place + odour | 16 | 23 | Not solvable — but see §2 |
| `tmaze_disambiguation`, place code only | 14 | 23 | Not solvable (no cue present at all) |
| T-maze, **single route** (control) | 11 | 14 | Not solvable — **and this is the trap, see §3** |

Section-by-section:

| Section | Why it is linearly solvable |
|---|---|
| `presentation_duration`, `sequence_length`, `multiple_sequences`, `noise_invariance`, `letter_noise`, `semantic_similarity`, `online_convergence`, `isi_tolerance`, `online_continual` | Every item in every list is distinct, so the transition set is a bijection. No cue ever demands two successors. These sections vary *load*, *noise* and *overlap* — all quantitative axes along a linearly solvable problem. |
| `tmaze_completion`, `spatial_sequence` | One route, one non-self-intersecting trajectory. Nothing to disambiguate. |
| `tmaze_reversal` | Overwriting an association. A linear map does this fine; the interesting axis is plasticity, not expressivity. |
| `tmaze_disambiguation` | The only section with genuine conjunctive structure. §2. |

**Elementwise nonlinearities do not count and the writeup must say so.** `TemporalPCNetwork`
predicts `W_r·f(x)` and `AsymmetricHopfieldNetwork` predicts `ReLU(W·x)`. Both are additive
across input blocks — `W·[p; o] = W_p·p + W_o·o` — so neither can form a cross-term between the
place block and the odour block no matter how the elementwise squashing is tuned. Only a learned
hidden layer (EqProp, `MultilayerTemporalPCNetwork`, GPT-2) can. "Has a nonlinearity" is not the
same property as "can represent a conjunction", and the suite should never use the first phrase.

## 2. The T-maze case: real conjunction, metric that hides it

The structure is genuinely non-linearly separable. Both routes share an identical stem place
code and differ only in the odour, so a single transition matrix would need

- `W_o·(o_A − o_B) = 0` on every stem step (both routes predict the same next place), and
- `W_o·(o_A − o_B) = left − right` at the branch step.

Inconsistent by construction. Adding explicit `p ⊗ o` features confirms the missing term is
exactly multiplicative — branch residual drops **0.362 → 0.031** (10×) while stem residual is
unchanged.

**But the metric does not see it.** `branch_accuracy` is nearest-of-two-arms on the decoded
suffix, so any correctly-signed lateral bias scores 1.0. Measured on the shipped configuration:

| Predictor | Branch accuracy | Distance to correct arm | to wrong arm | Margin |
|---|---|---|---|---|
| Optimal **linear** map | 1.00 | 0.0886 | 0.1113 | **+0.023** |
| Optimal **conjunctive** map | 1.00 | 0.0212 | 0.1761 | **+0.155** |

A 7× difference in margin, collapsed to an identical score. The committed results show the
ordering actually **inverted** against expressivity:

| Arm | `tmaze_disamb_full_branch_acc` |
|---|---|
| `AsymmetricHopfieldNetwork` (linear + ReLU readout) | **1.0** |
| `OriginalEqPropSequenceNetwork` (hidden layer) | **0.5** |
| `EqPropSequenceNetwork`, `DGEqPropSequenceNetwork` | 0.5 |

Two configuration findings from the same audit:

- `spatial_pipeline.py:384` sets `odour_scale=2.0`, commented *"odour weighted x2 so the cue is
  decisive for a linear map"*. The section is deliberately tuned **toward** linear solvability.
  That is a defensible choice for a disambiguation section, but it must not then be read as
  evidence about conjunctive coding.
- For a converged linear map that scale is a no-op — `W` absorbs any per-block rescaling, and the
  probe returns byte-identical residuals at ×1 and ×2. It only changes effective learning rate and
  saturation for gradient learners. The comment overstates what the knob does.

Both findings are consistent with the existing note that the odour is present on every stem step
(`docs`-level: the section scores concurrent binding, not maintenance over a delay).

## 3. The linear oracle

The single most useful thing to build, and the piece that would have caught §2. Two components.

### 3.1 Design-time certification

Before a benchmark is registered in `SPATIAL_BENCHMARKS` / `SYMBOLIC_BENCHMARKS`, its generator
output is run through the rank identity of §1. A task advertised as testing non-linearity that
returns `rank([X|Y]) == rank(X)` is not testing non-linearity, and the registration should fail
loudly — the same posture `resolve_benchmarks` already takes toward unknown benchmark names.

```python
def certify_nonlinear(X, Y, tol=None):
    """Return (is_nonlinear, rank_X, rank_XY). A linear W with W x_i = y_i exists
    iff rank([X|Y]) == rank(X)."""
    rx = np.linalg.matrix_rank(X, tol=tol)
    rxy = np.linalg.matrix_rank(np.hstack([X, Y]), tol=tol)
    return rxy > rx, rx, rxy
```

### 3.2 Run-time metric: the linear-oracle gap

Fit the optimal one-step linear map on the same transitions the model was trained on, score it
through **the same metric and the same rollout protocol** as the model, and report

```
linear_oracle_gap = model_score − linear_oracle_score
```

A gap near zero means the task is being solved linearly, whatever the model's architecture —
which is the actual result for `tmaze_disambiguation` today (gap = −0.5: the linear oracle
*beats* the EP arms). Reporting the oracle score alongside every headline number costs one extra
`lstsq` per section and makes every claim about expressivity falsifiable.

Two variants are worth reporting together, because their difference localises the non-linearity:

| Oracle | Features | Answers |
|---|---|---|
| `linear` | `[x]` | Is any non-linearity needed at all? |
| `conjunctive` | `[x ; block_i ⊗ block_j]` | Is the missing term specifically multiplicative? |

### 3.3 Caveats that must ship with the oracle

1. **Use SVD-based `lstsq`, not normal equations.** `solve(XᵀX + λI, XᵀY)` on rank-deficient
   place-cell data produces residuals that look like task non-linearity and are conditioning
   artefacts. This bit during the audit.
2. **Never quote an absolute residual as evidence in the spatial suite.** Place-cell codes along
   a smooth trajectory are rank-deficient — a *single* route is already not exactly linearly
   fittable (rank 11 for 14 transitions), because neighbouring codes are linear combinations of
   one another. That is representational degeneracy, not task structure. Only the **gap** between
   the linear and conjunctive oracles is interpretable, and only against the single-route control.
3. **Retro-report for the existing sections.** The oracle is cheap enough to run on everything
   already implemented; the §1 table is the expected output and belongs in the paper as a
   statement of what the suite does and does not currently discriminate.

## 4. Two families, and why the split determines cost

`dashboard/docs/benchmarks/` already names the families. The audit adds why the choice matters
for build order:

| Family | Requirement | Runs under the current harness? |
|---|---|---|
| **Concurrent non-linearity** — the gate is co-presented with the cue | A learned hidden layer, to form a cross-term | **Yes.** `predict_next` is a pure function of the cue and that is exactly enough. |
| **Requires recurrence** — the disambiguating information is in the past | Persistent state across steps | **No.** `measure_recall_associative` probes each position independently with a noisy cue, and every arm's `predict_next` is pure by contract. A history-dependent task scores every current arm at chance for a protocol reason, not a model reason. |

So the first family isolates expressivity on machinery that already exists. The second confounds
expressivity with a harness limitation until a state-carrying ingestion protocol is added
(prefix ingested through a state-updating call, not through `predict_next`). **Build the first
family first**, or the results are uninterpretable.

## 5. Proposed benchmarks

Ordered by build cost, which under §4 is also the order of interpretability.

### 5.1 Context-gated transitions (XOR) — concurrent family

**Construction.** Successor of item *x* is `f(x)` under context 0 and `g(x)` under context 1,
where `f` and `g` are two different permutations of the vocabulary. Sweep the number of gating
contexts and the vocabulary size.

**Why it breaks linearity.** The same cue demands two successors and the gate offset `W_c·c` is
shared across all items, so no single linear map fits for ≥2 items. No memory is required — the
gate is present at the moment of the decision.

**Implementation note.** Deliver the gate as extra **input dimensions**, in the manner of the
existing LEC block, *not* through the `context_data` channel: only 4 of the arms
(`cls_dg_eqprop`, `asymmetric_hopfield`, `dg_eqprop`, `ewc_dg_eqprop`) implement `n_context`,
so the context channel would silently exclude tPC, GPT-2 and `OriginalEqProp` from the
comparison.

**Metrics.** Transition accuracy; linear-oracle gap; accuracy vs number of gating contexts.

### 5.2 Negative patterning and transverse patterning — concurrent family

**Construction.** Negative patterning: `A→x`, `B→x`, compound `AB→y`, with the compound encoded
as the normalised sum of the two embeddings. Transverse patterning: `A>B`, `B>C`, `C>A` — no
item has a consistent value, so only the configuration determines the response.

**Why it is the better citation anchor.** These are *the* canonical non-linearly-separable
discriminations in the animal-learning literature and are hippocampus-dependent by lesion
(Rudy & Sutherland's configural account; Alvarado & Rudy for transverse patterning). XOR is the
same computation with no biological referent. A section grounded in negative and transverse
patterning connects the expressivity axis to the same literature that grounds the other five
capacities, rather than importing a machine-learning toy.

**Cost.** `SymbolicEncoder` plus superposition; no new model interface.

**Metrics.** Compound-vs-element accuracy; linear-oracle gap; per-rung accuracy across
{element, compound, transverse triad}.

### 5.3 Delayed-conjunction T-maze — crosses both families

**Construction.** One knob on `TMazeDisambiguationGenerator`: gate the odour to the first *k*
stem steps (`odour_duration`, the inverse of the existing `odour_on_arms`). Sweep *k* from full
stem down to the first step only.

**Why.** At `k = stem_end` this reduces exactly to the current section; as *k* shrinks the stem
becomes a genuine delay and the task becomes conjunction *over* that delay. It is the cheapest
way to place the existing section on a continuum rather than replacing it, and the prediction is
sharp: every current arm falls to chance as soon as `k < stem_end`, because none carries state.

**Also fix the metric.** Report branch **margin** (distance to wrong arm minus distance to
correct arm, normalised by arm separation) alongside branch accuracy, so a thin linear bias no
longer scores identically to a conjunctive solution — the failure documented in §2.

### 5.4 Aliased chain / high-order Markov — recurrence family

**Construction.** `A B C B D`: the same observable item requires different successors depending
on history. Sweep Markov order *k* and the fraction of aliased states. Certified by §1
(rank 4 vs 5 for the minimal case).

**Blocked on the protocol.** Needs the state-carrying ingestion path of §4. Worth specifying now
and building after 5.1–5.3, since it is the symbolic twin of splitter-cell disambiguation and
the natural partner to 5.3.

### 5.5 Cyclic free-running — recurrence family

Kept as specified in `dashboard/docs/benchmarks/limit_cycle.md`. Note for the oracle: a linear
map cannot hold a limit cycle except with eigenvalues exactly on the unit circle, so the
linear-oracle gap is well defined here and the oracle is expected to fail outright rather than
score thinly — a useful contrast with §2, where it passes for the wrong reason.

## 6. Traps

- **Transitive inference is not a non-linearity test.** It reads as one and is
  hippocampus-dependent, but `A>B, B>C ⊢ A>C` is solvable by a linear value model. Adding it
  would reproduce exactly the situation §1 documents: a section that looks like it discriminates
  expressivity and does not. If it is added, it should be added as a *generalisation* section
  with the oracle reported, and never as evidence about hidden layers.
- **A metric can hide a certified non-linearity.** §1 certification is necessary, not
  sufficient — `tmaze_disambiguation` passes certification and still fails to discriminate.
  Certification gates the *task*; the oracle gap gates the *metric*. Both are needed.
- **Do not describe elementwise activations as expressivity.** §1, final paragraph.

## 7. Reproduction

The audit numbers in §1–§2 come from three throwaway probes (rank identity, min-norm `lstsq`
residual split by stem/branch/arm, and a conjunctive-feature control). They should be folded into
the oracle utility of §3 rather than kept as scripts, at which point every number in this note
becomes a regression test.
