# Probe protocol — what cue a section presents, and how many times

> Companion to `docs/sections/rollout_protocol.md` (what `recall()` feeds back) and
> `docs/background/online_continual_benchmark.md` §1b (ingestion regime). This one is the
> third leg: how the harness *asks*.

## What the 30 noise trials were

Every retrieval probe drew `n_trials=30` Gaussian perturbations of the cue at a
fixed `noise_scale=0.05`, in every section, for every arm. Measured 2026-09-03
on the fruit list (100-dim unit-norm embeddings):

| fact | number |
|---|---|
| noise norm at σ=0.05 | 0.50 |
| cos(noisy cue, clean cue) | 0.894 ± 0.015 |
| cos(clean cue, nearest competitor) | 0.254 |
| theta, well-trained: positions with a graded outcome | **0 / 6** |
| EP, under-trained: positions with a graded outcome | 4 / 6 |
| Spearman(margin, 30-trial P(correct)) over 36 cells | 0.76 |
| sign(margin) agrees with trial-majority verdict | 89% |

So: for an arm that knows the sequence, all 30 draws agree and the trials are
pure cost. For an arm near a boundary they produce a graded fraction — but that
fraction is a Monte-Carlo estimate of the deterministic **margin**, at 30× the
cost, with an arbitrary σ baked in. And because the 30 draws share model,
weights, sequence and encoder, n=30 was pseudo-replication for any claim about
capability. The unit that varies capability is the training seed.

## The protocol, declared per section

`memval/benchmarks/probe.py` — `SECTION_PROBE_PROTOCOL`, resolved by
`resolve_probe(section, arm, n_trials)` and written to results metadata as
`probe_protocols`.

| protocol | sections | cue | repeats |
|---|---|---|---|
| `CLEAN_SINGLE` | every capability section | clean | 1 — unless the arm declares `stochastic_forward`, in which case repeats sample the *model's* noise with the cue held fixed |
| `NOISE_SWEEP` | `noise_invariance` | σ swept 0→1 | `n_trials` per level; P(correct \| σ) is the estimand |
| `MASK_SWEEP` | `cue_masking` | fraction masked, swept | `n_trials` per level |
| `NOISE_TRIALS` | none (legacy) | σ=0.05 | 30 — describes pre-refactor results only |

Every arm in the roster is deterministic at forward today, including the
codec-wrapped spiking arms (the wrapper reseeds its spike encoder on every
call). So `CLEAN_SINGLE` currently resolves to exactly one probe everywhere.

## What "clean" does not remove

- **Rollout drift.** Under `OBSERVATION` rollout every fed-back prediction is
  already off-manifold. That is the arm's own error and is what the
  autoregressive sections measure.
- **Material similarity.** `category_variance`, overlap fraction — properties of
  the material, not the probe.

Only the injected cue perturbation goes.

## The graded readout

Accuracy under a clean cue goes binary for a well-trained arm. That is not lost
information — it moves to `measure_recall_margin`: per position, cos(pred,
target) − best competitor cosine, from one clean probe. Positive ⇔ the clean
decode is correct; magnitude is how decisively; ties count as failures. It is
wired into `multiple_sequences` as `multiple_seq_margin_before/after` and
`delta_margin_forgetting`, alongside the MRR versions, and drawn per position as
the right-hand panel of `multiple_seq_forgetting.png`. Other sections can adopt
it the same way.

`score_capacities.py` also writes `capacity_scorecard_radar_margin.png` and a
`capacities_margin_scored` block: the same profile with Continual retention's
forgetting scored from `delta_margin_forgetting_lists` (as the fraction of margin
retained over list pairs) instead of `delta_mrr_forgetting`. It is an alternate
view, not the published profile.

## Putting a probability back on the y-axis: replicate over material

A clean single probe makes each serial position a 0/1 hit, so a per-position
"recall probability" needs a replication axis. Cue noise was the wrong one (it
measured basin width). Seeds are honest but a no-op for hopfield and theta, whose
`W` starts at zero. **List pairs** work for every arm and are the standard
construction of a serial-position curve: `multiple_sequences` now runs
`n_lists` A→B pairs (default 5, rotated through the categories with ≥ 7 words,
disjoint within a pair), reports `multiple_seq_prob_before/after` and
`delta_prob_forgetting`, and plots P(recalled | clean cue) over pairs on the
left with the margin panel on the right. Pair 0 is the canonical fruit→animal
pair, so every pre-existing scalar keeps its meaning and `n_lists=1` reproduces
the old section. At ceiling the probability axis stays flat and only the margin
panel moves — theta: P 1.00→1.00, margin 0.369→0.363 over five pairs.

**Why it is at ceiling, and the overlap condition.** A linear associator's B step
disturbs A by `lr · err_B · (x_B · x_A)`: orthogonal inputs cannot interfere. The
canonical fruit→animal pair has only ~23% of each B input inside span(A).
`multiple_sequences` therefore also runs an **overlap** condition (default on,
`overlap=off` to skip): B drawn from A's own category (in-span ~0.37) against a
disjoint-category control, both at L=6 with identical budgets and variance, and
writes the model-free `in_span` covariate, per-condition P and margin deltas, and
`multiple_seq_overlap_cost_{prob,margin}` (same-category minus disjoint).
`b_exposure_multiplier` (default 1) scales B's budget for the plasticity-asymmetry
axis. The retired `standalone_overlap_demo.py` (git tag `archive/pre-cleanup`) shows the full dial.

## Sites that had their own copy of the legacy protocol

- `schema_consistency` carried a private `_schema_probe` drawing 20 noisy cues at
  σ=0.05, independent of the rest of the suite. It now takes the section's
  resolved `(n_probe_trials, noise_scale)` from `resolve_probe`.
- `SpatialDisambiguationBenchmark.evaluate` looped `n_trials=20` over a body with
  no randomness in it (the test sequences are built once; the rollout is a
  deterministic function of them). Twenty identical recomputations. It now runs
  once unless the arm declares `stochastic_forward`, and records
  `n_trials_requested` / `n_trials_effective` in its result.

- `ContinualChainBenchmark` and `PairedAssociateBenchmark` each defaulted to
  `n_trials=30, noise_scale=0.05` and were constructed from the pipeline with
  only `n_trials` passed, so their cue noise never went through the shared
  argument. Both now take the section's resolved pair and default to clean.
- `interval_timing` already probed with a clean cue but looped `n_trials=30`
  over it -- thirty identical evaluations. Default is now 1.

## Reading old results files

A results file without `metadata.probe_protocols` was produced under
`NOISE_TRIALS`. Its accuracies for near-ceiling arms are identical to what
`CLEAN_SINGLE` gives; its accuracies for sub-ceiling arms are σ=0.05-basin
fractions and are not comparable with clean numbers.
