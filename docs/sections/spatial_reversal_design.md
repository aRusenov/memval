# Spatial reversal benchmark — design note

**Status:** implemented as the `tmaze_reversal` section of the spatial suite.
**Motivating paper:** Hasselmo, Bodelón & Wyble (2002), *A proposed function for
hippocampal theta rhythm: separate phases of encoding and retrieval enhance
reversal of prior learning*, Neural Computation 14:793–817.

---

## 1. Why a reversal task

MemVal's retention matrix scores how well an old memory is **protected**. Reversal
scores how fast an invalid association is **overwritten**. These are opposite
failure modes:

| | measures | failure mode |
|---|---|---|
| Retention matrix | protection of old memories | catastrophic forgetting |
| Reversal | overwriting of invalid associations | perseveration |

A model can ace one by failing the other, and nothing in the suite currently
measures the second. `TemporalPCNetwork`, which shows ~zero forgetting across the
whole continual chain, is the obvious suspect: an arm that never overwrites
anything should be slow to reverse.

The intended headline figure is a **retention × reversal scatter**, with the
prediction that the EP mitigation ladder (DG → DG+XdG → EWC) walks the wrong way
along it — that retention was bought by paying plasticity. That reframes the
mitigations as a trade-off curve rather than a leaderboard.

## 2. What Hasselmo actually does (and what we can borrow)

Verified against the paper, §2.3, eq. 2.7–2.9. Four stages: initial learning
(left ↔ food), reversal with erroneous trials, correct post-reversal trials
(right ↔ food), and retrieval at the choice point. During the error trials the
paper represents the absent food reward as an **absence of entorhinal input**,
`a_EC^(e) = 0`.

The zeroing is principled there because EC *is* the reward afferent — no food, no
drive. But the zeroing is not what performs the extinction. With `a_EC = 0`, CA1
activity during the retrieval phase is purely `W·a_CA3` — the stale prediction —
and `h_LTP` is negative at that phase, so

```
ΔW_CA3 = Σ_e ∫ h_LTP(t) · h_CA3(t) · W_CA3 a_CA3^(e) · a_CA3^(e)ᵀ dt      (eq 2.9)
```

subtracts exactly what was retrieved. Net over a theta cycle this is the delta
rule with target zero: `ΔW ∝ (0 − W x)xᵀ`.

Two things carry over to MemVal; one does not.

**Carries over — orthogonal goal representations.** The paper assumes the EC
patterns for left-arm food and right-arm food are orthogonal. We use orthogonal
unit vectors `e₀` and `e₁` in the reward block, which additionally makes *which
goal the model predicts* directly readable by projection.

**Carries over — extinction as prediction mismatch.** During extinction the
exogenous reward channel is zero while the model still predicts reward. That
mismatch is exactly Hasselmo's `W·a_CA3` under `a_EC = 0`.

**Does not carry over — zeroing the target.** MemVal's target is the next
position, which is never absent. Zeroing a trajectory target teaches "predict the
origin", a coordinate, not an absence. Hence the reward channel: it gives the
task a dimension along which absence is representable without corrupting the
spatial prediction problem.

## 3. Scope decision — the reward channel is not retrofitted

The goal/reward channel is added **only to the reversal section**, not to the
whole suite. Reasoning:

- The suite already mixes input formats. `tmaze_completion` is pure place cells;
  `tmaze_disambiguation` runs 400 place cells + 20 odour dims via
  `TMazeDisambiguationGenerator`; `ObjectArenaGenerator` appends one-hot object
  dims. A reward block is the third instance of an existing pattern.
- Comparability binds **within a figure, not across sections**. Every arm inside
  the reversal section sees the same format, so the section is internally valid.
- An all-zero reward block on the other sections is not free: it changes vector
  norms and effective dimensionality, and therefore the EWC Fisher scale (see the
  `SPATIAL_EWC_LAMBDA` note in `spatial_pipeline.py`).

The one place the requirement is real: **both axes of the retention × reversal
scatter must come from the same encoder condition.** State the condition on every
figure — this is the standing caveat from the load sweeps, applied here.

**Dead-channel control.** `reward_gain=0.0` yields a reward block that is present
but always zero. Running the section that way isolates the effect of the added
dimensionality from the effect of the reward signal. Exposed as
`--benchmark-args 'tmaze_reversal:control_dead_channel=1'`; off by default.

## 4. Scope decision — anticipation is measured, not injected

A tempting design is to advance the reward "spike" earlier on each successive
exposure, simulating an acquired anticipatory signal. We do not, for four
reasons:

1. **It hands every arm the answer.** Anticipatory firing ahead of a goal is the
   classic *emergent* result in this literature — place fields expand and shift
   backward against the direction of travel within a few laps (Mehta, Barnes &
   McNaughton 1997, PNAS; Mehta, Quirk & Wilson 2000, Neuron), and the 1997 paper
   frames it as the prediction of temporally asymmetric Hebbian learning, i.e.
   the very rule class our arms implement. It is an output of the plasticity
   rule, not a property of the world.
2. **It makes the input non-stationary across trials**, which for a
   continual-learning benchmark confounds "learned the reversal" with "tracked a
   drifting input distribution".
3. **The schedule is an arbitrary difficulty knob** — how much earlier per trial
   directly sets how hard reversal is, with no principled value.
4. **A moving spike models the phenomenon wrongly anyway.** The canonical
   "response migrates to the earliest predictor" result is dopaminergic RPE
   (Schultz, Dayan & Montague 1997), where at asymptote the response *at reward
   time goes to zero* — it transfers, it does not duplicate.

So the exogenous reward channel is **fixed and stationary** (a box over the goal
zone), and anticipation becomes a dependent variable: `anticipation_lead`, the
number of steps before goal-zone onset at which the model's *predicted* reward
activation crosses threshold. Directly comparable to the Mehta backward shift.

If a learnable anticipatory cue is wanted later, the defensible form is a
**stationary distal cue** present from trial 1 (a landmark at the branch point),
not a moving stimulus. That is standard conditioning design and stays stationary.

## 5. Task definition

`TMazeReversalGenerator` (`memval/generators/t_maze_reversal.py`) wraps
`TMazeGenerator` + `PlaceCellEncoder` and emits, for each arm, a rewarded and an
unrewarded variant of the same trajectory:

```
input = [ place-cell block (n_place_cells) | reward block (n_reward_dims) ]
```

- Trajectories: shared stem, then left or right arm. The stem is the perfectly
  overlapping context — identical place-cell activations for both arms.
- Reward block: zero everywhere except a box over the final `reward_zone_fraction`
  of the arm, where it carries `e₀` (left goal) or `e₁` (right goal), scaled to
  `reward_gain`. The unrewarded variant is the same trajectory with that block
  left at zero — Hasselmo's `a_EC = 0`.
- `balance_modalities=True` normalises the place block to unit norm per timestep
  so the reward block's `reward_gain` is an explicit, documented signal strength
  rather than an accident of place-cell tiling. Place-cell decoding is a
  scale-invariant centre of mass, so this does not distort coordinates.

## 6. Protocol

Three conditions, so the presence of the extinction stage is a *measured
manipulation* rather than a design assumption:

| condition | stages |
|---|---|
| `direct` | acquire(left, rewarded) → reverse(right, rewarded) |
| `extinction` | acquire(left, rewarded) → extinguish(left, **unrewarded**) → reverse(right, rewarded) |
| `extinction_only` | acquire(left, rewarded) → extinguish(left, unrewarded) |

**A trial is one presentation — one `fit_sequence` call.** The internal
epochs-per-presentation is fixed by constructor kwargs and held equal across
arms, because `OriginalEqPropSequenceNetwork.fit_sequence` ignores an `epochs=`
kwarg and reads `self.n_epochs` (only `TemporalPCNetwork` honours the call-level
override). This is the offline/online commensurability caveat and must be stated
on any figure: a "trial" is one exposure to the trajectory, not one weight update.

Every stage runs a **fixed** `max_trials` presentations rather than stopping at
criterion, so learning curves are complete and equal-length across arms and
trials-to-criterion is computed post hoc. Consolidation (`consolidate()`, for the
EWC arms) fires at **stage boundaries only**, matching `_maybe_consolidate` in
`bin/continual_chain_experiment.py`. Consolidating every trial would re-anchor
mid-stage and change the mechanism under test.

**Probes are fully autoregressive on both blocks.** This is the key difference
from `tmaze_disambiguation`, which clamps the odour exogenously throughout the
rollout. An odour is a cue available in advance; a reward is an *outcome*.
Clamping it would hand the model the goal and destroy the anticipation
measurement. Probes never train.

## 7. Metrics

Per trial, from one probe rollout seeded on the stem:

- **`arm_accuracy`** — fraction of arm-phase steps whose decoded position is
  closer to the currently-correct arm than to the other. The branch readout.
- **`perseveration`** — same, for the *old* arm. **Reported, not penalised** (§8).
- **`reward_pred`** — mean projection of the predicted reward block onto the
  stage's goal vector *inside* the goal zone, over `reward_gain`. This is where
  extinction is scored: during extinction the rat still runs the same arm, so
  what should change is the reward prediction, not the branch.
- **`goal_identity_margin`** — `proj(target goal) − proj(other goal)` in the goal
  zone. Separates "predicts a reward" from "predicts the *right* reward",
  independently of where the trajectory went.
- **`reward_pre_goal`** — mean projection over the arm steps *before* the goal
  zone. Threshold-free measure of the anticipatory ramp.
- **`anticipation_lead`** — steps before goal-zone onset at which that pre-goal
  projection first crosses `anticipation_threshold × reward_gain`.

**`anticipation_lead` is a bio-signature, not a performance score.** A model that
predicts the next state exactly scores 0 by construction — the reward genuinely
is not there yet, so predicting it early is, strictly, a prediction error. A
positive lead means the reward representation has smeared backward along the arm,
which is precisely the hippocampal signature (Mehta et al. 1997/2000) that
motivated measuring anticipation rather than injecting it. Read it alongside
`arm_accuracy`, never as a substitute.

That also fixes the threshold: `anticipation_threshold` defaults to **0.1**, not
0.5. Anticipation is a sub-threshold ramp ahead of onset, so a high threshold
measures nothing but the onset itself and reports 0 for every arm. Observed
profile for a delta-rule arm after six presentations (goal onset at step 13):

```
step:  8      9      10     11     12  |  13     14     15
proj: -0.002 -0.032  0.022  0.186  0.395 | 0.576  0.681  0.702
```

— a clean ramp starting two steps early, invisible at a 0.5 threshold.

Aggregated per stage:

- **`trials_to_criterion`** — first trial after which `arm_accuracy ≥ criterion`
  holds for `criterion_window` consecutive trials; `max_trials` if never.
- **`savings`** / **`savings_vs_reversal`** — `trials_to_criterion` for a second
  reversal back to the original arm, against the original acquisition and against
  the first reversal respectively. Faster re-acquisition indicates a surviving
  latent trace. **Prefer `savings_vs_reversal`**: acquisition saturates on trial 1
  (no competitor), so `savings` against it is floored, whereas the first reversal
  faced a competing association just as the re-reversal does. Optional
  (`measure_savings`), off by default — it adds a full stage of compute.
- **`recovery_after_interference`** — arm accuracy and perseveration re-probed
  after `interference_trials` presentations of an unrelated trajectory in another
  part of the arena.

## 8. Why perseveration is reported, not penalised

Extinction is not erasure. Renewal, spontaneous recovery, reinstatement and rapid
reacquisition all show that the original association survives extinction (Bouton
2004, *Learn Mem* 11:485; Physiol Rev 2021, 101:611). A benchmark that scored
"how completely was the old association destroyed" would reward models for doing
something animals demonstrably do not do.

So `trials_to_criterion` is the headline, residual old-arm mass is a descriptor,
and the recovery probes are the interesting measurement.

**Naming honesty:** we do not measure *spontaneous* recovery. Spontaneous
recovery is driven by the passage of time, and a deterministic model with no
state drift would return an identical probe — trivially zero. What we implement
is recovery after **retroactive interference**: a delay filled with unrelated
experience. The metric is named `recovery_after_interference` accordingly, and it
should not be written up as spontaneous recovery.

## 9. Parameters that are choices, not defaults

Three knobs set the task's difficulty and must be swept once and reported, not
tuned to taste:

- **`reward_gain`** — the teaching-signal strength. Currently implicit elsewhere
  in the suite (`bifurcating_route` gives LEC a unit vector against whatever norm
  the thresholded place tiling produces). Here it is explicit. Sweep once, report
  the plateau, same as the `SPATIAL_EWC_LAMBDA` precedent.
- **`reward_zone_fraction`** — the width of the reward box. A single-timestep
  spike is fragile under rollout decoding; a box is robust. Sweep as a robustness
  check, not a tuning knob.
- **`criterion` / `criterion_window`** — trials-to-criterion is sensitive to the
  threshold, so the full learning curve is always returned alongside the scalar.
- **epochs per presentation** (`--benchmark-args tmaze_reversal:epochs=N`,
  default 3) — the single most consequential knob. At 10 every delta-rule arm
  reverses by trial 2 and the measurement disappears into the ceiling; at 3 the
  reversal stage takes ~5 trials and has range. It trades off against learning
  rate, so compare arms at matched *total* exposure and state the value used.

**Acquisition saturates on trial 1, by design.** A blank memory learning its
first association has no competitor, so `arm_accuracy` is at ceiling immediately.
The discriminating measurement is the **reversal** stage, where the old
association is competing; acquisition is there to establish the thing that has to
be overwritten, and to serve as the baseline for `savings`.

Scoring keeps the blocks separate: spatial error is computed on the place block
only, and reward readout on the reward block only. Mixing them would make arms
with strong reward prediction look spatially better.

## 10. Prior art to settle before writing this up

[Ketz, Morkonda & O'Reilly 2013](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1003067)
(PLoS Comput Biol) recasts the theta phases as the minus/plus phases of an
error-driven learning rule in hippocampus and shows it beats Hebbian on capacity.
That is structurally the same trick EP uses — two settling phases, learn from the
contrast — published in a hippocampal setting. It is simultaneously the strongest
support for the EP+theta direction in `docs/proposals/ep_theta_phase_exploration.md` and the
clearest novelty risk. The differentiator has to be the *measurement* — retention
and reversal scored across a common roster — which is what this section provides.

Other successors of the 2002 model, for the related-work section: Kunec, Hasselmo
& Kopell 2005 (spiking CA3 implementation of the same phase separation);
Cutsuridis, Cobb & Graham 2010 (detailed CA1 microcircuit — note that phase
conventions differ between papers by recording site and reference). Experimental
support: Hyman et al. 2003 (peak → LTP, trough → LTD in behaving rats);
Douchamps et al. 2013; Siegle & Wilson 2014 (closed-loop phase-specific
inhibition); Kerrén et al. 2022 (human MEG, competing memories separated by theta
phase).

## 11. The experiment only a theta arm can run

The paper's performance measure reduces to

```
M = cos(φ_LTP − φ_EC) − K − cos(φ_LTP − φ_CA3)                          (eq 2.16)
```

i.e. reversal performance is maximised when LTP peaks in antiphase to CA3
transmission. Sweeping the phase offset of a theta-gated arm against
`trials_to_criterion` on this section tests the paper's actual thesis. No other
arm in the roster can be probed that way. Not implemented here — it needs the
theta arm — but the benchmark is the substrate for it.
