# T-Maze Reversal

**Capacity:** Continual retention — the *opposite* failure from the retention
matrix. Retention asks what survives; reversal asks whether an association that
has become invalid can be given up. A memory that never forgets fails here.

## Protocol

A T-maze whose stem is a perfectly overlapping context. One arm is rewarded,
then that association is invalidated and the opposite arm becomes correct. Two
protocols run independently so the value of an explicit extinction stage is
measured rather than assumed:

- `direct` — acquisition → reversal
- `extinction` — acquisition → extinction → reversal

A **trial is one presentation** (a single `fit_sequence` call), not an epoch.
Every stage runs to `max_trials` rather than stopping at criterion, so learning
curves are complete and equal-length across arms and trials-to-criterion is
derived post hoc.

## Metrics

Per stage: `trials_to_criterion`, `criterion_reached`, and final values of
`arm_accuracy`, `perseveration`, `reward_pred`, `goal_identity_margin`,
`reward_pre_goal`, `anticipation_lead`.

- **`perseveration`** is the headline failure mode — continuing to predict the
  formerly-correct arm.
- **`extinction_reward_drop`** scores extinction on the *reward* channel, not
  the branch: the animal still runs the same arm, so what should change is the
  predicted reward.
- **`anticipation_lead`** is a bio-signature, not a performance score. A model
  that predicts the next state exactly scores 0 by construction; a positive lead
  means the reward representation has smeared backward along the arm, the
  hippocampal signature that motivated measuring anticipation rather than
  injecting it.
- **`savings` / `savings_vs_reversal`** — faster re-acquisition indicates a trace
  that survived extinction, i.e. extinction is not erasure. Compare against the
  *first reversal*, not against acquisition from a blank memory, which saturates
  on trial 1 and floors the comparison.

## Reading it

`reward_gain` is a choice, not a default — sweep it and report the plateau.
`control_dead_channel=1` gives the dead-channel control (block present, always
zero), isolating the effect of the added dimensionality from the effect of the
reward signal. Trials-to-criterion interacts with learning rate, so state
epochs-per-trial on every figure and prefer comparing arms at matched total
exposure.

See `docs/spatial_reversal_design.md`.
