# Delayed Dependency (n-Back) Benchmark

> **Status: Proposed** — a linearity-breaking benchmark in the *requires-recurrence*
> family. Not yet implemented.

## Biological Context
Memory frequently has to bridge a temporal gap: the item that determines the current
response was experienced several steps ago, with unrelated events in between. "Time
cells" and sustained delay-period activity in the hippocampus and prefrontal cortex are
the biological substrate for holding information across such gaps.

## Computational Assay
The target at position $t$ is determined by the item presented $k$ steps earlier — a
copy / delayed-match / **n-back** structure. We sweep the delay $k$, and optionally
fill the intervening positions with distractor items to increase interference within
the gap. Accuracy is measured as a function of delay.

## Why it breaks linearity
The successor is **not a function of the current observable** — it depends on an item
that is no longer present in the input. The information must be actively maintained in
recurrent state across the delay, which a one-step linear map has no mechanism to do.

## Metrics Evaluated
- **Accuracy vs. delay $k$** — the delay-tolerance curve.
- **Max reliable delay** — largest $k$ retrieved above threshold.
- **Distractor sensitivity** — accuracy drop when the gap is filled with distractors
  vs. left blank.
