# Rollout protocol — what `recall()` feeds back

> Companion to `docs/background/online_continual_benchmark.md` §1b, which covers the
> *ingestion* regime. This one covers the *retrieval* regime.
> The third leg — what cue a section presents and how many times — is
> `docs/sections/probe_protocol.md`.

There are **two** autoregressive rollout paths in MemVal, and only one of them
was ever a declared protocol.

## 1. The harness's loop — already declared

`measure_recall_autoregressive` (`benchmarks/symbolic_pipeline.py`) drives
`predict_next` itself and projects the prediction back onto the input manifold
according to an explicit `feedback_mode`:

| mode | fed back | effect |
|---|---|---|
| `raw` | the prediction verbatim | magnitude unconstrained; drifts off the encoder manifold, error compounds |
| `l2` | prediction renormalised to the unit sphere | fixes magnitude drift only |
| `quantized` | the *clean* embedding of the decoded symbol | "drift-free upper bound … hides sub-threshold representational degradation" |

The symbolic suite sweeps all three (`symbolic_pipeline.py`, `feedback_modes`),
and `docs/figures/protocol_dissociation_table.py` reports the dissociation. Good:
the protocol is a condition, stated and varied.

## 2. `model.recall()` — was undeclared

`benchmarks/pattern_completion.py` scores
`model.recall()`, which every arm implements privately. Those private
implementations were **not** the same protocol, and nothing said so.

`rollout_mode` (`memval/models/capabilities.py`) now declares it per arm, and
`tests/test_rollout_mode.py` both enforces the declaration and verifies it
against what the code actually does.

| mode | arms | what happens per step |
|---|---|---|
| `OBSERVATION` | theta, tPC (single-layer), the whole EP family (original / eq_prop / online_eq_prop / dg / dg_xdg / ewc_* / cls_dg / spiking), Hopfield, AHN, kNN-episodic | raw prediction fed straight back — equivalent to `feedback_mode="raw"` |
| `ENCODER` | GPT-2, HiCL | prediction decoded to a symbol, that symbol's **clean codebook embedding** fed back — equivalent to `feedback_mode="quantized"`, hardcoded |
| `LATENT` | `MultilayerTemporalPCNetwork` | latent inferred once from the cue, then run forward generatively; observations read out but never fed back |
| `HYBRID` | DTS-ESN | reservoir state carried forward **and** `decode_prediction(y)` fed back into it — the only arm that honours the `decode_prediction` hook inside its own rollout |

## Why this matters

GPT-2 and HiCL run, inside `recall()`, the protocol the symbolic suite itself
documents as a **drift-free upper bound that hides sub-threshold
representational degradation**. The EP family runs the maximal-drift protocol.
Both are scored by Pattern completion and Sequence disambiguation, under the
same metric, as though the numbers were commensurable. They are not.

`rollout_modes_comparable(a, b)` is the check; `rollout_mode` is now recorded in
each suite's results metadata so a scorecard can surface it.

## Open decision

This change **declares and records** the discrepancy; it deliberately does not
resolve it, because every resolution moves published numbers:

1. **Make `recall()` take a `feedback_mode`**, as the harness loop does, and let
   the benchmark choose — the most consistent option, and the one that turns
   retrieval protocol into a swept condition like ingestion. Changes GPT-2 and
   HiCL scores on Pattern completion and Sequence disambiguation.
2. **Normalise every arm to `OBSERVATION`.** Simplest; also changes those scores.
3. **Keep as-is and report the mode alongside the score**, treating ENCODER arms
   as a separate protocol class rather than a comparable one.

`LATENT` does not collapse into this choice — a latent rollout is not a feedback
mode, it is a different generative story, and for `MultilayerTemporalPCNetwork`
it is the behaviour the source reference specifies. Whatever is decided for
1–3, `LATENT` stays its own class.
