# Free recall — design note

> **DEMOTED 2026-09-02.** Temporal contiguity was removed as a promise of the Serial
> order capacity, so the lag-CRP — this note's whole case, and the subject of §3's
> dissociation table — is no longer a target statistic. `binding_ordinal` is now covered
> by cued-recall order errors (`order_error_fraction`, `transposition_locality`,
> `transposition_asymmetry`, `intrusion_rate`; see `docs/capacity_coverage_audit.md`
> sec 5, S-O1/S-O2) at a fraction of the cost. Everything below still stands as a design;
> its unique remainders are the initiation curve and unconstrained-order recall, neither
> of which any paragraph-11 row currently asks for. Read §§1-2 and 4-5 as the record of a
> protocol worth building if contiguity ever returns to scope; disregard §3's framing.

**Status:** design only. Nothing built. Symbolic suite. Unlike the interval section
(docs/interval_encoding_design.md), this needs **no new model interface and no
state-carrying protocol** — rollout, decode, and feedback modes all exist in
`measure_recall_autoregressive`. It is therefore the cheapest buildable binding metric,
and the natural first one to ship.

**Capacity:** Serial order, binding readout — third instrument alongside the cued
transposition gradient and the forward/backward probe (neither built). This is the only
one of the three that yields a *genuine* lag-CRP: the lag-CRP is defined over free
recall, and is not computable from cued next-item probes (see the methods-precision note
at the end).

**Literature anchors.**
- Kahana (1996), Mem Cognit 24:103–109 — introduces the lag-CRP.
- Howard & Kahana (2002), J Math Psychol 46:269–299 — retrieved-context account (TCM):
  gradient from gradual context drift, forward asymmetry from the recalled item's own
  context input being present only during encoding of *subsequent* items.
- Healey, Long & Kahana (2019), Psychon Bull Rev — contiguity as a candidate law:
  ubiquity, scale invariance (survives filled delays; appears across lists), steeper
  contiguity predicts better recall.
- Kahana (2020) — already in the reference set; carries the retrieved-context framing in
  the results intro.

## 1. Protocol

1. **Study** — unchanged: train the list as today.
2. **Initiate without an item cue** — the defining move. Cue with a non-item state:
   pure Gaussian noise, zero vector, or context/LEC block alone with the item block
   zeroed (arms that have one). "Here is the list context, produce what you remember."
3. **Generate** — iterate `predict_next`, decode each output to the nearest vocabulary
   item (record item + decode confidence), feed back under the `quantized` mode, until a
   step budget `T_max` or cycle detection (state revisit within tolerance).
4. **Repeat under noise** — `predict_next` is deterministic, so one rollout is one
   trajectory. Many trials × fresh cue noise = sampling from the basin structure;
   distributional statistics come from the trial ensemble.

## 2. Scoring

- **Recall span** — unique list items produced before the first repetition (parallels
  the existing memory-span metric).
- **Initiation curve** — distribution over which serial position is produced first.
  Primacy/recency analogue; for attractor arms, a direct readout of basin sizes from a
  neutral start. Feeds the retention story too (primacy/recency after interference).
- **Lag-CRP** — over successive novel-item transitions, conditionalised on availability
  with exactly Kahana's formula (divide transition counts at each lag by the number of
  times that lag was available: item not yet recalled, within list bounds). Report the
  two signatures separately: gradient (peak at |lag|=1, monotone falloff) and
  asymmetry ratio (+1 : −1).
- **Recovery after derailment** — when noise knocks the rollout off the chain, the lag
  between the derailment position and the re-entry position. Context-carrying memories
  should resume near where they derailed; pairwise chains re-enter wherever the noisy
  state lands. Computable from the same rollouts.
- **Termination stats** — where and how streams die or cycle.

## 3. Why it discriminates (the point of the section)

Cued probes hand every architecture the same retrieval route; this protocol does not.

| architecture | predicted lag-CRP |
|---|---|
| pure chaining | spike at +1, nothing else (free recall collapses into serial recall) |
| symmetric pairwise associator | ±1 symmetric, no forward bias |
| context-carrying (TCM-like) | graded gradient, forward asymmetry ~2:1 — the human band |

The asymmetry ratio locates each arm on the associative ↔ pure-chain continuum with
human data as the reference band. This is a dissociation no cued section can produce.

## 4. Harness opinions — three declared choices

1. **Response suppression.** Humans do not re-recall; the models will cycle A→B→A.
   Model-side suppression needs a new interface, so: (a) *no intervention* — score only
   transitions before the first repeat; honest, but strong attractors give short streams
   and thin lag statistics; (b) *masked requantization* — `quantized` feedback snaps to
   the nearest **not-yet-recalled** item; a declared harness rule standing in for the
   suppression humans do internally, yielding full-length sequences. Same epistemic
   status as the peak-time-vs-threshold decision in the interval doc. **Report (a) as
   primary, (b) as the labelled extended readout.**
2. **Initiation cue type** is a real degree of freedom (noise / zeros / context-only),
   and arms without a context block can only get the noise variant — the `n_context`
   exclusion precedent applies. Report cue type per arm; never compare across cue types
   silently.
3. **Noise scale conflates with basin geometry.** Transition statistics are
   f(attractor landscape, noise_scale) jointly. Report the lag-CRP at a stated σ and as
   a small sweep showing the gradient shape is stable across σ, not an artefact of one.

## 5. Caveats

- **Not strategic recall.** Human free recall includes rehearsal and deliberate search;
  this is pure retrieval dynamics. One scoping sentence in the methods, not a weakness.
- **Deterministic-plus-noise is not true stochastic sampling.** Transition
  "probabilities" are basin statistics under perturbation. The σ-sweep of §4.3 is the
  control; do not read absolute probabilities off any single noise level.
- **Cycle detection needs a tolerance parameter** (state revisit within ε); declare it
  with the protocol parameters in the methods, next to `T_max`.
- **Future: multi-list free recall** — across-list intrusions and their lag structure
  (contiguity across lists is part of the scale-invariance evidence). Needs the
  retention stream machinery; not part of the first build.

## 6. Placement in the capacity map

One row under Serial order (binding), designed-not-built until implemented:

| Serial order (binding, ordinal) | `free_recall` (symbolic, designed) | noise σ; initiation cue type | recall span; initiation curve; lag-CRP gradient + asymmetry; derailment-recovery lag |

**Methods-precision note:** the cued transposition gradient (from `measure_recall_associative`
error identities) is the *serial-recall analogue* of contiguity, not a lag-CRP; only this
section produces the real thing. Keep the terms distinct in the paper — a Kahana-adjacent
reviewer will check.
