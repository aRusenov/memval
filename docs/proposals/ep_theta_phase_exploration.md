# EP + Theta Phase — Exploration Notes & Handoff

> **Note (2026-09-27):** the `standalone_*.py` scripts cited here were retired
> from the repo; recover any of them with `git show archive/pre-cleanup:<script>`.

> **Purpose:** self-contained handoff so this line of thinking can continue in a
> fresh chat. Captures the core idea, the mechanistic brainstorm, the
> literature/novelty assessment, and prioritized next steps.
>
> **Reads with:** `docs/proposals/true_online_learning.md` (the raw-signal / true-online
> spec this grows out of) and `docs/background/online_continual_benchmark.md` (the current
> theta-*abstracted* benchmark). Project memory: `dg-ep-retention-direction.md`.

---

## 0. Where this came from

The online-continual benchmark abstracts raw serial experience
(`A A A B B B C C C`) into a theta-compressed replay (`A B C | A B C | …`) — see
`true_online_learning.md`. This exploration asks: **what if the model produced
that compression itself, by explicitly modeling the theta rhythm**, removing the
implicit abstraction? Substrate is `OriginalEqPropSequenceNetwork` (Equilibrium
Propagation, energy-based), the project's model of record.

Requirements the user articulated for a "true online" model:
1. **One-shot local attractors** (place-cell-like) from dwelling input.
2. **A global clock** that orders events (theta / phase-precession style) and
   reinforces their weights.
Current gap: we have only local learning rules; nothing that gives rise to these
phenomena explicitly.

---

## 1. The core idea — theta *is* the orchestrator of EP's two phases

The centerpiece. EP already has a two-phase contrastive rule (free / nudge),
currently **hand-orchestrated** (an abstraction). There is a clean correspondence
to Hasselmo's theta encode/retrieve model:

| Theta (Hasselmo) | EP phase |
|---|---|
| **peak** — strong afferent input, high plasticity → **encoding** | **nudge** phase (β>0, clamp toward input/target, learn) |
| **trough** — strong recurrent dynamics, low plasticity → **retrieval** | **free** phase (β=0, settle from internal dynamics, predict) |

**Proposal:** let a global theta phase `φ(t)` drive `β`. `β(t)=β₀·max(0,cos φ)`.
The contrastive weight update then fires once per theta cycle — the two-phase
structure becomes an *emergent consequence of the rhythm*, not a hand-coded
procedure.

**What this single oscillator buys, for free:**
1. **A global clock** (`φ`) that windows and resets the net (inhibition peak
   silences all units each cycle → "activity ceases at boundary" becomes
   emergent, not hand-cued).
2. **De-abstracts the two-phase rule** (removes hand-orchestration).
3. **Change/novelty-gating emerges:** at the trough the recurrent net *retrieves*
   the predicted next state; the following encode phase clamps the *actual*
   input. Match → tiny nudge (nothing to learn); mismatch → big nudge (prediction
   error → learn). Dwelling → ~0 error → no spurious self-transition learning.
   (This is exactly the raw-signal fix from `true_online_learning.md`, now
   emergent.)
4. **Phase precession falls out:** retrieved *future* states fire at the trough,
   ahead of the animal's actual position → they lead in phase = precession; the
   compressed trough sweep = the theta sequence that STDP binds.

---

## 2. Neural-network-level ingredients

On top of the existing `input → DG → hidden → output` EP core:

1. **Global theta variable `φ(t)`** cycling at ω. Two effects from one scalar:
   `β(t)=β₀·max(0,cos φ)` (encode/retrieve gate) and global inhibition
   `I(t)=γ(1+cos φ)·Σs` (windowing + phase reset).
2. **Recurrent hidden↔hidden weights** (not present yet — current loop is
   hidden↔output). CA3-collateral analog that *holds* attractors; fast, one-shot
   Hebbian during the encode phase gated by an eligibility tag (BTSP-style). The
   **DG already supplies the sparse ensemble** (which units) → DG + one-shot
   recurrent Hebbian = one-shot local attractor (subsystem 1, reusing existing
   pattern-separation machinery).
3. **Per-unit adaptation current** (spike-frequency adaptation). Makes the
   retrieval sweep *move*: a visited attractor self-inhibits, so the trajectory
   rolls A→B→C→(silence) instead of locking in the first attractor.
4. **Eligibility traces** on the slow (transition) synapses — decaying pre-tag
   converted to Δw when post fires. The glue: bridges behavioral time and
   bootstraps the whole thing (see §3b).

Named biological correlates (so we build, not reinvent):
- one-shot attractor ↔ **BTSP** (Bittner/Magee) + CA3 one-shot storage (Marr/Rolls)
- global clock ↔ **septal theta pacemaker**
- phase-ordered reinforcement ↔ **phase precession / theta sequences**
  (O'Keefe & Recce; Skaggs)
- encode/retrieve gating ↔ **Hasselmo theta model**

---

## 3. The two hard problems

**(a) Equilibrium vs. traveling activity.** EP settles to a *fixed point*; theta
sequences are *non-equilibrium traveling waves*. Resolution: attractors stay the
equilibria (EP-native — an attractor *is* an energy minimum); the sweep is a fast
tour of a **slowly-tilting landscape** — adaptation raises the current well so the
ball rolls to the next (a heteroclinic chain / chain of quasi-equilibria). EP
trains the slow transition weights; the fast sweep is the readout. **Most likely
to need a hybrid** (EP for slow weights, non-EP fast dynamics for the sweep).

**(b) Bootstrapping chicken-and-egg.** The compressed sweep needs transition
weights to exist, but learning them needs the sweep. **Eligibility traces break
it:** on the first traversal there's no compression, but A's trace persists
(seconds) so when B arrives at behavioral timescale, STDP binds a weak A→B (BTSP).
Once weak A→B exists, later theta cycles retrieve & compress it, and the tight
STDP window strengthens it. **Behavioral-time eligibility bootstraps; theta
compression refines.**

**(c) The real crux — symmetric vs. asymmetric recurrence.** Symmetric recurrent
weights keep a well-defined energy (EP stays valid) but can't encode *directed*
sequences; asymmetric weights give directed sweeps but break the global energy
function. This is the same tension `memval/models/baselines/asymmetric_hopfield.py`
already lives in. Probably the deepest design decision.

---

## 4. Literature & novelty assessment (searched 2026-07-08)

**The specific synthesis — theta rhythm as the clock orchestrating EP's free/nudge
phases, on an energy-based substrate, for continual/sequence memory — was NOT
found. It looks open.** But adjacent work is close enough to require deliberate
positioning:

- **EP on oscillatory substrates:** [Training Coupled Phase Oscillators via EP](https://arxiv.org/html/2402.08579v1)
  — oscillators as *units* (hardware), not a theta *clock* gating encode/retrieve.
  Encouraging that EP works on oscillatory dynamics; different use of "phase."
- **EP ≡ predictive coding ≡ contrastive Hebbian** (formal unification):
  [Backprop at the Infinitesimal Inference Limit](https://arxiv.org/pdf/2206.02629).
  **Key caveat:** because of this equivalence, "EP + theta" is conceptually
  adjacent to "predictive coding + theta," which already exists (below). Novelty
  can NOT rest on the phenomenology.
- **Theta encode/retrieve neuroscience (Hasselmo scheme):**
  [J. Neurosci.](https://www.jneurosci.org/content/33/20/8689),
  [phase-specific manipulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC4384761/)
  — settled biology, not connected to EP.
- **Closest prior art — theta + sequence learning via PC/STDP (NOT EP):**
  [Rapid predictive maps with STDP + theta phase precession](https://www.biorxiv.org/content/10.1101/2022.04.20.488882.full.pdf)
  and [Predictive sequence learning in the hippocampal formation (Neuron 2024)](https://www.cell.com/neuron/fulltext/S0896-6273(24)00371-4).
  A reviewer would cite these.

**Where the novelty must live (EP-specific value over existing theta+PC/STDP work):**
1. **Energy-based / analog-hardware implementability** (EP's selling point).
2. **The clean theta ↔ two-phase identity** as a mechanistic *why* for the two
   theta half-cycles (a "why" the PC work doesn't foreground).
3. **Continual retention / one-shot attractors / catastrophic-forgetting angle**
   — largely ignored by the predictive-map papers (they focus on rapid learning,
   not retention over a task chain). **This is the project's real differentiator.**

Framing to adopt: *"EP gives an energy-based, hardware-plausible, continual-memory
account of theta-phased sequence learning,"* NOT *"we discovered theta helps
sequence learning."*

---

## 5. Next steps (prioritized)

1. **Read the two closest papers closely** to firm up novelty — specifically
   whether either frames the two theta half-cycles as a *contrastive (two-phase)*
   learning rule, which is the heart of the idea:
   - Rapid predictive maps w/ STDP + phase precession (biorxiv 2022, link above)
   - Predictive sequence learning in hippocampal formation (Neuron 2024, link above)
2. **Minimal first prototype** (don't build everything): add just
   **theta-phase-driven `β`** + **per-unit adaptation** + **eligibility trace**
   to `OriginalEqPropSequenceNetwork` (skip explicit recurrent CA3 at first — reuse
   the existing hidden loop). Feed the **raw dwelling stream** `A A A B B B C C C`
   (variable dwell) and check:
   - (a) does a stable attractor form one-shot per state (ensemble stability
     across the dwell)?
   - (b) after a few exposures, does a **compressed retrieval sweep** appear at the
     theta trough (net predicts B before B arrives)?
   - (c) does transition learning emerge with **no hand-cued presentations or
     boundaries**?
   Success = two abstractions removed (hand-orchestrated phases, hand-cued
   boundaries) + emergent change-gating, validated against the theta-abstracted
   harness that was *handed* `A B C | A B C`.
3. **Resolve the symmetric/asymmetric recurrence decision** (§3c) — likely the
   gating design choice; compare against `asymmetric_hopfield.py`.

Open questions:
- How much explicit theta compression is really needed if change-gating +
  eligibility traces already yield correct transition learning? (May matter only
  for long sequences / spatial trajectories.)
- Boundary-detection threshold and its interaction with `category_variance`
  (similar consecutive items → small change signal → under-segmentation).
- Does change-gated plasticity (updates concentrated at transitions) alter the
  forgetting dynamics / EWC `λ` tuning vs. the abstracted harness?
- Transfer to spatial paradigms (`standalone_tmaze.py`, `standalone_random_walk.py`,
  `generators/bifurcating_route.py`) where dwell is variable and transitions are
  one-shot — the setting where the true-online mechanisms are *required*.

---

## 6. Pointers

- Substrate: `memval/models/baselines/original_eqprop.py` (has `fit_event`,
  `_transition_deltas`, `_on_update_end` hook, `reset_context`).
- Benchmarks: `memval/benchmarks/online_symbolic_pipeline.py` (`stream_sequence`),
  `memval/metrics/retention.py` (retention matrix / ACC / BWT; the
  `online_continual_pipeline.py` that used to host them was removed 2026-09-01).
- Related model with the asymmetry tension: `memval/models/baselines/asymmetric_hopfield.py`.
- Companion docs: `docs/background/online_continual_benchmark.md`, `docs/proposals/true_online_learning.md`.
- Project memory: `dg-ep-retention-direction.md`.

---

## 7. Prototypes built (session 2026-07-08/09)

Three standalone files, each a rung up from the last. All use low-dim one-hot
symbol codes (decode = argmax) for legibility.

**`standalone_theta_ep.py` — theta-gated EP core + the symmetric/asymmetric split.**
Single continuous settling loop with `beta(t)=beta0*max(0,cos phi)`; one theta
cycle == one stream step; retrieve (trough) then encode (peak) with state
carrying across the seam. Trained on the fixed-dwell `A A A B B B C C C`.
- v1 (single hidden loop, EP does everything): learned A->B, B->C but dwell |dW|
  never zeroed and C leaked to B (0.44) — the single matrix could not separate
  auto-association from transition. This *is* the section-3c tension, observed.
- v2 (**two populations**, doc section 3c resolved): symmetric EP net made
  *auto-associative* (nudge target = current symbol) stores states; a separate
  asymmetric `W_asym` (delta rule, novelty-gated) holds directed transitions and
  is NOT in the EP energy (so EP stays valid). Result: symmetric boundary update
  -> 0 (stops chasing transitions), novelty gate crisp (0 dwell / 1 boundary),
  A->B / B->C clean, C->B leak gone, autonomous sweep A->B->C. Change-gating is
  emergent from continuity; direction lives entirely in `W_asym`.

**`standalone_theta_online.py` — TRUE ONLINE, single pass (Stage 1).**
Removes the `n_passes` outer loop entirely. Only entry point is
`observe(frame)`, called once per second in stream order; no future access, no
replay. Motivated by: `n_passes` was a stand-in for repeated traversals + replay
*plus* an unbiological gradient-stability crutch. A single pass has no repetition
for EP's contrastive rule, so Stage 1 **replaces it with theta-gated fast
one-shot Hebbian plasticity (BTSP, section 2.2)** for attractor formation and an
**eligibility latch** (last committed location, held across the dwell = the
behavioural-time bridge, section 3b) for transition binding. Change-gating via a
novelty signal (cleaned current vs held attractor). Trained on the variable-dwell
mouse traversal `A A A B B C C C C C C D D D E E E` (dwell 3,2,6,3,3s).
- Result (single pass): 5 onsets at exactly the 5 location changes, novelty 1.0
  at onset / 0.0 through every dwell, 4 transition writes (one per boundary,
  dwell wrote nothing). Transition matrix a perfect super-diagonal A->B->C->D->E
  (all 1.00, zero leaks); autonomous sweep A->B->C->D->E, parks at E. Variable
  dwell handled with no misbinding.
- Design note: EP's contrastive rule is dropped for the single pass (deferred to
  a replay-driven refinement stage); the theta read/write phase scaffold and the
  two-population split are kept. For orthogonal codes `W_auto`'s cleaning is
  near-identity — it becomes load-bearing only under overlap (Stage 2). The
  `frame` builder already supports `overlap>0` so Stage 2 is a one-line change.

**Next: Stage 2 — overlapping place fields.** Set `overlap>0`: each frame is a
blend (current strong, spatial neighbours weak). This is the realistic hard case
and it stresses two things at once: (1) the novelty gate goes *graded* (adjacent
frames are similar -> under-segmentation risk, open question #2); likely needs
the gate computed against the auto-associative attractor (prediction error) and
possibly DG pattern separation. (2) Direction can no longer be read off a single
frame — the instantaneous blend at B is ~symmetric (A behind and C ahead both
weakly on), so `W_asym` must recover direction from temporal order alone. This
is the setting where `W_auto`'s cleaning and the DG machinery earn their place,
and where the graded decaying eligibility trace (vs the Stage-1 latch) may become
necessary.
