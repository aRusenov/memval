# Porting Vieth & Triesch: the roster's spiking x online arm

*Stabilizing sequence learning in stochastic spiking networks with GABA-Modulated STDP*,
Neural Networks 183 (2025). Upstream: `gitmv/GABA_Modulated_STDP_Paper` (MIT),
on PymoNNto.

Selected by the search in `docs/spiking_online_arm_search.md` as the only
complete spiking, online-trained, hetero-associative model with runnable
permissively-licensed code. Built 2026-09-04.

## What was built

| thing | where |
|---|---|
| the arm | `memval/models/baselines/vieth_gaba_stdp.py` (`ViethGabaSTDPNetwork`, `CodecViethNetwork`) |
| equivalence + contract tests (23) | `tests/test_vieth_gaba_stdp.py` |
| recorded upstream trace | `tests/fixtures/vieth_upstream_trace.npz` |
| correctness gate on the paper's own task | `bin/vieth_stdp_gate.py` |
| upstream driver (headless, no Qt) | `scratch/vieth_ref.py` |
| reference clones | `scratch/vieth_repo`, `scratch/pymonnto_repo`, venv `scratch/vieth_venv` |
| registry entry | `vieth_gaba_stdp`, all three modalities |

## Provenance: reimplemented against a reference, and why not vendored

The memory note from the search said "vendorable verbatim". That was wrong, and
the reason is worth recording: PymoNNto's install pulls **PyQt5, paramiko and
scp**. Depending on a GUI toolkit and an SSH stack to run one arm is not
proportionate, so this joins EP, tPC and DTS-ESN in the
*reimplemented-against-a-reference* class rather than the vendored one
(`docs/paper_models_roster.md`).

The reference is genuinely runnable, which is what makes the class honest here
and is more than the EP/tPC ports had:

* the **PyPI package is broken** -- `PymoNNto==3.0.3` ships without
  `NetworkCore.NumPy_Backend` and dies on import. Install from the GitHub
  master clone instead.
* it needs **numpy < 2** (`np.set_string_function`), which in turn needs
  **scipy < 1.14** (`np.long`), and `Behavior_Text_Modules` imports `tiktoken`
  unconditionally even for the character experiment.

## How fidelity was established

Three layers, weakest to strongest.

1. **Rule by rule.** Each rule is checked against upstream's recorded values
   *given upstream's own inputs to it* -- STDP on both matrices, the Sinkhorn
   initialisation, normalisation, intrinsic plasticity, the GABA modulation, the
   membrane drive, the inhibitory running average, the readout, and the spike
   probability. Exact, not statistical.
2. **Whole trajectory.** The port's own iteration is run end to end from
   upstream's recorded starting state with upstream's recorded spike outcomes
   substituted for the two stochastic draws. Every intermediate tracks upstream
   for the length of the fixture. This is the layer that catches a *wiring*
   error -- a correct rule handed the wrong variable -- which no amount of
   rule-by-rule testing can.
3. **Behaviour, on the paper's own task.** `bin/vieth_stdp_gate.py`, below.

The one thing not covered is the threshold draw itself, since the two
implementations use different generators.

### The execution order is load-bearing

PymoNNto sorts behaviours by numeric key **across all groups**, giving
3, 3.1, 10, 12, 20, 30, 40, 41, 50, 51, 60, 70, 80. Three consequences that a
natural-looking reimplementation gets wrong:

* Plasticity (key 41) runs **before** the groups emit (50, 51), so the pair
  learned at iteration *t* is `(t-2, t-1)`, not `(t-1, t)`.
* Excitatory input and inhibition are **delayed one iteration**; the inhibitory
  population, uniquely, reads the **current** iteration's excitatory spikes.
* So the network pipeline is **two iterations deep**: input at *t* reaches the
  excitatory population at *t+1* and its recurrent consequence at *t+2*. Hence
  `cue_lag=2` -- reading at lag 1 returns the cue's own item and would score the
  arm on autoencoding.

## What the model does, and its two failure modes

`ES` (input->exc) binds an item to an assembly *and* is the readout; `EE`
(exc->exc) is the sequence memory. Trained on upstream's character stream at
`n_exc=600`, the port reaches 0.998 assembly selectivity and a transition matrix
whose forward weight is ~1000x the off-sequence mean -- structurally
indistinguishable from upstream's (forward 0.01756 vs 0.01728).

### 1. It is bimodal across seeds, and that is the model

The decisive measurement. On upstream's own task, `n_exc=600`, 20000 steps:

| implementation | reaches the generating mode | score, success | score, failure |
|---|---|---|---|
| upstream, seeds 0-9 | 5 / 10 | 3.99 - 4.13 | 1.72 - 2.07 |
| this port, seeds 0-7 | 3 / 8 | 4.09 - 4.12 | 1.74 - 2.18 |

Nothing lands between the modes. A successful run regenerates its training text
verbatim; a failed one emits noise at the correct firing rate. **Never report a
mean over seeds for this arm, and never a single run.** The gate is built around
the distribution for exactly this reason.

At upstream's *published* size (`n_exc=2400`, 60000 steps) both fail on this
one-sentence grammar -- upstream 1.81, port 1.81-1.86 -- so 600 units is the
operating point, not a reduction for speed. The published hyperparameters were
evolved against a three-sentence text and do not transfer by scaling up.

### 2. Free-running needs a settled homeostat, not just learned weights

The failure that cost the most to diagnose, and the most reusable finding.

After training, the port's weights were already correct -- transplanting
**upstream's** final state into the port reproduced upstream's generation
exactly (rate 0.06633 vs 0.06636, score 4.125 vs 4.127), proving the dynamics
were right. What differed was the intrinsic-plasticity bias: upstream ended
training at `sensitivity` ~= 0, the port at -0.64.

That matters because `ES` supplies most of the excitatory drive while input is
present. Switch the input off and a network whose threshold was tuned *with* it
sits far below firing. Upstream's recovery phase is not a formality: it is where
homeostasis re-tunes the threshold for the input-free regime. Both
implementations need it, and the seed decides whether the network re-settles
into the chain or into rate-correct noise.

## Where the codec meets the model

Upstream's input layer is `Grid(width=10, height=n_chars)` and its readout sums
the reconstruction over `width`. The codec's layout is the same shape --
D features x 2 polarities x `n_per_feature` duplicates -- and
`PopulationSpikeDecoder.counts` sums over exactly that duplicate axis. So the
codec substitutes for upstream's input grid **without changing the readout**;
the arm returns `ES^T . exc_spike` per recall step and the decoder reduces it.
That is upstream's `TextReconstructor` with its argmax removed, per spec S2.4.

Two things to state in any caption:

* **The arm ships a learned linear readout** (`ES`). Unlike Bush, whose output
  is literally spikes in codec space, this arm's output passes through a trained
  map. That is upstream's own design, not our addition, but it carries the same
  caveat as AHN's pseudoinverse fit when comparing against a linear reference.
* **`presentation_steps=1` is upstream's protocol and the default.** The STDP
  window is exactly one iteration wide and autapses are not removed, so
  presenting a 50-step codec window instead makes 49 of every 50 pairs a
  within-item self-pair and buries the transition.

## Sanity check: does it actually do cued and autoregressive recall?

`bin/vieth_stdp_sanity.py`. A 6-item sequence, 400 passes, 5 seeds, scored two
ways: **cued** (probe position *t* independently, ask for *t+1*) and
**autoregressive** (cue item 0, roll the whole sequence forward). Medians over
seeds; chance is 0.17 and the untrained control sits at 0.20 (an untrained
network emits a constant, which lands on exactly one of the five positions).

Trained continuously, as upstream trains:

| substrate | cued | autoregressive |
|---|---|---|
| native one-hot, no codec | **1.00** | **1.00** |
| one-hot through the spike codec | **1.00** | **1.00** |
| `HierarchicalEncoder` through the codec | 0.80 | 0.60 |

**It works.** Perfect cued *and* full-sequence autoregressive recall on disjoint
items, and the codec costs nothing on its own. Every seed succeeds, so the
bimodality seen on upstream's 10-symbol text task does not appear at this scale.
Overlap costs it, as expected: on the hierarchical substrate cued recall holds
up but the rollout degrades, which is the signature of a chain whose links are
individually right but compound.

### Two defects the sanity check found, both now fixed

**1. `recall` fed a real-valued readout back through a boolean cast.** The arm's
output is `ES^T . exc_spike`, a weighted sum, not a spike pattern; feeding it
back as the next cue made every input neuron with any nonzero activation active.
`recall` is now upstream's protocol -- present the prompt, then free-run the
recurrent population and read out each step, feeding nothing back -- and the arm
declares `RolloutMode.LATENT` rather than `OBSERVATION`. This took
autoregressive recall from 0.40 to 1.00. Note `CodecViethNetwork` re-declares
`OBSERVATION`, because `CodecWrappedModel.recall` runs its own R^D rollout and
never calls the inner arm's: two different protocols, declared separately.

**2. Sequence seams destroyed what the network had not finished computing.**
Two independent structural causes, both in `on_event_boundary`:

* The network is `cue_lag` iterations deep, so at the moment the last item is
  presented none of its consequences exist yet. Resetting there cost the last
  two items their input bindings and the last transition. The boundary now
  drains the pipeline with `cue_lag` silent but plastic iterations first
  (0.00 -> 0.40).
* Zeroing the inhibitory running average at every seam removed the competition
  that separates assemblies, so the first items of each pass were learned with
  none. Measured assembly sizes were `[117, 0, 2, 46, 74, 61]` on six items --
  item 1 got no assembly at all -- against `[48, 43, 53, 57, 52, 47]` when it is
  kept. A seam means "no transition forms here", which requires clearing the
  spike history, not the homeostatic state (0.40 -> 1.00 on the best seeds).

### The remaining gap is ingestion, and it does not close with exposure

Same material and exposure, trained through MemVal's fenced `fit_sequence`
instead of continuously:

| substrate | cued | autoregressive |
|---|---|---|
| native one-hot | 0.40 | 0.00 |
| one-hot through the codec | 0.40 | 0.20 |
| `HierarchicalEncoder` | 0.20 | 0.00 |

An exposure sweep from 150 to 9600 iterations does not move it: fenced plateaus
at 0.20-0.40 while continuous reaches 1.00 by 600 iterations. **So for this arm
the ingestion protocol costs more than representational overlap does.** Each
pass restarts the sequence from a cold spike history, and assemblies here are
formed by recurrent competition that a cold start disrupts. That is a genuine
property of the model rather than a harness defect, and it is the kind of result
MemVal exists to produce -- but it must be reported as an ingestion effect, not
folded into a substrate claim.

## First pipeline result

`online_symbolic / isi_tolerance`, 2 trials, `n_epochs=200`: MRR
`[0.0, 0.0, 0.17, 0.17, 0.0]`. A floor -- and the sanity check above **revises
what it should be attributed to**. Three candidates, now ordered by the
evidence:

1. **Fenced ingestion**, which alone takes this arm from 1.00 to 0.40 on
   material it otherwise handles perfectly, and does not recover with exposure.
2. **Exposure** -- 200 passes against the ~20000 iterations the homeostat needed
   on upstream's own task.
3. **Substrate overlap**, which is real (1.00 -> 0.80/0.60 from disjoint to
   hierarchical) but is the *smallest* of the three, not the largest.

The earlier reading of this floor named only overlap and exposure and put
overlap first. On the measured evidence that ordering is wrong. Any section
reporting this arm on the symbolic suites should state the ingestion regime
explicitly, because for this arm it is not a neutral condition.

## Why the forgetting curve looks the way it does

`standalone_vieth_cf.py` (2026-09-04). Start from the sanity configuration that
scores 1.00 on one 6-item list, train a second 6-item list, and walk toward the
pipeline's `multiple_sequences` configuration one factor at a time. Three
separate mechanisms, each pinned with a direct read rather than inferred.

### 1. Forgetting is re-labelling of a shared positional scaffold, not erasure

One-hot material, continuous ingestion, `target_activity = 1/L`, noise-free
probes: A before B **1.00**, A after B **0.00**, B after B **1.00**. And yet:

| read-out after B | value |
|---|---|
| EE weight along A's chain (A's original assemblies) | 0.050 -> 0.053, unchanged |
| A's items' ES binding, top units kept | 98-99% |
| assembly Jaccard, B_t vs A_t (same position) | **0.91-0.95** |
| assembly Jaccard, B_t vs A_u (other positions) | 0.00 |
| ES mass onto assembly_t from B_t vs from A_t | **9x** |
| free-run from an **A0** cue walks | B1 B2 B3 B4 B5 |

B's item *t* recruited exactly A's item-*t* assembly. Under continuous
ingestion B0 arrives after B5, when the recurrent state is predicting A0's
assembly (the wrap-around), so that assembly fires and gets bound to B0, and so
on down the list: B rides A's chain position by position. The chain survives
intact -- cue A0 and the network walks positions 1..5 correctly -- but every
assembly's `ES` column, which is normalised to unit sum, has its input mass
re-assigned to the most recent binding, so the readout names each position
with the newer list. `delta_mrr_forgetting` reads this as total loss. A
position-aware read would show retention of order and loss of label.

**The homeostat is why the two lists share units.** `target_activity = 1/L`
sizes every assembly at `n_exc / L`, so one list uses the entire excitatory
population and the second has nowhere else to go (budget: 12 assemblies x 50
units on 300 units). Under fenced ingestion the seam clears the recurrent
state, so B is placed less deterministically (68% shared instead of 97%) --
but the network is still oversubscribed, and A is still lost.

### 2. A target below the driven rate makes the homeostat drift into silence

Setting `target_activity = 1/(2L)` so both lists fit (assemblies then share
only 10-18% of units, and A keeps 84-90% of its units after B) does not rescue
recall, and *more* training makes it worse: at 1600 passes both lists read
0.00 and cued probes evoke nothing.

| target | driven rate | sensitivity at 200 / 400 / 800 / 1600 passes | probe-time rate at 1600 |
|---|---|---|---|
| 1/L = 0.167 | 0.167, at target | +0.04 / -0.07 / -0.07 / -0.07, settles | 0.34 |
| 1/(2L) = 0.083 | 0.090-0.100, **above target** | +0.12 / +0.02 / -0.23 / **-0.60**, diverging | **0.02** |

A 6-item list drives the network at ~0.09 whatever the target says; when the
target is below that floor, intrinsic plasticity lowers every unit's bias
without bound, and the moment input stops -- the `cue_lag` steps of every probe
-- the network is silent. The registry default `target_activity = 1/15 = 0.067`
is below the driven rate of every short list in the suite, so this drift is
running in every pipeline section. `set_target_activity` exists on the arm and
nothing in the wrapper calls it; the default should be re-sited, but that is a
change to the zoo run's meaning and is left as a recommendation here.

#### Re-siting the target does not rescue two lists (checked 2026-09-04)

Two checks, both in `standalone_vieth_cf.py`. First, a sweep for a target at
which rate matches target and the bias stays bounded, one-hot material,
continuous ingestion, 300 units:

| target | rate @400 / @800 | bias @400 / @800 | A before | A after B | B after | units shared |
|---|---|---|---|---|---|---|
| **1/6 = 1/L** | 0.167 / 0.167 | -0.07 / -0.07, **settled** | 1.00 | 0.00 | 1.00 | 97% |
| 1/7 | 0.166 / 0.166 | -0.16 / -0.65, drifting | 0.60 | 0.00 | 0.00 | 73% |
| 1/8 | 0.153 / 0.150 | -0.14 / -0.61 | 0.00-0.20 | 0.00 | 0.00 | 57-73% |
| 1/10 | 0.121 / 0.115 | -0.03 / -0.35 | 0.80-1.00 | 0.00 | 0.20-1.00 | 29-50% |
| 1/12 = 1/(2L) | 0.098 / 0.096 | +0.02 / -0.23, drifting | 0.60-1.00 | 0.40-0.60 | 0.60 | 16% |

`1/L` is the **only** self-consistent operating point for an L-item list: the
material forces a unit to fire at about 1/L whatever the target says, so every
target below it starts the drift. The one row where A partly survives B,
`1/(2L)`, is a transient of a homeostat that is still diverging; it does not
persist.

Second, a fixed-point calibration under normal intrinsic plasticity -- train,
set the target to the steady-state rate, repeat -- which is what "the driven
rate" operationally means. On one-hot it converges to 1/L from above
(0.167 -> 0.184 -> 0.169 -> 0.167) and walks toward it from the registry's
1/15 (0.067 -> 0.073 -> 0.082 -> 0.097, still rising at four rounds); on the
dense symbolic substrate it converges to **~0.25 with the bias at -1.0**, the
active fraction the codec reports for that material (247 of 1000 neurons). A
first attempt that measured the rate with the bias switched off was wrong: that
regime is sparse (0.02-0.06 on one-hot) and the resulting targets starved
acquisition to 0.00 everywhere. `calibrate_target_activity` now implements the
fixed point.

One row did change, and it is a transient. Calibrating from the registry's
1/15 stops at ~0.10 after four rounds (not yet converged), and at that target
A *survives* B at 400 passes of B -- while B itself fails to acquire. Followed
further:

| passes of B | rate | bias | A recall | B recall | units shared |
|---|---|---|---|---|---|
| 400 | 0.112-0.117 | -0.12 / -0.16 | **1.00 / 0.80** | 0.00 / 0.40 | 21-29% |
| 1600 | 0.106-0.108 | -0.71 / -0.73 | 0.00 | 0.00 | 17-24% |
| 3200 | 0.098-0.109 | -0.90 / -0.93 | 0.00 | 0.00 | 25-30% |

A survived only because B had not been learned yet (nothing to re-label with)
and the bias had not yet fallen far enough to silence the probe. Both arrive
with more exposure. There is no setting of the target at which two 6-item
lists are held.

So re-siting the target changes nothing about the forgetting. At the
self-consistent target the second list necessarily lands on the first list's
assemblies (mechanism 1); at any lower target the homeostat diverges
(mechanism 2). The model has no way to hold a new list on *new* units while
keeping old bindings: the normalised input columns and the homeostat together
assign the whole excitatory population to the first material they see. That is
a property of the model, and it is the result.

#### What the paper does, and does not, say about this (checked 2026-09-04)

Vieth & Triesch, *Stabilizing sequence learning in stochastic spiking networks
with GABA-Modulated STDP*, Neural Networks 183 (2025) 106985. Read from the
DNB deposit copy; the earlier title recorded here was wrong and is corrected.

**They never train sequences one after another.** The character experiment
trains "three simple character encoded sentences in random order" -- 52
characters, 23 unique, 60k training steps, 15k recovery, 5k free -- and
"throughout the training period, the three sentences ... are activated in a
randomized sequence". Every experiment is a single interleaved stream. The
words *forgetting*, *interference*, *overwriting*, *catastrophic* and
*continual* do not occur. The blocked A-then-B protocol that
`multiple_sequences` runs is outside anything the paper tests.

**The activity target is set by the whole stream, not by one sequence.**
The appendix gives h = 0.01923 for the character text (= 1/52, one over all
characters in all three sentences), 0.00709 for tokens, 0.02778 for bars, and
states that h is the only parameter that changes between experiments. That is
upstream's `1 / n_chars(grammar)`, and it is the fixed point measured above:
the self-consistent target is one over the number of items *in the stream*.
Sections 4.5-4.6 say why in the paper's own terms: "the size of the clusters
should be proportional to the frequency of the corresponding patterns they
represent" (their Fig. 6A shows it), and "given that every neuron strives to
attain its target activity h via sensitivity regulation ... N_active = N_E * h"
(their Eq. 14). A list presented on its own occurs 100% of the time and is
allotted the whole network; that is mechanism 1. And Eq. 14 is mechanism 2 read
forwards: h fixes how many units may be active per step, so an h below what
the stream forces has no solution.

**They report a capacity, and name a different limit.** "Our network of 2400
excitatory neurons can learn an input pattern consisting of three sentences
with 52 characters (23 unique), thus requiring around 46 neurons per input
pattern ... C = N / 46 or C = N * 0.0217." Then: "the primary factor reducing
the memory capacity is the random branching points between the sentences" --
the sentences share characters, and at a shared character the free-running
network picks a continuation at random. That is the disambiguation axis, and
it is the limit they discuss. Future work is "deeper recurrent layers" and
"multiple concurrently active patterns"; nothing about adding material later.

**Under their protocol the two lists are held.** A and B interleaved in random
block order, target 1/(2L), continuous, 300 units: A 1.00 / 0.80, B 0.60 /
1.00, assemblies share 6-7% of units, and the bias *settles* (+0.03 / +0.00)
-- so the earlier statement that 1/(2L) is unstable was true only of a stream
containing L items. The self-consistent target is one over the items in the
stream, exactly as the paper sets it. A blocked tail of 400 passes of B alone
then leaves A at 0.60 / 0.80 (assemblies stay disjoint) while B itself falls
to 0.00, because the B-only stream again forces a rate above the target and
the bias starts to fall.

So the two-list result stands as a property of the model under a protocol the
paper does not use: the model stores whatever is in the stream at the frequency
it occurs, and has no mechanism for material that arrives later. Its native
regime is interleaved, and there it does what the paper claims.

### 3. The pipeline's probe noise is amplified into a mostly-active cue -- on sparse material only

`measure_recall_associative` adds N(0, 0.05) to every cue. Through the codec
(200 Hz, 50 steps) a component of magnitude 0.05 fires with p = 0.01 per
step, so `P(any spike in the window) = 0.39`, and the arm's OR-collapse of the
window (`presentation_steps = 1`, `input_mode = "any"`) makes every such
neuron *active*: a clean one-hot cue of 5 neurons becomes 19 with the
pipeline's noise. That alone takes A-before from 1.00 to 0.20-0.40 on one-hot
material. A count-threshold collapse (`>= 6` of 50 steps) restores 1.00 with
the noise present.

It does **not** help on the suite's own `SymbolicEncoder` substrate. There
every item lights 247 of 1000 codec neurons (within-category Jaccard 0.22),
and no threshold gives a stable item-specific set: `>= 3` leaves 36 neurons,
`>= 6` leaves 4, and recall is 0.20 at every setting, noise or no noise. That
is the Bush finding again -- dense unit-norm rows spread magnitude too thin for
a population code -- and it is a substrate property, not a probe artefact.

### What the pipeline number was

`multiple_seq_mrr_before` 0.206 is the staircase's best censored point on
material the arm cannot acquire (mechanism 3), under a target that is drifting
its bias negative (mechanism 2), through fenced passes; `after` 0.000 is what
remains once the second list re-labels whatever positional structure existed
(mechanism 1). The curve is three effects stacked, and the one the metric's
name suggests -- the second list erasing the first -- is the one that is
*not* happening.
