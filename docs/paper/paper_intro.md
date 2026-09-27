# MemVal: Can Your Hippocampus Model Actually Remember?

## Introduction (draft)

> **Status:** first draft. Citations are placeholders — verify against the
> reference library before submission. Anchors used below: Buzsáki & Tingley
> 2018 (*Space and Time: The Hippocampus as a Sequence Generator*, Trends Cogn
> Sci); Tulving 1972 (episodic memory); McCloskey & Cohen 1989 / French 1999
> (catastrophic forgetting); McClelland, McNaughton & O'Reilly 1995 (CLS);
> Graybiel (striatal procedural sequences); Lopez-Paz & Ranzato 2017 (GEM;
> BWT/FWT metrics); Chaudhry et al. 2018 (forgetting measure); van de Ven &
> Tolias 2019 (*Three scenarios for continual learning*); van de Ven,
> Tuytelaars & Tolias 2022 (*Three types of incremental learning*, Nat Mach
> Intell — **verify venue/year**); Rebuffi et al. 2017 (iCaRL); Hahnloser,
> Kozhevnikov & Fee 2002 (*Nature* — ultra-sparse code / songbird HVC sequences).

---

Episodic memory — the capacity to encode and later relive specific past
experiences [Tulving 1972] — is fundamentally *sequential*. An episode is not a
bag of features but an ordered trajectory of events unfolding in time: a route
through a building, a conversation, the steps of a recipe. Recalling one is
inherently a matter of sequence — retrieving *what came next* from a partial
cue, in the right order. Any account of episodic memory, biological or
artificial, is therefore in large part an account of how ordered sequences are
encoded, stored, and retrieved.

Computational models of sequence memory have proliferated — from attractor
networks and Hopfield-style associative memories to reservoir computers, RNNs,
and energy-based learners such as Equilibrium Propagation [refs]. Yet progress
is hard to assess, for three reasons. **First, catastrophic forgetting** remains
pervasive: models that learn a new sequence overwrite earlier ones [McCloskey &
Cohen 1989; French 1999], and the field lacks a common yardstick for how
gracefully a given model degrades. **Second, the structural properties of
sequences themselves** — length, temporal spacing, sensory noise, and especially
*overlap* between episodes — are rarely varied systematically, even though these
are exactly the dimensions that determine whether a model can disambiguate
similar memories or complete a degraded one. **Third, biological and cognitive
fidelity is often an afterthought**: models are tuned to a single dataset or task
and are rarely asked whether they reproduce the behavioral signatures — one-shot
acquisition, pattern completion, interference profiles — that define episodic
memory in the first place. The result is a literature of models that are
difficult to compare head-to-head, because each is evaluated on its own terms.

What is missing is a *neutral proving ground*: a way to generate sequence data
with controlled structure and to score any model on the same cognitively-
motivated tasks. Such a framework would let us ask sharp, comparative questions —
which models resist interference as episodes overlap? which acquire a novel
sequence in one shot? how does retention scale as more sequences arrive? — and
get answers that are commensurable across architectures.

We ground our framework in the hippocampus, the most thoroughly characterized
sequence-memory system in the brain. Beyond its classical role in episodic
encoding, the hippocampus can be viewed as a general *sequence generator*: its
circuitry produces ordered spatiotemporal trajectories — place-cell sequences,
theta sequences, and offline replay — that recur across spatial, temporal, and
even abstract relational domains [Buzsáki & Tingley 2018]. This makes it a
natural reference for what a sequence-memory model should do. But sequence memory
is not the hippocampus's alone: complementary cortical systems support gradual,
interleaved consolidation [McClelland, McNaughton & O'Reilly 1995], the striatum
supports procedural, habitual sequence learning [Graybiel; refs], and — beyond
the mammalian brain entirely — the songbird premotor nucleus HVC generates the
sparse, precisely timed neural sequences that drive learned song [Hahnloser,
Kozhevnikov & Fee 2002]. Each of these systems solves the sequence problem on
very different terms. We therefore treat the hippocampus as our *reference frame*
rather than a constraint, and define our benchmark at the level of
**computational capacities** — pattern completion, sequence disambiguation, rapid
on-the-fly learning, and continual retention — that any sequence-memory system
must exhibit, however it is implemented. This keeps the benchmark open to
biologically detailed models, abstract associative memories, and standard
deep-learning baselines alike.

In this work we (i) systematically characterize how the structural properties of
sequences — length, inter-event interval, noise, and contextual overlap — shape
the difficulty of episodic recall; (ii) introduce **MemVal**, a synthetic
benchmark that operationalizes this analysis into a reproducible suite, pairing a
controllable multi-modal sequence generator (spatial, temporal, and relational)
with a standardized model interface and a battery of biologically-motivated
evaluations; and (iii) demonstrate the framework's generality by benchmarking a
deliberately diverse set of memory models under a single protocol.

To show that MemVal accommodates fundamentally different classes of model, we
evaluate six representatives chosen to span two design axes — the
**representational substrate** (rate-based vs. spiking) and the **learning
schedule** (how the arm ingests a sequence). The schedule axis is not a binary.
Because a benchmark can present the same sequence either as a whole or one
event at a time, "online" turns out to be three distinct claims, and the arms
separate on which of them they satisfy. At one end, the **asymmetric Hopfield
network** — a single matrix trained by the delta rule, our classical rate-based
baseline — exposes no streaming interface at all; it is offline in the sense of
the API, though its update is itself per-transition. In the middle,
**Equilibrium Propagation** and its **spiking variant** — an energy-based rule
whose error is derived from two settled states, and the same rule with binary
spikes on every channel between neurons — can ingest one event at a time, but
their batch and streamed paths are *different rules*: the algorithm is defined
per example, and it is our sequence adaptation that averages the update across
a sequence's transitions. At the other end, the **theta-phase model**,
**temporal predictive coding**, and a **reservoir with diverse timescales**
stream by construction — their batch path is literally a loop over the
streamed one — and they were chosen because each also gives up one commitment
the first three share: the theta model derives the delta rule from oscillatory
phase rather than subtraction, tPC carries error in dedicated units across a
persistent latent, and the reservoir keeps no plasticity in its dynamics at all
and is the only arm clocked by elapsed time. (For the reservoir, too, "online"
is partly ours: its source paper fits the readout offline, and we train the
same least-squares problem recursively.) None of the six learns from the raw
serial signal — the dwelling stream in which a state persists for many steps
and nothing announces a boundary — which marks the far end of the schedule
axis the roster does not yet reach. That a single benchmark can pose the same
tasks — pattern completion, disambiguation, one-shot acquisition, and continual
retention — to models this heterogeneous, and score them commensurably, is
itself a central result: it is what makes head-to-head comparison across the
sequence-memory literature possible.

---

## Related Work (stub)

### Positioning against incremental-learning scenarios

The continual-learning literature organizes its problems around three canonical
scenarios — **task-incremental (TIL)**, **domain-incremental (DIL)**, and
**class-incremental (CIL)** learning [van de Ven & Tolias 2019; van de Ven,
Tuytelaars & Tolias 2022] — distinguished by whether task identity is supplied at
test time and whether the model must discriminate across all tasks seen so far.
This framework, however, is defined for **classification**: its scenarios are
posed in terms of task-specific output heads and growing sets of classes
[Rebuffi et al. 2017], neither of which has a native analogue in associative
*sequence recall*, where retrieval is cued completion rather than labeling. This
is why MemVal does not inherit the trichotomy directly.

Its central difficulty axis nonetheless transfers. What separates TIL from CIL is
fundamentally **how much retrieval context the model is given**: in
classification that context is the task label; in sequence memory it is the
**retrieval cue**. MemVal realizes the same axis through *cue overlap* rather
than explicit task labels. A cue whose items belong to a single episode supplies
its own context — the easy, TIL-like regime; an ambiguous cue shared across
episodes (our disambiguation task, `A→B→X` vs. `C→B→Y`) forces cross-episode
discrimination — the hard, CIL-like regime; the same structure re-presented under
shifted statistics (added noise, a modality change) mirrors DIL. MemVal thus does
not occupy one scenario but **spans the easy-to-hard spectrum via the cue**,
extending the incremental-learning difficulty axis from classification into
sequence recall.

> **TODO:** flesh out remaining Related Work threads — (i) CF metrics we adopt
> (ACC, BWT, forgetting) and existing benchmarks (Split-CIFAR, CORe50, CLEAR)
> and why they are classification-centric; (ii) biologically plausible sequence
> models (Hopfield/attractor, EqProp, spiking); (iii) recent (2024–2026)
> sequence-native continual-learning work, if any, to situate MemVal's niche.
