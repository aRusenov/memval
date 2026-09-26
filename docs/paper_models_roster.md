# Methods — The model roster

> Draft prose for the section that introduces the arms. Written 2026-09-04
> against the shipped code and the zoo capacity run
> (`docs/capacity_report_zoo.md`). **Citations follow the placeholder
> convention of `docs/paper_intro.md` — anchors, not verified page
> references.** A verify-list is at the end.
>
> The intro (`docs/paper_intro.md`) frames the roster on two axes,
> substrate × schedule. This section keeps that frame and then says why it is
> not enough: the last three arms sit in the *same* cell of that 2×2 as each
> other, and what separates them is how the error signal is derived and what
> persists between events.

---

## Draft

We chose the arms to walk the reader across a space, not to cover it. Each one
was selected because it removes exactly one assumption the previous one made,
so that a change in the capacity profile can be read against a single design
decision rather than a bundle of them. Two constraints applied to every
candidate and are the only grounds on which we rejected any: the arm must have a
**cue pathway and a target pathway that can be re-pointed to `x_t → x_{t+1}`
without altering its learning rule**, and its output must be **scorable by the
same decoder** as every other arm. Whether the source paper's own demonstration
was sequential was not a criterion — none of these models was published as a
sequence memory, and all of them became one the same way, by feeding `x_t` as
the cue and `x_{t+1}` as the target.

**We begin with the asymmetric Hopfield network** [Sompolinsky & Kanter 1986;
Kleinfeld 1986], a single weight matrix trained by the delta rule to map each
event onto its successor. It is the minimal hetero-associative memory: one
associative matrix, an error signal obtained by explicitly subtracting the
prediction from the target, and no state carried between events. It is also
linear, and its delta rule on near-orthogonal codes converges toward the
least-squares solution — every section in the suite is solvable by it (§X),
which is precisely why it is the reference rather than a competitor. Its profile sets the floor that any richer model has to justify
departing from.

**Its two limitations motivate Equilibrium Propagation** [Scellier & Bengio
2017]. The first is that it is linear: no section in the suite *requires* a
nonlinearity, so to ask whether one helps we have to introduce one. The second
is that its error signal is computed by subtraction, an operation with no
accepted biological implementation. EP replaces both. It adds a plastic hidden
layer and derives the error from the difference between two settled states of
an energy function — a free phase and a nudged phase — so that no unit ever
receives an explicit gradient. What it costs is the finding the retention
mitigations in §Y exist for: an all-plastic hidden layer interferes with itself,
and EP forgets a learned list almost entirely when the next one is learned
(ΔMRR −0.62 against 0.00 for the linear store), while reversing a stale
contingency as readily as anything in the roster.

**Spiking EP** [O'Connor, Gavves & Welling 2019] keeps that rule unchanged and
changes only the channel between neurons: each unit emits binary spikes through
a sigma-delta encoder and its neighbours reconstruct a rate with a leaky
predictive decoder. We include it not as a spiking memory in its own right but
as the controlled counterpart to EP — the authors' real-valued layer runs
through the identical code path, so any difference between the two is
attributable to the spike channel alone. It is a spiking network in the sense
that matters: every signal any neuron sends to any other is a binary spike
train, from the first hop — the input layer's potential is clamped to the event
vector, but what it *transmits* to the hidden layer is its quantiser's output,
exactly as for every other layer. What is real-valued is only the two points
where the model meets the world: the clamp on the input potential and the
readout of the output potential. It therefore does not demonstrate a
spike-train *interface* — that is a population codec between the vector
encoder and the model, a separate tier that is built but whose first arm is
paused (see below).

Everything so far shares two commitments. The error is *computed* — by
subtraction in the first arm, between two phases in the next two — and nothing
persists from one event to the next: every probe is a pure function of its cue.
The remaining arms each give up one of these, and they were chosen because each
gives up a different one.

**The theta-phase model** [Hasselmo, Bodelón & Wyble 2002] gives up computed
error. It takes the asymmetric Hopfield network's exact delta rule and derives
it instead from the phase relationship between three theta-modulated
oscillations: encoding is gated to one half of the cycle, retrieval to the
other, and the sign of plasticity reverses between them. At the paper's optimal
phases the cycle integral *is* the delta rule — we verify that the two arms take
identical weight steps to machine precision — so the pair is a control on the
harness as much as an arm: two provably identical rules must produce identical
profiles, and any disagreement is a defect in the instrument. It did find two
(§Z). What the pair does not yet test is the part of the paper that is about
theta: the modulation depth and the phase offsets are free parameters no section
sweeps.

**Temporal predictive coding** [Tang, Barron & Bogacz 2023] gives up the
two-phase comparison. The error is carried by dedicated units at every step and
each weight update is a local outer product of an error with an activation — one
phase, no nudging. Its authors prove that the single-layer form is the
asymmetric Hopfield network with an implicit whitening of the input covariance,
which makes it a second controlled comparison against the reference; the
multilayer form we run adds a carried latent, the first state that persists
between events. That persistence is declared and probed (the arm implements the
non-learning `observe` step that lets a benchmark prime it), and it is the
reason tPC could be asked the delay question at all.

**The reservoir** [Tanaka et al. 2022; Jaeger et al. 2007] gives up plasticity in
the dynamics altogether. A fixed recurrent network with a spread of
leak timescales carries state; only a linear readout learns, by recursive least
squares. It is the roster's only arm clocked by elapsed time rather than event
ordinal, and the only one that can emit an interval as well as an item — the
two halves of Serial order that no other arm can be asked about. It is also,
empirically, the only arm that holds a withdrawn discriminator across a stretch
of shared observations: tPC declares the same capability and floors on it. That
result is what the state axis was included to find.

A seventh arm — the STDP network of Bush et al. [2010], a Hebbian
contrast case consuming and emitting spike trains through a population codec —
is built and gated but **paused**: it is a reimplementation from the paper's
Methods, the paper ships no code, and the cost of establishing fidelity that
way has proved high. It is not scored here.

Two things the roster does not yet contain are worth naming. No arm learns from
the raw serial signal — the dwelling stream in which each state persists for
many steps and nothing announces a boundary (§W); every arm here consumes the
pre-segmented sequence, including the ones that ingest it one event at a time.
And only one arm has any sense of elapsed time. Both are gaps in the roster,
not in the benchmark: the questions exist, and the reservoir shows the harness
can pose them.

---

## Provenance, stated once

The arms split three ways on how they were built, and the paper should say
which is which rather than "published models":

| arm | source | how it exists here |
|---|---|---|
| asymmetric Hopfield | textbook; [Sompolinsky & Kanter 1986; Kleinfeld 1986] | written from the delta rule; **no code citation in the repo** |
| Equilibrium Propagation | Scellier & Bengio 2017 | NumPy reimplementation, checked against the authors' Theano release (`scratch/ep_repo`) |
| spiking EP | O'Connor et al. 2019 | authors' code **vendored verbatim** (`memval/vendor/spiking_eqprop`, only imports changed) |
| theta phase | Hasselmo et al. 2002 | **from the published equations** (2.2, 2.3, 2.5, 2.7, 2.14); no public implementation exists |
| temporal PC | Tang et al. 2023 | reimplementation matching the authors' reference (`C16Mftang/sequential-memory`) |
| predictive recirculation | Chen, Zhang, Cameron & Sejnowski 2024 | **from the published equations** (Results Eq. 2; STAR Eqs. 5–11); the authors' repo (`yschen13/HCPrediction`) carries no license and is not used. Only the paper's *local* rule is ported, not its BPTT-trained headline model |
| DTS reservoir | Tanaka et al. 2022 | **ported to NumPy** from the reference; configuration matches the paper |
| Bush STDP *(paused)* | Bush et al. 2010 | **from the published Methods**; the paper ships no code — the load that paused it |
| spiking BCPNN | Tully et al. 2016 | **rule reimplemented against the authors' NEST 2.2 synapse module** (transliterated as the test oracle; it no longer builds), **network from the Methods and S1 Appendix**; gains calibrated to the paper's plotted weights, not its text (`docs/bcpnn_spiking_port.md`) |

So "Hasselmo is the exception" is not right: two arms are equation-derived (one
of them now paused for exactly that cost), one is verbatim vendored, and the
rest are reimplementations against a reference.
The cleaner claim is that every arm has a published source and a stated
relationship to that source.

## On "supports transitions" — confirmed, with a distinction

Every arm is hetero-associative *as run*: the harness points each one at
`x_t → x_{t+1}`. That is a property of the protocol, not a selection criterion,
and it would be true of any arm admitted. What differs is whether the direction
is native to the substrate:

| arm | where the asymmetry lives |
|---|---|
| asymmetric Hopfield | the weight matrix itself: `W = Σ x_{t+1} x_tᵀ` |
| theta | CA3 → CA1 with `a_CA3 = x_t`, `a_EC = x_{t+1}` — the paper's own mapping |
| temporal PC | the model *is* a next-step predictor: `x̂_t = W_r f(x_{t-1})` |
| predictive recirculation | the model *is* a next-step predictor: `h_t = tanh(W h_{t-1} + U x_{t-1})`, `x̂_t = V h_t`, scored against `x_t` |
| Bush STDP | the causal asymmetry of the STDP window: across-item pairs potentiate, within-item pairs net-depress |
| spiking BCPNN | the *shifted* co-activation window: presynaptic NMDA trace slow (150 ms), postsynaptic fast (5 ms), so `P_ij` credits a pattern's successor and not its predecessor; swapping the two time constants reverses recall (paper Fig 4G) |
| Equilibrium Propagation | **not native** — the energy is symmetric; direction comes from clamping `x_t` at input and nudging `x_{t+1}` at output |
| spiking EP | as EP |
| DTS reservoir | **not native** — a fixed reservoir; the readout is trained to predict the next item |

The selection criterion that actually did the work is the first paragraph's:
a cue pathway and a target pathway that can be re-pointed without touching the
rule. All seven pass it. Cutsuridis et al. 2009 was rejected on cost and on the
spike-interface gap, not on "not sequence-native" — that reasoning was tried and
withdrawn, because it would have excluded EP.

## On "offline" vs "online" — three categories, and they already exist in code

The intuition "EP batches by design; theta, tPC and the reservoir see one item
at a time" is right, but the two-way split it suggests fails on the reference
arm. The registered `AsymmetricHopfieldNetwork` does **not** batch: its
`fit_sequence` is a per-transition delta-rule loop, taking one step per
`(x_t, x_{t+1})` in order. It is "offline" only in that it exposes no `fit_event`
— there is no streaming *interface*, not a batched *rule*. "Batches by default ⇒
offline" would classify it as online, against the intro's use of it as the
classical offline baseline.

What survives is three-way, and it is already declared per arm
(`OnlineTrainable` × `online_equivalent`, `memval/models/capabilities.py`):

| | arms | batch path | streamed path |
|---|---|---|---|
| **no streaming interface** | AHN | per-transition LMS, in order | — |
| **streams, but the batch path is a different rule** | EP, spiking EP | one gradient-*averaged* update per epoch over all transitions | one update per transition |
| **streams, and the batch path *is* the streamed rule** | theta, tPC, predictive recirculation, DTS-ESN, spiking BCPNN | a `fit_event` loop, byte-identical to streaming | same |

The middle row is what "batched by design" actually means, and it is a fact
about **our sequence adaptation, not about the algorithm**. Equilibrium
Propagation is defined per example: the paper states the rule as "for each
training example (x, d) in the dataset … update each synapse", and uses
minibatches of 20 "for efficiency of the experiments" [Scellier & Bengio
2017]. In both reference implementations the energy is separable across the
batch — each example relaxes independently and the update is the mean of the
per-example updates — so no phase relaxes "after a batch"; batching is a
gradient-noise and efficiency choice with no dynamical role. The spiking
variant inherits this unchanged and its paper never mentions batching at all.
What our adaptation does is treat the L−1 transitions of *one sequence* as a
minibatch (`fit_sequence`) or as L−1 separate examples (`fit_event`) — both are
legitimate EP, neither is the papers' minibatch-of-20 regime, and they differ
only in gradient averaging and step magnitude (`online_equivalent = False`).
The bottom row is the one where "online" is a property of the rule rather than
of the API — with one qualification that is the mirror image of the EP case.
For **theta** and **tPC** the per-transition update is the algorithm as
published: Hasselmo's cycle integral is defined per transition, and Tang et al.
state that "the weight parameter W is updated at each step" — their reference
code steps the optimiser inside the timestep loop, on one sequence, and where it
uses a batch it is a batch of *parallel sequences*, never an accumulation across
time. For the **reservoir**, the dynamics are online by construction — the
state advances as each input arrives — but the *readout* in Tanaka et al. is
trained by a single ridge-regression solve over the collected states of a
training phase [their Eq. 6], an offline least-squares fit. Our port trains the
same readout by recursive least squares — the recursive form of the same
regularised least-squares problem, updated one sample at a time — which is
what makes `fit_event` primary for this arm and `fit_sequence` a loop over it. So EP's "batched" and the reservoir's "online" are both, in part,
decisions of the adaptation: EP's per-example algorithm batched across a
sequence's transitions, and the reservoir's offline readout replaced by its
recursive form. The prose should own both rather than present either as a
property of the source paper.

This has a consequence for the numbers that should be stated wherever they are
compared. Every score in the spatial and symbolic suites, for every arm, comes
through `fit_sequence` (`ingest` defaults to batch; only the streamed suite
passes `regime="streamed"`). So EP's and spiking EP's main-suite scores are
from the **averaged** rule, while theta's, tPC's and the reservoir's are from
the streamed rule whichever suite ran them. A batch-vs-streamed contrast on the
EP arms confounds ingestion with a change of rule; on the bottom-row arms it
measures ingestion alone.

None of the seven is **true online** — none learns from the raw dwelling stream
in which a state persists for many steps and nothing announces a boundary
(`docs/true_online_learning.md`). That is a fourth thing, and the roster does
not have it.

## Verify-list

- Sompolinsky & Kanter 1986, *Phys. Rev. Lett.* 57:2861; Kleinfeld 1986, *PNAS*
  83:9469 — the asymmetric-Hopfield sequence citation. **The repo carries no
  citation for AHN at all**; these are the standard ones and need confirming.
- "Converges toward least squares" — the registered AHN is an LMS loop, *not*
  a pseudoinverse (that is the unregistered `HopfieldSequenceNetwork`). The
  linear-oracle claim in §X should be re-checked against the registered class
  before it is quoted as definitional.
- Hasselmo, Bodelón & Wyble 2002, *Neural Computation* 14(4):793–817 — equation
  numbers as cited in `theta_phase.py`.
- Tang, Barron & Bogacz 2023, NeurIPS — the AHN-with-whitening theorem number.
- Tanaka et al. 2022, *Phys. Rev. Research* 4, L032014; Jaeger et al. 2007,
  *Neural Networks* 20(3):335–352.
- Bush, Philippides, Husbands & O'Shea 2010, *PLOS Comput Biol* 6(7):e1000839.
- O'Connor, Gavves & Welling 2019, AISTATS (PMLR v89) — settling assignment
  `(n_negative, n_positive) = (100, 50)` confirmed from the authors' code, not
  the prose.
- Tang, Barron & Bogacz 2023: "the weight parameter W is updated at each step
  following gradient descent"; Algorithm 1 loops over timesteps μ=1…P. The
  reference `train_multilayer_tPC` steps Adam per timestep on one sequence;
  `sequential_memory.py` batches five *parallel* sequences, still per timestep.
  Our port: plain SGD, one sequence — the Adam deviation is documented in the
  arm.
- Tanaka et al. 2022 Eq. 6: `W_out` by ridge regression over the training
  window, "training phase … testing phase", "only W_out is trainable". **No RLS
  in the paper**; RLS is our port's choice (the arm docstring says so). The
  offline/online contrast is documented in the library the arm's docstring
  names: reservoirpy's `Ridge` is "Tikhonov linear regression",
  `Ŵ_out = YXᵀ(XXᵀ+λI)⁻¹`, with an "offline fitting method" over collected
  states; its `RLS` node "fit[s] the Node in an online fashion". Jaeger et al.
  2007 is cited for the leaky-integrator neuron only.
- Scellier & Bengio 2017 §"Experiments": per-example rule statement, minibatch
  20 "for efficiency", and the *persistent particles* trick — per-example hidden
  state stored across epochs. O'Connor et al. dropped that trick (their paper
  says so) and **so do we**: every settle in both EP arms starts fresh
  (`renew_activations=True`). Fidelity note, not a batching issue.
- Cross-refs §X (linear solvability), §Y (mitigation ladder), §Z (the
  controlled pair's defects — `capacity_report_zoo.md` §1.4, §4.2), §W (true
  online) — to be resolved once the section order is fixed.
