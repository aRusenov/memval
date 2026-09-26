# Porting ca3net as a MemVal arm — implementation plan

**Status:** Phases 0 and 1 complete (2026-09-04), spatial-only scope agreed. Phase 2
(direction control) is next and is the gate that decides whether the arm can be scored at
all. Upstream verified hands-on 2026-08-28 (both probing protocols reproduced end to end;
see the evaluation notes in memory).

| Phase | State | Artefact |
|---|---|---|
| 0 vendor + landmines | **done** | `memval/vendor/ca3net/` (4 modules + `_shims.py`), `scratch/ca3net_repo/`, `brian2` extra in `pyproject.toml` |
| 0 licence | **open** | copyright line still wrong upstream — see below |
| 1 encoder/decoder | **done, gate passed** | `memval/models/baselines/_ca3net_io.py`, `tests/test_ca3net_io.py` (13 tests) |
| 2 direction control | **done — GATE FAILED, decision needed** | `examples/ca3net_direction_probe.py` |
| 3 wrapper | not started | — |
| 4 symbolic | out of scope for v1 | — |
| 5 registration | not started | — |
| 6 validation | not started | — |

**Phase 1 gate result.** Round-trip cosine (encode → decode, no network between), worst case
over 8 `PlaceCellEncoder` items: **0.974 at the 100 ms default**, against a 0.95 gate. The
same gate passes on sparse near-orthogonal vectors, which is the regime a symbolic encoder
would produce — so Phase 4 is not blocked by the adapter. Note the default presentation
window moved from upstream's 25 ms gamma rhythm to 100 ms: 25 ms scores 0.919 and does not
clear the gate. That is a fidelity requirement of *our* encoding, not a change to the model.

**Upstream:** [KaliLab/ca3net](https://github.com/KaliLab/ca3net) — Ecker, Bagi, Vértes,
Steinbach-Németh, Kerekes, Papp, Káli, Freund, Hájos, Gulyás (2022), *eLife* 11:e71850.
Brian2, Python, ~3.8k lines across `scripts/`.

---

## 1. What this arm actually adds — and a framing correction

The motivating phrase was "our only spiking × online model". Two corrections before we build
on that.

**`spiking_eqprop` already exists and already declares `online_symbolic`.** The cell is not
empty. And per `memval/models/capabilities.py`, regime is deliberately *a property of the
protocol, not of the arm* — arms declare `OnlineTrainable` and the suite varies ingestion.
So "the spiking × online arm" is not a slot this codebase actually has.

What ca3net genuinely adds, none of which any current arm has:

1. **Recall is a network dynamical process, not a feedforward map.** Every other arm computes
   `predict_next` as a function evaluation. Here a cue perturbs a recurrent network and a
   travelling wave of activity is the recall. That is a different computational claim, and it
   is what makes the autonomous-generation half of Serial order a native measurement rather
   than an unrolled `predict_next`.
2. **Real metric time.** Learning is STDP over millisecond spike timing, and the network runs
   on a wall clock independent of event arrival. This is the only arm where an inter-event
   gap is *endured* rather than counted. Note `docs/interval_encoding_design.md` currently
   expresses gaps as blank ticks — ca3net satisfies that spec by a genuinely different
   mechanism, which makes it the interesting comparison, not a redundant one.
3. **A different error-signal route.** Adds "spike-timing correlation" to the AHN /
   tPC / EP / theta table.

Position it that way in the paper. "Fills the empty spiking cell" would not survive review of
our own taxonomy.

---

## 2. The enabling decision: treat the PC population as a generic assembly substrate

This is the design choice everything else follows from, and it is what makes the port
tractable at all.

ca3net's 8000 pyramidal cells are *place* cells: upstream assigns each a place field `phi`,
generates spike trains from simulated maze running (`generate_spike_train.py`), and decodes
recall with `bayesian_decoding.py`, which infers a **1-D scalar position** from per-neuron
tuning curves and fits a straight line through the posterior.

**Do not port that.** It ties the arm to 1-D spatial tasks and cannot express the symbolic
suite. Instead:

- Partition the 8000 PCs into `D` disjoint blocks, where `D` is the encoder's feature
  dimension (`PlaceCellEncoder` defaults to 400; symbolic encoders differ).
- **Encode** item vector `v` by driving block `j` as an inhomogeneous Poisson source with
  rate proportional to `v[j]`.
- **Decode** by counting spikes per block in a time window and normalising to a `D`-vector,
  then handing that to the arm's `decode_prediction` / the suite's existing decoder.

Consequences:

- `generate_spike_train.py` is **replaced** by a MemVal-sequence-driven encoder. `stdp.py`'s
  learning and `spw_network.py`'s recall are kept as-is.
- `bayesian_decoding.py`, `detect_replay.py`, `analyse_movement.py` are **not vendored**.
  This also retires my earlier claim that the Bayesian decoder solves the decode problem —
  it solves a narrower one we no longer need.
- The arm becomes modality-agnostic in principle. Whether symbolic actually works is an
  empirical question (§5, Phase 4), not an architectural bar.

Sanity note: the diagonal band that upstream's weight matrix shows is a *consequence* of the
sequence presented during learning, not a prerequisite. Presenting a MemVal chain produces
whatever structure that chain implies.

---

## 3. Scope

**Register `modalities: ["spatial"]` first.** Spatial is where the substrate match is
closest (both sides are sparse, overlapping, positive activations) and where a travelling
wave is the natural readout. Raise `UnsupportedRegime` elsewhere until Phase 4 says
otherwise — per `capabilities.py` the pipeline records that as *not applicable*, never as a
score of 0.

Out of scope for v1: the ripple/gamma oscillation analysis, LFP estimation, the 2nd-environment
scripts, and `bluepyopt` network optimisation.

---

## 4. Interface mapping

`HippocampalModel` requires `fit_sequence`, `predict_next`, `recall`, `get_latent_state`,
`reset_context`. Optional: `decode_prediction`, `OnlineTrainable.fit_event` /
`on_event_boundary`, `StatePrimeable.observe` / `observe_sequence`, `batched_probe`.

| MemVal method | ca3net operation |
|---|---|
| `fit_sequence(seq, epochs)` | Encode `seq` to spike trains, repeat `epochs` traversals, run `stdp.learning()` → weight matrix. Held as the arm's "synapses". |
| `fit_event(evt)` | Keep a **persistent** Brian2 STDP network; inject this event's spikes and advance the clock by the inter-event interval. This is the path that gives metric time. |
| `predict_next(cue)` | Cue the corresponding PC block for 200 ms, run ~1 event duration, decode the block active shortly after cue offset. **Must be pure** — instantiate a fresh recall network per call, or verify independence (§6). |
| `recall(prompt, length)` | The native operation. Cue, run one event, bin the sweep into `length` windows, decode each. |
| `reset_context()` | Rebuild the recall network from the stored weight matrix; membrane potentials and conductances to rest. Must **not** touch weights. |
| `get_latent_state()` | `{"wmx": sparse weights, "last_spikes": (t, i), "rate": pop rate}` |
| `decode_prediction(raw)` | Block-count → normalised vector cleanup. |
| `on_event_boundary()` | Advance the persistent learning network through a gap with no input. |

`observe_sequence` (StatePrimeable) is worth declaring: it maps cleanly onto "run the network
through a prefix without plasticity", and per the criterion-probe memory, the probe must
match the training input regime — a cold-cued ca3net and a primed one will differ.

---

## 5. Phases

### Phase 0 — vendor and licence  *(blocking)*
Follow the `spiking_eqprop` precedent exactly: verbatim upstream at
`memval/vendor/ca3net/`, import lines the only edit, reference clone at
`scratch/ca3net_repo/`, wrapper at `memval/models/baselines/ca3net.py`, demo at
`examples/ca3net_demo.py`.

**Licence caveat to resolve first.** `LICENSE` is verbatim MIT but reads
`Copyright (c) 2015 yaringal` — Yarin Gal, unrelated to the Káli lab, seven years before the
paper. GitHub reports "MIT" on text match alone. This is the *sixth* unlicensed-or-ambiguous
upstream in the pipeline (tPC, Chen, Tadros, DeepSITH, simpleBTSP, ca3net). Worth one
deliberate decision covering all six rather than a sixth ad-hoc one. Cheapest resolution
here: email Káli/Ecker and ask them to correct the copyright line.

**Also pin the three landmines in the vendored copy**, with comments:
- `len_sim = 10000` hardcoded in `helper.py`, `plots.py`, `analyse_movement.py`;
  `slice_high_activity` converts bin indices to ms with it, so any run ≠ 10 s silently
  reports mis-scaled times. Make it a parameter.
- `nPCs = 8000` module-level in `stdp.py` and `spw_network.py`. Make it a parameter.
- `prefs.codegen.target = "numpy"`. Switching to cython took a 10 s run from ~12 min to
  ~20 s; standalone mode ran 400 s of STDP in 21 s. Both matter for viability.

### Phase 1 — encoder / decoder adapter
`memval/models/baselines/_ca3net_io.py`: `encode_events(events, dt) -> (spike_times, ids)`
and `decode_window(spike_times, ids, t0, t1) -> vector`. Pure NumPy, no Brian2, unit-tested
standalone against round-trip fidelity (encode → decode with no network in between should
recover the input to high cosine).

**Gate:** round-trip cosine > 0.95 on spatial items at the chosen block size.

### Phase 2 — direction control  *(decides whether the arm can be scored at all)*
Verified upstream: the **symmetric** STDP kernel gives reverse-dominant sweeps — 6/7 reverse
spontaneous, 8/10 cued. Under `predict_next` a reverse sweep is a wrong answer, so ~⅔ of
events score as failures for reasons that have nothing to do with memory.

`stdp.py` supports `asym` mode (`taup=taum=20 ms`, `Ap=-Am`, `wmax=4e-8`,
`scale_factor=1.27`). Run the §5 Phase-1 harness under `asym` and measure the forward:reverse
ratio.

**Gate:** `asym` yields ≥ 80% forward sweeps. If not, the arm can only be scored with a
direction-agnostic metric, which is a change to the suite, not to the arm — escalate before
proceeding.

#### Result (2026-09-04): GATE FAILED, and `asym` is not the fix

Stimulus: 16 Gaussian bumps marching across the feature dimension, 150 traversals, cued at the
chain **start**. Three seeds per mode, 10 s each.

| | weight asymmetry | usable scale | events (3 seeds) | forward |
|---|---|---|---|---|
| `sym` | **−0.000** | 0.35 | 27 | **20 / 27 = 0.74** |
| `asym` | **+0.273** | none found | **0** at every scale 0.35 → 1.15 | n/a |

**1. The kernel biases the weights exactly as predicted.** Symmetric STDP leaves forward and
backward mass identical (−0.000); asymmetric puts 1.75x more forward (+0.273). The mechanism
is real and measurable in the weight matrix.

**2. But `asym` produces no replay at all** — and the cause is *our* parameter choice, not the
LTD lobe. Population rate rises smoothly with scale (2.6 Hz at 0.70 → 8.0 Hz at 1.15) while
detected events stay at **zero**: activity is elevated but fragmented, never sustaining the
≥260 ms contiguous run an event requires.

My first explanation — that the LTD lobe cancels potentiation under our denser drive — was
wrong. The two modes differ in *time constant* as well as sign: 62.5 ms vs **20 ms**. Phase 1
set the presentation window to 100 ms per item for decode fidelity, with no reference to the
kernel. At that spacing, consecutive items pair with weight

    sym :  exp(-100 / 62.5) = 0.20
    asym:  exp(-100 / 20  ) = 0.0067      (~30x weaker, effectively nothing)

So the `asym` kernel cannot bridge our item spacing and never builds a chain to replay. This is
a **Phase 1 / Phase 2 parameter interaction**, not evidence about LTD.

The real option-2 experiment is therefore to shorten the presentation window toward the kernel
width — upstream's own 25 ms gamma rhythm — and recover the lost decode fidelity by raising
`infield_rate` rather than the window: 25 ms at ~80 Hz carries the same spike count as 100 ms
at 20 Hz, and 80 Hz is within CA3 burst range.

**3. Task geometry matters more than the kernel.** `sym` scores 0.74 forward here against
~0.25 on upstream's own place-cell setup — with an *identical* learning rule. The difference
is that our chain is unidirectional and cued at its start, where upstream's is a bidirectional
track cued at its end. Worth knowing independently of this arm: the suite's task geometry, not
the model, sets most of the direction bias.

#### 10-seed follow-up (option 1, resolved): still failed, and the variance is the story

| | events | pooled forward | per-seed mean ± sd | 95% CI (seed-level) |
|---|---|---|---|---|
| 3 seeds | 27 | 0.74 | — | — |
| **10 seeds** | **103** | **0.718** | **0.691 ± 0.244** | **[0.540, 0.842]** |

Per-seed forward fraction: `0.57 1.00 0.50 0.78 0.70 0.92 0.73 0.27 0.44 1.00`.

The point estimate did not move (0.74 → 0.72), so **more seeds will not rescue the gate** —
option 1 is exhausted. Note the 0.80 threshold does fall inside the seed-level 95% CI, but only
because that interval is enormous, and a threshold you can only meet by widening your error
bars is not met.

The real finding is the **spread**: sd 0.244, with individual seeds running from 0.27 to 1.00.
Direction is close to a per-seed coin flip with strong *within*-run consistency — two seeds
replayed entirely forward, one mostly reverse. That is a bistability of the recurrent dynamics,
not measurement noise, and it means per-arm direction cannot be stabilised by averaging.

**A prerequisite bug found on the way.** `np.tile`-ing the chain for repeated traversals puts
item N back-to-back with item 1, so STDP also learns the wrap-around link and the chain becomes
a closed **loop**: once cued, activity cycles forever and the network shows one run-long event
instead of discrete ones. Before the fix there was *no* discrete-event regime at any weight
scale in either mode. `stage_learn` now inserts `INTER_TRAVERSAL_MS` of silence between
traversals. Any future sequence-presentation code must do the same.

#### What `sym` and `asym` actually are — and which one ships

The rule accumulates two traces:

    on_pre :  w += A_postsyn      # post-before-pre, amplitude Am
    on_post:  w += A_presyn       # pre-before-post, amplitude Ap

| | Ap | Am | tau | behaviour |
|---|---|---|---|---|
| `sym` | +0.004 | **+0.004** | 62.5 ms | **both** orders potentiate — order-blind coincidence detection |
| `asym` | +0.01 | **−0.01** | 20 ms | causal potentiates, anti-causal depresses — classic Hebbian |

**`sym` is the shipped default.** `stdp.py`'s `__main__` falls back to `STDP_mode = "sym"` when
given no argument, and the module docstring records that it was *"updated to produce symmetric
STDP curve as reported in Mishra et al. 2016 (10.1038/ncomms11552)"*.

It is also the empirically grounded one. `files/original_STDP_data.json` carries the note *"The
data was sent by Jose Guzman on 11 July 2017 (P. Jonas and R. Mishra in CC) for the request of
Szabolcs Káli"* along with a Gaussian fit whose width parameter is `gauss_c = 62.243` — which
is exactly where `sym`'s 62.5 ms comes from. A Gaussian is symmetric by construction. The
`asym` numbers are fitted to nothing in this repository.

**Consequence for us:** switching to `asym` means abandoning the model's fit to CA3–CA3
recurrent-synapse data in order to gain a scoring convenience. Reverse replay is not a defect
of this model — it is a reproduced experimental phenomenon and a headline result of the paper.

#### Options, in the order I would now try them

1. **More seeds on `sym`.** 0.74 with that spread is not clearly below 0.80; ~10 seeds settles
   it. Cheapest by far (~6 min).
2. **Direction-agnostic scoring** — score `recall` against the sequence and its reverse, take
   the better. Given the point above, scoring reverse replay as failure is arguably *our metric*
   being wrong rather than the arm. Cost: a suite change that weakens `predict_next`
   comparability with every other arm.
3. **Shorten the presentation window to match `asym`'s 20 ms kernel** (25 ms at a raised
   `infield_rate`). Only worth doing if 1 and 2 both fail, since it buys direction at the price
   of the empirical grounding.
4. **Drop the arm**, only if 1–3 fail.

### Phase 3 — the wrapper
Implement the §4 table. `HippocampalModel + OnlineTrainable + StatePrimeable`,
`online_equivalent = False` (the batch path repeats traversals; the streamed path does not —
they are not the same rule).

### Phase 4 — modality extension (optional)
Try the symbolic suite with the same adapter. Expect trouble: symbolic items are
near-orthogonal, so the "band" the STDP learning builds has no local structure to exploit and
the travelling wave may not propagate. A negative result here is publishable — it says
something real about what the recurrent-wave mechanism needs from its inputs.

### Phase 5 — registration and controls
Registry entry in `bin/run_benchmark.py` with a comment block in the style of the
`spiking_eqprop` entry, documenting the direction choice and the `len_sim` landmine.

**Controls to register alongside** (this is what makes it a comparison rather than a demo):
- **Rate control:** same network, same weights, real-valued rates instead of spikes. Isolates
  the spike channel — exactly the role `quantizer=None` plays for `spiking_eqprop`.
- **Symmetric-vs-asymmetric STDP:** isolates the direction mechanism.

### Phase 6 — validation gates before any figure
- Reproduce upstream's own result once (learned matrix → spontaneous replay, all events
  significant) as a regression test on the vendored copy.
- `theta ≡ AHN`-style controlled-pair check: confirm the rate control and the spiking arm
  agree on inference when given identical weights, the diagnosis method that worked for
  `spiking_eqprop`.
- Confirm `predict_next` purity (§6).

---

## 6. Cost, and the one trick that might save it

Measured: ~20–33 s wall per 10 s simulation (cython); one-time setup (spike generation +
400 s STDP in standalone) ≈ 2 min.

**The cue is a one-shot trigger, not a persistent probe.** By 400 ms activity has left the
cued block; by 5 s cued and spontaneous runs are indistinguishable. So naively it is **one
probe per 10 s simulation**.

`measure_recall_associative` defaults to `n_trials=30` over `L-1` positions. At L=5 that is
120 probes ⇒ 120 simulations ⇒ ~50 min *per sequence per condition*. That is not viable
across a sweep.

**The trick:** events last 380–780 ms, so a single 10 s run can carry ~12 cues spaced ~800 ms
apart. That is a 12× reduction, bringing L=5 to ~4 min.

**But it conflicts with the `batched_probe` contract.** `supports_batched_probe`'s docstring
is explicit: an arm that carries state across probes must not declare it. ca3net's network
state persists between cues within a run. So this needs an empirical independence check
before the flag can honestly be set:

> Present the same cue at position 1 of a batch and at position 12, in otherwise identical
> runs, and compare decoded outputs. If they diverge beyond seed noise, the flag is invalid
> and the spacing must grow (or the trick is abandoned).

Treat that check as a gate, not a formality. Declaring `batched_probe` falsely would corrupt
every number this arm produces, silently.

Even with the trick this is the most expensive arm in the registry, displacing
`spiking_eqprop`. Budget accordingly, and consider registering it for a reduced set of sweep
points from the start rather than discovering the cost mid-run.

---

## 7. Risks, in order of how much they cost

1. **Direction (Phase 2).** If `asym` does not fix it, scoring needs a suite change.
2. **Probe purity / batching (§6).** Determines whether the arm is 4 min or 50 min per
   sequence.
3. **Licence (Phase 0).** Blocks redistribution, not development.
4. **Symbolic may simply not work (Phase 4).** Acceptable — it is a finding, and the arm
   still stands on spatial.
5. **Parameter sensitivity.** The network is bistable with a narrow usable window: below
   ~6 nS peak recurrent weight it is silent (0.43 Hz, no events), above ~11 nS it saturates
   into one continuous event. Any change to encoder scaling re-enters this regime problem, so
   the adapter must hold total recurrent drive roughly invariant, and `calibrate.py` from the
   evaluation should be vendored as a tuning tool.

---

## 8. Decisions I need from you

1. **Scope:** spatial-only v1 as recommended, or attempt symbolic in the same pass?
2. **Direction:** if `asym` fails Phase 2's gate, do we change the metric to be
   direction-agnostic, or drop the arm?
3. **Licence:** resolve all six ambiguous upstreams as one decision, or proceed on ca3net
   alone and defer?
4. **Budget:** register at full sweep density and accept the cost, or reduced density from
   the start?
