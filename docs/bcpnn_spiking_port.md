# Porting Tully et al. 2016: spike-based BCPNN as the spiking x online arm

*Spike-Based Bayesian-Hebbian Learning of Temporal Sequences*, PLOS Comput Biol
12(5):e1004954. Chosen from `docs/spiking_online_arm_search.md` as the
candidate with a real membrane and a readable reference for its rule. Built
2026-09-05.

## What was built

| thing | where |
|---|---|
| the arm | `memval/models/baselines/bcpnn_spiking.py` (`BCPNNSpikingNetwork`, `CodecBCPNNNetwork`) |
| fidelity + contract tests (22) | `tests/test_bcpnn_spiking.py` |
| reference-module transliteration (the oracle) | `reference_module_traces` in the same test file |
| correctness gate on the paper's own task, calibration, stimulus and codec siting scans | `bin/bcpnn_gate.py` |
| registry entry | `bcpnn_spiking`, all three modalities |
| paper + S1 Appendix + reference C++ | `scratch/` copies were read; nothing vendored |

## Provenance: rule against a reference, circuit from the equations

The learning rule has a public reference: the authors' NEST 2.2 synapse
module `bcpnn_connection.h/.cpp` (Tully & Kaplan 2011-12, in
`Florian-Fiebig/BCPNN-for-NEST222-MPI`, no LICENSE file). It does not build
against any current NEST, so it could not be *run*; it was transliterated
line by line into `reference_module_traces` and used as an oracle. The
network -- AdEx cells, columnar WTA, conductance synapses, delays, STD, the
stimulation protocol -- has no public code ("all relevant data within the
paper"), so it is built from the Methods and the S1 Appendix parameter
tables, like the Bush arm.

## How fidelity was established

Three layers, weakest to strongest, all in `tests/test_bcpnn_spiking.py`.

1. **Transliteration vs fixed-grid stepping.** The module is event-driven
   (it replays the postsynaptic history at each presynaptic spike, with its
   own quirks: the pre spike is applied at step 0 of the *next* interval, so
   a spike at t=0 is never applied; a post spike coincident with the current
   pre spike is dropped). A scalar per-synapse stepper of the same equations
   agrees with it at every presynaptic spike to 1e-12, E level included.
2. **The arm's vectorised engine vs the stepper**, synapse by synapse, with
   the E level disabled (the paper's Eq 3 has two trace levels; the module
   has three). Z, P_i, P_j and the lazily accumulated P_ij all match to 1e-10
   for every AMPA and NMDA synapse of a 24-cell network under random spike
   counts, including doublets and flushes at arbitrary points.
3. **Behaviour on the paper's own task**, `bin/bcpnn_gate.py`, below; and in
   miniature (4 x 5 x 12, five patterns) as a unit test.

## The gate: the paper's 9 x 10 x 30 network

Ten orthogonal patterns, 100 ms each, IPI = 0, trained cyclically as the
paper does (the wrap-around 9 -> 0 is learned); recall scored with the
paper's own tools -- Eq 10 attractor detection with 25 ms persistence, the
CRP over lags -4..5, and the Levenshtein distance of each recalled run from
the template (Eq 13). 10 epochs (S7 Fig: converged by 10).

| | seed 0 | seed 1 | paper |
|---|---|---|---|
| training rate of stimulated cells | 17.3 Hz | 17.2 Hz | f_max = 20 Hz |
| AMPA weight in-pattern / other | 4.3 / -2.6 nS | 4.1 / -2.6 nS | ~4 / -8 nS (Fig 3E) |
| NMDA weight forward / backward | 0.038 / -0.041 nS | 0.038 / -0.042 nS | ~0.015 / -0.03 nS (Fig 4B) |
| cued: CRP lag +1 | 0.97 | 1.00 | ~0.9 (Fig 5B) |
| cued: mean edit distance | 0.75 | 0.75 | small (Fig 8-10) |
| cued: attractor rate | 34 Hz | 35 Hz | 20-40 Hz (Fig 3H, 4F) |
| cued: dwell / speed | 78 ms, 12.8/s | 83 ms, 12.0/s | ~150-300 ms, 3-7/s (Fig 9) |
| free: CRP lag +1 | 0.84 | 0.63 | forward-dominant (Fig 4E) |

**GATE PASS** (cued CRP lag-1 >= 0.8, D_L <= 2). The one quantity that does
not match is replay speed, 2x the paper's.

## The calibration finding, and why the default is not the text's gains

Eq 4 with the gains as stated (6.02 nS AMPA, 1.22 nS NMDA) and the stated
protocol gives terminal weights of **17.8 nS** in-pattern AMPA and **1.46 nS**
forward NMDA here. The paper's own figures show **~4 nS** (Fig 3E) and
**~0.015-0.03 nS** (Fig 3F, 4B), while the negative AMPA weights (~-8 nS)
agree with mine (-9.1 nS). Under the stated gains this implementation
replays sequences in perfect order but at **207 Hz** with 390 ms dwell
(`--calibrate`); under gains multiplied by 0.3 (AMPA) and 0.025 (NMDA) it
reproduces both the plotted weights and the plotted regime. The factors are
therefore matched to the paper's plotted weights and validated against its
plotted dynamics, and are the default (`PAPER["ampa_calib"]`,
`PAPER["nmda_calib"]`); 1/1 is the equations-as-stated ablation.

I could not identify the source of the discrepancy in the text. The ratio
for AMPA (~0.25-0.3) is suspiciously close to the connection probability, as
if the figure averaged over unconnected pairs; that does not explain NMDA.
The traces are mean-preserving filters, so I can find no protocol under
which P_ij/(P_i P_j) for perfectly co-active cells comes out near 2 rather
than near 10. Open.

Two further calibrations, both in the paper's own words rather than against
it: the stimulus synapse is 15 nS, not 5 nS, because the paper's criterion
is that stimulated cells fire at f_max and 5 nS gives 7 Hz here
(`--stim`); and a 2 ms absolute refractory period is added (the paper is
silent, NEST's default is 0, and without one a driven cell fires on every
membrane substep).

## Deviations from the paper, all stated in the module docstring

No spatial layout (per-presynaptic delays drawn around the paper's grid
average, 6.45 ms); traces and delivery on a 1 ms grid with membranes at
0.2 ms, semi-implicit in the conductance terms (explicit Euler seized once
basket conductances reached ~1 muS -- the same lesson as the Bush arm's
inhibition); no E trace; the codec window tiled to the paper's 100 ms; and
the **size normalisation**: recurrent gains scaled by `270 / pattern_cells`,
pyr->basket by `30 / n_per_mc`, basket->pyr by `30 / n_basket_per_hc`, all
exactly 1 at the paper's geometry, so that a full pattern delivers the
paper's drive whatever the codec makes the cell counts.

Two implementation facts worth recording because each cost a debugging
round: the delay ring's row must be copied before it is cleared (a view
made every recurrent conductance read exactly zero); and `P_ij` is
accumulated as one matrix product per item with closed-form per-step
weights, which is exact (tested) and is what makes training 20 ms per item
instead of seconds.

## Behind the codec: what was measured

Mapping: hypercolumn = feature, its two minicolumns = ON/OFF, codec neuron k
= pyramidal cell k through the stimulus synapse, readout = the raster.
`bin/bcpnn_gate.py --codec`, 6-item lists, 20 epochs, next-item accuracy
over the 5 cue positions at the best (lag, window) of 12 scanned:

| substrate | n_per_feature | w_stim | cells / item | training rate | best acc | margin |
|---|---|---|---|---|---|---|
| hierarchical (b=[2,3], fpn=4) | 5 | 15 | 40 | 9 Hz | 0.60 | -0.02 |
| hierarchical | 5 | 50 | 40 | 20 Hz | 0.60 | -0.27 |
| hierarchical | 10 | 50 | 80 | 20 Hz | 0.40 | -0.04 |
| symbolic (D=100, cv=0.2) | 5 | 15 | 500 | 4.6 Hz | 0.40 | -0.08 |
| symbolic | 5 | 50 | 500 | 16 Hz | 0.40 | -0.04 |
| symbolic | 10 | 50 | 1000 | 14 Hz | 0.60 | +0.03 |
| symbolic | 10 | 150 | 1000 | 28 Hz | 0.60 | +0.02 |
| orthogonal one-hot | 5-20 | 15 | 5-20 | -- | 0.40 | < 0 |

Against 1.00 on the paper's geometry, this is a floor, and the reasons are
structural rather than a missed knob (gains, background, readout window,
stimulus and pattern-size normalisation were all scanned):

* **Dense substrates give 50% cell overlap by construction.** With two
  minicolumns per hypercolumn, a random-sign item shares half its cells with
  every other item; the paper's patterns share none. BCPNN weights are
  log co-activation *above chance*, and for such items that is near zero.
* **Sparse substrates have nothing to hand over within a hypercolumn.**
  The paper's transition is a within-hypercolumn switch between minicolumns
  under shared basket inhibition, and every pattern occupies every
  hypercolumn. A hierarchical item occupies 8 of 32 hypercolumns; its
  successor mostly lives in others, where there is no competition to win.
* Items are small: 40-80 cells against the paper's 270, i.e. 10-20
  presynaptic partners per cell at p = 0.25 against 67.

The registry defaults are the sited point for the symbolic pipeline's
substrate (`w_stim = 50`, `pattern_cells = 250`, readout the 100 ms after the
cue). A section run on another substrate must re-site the two named
conditions and say so.

## What this arm is for

It is the roster's spiking x online arm with sub-threshold dynamics and a
natively directional rule, and it reproduces its paper. Read its MemVal
scores as what a columnar attractor network does when handed a population
code it was not designed around -- which is a result about the code as much
as the model, and the same result the Bush and Vieth arms gave from
different mechanisms.

## Addendum (2026-09-05): the columnar codec

The floor above is the shared codec's, not the model's. Run on the interval
code the paper's S1 Appendix prescribes -- `memval/encoders/columnar_codec.py`,
registry entry `bcpnn_spiking_columnar`, fixed from the paper's side (one
hypercolumn per dimension, 10 minicolumns, 3 cells each, 200 Hz for 100 ms,
bins over +-2.5 SD of an isotropic coordinate) -- the same seven words recall
at 1.00 and the weights return to the paper's range. What remains weak is
reported as the arm's result: vector-space cue noise, load past ~10 items,
and sparse embeddings. Section-by-section numbers and the reasoning about
which parameters may be sited and which may not:
`docs/capacity_report_bcpnn.md`.
