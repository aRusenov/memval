# Spike Codec — specification and implementation plan

Status: **Phases 0, 1 and 2 built and gated** (2026-09-04). Written 2026-09-04.
Phase 3 (reporting spiking arms into the results suite) is not done. See
[S9 -- Results](#9-results-what-was-built-and-what-it-measured) at the foot of
this document for every number. Sections 1-8 are left exactly as designed, so
that where a measurement contradicted the design -- S2.1-S2.2's stability, S8's
open questions 1, 2 and 4 -- the disagreement stays visible instead of being
edited away. S9 says which.

Purpose: let spiking arms be scored on the existing symbolic sections without
changing a single metric, and — equally important — make the codec's own cost
**measurable** rather than silently attributed to the arm.

This is the "spike-encoder protocol tier" that has kept the `spiking x online`
cell of the model taxonomy empty. It is a prerequisite for any spiking arm
(Bush et al. 2010, `ca3net`, Cutsuridis et al. 2009), not a feature of one.


## 1. Where the codec sits

```
SymbolicEncoder / HierarchicalEncoder
        |
        v   R^D vector, unit-norm
   [ PROBE CORRUPTION HAPPENS HERE ]         <-- noise, cue masking, partiality
        |
        v
   PopulationSpikeEncoder   (new)
        |
        v   spike trains, (N neurons, T steps)
   spiking model  (fit_event / predict_next)
        |
        v   spike trains, (N neurons, T steps)
   PopulationSpikeDecoder   (new)
        |
        v   R^D vector, unit-norm
   SymbolicDecoder  (unchanged)              <-- cosine vs embeddings, top_k -> MRR
```

**The placement of probe corruption is the load-bearing decision.** Corrupting
in vector space, *upstream* of the codec, means every arm — rate and spiking —
sees the identical corrupted R^D vector on the identical probe grid. The probe
stays definable at full resolution regardless of how many neurons encode an
item. Corrupting downstream, in spike space, would quantise the probe grid to
`1 / n_per_feature` and make spiking arms incomparable with rate arms. Do not do
that.

What is at risk under upstream corruption is **sensitivity**, not definability:
the codec may wash out small corruptions before the model sees them. Section 5
is how that gets measured instead of assumed.


## 2. Hard constraints

These come from the existing code and are not negotiable without changing
metrics.

1. **`PopulationSpikeDecoder` returns a vector in R^D, never a symbol.**
   `SymbolicDecoder.decode` (`memval/encoders/symbolic.py:95`) expects
   `(seq_len, embedding_dim)` and does cosine similarity against the embedding
   matrix. Returning a vector keeps `top_k`, MRR, and every downstream metric
   untouched.
2. **Output lands on the unit-norm manifold.** `SymbolicDecoder` and the
   `l2` / `quantized` feedback modes assume it. The decoder L2-normalises as its
   last step.
3. **Sign is preserved.** Embeddings are unit-norm and signed. Spike rates are
   non-negative. See ON/OFF populations, S3.3 — this is not optional, and the
   repo has already been bitten by sign handling (`divergence_margin` compression
   for negative-output arms).
4. **Explicit seeding, no global RNG.** The codec is stochastic (Poisson). It
   takes a `seed` and owns a `np.random.Generator`. It must never consume the
   global RNG, and two runs with the same seed must be byte-identical. This is
   the defect class already found in `semantic_similarity`.
5. **The codec is a stated condition, never an invisible default.** Every figure
   produced through it names the codec and its parameters, the same way encoder
   condition is already stated for load sweeps.


## 3. Codec specification

### 3.1 Substrate: prefer the hierarchical encoder

Build and validate against `HierarchicalEncoder` first, not `SymbolicEncoder`.

Its `feature_matrix` is **binary and sparse**, with columns owned by nodes of a
tree, so a feature maps onto a neuron population with no arbitrary quantisation,
and the `shared` / `identity` block partition survives into spike space — which
means the cue-masking modes remain meaningful after transport. `SymbolicEncoder`
embeddings are `category_base + noise`, L2-normalised: arbitrary coordinates in
a random basis. The same reason masking a random basis "measures almost nothing"
(`memval/benchmarks/cue_masking.py`) applies to spike-encoding one.

Support `SymbolicEncoder` too, but treat hierarchical as the reference substrate.

### 3.2 Parameters

| Name | Meaning | Default |
|---|---|---|
| `n_per_feature` | neurons per embedding dimension, per polarity | 5 |
| `window_steps` | simulation steps per item presentation, `T` | 50 |
| `dt_ms` | step duration | 1.0 |
| `r_max` | firing rate at unit feature magnitude (Hz) | 200 |
| `mode` | `rate` \| `latency` | `rate` |
| `seed` | RNG seed | required |

Total neurons `N = 2 * D * n_per_feature` (factor 2 = ON/OFF).

### 3.3 Encoding — `rate` mode (default)

For embedding vector `v` in R^D:

```
v_on[d]  = max( v[d], 0)
v_off[d] = max(-v[d], 0)
```

Each of the `n_per_feature` neurons in population `(d, ON)` emits an independent
Poisson spike train over `T` steps with per-step probability

```
p = clip(v_on[d] * r_max * dt_ms / 1000, 0, 1)
```

and likewise for `(d, OFF)` with `v_off[d]`. Output: boolean array
`(N, T)`.

Rationale for rate over latency as the default: firing probability is **linear
in feature magnitude**, so a corruption of size eps upstream produces a
proportional change in expected spike count. That is what preserves the probe
grid. Latency coding compresses magnitude into time and is far less faithful to
graded corruption.

### 3.4 Encoding — `latency` mode

Provided because STDP-based arms need spike *order* to carry information, and a
Poisson rate code gives them little to work with.

Each population `(d, polarity)` emits one spike per neuron at

```
t[d] = round( (1 - |v[d]|) * (T - 1) )      # strong feature -> early spike
```

jittered by `+/- 1` step per neuron (seeded). Zero-magnitude features emit no
spike. This is a *secondary* mode: it does not preserve graded corruption
linearly, and any section run under it must say so.

### 3.5 Decoding

Given spike array `(N, T)`:

```
c_on[d]  = total spikes in population (d, ON)
c_off[d] = total spikes in population (d, OFF)

v_hat[d] = (c_on[d] - c_off[d]) / (n_per_feature * T * r_max * dt_ms / 1000)

v_hat = v_hat / ||v_hat||_2            # constraint 2; if ||.|| == 0, return zeros
```

The same estimator is used for both modes; under `latency` it degrades to a
count-based approximation, which is one more reason `latency` is secondary.

### 3.6 API

New file `memval/encoders/spike_codec.py`:

```python
class PopulationSpikeEncoder:
    def __init__(self, embedding_dim: int, n_per_feature: int = 5,
                 window_steps: int = 50, dt_ms: float = 1.0,
                 r_max: float = 200.0, mode: str = "rate",
                 seed: Optional[int] = None): ...

    @property
    def n_neurons(self) -> int: ...          # 2 * D * n_per_feature

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """(D,) -> (n_neurons, window_steps) bool."""

    def encode_sequence(self, vectors: np.ndarray) -> np.ndarray:
        """(L, D) -> (n_neurons, L * window_steps) bool. Items in order,
        one window each, no gap by default."""


class PopulationSpikeDecoder:
    def __init__(self, encoder: PopulationSpikeEncoder): ...

    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """(n_neurons, window_steps) -> (D,), L2-normalised."""

    def decode_sequence(self, spikes: np.ndarray) -> np.ndarray:
        """(n_neurons, L * window_steps) -> (L, D)."""
```

Decoder holds a reference to the encoder, mirroring
`SymbolicDecoder.__init__(encoder)`.

### 3.7 Inter-item gaps carry Delta t

`encode_sequence` takes an optional `gap_steps` — per-item, not global — so the
codec is the natural place to inject inter-event intervals when the interval
section is built. Nothing in this spec depends on it; it is noted so the
signature does not have to change later.


## 4. The wrapper that makes the control cheap

New `memval/models/codec_wrapper.py`:

```python
class CodecWrappedModel(HippocampalModel):
    """Runs any model behind a spike codec.

    For a spiking arm, the inner model consumes and emits spike arrays.
    For a RATE arm, the codec round-trips vector -> spikes -> vector on the
    way in and out, so the arm sees only codec-degraded vectors. That second
    use is the transport control (S5).
    """
```

It implements `fit_event`, `fit_sequence`, `predict_next`, `reset_context` and
`on_event_boundary` by delegating, encoding on the way in and decoding on the
way out. `predict_next` must remain a **pure function of the cue**, per the rule
in `tests/test_temporal_pc.py` — the codec must not leak state between probes.


## 5. The transport control — the point of the exercise

Run `AsymmetricHopfieldNetwork` twice on the identical probe grid:

- **native** — as today
- **`CodecWrappedModel(AsymmetricHopfieldNetwork, codec)`** — vectors round-trip
  through the codec on input and output; the arm itself is unchanged

The difference is the codec's contribution, **measured**. Without it, any
deficit a spiking arm shows is unattributable: a model that cannot exploit
graded input and a codec that quantised the gradation away are indistinguishable
from the outside.

Report every spiking arm against this control, not against native rate arms.

Second control, cheap and worth having: the **codec-identity** check — a
`predict_next` that returns the cue unchanged, run through the codec. It bounds
what transport alone can score.


## 6. Implementation phases

### Phase 0 — codec in isolation
No model. Characterise round-trip fidelity:
`cos(decode(encode(v)), v)` over the vocabulary, as a function of
`n_per_feature`, `window_steps`, `r_max`.

**Acceptance:** median round-trip cosine >= 0.95 at defaults; fidelity monotone
non-decreasing in `n_per_feature` and `window_steps`; identical output for
identical seed; global RNG untouched (assert via `np.random.get_state()`
before/after).

**Deliverable:** a fidelity curve. If the ceiling is low, everything downstream
is capped and the tier is not worth building — this phase is the go/no-go.

### Phase 1 — transport control
Build `CodecWrappedModel`. Run AHN native vs AHN-through-codec on the existing
symbolic sections, including the `noise_invariance` and `cue_masking` sweeps.

**Acceptance:** the gap is quantified per section and per probe level and
written down. There is no pass/fail threshold here; the number *is* the result.

### Phase 2 — first spiking arm: port of Bush et al. 2010

Bush, Philippides, Husbands & O'Shea (2010), *Dual Coding with STDP in a Spiking
Recurrent Neural Network Model of the Hippocampus*, PLOS CB 6(7):e1000839.

Chosen over `ca3net` and Cutsuridis because it is the only candidate that is
simultaneously spiking, genuinely hetero-associative (it learns *directional*
connections rather than auto-associating), and cued — and because it has no
published code, hence no license question in a pipeline already carrying six
unlicensed repos. This is a **reimplementation from published equations**, not a
port of source.

Target the paper's **dual-coded** configuration (20 fields x 5 neurons), not the
hetero-associative one (100 fields x 1 neuron). The latter is a localist code and
does not compose with a population codec.

#### 2.1 Dynamics

Izhikevich point neurons, excitatory parameters `a=0.02, b=0.2, c=-65, d=6`:

```
v' = 0.04 v^2 + 5 v + 140 - u + I
u' = a (b v - u)
if v >= 30:  v <- c ;  u <- u + d
```

Integrate with two 0.5 ms substeps per 1 ms step (Izhikevich's standard scheme);
the single-step form is unstable at these parameters. Recurrent, no self-
connections. Weights in `[0, w_max]`, `w_max = 1`, initialised at `0.01 w_max`.
Start fully connected; the paper reports no significant difference at 15 random
presynaptic connections per neuron, so sparsity is a later optimisation, not a
correctness question.

#### 2.2 Plasticity

Additive pair-based STDP in online trace form:

```
x_pre  <- x_pre  * exp(-dt/tau_plus)  + spike_pre         # tau_plus  = 20 ms
y_post <- y_post * exp(-dt/tau_minus) + spike_post        # tau_minus = 50 ms

on post spike:  w += Phi * A_plus  * x_pre                # A_plus  = +0.02 w_max
on pre  spike:  w += Phi * A_minus * y_post               # A_minus = -0.01 w_max

w <- clip(w, 0, w_max)
```

`Phi` is the ACh gate: **1 during `fit_event`, 0 during `predict_next`**. This is
the encode/retrieve separation and it is what makes `predict_next` non-mutating.

Implement the pair-based BCM variant first. The triplet and non-BCM variants
(`A_minus = -0.021`, `tau_plus = tau_minus = 20 ms`) are ablations, not the
default — build them only if the default fails 2.5.

Theta-phase modulation of plasticity (potentiation at LFP peak, depression at
trough) and the 8 Hz inhibitory drive are **deferred**. They are what make this
model a spiking instance of the SPEAR encode/retrieve mechanism and are the most
interesting thing about it scientifically, but they are not needed for a first
scored arm. Add after 2.5 passes, as an ablation pair.

#### 2.3 Interface mapping

| `HippocampalModel` | Bush model |
|---|---|
| `fit_event(x)` | codec-encode `x`, present for `window_steps`, `Phi = 1` |
| `fit_sequence(X)` | `fit_event` over rows, repeated `n_presentations` times |
| `predict_next(x)` | codec-encode `x` as cue, inject `I_cue = 30` for 1 ms to its active populations, run the recall window with `Phi = 0`, return the spike array |
| `reset_context()` | zero `v`, `u`, traces; **leave `w` untouched** |
| `on_event_boundary()` | default (calls `reset_context`) |

**`predict_next` must be a pure function of the cue.** It runs with `Phi = 0`, and
it must save and restore `v`, `u`, `x_pre`, `y_post` around the call — otherwise
membrane state leaks between the independent probes that
`measure_recall_associative` assumes are independent. This is the same defect
that `MultilayerTemporalPCNetwork` had to avoid; see `tests/test_temporal_pc.py`.

`n_presentations` defaults to 10, per the paper. It is a constructor argument on
the instance, **not** an `epochs=` kwarg — nine EP-family arms already discard
that kwarg, and any staircase that relies on it is wrong.

#### 2.4 Recall window

The paper reads out over a ~33 ms sharp-wave-ripple window. Use that as the
default `recall_steps`, and return the whole `(n_neurons, recall_steps)` array;
`PopulationSpikeDecoder` handles the reduction to a vector. Do not reduce inside
the model — that would bake a decode choice into the arm.

#### 2.5 Correctness gate — before any benchmarking

Reproduce the paper's own result first: **>90% recall fidelity** on the
dual-coded configuration, measured as temporal order fidelity across the recall
window, over 50 seeded runs. If the reimplementation cannot hit that on the
paper's own task, its scores on MemVal sections mean nothing.

Unit tests alongside:
- STDP kernel shape matches the analytic exponentials at both signs
- weights stay within `[0, w_max]` under adversarial spike trains
- identical output for identical seed; global RNG untouched
- **`predict_next` purity**: weights and membrane state byte-identical before and
  after; calling it twice on the same cue returns the same array

#### 2.6 Known limitations — record up front, do not discover later

- **Not one-shot.** Needs ~10 presentations. The rapid one-shot section should be
  expected to fail. That is a result, not a bug.
- **Purely local, no error signal** beyond what STDP asymmetry provides — unlike
  AHN, EP, tPC and the theta arm, all of which are error-correcting. It is the
  roster's Hebbian contrast case.
- Native readout is order fidelity over a SWR window, which is span-*like* but
  not identical to `max_memory_span`. State the difference; do not silently
  equate them.
- The paper does not name its simulator or language, so synaptic current shape
  and axonal delays are **inferred**. Record every inferred parameter in the
  module docstring, the way `temporal_pc.py` records its deviations from the
  reference implementation.

### Phase 3 — report
Spiking arms reported relative to the Phase-1 control. Results tables must keep
**N/A** (probe cannot be constructed) distinct from **low score** (probe ran, arm
did badly). Under upstream corruption the noise and masking sections are the
second kind and must be scored, not blanked.


## 7. Risks

| Risk | Mitigation |
|---|---|
| Codec fidelity ceiling caps all downstream results | Phase 0 is an explicit go/no-go |
| Poisson variance inflates run-to-run spread | more seeds; report spread; consider `n_per_feature` up |
| Runtime: `T` steps x sequence length x conditions x seeds | tune `window_steps` down in Phase 0; the codec is pure NumPy |
| Codec silently becomes an invisible default | constraint 5; codec params in every figure caption |
| Latency mode quietly used where rate is assumed | mode recorded per run; sections state which |


## 8. Open questions

1. Does `HierarchicalEncoder`'s `shared` / `identity` block asymmetry survive
   transport? Phase 0 should measure the masking asymmetry read-out through the
   codec with no model attached — if the codec destroys it, the block modes
   cannot be run on spiking arms and that section is N/A for them.
2. Renormalisation interacts with the codec twice — once when a masked cue is
   re-L2-normalised upstream, once in `decode`. Check the composition does not
   double-shrink.
3. Whether `latency` mode is needed at all depends on whether Bush's STDP learns
   anything from a Poisson rate code. A rate code carries no reliable spike
   *order*, and STDP is a function of order — this is the most likely way Phase 2
   fails. Defer building `latency` until Phase 2 shows it is needed, but treat it
   as probable rather than speculative.

4. **Neuron count is set by the codec, not chosen.** The model must have
   `N = 2 * D * n_per_feature` neurons to receive the codec's output. At the
   defaults with `D = 100` that is **1000 neurons**, an order of magnitude above
   the 100 Bush validated at. Three ways out, to be decided in Phase 0: reduce
   `D` (the hierarchical encoder's feature count is a free parameter in a way
   `SymbolicEncoder`'s 100 dimensions are not), reduce `n_per_feature`, or accept
   the larger network and re-establish the 2.5 correctness gate at that scale.
   Recurrent connectivity is O(N^2), so this is a runtime question as well as a
   fidelity one.


## 9. Results — what was built, and what it measured

Files:

| Thing | Where |
|---|---|
| codec | `memval/encoders/spike_codec.py` |
| wrapper + transport control | `memval/models/codec_wrapper.py` |
| Bush et al. 2010 arm | `memval/models/baselines/bush_stdp.py` |
| Phase 0 characterisation | `bin/spike_codec_phase0.py` |
| Phase 1 transport control | `bin/spike_codec_transport_control.py` |
| Phase 2.5 gate + calibration | `bin/bush_stdp_gate.py` |
| overlap diagnostic | `bin/bush_stdp_overlap_diagnostic.py` |
| N=1560 siting sweep | `bin/bush_stdp_siting_sweep.py` |
| tests (59) | `tests/test_spike_codec.py`, `tests/test_codec_wrapper.py`, `tests/test_bush_stdp.py` |


### 9.1 Phase 0 — GO, on the hierarchical substrate

Median round-trip cosine at the defaults (`n_per_feature=5`, `T=50`,
`r_max=200 Hz`), 3 codec seeds:

| substrate | median | min | gate 0.95 |
|---|---|---|---|
| hierarchical D=56 | 0.975 | 0.946 | **PASS** |
| hierarchical D=156 | 0.974 | 0.917 | **PASS** |
| symbolic D=100 | 0.934 | 0.908 | **FAIL** |

S3.1's preference for `HierarchicalEncoder` is confirmed quantitatively, and the
reason is mechanical rather than aesthetic: a sparse binary row concentrates its
norm in few dimensions, and its zero dimensions decode to *exactly* zero with no
Poisson variance at all. A dense unit-norm row spreads the same norm over every
dimension, putting each one near the noise floor. `SymbolicEncoder` needs
`n_per_feature=10` (median 0.966) to clear the gate; run it there or state the
lower ceiling.

Fidelity is monotone non-decreasing in `n_per_feature`, `window_steps` and
`r_max` (all three buy the same thing: Poisson events per dimension). Identical
seed gives byte-identical output; the global RNG is untouched.

**Sensitivity (S1's real risk).** Upstream corruption survives transport with
room to spare — at every noise level the downstream cosine loss is 1.02x to 1.6x
the upstream loss, i.e. the codec adds its own floor rather than washing the
corruption out. The probe grid transports.

**Open question 1 — answered, yes.** The hierarchical `shared`/`identity`
asymmetry survives: masking shared vs identity separates by 0.239 upstream and
0.225 downstream, **0.94x retained**. The `cue_masking` block modes are runnable
on spiking arms; that section is not N/A for them.

**Open question 2 — answered, no double-shrink.** The raw estimator is unbiased
in scale (`||decode_raw(encode(v))|| / ||v||` = 1.028 ± 0.067), so the decoder's
L2 only fixes scale and cannot compound the upstream one.

### 9.2 Phase 1 — the transport control, measured

AHN native vs AHN-through-codec, identical exposure, on `cue_masking`'s own
probe grid (hierarchical `(2,12)`, 8-item across-branch list, D=156).

| probe | native | via codec | gap |
|---|---|---|---|
| clean cued recall | 1.000 | 1.000 | 0.000 |
| clean margin | 0.424 | 0.343 | 0.081 |
| noise sigma 0.1 / 0.2 | 1.000 / 0.976 | 0.976 / 0.905 | 0.024 / 0.071 |
| mask random 0.5 / 0.75 | 1.000 / 0.929 | 0.952 / 0.881 | 0.048 / 0.048 |
| mask shared (all levels) | 1.000 | 1.000 | 0.000 |
| mask identity 0.5 / 0.75 | 0.286 / 0.286 | 0.286 / 0.333 | 0.000 / −0.048 |

**Transport costs at most 0.07 recall anywhere on this grid**, and shows up
earlier in `margin` than in `recall` — which is the expected ordering, and
another instance of margin resolving below the accuracy floor. Every spiking
score must be read against the *via codec* row, never against native.

### 9.3 Phase 2.5 — the Bush correctness gate: **PASS**, with one stated deviation

Dual-coded configuration (20 fields x 5 neurons), 50 seeds, temporal order
fidelity, a field that never replays counting as a failure:

```
recall window 40 steps   fidelity mean 0.958  sd 0.046  median 0.947  min 0.842
                         19.6 / 20 fields over 35.5 steps
recall window 33 steps   fidelity mean 0.821  sd 0.058  min 0.68
                         17.0 / 20 fields over 29.4 steps
directionality           forward 0.109   backward 0.000   all-synapse mean 0.016
GATE                     PASS (0.958 vs 0.90)
```

**The deviation is in replay speed, not order.** The reimplementation replays the
full 20-field sequence in the correct order and takes about **36 steps** to do
it, so the paper's ~33 ms ripple window truncates the last two or three fields.
`recall_steps` therefore defaults to 40 and the 33-step number is reported
beside it; a comparison against the paper's compression ratio must use 33 and
quote 0.82. Every earlier version of this file reported 0.953 at 33 steps —
that number was partly an artefact of the inhibition defect in S9.6, whose
rebound accelerated propagation.

The learned matrix is cleanly directional, the check the `ca3net` port failed.

**S2.1–S2.2 read literally do not run.** With recurrent excitation on during
encoding and no inhibition, STDP pairs the network's own reverberation instead
of the stimulus and forward and backward weights come out *equal*: direction
gone, fidelity 0.05–0.16, at every `g_syn` tried. Two stabilisers fix it, both
from the model's own SPEAR lineage and both recorded as inferred in the module
docstring: ACh suppressing recurrent transmission during encoding, and a
conductance-based feedback-inhibition pool. They trade off along a **ridge** —
excitation and inhibition rise together — and shunting inhibition slows
integration, which is why the calibrated gain is large (550) where a
current-based pool needed 160.

### 9.4 The arm through the codec — the limit is overlap, not scale

**Corrected 2026-09-04.** An earlier version of this section reached the right
conclusion — the limit is overlap — on two broken measurements, and quoted
numbers far worse than the arm's. Both errors are recorded because both are easy
to repeat, and because a conclusion that survives its own evidence being wrong
still has to be re-earned.

*Error 1 — a frozen Poisson sample.* `CodecWrappedModel.fit_sequence` encoded
the material once and let the inner arm replay that one realisation on every
pass, while streamed ingestion re-encoded per pass by construction. The two
regimes were therefore not the same protocol behind a stochastic codec
(`max|dW| = 1.3e-1` on a `w_max = 1` matrix), and `online_equivalent = True` was
false for the wrapper. Fixed by `resample_per_pass` (default True): the wrapper
now owns the pass loop and re-encodes each pass, and the two paths are
byte-identical. Pinned by `test_batch_and_streamed_ingestion_are_the_same_protocol`.

*Error 2 — an off-ridge operating point.* The stable region in
`(g_syn, k_inh)` is a **ridge**, not a low-gain corner: excitation and inhibition
must rise together, exactly as S9.3's calibration grid shows at paper scale. Its
coordinates move with the number of co-active neurons, so a codec-driven network
has to be re-sited on it. The overlap diagnostic's grid had been narrowed for
runtime in a way that missed the ridge entirely, and reported 0.57 where the arm
reaches 0.93.

With both fixed, on an orthogonal-to-overlapping block code at D=64 (N=640, 38
co-active), 10 passes, best point on the ridge:

| item overlap | mean abs cosine | Bush via codec | AHN via codec | best (g_syn, k_inh) |
|---|---|---|---|---|
| 0 | 0.00 | **1.00** | 1.00 | (45, 0.3) |
| 2 | 0.06 | **1.00** | 1.00 | (45, 0.3) |
| 4 | 0.12 | **1.00** | 1.00 | (45, 0.3) |
| 6 | 0.34 | 0.86 | 1.00 | (90, 0.3) |

So the arm **does** learn a sequence through the codec, at parity with AHN over
most of this range. Overlap costs it something at the top end (0.86 vs 1.00),
which is the expected direction for a rule with no error term — but it is a mild
effect, not the cliff the earlier reading claimed, and it does not explain the
symbolic floor.

**Resolved at N=1560** (`bin/bush_stdp_siting_sweep.py`), and **re-run after the
S9.6 fix** — the conclusion is unchanged, which is why it is stated here rather
than hedged. The sweep holds size and co-activity fixed — D=156, N=1560, 12
active dims and ~60 co-active neurons per item — and varies only overlap, across
a 5x5 `(g_syn, k_inh)` grid and 2..40 passes, streamed:

| material | mean abs cos | best recall | at | exposure |
|---|---|---|---|---|
| orthogonal (matched co-activity) | 0.00 | **1.00** | (160, 0.02) | 5 passes |
| hierarchical `(2,12)` list | 0.21 | **0.07** | nowhere in the grid | no exposure helps |

**Scale is not the limit.** N=1560 works. **Overlap is**, with size and
co-activity held fixed. The weight diagnostic gives the mechanism:

| material | `w_fwd` | `w_bwd` | ratio |
|---|---|---|---|
| orthogonal | 0.046 | 0.0000 | clean |
| hierarchical | 0.033 | 0.0056 | ~6 |

Forward potentiation is comparable in both. What overlap destroys is the
*suppression of the backward weight*: a neuron shared between item `k` and
`k+1` fires in both windows, pairs with itself across the boundary in both
directions, and is potentiated each way. The asymmetry STDP's temporal window
exists to create is cancelled by the representation, not by the rule. An
error-correcting arm subtracts its own prediction and can still separate two
overlapping items; STDP has nothing to subtract with.

The N=1560 sweep is what carries this conclusion. The D=64 table above is a
coarse single-seed corroboration and its *middle* cells are run-to-run spread,
not structure — quote the endpoints only, or add seeds.

**Cosine is the wrong overlap axis.** Measured on three materials through the
same codec:

| material | mean abs cos | pop. Jaccard | active fraction |
|---|---|---|---|
| `SymbolicEncoder` D=100 | 0.091 | **0.213** | **24.7%** |
| `HierarchicalEncoder` | 0.214 | 0.144 | 3.7% |
| orthogonal blocks | 0.000 | 0.000 | 3.6% |

Cosine ranks the dense symbolic substrate as the *easiest* of the three;
population overlap ranks it the *hardest*. This arm tracks the second, which is
mechanical: STDP chains item-specific populations, and a code where every item
drives a quarter of the network has none. On that substrate the failure is
cleanly bimodal — 77% of neuron-steps firing at low inhibition, 1.9% (chain
dead) at high, no band between — so there is no operating point, not merely a
badly-sited one. Report population Jaccard alongside cosine for any spiking arm.

### 9.5 Open question 4 — answered by measurement

**Answered: take the larger network.** Of the three ways out the spec offered —
reduce D, reduce `n_per_feature`, or accept the larger network and re-establish
the gate at that scale — the third works. N=1560 reaches 1.00 one-step recall on
an overlap-free code at (160, 3.0) with 20 passes, so neuron count is not the
binding constraint and D need not be shrunk for the arm's sake.

What *is* required is recalibration per configuration, and the failure mode is
subtler than "wrong scalar". The stable region is a **ridge** in
`(g_syn, k_inh)`: excitation and inhibition rise together, so recalibration means
moving along the ridge, not scanning gains at fixed inhibition. A grid that scans
one axis reports a working arm as a broken one (S9.4, error 2). The ridge runs
through (45, 0.3) and (90, 3.0) at N=640, and through (160, 3.0) at N=1560.
Sweep along it, and record the point used.

Runtime is not the constraint S8 feared either: the recurrent matvec and the STDP
clip are both restricted to the neurons that actually fired, so cost scales with
activity rather than N^2 — the full 5x5x5 sweep at N=1560 runs in 52 s. Runtime is no longer the constraint it
looked like: the recurrent matvec and the STDP clip are both restricted to the
neurons that actually fired, so cost scales with activity rather than N².

### 9.6 Fixed defect — the inhibitory pool caused seizures

**Found and fixed 2026-09-04, while registering the arm.**

`k_inh` was documented as a pool that only ever reduces firing. It did the
opposite. With recurrent excitation switched off entirely (`g_syn = 0`), so that
nothing but the cue and the pool could drive the network:

| `k_inh` | before | after |
|---|---|---|
| 0.0 | 2.5% | 2.5% |
| 0.3 | 2.5% | 2.5% |
| 1.5 | **87.5%** | 2.5% |
| 6.0 | **87.5%** | 2.5% |
| 30.0 | — | 2.5% |

The fix has **two halves, and the first alone is not enough** — worth recording,
because the obvious fix looked complete and was not:

1. *Conductance-based, not current-based.* The pool now accumulates a
   conductance and contributes `g_inh * (e_inh - v)`, shunting the membrane
   toward `e_inh = -80` rather than subtracting an unbounded current that drove
   the recovery variable `u` deeply negative and produced a synchronous rebound.
2. *Integrated semi-implicitly.* With the term added explicitly, `h * g > 1`
   overshoots: one substep threw `v` hundreds of mV past the reversal potential,
   the next saw a large *depolarising* `(e_inh - v)`, and the neuron fired — the
   same pathology by a different route, and the first attempt at the fix
   reproduced it exactly. Backward Euler on the linear part,
   `v <- (v + h*(f + g*e_inh)) / (1 + h*g)`, is unconditionally stable in `g`.

Pinned by `test_inhibition_never_increases_firing`, parametrised to `k_inh = 30`.

**What it changed.** `k_inh` is now a conductance, roughly 15x below the
current-based scale, and shunting slows integration so matching gains are larger
— the paper-scale ridge moved from (160, 2.0) to **(550, 0.09)**. The gate's
33-step score fell from 0.953 to 0.821 because the rebound had been accelerating
replay; at a 40-step window it is 0.958 (S9.3). Every `(g_syn, k_inh)` grid in
this section was re-run. **The S9.4 conclusion — overlap, not scale — survived
unchanged**, which is the main reason to trust it.

### 9.7 Not done

* **Phase 3.** No spiking arm is registered in `MODEL_REGISTRY` or reported into
  the results suite yet.
* **Sections.** The arm has not been run through `cue_masking` /
  `noise_invariance` as *sections* — only as the fixed-exposure probes used
  here. Nothing blocks it: the interface is ready (S9.7) and the siting is
  resolved (S9.4). Its scores there will be a floor, and that floor is now
  attributable to representational overlap rather than to transport, scale or
  calibration.
* **`latency` mode** is implemented and unit-tested but unused. Open question 3
  is still open: the arm learns fine from the Poisson rate code at low overlap,
  so a spike-order code has not yet been shown to be *needed*.
* **Theta-phase modulation and the 8 Hz drive** remain deferred, as designed.
* The **triplet** and **non-BCM** STDP ablations: `non_bcm` is available as a
  `variant`, the triplet rule is not implemented (the default passed 2.5, and
  S2.2 makes them conditional on it failing).

### 9.8 Ingestion — this arm is run streamed

**Decision (2026-09-04): the Bush arm is run through `fit_event`, not
`fit_sequence`.** It is an online arm and the online path is the one it is
scored under; `bin/spike_codec_transport_control.py`,
`bin/bush_stdp_overlap_diagnostic.py` and `bin/bush_stdp_siting_sweep.py` all
stream. Registration should list `online_symbolic`.

The choice is free rather than consequential, and that is the point:
`resample_per_pass` (S9.4, error 1) makes the batch and streamed paths
**byte-identical** behind the codec, so nothing about any number above changes
with it. Re-running the overlap diagnostic streamed reproduces the batch table
exactly. `AsymmetricHopfieldNetwork` declares no streamed path and continues to
batch — the regime is a property of the protocol, not of the arm, so both are
stated in the scripts' output rather than assumed.

### 9.9 Exposure — a presentation is an epoch

S2.3 says `n_presentations` is a constructor argument and "not an `epochs=`
kwarg". Read as *don't silently swallow the kwarg* that is right, and it is now
enforced. Read as *this arm needs its own exposure unit* it is wrong, and the
measurement says so.

Every rate online arm applies **exactly one weight update per transition**, and
its internal dynamics do not multiply that: EP settles twice and updates once
(`original_eqprop.py:236`), tPC-2 relaxes for `inf_iters` and updates once,
theta runs a cycle and updates once (`theta_phase.py:289`). So one pass is
`L−1` updates for all of them. A Bush presentation is also one pass over the
material. Same unit.

`n_epochs` is therefore an alias for `n_presentations` — readable, settable, and
accepted as a constructor kwarg — and `fit_sequence` honours a call-level
`epochs=`. All three paths are wired because the suite drives all three, and
`tests/test_bush_stdp.py` pins that they agree. The arm staircases unchanged;
`test_exposure_staircases_like_any_other_arm` runs it under the real
`epochs_to_criterion`.

Passes to a 0.95 top-1 criterion, identical orthogonal 8-item material,
identical read-out:

| arm | passes |
|---|---|
| AsymmetricHopfield | 1 |
| TemporalPC | 1 |
| ThetaPhase | 1 |
| **Bush via codec** | **12** |

That 12 is the comparable number, and it is a result about a Hebbian rule with
no error term: additive STDP at `A_plus = 0.02 w_max` needs ~50 pairings to move
a synapse where a delta rule corrects in one.

**What genuinely differs, and it is not the unit.** A pass is not the same
*amount of learning*: one plasticity event per spike rather than one per
transition — 383 vs 7 on an 8-item list, **55x** — and the multiplier is set by
the codec's `window_steps` and `r_max`, not by the sequence. No rate arm has a
knob that changes how much learning one pass contains. Two consequences:

1. The codec parameters must be held fixed and stated across any comparison that
   turns on exposure. That is spec constraint 5 already; this is a second reason
   for it, and a sharper one, because Phase 0 (S9.1) invites raising
   `window_steps` and `r_max` to buy round-trip fidelity — which silently
   retrains this arm.
2. Presentations, within-window firing rate and window duration do **not** trade
   one-for-one. On the paper task: 10 x 50 ms @ 100 Hz scores 0.954, but
   5 x 50 ms @ **200 Hz** scores 1.000 (half the passes, better), while
   10 x **25 ms** @ 200 Hz — the same total spike count as the first — scores
   0.783. Rate is superlinear in pairings where passes are linear, and shortening
   the window pulls non-adjacent fields inside `tau_minus` and potentiates
   skip-connections. So only passes are a clean exposure axis, and only with the
   other two pinned.

The exposure curve itself is well behaved and staircases cleanly (8 seeds, paper
task): 1 pass 0.03, 2 → 0.24, 4 → 0.59, 6 → 0.76, 8 → 0.88, **10 → 0.954**,
16 → 1.000. The paper's 10 is the criterion crossing almost exactly, which means
the default sits *on* the threshold and straddles it run to run; use 16 wherever
a section needs a fixed exposure that reliably clears.
