# Search: a spiking, online-trained, hetero-associative arm with published code

Date: 2026-09-04. Motivation: the spiking x online cell of the roster is still
empty. Bush 2010 ships no code (built from the Methods), ca3net is 8000
Brian2 neurons whose Phase-2 direction gate failed. This search looked for a
model that satisfies all four of: published source, hetero-association
(cue -> target, or x_t -> x_{t+1}), spiking units, non-batched training.

Method: three parallel literature/code sweeps (classic comp-neuro lineage;
SNN frameworks and HTM-like models; 2019-2026 work and ModelDB), then direct
inspection of the source of every top candidate. Every "verified" claim below
was checked against the fetched code, not the paper.

## Verdict

Two candidates are worth a port; they trade off on what "spiking" means.

| rank | candidate | why | what it costs |
|---|---|---|---|
| 1 | Vieth & Triesch 2025, GABA-modulated STDP | the only complete model that can be vendored verbatim: MIT, pure Python/numpy (PymoNNto), ~250 LoC of model code, next-symbol prediction with a population decoder shipped, plasticity is one-step pre/post STDP applied every step | units are memoryless stochastic-threshold neurons (voltage zeroed each step, no leak, no membrane state); 2640 units; afferent+efferent weight normalisation every 200 steps; hyperparameters are evolved floats |
| 2 | Spiking BCPNN synapse (Tully et al. 2016 / Fiebig & Lansner 2017) | genuine LIF/AdEx substrate; per-presynaptic-spike trace rule (verified in `bcpnn_connection.h`, ~150 lines of logic); hetero-association is a parameter (tau_i != tau_j); the rule is the reference, so provenance would be "reimplemented against a reference" | only the synapse is public (NEST 2.2 C++ module); the network around it is ours to build, as with Bush; no LICENSE file |
| 3 | Nengo `learn-associations` (PES + Voja on LIF) | ~60 lines, pip install, GPL-2 (verified: LICENSE.rst and PyPI classifier), explicit key -> value with per-timestep learning, vector I/O | learning lives on NEF decoders driven by a rate-derived error, so it is a spiking substrate under a rate interface, like spiking EP; supervised error signal |

Recommendation: port Vieth & Triesch as the arm, stating the neuron model
plainly in the roster. If the bar for "spiking" is membrane dynamics, build
the BCPNN arm on the codec instead; it is the lightest rule with a verified
reference implementation.

## Verified details, top candidates

### Vieth & Triesch 2025 (Neural Networks 183)
- Repo: https://github.com/gitmv/GABA_Modulated_STDP_Paper, MIT, default
  branch `master`, ~72 kB. Model in `Experiments/Behavior_Core_Modules.py`
  (7 kB) and `Behavior_Text_Modules.py` (14 kB); experiment
  `Experiment_char.py` (89 lines). Depends on PymoNNto 3.0.3 (pip, numpy).
- Neuron (`Output_Excitatory`): `spike = clip(voltage*mul,0)^exp > uniform`,
  then `voltage.fill(0)`. No leak, no reset dynamics, no refractoriness.
  Inhibitory units: running-average activity thresholded stochastically.
- STDP: for each E->E synapse group, `W[src.spike_old, dst.spike] +=
  eta * li_stdp_mul[dst]`, i.e. a one-step causal window; the multiplier is
  `clip((1 + GABA_input/avg_inh) * strength, min, max)`, sign can go negative
  (min = -0.15) which is the paper's contribution. Weights clipped at 0.
- Housekeeping: `Normalization` of afferent+efferent E->E sums every 200
  steps; `IntrinsicPlasticity` toward a target rate each step.
- Task: 2400 E + 240 I; input grid 10 x n_chars, one row per character, one
  step per character; 60 000 training steps then 10 000 recovery, 5 000
  free-running. `TextReconstructor` decodes the predicted character as
  argmax over rows of `W_input^T . spike`.
- Fit to memval: input rows would be replaced by codec spike trains on the
  hierarchical substrate; decoder is already population-based.

### Spiking BCPNN (Tully, Linden, Hennig & Lansner 2016 PLoS CB)
- Repo: https://github.com/Florian-Fiebig/BCPNN-for-NEST222-MPI (also
  Nikolaos-Chrysanthidis/BCPNN-NEST2.2.2-MPICH-SINGULARITY). No LICENSE.
  `bcpnn_connection.h/.cpp` (11 kB + 8 kB), authors Tully & Kaplan 2011-12.
- Rule (verified in `send()`): on each presynaptic spike, replay the
  post-spike history since the last pre spike and integrate per step
  `z_i += (y_i - z_i + eps) dt/tau_i`, `z_j` likewise with `tau_j`;
  `e_i, e_j, e_ij` with `tau_e`; `p_i, p_j, p_ij` with `tau_p` gated by a
  print-now signal `K`; `w = gain * log(p_ij / (p_i p_j))`,
  `bias = log(p_j)`. Optional Tsodyks STP on top.
- Hetero-association: `tau_i` (pre) vs `tau_j` (post) asymmetry yields a
  forward-shifted kernel; Tully 2016 uses this for sequences without any
  other change. Symmetric taus give autoassociation.
- Not public: Tully's and Fiebig's network scripts ("available upon
  request"). Paper network: 9 HC x 10 MC x 30 AdEx.
- NEST 2.2.2 only; will not build on NEST 3. A numpy port of the rule is
  ~100 lines; the arm would be codec + LIF population + this matrix rule.

### Nengo learn-associations
- https://www.nengo.ai/nengo/examples/learning/learn-associations.html;
  nengo 4.1.0 on PyPI, GPL-2 (LICENSE.rst, "OSI Approved :: GPLv2").
- 200 LIF neurons; Voja on encoders (unsupervised), PES on decoders with the
  error from a spiking ensemble, applied every 1 ms step. Key -> value;
  x_t -> x_{t+1} by feeding the next key as value.

## Checked and set aside

| candidate | code | why not first |
|---|---|---|
| Bouhadjar et al. 2022 spiking Temporal Memory (PLoS CB) and the CC-BY memristive variant (Zenodo 6754964, 182 kB, inspected) | Zenodo; NEST + NESTML | needs the author's NEST fork (`reram_synapse` branch) and NESTML fork built from source; `shtm/model.py` 671 lines + `helper.py` 695 on top; symbol-per-subpopulation input (n_E per symbol, ~10^3 neurons). Conceptually the best match (unsupervised, event-driven structural STDP, defined prediction error); large port. Re-deriving the permanence rule in numpy is medium. Main 2022 Zenodo has no license; memristive drop is CC-BY-4.0 |
| Asabuki & Fukai 2024 PriorNet (eLife) | https://github.com/TAsabuki/PriorNet_codes, Apache-2.0 header, one 370-line numpy file | two-compartment sigmoid rate units sampled to Bernoulli spikes; plasticity uses compartment potentials (rate); 500+500 units; 10^6 ms training. Learns a 5-pattern Markov chain and replays transitions |
| Saponati & Vinck 2023 (Nat Commun) | https://github.com/matteosaponati/predictive_neuron, no license | LIF with reset, but the numpy trainer draws minibatches and averages `grad.mean(axis=0)` (online only at batch=1); learns to predict its own inputs, not cue -> target; demo net 10 neurons |
| Cone & Shouval 2021 (eLife) | ModelDB 266774 (MATLAB); NEST replication zbarni/re_modular_seqlearn (custom NEST 2.20 modules) | reward-gated eligibility traces; one column per element; original code had a wiring bug fixed 2023 |
| Haga & Fukai 2018 (eLife) | https://github.com/TatsuyaHaga/reversereplaymodel_codes, MIT, Izhikevich C++ + numpy | input builder is a 1-D place field; reverse-replay is the point |
| Maes, Barahona & Clopath 2020 | ModelDB 257609, Julia + MATLAB, no license | 3000 AdEx/LIF, 60 min simulated training at 0.1 ms |
| Kim & Kim 2025 (PLoS CB) hippocampus | https://github.com/kgt1220/Hippocampus_SNN, numpy Izhikevich, no license | autoassociative pattern completion only; one Python object per neuron |
| Casanueva-Morato et al. 2024 / sPyMem | GPL-3 | sPyNNaker only (needs a SpiNNaker board) |
| Zenke et al. 2015 | fzenke/pub2015orchestrated, GPL-2, Auryn C++ | autoassociative assemblies; 5120 neurons, hours |
| Litwin-Kumar & Doiron 2014 | author site unreachable (TLS); Julia derivative comp-neural-circuits/novelty-via-inhibitory-plasticity | autoassociative |
| Vignoud et al. 2024, Reifenstein 2021, Coppolino 2021 | code exists | single-neuron or two-neuron studies, not memories |
| PyNAM (Willshaw on PyNN) | GPL-3 | weights computed offline |
| Lava, BindsNET, Norse, Brian2, ngc-learn, NEST e-prop | frameworks | STDP primitives only, no associative model shipped |

No public code found: Tully 2016 / Fiebig 2017 network scripts, Brea 2013,
Pokorny 2020, Chenkov 2017 (chain hardwired anyway), Kappel 2014, Klos 2018,
Lindsey & Aimone 2022 (Loihi), Bouhadjar ICONS 2025.

## Traps found while verifying
- Agents disagreed on Vieth's neuron model; the code settles it (memoryless).
- Saponati's README topic "backpropagation-through-time" is misleading for
  the numpy path, but the numpy trainer is still minibatched.
- Nengo's license is GPL-2, not the old non-commercial licence.
- PriorNet has an Apache-2.0 header despite no LICENSE file.

## Outcome (2026-09-05)

Both viable ports were built. Vieth & Triesch first (`docs/vieth_stdp_port.md`,
2026-09-04) -- reimplemented against the reference rather than vendored,
because PymoNNto drags in PyQt5. Then Tully et al. 2016 spiking BCPNN
(`docs/bcpnn_spiking_port.md`, 2026-09-05), the arm with a real membrane: its
rule is held to the authors' NEST synapse module by transliteration and its
gate passes on the paper's own 9x10x30 task, with the gains calibrated to the
paper's plotted weights. Behind the codec both floor, for structural reasons
each port note records.
