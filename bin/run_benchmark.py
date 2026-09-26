#!/usr/bin/env python
from memval.models.baselines import EqPropSequenceNetwork
from memval.models.baselines import OriginalEqPropSequenceNetwork
from memval.models.baselines import DGOriginalEqPropSequenceNetwork
from memval.models.baselines import DGXdGEqPropSequenceNetwork
from memval.models.baselines import EWCDGXdGEqPropSequenceNetwork
from memval.models.baselines import EWCOriginalEqPropSequenceNetwork
from memval.models.baselines import AsymmetricHopfieldNetwork
from memval.models.baselines import MultilayerTemporalPCNetwork
from memval.models.baselines import PredictiveRecirculationNetwork
from memval.models.baselines import DTSESNSequenceNetwork
from memval.models.baselines import ThetaPhaseSequenceNetwork
from memval.models.baselines import SpikingEqPropSequenceNetwork
from memval.models.baselines import CodecBushNetwork
from memval.models.baselines import CodecViethNetwork
from memval.models.baselines import CodecBCPNNNetwork
import os
import sys
import argparse
from typing import Dict, Any, Type, Optional

# Add the project root to sys.path so we can import memval
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.models.baselines.hopfield import HopfieldSequenceNetwork
# ARCHIVED -- outside the model taxonomy; kept on disk, not run. See the
# MODEL_REGISTRY note below.
# from memval.models.baselines.gpt2_wrapper import GPT2SequenceModel
from memval.models.baselines.dg_eqprop import DGEqPropSequenceNetwork
from memval.models.baselines.ewc_dg_eqprop import EWCDGEqPropSequenceNetwork
from memval.benchmarks.spatial_pipeline import run_spatial_pipeline, SPATIAL_BENCHMARKS
from memval.benchmarks.symbolic_pipeline import run_symbolic_pipeline, SYMBOLIC_BENCHMARKS
from memval.benchmarks.online_symbolic_pipeline import (
    run_online_symbolic_pipeline,
    ONLINE_SYMBOLIC_BENCHMARKS,
)

# Benchmark sections available per suite, for --benchmarks subsetting.
SUITE_BENCHMARKS = {
    "spatial": SPATIAL_BENCHMARKS,
    "symbolic": SYMBOLIC_BENCHMARKS,
    "online_symbolic": ONLINE_SYMBOLIC_BENCHMARKS,
}

# Central Registry of Models
MODEL_REGISTRY = {
    "hopfield": {
        "class": AsymmetricHopfieldNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 1/1 seeds reached, epochs [1], mean 1.0
            # spatial:  1/1 seeds reached, epochs [2], mean 2.0 (400-cell grid)
            # Zero-init: seed-invariant, so 1 seed not 5 (sd=0 is a property of
            # the arm, never a finding).
            "n_epochs": 2,
            "learning_rate": 0.1,
            "activation": "relu",
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 4}}
    },
    # --- ARCHIVED: outside the model taxonomy ------------------------------
    # GPT-2 (and KNNEpisodicModel, which was never registered here) sit outside
    # the taxonomy this project compares within. The classes stay in
    # memval/models/baselines/ for archival reference, but they are not run and
    # their outputs are not compared against the taxonomy's arms. Any figure or
    # scorecard that still lists them should drop the row, not re-run them.
    # Uncomment the import above together with this entry to reinstate.
    # "gpt2": {
    #     "class": GPT2SequenceModel,
    #     "modalities": ["symbolic"],
    #     "default_kwargs": {
    #         "n_epochs": 64, # low epochs by default so it runs fast during benchmark
    #         "from_pretrained": True,
    #         "device": "auto"
    #     }
    # },
    # Bush, Philippides, Husbands & O'Shea (2010) behind the population spike
    # codec. It is an online arm and the online_symbolic suite runs it through
    # fit_event (docs/spike_codec_spec.md S9.8); the spatial and symbolic suites
    # are batch, which is byte-identical here (resample_per_pass), so the arm is
    # registered for all three and the regime is the protocol's choice.
    # It is also the roster's Hebbian contrast case -- purely local, no error
    # signal -- against AHN, EP, tPC and theta, all of which are error-correcting.
    #
    # Expect a floor on this suite, and read it as a result rather than a bug:
    # transport costs at most 0.07 recall (S9.2, measured), and the arm reaches
    # 1.00 one-step recall through the same codec at comparable network size on
    # a sparse code (S9.4). What floors it here is the substrate --
    # SymbolicEncoder's rows are dense, so ~25% of the network fires for every
    # item and any two items share ~21% of their active neurons, leaving STDP no
    # item-specific populations to chain. Note that cosine ranks this substrate
    # as the EASIEST of the three measured and population overlap ranks it the
    # hardest; this arm tracks the second.
    #
    # g_syn/k_inh sit on a stability ridge that moves with co-activity, so these
    # are the ridge point established at comparable N (1560), where the arm
    # reaches 1.00 on an overlap-free code. On THIS substrate nothing works:
    # a 4x4 probe (g 45..1200, k_inh 0.005..0.25, 5 and 20 passes) finds no
    # cell above 0.14, and the failure is cleanly bimodal -- 77% of neuron-steps
    # firing at low inhibition, 1.9% (chain dead) at high, with no band in
    # between. That is what 25% co-activity does: there is no operating point at
    # which a quarter of the network can be driven as a chain.
    # Re-site with bin/bush_stdp_siting_sweep.py for any new configuration.
    # "bush_stdp": {
    #     "class": CodecBushNetwork,
    #     "modalities": ["spatial", "symbolic", "online_symbolic"],
    #     "default_kwargs": {
    #         # Presentations. One pass over the material, the same unit every
    #         # other arm's n_epochs denotes. 20 is where the siting sweep peaked;
    #         # note recall is NON-monotone in exposure (it collapses by 40 as the
    #         # weights saturate into a burst), so a criterion staircase's
    #         # reached=False here does not mean "train longer".
    #         "n_epochs": 20,
    #         # 5, the codec default. NOTE this is BELOW Phase 0's round-trip
    #         # fidelity gate on a dense substrate -- median cosine 0.934 at 5
    #         # against the 0.95 gate, 0.966 at 10 -- and is a deliberate
    #         # cost/fidelity trade: N drops from 2000 to 1000 and the suites run
    #         # ~4x faster. Defensible only while these runs are confirming a
    #         # floor. Raise to 10 before reading any non-floor score off them,
    #         # and state the value in the caption either way (spec constraint 5).
    #         "n_per_feature": 5,
    #         "window_steps": 50,
    #         # The one-step readout window. The paper's ~33 ms sharp-wave-ripple
    #         # value is what bin/bush_stdp_gate.py scores replay over; this
    #         # pipeline probes one-step prediction, where a shorter window is the
    #         # right reduction -- 33 pools several links of the replay chain into
    #         # one decoded vector.
    #         "recall_steps": 12,
    #         # k_inh is a CONDUCTANCE (shunting toward e_inh), not a current.
    #         "g_syn": 160.0,
    #         "k_inh": 0.02,
    #         "seed": 42,
    #     }
    # },
    # Vieth & Triesch (2025) GABA-modulated STDP -- the roster's spiking x online
    # arm. Reimplemented against the upstream MIT release, which
    # tests/test_vieth_gaba_stdp.py replays step by step; the correctness gate on
    # the paper's own task is bin/vieth_stdp_gate.py.
    #
    # TWO THINGS TO KNOW BEFORE READING A SCORE OFF THIS ARM.
    #
    # 1. It is BIMODAL ACROSS SEEDS, and that is the model, not the port. On
    #    upstream's own character task at n_exc=600, upstream itself regenerates
    #    its training text on 5 of 10 seeds and produces noise on the other 5,
    #    with nothing in between (scores 3.99-4.13 vs 1.72-2.07). The port lands
    #    3 of 8. So a single seed here is a coin flip: report a distribution over
    #    seeds, never a mean, and never a single run.
    # 1b. IT CANNOT INGEST SEQUENCES ONE AFTER ANOTHER (2026-09-05, the arm's
    #    headline limitation). Its source paper sizes every assembly by how
    #    often its item occurs in the stream (Vieth & Triesch 2025, sec. 4.5-4.6:
    #    cluster size proportional to pattern frequency, N_active = N_E * h),
    #    so a list presented alone takes the whole excitatory population and a
    #    later list re-labels it: chain intact, input bindings overwritten
    #    (docs/vieth_stdp_port.md). The paper only ever trains in random block
    #    order and never tests blocked lists. Every multi-list section here --
    #    multiple_sequences, continual_chain, paired_associate, schema -- is
    #    blocked, and it scores zero on all of them for this reason. The
    #    interleaved condition in multiple_sequences is the control that shows
    #    it holds both lists when they share the stream. A property of the
    #    model under a protocol its paper does not use; reported, not patched.
    # 2. FENCED INGESTION COSTS IT MORE THAN OVERLAP DOES. On a 6-item sequence
    #    it reaches 1.00 cued and 1.00 autoregressive recall trained as one
    #    continuous stream, and 0.40 / 0.20 trained through fit_sequence's
    #    fenced passes -- and that gap does NOT close with exposure
    #    (bin/vieth_stdp_sanity.py). Its assemblies form by recurrent
    #    competition, which a cold start at every pass disrupts. State the
    #    ingestion regime in any caption; for this arm it is not neutral.
    # 3. It needs FAR more exposure than the rate arms. Its homeostatic
    #    threshold has to settle before the recurrent chain can free-run at all,
    #    which took ~20000 presentations upstream. n_epochs=200 is a compromise
    #    that keeps the suites runnable; expect an unsettled network and score it
    #    as such rather than as a capability claim.
    # "vieth_gaba_stdp": {
    #     "class": CodecViethNetwork,
    #     # Spiking arm behind the codec, trained through fit_event, so the same
    #     # three modalities the Bush arm carries.
    #     "modalities": ["spatial", "symbolic", "online_symbolic"],
    #     "default_kwargs": {
    #         "n_epochs": 200,
    #         # Codec condition, matched to the Bush arm's so the two spiking arms
    #         # are read on the same substrate (spec S2.5 -- state it in captions).
    #         "n_per_feature": 5,
    #         "window_steps": 50,
    #         # Upstream's protocol: ONE network iteration per item. The STDP
    #         # window is exactly one iteration wide, so presenting a 50-step codec
    #         # window instead makes 49 of every 50 pairs a within-item self-pair
    #         # and buries the transition. See the module docstring.
    #         "presentation_steps": 1,
    #         # The network's pipeline is two iterations deep, so lag 2 is the
    #         # first genuinely predictive state; lag 1 reads the cue back.
    #         "cue_lag": 2,
    #         "recall_steps": 1,
    #         "exc_per_input": 0.5,
    #         "seed": 42,
    #     }
    # },
    # Tully, Linden, Hennig & Lansner (2016) spike-based BCPNN -- the roster's
    # spiking x online arm WITH a real membrane (AdEx, conductance synapses,
    # adaptation, short-term depression). Rule reimplemented against the
    # authors' NEST 2.2 synapse module (transliterated as the oracle in
    # tests/test_bcpnn_spiking.py); network from the Methods and S1 Appendix.
    # Correctness gate on the paper's own 9x10x30 task: bin/bcpnn_gate.py
    # (PASS: cued CRP lag-1 0.99, D_L 0.75, 34 Hz attractors, 2 seeds).
    #
    # THREE THINGS TO KNOW BEFORE READING A SCORE OFF THIS ARM.
    #
    # 1. The gains are CALIBRATED TO THE PAPER'S FIGURES, not its text. Eq 4
    #    with the stated gains gives weights 3x (AMPA) and 40x (NMDA) larger
    #    than the paper plots and a replay 4x hotter than it shows; with
    #    ampa_calib=0.3 / nmda_calib=0.025 the reimplementation matches the
    #    plotted weights AND the plotted recall regime (20-40 Hz attractors).
    #    Replay is still ~2x faster than the paper's 3-7 attractors/s.
    #    bin/bcpnn_gate.py --calibrate prints both. docs/bcpnn_spiking_port.md.
    # 2. Its columnar structure is the codec's: hypercolumn = feature, the two
    #    minicolumns = ON/OFF. That makes every dense item share ~50% of its
    #    cells with every other -- the paper's patterns share 0% -- so the
    #    log-ratio weights sit near chance and attractors are weak. On the
    #    hierarchical substrate items are small (8 features x n_per_feature
    #    cells) and live in disjoint hypercolumns, so the paper's
    #    within-hypercolumn handover has nothing to act on. Measured next-item
    #    accuracy through the codec is 0.4-0.6 on 6-item lists at every
    #    readout window scanned (bin/bcpnn_gate.py --codec), against 1.00 on
    #    the paper's geometry. A floor, and an attributable one; see the
    #    port note before reading anything into it.
    # 3. Two conditions are SITED PER SUBSTRATE and belong in every caption:
    #    w_stim (the paper's 5 nS at 200 Hz is 7 Hz here; unit-norm items
    #    drive 20-70 Hz, and 50 nS brings training to ~f_max on both
    #    substrates) and pattern_cells (the recurrent-gain normalisation;
    #    250 is the symbolic pipeline's measured active count, ~40 is the
    #    hierarchical encoder's at n_per_feature=5).
    # DROPPED 2026-09-23: spiking arms are out of the paper roster (rate arms
    # only). Entry kept verbatim below so it can be restored; its budget was
    # never re-sized after the 2026-09-06 paper-value changes.
    # "bcpnn_spiking": {
    #     "class": CodecBCPNNNetwork,
    #     "modalities": ["spatial", "symbolic", "online_symbolic"],
    #     "default_kwargs": {
    #         # P traces converge within ~10 passes (paper S7 Fig; tau_p = 5 s
    #         # against 0.6 s per pass); 20 matches the Bush arm's unit.
    #         "n_epochs": 20,
    #         # Codec condition matched to the other two spiking arms (spec S2.5).
    #         "n_per_feature": 5,
    #         "window_steps": 50,
    #         # Two 50 ms windows = the paper's 100 ms stimulus.
    #         "cue_repeats": 2,
    #         # Readout: the 100 ms right after the cue. The successor appears
    #         # within 0-50 ms of cue offset on every substrate scanned and the
    #         # cued attractor's tail is what a lag would have to trade against.
    #         "readout_lag_ms": 0,
    #         "recall_steps": 100,
    #         # Sited on the symbolic pipeline's substrate (note 3).
    #         "pattern_cells": 250,
    #         "overrides": {"w_stim": 50.0},
    #         # Cost. A probe simulates 200 ms of the whole circuit and the suite
    #         # makes 180 probes per score, so the circuit is run at 3 basket
    #         # cells per hypercolumn (the basket->pyr gain is normalised by the
    #         # count, so a volley delivers the paper's inhibition either way;
    #         # 1 per hypercolumn is too coarse a WTA and floors) and 0.5 ms
    #         # membrane substeps (semi-implicit, so stable). Probe 0.47 s ->
    #         # 0.08 s, same scores; paper gate re-checked at these settings.
    #         "n_basket_per_hc": 3,
    #         "substeps": 2,
    #         "seed": 42,
    #     }
    # },
    # The same arm on the interval-coded COLUMNAR codec -- the format its
    # paper prescribes for a continuous variable (S1 Appendix: one
    # hypercolumn per variable, one minicolumn per interval, one active at a
    # flat rate). Fixed from the paper's side, never re-sited per section:
    # 10 minicolumns per dimension, 3 cells each (300 cells per item against
    # the paper's 270), 200 Hz for 100 ms, bins over +-2.5 SD of an isotropic
    # unit-norm coordinate. On the population codec above the arm's memory is
    # empty (in-item weight 0.36 nS vs the paper's ~4); on this code it learns
    # the paper's weights and recalls 7-item lists at 1.00 (standalone
    # verification in docs/capacity_report_bcpnn.md sec 1b-1c). Its known
    # weaknesses are the code's resolution -- vector-space cue noise flips
    # bins -- and the paper's own load limit (10 patterns in 10 minicolumns).
    # Both are results, not things to tune away.
    # DROPPED 2026-09-23: spiking arms are out of the paper roster (rate arms
    # only). Entry kept verbatim below so it can be restored; its budget was
    # never re-sized after the 2026-09-06 paper-value changes.
    # "bcpnn_spiking_columnar": {
    #     "class": CodecBCPNNNetwork,
    #     "modalities": ["spatial", "symbolic", "online_symbolic"],
    #     "default_kwargs": {
    #         "codec": "columnar",
    #         "n_mc": 10,
    #         "n_per_mc": 3,
    #         "z_range": 2.5,
    #         "n_epochs": 20,
    #         # cue = the codec's 100 ms window once; readout the 100 ms after.
    #         "readout_lag_ms": 0,
    #         "recall_steps": 100,
    #         # Paper's stimulus synapse at the paper's 200 Hz (the 50 nS of the
    #         # population entry compensated for graded, weaker drive).
    #         "overrides": {"w_stim": 15.0},
    #         "n_basket_per_hc": 3,
    #         "substeps": 2,
    #         "seed": 42,
    #     }
    # },
    # "eqprop": {
    #     "class": EqPropSequenceNetwork,
    #     "modalities": ["spatial", "symbolic"],
    #     "default_kwargs": {
    #         "learning_rate": 0.1,
    #         "n_epochs": 100,
    #         "beta": 0.5,
    #         "activation": "tanh",
    #         "seed": 42
    #     }
    # },
    "original_eqprop": {
        "class": OriginalEqPropSequenceNetwork,
        # online_symbolic added 2026-09-03. The arm has declared OnlineTrainable
        # all along, so capacity question 2.3 (one-shot from a single streamed
        # pass) is answerable for it; omitting the modality reported that as a
        # missing instrument rather than as a result. The pipeline itself already
        # gates on supports_online() and writes status: not_applicable, so the
        # modality list was the only thing withholding the run.
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            # Paper values (Scellier & Bengio 2017, Table 2, 784-500-10 row), set
            # 2026-09-06 after the provenance audit (docs/model_table.md). Not
            # paper-matchable here: the paper's hard sigmoid rho(s)=0 v s ^ 1 (the
            # class offers tanh|sigmoid; kept tanh) and its asymmetric 20 free / 4
            # clamped iterations (one budget serves both phases; set to the free
            # count). dt=0.5 and learning_rate=0.1 already matched (eps, alpha_1).
            "n_hidden": 500,
            "n_settle_steps": 20,
            "learning_rate": 0.1,
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 5/5 seeds reached, epochs [19, 30, 17, 22, 19], mean 21.4
            # spatial:  5/5 seeds reached, epochs [40, 47, 43, 42, 40], mean 42.4 (400-cell grid)
            "n_epochs": 43,
            "beta": 1.0,
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 85}}
    },
    # "dg_original_eqprop": {
    #     "class": DGOriginalEqPropSequenceNetwork,
    #     "modalities": ["spatial", "symbolic"],
    #     "default_kwargs": {
    #         "n_dg": 1000,
    #         "dg_target_sparsity": 0.05,
    #         "dg_inhibition": 1.0,
    #         "learning_rate": 0.1,
    #         "n_epochs": 300,
    #         "beta": 0.5,
    #         "seed": 42,
    #         "dg_seed": 42
    #     }
    # },
    # "dg_xdg_eqprop": {
    #     "class": DGXdGEqPropSequenceNetwork,
    #     "modalities": ["spatial", "symbolic"],
    #     # Wide + sparse: large n_hidden with a small gate keeps per-item capacity
    #     # (~41 units/item) while keeping subnetworks disjoint under many
    #     # interferers — cumulative coverage ~(1-gate_sparsity)^n_interferers.
    #     # Heavier compute than a narrow net; shrink n_hidden to trade retention
    #     # for speed.
    #     "default_kwargs": {
    #         "n_dg": 1000,
    #         "dg_target_sparsity": 0.05,
    #         "dg_inhibition": 1.0,
    #         "n_hidden": 2048,
    #         "gate_sparsity": 0.02,
    #         "learning_rate": 0.1,
    #         "n_epochs": 300,
    #         "beta": 0.5,
    #         "seed": 42,
    #         "dg_seed": 42,
    #         "gate_seed": 42
    #     }
    # },
    # "ewc_dg_xdg_eqprop": {
    #     "class": EWCDGXdGEqPropSequenceNetwork,
    #     "modalities": ["spatial", "symbolic"],
    #     # Small/fast hidden layer: EWC protects A-important weights directly, so
    #     # retention no longer needs a wide layer. gate_sparsity=1.0 disables XdG
    #     # (EWC alone beats EWC+XdG here — XdG's gating costs learning capacity).
    #     # ewc_lambda sweet spot is a flat plateau ~[3e4, 1e5] (A~0.89 retained,
    #     # B~0.98 learned); it collapses by ~3e5 (over-constrained). Fisher is
    #     # tiny (~1e-3) so lambda must be large.
    #     "default_kwargs": {
    #         "n_dg": 1000,
    #         "dg_target_sparsity": 0.05,
    #         "dg_inhibition": 1.0,
    #         # "n_hidden": 128,
    #         "gate_sparsity": 1.0,
    #         "ewc_lambda": 30000.0,
    #         "ewc_decay": 0.0,
    #         "learning_rate": 0.1,
    #         "n_epochs": 300,
    #         "beta": 0.5,
    #         "seed": 42,
    #         "dg_seed": 42,
    #         "gate_seed": 42
    #     }
    # },
    # "ewc_original_eqprop": {
    #     "class": EWCOriginalEqPropSequenceNetwork,
    #     "modalities": ["spatial", "symbolic"],
    #     # EWC directly on the plain net (no DG, no XdG) — the control for
    #     # measuring whether the DG front-end adds anything over plain EWC.
    #     "default_kwargs": {
    #         # "n_hidden": 128,
    #         "ewc_lambda": 30000.0,
    #         "ewc_decay": 0.0,
    #         "learning_rate": 0.1,
    #         "n_epochs": 300,
    #         "beta": 0.5,
    #         "seed": 42
    #     }
    # },
    # "dg_eqprop": {
    #     "class": DGEqPropSequenceNetwork,
    #     "modalities": ["spatial", "symbolic"],
    #     "default_kwargs": {
    #         "n_dg": 1000,
    #         "sparsity": 0.05,
    #         "learning_rate": 0.1,
    #         "n_epochs": 100,
    #         "beta": 0.5,
    #         "activation": "tanh",
    #         "seed": 42
    #     }
    # },
    # "ewc_dg_eqprop": {
    #     "class": EWCDGEqPropSequenceNetwork,
    #     "modalities": ["spatial", "symbolic"],
    #     "default_kwargs": {
    #         "n_dg": 1000,
    #         "sparsity": 0.05,
    #         "learning_rate": 0.1,
    #         "n_epochs": 100,
    #         "beta": 0.5,
    #         "activation": "tanh",
    #         "seed": 42,
    #         "ewc_lambda": 100.0,
    #         "ewc_method": "lr"
    #     }
    # },
    # --- non-EP arm: temporal predictive coding (Tang, Barron & Bogacz 2023) --
    # Registered 2026-09-03. Its absence was an oversight, not a decision:
    # models/baselines/__init__.py records tPC and DTS-ESN as "a crucial part of
    # the taxonomy ... compared normally", yet neither was reachable from this
    # registry, so no pipeline could run them.
    #
    # This is the ONLY registered arm that declares StatePrimeable, which makes
    # it the only arm for which the graded disambiguation sweep's `observe`
    # priming path fires at all. Every other entry here reports
    # state_primed=False, so their delay > 0 rows are protocol floors rather
    # than model results (docs/disambiguation_design.md sec 3, C2).
    #
    # kwargs follow bin/continual_chain_experiment.py's "tpc2" precedent
    # (n_hidden=128, learning_rate=0.05). inf_iters is the relaxation budget per
    # timestep and dominates runtime; the reference default is 100.
    "temporal_pc": {
        "class": MultilayerTemporalPCNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            # Paper ratio (Tang et al. 2023, Table 1: latent 480 for 784-d, 630 for
            # 1024-d, i.e. ~0.6x input) applied to the 100-d embedding, and the
            # paper's 100 inference iterations at step 1e-2 (the class default).
            # Set 2026-09-06; was 128 / 50. learning_rate has no paper value to
            # inherit (the paper's span 1e-5..5e-1) and stays 0.05.
            "n_hidden": 60,
            "learning_rate": 0.05,
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 5/5 seeds reached, epochs [39, 34, 48, 41, 44], mean 41.2
            # spatial:  5/5 seeds reached, epochs [32, 42, 12, 35, 14], mean 27.0 (400-cell grid)
            "n_epochs": 83,
            "inf_iters": 100,
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 54}}
    },
    # --- non-EP arm: predictive recirculation (Chen, Zhang, Cameron & Sejnowski 2024)
    # Registered 2026-09-08 as the CONTROLLED PAIR for `temporal_pc`: same family
    # (rate x online, prediction-error driven, tanh latent carried across
    # events, local outer-product updates), differing on exactly two design
    # decisions -- the latent is a forward pass rather than a relaxation, and
    # the error reaches the hidden layer through the forward input weights U
    # rather than W_out^T. It is also the only arm carrying anatomy beyond a
    # motif (EC/DG input, CA3 recurrent, CA1 error units).
    # Ported from the paper's equations (the repo has no license); only the
    # LOCAL rule is ported, not the BPTT-trained headline model.
    # n_hidden / learning_rate copy the tPC entry so the pair differs only where
    # the equations differ. n_epochs sized 2026-09-08 by the docs/model_table.md
    # rule (2x epochs-to-criterion on the canonical material, larger of the two
    # modalities): 7-item list reaches at 7 epochs, the 15-step T-maze at 64
    # (48 fails, 64 passes; ladder 1..64), so 2 x 64 = 128. Fast to criterion on
    # short material, but the 24-item cue_masking list and the length sweep are
    # NOT acquired at 512 epochs -- the paper's own long-sequence caveat.
    "predictive_recirculation": {
        "class": PredictiveRecirculationNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            "n_hidden": 60,
            "learning_rate": 0.05,
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 4/5 seeds reached, epochs [10, 512, 18, 40, 18], mean 21.5
            # spatial:  1/5 seeds reached, epochs [512, 4, 512, 512, 512], mean 4.0 (400-cell grid)
            "n_epochs": 43,
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 8}}
    },
    # The only arm clocked by elapsed TIME rather than event ordinal: it declares
    # TemporallyClocked and TimingPredictive (memval/models/capabilities.py), so
    # it is the only one that can answer Serial order's interval questions
    # (docs/capacity_questions.md 5.4, 5.5 -- half that capacity's weight). It
    # also declares OnlineTrainable. kwargs follow examples/dts_esn_*_demo.py;
    # predict_timing=True enables the timing head that 5.5 needs. The reservoir
    # is fixed; the RLS readout keeps updating across passes, and fit_sequence
    # loops `epochs` times (resetting context each pass). n_epochs=1 is the
    # ordinal-clocked baseline exposure, not a statement that more is inert.
    # --- non-EP arm: theta-phase encode/retrieve (Hasselmo, Bodelon & Wyble 2002) -
    # Registered 2026-09-03 for the zoo capacity run. The arm was implemented and
    # demoed (examples/theta_phase_demo.py) but never reachable from this
    # registry, so no suite could run it.
    #
    # It is the CONTROLLED PAIR for `hopfield`: at the paper's optimal phases
    # (phi_EC = phi_LTP = 0, phi_CA3 = pi, X = 1) the theta cycle integral
    # reduces exactly to the delta rule, so this arm takes the *identical* weight
    # step as AsymmetricHopfieldNetwork at the same learning_rate. The one
    # remaining difference is the readout -- AHN rectifies, Hasselmo eq 2.4 is
    # linear -- so `rectify_output=True` makes the pair exact on predictions too.
    # That is the point of running it: same rule, error derived by oscillatory
    # phase separation instead of explicit subtraction. Any divergence from the
    # AHN scorecard is a bug or a readout effect, not a model difference.
    #
    # n_epochs=1 matches hopfield: the arm is naturally online and one pass is the
    # honest default. Sections that need more staircase up to it.
    "theta": {
        "class": ThetaPhaseSequenceNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 1/1 seeds reached, epochs [1], mean 1.0
            # spatial:  1/1 seeds reached, epochs [2], mean 2.0 (400-cell grid)
            # Zero-init: seed-invariant, so 1 seed not 5 (sd=0 is a property of
            # the arm, never a finding).
            "n_epochs": 2,
            "learning_rate": 0.1,
            "rectify_output": True,
            "modulation_depth": 1.0,
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 4}}
    },
    # --- spiking EP (O'Connor, Gavves & Welling, AISTATS 2019) ----------------
    # Registered 2026-09-03. Thin wrapper over the authors' vendored code at
    # memval/vendor/spiking_eqprop. The LEARNING RULE is unchanged EP; what
    # changes is the communication channel between neurons (binary sigma-delta
    # spikes + leaky predictive decode instead of real-valued rates). So it is
    # read against `original_eqprop`, and `quantizer=None` through the same code
    # path is its own exact real-valued control.
    #
    # `random_flip_beta=False` departs from upstream deliberately and this is a
    # REGIME landmine, not a tuning choice: upstream's MNIST run takes ~3000
    # minibatch updates per epoch so the sign noise averages out; MemVal takes
    # one full-batch update per epoch over a handful of transitions, where it
    # does not (0.24 vs 0.88 next-item accuracy on the demo chain over 5 seeds).
    #
    # 100/50 settling ticks is upstream's "longer" setting, and it is NOT a knob
    # to trade for speed: it is the paper's own calibration. Their ablation
    # (demo_mnist_quantized_eqprop.py, X_osa_1hid_longer) at epoch 25 --
    #     n_neg/n_pos  20/4 -> 6.20% test error
    #                 50/20 -> 4.68%
    #                100/20 -> 3.25%
    #                100/50 -> 2.58%   <- this setting, and the paper's result
    # Note the assignment: the paper's prose reads "the positive and negative
    # phases to (100, 50) respectively", which is backwards; the code is
    # authoritative and says n_negative_steps=100, n_positive_steps=50.
    #
    # `input_gain` is left at the arm's own default of sqrt(n_features) rather
    # than pinned here. That default exists because MemVal's encoders emit
    # UNIT-NORM vectors, so per-component magnitude falls as 1/sqrt(n_features),
    # while upstream's MNIST pixels already fill the neurons' [0, 1] domain. At
    # the old gain of 1.0 a 100-dim embedding mapped into [0.15, 0.79] -- every
    # neuron saw a near-constant 0.5 -- and the arm floored at chance (MRR 0.167
    # on a 7-item list) for ANY budget from 1 to 512 updates. With the gain it
    # reaches criterion in 32. Diagnosed 2026-09-04; see the arm's module
    # docstring for the measured sweep.
    #
    # Cost note: `n_epochs=100` here means 100 full-batch updates, because a
    # MemVal sequence is a handful of transitions that fit in ONE batch.
    # Upstream's n_epochs=25 over 50k MNIST images at minibatch_size=20 is
    # ~2.5k updates per epoch, ~62k in total. These are not the same unit, and
    # our budget is the far smaller one -- do not read `n_epochs=100` as
    # "more training than the paper".
    # DROPPED 2026-09-23: spiking arms are out of the paper roster (rate arms
    # only). Entry kept verbatim below so it can be restored; its budget was
    # never re-sized after the 2026-09-06 paper-value changes.
    # "spiking_eqprop": {
    #     "class": SpikingEqPropSequenceNetwork,
    #     "modalities": ["spatial", "symbolic", "online_symbolic"],
    #     "default_kwargs": {
    #         # Paper width (O'Connor, Gavves & Welling 2019: 784-500-10), set
    #         # 2026-09-06; was (100,) on the argument that MemVal sequences are
    #         # small. beta and learning_rate are not stated in the paper's main
    #         # text and stay as they were.
    #         "hidden_sizes": (500,),
    #         "quantizer": "sigma_delta",
    #         "n_negative_steps": 100,
    #         "n_positive_steps": 50,
    #         "beta": 0.5,
    #         "random_flip_beta": False,
    #         "learning_rate": 0.05,
    #         # STALE BUDGET (2026-09-06): n_epochs was sized as 2x epochs-to-criterion
    #         # under the PREVIOUS hyper-parameters. The settings below were then reset
    #         # to the mother paper's values without re-running; re-size before use.
    #         "n_epochs": 22,
    #         "seed": 42
    #     },
    #     # Place codes are sparse and already in [0,1]; the symbolic mapping
    #     # (offset .5, gain sqrt(n)) flattens them to a constant 0.5 and the arm
    #     # scores 0.0 at ANY budget. With the [0,1] mapping it reaches criterion
    #     # at 257 -> spatial budget 514 (2x); symbolic/online use 22 (2x L=7).
    #     "modality_kwargs": {
    #         "spatial": {"input_offset": 0.0, "input_gain": 1.0, "n_epochs": 514}
    #     }
    # },
    "dts_esn": {
        "class": DTSESNSequenceNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            "n_units": 400,
            "tau_min": 0.1,
            "tau_max": 20.0,
            # Kept at 0.9 DELIBERATELY (audit 2026-09-06). The paper tunes rho per
            # system -- 1.0 for three of them, 0.1 for the N=400 Lorenz setting we
            # take the size from -- and the class requires rho < 1 for a stable rest
            # state (gaps must decay). n_units=400, log-uniform leak rates and
            # connectivity=0.1 ARE the paper's. RLS readout is ours (paper: ridge).
            "spectral_radius": 0.9,
            "dt": 0.05,
            "predict_timing": True,
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 5/5 seeds reached, epochs [1, 1, 1, 1, 1], mean 1.0
            # spatial:  criterion UNREACHED on every seed (400-cell grid) -> no
            # spatial budget; the suite runs at this budget with
            # criterion_reached=False recorded.
            "n_epochs": 2,
            "seed": 42
        }
    }
}

# EP (Equilibrium Propagation) models settle iteratively and need a far larger
# epoch budget to reach the convergence threshold than one-shot associators, so
# the symbolic convergence sweep runs them out further by default.
EP_SYMBOLIC_MAX_EPOCHS = 2048
DEFAULT_SYMBOLIC_MAX_EPOCHS = 500

def resolve_symbolic_max_epochs(model_class: Type[Any], cli_max_epochs: Optional[int]) -> int:
    """Resolve the symbolic convergence epoch budget: an explicit CLI value wins;
    otherwise EP models get EP_SYMBOLIC_MAX_EPOCHS and everything else the default."""
    if cli_max_epochs is not None:
        return cli_max_epochs
    if "EqProp" in model_class.__name__:
        return EP_SYMBOLIC_MAX_EPOCHS
    return DEFAULT_SYMBOLIC_MAX_EPOCHS

def _coerce_arg_value(raw: str) -> Any:
    """Best-effort typing of a CLI override value: bool, then int, then float,
    else the original string."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_benchmark_args(spec: str, available) -> Dict[str, Dict[str, Any]]:
    """Parse a ``--benchmark-args`` spec into ``{benchmark: {key: value}}``.

    Format: comma-separated ``benchmark:key=value`` entries. Values are typed via
    ``_coerce_arg_value``. Raises ``ValueError`` on malformed entries or unknown
    benchmark names (validated against ``available``).
    """
    result: Dict[str, Dict[str, Any]] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        bench, sep, rest = token.partition(":")
        if not sep or "=" not in rest:
            raise ValueError(
                f"Malformed --benchmark-args entry {token!r}; "
                f"expected 'benchmark:key=value'."
            )
        key, _, value = rest.partition("=")
        bench, key = bench.strip(), key.strip()
        if not bench or not key:
            raise ValueError(
                f"Malformed --benchmark-args entry {token!r}; "
                f"expected 'benchmark:key=value'."
            )
        result.setdefault(bench, {})[key] = _coerce_arg_value(value.strip())
    unknown = [b for b in result if b not in available]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s) in --benchmark-args: {unknown}. "
            f"Available: {list(available)}"
        )
    return result


def print_summary_table(metrics: Dict[str, Any], title: str):
    """Prints a clean summary table of calculated metrics in the CLI."""
    print("\n" + "=" * 50)
    print(f" {title} METRICS SUMMARY ")
    print("=" * 50)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:<35}: {v:.5f}")
        else:
            print(f"  {k:<35}: {v}")
    print("=" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run MemVal sequence memory benchmark pipelines.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Name of the model to benchmark (required unless --list-benchmarks)."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["spatial", "symbolic", "online_symbolic", "all"],
        help="Benchmark suite to run (default: all)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory where results and plots will be saved (default: ./results)."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of trials for evaluation (default: 30)."
    )
    parser.add_argument(
        "--mrr-threshold",
        type=float,
        default=0.95,
        help="MRR threshold for convergence (default: 0.95)."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max epochs for symbolic convergence sweep. Defaults per model: "
             f"{EP_SYMBOLIC_MAX_EPOCHS} for EP (Equilibrium Propagation) models, "
             f"{DEFAULT_SYMBOLIC_MAX_EPOCHS} otherwise. Pass a value to override."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Global training-epoch override for the fixed-epoch benchmark "
             "sections (spatial and symbolic). Precedence: --benchmark-args "
             "(per section) > --epochs > model default > each section's own "
             "default. Does not affect the epoch sweeps (symbolic "
             "'presentation_duration', governed by --max-epochs, and spatial "
             "'tmaze_reversal')."
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.05,
        help="Standard deviation of the Gaussian ISI filler events streamed between items "
             "in the online suite (default: 0.05). Retrieval cues are clean by protocol; "
             "cue degradation is swept only in noise_invariance / cue_masking -- see docs/probe_protocol.md."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated subset of benchmark sections to run within the "
             "chosen suite (default: all). Requires a single --suite (not 'all') "
             "since section names are suite-specific. Use --list-benchmarks to "
             "see the available names per suite."
    )
    parser.add_argument(
        "--benchmark-args",
        type=str,
        default=None,
        help="Per-benchmark parameter overrides, as a comma-separated list of "
             "'benchmark:key=value' entries (e.g. 'sequence_length:epochs=500'). "
             "Values are parsed as bool/int/float when possible, else kept as "
             "strings. Requires a single --suite (not 'all'). The trainable "
             "sections honour an 'epochs' key."
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="Print the benchmark sections available in each suite and exit."
    )

    args = parser.parse_args()

    if args.list_benchmarks:
        print("Available benchmark sections per suite:")
        for suite, names in SUITE_BENCHMARKS.items():
            print(f"  {suite}:")
            for name in names:
                print(f"    - {name}")
        sys.exit(0)

    if args.model is None:
        parser.error("--model is required (unless --list-benchmarks).")

    # Parse and validate the --benchmarks subset. It is suite-specific, so a
    # single concrete suite must be chosen (not "all").
    selected_benchmarks = None
    if args.benchmarks is not None:
        if args.suite == "all":
            parser.error("--benchmarks requires a single --suite (not 'all'), "
                         "because section names are specific to each suite.")
        selected_benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        available = SUITE_BENCHMARKS[args.suite]
        unknown = [b for b in selected_benchmarks if b not in available]
        if unknown:
            parser.error(f"Unknown {args.suite} benchmark(s): {unknown}. "
                         f"Available: {list(available)}")

    # Parse and validate the --benchmark-args overrides. Like --benchmarks, the
    # section names are suite-specific, so a single concrete suite is required.
    benchmark_args = None
    if args.benchmark_args is not None:
        if args.suite == "all":
            parser.error("--benchmark-args requires a single --suite (not 'all'), "
                         "because section names are specific to each suite.")
        try:
            benchmark_args = parse_benchmark_args(
                args.benchmark_args, SUITE_BENCHMARKS[args.suite]
            )
        except ValueError as e:
            parser.error(str(e))

    model_entry = MODEL_REGISTRY[args.model]
    model_class = model_entry["class"]
    model_kwargs = model_entry["default_kwargs"]
    # Per-modality overrides (input mapping, budget) layered on the defaults.
    _mod_kw = model_entry.get("modality_kwargs", {})
    def kwargs_for(modality):
        return {**model_kwargs, **_mod_kw.get(modality, {})}
    supported_modalities = model_entry["modalities"]

    print(f"Starting benchmark run for model: {args.model}")
    print(f"Supported modalities: {supported_modalities}")
    print(f"Target suite: {args.suite}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of trials: {args.n_trials}")
    print("-" * 50)

    # 1. Spatial Suite Execution
    if args.suite in ["spatial", "all"]:
        if "spatial" not in supported_modalities:
            print(f"\n[SKIP] Modality 'spatial' is not supported by model '{args.model}'.")
            if args.suite == "spatial":
                sys.exit(1)
        else:
            print("\nExecuting Spatial Benchmarks...")
            spatial_metrics = run_spatial_pipeline(
                model_class=model_class,
                model_kwargs=kwargs_for("spatial"),
                output_dir=args.output_dir,
                n_trials=args.n_trials,
                benchmarks=selected_benchmarks,
                benchmark_args=benchmark_args,
                epochs=args.epochs
            )
            print_summary_table(spatial_metrics, f"{args.model.upper()} SPATIAL")

    # 2. Symbolic Suite Execution
    if args.suite in ["symbolic", "all"]:
        if "symbolic" not in supported_modalities:
            print(f"\n[SKIP] Modality 'symbolic' is not supported by model '{args.model}'.")
            if args.suite == "symbolic":
                sys.exit(1)
        else:
            symbolic_max_epochs = resolve_symbolic_max_epochs(model_class, args.max_epochs)
            print(f"\nExecuting Symbolic Benchmarks (max convergence epochs: {symbolic_max_epochs})...")
            symbolic_metrics = run_symbolic_pipeline(
                model_class=model_class,
                model_kwargs=kwargs_for("symbolic"),
                output_dir=args.output_dir,
                n_trials=args.n_trials,
                mrr_threshold=args.mrr_threshold,
                max_epochs=symbolic_max_epochs,
                benchmarks=selected_benchmarks,
                benchmark_args=benchmark_args,
                epochs=args.epochs
            )
            print_summary_table(symbolic_metrics, f"{args.model.upper()} SYMBOLIC")

    # 3. Online Symbolic Suite Execution
    if args.suite in ["online_symbolic", "all"]:
        if "online_symbolic" not in supported_modalities:
            print(f"\n[SKIP] Modality 'online_symbolic' is not supported by model '{args.model}'.")
            if args.suite == "online_symbolic":
                sys.exit(1)
        else:
            print("\nExecuting Online Symbolic Benchmarks...")
            online_metrics = run_online_symbolic_pipeline(
                model_class=model_class,
                model_kwargs=kwargs_for("online_symbolic"),
                output_dir=args.output_dir,
                n_trials=args.n_trials,
                noise_scale=args.noise_scale,
                benchmarks=selected_benchmarks,
                benchmark_args=benchmark_args
            )
            print_summary_table(online_metrics, f"{args.model.upper()} ONLINE SYMBOLIC")

    print("Benchmarking run completed.")

if __name__ == "__main__":
    main()
