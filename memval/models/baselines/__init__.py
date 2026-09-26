from .hopfield import HopfieldSequenceNetwork
from .eq_prop import EqPropSequenceNetwork
from .original_eqprop import OriginalEqPropSequenceNetwork
from .dg_original_eqprop import DGOriginalEqPropSequenceNetwork
from .dg_xdg_eqprop import DGXdGEqPropSequenceNetwork
from .ewc_dg_xdg_eqprop import EWCDGXdGEqPropSequenceNetwork
from .ewc_original_eqprop import EWCOriginalEqPropSequenceNetwork
from .dg_eqprop import DGEqPropSequenceNetwork
from .cls_dg_eqprop import CLSDGEqPropNetwork
from .ewc_dg_eqprop import EWCDGEqPropSequenceNetwork
from .asymmetric_hopfield import AsymmetricHopfieldNetwork
from .theta_phase import ThetaPhaseSequenceNetwork
from .temporal_pc import TemporalPCNetwork, MultilayerTemporalPCNetwork
from .predictive_recirculation import PredictiveRecirculationNetwork
from .dts_esn import DTSESNSequenceNetwork
from .spiking_eqprop import SpikingEqPropSequenceNetwork
from .bush_stdp import BushSTDPSpikingNetwork, CodecBushNetwork
from .vieth_gaba_stdp import ViethGabaSTDPNetwork, CodecViethNetwork
from .bcpnn_spiking import BCPNNSpikingNetwork, CodecBCPNNNetwork
# --- ARCHIVED / PARKED ---------------------------------------------------
# Not run as benchmark arms, and their outputs are NOT compared against the
# taxonomy's models. Still importable so old notebooks and results keep working.
# Deregistered from bin/run_benchmark.py's MODEL_REGISTRY and from
# docs/figures/.
#
#   GPT2SequenceModel, KNNEpisodicModel -- archived: outside the taxonomy.
#   HiCLSequenceModel                   -- parked (2026-09-01): set aside for
#       now rather than ruled out. Reinstating it is a decision, not a
#       rediscovery; the vendored package under memval/vendor/hicl stays put.
#
# NOT in this set, and deliberately so: DTSESNSequenceNetwork is a crucial part
# of the taxonomy and is compared normally.
#
# Use of KNNEpisodicModel as a *test fixture* (exercising benchmark machinery
# end-to-end, e.g. tests/test_spatial_disambiguation.py) is unaffected -- that
# is harness plumbing, not a model comparison.
from .gpt2_wrapper import GPT2SequenceModel
from .knn_episodic import KNNEpisodicModel
from .hicl_wrapper import HiCLSequenceModel

#: Model class names excluded from benchmark runs and cross-model comparison.
#: Anything building a comparison should filter these out rather than
#: special-casing them at the call site.
ARCHIVED_MODELS = frozenset({
    "GPT2SequenceModel",
    "KNNEpisodicModel",
    "HiCLSequenceModel",
})

__all__ = [
    "HopfieldSequenceNetwork",
    "AsymmetricHopfieldNetwork",
    "ThetaPhaseSequenceNetwork",
    "TemporalPCNetwork",
    "MultilayerTemporalPCNetwork",
    "PredictiveRecirculationNetwork",
    "DTSESNSequenceNetwork",
    "SpikingEqPropSequenceNetwork",
    "BushSTDPSpikingNetwork",
    "CodecBushNetwork",
    "ViethGabaSTDPNetwork",
    "CodecViethNetwork",
    "BCPNNSpikingNetwork",
    "CodecBCPNNNetwork",
    "EqPropSequenceNetwork",
    "OriginalEqPropSequenceNetwork",
    "DGOriginalEqPropSequenceNetwork",
    "DGXdGEqPropSequenceNetwork",
    "EWCDGXdGEqPropSequenceNetwork",
    "EWCOriginalEqPropSequenceNetwork",
    "DGEqPropSequenceNetwork",
    "CLSDGEqPropNetwork",
    "EWCDGEqPropSequenceNetwork",
    # Archived -- see ARCHIVED_MODELS above.
    "GPT2SequenceModel",
    "KNNEpisodicModel",
    "HiCLSequenceModel",
    "ARCHIVED_MODELS",
]



