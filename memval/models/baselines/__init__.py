# The model zoo: the six arms this version of MemVal compares. Retired arms
# (spiking, DG/EWC EqProp variants, GPT-2, KNN, HiCL, ...) live at git tag
# `archive/pre-cleanup`.
from .asymmetric_hopfield import AsymmetricHopfieldNetwork
from .original_eqprop import OriginalEqPropSequenceNetwork
from .theta_phase import ThetaPhaseSequenceNetwork
from .temporal_pc import TemporalPCNetwork, MultilayerTemporalPCNetwork
from .predictive_recirculation import PredictiveRecirculationNetwork
from .dts_esn import DTSESNSequenceNetwork

__all__ = [
    "AsymmetricHopfieldNetwork",
    "OriginalEqPropSequenceNetwork",
    "ThetaPhaseSequenceNetwork",
    "TemporalPCNetwork",
    "MultilayerTemporalPCNetwork",
    "PredictiveRecirculationNetwork",
    "DTSESNSequenceNetwork",
]
