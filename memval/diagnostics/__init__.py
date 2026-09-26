"""
memval.diagnostics — model-agnostic network debugging utilities.

Pairs the behavioural view (what the model recalls / forgets) with the
geometric and weight-space views (where in the network it fails), so a
forgetting curve is read next to its mechanistic cause. Utilities are
capability-gated (see `protocols`): a model gets whichever tiers it can expose.

Tiers
-----
behavioural  : one_step_accuracy, recall_accuracy, one_step_fidelity
geometry     : representation_overlap            (needs named_representations)
weights      : update_interference, fisher_diagonal, fisher_attribution
                                                 (needs transition_grads + named_parameters)
report       : ab_forgetting_report, format_report, plot_report
"""

from .protocols import (RepresentationProbe, GradientProbe,
                        supports_representations, supports_gradients)
from .behavioral import (transitions_of, one_step_accuracy, one_step_fidelity,
                         recall_accuracy, retrieval_margin, retrieval_mrr)
from .geometry import representation_overlap
from .weights import (transition_signatures, update_interference, fisher_diagonal,
                      snapshot_parameters, fisher_attribution)
from .report import ab_forgetting_report, format_report, plot_report

__all__ = [
    "RepresentationProbe", "GradientProbe",
    "supports_representations", "supports_gradients",
    "transitions_of", "one_step_accuracy", "one_step_fidelity", "recall_accuracy",
    "retrieval_margin", "retrieval_mrr",
    "representation_overlap",
    "transition_signatures", "update_interference", "fisher_diagonal",
    "snapshot_parameters", "fisher_attribution",
    "ab_forgetting_report", "format_report", "plot_report",
]
