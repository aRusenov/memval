"""Vendored core of `QUVA-Lab/spiking-eqprop`.

Reference implementation for:

    P. O'Connor, E. Gavves, M. Welling (2019).
    "Training a Spiking Neural Network with Equilibrium Propagation."
    AISTATS 2019.  https://github.com/QUVA-Lab/spiking-eqprop

Only the two library modules are vendored (`eq_prop`, `quantized_eqprop`);
the upstream demo/experiment scripts depend on `artemis`, `skopt` and MNIST
loaders and are not needed here. The vendored sources are unmodified except
for their import lines, which now point at `_artemis` (see that module).

Upstream commit: d56a0fe6559083dcd148243aaa1f4ccea14f6ce8

What the two modules provide
----------------------------
`eq_prop`
    Scellier-&-Bengio Equilibrium Propagation with a *functional* layer
    interface: a layer is an immutable dataclass that, when called with its
    neighbours' outputs, returns a **new** layer state. `SimpleLayerController`
    is the ordinary real-valued neuron; `run_eqprop_training_update` runs the
    negative (free) and positive (nudged) phases and applies the two-phase
    contrastive weight update.

`quantized_eqprop`
    The paper's contribution: `EncodingDecodingNeuronLayer`, a drop-in
    replacement for `SimpleLayerController` in which neurons communicate with
    their neighbours only through **binary spikes**. Each neuron encodes its
    rate `rho(potential)` with a sigma-delta (or stochastic / threshold)
    quantizer, and the receiving neuron reconstructs an estimate with a
    predictive (leaky-average) decoder. Because the layer interface is
    unchanged, the same EP dynamics and update rule run on spikes.

Known upstream defect (left unfixed, so the vendored source stays faithful)
--------------------------------------------------------------------------
``SecondOrderSigmaDeltaQuantizer.__call__`` reads ``self.phi_1``/``self.phi_2``
while its dataclass fields are ``phi1``/``phi2``, so selecting
``quantizer='second_order_sd'`` raises ``AttributeError`` on the first tick. It
is dead code upstream -- no experiment in the paper selects it. The MemVal arm
rejects that option up front with an explanatory error.
"""

from .eq_prop import (
    IDynamicLayer,
    LayerParams,
    PropDirectionOptions,
    SimpleLayerController,
    initialize_params,
    initialize_states,
    output_from_state,
    rho,
    run_eqprop_training_update,
    run_inference,
)
from .quantized_eqprop import (
    ConstantStepSizer,
    EncodingDecodingNeuronLayer,
    IdentityFunction,
    OptimalStepSizer,
    PredictiveDecoder,
    PredictiveEncoder,
    ScheduledStepSizer,
    SigmaDeltaQuantizer,
    StochasticQuantizer,
    ThresholdQuantizer,
    create_quantizer,
    create_step_sizer,
)

__all__ = [
    "IDynamicLayer",
    "LayerParams",
    "PropDirectionOptions",
    "SimpleLayerController",
    "initialize_params",
    "initialize_states",
    "output_from_state",
    "rho",
    "run_eqprop_training_update",
    "run_inference",
    "ConstantStepSizer",
    "EncodingDecodingNeuronLayer",
    "IdentityFunction",
    "OptimalStepSizer",
    "PredictiveDecoder",
    "PredictiveEncoder",
    "ScheduledStepSizer",
    "SigmaDeltaQuantizer",
    "StochasticQuantizer",
    "ThresholdQuantizer",
    "create_quantizer",
    "create_step_sizer",
]
