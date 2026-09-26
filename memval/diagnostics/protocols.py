"""
Optional capability interfaces that the diagnostics consume.

The diagnostics are model-agnostic: they never reach into a model's private
internals. Instead a model advertises what it can expose. Three tiers:

  * BEHAVIOURAL  — needs only the core `HippocampalModel` (`predict_next`/`recall`).
    Every model supports it.
  * REPRESENTATION geometry (Jaccard / RSA / sparsity) — needs
    `named_representations(x) -> {layer: vector}`.
  * WEIGHT space (update-cosine / Fisher / attribution) — needs
    `transition_grads(x_t, x_next) -> {group: array}` and
    `named_parameters() -> {group: array}`.

A diagnostic checks the capability and degrades gracefully (skips its tier) when
a model does not provide it — e.g. a kNN or Hopfield model has no per-transition
weight gradient and simply opts out of the weight tier.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RepresentationProbe(Protocol):
    def named_representations(self, x: np.ndarray) -> dict: ...


@runtime_checkable
class GradientProbe(Protocol):
    def transition_grads(self, x_t: np.ndarray, x_next: np.ndarray) -> dict: ...
    def named_parameters(self) -> dict: ...


def supports_representations(model) -> bool:
    return callable(getattr(model, "named_representations", None))


def supports_gradients(model) -> bool:
    return (callable(getattr(model, "transition_grads", None))
            and callable(getattr(model, "named_parameters", None)))
