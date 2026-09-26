"""Minimal stand-ins for the three `artemis` helpers the vendored code uses.

Upstream (`QUVA-Lab/spiking-eqprop`) imports `artemis`, a large research
utility library (experiment framework, plotting, dataset loaders) that the
ported modules only touch for three tiny functions. Vendoring those three
here keeps the port dependency-free while leaving the upstream sources
otherwise byte-identical apart from their import lines.
"""

from typing import Optional, Union

import numpy as np


def get_rng(seed: Optional[Union[int, np.random.RandomState]] = None) -> np.random.RandomState:
    """`artemis.general.numpy_helpers.get_rng`: coerce a seed into an RNG."""
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)


def izip_equal(*iterables):
    """`artemis.general.should_be_builtins.izip_equal`: zip, but assert equal length."""
    iterators = [iter(it) for it in iterables]
    while True:
        stops, values = zip(*[_next_or_stop(it) for it in iterators])
        if all(stops):
            return
        assert not any(stops), "Iterables passed to izip_equal did not have the same length."
        yield values


def _next_or_stop(iterator):
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


def bad_value(value, explanation: str = None):
    """`artemis.general.should_be_builtins.bad_value`: raise on an unhandled case."""
    raise ValueError(
        f"Bad Value: {value}" + (f": {explanation}" if explanation is not None else "")
    )
