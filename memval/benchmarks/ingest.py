"""The single seam where a benchmark chooses an ingestion regime.

Every suite that varies batch-vs-streamed ingestion goes through ``ingest``.
Putting the capability check here -- rather than discovering it by catching an
exception at each sweep point -- is what lets a benchmark distinguish

  * "this arm has no streaming interface"  -> the sweep point is NOT APPLICABLE
  * "this arm streamed and failed"         -> the sweep point is a real 0.0

which were previously the same number in the results file.
"""

from typing import Any, Optional

import numpy as np

from ..models.capabilities import OnlineTrainable, UnsupportedRegime

#: The ingestion regimes ``ingest`` accepts. This is a *condition inside a
#: suite*, never a property of the arm -- an arm that declares OnlineTrainable
#: must be runnable under both.
REGIMES = ("batch", "streamed")


def ingest(
    model: Any,
    sequence_data: np.ndarray,
    regime: str = "batch",
    context_data: Optional[np.ndarray] = None,
    **kwargs,
) -> None:
    """Present one sequence to ``model`` under the requested regime.

    Args:
        model: The arm. Streaming requires it to declare ``OnlineTrainable``.
        sequence_data: Shape ``(Time, Features)``.
        regime: ``"batch"`` -> one ``fit_sequence`` call; ``"streamed"`` -> one
            ``fit_event`` per timestep, fenced by ``on_event_boundary`` at both
            seams so no transition forms across the sequence edge.
        context_data: Optional shape ``(Time, ContextFeatures)``, indexed
            per-event when streaming and passed whole when batching.
        **kwargs: Forwarded to ``fit_sequence`` (batch) or each ``fit_event``.

    Raises:
        UnsupportedRegime: ``regime="streamed"`` on an arm that does not declare
            ``OnlineTrainable``. Callers must record the point as not
            applicable, never as a score.
        ValueError: unknown regime.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown ingestion regime {regime!r}; expected one of {REGIMES}")

    if regime == "batch":
        # Keyword, not positional: several arms (original_eqprop, hopfield,
        # eq_prop) declare `fit_sequence(self, sequence_data, **kwargs)` with no
        # `context_data` parameter, so a positional pass raises TypeError. As a
        # keyword it lands in their **kwargs and is ignored, which is what those
        # arms do with context anyway.
        if context_data is not None:
            kwargs["context_data"] = context_data
        model.fit_sequence(sequence_data, **kwargs)
        return

    if not isinstance(model, OnlineTrainable):
        raise UnsupportedRegime(
            f"{type(model).__name__} does not declare OnlineTrainable, so it has "
            f"no streamed (event-by-event) regime."
        )

    model.on_event_boundary()
    for t, event in enumerate(sequence_data):
        model.fit_event(event, None if context_data is None else context_data[t], **kwargs)
    model.on_event_boundary()
