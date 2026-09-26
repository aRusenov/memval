"""Criterion-referenced exposure — the suite's unit for "how much training".

A section that trains for a fixed number of epochs scores every arm at a
different point on its own learning curve. The epoch count is resolved
CLI > model default > section fallback, and the model defaults in
``bin/run_benchmark.py``'s ``MODEL_REGISTRY`` span **1 to 300** across the
taxonomy, so a span, an MRR or a forgetting delta measured "at the default" is
not comparable between arms. Worse, the difference is invisible in the results
file: two arms report the same metric name from incomparable protocols.

The fix is to stop choosing exposure from outside and let each arm train until
it reaches a stated criterion on a stated read-out. Exposure then becomes a
*result* — ``epochs_to_criterion`` — rather than a parameter, and every metric
downstream of it is measured at the same functional point on every arm.

Three sections already worked this way and are the precedent, not an exception:
``presentation_duration`` (epochs to an MRR threshold), ``schema_consistency``
(trials to a criterion set as a fraction of the arm's own base MRR) and
``tmaze_reversal`` / ``paired_associate`` (trials to criterion per stage).

The criterion read-out must sit **upstream** of the scored one
-------------------------------------------------------------
Settling exposure on the same quantity a section then reports is circular: train
until MRR >= 0.95 and the reported MRR is 0.95 by construction. The criterion
should establish the *antecedent* of the claim, and the metric should measure
something else downstream of it. The shipped pairings follow that rule --
``sequence_length`` settles on CUED recall and scores ROLLOUT span,
``noise_invariance`` settles at sigma = 0 and scores the sigma sweep,
``tmaze_completion`` settles on teacher-forced one-step prediction and scores
open-loop coverage. Where the two coincide, the informative quantity is the
exposure itself and the section should say so rather than reporting a number
pinned to its own threshold.

Censoring is a result too. An arm that never reaches criterion inside the budget
returns ``reached=False``, and every metric measured at that point must be read
against it — which is why each rewired section emits a ``*_criterion_reached``
guard beside its ``*_epochs_to_criterion``.

Incremental vs. from-scratch
----------------------------
Most arms' ``fit_sequence`` accumulates (the schema and continual-chain sections
already depend on this), so the staircase can add one epoch at a time. Some do
not: ``HopfieldSequenceNetwork`` under ``fit_method="projection"`` *replaces*
``W`` with a least-squares solution, so calling it twice is not the same as
training twice. ``incremental=False`` (the default) therefore rebuilds the model
at every checkpoint, which is correct for every arm; pass ``incremental=True``
only for an arm known to accumulate, as a speed-up.
"""
from typing import Any, Callable, Dict, List, Optional

DEFAULT_CRITERION = 0.95
DEFAULT_MAX_EPOCHS = 512


def _ladder(max_epochs: int) -> List[int]:
    """Geometric checkpoints, so bracketing costs log(max_epochs) fits."""
    out, e = [], 1
    while e < max_epochs:
        out.append(e)
        e *= 2
    out.append(max_epochs)
    return out


def epochs_to_criterion(
    make_model: Callable[[], Any],
    fit: Callable[[Any, int], None],
    score: Callable[[Any], float],
    *,
    criterion: float = DEFAULT_CRITERION,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    incremental: bool = False,
    refine: bool = True,
) -> Dict[str, Any]:
    """Smallest exposure at which ``score`` reaches ``criterion``.

    Args:
        make_model: builds a fresh, untrained model. Called once per checkpoint
            unless ``incremental``.
        fit: ``fit(model, epochs)`` — trains for exactly ``epochs`` passes.
        score: ``score(model) -> float`` — the read-out the criterion is on.
        criterion: the threshold ``score`` must reach.
        max_epochs: budget. Exceeding it returns ``reached=False`` and the
            budget as ``epochs``, so the scalar stays plottable while the guard
            records that it is a lower bound.
        incremental: train one step at a time on a single model instead of
            rebuilding. Only for arms whose ``fit_sequence`` accumulates.
        refine: after the geometric ladder brackets the crossing, scan the
            interval below it so the answer is the true minimum rather than the
            checkpoint that happened to pass. Doubling checkpoints otherwise
            overstate the exposure by up to 2x.

    Returns:
        ``{"epochs", "reached", "score", "model", "history"}``. ``model`` is the
        trained model at the returned exposure, so a caller that needs to keep
        measuring does not have to retrain.
    """
    history: List[Dict[str, Any]] = []

    def _at(epochs: int):
        m = make_model()
        fit(m, epochs)
        s = float(score(m))
        history.append({"epochs": epochs, "score": s})
        return m, s

    if incremental:
        model = make_model()
        trained = 0

        def _at(epochs: int):                       # noqa: F811 - deliberate shadow
            nonlocal trained
            if epochs < trained:                    # cannot un-train; rebuild
                m = make_model()
                fit(m, epochs)
                history.append({"epochs": epochs, "score": float(score(m))})
                return m, history[-1]["score"]
            if epochs > trained:
                fit(model, epochs - trained)
                trained = epochs
            s = float(score(model))
            history.append({"epochs": epochs, "score": s})
            return model, s

    best_model, best_epochs, reached = None, None, False
    prev = 0
    for e in _ladder(max_epochs):
        m, s = _at(e)
        if s >= criterion:
            best_model, best_epochs, reached = m, e, True
            break
        prev = e

    if not reached:
        m, s = (best_model, float("nan")) if best_model is not None else _at(max_epochs)
        return {"epochs": max_epochs, "reached": False,
                "score": history[-1]["score"] if history else float("nan"),
                "model": m, "history": history}

    if refine and best_epochs > prev + 1:
        for e in range(prev + 1, best_epochs):
            m, s = _at(e)
            if s >= criterion:
                best_model, best_epochs = m, e
                break

    return {"epochs": best_epochs, "reached": True,
            "score": next(h["score"] for h in reversed(history)
                          if h["epochs"] == best_epochs),
            "model": best_model, "history": history}


#: Sections where exposure is a CONTROL -- the arm is trained to "good enough"
#: and something else is scored. These take the arm's fixed budget from
#: MODEL_REGISTRY (its `n_epochs`), sized 2026-09-05 as 2x the epochs each arm
#: needed to reach criterion on the 7-item list (spatial override where the
#: T-maze needs more). Everything NOT listed here is a RESULT section: the
#: epochs-to-criterion IS the reported number, so it stays on the ladder.
#: A budget is a hyperparameter the arm's builder owns; a result is not.
CONTROL_SECTIONS = frozenset({
    "sequence_length", "multiple_sequences", "noise_invariance", "cue_masking",
    "symbolic_disambiguation", "tmaze_completion", "tmaze_disambiguation",
})


def resolve_exposure(
    benchmark_args: Optional[Dict[str, Dict[str, Any]]],
    section: str,
    benchmark_arg: Callable[..., Any],
    fallback_epochs: int,
    global_epochs: Optional[int] = None,
    model_fixed_epochs: Optional[int] = None,
    force_criterion: bool = False,
) -> Dict[str, Any]:
    """Per-section exposure policy.

    Criterion-referenced by default. ``--benchmark-args <section>:epochs=N``
    still pins a fixed budget, which is how an old number is reproduced or a
    section is deliberately held at a set exposure; ``criterion`` and
    ``max_epochs`` tune the staircase.
    """
    # Precedence: per-section CLI > global --epochs > registry budget (control
    # sections only) > criterion ladder. `force_criterion` lets a section that
    # uses exposure as both control and result keep the ladder for the result.
    fixed = benchmark_arg(benchmark_args, section, "epochs", None)
    source = "cli:section" if fixed is not None else None
    if fixed is None and global_epochs is not None:
        fixed, source = global_epochs, "cli:global"
    if (fixed is None and not force_criterion and section in CONTROL_SECTIONS
            and model_fixed_epochs is not None):
        fixed, source = model_fixed_epochs, "registry"
    return {
        "mode": "fixed" if fixed is not None else "criterion",
        "epochs": int(fixed) if fixed is not None else None,
        "source": source,
        "criterion": float(benchmark_arg(benchmark_args, section, "criterion",
                                         DEFAULT_CRITERION)),
        "max_epochs": int(benchmark_arg(benchmark_args, section, "max_epochs",
                                        DEFAULT_MAX_EPOCHS)),
        "fallback_epochs": int(fallback_epochs),
    }
