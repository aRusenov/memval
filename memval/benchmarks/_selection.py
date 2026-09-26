"""Helpers for selecting a subset of benchmarks within a suite pipeline.

Each ``run_*_pipeline`` exposes a canonical, ordered list of the benchmark
sections it can run (e.g. ``SYMBOLIC_BENCHMARKS``) and accepts a ``benchmarks``
argument. Passing a subset lets callers skip the expensive, irrelevant sections
and iterate quickly on a single one.
"""
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def resolve_benchmarks(
    requested: Optional[Iterable[str]],
    available: Sequence[str],
    suite: str,
) -> List[str]:
    """Validate and normalize a requested benchmark subset for a suite.

    ``requested`` of ``None`` (or the single token ``"all"``) selects every
    benchmark in ``available``. Otherwise the request is filtered to the named
    benchmarks, preserving ``available``'s canonical order and de-duplicating.
    Unknown names raise ``ValueError`` listing the valid options so a typo fails
    loudly instead of silently running nothing.
    """
    if requested is None:
        return list(available)
    requested = list(requested)
    if requested == ["all"]:
        return list(available)
    unknown = [b for b in requested if b not in available]
    if unknown:
        raise ValueError(
            f"Unknown {suite} benchmark(s): {unknown}. "
            f"Available: {list(available)}"
        )
    requested_set = set(requested)
    return [b for b in available if b in requested_set]


def resolve_benchmark_args(
    benchmark_args: Optional[Mapping[str, Mapping[str, Any]]],
    available: Sequence[str],
    suite: str,
) -> Dict[str, Dict[str, Any]]:
    """Validate and normalize a per-benchmark override mapping for a suite.

    ``benchmark_args`` maps a benchmark section name to a dict of parameter
    overrides for that section, e.g. ``{"spatial_sequence": {"epochs": 200}}``.
    ``None`` (or empty) means no overrides. Unknown section names raise
    ``ValueError`` listing the valid options, so a typo fails loudly instead of
    silently doing nothing. Which keys a section honours is documented on the
    section itself (currently ``epochs`` on the trainable sections).
    """
    if not benchmark_args:
        return {}
    unknown = [b for b in benchmark_args if b not in available]
    if unknown:
        raise ValueError(
            f"Unknown {suite} benchmark(s) in benchmark_args: {unknown}. "
            f"Available: {list(available)}"
        )
    return {b: dict(v) for b, v in benchmark_args.items()}


def benchmark_arg(
    benchmark_args: Optional[Mapping[str, Mapping[str, Any]]],
    name: str,
    key: str,
    default: Any,
) -> Any:
    """Fetch a single per-benchmark override ``benchmark_args[name][key]``,
    falling back to ``default`` when the section or key is absent."""
    if not benchmark_args:
        return default
    return benchmark_args.get(name, {}).get(key, default)


def resolve_epochs_default(
    cli_epochs: Optional[int],
    model_kwargs: Optional[Mapping[str, Any]],
    fallback: Any,
) -> Any:
    """Resolve the epoch count for a trainable section by precedence:

        CLI ``--epochs`` > model default > section ``fallback``

    where "model default" is an ``epochs`` (or ``n_epochs``) value carried in the
    model's ``model_kwargs``. Pass the result as the ``default`` argument to
    ``benchmark_arg`` so a per-section ``--benchmark-args`` override (the most
    specific control) still wins over all three.
    """
    if cli_epochs is not None:
        return cli_epochs
    if model_kwargs:
        for key in ("epochs", "n_epochs"):
            if key in model_kwargs:
                return model_kwargs[key]
    return fallback
