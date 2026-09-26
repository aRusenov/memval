"""
Benchmark suite defining standard evaluation modes for memory models.
"""
from .base import Benchmark
from .ingest import REGIMES, ingest
from .pattern_completion import PatternCompletionBenchmark
from .novelty import NoveltyBenchmark
from .spatial_pipeline import run_spatial_pipeline
from .symbolic_pipeline import run_symbolic_pipeline

__all__ = [
    "Benchmark",
    "ingest",
    "REGIMES",
    "PatternCompletionBenchmark",
    "NoveltyBenchmark",
    "run_spatial_pipeline",
    "run_symbolic_pipeline",
]
