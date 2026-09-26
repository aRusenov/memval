from abc import ABC, abstractmethod
from typing import Any, Dict

from ..models.base import HippocampalModel


class Benchmark(ABC):
    """
    Abstract base class for all episodic memory benchmarks.
    """

    @abstractmethod
    def evaluate(self, model: HippocampalModel, datasets: Any, **kwargs) -> Dict[str, float]:
        """
        Evaluate the integrated HippocampalModel on given dataset(s).

        Args:
            model (HippocampalModel): The model to test.
            datasets (Any): Generated sequences or data required for testing.
            **kwargs: Extra hyperparameters for the evaluation procedure.

        Returns:
            Dict[str, float]: Dictionary of metric names and their corresponding values.
        """
        pass
