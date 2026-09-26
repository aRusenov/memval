from typing import Any, Dict, Optional

import numpy as np

from ..models.base import HippocampalModel
from .base import Benchmark


class PatternCompletionBenchmark(Benchmark):
    """
    Evaluates the capacity of a model to reconstruct a full memory 
    sequence from a partial or degraded prompt.
    """

    def __init__(self, metric: str = "mse"):
        """
        Initialize the benchmark.

        Args:
            metric (str): 'mse' for continuous (spatial) reconstruction,
                          'accuracy' for categorical/discrete reconstruction.
        """
        self.metric = metric

    def evaluate(self, model: HippocampalModel, datasets: np.ndarray, 
                 prompt_fraction: float = 0.3, **kwargs) -> Dict[str, float]:
        """
        Runs the pattern completion assessment.

        Args:
            model (HippocampalModel): The biological compute model.
            datasets (np.ndarray): Tensor of ground truth sequences (n_sequences, T, features).
            prompt_fraction (float): The fraction of the sequence [0, 1] to use as the cue.
            
        Returns:
            Dict containing the reconstruction performance metrics.
        """
        n_seqs, seq_len = datasets.shape[0], datasets.shape[1]
        prompt_len = max(1, int(seq_len * prompt_fraction))
        recall_len = seq_len - prompt_len
        
        reconstruction_errors = []

        for i in range(n_seqs):
            sequence = datasets[i]
            
            # The cue/prompt is the beginning fraction of the sequence
            prompt = sequence[:prompt_len]
            ground_truth_continuation = sequence[prompt_len:]
            
            # Reset internal state to avoid carrying over context from previous evaluations
            model.reset_context()

            # For models that require sequential cueing, we might feed the prompt incrementally.
            # In our general API, `recall` can take the full prompt and length to extrapolate.
            reconstructed_continuation = model.recall(prompt, length=recall_len)
            
            # Ensure model output matches expected length shape
            reconstructed_continuation = np.array(reconstructed_continuation)
            
            error = self._calculate_error(ground_truth_continuation, reconstructed_continuation)
            reconstruction_errors.append(error)

        avg_error = float(np.mean(reconstruction_errors))
        
        return {
            f"pattern_completion_{self.metric}": avg_error
        }

    def _calculate_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Helper to calculate designated metric."""
        if self.metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif self.metric == "accuracy":
            # For one-hot vectors or integer arrays, measuring exact match per timestep
            if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
                # Assuming one-hot if last dim > 1
                y_true_argmax = np.argmax(y_true, axis=-1)
                y_pred_argmax = np.argmax(y_pred, axis=-1)
                return np.mean(y_true_argmax == y_pred_argmax)
            return np.mean(y_true == y_pred)
        else:
            raise ValueError(f"Unknown metric {self.metric}")
