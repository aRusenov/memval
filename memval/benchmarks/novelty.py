from typing import Dict

import numpy as np

from ..models.base import HippocampalModel
from .base import Benchmark


class NoveltyBenchmark(Benchmark):
    """
    Evaluates the capacity of a model to detect novel objects or 
    displaced objects (Novel Object Recognition / Object Location Memory).
    """

    def __init__(self, metric: str = "mse"):
        self.metric = metric

    def evaluate(self, model: HippocampalModel, 
                 familiar_datasets: np.ndarray, 
                 test_datasets: np.ndarray, 
                 object_dim_start: int = 2,
                 **kwargs) -> Dict[str, float]:
        """
        Runs the novelty detection assessment.
        Fits the model on the familiar dataset, then computes prediction error
        on the test dataset specifically during moments of object interaction.

        Args:
            model (HippocampalModel): The biological compute model.
            familiar_datasets (np.ndarray): Tensor of ground truth familiar sequences (n_sequences, T, features).
            test_datasets (np.ndarray): Tensor of test sequences (novel or displaced) (n_sequences, T, features).
            object_dim_start (int): The index where object one-hot features begin in the vector.
            
        Returns:
            Dict containing the novelty score (prediction error).
        """
        n_seqs, seq_len, features = familiar_datasets.shape
        
        novelty_scores = []

        for i in range(n_seqs):
            # Reset context and train on familiar
            model.reset_context()
            model.fit_sequence(familiar_datasets[i])
            
            # Reset context for testing
            model.reset_context()
            
            test_seq = test_datasets[i]
            seq_novelty = []
            
            # Provide first step to start
            if hasattr(model, "predict_next"):
                # Track error across trajectory
                current_event = test_seq[0]
                model.predict_next(current_event) # process start step
                
                for t in range(1, seq_len):
                    pred = model.predict_next(current_event)
                    actual = test_seq[t]
                    
                    # Compute error specifically on object dimensions
                    if self.metric == "mse":
                        obj_error = np.mean((pred[object_dim_start:] - actual[object_dim_start:]) ** 2)
                    else:
                        obj_error = np.mean((pred[object_dim_start:] != actual[object_dim_start:]).astype(float))
                        
                    # Only score novelty when an object is actually present in the test sequence
                    # or was present in the prediction (to catch missing objects)
                    if np.sum(actual[object_dim_start:]) > 0 or np.sum(pred[object_dim_start:]) > 0.5:
                        seq_novelty.append(obj_error)
                        
                    current_event = actual
            
            if len(seq_novelty) > 0:
                novelty_scores.append(np.mean(seq_novelty))
            else:
                novelty_scores.append(0.0)

        avg_novelty = float(np.mean(novelty_scores))
        
        return {
            f"novelty_score_{self.metric}": avg_novelty
        }
