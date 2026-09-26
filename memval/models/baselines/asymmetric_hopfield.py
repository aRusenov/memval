from typing import Any, Optional, Union

import numpy as np

from ..base import HippocampalModel
from ..capabilities import RolloutMode


class AsymmetricHopfieldNetwork(HippocampalModel):
    """
    A continuous sequence-learning associative memory network (asymmetric
    Hopfield Network).
    """

    #: `recall` feeds the raw prediction back via `self.current_state`.
    rollout_mode = RolloutMode.OBSERVATION

    def __init__(self, n_features: int, n_context: int = 0, learning_rate: float = 0.1, **kwargs):
        """
        Initialize the Hopfield Sequence Network.

        Args:
            n_features (int): Dimensionality of the endogenous patterns.
            n_context (int): Dimensionality of the exogenous clamped context.
            learning_rate (float): Step size for delta/hebbian updates.
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_context = n_context
        # Default passes per ``fit_sequence`` call when the caller does not name
        # one. Stored (2026-09-03) because MODEL_REGISTRY supplies n_epochs to
        # every arm's constructor and this one silently dropped it into
        # **kwargs, while every other arm except DTSESNSequenceNetwork stored
        # it. SpatialReversalBenchmark's contract is explicitly the CONSTRUCTOR
        # value -- "a trial is one presentation ... the epochs-per-presentation
        # is whatever the model was constructed with, held equal across arms" --
        # and it calls fit_sequence(train_seq) with no epochs kwarg. So this arm
        # was training 1 epoch per trial where the section asked for 3 and every
        # storing arm did 3: a 3x exposure gap on the exact quantity that
        # section reports (trials to criterion). Sections that pass epochs=
        # explicitly are unaffected -- the named argument still wins.
        self.n_epochs = int(kwargs.get("n_epochs", 1))
        self.learning_rate = learning_rate
        
        # Asymmetric weight matrix for sequence transitions
        self.W = np.zeros((n_features, n_features + n_context))
        
        # Transient state for ongoing sequence processing
        self.current_state = np.zeros(n_features)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        # ReLU
        return np.maximum(0, x)

    def fit_sequence(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, epochs: Optional[int] = None, **kwargs):
        """
        Train the model on the sequence.

        Args:
            sequence_data: Target array of shape (seq_length, n_features).
            context_data: Exogenous context of shape (seq_length, n_context).
            epochs: Number of passes. None falls back to ``self.n_epochs``
                (the constructor value), which is what
                ``SpatialReversalBenchmark`` relies on.
        """
        seq_length = sequence_data.shape[0]
        if seq_length < 2:
            return

        n_passes = self.n_epochs if epochs is None else int(epochs)
        # Iterative LMS rule
        for _ in range(n_passes):
            for t in range(seq_length - 1):
                s_t = sequence_data[t].reshape(-1, 1)
                if self.n_context > 0 and context_data is not None:
                    c_t = context_data[t].reshape(-1, 1)[: self.n_context]
                    pre_state = np.concatenate([s_t, c_t], axis=0)
                else:
                    pre_state = s_t
                
                post_state = sequence_data[t + 1].reshape(-1, 1)
                
                prediction = self.W @ pre_state
                error = post_state - prediction
                # Update rule: dW = eta * error * input^T
                self.W += self.learning_rate * (error @ pre_state.T)

    def predict_next(self, current_event: np.ndarray, current_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Predict the next target given the current observation event and optional context.
        """
        s_t = current_event.reshape(-1, 1)
        if self.n_context > 0 and current_context is not None:
            c_t = np.asarray(current_context).reshape(-1, 1)[: self.n_context]
            pre_state = np.concatenate([s_t, c_t], axis=0)
        else:
            pre_state = s_t
            
        activation_input = self.W @ pre_state
        next_event = self._activate(activation_input).flatten()
            
        self.current_state = next_event
        return self.current_state

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Extrapolates auto-regressively starting from the end of the prompt sequence,
        optionally using a clamped sequence of future contexts.
        """
        if prompt_event.ndim > 1:
            current = prompt_event[-1]
        else:
            current = prompt_event
            
        self.current_state = current
        recalled = np.zeros((length, self.n_features))
        
        for t in range(length):
            c_t = prompt_context[t] if prompt_context is not None else None
            next_val = self.predict_next(self.current_state, current_context=c_t)
            recalled[t] = next_val
            self.current_state = next_val
            
        return recalled

    def get_latent_state(self) -> dict:
        return {"W": self.W.copy()}

    def reset_context(self):
        self.current_state = np.zeros(self.n_features)
