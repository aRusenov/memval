from typing import Any, Optional, Union

import numpy as np

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode


class HopfieldSequenceNetwork(HippocampalModel, OnlineTrainable):
    """
    A continuous sequence-learning associative memory network (asymmetric
    Hopfield Network).
    
    Supports various fitting methods:
    - projection: One-shot exact least-squares fit (Pseudo-inverse).
    - hebbian: Static outer-product rule (W += x_next @ x.T).
    - delta: Iterative error-correction (Delta Rule/LMS).
    """

    #: `recall` feeds the raw prediction back via `self.current_state`.
    rollout_mode = RolloutMode.OBSERVATION

    #: Config-dependent, so declared conservatively False: only some
    #: (activation, fit_method) combinations make the batch path the streamed
    #: rule. Measured (see tests/test_online_capability.py::
    #: test_hopfield_equivalence_is_config_dependent):
    #:
    #:   ============ ============ ============ =========
    #:   activation   projection   hebbian      delta
    #:   ============ ============ ============ =========
    #:   linear       match        DIFFER       match
    #:   sign         match        DIFFER       DIFFER
    #:   tanh         match        DIFFER       match
    #:   ============ ============ ============ =========
    #:
    #: False is the safe direction: it tells a reader not to assume a
    #: batch-vs-streamed contrast on this arm is pure ingestion.
    online_equivalent = False

    def __init__(self, n_features: int, learning_rate: float = 0.1, 
                 activation: str = "linear", fit_method: str = "auto", **kwargs):
        """
        Initialize the Hopfield Sequence Network.

        Args:
            n_features (int): Dimensionality of the patterns.
            learning_rate (float): Step size for delta/hebbian updates.
            activation (str): 'linear', 'sign' (±1), 'tanh', or 'relu'.
            fit_method (str): 'projection' (default for linear), 'hebbian', 'delta', or 'auto'.
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.activation_type = activation
        self.fit_method = fit_method
        
        # Asymmetric weight matrix for sequence transitions
        self.W = np.zeros((n_features, n_features))
        
        # Transient state for ongoing sequence processing
        self.current_state = np.zeros(n_features)
        
        # Online training state
        self._last_train_event = None
        self._X_pre_buf = []
        self._X_post_buf = []

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation_type == "linear":
            return x
        elif self.activation_type == "sign":
            # Map 0s to -1 during computation to ensure strong inhibition if using bipolar
            return np.where(x <= 0, -1.0, 1.0)
        elif self.activation_type == "tanh":
            return np.tanh(x)
        elif self.activation_type == "relu":
            return np.maximum(0, x)
        return x

    def fit_sequence(self, sequence_data: np.ndarray, epochs: int = 1, **kwargs):
        """
        Train the model on the sequence.

        Args:
            sequence_data: Target array of shape (seq_length, n_features).
            epochs: Number of iterations (only for 'delta' rule).
        """
        seq_length = sequence_data.shape[0]
        if seq_length < 2:
            return

        method = self.fit_method
        if method == "auto":
            method = "projection" if self.activation_type == "linear" else "hebbian"

        if method == "projection":
            # Exact least-squares fit (one-shot)
            X_pre = sequence_data[:-1].T
            X_post = sequence_data[1:].T
            self.W = X_post @ np.linalg.pinv(X_pre)
            return

        if method == "hebbian":
            norm_factor = max(1, seq_length - 1)
            for t in range(seq_length - 1):
                pre_state = sequence_data[t].reshape(-1, 1)
                post_state = sequence_data[t + 1].reshape(-1, 1)
                
                # Bipolar projection if requested
                if self.activation_type == "sign":
                    pre_state = np.where(pre_state <= 0, -1.0, 1.0)
                    post_state = np.where(post_state <= 0, -1.0, 1.0)

                self.W += (self.learning_rate / norm_factor) * (post_state @ pre_state.T)
            return

        if method == "delta":
            # Iterative LMS rule
            for _ in range(epochs):
                for t in range(seq_length - 1):
                    pre_state = sequence_data[t].reshape(-1, 1)
                    post_state = sequence_data[t + 1].reshape(-1, 1)
                    
                    prediction = self.W @ pre_state
                    error = post_state - prediction
                    # Update rule: dW = eta * error * input^T
                    self.W += self.learning_rate * (error @ pre_state.T)

    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None, **kwargs):
        """
        Train the model online on a single event.
        """
        method = self.fit_method
        if method == "auto":
            method = "projection" if self.activation_type == "linear" else "hebbian"

        if self._last_train_event is not None:
            pre_state = self._last_train_event.reshape(-1, 1)
            post_state = event.reshape(-1, 1)

            # Apply bipolar projection if requested for hebbian and delta
            if method in ["hebbian", "delta"] and self.activation_type == "sign":
                pre_state = np.where(pre_state <= 0, -1.0, 1.0)
                post_state = np.where(post_state <= 0, -1.0, 1.0)

            if method == "hebbian":
                # Outer product rule
                self.W += self.learning_rate * (post_state @ pre_state.T)

            elif method == "delta":
                # LMS update: dW = learning_rate * error * input^T
                prediction = self.W @ pre_state
                error = post_state - prediction
                self.W += self.learning_rate * (error @ pre_state.T)

            elif method == "projection":
                # Buffer and recompute pseudo-inverse
                self._X_pre_buf.append(self._last_train_event)
                self._X_post_buf.append(event)
                
                X_pre = np.array(self._X_pre_buf).T
                X_post = np.array(self._X_post_buf).T
                self.W = X_post @ np.linalg.pinv(X_pre)

        # Update last train event
        self._last_train_event = event.copy()

    def predict_next(self, current_event: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict the next target given the current observation event.
        """
        state_vec = current_event.reshape(-1, 1)
        
        # Internal bipolar mapping only if activation type is 'sign'
        if self.activation_type == "sign":
            state_vec = np.where(state_vec <= 0, -1.0, 1.0)
            
        activation_input = self.W @ state_vec
        next_event = self._activate(activation_input).flatten()
        
        # Post-process back to [0, 1] range if we used bipolar internally
        if self.activation_type == "sign":
            next_event = np.where(next_event < 0, 0.0, 1.0)
            
        self.current_state = next_event
        return self.current_state

    def recall(self, prompt_event: np.ndarray, length: int) -> np.ndarray:
        """
        Extrapolates auto-regressively starting from the end of the prompt sequence.
        """
        if prompt_event.ndim > 1:
            current = prompt_event[-1]
        else:
            current = prompt_event
            
        self.current_state = current
        recalled = np.zeros((length, self.n_features))
        
        for t in range(length):
            next_val = self.predict_next(self.current_state)
            recalled[t] = next_val
            self.current_state = next_val
            
        return recalled

    def get_latent_state(self) -> np.ndarray:
        return self.W.copy()

    def reset_context(self):
        self.current_state = np.zeros(self.n_features)
        self._last_train_event = None
        self._X_pre_buf = []
        self._X_post_buf = []
