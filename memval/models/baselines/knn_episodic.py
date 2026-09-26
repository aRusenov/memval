import numpy as np
from typing import Any, List, Optional, Tuple, Union
from ..base import HippocampalModel
from ..capabilities import RolloutMode

class KNNEpisodicModel(HippocampalModel):
    """
    A non-parametric k-NN episodic memory model that stores sequences exactly
    and recalls them using nearest-neighbor prefix matching over events and context.
    
    This serves as an upper bound baseline representing perfect memory retention 
    with zero biological fidelity.
    """

    #: `recall` replays the prompt through `predict_next`, which accumulates
    #: `history_events` -- this arm's `predict_next` is deliberately NOT pure,
    #: and the whole history is matched against the stored sequences.
    prompt_conditioned = True

    #: `recall` feeds `recalled_embs[-1]` (the raw prediction) back; its
    #: `decode_prediction` override is a pass-through.
    rollout_mode = RolloutMode.OBSERVATION

    def __init__(self, n_features: int, **kwargs):
        """
        Initialize the KNN Episodic Model.

        Args:
            n_features (int): Dimensionality of the patterns.
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.stored_sequences: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        
        # Internal state tracking for transient context
        self.current_t = 0
        self.history_events: List[np.ndarray] = []
        self.history_contexts: List[Optional[np.ndarray]] = []

    def fit_sequence(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, **kwargs):
        """
        Store a single sequence.
        
        Args:
            sequence_data: Target array of shape (seq_length, n_features).
            context_data: Optional array of shape (seq_length, context_features).
        """
        self.stored_sequences.append((sequence_data.copy(), context_data.copy() if context_data is not None else None))

    def fit_sequences(self, sequences: List[Tuple[np.ndarray, Optional[np.ndarray]]], **kwargs):
        """
        Store multiple sequences.
        """
        for seq, ctx in sequences:
            self.fit_sequence(seq, ctx, **kwargs)

    def predict_next(self, current_event: np.ndarray, current_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Predict the next event in the sequence given the current event and context.
        """
        # Slice history to match self.current_t (handles manual/benchmark overrides of current_t)
        if len(self.history_events) > self.current_t:
            self.history_events = self.history_events[:self.current_t]
        self.history_events.append(current_event.flatten())

        if len(self.history_contexts) > self.current_t:
            self.history_contexts = self.history_contexts[:self.current_t]
        self.history_contexts.append(current_context.flatten() if current_context is not None else None)

        history_len = len(self.history_events)

        best_seq_idx = -1
        best_score = -1.0

        for seq_idx, (s_events, s_contexts) in enumerate(self.stored_sequences):
            # The stored sequence must have at least history_len + 1 elements to predict the next event
            if s_events.shape[0] <= history_len:
                continue

            total_sim = 0.0
            for i in range(history_len):
                # Calculate event cosine similarity
                ev_sim = self._cosine_similarity(self.history_events[i], s_events[i])
                
                # Calculate context cosine similarity if available
                ctx_sim = 1.0
                if s_contexts is not None and self.history_contexts[i] is not None:
                    ctx_sim = self._cosine_similarity(self.history_contexts[i], s_contexts[i])
                
                total_sim += ev_sim + ctx_sim

            avg_sim = total_sim / history_len
            if avg_sim > best_score:
                best_score = avg_sim
                best_seq_idx = seq_idx

        # Advance internal tracking
        self.current_t += 1

        if best_seq_idx != -1:
            # Return the next event from the best matching sequence
            return self.stored_sequences[best_seq_idx][0][history_len].copy()
        else:
            # Fallback to zero vector if no match found
            return np.zeros(self.n_features)

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Recall a sequence from a partial prompt.
        """
        if length <= 0:
            return np.empty((0, self.n_features))

        self.reset_context()

        # Handle prompt loading
        # We process all prompt events and get the first prediction
        if prompt_event.ndim > 1:
            prompt_len = prompt_event.shape[0]
            first_pred = None
            for i in range(prompt_len):
                self.current_t = i
                c_i = None
                if prompt_context is not None:
                    if prompt_context.ndim > 1:
                        c_i = prompt_context[i] if i < len(prompt_context) else prompt_context[-1]
                    else:
                        c_i = prompt_context
                
                first_pred = self.predict_next(prompt_event[i], current_context=c_i)
        else:
            self.current_t = 0
            c_0 = prompt_context[0] if (prompt_context is not None and prompt_context.ndim > 1) else prompt_context
            first_pred = self.predict_next(prompt_event, current_context=c_0)
            prompt_len = 1

        recalled_embs = [first_pred]
        for t in range(1, length):
            self.current_t = prompt_len + t - 1
            
            c_t = None
            if prompt_context is not None:
                if prompt_context.ndim > 1:
                    idx = prompt_len + t - 1
                    c_t = prompt_context[idx] if idx < len(prompt_context) else prompt_context[-1]
                else:
                    c_t = prompt_context

            next_emb = self.predict_next(recalled_embs[-1], current_context=c_t)
            recalled_embs.append(next_emb)

        return np.array(recalled_embs)


    def decode_prediction(self, raw_prediction: np.ndarray) -> np.ndarray:
        # Pass-through by default
        return raw_prediction.copy()

    def get_latent_state(self) -> dict:
        # KNN doesn't have internal neural weights, so return the stored sequences
        return {"stored_sequences": self.stored_sequences}

    def reset_context(self):
        self.current_t = 0
        self.history_events = []
        self.history_contexts = []

    def clear_memory(self):
        self.stored_sequences = []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
