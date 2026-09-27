from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np

from .capabilities import RolloutMode


class HippocampalModel(ABC):
    """
    Abstract base class for standardized integration of computational
    hippocampal models into the MemVal framework.
    """

    #: What this arm's ``recall()`` feeds back at each autoregressive step.
    #: **Every concrete arm must declare this** (enforced by
    #: tests/test_rollout_mode.py). It is left None here rather than given a
    #: default because a silent default is exactly the failure this declaration
    #: exists to prevent: `recall()` is scored by
    #: benchmarks/pattern_completion.py, and arms were silently running
    #: different rollout protocols under the same metric.
    rollout_mode: Optional[RolloutMode] = None

    #: True when ``recall`` uses the whole 2D prompt trajectory to establish the
    #: state it rolls forward from; False when it reduces the prompt to
    #: ``prompt_event[-1]``. False is correct and honest for an arm whose
    #: ``predict_next`` is pure and which carries nothing between steps -- there
    #: is no state a longer prompt could establish. Declared per arm so a floor
    #: on a delayed-cue task is attributable.
    prompt_conditioned: bool = False

    #: True when the forward pass (``predict_next`` / ``recall``) is itself
    #: random, so repeating a probe with a FIXED clean cue samples the model's
    #: own variability. Every arm in the roster is deterministic today, so
    #: under ``ProbeProtocol.CLEAN_SINGLE`` a repeat is an exact copy and the
    #: harness runs the probe once. Declare True only where a repeat is
    #: genuinely a new sample; then, and only then, ``n_trials`` means
    #: something for that arm.
    stochastic_forward: bool = False

    def __init__(self, **kwargs):
        """
        Initialize the model.
        """
        pass

    def observe(self, event: np.ndarray, context: Optional[np.ndarray] = None) -> None:
        """Advance carried state by one observed event, without learning.

        Base-class default: a no-op. An arm with no carried state has nothing
        to advance, so a probe can deliver a prefix to every arm through the
        same call and the rollout that follows starts from the same cue on
        both families. Arms that do carry state declare ``StatePrimeable`` and
        override this (tests/test_observe_default.py checks that they do).
        """
        return None

    def observe_sequence(self, events: np.ndarray, context: Optional[np.ndarray] = None) -> None:
        """Advance carried state over a prefix, in order (no-op by default)."""
        for t, ev in enumerate(np.asarray(events, dtype=float)):
            self.observe(ev, None if context is None else context[t])

    @abstractmethod
    def fit_sequence(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, **kwargs):
        """
        Train or fit the model on a provided sequence.

        Args:
            sequence_data (np.ndarray): The sequence data to learn from. Must be a 2D 
                                        tensor of shape (Time, Features).
            context_data (np.ndarray, optional): Explicit context points parallel to the 
                                                 sequence data. Must be of shape (Time, ContextFeatures).
            **kwargs: Specific fitting arguments (e.g., learning rate, epochs).
        """
        pass

    # NOTE: ``fit_event`` / ``on_event_boundary`` deliberately do NOT live here.
    # They are declared by the ``OnlineTrainable`` capability in
    # ``memval.models.capabilities``; a concrete stub in this MRO would satisfy
    # that ABC's @abstractmethod and make ``isinstance(arm, OnlineTrainable)``
    # true for every arm. Probe streaming support with ``supports_online()``.

    @abstractmethod
    def predict_next(self, current_event: np.ndarray, current_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Predict the next event in a sequence given the current event
        and the model's internal context.

        Args:
            current_event (np.ndarray): The current state or observation. Must be a 1D tensor 
                                        of shape (Features,).
            current_context (np.ndarray, optional): Contextual signal tied to the current event.
                                                    Must be a 1D tensor of shape (ContextFeatures,).
            
        Returns:
            np.ndarray: The predicted next state or observation.
        """
        pass

    def decode_prediction(self, raw_prediction: np.ndarray) -> np.ndarray:
        """
        Decode or clean up a raw prediction vector from predict_next.
        By default, it is a pass-through returning the raw_prediction.
        Subclasses can override this to implement discretization, symbolic cleanup,
        or cleanup using a codebook / clean embeddings.

        Args:
            raw_prediction (np.ndarray): The raw prediction vector from predict_next.

        Returns:
            np.ndarray: The decoded or cleaned prediction vector.
        """
        return raw_prediction


    @abstractmethod
    def recall(self, prompt_event: np.ndarray, length: int, prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Perform pattern completion by recalling a full sequence from a partial prompt.

        Args:
            prompt_event (np.ndarray): The initial cue. Can be a 1D tensor of shape (Features,) 
                                       or a 2D sequence of shape (Time, Features) representing a prompt trajectory.
            length (int): The number of steps to recall forward.
            prompt_context (np.ndarray, optional): Context tied to the prompt. Must match the length of the recall 
                                                   or the prompt depending on the model logic.
            **kwargs: Additional parameters (like future_context).

        Returns:
            np.ndarray: The reconstructed sequence of events as a 2D tensor of shape (length, Features).
        """
        pass

    @abstractmethod
    def get_latent_state(self) -> dict:
        """
        Retrieve the internal latent representation or context state for
        the model (e.g., to perform Representational Similarity Analysis).

        Returns:
            dict: The current latent state variables.
        """
        pass

    @abstractmethod
    def reset_context(self):
        """
        Reset the model's transient internal context (e.g. hidden state of an RNN)
        without clearing the long-term encoded weights (synapses).
        """
        pass
