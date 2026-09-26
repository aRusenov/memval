import numpy as np
from typing import List, Optional, Tuple, Union, Dict
from ..base import HippocampalModel
from ..capabilities import RolloutMode
from .dg_eqprop import DGEqPropSequenceNetwork

class CLSDGEqPropNetwork(HippocampalModel):
    """
    Complementary Learning Systems (CLS) sequence network wrapping two
    DGEqPropSequenceNetwork instances:
      - HPC (Hippocampal store): fast learning (higher learning rate, fewer epochs)
      - CTX (Neocortical store): slow learning (lower learning rate, trained on replay)
      
    Features:
      - Shared Dentate Gyrus (DG) projection module between HPC and CTX
      - FIFO Replay Buffer for episodic consolidation
      - Blended retrieval using alpha interpolation
    """
    #: `recall` feeds the raw prediction back as the next cue.
    rollout_mode = RolloutMode.OBSERVATION

    def __init__(
        self,
        n_features: int,
        n_context: int = 0,
        n_dg: int = 1000,
        sparsity: float = 0.05,
        noise_scale: float = 0.05,
        n_hidden: Union[int, Tuple[int, int]] = 128,
        hidden_sparsity: Optional[float] = None,
        learning_rate: float = 0.05,
        beta: float = 0.5,
        n_settle_steps: int = 50,
        dt: float = 0.5,
        activation: str = "tanh",
        seed: Optional[int] = None,
        
        # CLS specific parameters
        hpc_epochs: int = 5,
        ctx_epochs: int = 100,
        ctx_lr_scale: float = 0.1,
        blend_alpha: float = 0.5,
        max_buffer_size: int = 10,
        auto_consolidate: bool = True,
        use_novelty_gate: bool = True,
        novelty_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_context = n_context
        self.hpc_epochs = hpc_epochs
        self.ctx_epochs = ctx_epochs
        self.ctx_lr_scale = ctx_lr_scale
        self.blend_alpha = blend_alpha
        self.max_buffer_size = max_buffer_size
        self.auto_consolidate = auto_consolidate
        self.use_novelty_gate = use_novelty_gate
        self.novelty_scale = novelty_scale
        self.last_pred_hpc = None
        self._replay_buffer: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []

        if isinstance(n_hidden, tuple):
            hpc_hidden, ctx_hidden = n_hidden
        else:
            hpc_hidden = ctx_hidden = n_hidden

        self.hpc = DGEqPropSequenceNetwork(
            n_features=n_features,
            n_context=n_context,
            n_dg=n_dg,
            sparsity=sparsity,
            noise_scale=noise_scale,
            n_hidden=hpc_hidden,
            hidden_sparsity=hidden_sparsity,
            learning_rate=learning_rate,
            beta=beta,
            n_settle_steps=n_settle_steps,
            n_epochs=hpc_epochs,
            dt=dt,
            activation=activation,
            seed=seed,
            **kwargs
        )

        self.ctx = DGEqPropSequenceNetwork(
            n_features=n_features,
            n_context=n_context,
            n_dg=n_dg,
            sparsity=sparsity,
            noise_scale=noise_scale,
            n_hidden=ctx_hidden,
            hidden_sparsity=hidden_sparsity,
            learning_rate=learning_rate * ctx_lr_scale,
            beta=beta,
            n_settle_steps=n_settle_steps,
            n_epochs=ctx_epochs,
            dt=dt,
            activation=activation,
            seed=seed + 1 if seed is not None else None,
            **kwargs
        )

        # Share Dentate Gyrus projection weights as per Open Question 3
        self.ctx.dg = self.hpc.dg

    def fit_sequence(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, **kwargs):
        """
        Train the model on a single sequence.
        
        Appends the sequence to the replay buffer, trains the hippocampal network online,
        and triggers auto-consolidation if enabled.
        """
        # 1. Append to buffer, respect capacity
        self._replay_buffer.append((sequence_data.copy(), context_data.copy() if context_data is not None else None))
        if len(self._replay_buffer) > self.max_buffer_size:
            self._replay_buffer.pop(0)

        # 2. Train HPC online
        self.hpc.fit_sequence(sequence_data, context_data, **kwargs)

        # 3. Auto-consolidate
        if self.auto_consolidate:
            self._auto_consolidate()

    def consolidate(self, n_cycles: int = 1):
        """
        Offline consolidation pass. Replays the contents of the episodic buffer
        and fits them to the neocortical network interleaved.
        """
        if not self._replay_buffer:
            return
        
        # Fit sequences in neocortex using slow learning
        for _ in range(n_cycles):
            for seq_data, ctx_data in self._replay_buffer:
                self.ctx.fit_sequence(seq_data, ctx_data)

    def _auto_consolidate(self):
        """
        Triggers automatic consolidation. Separated to allow future triggers
        and scheduling customization.
        """
        self.consolidate(n_cycles=1)

    def predict_next(self, current_event: np.ndarray, current_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Predict the next event by blending outputs from HPC and CTX networks.
        """
        # Extract blend_alpha override if present
        alpha = kwargs.get("blend_alpha", None)
        
        pred_hpc = self.hpc.predict_next(current_event, current_context, **kwargs)
        pred_ctx = self.ctx.predict_next(current_event, current_context, **kwargs)
        
        if alpha is None:
            if self.use_novelty_gate and self.last_pred_hpc is not None:
                error = np.linalg.norm(current_event - self.last_pred_hpc)
                alpha = 1.0 - np.exp(-error / self.novelty_scale)
            else:
                alpha = self.blend_alpha
                
        self.last_pred_hpc = pred_hpc.copy()
        
        return alpha * pred_hpc + (1.0 - alpha) * pred_ctx

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Autoregressively recall a sequence of events.
        """
        if len(prompt_event.shape) > 1:
            current = prompt_event[-1].copy()
        else:
            current = prompt_event.copy()

        recalled = np.zeros((length, self.n_features))

        for t in range(length):
            ctx = prompt_context[t] if (prompt_context is not None and t < len(prompt_context)) else None
            next_val = self.predict_next(current, current_context=ctx, **kwargs)
            recalled[t] = next_val
            current = next_val

        return recalled

    def get_latent_state(self) -> dict:
        """
        Return the combined latent representation variables for both HPC and CTX.
        """
        return {
            "hpc": self.hpc.get_latent_state(),
            "ctx": self.ctx.get_latent_state(),
        }

    def reset_context(self):
        """
        Reset transient activity dynamics for both networks.
        """
        self.hpc.reset_context()
        self.ctx.reset_context()
        self.last_pred_hpc = None
