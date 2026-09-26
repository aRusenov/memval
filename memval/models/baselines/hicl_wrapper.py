import sys
import os
import numpy as np
import torch
from typing import Any, List, Optional, Tuple, Union
from ..base import HippocampalModel
from ..capabilities import RolloutMode

class HiCLSequenceModel(HippocampalModel):
    """
    Wraps the DGGatedHippocampalMoE model from HiCL as-is to make it
    compatible with the memval framework's HippocampalModel API.
    """

    #: `recall` decodes the whole prompt into `context_words`, which conditions
    #: every subsequent step.
    prompt_conditioned = True

    #: `recall` decodes each prediction to a class, then feeds back that word's
    #: CLEAN codebook embedding. Same hardcoded quantized protocol as GPT-2.
    rollout_mode = RolloutMode.ENCODER

    def __init__(
        self,
        encoder: Any,
        n_epochs_phase1: int = 10,
        n_epochs_phase2: int = 5,
        memory_size: int = 200,
        target_sparsity: float = 0.05,
        device: str = "cpu",
        num_experts: int = 1,
        gating_strategy: str = "soft_hard",
        use_small_features: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.n_epochs_phase1 = n_epochs_phase1
        self.n_epochs_phase2 = n_epochs_phase2
        self.memory_size = memory_size
        self.target_sparsity = target_sparsity
        self.num_experts = num_experts
        self.gating_strategy = gating_strategy
        self.use_small_features = use_small_features
        
        # Device detection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.vocab_size = len(self.encoder.idx_to_word)
        self.classes_per_task = self.vocab_size
        
        self.model = None
        self._task_id = 0
        self._trained_tasks = 0
        
        # Keep track of context length / words for predict_next
        self.current_t = 0
        self.context_words = []

    def _init_hicl_model(self):
        # Inject the vendor path
        vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "hicl"))
        sys.path.insert(0, vendor_path)
        try:
            from l import DGGatedHippocampalMoE
        finally:
            sys.path.pop(0)
            
        self.model = DGGatedHippocampalMoE(
            num_experts=self.num_experts,
            classes_per_task=self.classes_per_task,
            input_channels=1, # 1D embedding reshaped to 1 channel 2D
            target_sparsity=self.target_sparsity,
            memory_size=self.memory_size,
            use_small_features=self.use_small_features
        )
        # Set task classes
        # Each expert has the full vocabulary classes
        task_classes = [list(range(self.vocab_size)) for _ in range(self.num_experts)]
        self.model.set_task_classes(task_classes)
        self.model.set_gating_strategy(self.gating_strategy)
        self.model.to(self.device)

    def _to_torch(self, embedding: np.ndarray) -> torch.Tensor:
        if embedding.ndim == 1:
            emb = embedding[np.newaxis, :]
        else:
            emb = embedding
            
        B, D = emb.shape
        if D <= 1024:
            padded = np.pad(emb, ((0, 0), (0, 1024 - D)), mode='constant')
        else:
            padded = emb[:, :1024]
            
        x = torch.tensor(padded, dtype=torch.float32, device=self.device).view(B, 1, 32, 32)
        return x

    def _embeds_to_classes(self, embeddings: np.ndarray) -> np.ndarray:
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        vocab_norms = np.linalg.norm(self.encoder.embeddings, axis=1, keepdims=True)
        
        emb_norms[emb_norms == 0] = 1.0
        vocab_norms[vocab_norms == 0] = 1.0
        
        norm_embeddings = embeddings / emb_norms
        norm_vocab = self.encoder.embeddings / vocab_norms
        
        sims = norm_embeddings @ norm_vocab.T
        return np.argmax(sims, axis=1)

    def _embed_to_class(self, embedding: np.ndarray) -> int:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            norm = 1.0
        norm_emb = embedding / norm
        vocab_norms = np.linalg.norm(self.encoder.embeddings, axis=1)
        vocab_norms[vocab_norms == 0] = 1.0
        norm_vocab = self.encoder.embeddings / vocab_norms[:, np.newaxis]
        sims = norm_vocab @ norm_emb
        return int(np.argmax(sims))

    def _sequence_to_tensors(self, sequence_data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        T = sequence_data.shape[0]
        if T < 2:
            raise ValueError("Sequence data must contain at least 2 steps for next-item prediction.")
        
        inputs = sequence_data[:-1]
        targets = sequence_data[1:]
        
        X = self._to_torch(inputs)
        y_indices = self._embeds_to_classes(targets)
        Y = torch.tensor(y_indices, dtype=torch.long, device=self.device)
        return X, Y

    def fit_sequence(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, **kwargs):
        if self.model is None:
            self._init_hicl_model()
            
        X, Y = self._sequence_to_tensors(sequence_data)
        
        # Decide which expert to train.
        expert_id = self._task_id % self.num_experts
        
        self.model.train()
        
        # Freeze all experts except the current one
        for name, p in self.model.named_parameters():
            is_current_expert = f"hippocampal_experts.{expert_id}" in name or f"output_layers.{expert_id}" in name
            is_shared_component = "feature_extractor" in name or "ca1_integration" in name
            p.requires_grad = is_current_expert or is_shared_component
            
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        learning_rate = kwargs.get("learning_rate", 0.001)
        weight_decay = kwargs.get("weight_decay", 1e-4)
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        
        epochs = kwargs.get("n_epochs_phase1", self.n_epochs_phase1)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Update class prototypes using EMA
            with torch.no_grad():
                features = self.model.feature_extractor(X).view(X.size(0), -1)
                dg_output, _ = self.model.hippocampal_experts[expert_id](features)
                self.model.update_class_prototype_ema(dg_output.detach(), Y, expert_id)
                
            # Forward pass (Oracle routing during training)
            outputs, _, _ = self.model(X, task_id=expert_id)
            
            # Extract outputs for current expert
            start_idx = expert_id * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            task_outputs = outputs[:, start_idx:end_idx]
            
            loss = criterion(task_outputs, Y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Add to replay buffer
            with torch.no_grad():
                self.model.add_to_replay_buffer(X.detach(), Y.detach(), expert_id)
                
        # Freeze prototypes for this expert
        self.model.freeze_expert_prototypes(expert_id)
        
        # Update trained experts count
        self.model.trained_experts = max(self.model.trained_experts, expert_id + 1)
        
        # Update DG prototypes from EMA
        self.model.update_dg_prototypes_from_ema()
        
        self._trained_tasks += 1
        self._task_id += 1
        
        # Phase 2 Contrastive Tuning (only if num_experts > 1)
        if self.num_experts > 1:
            self._run_phase2()

    def _run_phase2(self):
        # Freeze all layers except for the DG layers
        for name, p in self.model.named_parameters():
            if 'hippocampal_experts' in name and 'dg' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        
        vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "hicl"))
        sys.path.insert(0, vendor_path)
        try:
            from l import calculate_global_contrastive_loss, calculate_expert_balancing_loss, create_replay_balanced_loader
        finally:
            sys.path.pop(0)
            
        self.model.train()
        for epoch in range(self.n_epochs_phase2):
            replay_loader = create_replay_balanced_loader(self.model, batch_size=32)
            if not replay_loader:
                break
                
            for inputs, labels, task_ids in replay_loader:
                inputs = inputs.to(self.device)
                task_ids = task_ids.to(self.device)
                
                optimizer.zero_grad()
                
                features = self.model.feature_extractor(inputs).view(inputs.size(0), -1)
                all_dg_outputs = [expert(features)[0] for expert in self.model.hippocampal_experts]
                
                loss = calculate_global_contrastive_loss(
                    all_dg_outputs, task_ids, self.model.dg_prototypes, margin=0.5
                )
                
                # Global decorrelation loss on prototypes
                prototypes_norm = torch.nn.functional.normalize(self.model.dg_prototypes, p=2, dim=1)
                sim_matrix = prototypes_norm @ prototypes_norm.T
                num_experts = sim_matrix.size(0)
                if num_experts > 1:
                    off_diag = sim_matrix - torch.eye(num_experts, device=sim_matrix.device)
                    decorrelation_loss = (off_diag ** 2).sum() / (num_experts * (num_experts - 1))
                    loss = loss + 0.3 * decorrelation_loss
                    
                    # Expert balancing loss
                    all_dg_norm = torch.nn.functional.normalize(torch.stack(all_dg_outputs, dim=1), p=2, dim=2)
                    gate_logits = torch.einsum('bne,ne->bn', all_dg_norm, prototypes_norm)
                    balancing_loss = calculate_expert_balancing_loss(gate_logits)
                    loss = loss + 0.2 * balancing_loss
                
                loss.backward()
                optimizer.step()

    def predict_next(self, current_event: np.ndarray, current_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if self.model is None:
            self._init_hicl_model()
            
        word_idx = self._embed_to_class(current_event)
        w_t = self.encoder.idx_to_word[word_idx]
        
        if self.current_t == 0:
            self.context_words = [w_t]
        else:
            self.context_words = self.context_words[:self.current_t] + [w_t]
            
        x = self._to_torch(current_event)
        
        self.model.eval()
        with torch.no_grad():
            outputs, gate_logits, _ = self.model(x)
            
        # outputs shape is (1, num_experts * vocab_size)
        # Sum outputs over experts to get (1, vocab_size)
        logits = outputs.view(1, self.num_experts, self.vocab_size).sum(dim=1)[0]
        
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        
        # Soft lookup: weighted sum of embeddings
        pred_emb = (probs[:, None] * self.encoder.embeddings).sum(axis=0)
        
        self.current_t += 1
        return pred_emb.copy()

    def decode_prediction(self, raw_prediction: np.ndarray) -> np.ndarray:
        word_idx = self._embed_to_class(raw_prediction)
        return self.encoder.embeddings[word_idx].copy()

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if prompt_event.ndim > 1:
            prompt_indices = self._embeds_to_classes(prompt_event)
            prompt_words = [self.encoder.idx_to_word[idx] for idx in prompt_indices]
            self.context_words = list(prompt_words)
            self.current_t = len(prompt_words) - 1
        else:
            prompt_idx = self._embed_to_class(prompt_event)
            prompt_word = self.encoder.idx_to_word[prompt_idx]
            self.context_words = [prompt_word]
            self.current_t = 0
            
        recalled_embs = []
        for t in range(length):
            c_t = None
            if prompt_context is not None:
                if prompt_context.ndim > 1:
                    c_t = prompt_context[t] if t < len(prompt_context) else prompt_context[-1]
                else:
                    c_t = prompt_context
                    
            current_emb = self.encoder.embeddings[self.encoder.word_to_idx[self.context_words[-1]]]
            next_emb = self.predict_next(current_emb, current_context=c_t)
            recalled_embs.append(next_emb)
            
            next_idx = self._embed_to_class(next_emb)
            next_word = self.encoder.idx_to_word[next_idx]
            self.context_words.append(next_word)
            
        return np.array(recalled_embs)

    def reset_context(self):
        self.current_t = 0
        self.context_words = []

    def get_latent_state(self) -> dict:
        if self.model is None:
            return {}
        return {
            "model_state_dict": self.model.state_dict(),
            "dg_prototypes": self.model.dg_prototypes.cpu().numpy()
        }
