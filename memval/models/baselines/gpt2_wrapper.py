import numpy as np
from typing import Any, List, Optional, Tuple, Union
import torch
from ..base import HippocampalModel
from ..capabilities import RolloutMode

class GPT2SequenceModel(HippocampalModel):
    """
    A wrapper for HuggingFace GPT-2 to run language model baselines
    against episodic memory sequence benchmarks.
    """

    #: `recall` decodes the whole prompt into `context_words`, which conditions
    #: every subsequent generation step.
    prompt_conditioned = True

    #: `recall` decodes each prediction to a word and feeds back that word's
    #: CLEAN codebook embedding, not the prediction. This is
    #: `feedback_mode="quantized"` hardcoded into the arm -- a drift-free
    #: protocol that OBSERVATION arms do not get. See docs/rollout_protocol.md.
    rollout_mode = RolloutMode.ENCODER

    def __init__(
        self,
        encoder: Any,
        n_epochs: int = 100,
        learning_rate: float = 2e-4,
        from_pretrained: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the GPT-2 Sequence Model.

        Args:
            encoder (SymbolicEncoder): The encoder used to map words to embeddings.
            n_epochs (int): Default training epochs for fit_sequence.
            learning_rate (float): Learning rate for fine-tuning.
            from_pretrained (bool): Whether to load pre-trained GPT-2 weights or initialize randomly.
            device (str): Device to run the model on ('cpu', 'cuda', 'mps', or 'auto').
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        
        # Device auto-detection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if from_pretrained:
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        else:
            from transformers import GPT2Config
            config = GPT2Config.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel(config)
            
        self.model.to(self.device)
        self.input_proj = torch.nn.Linear(encoder.embedding_dim, self.model.config.n_embd, bias=False)
        self.input_proj.to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.input_proj.parameters()),
            lr=self.learning_rate
        )
        
        # Constrain generation to vocabulary tokens to prevent hallucination of out-of-vocab words
        self.vocab_token_ids = set()
        for word in self.encoder.idx_to_word:
            # Add token ids for word and space-prefixed word
            self.vocab_token_ids.update(self.tokenizer.encode(word))
            self.vocab_token_ids.update(self.tokenizer.encode(" " + word))
        self.vocab_token_ids = list(self.vocab_token_ids)

        # Build stable token_to_word and word_to_token maps for deterministic mapping
        self.token_to_word = {}
        self.word_to_token = {}
        for word in self.encoder.idx_to_word:
            # For labels construction: find preferred token ID representing the word
            ids = self.tokenizer.encode(" " + word)
            if len(ids) > 0:
                self.word_to_token[word] = ids[0]
            else:
                ids = self.tokenizer.encode(word)
                self.word_to_token[word] = ids[0]
            
            # Map all corresponding token IDs back to this word
            for token_id in self.tokenizer.encode(word) + self.tokenizer.encode(" " + word):
                self.token_to_word[token_id] = word
        
        # Internal state tracking
        self.current_t = 0
        self.context_words = []
        self.context_to_id = {}

    def _get_context_prefix(self, context_vector: Optional[np.ndarray]) -> str:
        """Map raw float/binary context vectors to a unique textual prefix token."""
        if context_vector is None:
            return ""
        key = tuple(np.round(context_vector.flatten(), 3))
        # If all elements are zero or very close to zero, ignore context
        if all(abs(x) < 1e-3 for x in key):
            return ""
        if key not in self.context_to_id:
            self.context_to_id[key] = f"ctx{len(self.context_to_id)}"
        return self.context_to_id[key]

    def _decode_embeddings(self, sequence_data: np.ndarray) -> List[str]:
        """Decode sequence embeddings back into vocabulary words using cosine similarity."""
        if sequence_data.ndim == 1:
            sequence_data = sequence_data[np.newaxis, :]
            
        norms = np.linalg.norm(sequence_data, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_seq = sequence_data / norms
        
        # Since self.encoder.embeddings is normalized, dot product is cosine similarity
        similarities = np.dot(normalized_seq, self.encoder.embeddings.T)
        indices = np.argmax(similarities, axis=-1)
        return [self.encoder.idx_to_word[idx] for idx in indices]

    def fit_sequences(self, sequences: List[Tuple[np.ndarray, Optional[np.ndarray]]], **kwargs):
        """
        Train on one or more sequences simultaneously.

        Args:
            sequences: List of (sequence_data, context_data) pairs.
        """
        epochs = kwargs.get("epochs", self.n_epochs)
        # Check if epochs was passed as an attribute during model init
        if epochs is None:
            epochs = self.n_epochs
        if epochs <= 0 or not sequences:
            return

        dataset = []
        for seq_data, ctx_data in sequences:
            # Check for degenerate sequences (fewer than 2 steps)
            if seq_data.shape[0] < 2:
                continue
            words = self._decode_embeddings(seq_data)
            
            # Construct prefix token IDs
            prefix_tokens = []
            if ctx_data is not None:
                prefix = self._get_context_prefix(ctx_data[0])
                if prefix:
                    prefix_tokens = self.tokenizer.encode(prefix + " ")
            
            # Get token IDs representing the words
            word_tokens = [self.word_to_token[w] for w in words]
            
            # Prepare target labels: combination of prefix tokens and word tokens
            label_ids = prefix_tokens + word_tokens
            labels = torch.tensor([label_ids], dtype=torch.long).to(self.device)
            
            # Save components for on-the-fly inputs_embeds construction during epochs
            sym_embs = torch.tensor(self.encoder.encode(words), dtype=torch.float32).to(self.device)
            dataset.append((prefix_tokens, sym_embs, labels))

        self.model.train()
        for epoch in range(epochs):
            for prefix_tokens, sym_embs, labels in dataset:
                self.optimizer.zero_grad()
                
                # 1. Embed the prefix tokens using the standard GPT-2 wte layer
                if prefix_tokens:
                    prefix_tensor = torch.tensor([prefix_tokens], dtype=torch.long).to(self.device)
                    prefix_embeds = self.model.transformer.wte(prefix_tensor)  # (1, len(prefix_tokens), n_embd)
                else:
                    prefix_embeds = None
                    
                # 2. Project the SymbolicEncoder word embeddings
                word_embeds = self.input_proj(sym_embs).unsqueeze(0)  # (1, len(words), n_embd)
                
                # 3. Concatenate prefix and word embeddings
                if prefix_embeds is not None:
                    inputs_embeds = torch.cat([prefix_embeds, word_embeds], dim=1)
                else:
                    inputs_embeds = word_embeds
                
                outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

    def fit_sequence(self, sequence_data: np.ndarray, context_data: Optional[np.ndarray] = None, **kwargs):
        """
        Fine-tune GPT-2 on a single sequence.
        """
        self.fit_sequences([(sequence_data, context_data)], **kwargs)

    def predict_next(self, current_event: np.ndarray, current_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Predict the next event in the sequence given the current event.
        """
        w_t = self._decode_embeddings(current_event)[0]
        
        if self.current_t == 0:
            self.context_words = [w_t]
        else:
            self.context_words = self.context_words[:self.current_t] + [w_t]
            
        # Get context prefix if context is provided
        prefix_tokens = []
        if current_context is not None:
            prefix = self._get_context_prefix(current_context)
            if prefix:
                prefix_tokens = self.tokenizer.encode(prefix + " ")
                
        # Build inputs_embeds
        # 1. Embed context prefix tokens
        if prefix_tokens:
            prefix_tensor = torch.tensor([prefix_tokens], dtype=torch.long).to(self.device)
            prefix_embeds = self.model.transformer.wte(prefix_tensor)
        else:
            prefix_embeds = None
            
        # 2. Get SymbolicEncoder embeddings for context words and project them
        word_syms = []
        for w in self.context_words:
            idx = self.encoder.word_to_idx[w]
            word_syms.append(self.encoder.embeddings[idx])
        word_syms_tensor = torch.tensor(np.array(word_syms), dtype=torch.float32).to(self.device)
        word_embeds = self.input_proj(word_syms_tensor).unsqueeze(0)  # (1, len(context_words), n_embd)
        
        # 3. Concatenate prefix and word embeddings
        if prefix_embeds is not None:
            inputs_embeds = torch.cat([prefix_embeds, word_embeds], dim=1)
        else:
            inputs_embeds = word_embeds
            
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs_embeds=inputs_embeds)
            
        # Restrict logits to benchmark vocabulary
        logits = outputs.logits[0, -1, :]
        mask = torch.full_like(logits, float("-inf"))
        mask[self.vocab_token_ids] = 0.0
        masked_logits = logits + mask
        
        next_token_id = torch.argmax(masked_logits).item()
        pred_word = self.token_to_word.get(next_token_id)
        
        # Map predicted word back to embedding space
        if pred_word is None or pred_word not in self.encoder.word_to_idx:
            pred_idx = 0
        else:
            pred_idx = self.encoder.word_to_idx[pred_word]
            
        pred_emb = self.encoder.embeddings[pred_idx]
        
        # Advance internal generic tracking
        self.current_t += 1
        return pred_emb.copy()

    def recall(self, prompt_event: np.ndarray, length: int, prompt_context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Recall a sequence from a partial prompt.
        """
        if prompt_event.ndim > 1:
            prompt_words = self._decode_embeddings(prompt_event)
            self.context_words = list(prompt_words)
            self.current_t = len(prompt_words) - 1
        else:
            prompt_word = self._decode_embeddings(prompt_event[np.newaxis, :])[0]
            self.context_words = [prompt_word]
            self.current_t = 0
            
        recalled_embs = []
        for t in range(length):
            # Select correct context for the current step t if prompt_context is 2D
            c_t = None
            if prompt_context is not None:
                if prompt_context.ndim > 1:
                    c_t = prompt_context[t] if t < len(prompt_context) else prompt_context[-1]
                else:
                    c_t = prompt_context

            current_emb = self.encoder.embeddings[self.encoder.word_to_idx[self.context_words[-1]]]
            next_emb = self.predict_next(current_emb, current_context=c_t)
            recalled_embs.append(next_emb)
            
            # Find the word corresponding to next_emb and append it to context_words
            next_word = self._decode_embeddings(next_emb[np.newaxis, :])[0]
            self.context_words.append(next_word)
            
        return np.array(recalled_embs)

    def decode_prediction(self, raw_prediction: np.ndarray) -> np.ndarray:
        pred_word = self._decode_embeddings(raw_prediction)[0]
        pred_idx = self.encoder.word_to_idx[pred_word]
        return self.encoder.embeddings[pred_idx].copy()

    def get_latent_state(self) -> dict:
        return {"model_state_dict": self.model.state_dict()}

    def reset_context(self):
        self.current_t = 0
        self.context_words = []
