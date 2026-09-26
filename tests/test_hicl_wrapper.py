import numpy as np
import pytest
from memval.encoders import SymbolicEncoder
from memval.models.baselines import HiCLSequenceModel

def test_hicl_wrapper_basic():
    vocab = ["apple", "banana", "cherry", "date", "fig", "grape", "kiwi", "lemon", "mango", "orange"]
    encoder = SymbolicEncoder(vocab, embedding_dim=100, seed=42)
    
    seq_words = ["apple", "banana", "cherry", "date", "fig"]
    seq_embeddings = encoder.encode(seq_words)
    
    model = HiCLSequenceModel(
        encoder=encoder,
        n_epochs_phase1=5,
        n_epochs_phase2=2,
        num_experts=1,
        device="cpu"
    )
    
    # Fit the sequence
    model.fit_sequence(seq_embeddings)
    
    # Predict next step-by-step
    model.reset_context()
    for i in range(len(seq_words) - 1):
        current_word_emb = seq_embeddings[i]
        pred_emb = model.predict_next(current_word_emb)
        assert pred_emb.shape == (100,)
        
        # Check decode_prediction
        decoded_emb = model.decode_prediction(pred_emb)
        assert decoded_emb.shape == (100,)
        
        # Verify it's one of the embeddings in the encoder
        diffs = np.linalg.norm(encoder.embeddings - decoded_emb, axis=1)
        assert np.any(diffs < 1e-5)
        
    # Test recall
    model.reset_context()
    recalled = model.recall(seq_embeddings[0], length=4)
    assert recalled.shape == (4, 100)
    
    # Verify reset_context doesn't delete model weights
    latent_state_1 = model.get_latent_state()
    model.reset_context()
    latent_state_2 = model.get_latent_state()
    assert latent_state_1.keys() == latent_state_2.keys()
    
    # Verify predict next returns inside vocabulary
    from memval.encoders import SymbolicDecoder
    decoder = SymbolicDecoder(model.encoder)
    pred_word, _ = decoder.decode_with_score(recalled[0])
    assert pred_word in vocab

def test_hicl_wrapper_multi_expert():
    vocab = ["apple", "banana", "cherry", "date", "fig", "grape", "kiwi", "lemon", "mango", "orange"]
    encoder = SymbolicEncoder(vocab, embedding_dim=100, seed=42)
    
    # Sequence A
    seq_words_a = ["apple", "banana", "cherry"]
    seq_a = encoder.encode(seq_words_a)
    
    # Sequence B
    seq_words_b = ["date", "fig", "grape"]
    seq_b = encoder.encode(seq_words_b)
    
    model = HiCLSequenceModel(
        encoder=encoder,
        n_epochs_phase1=3,
        n_epochs_phase2=2,
        num_experts=2,
        device="cpu"
    )
    
    # Fit both sequences sequentially
    model.fit_sequence(seq_a)
    model.fit_sequence(seq_b)
    
    # Verify trained_experts is 2
    assert model.model.trained_experts == 2
    
    # Recall seq_a
    model.reset_context()
    recalled_a = model.recall(seq_a[0], length=2)
    assert recalled_a.shape == (2, 100)
    
    # Recall seq_b
    model.reset_context()
    recalled_b = model.recall(seq_b[0], length=2)
    assert recalled_b.shape == (2, 100)
