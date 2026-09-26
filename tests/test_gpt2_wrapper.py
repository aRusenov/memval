import importlib.util

import numpy as np
import pytest
import torch
from memval.encoders import SymbolicEncoder, SymbolicDecoder
from memval.models.baselines.gpt2_wrapper import GPT2SequenceModel
from memval.utils.text_noise import corrupt_word, edit_distance_nearest

def test_text_noise_utility():
    vocab = ["apple", "banana", "cherry", "date"]
    
    # Test edit distance nearest
    assert edit_distance_nearest("aple", vocab) == "apple"
    assert edit_distance_nearest("bannana", vocab) == "banana"
    
    # Test corrupt_word substitutes
    rng = np.random.default_rng(42)
    corrupted_sub = corrupt_word("apple", noise_type="substitute", noise_level=0.5, rng=rng)
    assert len(corrupted_sub) == 5
    assert corrupted_sub != "apple" or 0.5 == 0.0 # Random possibility of no change but seed 42 should corrupt

    # Test corrupt_word deletes
    corrupted_del = corrupt_word("apple", noise_type="delete", noise_level=0.5, rng=rng)
    assert len(corrupted_del) < 5
    assert len(corrupted_del) >= 1  # at least one character preserved

    # Test corrupt_word inserts
    corrupted_ins = corrupt_word("apple", noise_type="insert", noise_level=0.5, rng=rng)
    assert len(corrupted_ins) > 5

    # Test corrupt_word transposes
    corrupted_tr = corrupt_word("apple", noise_type="transpose", noise_level=0.8, rng=rng)
    assert len(corrupted_tr) == 5

    # Test corrupt_word mixed
    corrupted_mx = corrupt_word("apple", noise_type="mixed", noise_level=0.5, rng=rng)
    assert len(corrupted_mx) >= 1

# GPT2SequenceModel is ARCHIVED -- outside the model taxonomy, so it is not run
# as a benchmark arm (see memval.models.baselines.ARCHIVED_MODELS). The wrapper
# is kept importable, and this test with it, but it is skipped by default rather
# than failing the suite on a `transformers` install that is no longer needed.
# test_text_noise_utility above is unaffected: text_noise is a standalone
# utility, deliberately retained (capacity_coverage_audit.md C3).
@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="archived model; transformers not installed",
)
def test_gpt2_wrapper_basic():
    vocab = ["apple", "banana", "cherry"]
    encoder = SymbolicEncoder(vocab, embedding_dim=100, seed=42)
    decoder = SymbolicDecoder(encoder)
    
    model = GPT2SequenceModel(
        encoder=encoder,
        n_epochs=2,
        from_pretrained=False,  # faster configuration for unit tests
        device="cpu"
    )
    
    # Verify input_proj and optimizer setups
    assert hasattr(model, "input_proj")
    assert isinstance(model.input_proj, torch.nn.Linear)
    assert model.input_proj.weight.shape == (768, 100)
    assert len(model.token_to_word) > 0
    assert len(model.word_to_token) == 3
    
    seq_words = ["apple", "banana", "cherry"]
    seq_embeddings = encoder.encode(seq_words)
    
    # Fit the sequence
    model.fit_sequence(seq_embeddings, epochs=2)
    
    # Predict next step-by-step
    model.reset_context()
    for i in range(len(seq_words) - 1):
        current_word_emb = seq_embeddings[i]
        pred_emb = model.predict_next(current_word_emb)
        assert pred_emb.shape == (100,)
        
        # Verify decode_prediction returns a valid reconstruction
        decoded_emb = model.decode_prediction(pred_emb)
        assert decoded_emb.shape == (100,)
        
        # Verify it decodes back to one of the vocab words
        pred_word, _ = decoder.decode_with_score(pred_emb)
        assert pred_word in vocab

    # Test recall
    model.reset_context()
    recalled = model.recall(seq_embeddings[0], length=2)
    assert recalled.shape == (2, 100)
