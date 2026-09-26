import sys
import os
sys.path.insert(0, '/Users/atanas/Documents/workspace/memval')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from memval.encoders import SymbolicEncoder
from memval.models.baselines import DGEqPropSequenceNetwork

vocab_multi = {
    'apple': 'fruit', 'banana': 'fruit', 'orange': 'fruit', 'grape': 'fruit', 'pear': 'fruit',
    'dog': 'animal', 'cat': 'animal', 'horse': 'animal', 'cow': 'animal', 'sheep': 'animal',
    'ford': 'car', 'chevy': 'car', 'dodge': 'car', 'toyota': 'car', 'honda': 'car'
}
encoder_multi = SymbolicEncoder(vocab_multi, embedding_dim=100, category_variance=0.2, seed=42)

lists = {
    'fruits': ['apple', 'banana', 'orange', 'grape', 'pear'],
    'animals': ['dog', 'cat', 'horse', 'cow', 'sheep'],
    'cars': ['ford', 'chevy', 'dodge', 'toyota', 'honda']
}

rng = np.random.default_rng(42)
context_map = {
    'fruits':  rng.integers(0, 2, size=32).astype(float),
    'animals': rng.integers(0, 2, size=32).astype(float),
    'cars':    rng.integers(0, 2, size=32).astype(float),
}

net_multi = DGEqPropSequenceNetwork(
    n_features=100, n_context=32,
    n_dg=1000, sparsity=0.05, learning_rate=0.1, n_epochs=1, seed=42
)

for scale in [1.0, 0.5, 0.2, 0.1, 0.05, 0.0]:
    dg_codes = []
    for cat, words in lists.items():
        enc_words = encoder_multi.encode(words)
        context_vec = np.array(context_map[cat]) * scale
        
        for i, word in enumerate(words):
            x_t = enc_words[i]
            dg_in = net_multi._make_dg_input(x_t, context_vec)
            dg_code = net_multi.dg.transform(dg_in)
            dg_codes.append(dg_code)
            
    dg_codes = np.array(dg_codes)
    cos_sim = cosine_similarity(dg_codes)
    
    # Calculate average within-category similarity (excluding diagonal)
    within_sims = []
    for idx, (cat, words) in enumerate(lists.items()):
        start_idx = idx * 5
        end_idx = start_idx + 5
        block = cos_sim[start_idx:end_idx, start_idx:end_idx]
        mask = ~np.eye(block.shape[0], dtype=bool)
        within_sims.extend(block[mask])
        
    # Calculate average cross-category similarity
    cross_sims = []
    for idx1 in range(3):
        for idx2 in range(idx1 + 1, 3):
            block = cos_sim[idx1*5:(idx1+1)*5, idx2*5:(idx2+1)*5]
            cross_sims.extend(block.flatten())
            
    print(f"Scale: {scale:.2f} | Within-Category Sim: {np.mean(within_sims):.3f} | Cross-Category Sim: {np.mean(cross_sims):.3f}")
