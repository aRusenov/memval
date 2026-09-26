import sys
import os
sys.path.insert(0, '/Users/atanas/Documents/workspace/memval')
import numpy as np

from memval.encoders import SymbolicEncoder, SymbolicDecoder
from memval.models.baselines import DGEqPropSequenceNetwork

vocab_multi = {
    'apple': 'fruit', 'banana': 'fruit', 'orange': 'fruit', 'grape': 'fruit', 'pear': 'fruit',
    'dog': 'animal', 'cat': 'animal', 'horse': 'animal', 'cow': 'animal', 'sheep': 'animal',
    'ford': 'car', 'chevy': 'car', 'dodge': 'car', 'toyota': 'car', 'honda': 'car'
}
encoder_multi = SymbolicEncoder(vocab_multi, embedding_dim=100, category_variance=0.2, seed=42)
decoder_multi = SymbolicDecoder(encoder_multi)

single_list = ['apple', 'banana', 'orange', 'grape', 'pear']
enc_single = encoder_multi.encode(single_list)
rng = np.random.default_rng(42)
context_map = {
    'fruits':  rng.integers(0, 2, size=32).astype(float),
}
context_single = np.tile(context_map['fruits'], (len(single_list), 1))

print("Testing direct DGEqPropSequenceNetwork:")
model = DGEqPropSequenceNetwork(
    n_features=100,
    n_context=32,
    n_dg=1000,
    n_hidden=128,
    learning_rate=0.05,
    n_epochs=100,
    seed=42
)

model.fit_sequence(enc_single, context_single)

model.reset_context()
correct = 0
for i in range(len(single_list) - 1):
    current_evt = enc_single[i]
    pred = model.predict_next(current_evt, current_context=context_map['fruits'])
    pred_word = decoder_multi.decode(pred, top_k=1)[0]
    is_correct = (pred_word == single_list[i+1])
    if is_correct:
        correct += 1
    print(f"  Input: {single_list[i]} -> Pred: {pred_word} (expected {single_list[i+1]}) {'[OK]' if is_correct else '[FAIL]'}")
print(f"  Accuracy: {correct}/4")
