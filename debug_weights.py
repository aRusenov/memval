import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from memval.generators.t_maze import TMazeGenerator
from memval.encoders.spatial import PlaceCellEncoder
from memval.models.baselines.eq_prop import EqPropSequenceNetwork

def run_debug():
    # 1. Generate T-Maze
    tmaze_gen = TMazeGenerator(seed=42)
    seq_len = 15
    
    tmaze_trajectories = tmaze_gen.generate(n_sequences=1, sequence_length=seq_len, turn_direction='right')
    tmaze_seq = tmaze_trajectories[0]

    # 2. Encode to place cells
    encoder = PlaceCellEncoder(seed=42, sigma_scale=1.0)
    tmaze_seq_encoded = encoder.encode(tmaze_seq)

    # 3. Init EqProp model (no DG)
    model = EqPropSequenceNetwork(n_features=encoder.n_cells, n_epochs=100)
    
    # Check pre-training weights
    w_ih_pre = model.W_ih.copy()
    w_ho_pre = model.W_ho.copy()

    # 4. Fit sequence
    model.fit_sequence(tmaze_seq_encoded)
    
    # Check post-training weights
    w_ih_post = model.W_ih.copy()
    w_ho_post = model.W_ho.copy()
    
    # Plot weights
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    sns.heatmap(w_ih_pre, cmap='coolwarm', ax=axes[0, 0], center=0)
    axes[0, 0].set_title('W_ih Pre-training')
    
    sns.heatmap(w_ih_post, cmap='coolwarm', ax=axes[0, 1], center=0)
    axes[0, 1].set_title('W_ih Post-training')
    
    sns.heatmap(w_ih_post - w_ih_pre, cmap='coolwarm', ax=axes[0, 2], center=0)
    axes[0, 2].set_title('W_ih Change (Delta)')
    
    sns.heatmap(w_ho_pre, cmap='coolwarm', ax=axes[1, 0], center=0)
    axes[1, 0].set_title('W_ho Pre-training')
    
    sns.heatmap(w_ho_post, cmap='coolwarm', ax=axes[1, 1], center=0)
    axes[1, 1].set_title('W_ho Post-training')
    
    sns.heatmap(w_ho_post - w_ho_pre, cmap='coolwarm', ax=axes[1, 2], center=0)
    axes[1, 2].set_title('W_ho Change (Delta)')
    
    plt.tight_layout()
    out_path = '/Users/atanas/.gemini/antigravity/brain/7664a859-2d61-4af6-a4c2-fd2d4fc70bba/scratch/eqprop_weights.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Weight plots saved to {out_path}")
    
    print(f"Max delta W_ih: {np.max(np.abs(w_ih_post - w_ih_pre)):.6f}")
    print(f"Max delta W_ho: {np.max(np.abs(w_ho_post - w_ho_pre)):.6f}")

if __name__ == "__main__":
    run_debug()
