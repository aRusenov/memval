import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from memval.generators.t_maze import TMazeGenerator
from memval.encoders.spatial import PlaceCellEncoder
from memval.models.baselines.dg_eqprop import DGEqPropSequenceNetwork

def run_debug():
    # 1. Generate T-Maze
    tmaze_gen = TMazeGenerator(seed=42)
    seq_len = 15
    
    tmaze_trajectories = tmaze_gen.generate(n_sequences=1, sequence_length=seq_len, turn_direction='right')
    tmaze_seq = tmaze_trajectories[0]

    # 2. Encode to place cells
    encoder = PlaceCellEncoder(seed=42, sigma_scale=1.0)
    tmaze_seq_encoded = encoder.encode(tmaze_seq)

    # 3. Init DG EqProp model
    model = DGEqPropSequenceNetwork(n_features=encoder.n_cells, n_epochs=100)
    
    # 4. Extract DG representations for each step in the sequence
    dg_activations = []
    for t in range(seq_len):
        x = tmaze_seq_encoded[t]
        dg_in = model._make_dg_input(x, context=None)
        dg_code = model.dg.transform(dg_in)
        dg_activations.append(dg_code)
        
    dg_activations = np.array(dg_activations) # Shape: (T, n_dg)
    
    # 5. Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(dg_activations.T, cmap='viridis', cbar=True, yticklabels=False)
    plt.xlabel('Timestep')
    plt.ylabel('DG Neurons')
    plt.title('Dentate Gyrus Layer Activations over T-Maze Sequence')
    plt.tight_layout()
    
    out_path = '/Users/atanas/.gemini/antigravity/brain/7664a859-2d61-4af6-a4c2-fd2d4fc70bba/scratch/dg_heatmap.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {out_path}")
    
    # Check sparsity / dead neurons
    active_neurons = np.sum(dg_activations > 0, axis=0)
    dead_neurons = np.sum(active_neurons == 0)
    avg_sparsity = np.mean(np.sum(dg_activations > 0, axis=1)) / model.n_dg
    print(f"Total DG neurons: {model.n_dg}")
    print(f"Dead neurons (never fired): {dead_neurons}")
    print(f"Average sparsity per timestep: {avg_sparsity:.4f}")
    
if __name__ == "__main__":
    run_debug()
