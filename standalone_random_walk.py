import os
import numpy as np
import matplotlib.pyplot as plt

from memval.generators.spatial import SpatialSequenceGenerator
from memval.encoders.spatial import PlaceCellEncoder
from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork

def run_standalone_random_walk():
    print("=" * 50)
    print(" Running Standalone Random Walk Pattern Completion")
    print("=" * 50)

    # 1. Generate the Raw 2D Trajectory
    spatial_gen = SpatialSequenceGenerator(n_dimensions=2, seed=42)
    seq_len_rw = 25
    prompt_len_rw = int(seq_len_rw * 0.1) # 2 steps
    recall_len_rw = seq_len_rw - prompt_len_rw # 23 steps

    print(f"Generating Random Walk sequence (Total: {seq_len_rw} steps, Prompt: {prompt_len_rw} steps, Recall: {recall_len_rw} steps)")
    rw_trajectories = spatial_gen.generate(
        n_sequences=1, 
        sequence_length=seq_len_rw, 
        step_size=0.3, 
        momentum=0.9, 
        drift=0.3, 
        start_pos="center"
    )
    rw_seq = rw_trajectories[0]
    prompt_rw = rw_seq[:prompt_len_rw]
    true_suffix_rw = rw_seq[prompt_len_rw:]

    # 2. Encode into Place Cells
    encoder_rw = PlaceCellEncoder(n_cells_per_dim=20, env_bounds=1.0, sigma_scale=1.0, seed=42)
    rw_seq_encoded = encoder_rw.encode(rw_seq)
    prompt_rw_encoded = rw_seq_encoded[:prompt_len_rw]
    
    print(f"Encoded into {encoder_rw.n_cells} place cells.")

    # 3. Initialize Model
    print("Using model: OriginalEqPropSequenceNetwork")
    model_rw_pc = OriginalEqPropSequenceNetwork(
        n_features=encoder_rw.n_cells, 
        n_hidden=64, 
        n_epochs=20,
        learning_rate=0.1,
        beta=0.5,
        seed=42
    )

    # 4. Train Model
    print("Fitting sequence...")
    model_rw_pc.fit_sequence(rw_seq_encoded)
    model_rw_pc.reset_context()

    # 5. Autoregressive Recall & Decoding
    print("Running autoregressive recall...")
    recon_rw_encoded = []
    current_rw = prompt_rw_encoded[-1].copy()
    
    for _ in range(recall_len_rw):
        next_val = model_rw_pc.predict_next(current_rw)
        recon_rw_encoded.append(next_val)
        current_rw = next_val
        
    recon_rw_encoded = np.array(recon_rw_encoded)
    reconstructed_rw_pc = encoder_rw.decode(recon_rw_encoded)
    
    rw_pc_mse = float(np.mean((true_suffix_rw - reconstructed_rw_pc) ** 2))
    print(f"\nFinal Pattern Completion MSE: {rw_pc_mse:.5f}")

    # 6. Plot Results
    plt.figure(figsize=(6, 6))
    plt.plot(rw_seq[:, 0], rw_seq[:, 1], '--', color='gray', alpha=0.5, label='True Path')
    plt.plot(prompt_rw[:, 0], prompt_rw[:, 1], '-o', color='green', label='Prompt Cue')
    
    if not np.isnan(rw_pc_mse):
        plt.plot(reconstructed_rw_pc[:, 0], reconstructed_rw_pc[:, 1], '-b^', alpha=0.8, label=f'PC Model (MSE: {rw_pc_mse:.4f})')
        
    plt.title('Standalone Spatial Sequence Pattern Completion')
    plt.legend()
    plt.tight_layout()
    
    output_img = "standalone_random_walk_result.png"
    plt.savefig(output_img, dpi=150)
    print(f"Plot saved to ./{output_img}")
    plt.close()

if __name__ == "__main__":
    run_standalone_random_walk()
