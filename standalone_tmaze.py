import os
import numpy as np
import matplotlib.pyplot as plt

# Import the generators and encoders
from memval.generators.t_maze import TMazeGenerator
from memval.encoders.spatial import PlaceCellEncoder

# Import the models to test
from memval.models.baselines.eq_prop import EqPropSequenceNetwork
from memval.models.baselines.online_eq_prop import OnlineEqPropSequenceNetwork
from memval.models.baselines.dg_eqprop import DGEqPropSequenceNetwork
from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork


def run_standalone_tmaze():
    print("=" * 50)
    print(" Running Standalone T-Maze Pattern Completion")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Generate the Raw 2D Trajectory
    # ---------------------------------------------------------
    tmaze_gen = TMazeGenerator(seed=42)
    seq_len = 15
    prompt_len = int(seq_len * 0.3)  # 4 steps
    recall_len = seq_len - prompt_len  # 11 steps

    print(f"Generating T-Maze sequence (Total: {seq_len} steps, Prompt: {prompt_len} steps, Recall: {recall_len} steps)")
    tmaze_trajectories = tmaze_gen.generate(
        n_sequences=1, 
        sequence_length=seq_len, 
        turn_direction='right'
    )
    tmaze_seq = tmaze_trajectories[0]
    prompt = tmaze_seq[:prompt_len]
    true_suffix = tmaze_seq[prompt_len:]

    # ---------------------------------------------------------
    # 2. Encode into Place Cells
    # ---------------------------------------------------------
    # Using n_cells_per_dim=10, sigma_scale=1.0 as set in the pipeline
    encoder = PlaceCellEncoder(seed=42, n_cells_per_dim=20, sigma_scale=1.0)
    tmaze_seq_encoded = encoder.encode(tmaze_seq)
    prompt_encoded = tmaze_seq_encoded[:prompt_len]

    print(f"Encoded into {encoder.n_cells} place cells.")

    # ---------------------------------------------------------
    # 3. Initialize Model
    # ---------------------------------------------------------
    # Toggle which model you want to test here:
    # USE_DG_LAYER = False  # Set to True to test DGEqPropSequenceNetwork instead
    # USE_ORIGINAL_EP = True
    
    # if USE_DG_LAYER:
    #     print("Using model: DGEqPropSequenceNetwork")
    #     model_pc = DGEqPropSequenceNetwork(
    #         n_features=encoder.n_cells, 
    #         n_hidden=1024, 
    #         n_epochs=100,
    #         activation="sigmoid",
    #     )
    # elif USE_ORIGINAL_EP:
    print("Using model: OriginalEqPropSequenceNetwork")
    model_pc = OriginalEqPropSequenceNetwork(
        n_features=encoder.n_cells, 
        n_hidden=64, 
        n_epochs=100,
        learning_rate=0.1,
        beta=0.5,
        seed=42
    )
    # else:
    #     print("Using model: OnlineEqPropSequenceNetwork")
    #     ONLINE = False
    #     if ONLINE:
    #         model_pc = OnlineEqPropSequenceNetwork(
    #             n_features=encoder.n_cells, 
    #             n_hidden=64, 
    #             n_epochs=300,
    #             learning_rate=0.1,
    #             activation="sigmoid",
    #             output_activation="clip",
    #             seed=42
    #         )
    #     else:
    #         model_pc = EqPropSequenceNetwork(
    #             n_features=encoder.n_cells, 
    #             n_hidden=64, 
    #             n_epochs=300,
    #             learning_rate=0.1,
    #             activation="sigmoid",
    #             output_activation="clip",
    #             seed=42
    #         )

    # ---------------------------------------------------------
    # 4. Train Model
    # ---------------------------------------------------------
    print("Fitting sequence...")
    model_pc.fit_sequence(tmaze_seq_encoded)
    model_pc.reset_context()

    # ---------------------------------------------------------
    # 5. Autoregressive Recall & Decoding
    # ---------------------------------------------------------
    print("Running autoregressive recall...")
    recon_encoded = []
    current = prompt_encoded[-1].copy()
    
    for t in range(recall_len):
        next_val = model_pc.predict_next(current)
        recon_encoded.append(next_val)
        current = next_val
        
    recon_encoded = np.array(recon_encoded)
    reconstructed_pc = encoder.decode(recon_encoded)
    
    pc_mse = float(np.mean((true_suffix - reconstructed_pc) ** 2))
    print(f"\nFinal Pattern Completion MSE: {pc_mse:.5f}")

    # ---------------------------------------------------------
    # 6. Plot Results
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(tmaze_seq[:, 0], tmaze_seq[:, 1], '--', color='gray', alpha=0.5, label='True T-Maze Path')
    plt.plot(prompt[:, 0], prompt[:, 1], '-o', color='green', label='Prompt Cue')
    
    if not np.isnan(pc_mse):
        plt.plot(reconstructed_pc[:, 0], reconstructed_pc[:, 1], '-b^', alpha=0.8, label=f'PC Model (MSE: {pc_mse:.4f})')
        
    plt.title('Standalone T-Maze Pattern Completion')
    plt.legend()
    plt.tight_layout()
    
    # Save the trajectory plot
    output_img = "standalone_tmaze_result.png"
    plt.savefig(output_img, dpi=150)
    print(f"Plot saved to ./{output_img}")
    plt.close()

    # ---------------------------------------------------------
    # 7. Plot Diagnostic Output Layer Activations
    # ---------------------------------------------------------
    # dim = encoder.n_cells_per_dim
    # steps_to_plot = [0, 2, 7]
    # fig, axes = plt.subplots(len(steps_to_plot), 3, figsize=(12, 3 * len(steps_to_plot)))
    
    # for i, t in enumerate(steps_to_plot):
    #     # Ground Truth
    #     gt_grid = true_suffix_encoded[t].reshape(dim, dim)
    #     im0 = axes[i, 0].imshow(gt_grid, vmin=0, vmax=1, cmap='viridis')
    #     axes[i, 0].set_title(f"Step {t} - Ground Truth")
    #     fig.colorbar(im0, ax=axes[i, 0])
        
    #     # Pre-clip Predictions
    #     pre_grid = pre_clip_encoded[t].reshape(dim, dim)
    #     im1 = axes[i, 1].imshow(pre_grid, cmap='coolwarm')
    #     axes[i, 1].set_title(f"Step {t} - Predicted (Pre-clip)")
    #     fig.colorbar(im1, ax=axes[i, 1])
        
    #     # Post-clip Predictions
    #     post_grid = recon_encoded[t].reshape(dim, dim)
    #     im2 = axes[i, 2].imshow(post_grid, vmin=0, vmax=1, cmap='viridis')
    #     axes[i, 2].set_title(f"Step {t} - Predicted (Post-clip)")
    #     fig.colorbar(im2, ax=axes[i, 2])
        
    #     for ax in axes[i]:
    #         ax.set_xticks([])
    #         ax.set_yticks([])
            
    # plt.suptitle("Output Layer Place Cell Activations (10x10 Grid)")
    # plt.tight_layout()
    
    # output_act_img = "standalone_tmaze_activations.png"
    # plt.savefig(output_act_img, dpi=150)
    # print(f"Diagnostic activation plot saved to ./{output_act_img}")
    plt.close()

if __name__ == "__main__":
    run_standalone_tmaze()
