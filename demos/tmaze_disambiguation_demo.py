#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.generators.tmaze_disambiguation import TMazeDisambiguationGenerator
from memval.benchmarks.spatial_disambiguation import SpatialDisambiguationBenchmark
from memval.models.baselines import MODEL_REGISTRY

def run_demo(model_name: str, fit_epochs: int):
    print(f"=== Running T-Maze Odour Disambiguation Demo ===")
    print(f"Model: {model_name}")
    print(f"Fit Epochs: {fit_epochs}")
    print("-" * 50)

    # 1. Generate T-Maze sequences with Odours
    generator = TMazeDisambiguationGenerator(seed=42)
    route_pair = generator.generate(
        sequence_length=30,
        stem_fraction=0.50,
        n_place_cells=400,
        n_odour_dims=20,
        seed=42,
        odour_on_arms=True
    )

    input_A = route_pair["input_A"]
    input_B = route_pair["input_B"]
    stem_end = route_pair["shared_end"]
    encoder = route_pair["encoder"]
    n_place_cells = encoder.n_cells
    n_odour_dims = input_A.shape[-1] - n_place_cells
    L = len(input_A)
    recall_len = L - stem_end

    # Get true paths for plotting
    traj_A = encoder.decode(input_A[:, :n_place_cells])
    traj_B = encoder.decode(input_B[:, :n_place_cells])

    # 2. Instantiate and train model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_REGISTRY.keys())}")
        
    model_entry = MODEL_REGISTRY[model_name]
    model_class = model_entry["class"]
    model_kwargs = dict(model_entry["default_kwargs"])
    
    # Configure model dimensions
    model_kwargs["n_features"] = n_place_cells + n_odour_dims
    if "n_epochs" in model_kwargs:
        model_kwargs["n_epochs"] = fit_epochs
    if "epochs" in model_kwargs:
        model_kwargs["epochs"] = fit_epochs

    print("Training model on both paths...")
    # Train A then B
    model = model_class(**model_kwargs)
    model.reset_context()
    model.fit_sequence(input_A, epochs=fit_epochs)
    if hasattr(model, "consolidate"):
        model.consolidate()
        
    model.fit_sequence(input_B, epochs=fit_epochs)
    if hasattr(model, "consolidate"):
        model.consolidate()

    # 3. Evaluate using SpatialDisambiguationBenchmark
    benchmark = SpatialDisambiguationBenchmark()
    
    print("\nRunning standard benchmark evaluation...")
    metrics_full = benchmark.evaluate(
        model=model,
        route_pair=route_pair,
        condition="full",
        n_trials=20,
        fit_epochs=fit_epochs
    )
    
    metrics_mec = benchmark.evaluate(
        model=model,
        route_pair=route_pair,
        condition="mec_only",
        n_trials=20,
        fit_epochs=fit_epochs
    )

    print("\nRESULTS:")
    print("=" * 60)
    print(f"{'Condition':<20} | {'Branch Accuracy':<20} | {'Confusion Rate':<15}")
    print("-" * 60)
    print(f"{'Full (with Odour)':<20} | {metrics_full['branch_accuracy']:<20.4f} | {metrics_full['confusion_rate']:<15.4f}")
    print(f"{'MEC-only (No Odour)':<20} | {metrics_mec['branch_accuracy']:<20.4f} | {metrics_mec['confusion_rate']:<15.4f}")
    print("=" * 60)

    # 4. Generate trajectories for visualization
    # We execute a single recall trial to plot the resulting path
    def get_recall_path(test_seq):
        model.reset_context()
        # Ingest stem cue
        for t in range(stem_end - 1):
            model.predict_next(test_seq[t])
        
        # Recall autoregressively
        current_input = test_seq[stem_end - 1].copy()
        recalled = []
        for t in range(recall_len):
            pred = model.predict_next(current_input)
            recalled.append(pred[:n_place_cells])
            # Outside stem, odour is 0
            current_input = np.concatenate([pred[:n_place_cells], np.zeros(n_odour_dims)])
            
        return encoder.decode(np.array(recalled))

    # Test full inputs
    recall_full_A = get_recall_path(input_A)
    recall_full_B = get_recall_path(input_B)

    # Test mec-only inputs (odour masked to 0)
    mec_only_input_A = input_A.copy()
    mec_only_input_A[:, n_place_cells:] = 0.0
    mec_only_input_B = input_B.copy()
    mec_only_input_B[:, n_place_cells:] = 0.0
    
    recall_mec_A = get_recall_path(mec_only_input_A)
    recall_mec_B = get_recall_path(mec_only_input_B)

    # 5. Plotting
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Panel 1: Full Condition (With Odour)
    axes[0].plot(traj_A[:, 0], traj_A[:, 1], '--', color='blue', alpha=0.3, label='Target Path A (Left)')
    axes[0].plot(traj_B[:, 0], traj_B[:, 1], '--', color='red', alpha=0.3, label='Target Path B (Right)')
    axes[0].plot(traj_A[:stem_end, 0], traj_A[:stem_end, 1], '-', color='green', linewidth=3, label='Stem Cue (with Odour)')
    axes[0].plot(recall_full_A[:, 0], recall_full_A[:, 1], '-b^', label='Recall A (Left Odour)')
    axes[0].plot(recall_full_B[:, 0], recall_full_B[:, 1], '-rx', label='Recall B (Right Odour)')
    axes[0].set_title(f"Full Recall (Odour Active)\nBranch Acc: {metrics_full['branch_accuracy']:.2%}")
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend()

    # Panel 2: MEC-Only Condition (No Odour)
    axes[1].plot(traj_A[:, 0], traj_A[:, 1], '--', color='blue', alpha=0.3, label='Target Path A')
    axes[1].plot(traj_B[:, 0], traj_B[:, 1], '--', color='red', alpha=0.3, label='Target Path B')
    axes[1].plot(traj_A[:stem_end, 0], traj_A[:stem_end, 1], '-', color='gray', linewidth=3, label='Stem Cue (No Odour)')
    axes[1].plot(recall_mec_A[:, 0], recall_mec_A[:, 1], '-b^', label='Recall A')
    axes[1].plot(recall_mec_B[:, 0], recall_mec_B[:, 1], '-rx', label='Recall B')
    axes[1].set_title(f"MEC-Only Recall (Odour Masked)\nBranch Acc: {metrics_mec['branch_accuracy']:.2%}")
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()

    plt.suptitle(f"T-Maze Odour Disambiguation: {model_name.upper()}")
    plt.tight_layout()
    
    os.makedirs("./results", exist_ok=True)
    plot_path = f"./results/tmaze_disambiguation_{model_name}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run T-Maze odour disambiguation demo.")
    parser.add_argument(
        "--model",
        type=str,
        default="hopfield",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model baseline to use (default: hopfield)"
    )
    parser.add_argument(
        "--fit-epochs",
        type=int,
        default=100,
        help="Number of epochs to train (default: 100)"
    )
    args = parser.parse_args()
    run_demo(args.model, args.fit_epochs)
