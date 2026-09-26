import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

def plot_imprinted_coordinate(model, encoder, coordinate, context=None, ax=None):
    """
    Visualizes how a specific 2D coordinate is "imprinted" in the model's weights/transitions.
    For predictive models, this shows the spatial activation the network predicts for the 
    next step given the current coordinate.
    
    Args:
        model: The trained HippocampalModel (Hopfield or EqProp).
        encoder: The PlaceCellEncoder used for spatial state.
        coordinate: A tuple or array [x, y] representing the 2D coordinate.
        context: Optional context vector. If None, uses model's internal defaults.
        ax: Optional matplotlib axis to plot on.
    """
    # 1. Encode the target coordinate
    activations = encoder.encode(np.array([coordinate])).flatten()
    
    # 2. Get prediction/imprint from the model
    if hasattr(model, 'predict_next'):
        # For predictive models (EqProp, etc.)
        # If no context provided, try to get a neutral one (start of sequence)
        if context is None and hasattr(model, '_get_context'):
            context = model._get_context(0, getattr(model, 'trained_seq_len', 30))
        
        imprint = model.predict_next(activations, current_context=context)
    elif hasattr(model, 'W'):
        # For baseline Hopfield models
        W_spatial = model.W[:, :encoder.n_cells]
        imprint = W_spatial @ activations
    else:
        raise ValueError("Model does not have a recognizable weight matrix or prediction method.")
    
    # 3. Reshape the 1D imprint back to the 2D spatial grid
    n = encoder.n_cells_per_dim
    imprint_2d = imprint.reshape(n, n)
    
    # 4. Plot as a heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        
    bounds = encoder.env_bounds
    extent = [-bounds, bounds, -bounds, bounds]
    
    im = ax.imshow(imprint_2d, origin='lower', extent=extent, cmap='magma', aspect='auto')
    
    # Plot the original target coordinate
    ax.plot(coordinate[0], coordinate[1], 'cx', markersize=10, markeredgewidth=2, label='Current Pos')
    
    ax.set_title(f'Transition Imprint for ({coordinate[0]:.2f}, {coordinate[1]:.2f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    ax.legend()
    plt.colorbar(im, ax=ax, label='Activation Strength')
    
    return ax
