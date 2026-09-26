import numpy as np
from memval.generators.t_maze import TMazeGenerator
from memval.encoders.spatial import PlaceCellEncoder
from memval.models.baselines.dg_eqprop import DGEqPropSequenceNetwork

def run_debug():
    tmaze_gen = TMazeGenerator(seed=42)
    seq_len = 15
    prompt_len = int(seq_len * 0.3)
    
    tmaze_trajectories = tmaze_gen.generate(n_sequences=1, sequence_length=seq_len, turn_direction='right')
    tmaze_seq = tmaze_trajectories[0]

    encoder = PlaceCellEncoder(seed=42, sigma_scale=1.0)
    tmaze_seq_encoded = encoder.encode(tmaze_seq)

    model = DGEqPropSequenceNetwork(n_features=encoder.n_cells, n_epochs=100)
    model.fit_sequence(tmaze_seq_encoded)
    
    prompt = tmaze_seq_encoded[:prompt_len]
    recon_encoded = []
    current = prompt[-1].copy()
    for _ in range(seq_len - prompt_len):
        nxt = model.predict_next(current)
        nxt = (0.5 * nxt) + 0.5
        recon_encoded.append(nxt)
        current = nxt
    
    reconstructed_pc = encoder.decode(np.array(recon_encoded))
    
    print("True trajectory:")
    for p in tmaze_seq:
        print(f"  {p[0]:.3f}, {p[1]:.3f}")
        
    print("\nReconstructed trajectory (from prompt end):")
    for p in reconstructed_pc:
        print(f"  {p[0]:.3f}, {p[1]:.3f}")

if __name__ == "__main__":
    run_debug()
