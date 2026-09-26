# Noise Tolerance & Invariance Benchmark

## Biological Context
Sensory systems are inherently noisy. An animal returning to a nest might experience changing weather, lighting conditions, or partial occlusion of landmarks. The hippocampus is able to retrieve the stored memory of the nest's location despite these variations, demonstrating noise invariance and robust error-correction.

## Computational Assay
This benchmark sweeps the variance of sensory noise added to retrieval prompts. The model is trained on clean sequences, and then prompted with noisy sequences at various noise levels (e.g., noise standard deviation $\sigma \in [0.0, 1.0]$). We measure recall accuracy (MRR) at each noise level.

### Metrics Evaluated
- **Noise Tolerance Threshold**: The maximum noise level at which the model can still recall the sequence with an MRR above a target threshold (e.g., 0.5).
- **Noise Sweep Curve**: The decay curve of MRR as noise level increases. Models with strong attractor dynamics typically show a flat curve up to a critical threshold, followed by a sharp drop-off (phase transition).
