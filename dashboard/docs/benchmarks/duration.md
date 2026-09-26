# Presentation Duration & Convergence Benchmark

## Biological Context
In biological brains, learning a new sequence or establishing a memory trace requires varying numbers of exposures (stimulus presentations) depending on the complexity of the sequence and the plasticity of the neural substrate. Rapid, one-shot episodic learning (associated with the hippocampus) contrasts with slow, gradual statistical learning (associated with neocortical consolidation). The rate at which recall memory converges over repeated presentations reflects the model's sample efficiency and sequence-encoding dynamics.

## Computational Assay
This benchmark sweeps the **presentation duration** (measured in training epochs or stimulus presentations) rather than any delay interval. Specifically, it trains the model on a stimulus sequence for an increasing number of epochs (e.g., $1, 2, 4, 8, 16, 32, 64, 128, 256, 500$ presentations) to track how quickly it converges to reliable recall. The model is evaluated at each checkpoint to observe learning trajectories (recall accuracy and memory span) over exposure time.

### Metrics Evaluated
- **Convergence MRR**: The Mean Recall Rate (MRR) of sequence recall achieved once training converges or at the maximum epoch limit.
- **Convergence Epochs**: The number of training epochs (exposures) required for the model's recall accuracy to cross a predefined success threshold (e.g., $\text{MRR} \ge 0.95$). If a model fails to cross this threshold within the epoch budget, it is flagged as not converged.
- **Memory Span at Convergence**: The sequence length capacity (longest sequence recallable above threshold) the model achieves once it has converged.
- **Convergence Curves**: Trajectory of Mean Recall Rate (MRR) plotted as a function of training epochs (typically on a log-scale).

