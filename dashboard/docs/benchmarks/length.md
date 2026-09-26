# Memory Span & Length Sweep Benchmark

## Biological Context
Human and animal working/episodic memory has a finite capacity. For example, humans can easily remember a sequence of 5-7 items (Miller's Law) but struggle with 20 items. In computational neuroscience, a model's *memory span* represents its storage capacity limits before interference degrades recall.

## Computational Assay
The Length Sweep benchmark evaluates sequence recall accuracy (measured via MRR) as a function of the sequence length $L$. The length is systematically swept (e.g., $L = 3, 5, 7, 9, 11$), and the model is trained and tested on each length.

### Metrics Evaluated
- **Max Memory Span**: The maximum sequence length at which the model achieves perfect or near-perfect recall (e.g., MRR > 0.8).
- **MRR Curve**: Mean Recall Rate plotted across sequence lengths to track the decay rate of capacity.
- **Span Curve**: Reconstructed sequence length versus true sequence length.
