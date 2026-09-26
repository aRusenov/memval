# Semantic Similarity & Interference Benchmark

## Biological Context
Similar memories are prone to interference. In rodent experiments, placing animals in two very similar rooms leads to the firing of similar place cell ensembles. If the rooms are too similar, the representations might merge (generalization), whereas if the animal can distinguish them, it must *orthogonalize* the representations (pattern separation). The dentate gyrus subfield of the hippocampus is known to perform pattern separation to avoid interference.

## Computational Assay
This benchmark sweeps the category variance or overlap similarity between two sequences. The model is trained on two highly similar sequences. We evaluate how the degree of cosine similarity between the sequence elements affects the model's ability to recall them independently.

### Metrics Evaluated
- **MRR High Similarity**: Recall accuracy when sequences have high overlap/cosine similarity.
- **MRR Low Similarity**: Recall accuracy when sequences have low overlap.
- **Similarity Effect MRR Drop**: The performance drop between low and high similarity settings (`MRR Low - MRR High`). A high drop indicates that the model suffers from severe interference when items are semantically similar.
- **Similarity Sweep Curve**: Recall accuracy plotted as a function of cosine similarity between the stored sequences.
