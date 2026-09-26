# Multiple Sequences & Forgetting Benchmark

## Biological Context
A key feature of the biological brain is *continual learning*. When an animal learns a new task or experience, it does not immediately overwrite old ones. The brain maintains a balance between plasticity (encoding new information) and stability (retaining old information). In artificial systems, training on a second task typically leads to *catastrophic forgetting* of the first.

## Computational Assay
This benchmark tests the model's ability to retain previous memories when learning new sequences. The model sequentially fits task/sequence 1, then task/sequence 2. We evaluate recall performance on sequence 1 *before* and *after* fitting sequence 2.

### Metrics Evaluated
- **MRR Before**: Mean Recall Rate on sequence 1 after initial encoding.
- **MRR After**: Mean Recall Rate on sequence 1 after encoding sequence 2.
- **Delta MRR Forgetting**: The difference (`MRR After - MRR Before`). A value close to 0 indicates high stability (no forgetting), while a value close to -1 indicates catastrophic forgetting.
- **Forgetting Curves**: Recall accuracy on older memories as a function of the number of newly learned sequences.
