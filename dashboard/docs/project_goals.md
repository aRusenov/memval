# MemVal: Hippocampal Sequence Learning Benchmarking Framework

## Project Overview
The mammalian hippocampus serves as the brain's premier engine for rapid, episodic, and sequential memory. Understanding how it stores and recalls sequence representations is a central problem in both cognitive neuroscience and artificial intelligence. 

While modern artificial neural networks (e.g., LSTMs, Transformers) excel at processing sequences in static, offline settings, they fail catastrophically when forced to learn new sequences incrementally (the stability-plasticity dilemma). Biologically-plausible computational architectures, such as attractor networks and equilibrium propagation models, attempt to address these issues but have lacked a unified, standardized framework for rigorous evaluation.

**MemVal** addresses this gap. It provides a standardized testing and benchmarking platform to evaluate computational models on their sequence learning, pattern completion, and disambiguation capabilities, bridging the gap between biological findings and machine learning algorithms.

---

## Core Framework Objectives

1. **Systematic Sequence Generation**
   Providing a synthetic sequence generation engine (`memval.generators`) that produces diverse, structurally structured event sequences with fine-grained control over underlying physical and temporal properties.

2. **Standardized Model Adapter API**
   Defining clean, unified APIs (`fit_sequence()`, `recall()`, `get_latent_state()`, `reset_context()`) to allow researchers to integrate and compare very different model classes (e.g., asymmetric Hopfield networks, Sparse Dentate Gyrus models with Equilibrium Propagation, RNNs).

3. **Rigorous Biological Assays**
   Replicating classical behavioral paradigms from animal studies to test specific cognitive requirements, including spatial path integration, pattern completion of partial trajectories, sequence disambiguation under high context overlap, and continual sequence storage.

4. **Quantitative Diagnostics & Metrics**
   Providing evaluation tools to analyze model accuracy, memory decay rates, noise tolerance, and representational dynamics (such as Representational Similarity Analysis and neural manifold visualizations).
