# MemVal: Hippocampal Sequence Learning Benchmarking Framework

## 1. Executive Summary
The goal of this project is to develop a robust, biologically-grounded testing and benchmarking framework to evaluate computational models of the hippocampus. While existing studies highlight the hippocampus as a central processor for episodic memories and sequential events, a unified benchmark for evaluating artificial and computational hippocampal models is lacking. `MemVal` addresses this by systematically generating diverse, structurally rigorous event sequences across modalities and evaluating candidate models on core cognitive capacities (pattern completion, disambiguation, and on-the-fly learning).

## 2. Core Framework Objectives
- **Systematic Sequence Generation:** Delivery of a novel synthetic dataset generation algorithm capable of producing modalities like spatial, temporal, auditory, and relational event sequences.
- **Model Standardisation:** Provide a standardized abstraction (API) for integrating existing biologically plausible computational models.
- **Performance Evaluation:** Rigorously evaluate models to identify how effectively they handle rapid, on-the-fly episodic learning without catastrophic forgetting.

## 3. High-Level Architecture

The framework is divided into four primary modules:

### 3.1. Synthetic Data Generation Engine (`memval.generators`)
Responsible for synthesizing event sequences with granular control over underlying structural properties.
- **Modalities**:
  - **Spatial:** Multi-dimensional trajectories simulating physical navigation.
  - **Temporal:** Sequences of identical stimuli separated by varying inter-event intervals (ITIs) to test time-cell encoding.
  - **Relational / Abstract:** Sequences representing social hierarchies, auditory tone sweeps, or transitive inference trees.
- **Configurable Structural Properties**:
  - Sequence Length ($L$)
  - Inter-Event Intervals ($\Delta t$)
  - Noise Variance ($\sigma^2$), both Gaussian sensory noise and temporal variability.
  - Contextual Overlap (e.g., shared sub-sequences to test disambiguation).

### 3.2. Model Adapter Interface (`memval.models`)
A standardized interface allowing seamless integration of varying hippocampal models (e.g., Continuous Attractor Networks, Equilibrium Propagation, Reservoir Computers).
- Expected APIs: `fit_sequence()`, `recall()`, `get_latent_state()`, `reset_context()`.

### 3.3. Evaluation & Benchmarking Suite (`memval.benchmarks`)
Standardized tasks designed to replicate biological behavioral assays:
- **Pattern Completion:** Given a partial or highly degraded input snippet securely recall the full original sequence.
- **Sequence Disambiguation:** Differentiating overlapping sequences (e.g., $A \rightarrow B \rightarrow X$ vs. $C \rightarrow B \rightarrow Y$).
- **On-the-Fly Learning:** Assessing the model's ability for one-shot or few-shot acquisition of novel information.
- **Continual Learning:** Measuring resistance to catastrophic forgetting as new sequences are introduced over time.

### 3.4. Metrics & Diagnostics (`memval.metrics`)
Tools to quantitatively assess model fidelity and representation quality.
- **Behavioral Metrics:** Error rates in sequence recall, forgetting curves, one-shot accuracy.
- **Representational Diagnostics:** Analyzing the learned manifolds (via PCA/UMAP) or representational similarity analysis (RSA) to see how effectively the model disentangles overlapping memories.

## 4. Proposed Package Structure
```text
memval/
├── data/                    # Generated datasets and caching
├── memval/
│   ├── __init__.py
│   ├── generators/          # Synthetic sequence generation algorithms
│   │   ├── base.py          # Abstract Generator class
│   │   ├── spatial.py       # Spatial trajectory generation
│   │   ├── temporal.py      # Variable delay sequence generation
│   │   └── abstract.py      # Abstract/Relational modalities
│   ├── models/              # Standardized Model Wrappers
│   │   ├── base.py          # API specification for models
│   │   └── baselines/       # Reference baseline models (RNN, etc.)
│   ├── benchmarks/          # Definition of core biological tests
│   │   ├── pattern_completion.py
│   │   └── fast_learning.py
│   └── metrics/             # Scoring and diagnostic tools
│       ├── accuracy.py
│       └── rsa.py           # Representational Similarity Analysis
├── tests/                   # Framework unit tests
├── notebooks/               # Demonstration and visualization notebooks
├── setup.py                 
└── pyproject.toml           # Project dependencies
```

## 5. Technology Stack
- **Core Environment:** Python 3.10+
- **Data & Computations:** `numpy`, `scipy`
- **Machine Learning Integration:** `torch` or `jax` (for deep learning baselines and GPU acceleration if evaluating large models)
- **Visualization:** `matplotlib`, `seaborn`, `Plotly` (for interactive manifold evaluation)
- **Testing:** `pytest`

## 6. Implementation Phasing
- **Phase 1: Generation Engine:** Implement dataset generation algorithms encompassing spatial, temporal, and abstract modalities. Include noise and sequence overlap configuration.
- **Phase 2: Baseline & Adapter:** Define the standard model API and implement simple analytical or neural baselines (e.g. LSTM, Hopfield Networks) for reference.
- **Phase 3: Benchmark Suite:** Build the logic for 'Pattern Completion' and 'Disambiguation' testing. Incorporate metrics calculating reconstruction error.
- **Phase 4: Advanced Testing:** Implement the protocol for 'Continual / On-the-fly' learning to track catastrophic forgetting and sequential acquisition.

## 7. Open Questions / Next Steps
- Are there specific baseline models (e.g., established computational models from literature) you want to include out-of-the-box?
- Should the data generation engine explicitly export datasets into common formats like `.npz` or standard PyTorch `Dataset` loaders?
- Are there specific spatial formats required (e.g., place cell / grid cell population rate activities vs. raw coordinate representation)?
