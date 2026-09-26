# Original Equilibrium Propagation Sequence Network

> **Based on** Scellier & Bengio (2017), *"Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation,"* Frontiers in Computational Neuroscience **11**:24 — the original two-phase (free / nudge) energy-based learning rule. Here it is adapted from static classification to autoregressive sequence prediction.

## Architecture & Biological Rationale
The `OriginalEqPropSequenceNetwork` is a multi-layer, energy-based recurrent network trained with **Equilibrium Propagation (EP)**, serving as the canonical EP baseline against which the pattern-separating (DG) and consolidation (EWC) variants are compared. Rather than back-propagating errors through a computation graph — which requires non-local information and a separate feedback path (biologically implausible) — EP finds the steady-state equilibrium of an energy function and reads out a gradient from the difference between two settled states.

The network is a simple three-layer feedforward-with-feedback stack:

$$
\text{input } (N) \;\rightarrow\; \text{hidden } (H,\ \text{clipped [0,1]}) \;\rightarrow\; \text{output } (N).
$$

To learn *sequences*, static EP is applied to consecutive pairs $(\mathbf{x}_t, \mathbf{x}_{t+1})$ of a trajectory: the current item $\mathbf{x}_t$ is clamped to the input layer and the next item $\mathbf{x}_{t+1}$ is used as the nudge target on the output layer. Weight updates accumulate across all transitions and across training epochs, so the network learns the $t \rightarrow t+1$ transition map as a set of energy minima.

## Model Formulation

Let $N$ be the feature dimensionality (`n_features`) and $H$ the hidden width
(`n_hidden`). The network has two weight matrices $W_{ih} \in \mathbb{R}^{H \times N}$
(input → hidden) and $W_{ho} \in \mathbb{R}^{N \times H}$ (hidden → output), plus
biases $\mathbf{b}_h \in \mathbb{R}^{H}$, $\mathbf{b}_o \in \mathbb{R}^{N}$.

**Energy.** With input $\mathbf{s}_i$ clamped and hidden/output states
$\mathbf{s}_h, \mathbf{s}_o$ free, the network minimizes

$$
E \;=\; -\tfrac{1}{2}\,\mathbf{s}_h^{\top} W_{ih}\, \mathbf{s}_i
\;-\; \tfrac{1}{2}\,\mathbf{s}_o^{\top} W_{ho}\, \mathbf{s}_h
\;-\; \mathbf{b}_h^{\top}\mathbf{s}_h
\;-\; \mathbf{b}_o^{\top}\mathbf{s}_o
\;+\; \tfrac{1}{2}\,\beta\,\lVert \mathbf{s}_o - \mathbf{y} \rVert^2,
$$

where $\beta$ is the **nudge strength** (0 in the free phase, $>0$ in the nudge
phase) and $\mathbf{y}$ is the target.

**Settling dynamics.** Each phase relaxes the states toward equilibrium by
discretized gradient flow ($\dot{\mathbf{s}} \propto -\partial E/\partial \mathbf{s}$),
with states clipped to $[0,1]$ (so the state itself acts as its own activation):

$$
\mathbf{s}_h \leftarrow \mathrm{clip}\big(\mathbf{s}_h + \Delta t\,(-\mathbf{s}_h + W_{ih}\mathbf{s}_i + W_{ho}^{\top}\mathbf{s}_o + \mathbf{b}_h),\ 0, 1\big),
$$

$$
\mathbf{s}_o \leftarrow \mathrm{clip}\big(\mathbf{s}_o + \Delta t\,(-\mathbf{s}_o + W_{ho}\mathbf{s}_h + \mathbf{b}_o + \underbrace{2\beta(\mathbf{y} - \mathbf{s}_o)}_{\text{nudge, if } \beta>0}),\ 0, 1\big).
$$

This is iterated for `n_settle_steps` steps with integration step $\Delta t$
(`dt`) in **both** phases.

**Two-phase learning rule.** For each transition, the network settles first
freely ($\beta = 0$) to obtain $(\mathbf{s}_h^{\text{free}}, \mathbf{s}_o^{\text{free}})$,
then with the target nudged ($\beta > 0$) to obtain
$(\mathbf{s}_h^{\text{nudge}}, \mathbf{s}_o^{\text{nudge}})$. The EP gradient is
the finite-difference of local correlations between the two equilibria:

$$
\Delta W_{ho} \propto \tfrac{1}{\beta}\big(\mathbf{s}_o^{\text{nudge}} \otimes \mathbf{s}_h^{\text{nudge}} - \mathbf{s}_o^{\text{free}} \otimes \mathbf{s}_h^{\text{free}}\big),
$$

$$
\Delta W_{ih} \propto \tfrac{1}{\beta}\big(\mathbf{s}_h^{\text{nudge}} \otimes \mathbf{x}_t - \mathbf{s}_h^{\text{free}} \otimes \mathbf{x}_t\big),
$$

with analogous updates for the biases $\mathbf{b}_h, \mathbf{b}_o$. Updates are
accumulated over all $L-1$ transitions of a sequence, averaged, and applied with
learning rate $\eta$ scaled by the transition count. Because both correlations
are computed from **locally available** pre- and post-synaptic activity, the rule
needs no back-propagated error signal.

**Recall (autoregressive rollout).** Given a prompt, generation seeds with the
last prompt vector and feeds each free-phase output back as the next input:

$$
\mathbf{s}_{t+1} = \mathrm{settle}_{\beta=0}(\mathbf{s}_t)\big|_{\text{output}},
\qquad t = 0, 1, \dots, L-1.
$$

## Strengths & Weaknesses
- **Strengths**: Biologically plausible, fully local two-phase learning rule (no back-propagation, no separate feedback path); a genuine hidden layer with nonlinear energy dynamics gives it more expressive capacity than the single-matrix Hopfield baseline; the canonical EP formulation makes it a clean reference point for the DG / EWC extensions.
- **Weaknesses**: Vulnerable to catastrophic forgetting — sequential training on new sequences overwrites the energy minima of earlier ones (this is the motivation for the DG pattern-separation and EWC consolidation variants); each transition requires two full settling relaxations, so training is markedly slower than closed-form baselines; sensitive to the settling hyperparameters ($\Delta t$, `n_settle_steps`, $\beta$), which trade off equilibrium accuracy against stability and cost.

## Hyperparameters

Benchmark defaults are set in the model registry (`bin/run_benchmark.py`, key
`original_eqprop`); the feature dimensionality $N$ is inferred from the dataset's
embedding width at fit time.

| Hyperparameter | Symbol | Default | Description |
|---|---|---|---|
| Feature dimension | $N$ (`n_features`) | dataset-dependent | Width of the input/output observation vector. |
| Hidden width | $H$ (`n_hidden`) | `64` | Number of hidden units in the energy layer. |
| Learning rate | $\eta$ (`learning_rate`) | `0.1` (registry); `0.01` (call default) | Step size of the EP weight update. |
| Nudge strength | $\beta$ (`beta`) | `0.5` | Clamping strength in the nudge phase; small-but-positive. |
| Settling steps | — (`n_settle_steps`) | `50` | Iterations to approximate steady-state per phase. |
| Training epochs | — (`n_epochs`) | `20` (registry); `100` (call default) | Full passes over each sequence's adjacent pairs. |
| Integration step | $\Delta t$ (`dt`) | `0.5` | Step size of the settling dynamics (stability vs. speed). |
| Hidden activation | — (`activation`) | `"tanh"` | Activation for the settling helpers (`tanh` / `sigmoid`); note the settle loop clips state to $[0,1]$. |
| Output activation | — (`output_activation`) | `None` (→ matches hidden) | Optional distinct output nonlinearity (`clip` / `tanh` / `sigmoid`). |
| Random seed | — (`seed`) | `42` | RNG seed for reproducibility. |

**Trainable parameter count.** The model has two weight matrices and two bias
vectors:

$$
\#\text{params} = \underbrace{H \times N}_{W_{ih}} + \underbrace{N \times H}_{W_{ho}} + \underbrace{H}_{\mathbf{b}_h} + \underbrace{N}_{\mathbf{b}_o} = 2HN + H + N.
$$

For example, at $N = 128$, $H = 64$ the network holds $2(64)(128) + 64 + 128 = 16{,}576$ parameters.
