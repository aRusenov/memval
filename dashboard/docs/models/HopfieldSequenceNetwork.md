# Hopfield Sequence Network

> **Based on** Sompolinsky & Kanter (1986), *"Temporal Association in Asymmetric Neural Networks,"* Physical Review Letters **57**(22), 2861–2864 — the canonical asymmetric-Hopfield model for sequence recall. The error-correcting learning rule follows the Widrow–Hoff / delta rule (Widrow & Hoff, 1960; applied to associative memories by Diederich & Opper, 1987).

## Architecture & Biological Rationale
The `HopfieldSequenceNetwork` is a classic recurrent autoassociative memory network serving as a foundational baseline. The standard Hopfield network stores static binary patterns by creating deep energy basins (attractors) at the stored states using Hebbian learning rules. 

To store *sequences*, the network is modified with **asymmetric, time-delayed connections** (asymmetric Hopfield model). This introduces sequence transitions: when the network settles into a state representing item $t$, the asymmetric delayed synapses nudge the state dynamics to transition into the attractor representing item $t+1$. This is a standard model for associative chain recall in the hippocampus.

## Model Formulation

The network is a single-layer **linear heteroassociator** that maps the current
state (optionally augmented with an exogenous context vector) to the next state,
followed by a rectifying readout. Let $N$ be the feature dimensionality
(`n_features`) and $C$ the context dimensionality (`n_context`).

**State and weights.** At step $t$ the input is the concatenation of the current
pattern $\mathbf{s}_t \in \mathbb{R}^{N}$ and the clamped context
$\mathbf{c}_t \in \mathbb{R}^{C}$:

$$
\mathbf{x}_t = \begin{bmatrix} \mathbf{s}_t \\ \mathbf{c}_t \end{bmatrix} \in \mathbb{R}^{N+C}.
$$

The (asymmetric, non-square) weight matrix $W \in \mathbb{R}^{N \times (N+C)}$
encodes the time-delayed $t \rightarrow t+1$ transitions. Because $W$ is not
symmetric, the dynamics have no Lyapunov energy and instead flow *through* a
chain of transient attractors rather than settling into a single fixed point —
this is precisely the mechanism of Sompolinsky & Kanter's temporal association.

**Prediction (one step).** The next state is produced by a linear map followed
by a ReLU nonlinearity $\phi(u) = \max(0, u)$:

$$
\hat{\mathbf{s}}_{t+1} = \phi\!\left( W \mathbf{x}_t \right).
$$

**Learning (delta / Widrow–Hoff rule).** Rather than the classic Hebbian
outer-product rule, weights are learned by error correction. For each transition
in a training sequence the prediction error is

$$
\mathbf{e}_t = \mathbf{s}_{t+1} - W \mathbf{x}_t,
$$

and the weights are updated by local LMS gradient descent with learning rate
$\eta$:

$$
\Delta W = \eta \, \mathbf{e}_t \, \mathbf{x}_t^{\top}.
$$

This is iterated over all adjacent pairs $(t, t{+}1)$ of a sequence for a number
of `epochs`. In the limit it drives $W$ toward the least-squares (pseudo-inverse)
solution $W^\star = S_{+} X^{\dagger}$, where $X$ stacks the input states
$\mathbf{x}_t$ and $S_{+}$ the targets $\mathbf{s}_{t+1}$ — giving markedly
higher capacity and lower cross-talk than the Hebbian rule.

**Recall (autoregressive rollout).** Given a prompt, generation proceeds by
feeding each prediction back as the next input, optionally clamping a supplied
future-context sequence $\{\mathbf{c}_t\}$:

$$
\mathbf{s}_{t+1} = \phi\!\left( W \begin{bmatrix} \mathbf{s}_t \\ \mathbf{c}_t \end{bmatrix} \right),
\qquad t = 0, 1, \dots, L-1.
$$

## Strengths & Weaknesses
- **Strengths**: Simple, mathematically tractable formulation; very fast, single-step closed-form learning (Hebbian updates); robust pattern completion for static patterns.
- **Weaknesses**: Low storage capacity (capping at $\approx 0.14N$ patterns); highly vulnerable to interference when sequences have high contextual overlap (spurious attractors); struggles with continuous spatial coordinates (since it is naturally formulated for discrete/binary patterns).

## Hyperparameters

Benchmark defaults are set in the model registry (`bin/run_benchmark.py`, key
`hopfield`); dimensionalities $N$ and $C$ are inferred from the dataset's
embedding/context width at fit time.

| Hyperparameter | Symbol | Default | Description |
|---|---|---|---|
| Feature dimension | $N$ (`n_features`) | dataset-dependent | Width of the endogenous pattern vector. |
| Context dimension | $C$ (`n_context`) | `0` | Width of the exogenous clamped context; `0` disables context. |
| Learning rate | $\eta$ (`learning_rate`) | `0.1` | Step size of the delta / Widrow–Hoff update. |
| Training epochs | `epochs` | `1` (call default); `100` in the disambiguation demo | Passes over each sequence's adjacent pairs during `fit_sequence`. |
| Readout activation | — (`activation`) | `"relu"` | Output nonlinearity $\phi(u)=\max(0,u)$. |
| Random seed | — (`seed`) | `42` | RNG seed for reproducibility. |

**Trainable parameter count.** The model has a single weight matrix
$W \in \mathbb{R}^{N \times (N+C)}$ and no biases, so:

$$
\#\text{params} = N \times (N + C).
$$

With context disabled $(C = 0)$ this reduces to $N^2$. For example, at
$N = 128$, $C = 0$ the network holds $16{,}384$ parameters.
