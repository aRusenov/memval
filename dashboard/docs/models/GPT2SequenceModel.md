# GPT-2 Sequence Baseline Model

> **Based on** Radford et al. (2019), *"Language Models are Unsupervised Multitask Learners"* (OpenAI) — the GPT-2 autoregressive Transformer language model. Loaded and fine-tuned through the HuggingFace **Transformers** library (Wolf et al., 2020), *"Transformers: State-of-the-Art Natural Language Processing,"* EMNLP 2020 (System Demonstrations).

## Architecture & Rationale
The `GPT2SequenceModel` is an autoregressive Transformer language-model baseline. It fine-tunes a pre-trained **GPT-2 small** backbone (the HuggingFace `gpt2` checkpoint, ~124M parameters) to evaluate how a modern deep-learning architecture doing next-token prediction compares against the biologically constrained hippocampal models. It is the "capacity ceiling / biological-implausibility" reference point: strong sequence modelling, but trained by global back-propagation with none of the locality or pattern-separation constraints of the EqProp and Hopfield baselines.

Because the benchmarks operate on continuous `SymbolicEncoder` embeddings rather than natural-language text, the wrapper bridges the two: benchmark events are mapped to discrete vocabulary words (e.g. `"apple"`, `"banana"`, `"orange"`), and context vectors are mapped to unique textual prefix tokens (e.g. `"ctx0"`) prepended to each prompt. This lets the same `fit_sequence` / `predict_next` / `recall` interface used by every other model drive a Transformer unchanged.

## Integration & API Usage
The model is used entirely through the HuggingFace `transformers` API rather than a from-scratch implementation:

- **Loading.** `GPT2LMHeadModel.from_pretrained("gpt2")` and `GPT2Tokenizer.from_pretrained("gpt2")` load the pre-trained backbone and tokenizer (with `from_pretrained=True`)
- **Feeding continuous embeddings.** Instead of passing token IDs, the benchmark's `SymbolicEncoder` embeddings are pushed through a trainable `torch.nn.Linear` projection (from the encoder's `embedding_dim` to GPT-2's hidden size `n_embd = 768`) and handed to the model via the `inputs_embeds` argument. This allows us to train the model on continuous embeddings, opening the possibility of injecting graded noise into the models's input. 
- **Fine-tuning.** Training uses the model's built-in causal-LM objective: a single `model(inputs_embeds=..., labels=...)` call returns the cross-entropy `loss`, which is optimized with `torch.optim.AdamW` over **both** the Transformer weights and the input projection. `fit_sequence` fine-tunes on one sequence; `fit_sequences` accepts several `(sequence, context)` pairs at once.
- **Retrieval.** `predict_next` runs a forward pass under `torch.no_grad()`, then **masks the output logits down to the benchmark vocabulary's token IDs** before taking the argmax — this prevents the model from hallucinating out-of-vocabulary words. The selected token is mapped back to its word and then to the corresponding `SymbolicEncoder` embedding, so the return type matches the rest of the framework. `recall` seeds from the prompt and rolls this forward autoregressively.

## Strengths & Weaknesses
- **Strengths**: High capacity for long and complex sequences; pre-trained weights provide strong prior representations; excellent recall under noise when trained on single items; drops into the shared model interface unchanged, so it benchmarks head-to-head with the hippocampal models.
- **Weaknesses**: Biologically implausible (relies on global back-propagation, dense attention, and backprop-through-time); suffers from severe **catastrophic forgetting** when fine-tuned sequentially on multiple tasks without replay or regularization; cannot directly process continuous/spatial coordinates without the tokenization bridge, so it is registered for the **symbolic** modality only (inapplicable to the raw spatial path-integration suite).

## Hyperparameters

Benchmark defaults are set in the model registry (`bin/run_benchmark.py`, key
`gpt2`); the encoder (and therefore the input `embedding_dim`) is supplied by the
benchmark at construction time.

| Hyperparameter | Default | Description |
|---|---|---|
| Encoder | required | `SymbolicEncoder` used to map words ↔ embeddings and to define the allowed vocabulary. |
| Training epochs (`n_epochs`) | `10` (registry); `100` (call default) | Fine-tuning passes over each sequence. The registry keeps this low so the suite runs fast. |
| Learning rate (`learning_rate`) | `2e-4` | Step size for the AdamW fine-tuning of the Transformer + projection. |
| Pre-trained weights (`from_pretrained`) | `True` | Load the `gpt2` checkpoint; `False` initializes the same architecture randomly (untrained control). |
| Device (`device`) | `"auto"` (registry); `"cpu"` (call default) | Resolves `"auto"` to `mps` / `cuda` / `cpu`. |

**Trainable parameter count.** Fine-tuning updates the full pre-trained GPT-2
small backbone (~124M parameters) together with the added input projection
(`embedding_dim × 768`, bias-free) — the projection is the only architecture-side
addition; everything else is the stock HuggingFace model.
