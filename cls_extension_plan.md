# CLS Extension for DGEqPropSequenceNetwork

## Overview

This document describes a proposed extension to
[`DGEqPropSequenceNetwork`](memval/models/baselines/dg_eqprop.py) that
implements a two-memory **Complementary Learning Systems (CLS)** architecture
modelled after hippocampal-neocortical interaction.

The current `DGEqPropSequenceNetwork` already provides strong pattern
separation via the Dentate Gyrus k-WTA layer and contrastive Hebbian plasticity
via the Equilibrium Propagation learning rule. However, all learned associations
are stored in a single weight matrix. The CLS extension splits this into two
complementary memories with distinct learning dynamics:

| Property | Hippocampal store (fast) | Neocortical store (slow) |
|---|---|---|
| Learning speed | Immediate (1 epoch) | Gradual (many replay cycles) |
| Capacity | Limited (episodic buffer) | Effectively unlimited |
| Representation | Sparse, context-tagged | Dense, compressed |
| Forgetting | Deliberately bounded by buffer size | Resistant (distributed weights) |
| Biological analog | CA3 / CA1 rapid Hebbian binding | Neocortical slow consolidation |

---

## Proposed Architecture

```
  New sequence ──► DG (k-WTA)
                      │
               ┌──────▼──────────────────────┐
               │  Hippocampal EqProp (fast)   │  ← fits in 1–5 epochs
               │  High LR, small weight mat.  │
               └──────┬──────────────────────┘
                      │ episodic replay buffer
               ┌──────▼──────────────────────┐
               │  Neocortical EqProp (slow)   │  ← trains only during consolidation
               │  Low LR, larger weight mat.  │
               └─────────────────────────────┘
                      │
              predict_next / recall
              (weighted blend or fallback)
```

During **online encoding**, only the hippocampal network is updated, and the
sequence is appended to an internal replay buffer.

During **offline consolidation** (`consolidate()`), all buffered sequences are
replayed interleaved (the existing `fit_sequences` pattern) and applied to the
neocortical network with a slower learning rate.

---

## Proposed Changes

### New file: `memval/models/baselines/cls_dg_eqprop.py`

A new class `CLSDGEqPropNetwork(HippocampalModel)` that wraps two instances of
`DGEqPropSequenceNetwork` with different hyperparameters.

#### Key attributes

```python
self.hpc: DGEqPropSequenceNetwork   # fast, high-LR hippocampal store
self.ctx: DGEqPropSequenceNetwork   # slow, low-LR neocortical store

self._replay_buffer: List[Tuple[np.ndarray, Optional[np.ndarray]]]
# (sequence_data, context_data) pairs stored after each fit_sequence call

self.hpc_epochs: int                # epochs per online encoding (e.g. 5)
self.ctx_epochs: int                # epochs per consolidation pass (e.g. 100)
self.ctx_lr_scale: float            # neocortical LR as fraction of hpc LR (e.g. 0.1)
self.blend_alpha: float             # interpolation weight for prediction blending
self.max_buffer_size: int           # cap on replay buffer (oldest evicted first)
```

#### Methods to implement

**`fit_sequence(sequence_data, context_data, **kwargs)`**  
1. Append `(sequence_data, context_data)` to `self._replay_buffer`, evicting
   oldest if at capacity.
2. Call `self.hpc.fit_sequences([(sequence_data, context_data)], epochs=self.hpc_epochs)`.
3. Does **not** touch the neocortical network.

**`fit_sequences(sequences, **kwargs)`**  
Batch variant — adds all pairs to the buffer and calls `hpc.fit_sequences` with
the full list. Useful for deliberate joint training.

**`consolidate(n_cycles: int = 1)`**  
The offline replay entry point:
1. For each cycle, call `self.ctx.fit_sequences(self._replay_buffer, epochs=self.ctx_epochs)`.
2. Optionally scale the neocortical LR by `ctx_lr_scale` relative to `hpc.learning_rate`.
3. Can be called explicitly after a series of `fit_sequence` calls, or wired
   into a training schedule.

**`predict_next(current_event, current_context, **kwargs)`**  
Blend the predictions from both networks:
```python
pred_hpc = self.hpc.predict_next(current_event, current_context)
pred_ctx = self.ctx.predict_next(current_event, current_context)
return self.blend_alpha * pred_hpc + (1 - self.blend_alpha) * pred_ctx
```
`blend_alpha` defaults to 0.5 but can be tuned: setting it to 1.0 isolates
hippocampal recall (good for novel/recent episodes); 0.0 isolates neocortical
recall (good for well-consolidated sequences).

**`recall(prompt_event, length, prompt_context, **kwargs)`**  
Delegates to `predict_next` iteratively (same pattern as in
`DGEqPropSequenceNetwork.recall`).

**`reset_context()`**  
Calls `reset_context()` on both `hpc` and `ctx`.

**`get_latent_state()`**  
Returns a dict combining both networks' latent states under `"hpc"` and `"ctx"` keys.

---

### Modified file: `memval/models/baselines/__init__.py`

Export `CLSDGEqPropNetwork` alongside the existing classes.

---

### New demo notebook: `demos/cls-eqprop-demo.ipynb`

A benchmark notebook parallel to `gpt2-word-span-demo.ipynb` and
`eqprop-demo.ipynb` with the following sections:

1. **Online-only vs consolidated recall** — train `CLSDGEqPropNetwork` on
   three categories sequentially, measure forgetting before and after calling
   `consolidate()`.
2. **Consolidation sweep** — vary `n_cycles` in `consolidate()` (1, 5, 20) and
   plot neocortical recall improvement vs. hippocampal degradation.
3. **Blend alpha sweep** — fixed post-consolidation model, sweep `blend_alpha`
   from 0.0 to 1.0, measure MRR, Span, and Fidelity. This quantitatively
   captures the hippocampal-neocortical trade-off.
4. **Side-by-side comparison** against `DGEqPropSequenceNetwork` and
   `GPT2SequenceModel` on the catastrophic forgetting benchmark already defined
   in `demos/gpt2-word-span-demo.ipynb`.

---

## Open Questions / Design Decisions

1. **Buffer eviction policy**: FIFO (oldest-first) is simplest and mirrors
   biological forgetting curves. An alternative is priority-based eviction
   (keep sequences that the ctx network recalls worst), but this adds complexity.

2. **When to auto-consolidate**: Should `consolidate()` be called manually, or
   should `fit_sequence` trigger it automatically after every N episodes? The
   latter would be more biologically faithful (continuous SWS-like replay) but
   harder to benchmark cleanly.

3. **Shared vs separate DG**: Both `hpc` and `ctx` currently instantiate
   separate `DGEqPropSequenceNetwork` instances, each with their own DG
   projection weights. Sharing the DG weights (`self.hpc.dg = self.ctx.dg`)
   would better reflect biology (a single DG feeds both CA3 and neocortex) and
   halve the number of random projection parameters.

4. **Asymmetric weight matrices**: The neocortical EP network could use a wider
   `n_hidden` (e.g. 512 vs 128) to reflect the larger representational capacity
   of neocortex relative to hippocampus proper.

---

## Verification Plan

### Automated

- Add `tests/test_cls_dg_eqprop.py` covering:
  - `fit_sequence` populates `_replay_buffer`.
  - `consolidate()` reduces recall loss on previously trained sequences.
  - `reset_context()` does not erase weights.
  - `blend_alpha=1.0` and `blend_alpha=0.0` produce outputs matching the
    isolated `hpc` and `ctx` networks respectively.

### Manual

- Run `demos/cls-eqprop-demo.ipynb` end-to-end.
- In the catastrophic forgetting benchmark (Section 3 of the GPT-2 demo),
  verify that `CLSDGEqPropNetwork` after `consolidate()` retains MRR > 0.7 for
  all three categories, while `GPT2SequenceModel` (sequential training, no
  replay) drops to near 0 for the earliest category.
