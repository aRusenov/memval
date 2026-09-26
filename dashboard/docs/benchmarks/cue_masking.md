# Cue Masking Benchmark

## Biological Context
An animal that once met a lion at a water hole should not need the whole scene back to re-experience it: the water hole alone, or a lion-shaped silhouette, ought to be enough. Cues arrive incomplete in two independent ways. They can be **corrupted** — noise entering at every stage from the receptor to the trace — which is what the [Noise Tolerance](noise_tolerance.md) benchmark sweeps. Or they can be **partial**: some of the original experience is simply not present. This benchmark is the second one.

The behavioural evidence is specific about what partial recall looks like. When people study events made of several elements and are cued with one, the rest return *together* rather than independently (Horner et al., 2015), and recognition falls off gradually as less of a learned image is shown (Vieweg et al., 2019). The attractor account puts this in CA3, where plasticity among features co-active at one moment binds them into a single stored pattern, so a fragment drives the remainder back into activity.

## Computational Assay
A swept fraction of the cue's **feature dimensions** is removed and the rest left exact, then the masked cue is re-normalised onto the unit sphere. Renormalisation is not cosmetic: without it, masking mostly shrinks the cue, and since decoding is cosine-based and many arms are linear, a smaller cue gives a smaller prediction in nearly the same direction — masking would look almost free for a reason unrelated to completion.

The substrate is the **hierarchical encoder**, not the usual symbolic one. Symbolic word embeddings are random vectors on a hypersphere: their dimensions carry no individual meaning, so masking a subset acts like a random projection that shrinks similarity to everything at once, measuring the vocabulary's geometry rather than the model. The hierarchy's columns are owned by nodes of a tree, so each item's features split into two blocks that mean different things:

- **shared** — inherited from its ancestors. Category structure, held in common with its siblings. *"It is a bird."*
- **identity** — owned by its own leaf. What separates it from its siblings. *"It is* this *bird."*

Three masking modes run over the same grid: `random` (the structure-agnostic graded fragment), `shared` (category features first), and `identity` (identity features first). Both probes of the completion definition are run — cued recall masks every position independently; autoregressive rollout masks **only the initial cue** and then free-runs, which is what makes an initial error's propagation visible.

### The model-free reference
Before any model is built, the section computes whether a masked cue still *names* its own item under a plain matched filter. This does two jobs. Below it, failures are not the model's: once several studied items' masked cues are identical the network receives one input for several targets and cannot do better than guess, so every scored metric is restricted to the region where the reference still holds. Above it, it is a baseline worth beating — an arm recalling more than the reference is recovering a target the degraded cue no longer specifies, which is pattern completion as against cue matching.

### Metrics Evaluated
- **In-bound recall** (`mask_<mode>_recall_in_bound`): cued recall averaged only over masked fractions where the cue still identifies its item. The scored readout, and the one that cannot be inflated by cue collisions.
- **Mask tolerance** (`mask_random_tolerance`): the largest fraction of the cue that can be discarded with recall still at or above 0.5. The structural counterpart of the noise tolerance threshold, same sign convention — higher is more robust.
- **Margin** (`mask_random_margin_in_bound`): target-minus-best-competitor cosine. Keeps resolving after recall has floored, which is where a masking curve gets interesting.
- **Completion advantage** (`mask_<mode>_completion_advantage`): recall minus the matched-filter reference. Positive means the arm completes rather than merely matches.
- **Block asymmetry** (`mask_block_asymmetry`): recall under identity-masking minus recall under shared-masking, in the window where both remove the same number of features and both cues remain identifiable. An architectural fingerprint, never a quality: above zero the arm leans on category structure, below zero on identity features, and near zero the cue is a bag of dimensions to it — the holistic-retrieval null.
