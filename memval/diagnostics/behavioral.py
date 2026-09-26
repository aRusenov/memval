"""
Behavioural tier: what the network actually recalls. Needs only the core
`HippocampalModel` interface (`predict_next` / `recall`) plus an encoder that can
`encode(list_of_symbols) -> array` and `decode(vector) -> symbol`.

`one_step_accuracy` (per-transition, drift-free) is the recommended forgetting
probe: it scores each stored transition independently by clamping the clean cue,
so it isolates the stored association from autoregressive drift (L-step recall
compounds ~acc**L and understates retention).
"""

import numpy as np

__all__ = ["transitions_of", "one_step_accuracy", "recall_accuracy", "one_step_fidelity",
           "retrieval_margin", "retrieval_mrr"]


def transitions_of(seq):
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def _candidate_matrix(encoder, candidates):
    C = encoder.encode(list(candidates))
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C, {w: i for i, w in enumerate(candidates)}


def retrieval_margin(model, encoder, transitions, candidates) -> list:
    """Noise-free cosine margin per transition, ranked against `candidates`.

    ``margin = cos(pred, true_successor) - max_{d != true} cos(pred, d)``. A single
    deterministic pass (no cue noise): sign gives correct/wrong, magnitude gives
    confidence. Isolates representation quality from noise-robustness — a uniformly
    smaller-but-positive margin is graceful degradation; negative margins are
    associations that did not store. Returns one margin per transition."""
    C, idx = _candidate_matrix(encoder, candidates)
    out = []
    for x, y in transitions:
        pred = model.predict_next(encoder.encode([x])[0])
        pv = pred / (np.linalg.norm(pred) + 1e-12)
        sims = C @ pv
        ti = idx[y]
        true_sim = sims[ti]
        others = np.delete(sims, ti)
        out.append(float(true_sim - (others.max() if others.size else -1.0)))
    return out


def retrieval_mrr(model, encoder, transitions, candidates) -> list:
    """Noise-free reciprocal rank per transition: ``1 / rank`` of the true
    successor among `candidates` by cosine to the prediction (deterministic, one
    pass). Graded companion to `retrieval_margin`."""
    C, idx = _candidate_matrix(encoder, candidates)
    out = []
    for x, y in transitions:
        pred = model.predict_next(encoder.encode([x])[0])
        pv = pred / (np.linalg.norm(pred) + 1e-12)
        sims = C @ pv
        ti = idx[y]
        rank = 1 + int(np.sum(sims > sims[ti]))
        out.append(1.0 / rank)
    return out


def one_step_accuracy(model, encoder, transitions) -> float:
    """Fraction of transitions whose one-step prediction decodes to the true
    successor. `transitions`: iterable of (symbol_x, symbol_y)."""
    hits = []
    for x, y in transitions:
        pred = model.predict_next(encoder.encode([x])[0])
        hits.append(encoder.decode(pred) == y)
    return float(np.mean(hits)) if hits else float("nan")


def one_step_fidelity(model, encoder, transitions) -> float:
    """Mean cosine between one-step prediction and the true successor (continuous
    companion to `one_step_accuracy`)."""
    fids = []
    for x, y in transitions:
        pred = model.predict_next(encoder.encode([x])[0])
        pv = pred / (np.linalg.norm(pred) + 1e-12)
        tv = encoder.encode([y])[0]
        tv = tv / (np.linalg.norm(tv) + 1e-12)
        fids.append(float(pv @ tv))
    return float(np.mean(fids)) if fids else float("nan")


def recall_accuracy(model, encoder, seq) -> float:
    """Autoregressive recall accuracy from the first symbol (drifts; use
    `one_step_accuracy` to isolate per-transition retention)."""
    codes = encoder.encode(seq)
    recalled = model.recall(codes[0:1], length=len(seq) - 1)
    decoded = [encoder.decode(r) for r in recalled]
    return float(np.mean([d == t for d, t in zip(decoded, seq[1:])]))
