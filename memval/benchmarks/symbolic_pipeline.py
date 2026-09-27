import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Callable, Dict, List, Type, Optional, Tuple

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.models.capabilities import rollout_mode_of
from memval.benchmarks.probe import (
    SECTION_PROBE_PROTOCOL, resolve_probe, probe_record,
    measure_recall_margin, mean_margin,
)
from memval.benchmarks._selection import (resolve_benchmarks, resolve_benchmark_args,
                                          benchmark_arg, resolve_epochs_default)
from memval.benchmarks.interval_timing import (
    interval_sections_applicable,
    run_interval_generation,
    run_interval_retention,
)
from memval.benchmarks.exposure import (DEFAULT_CRITERION, DEFAULT_MAX_EPOCHS,
                                        epochs_to_criterion, resolve_exposure)

# Canonical, ordered list of benchmark sections in the symbolic suite. Pass any
# subset as ``benchmarks=`` to run_symbolic_pipeline to skip the rest.
# Factor by which a section's exposure may differ from the arm's reference
# exposure before the run flags it. Suite-level policy, not a section knob, so it
# lives here rather than behind --benchmark-args (a pseudo-section name would be
# rejected by the CLI's validation against SYMBOLIC_BENCHMARKS anyway).
#
# The band is on the ratio and is deliberately loose. Reference exposures are
# often 1-3 epochs, where a one-epoch difference is already 2-3x and means
# nothing; 8x is where a section is training at a genuinely different point
# rather than at small-integer noise.
EXPOSURE_DEVIATION_BAND = 8.0

SYMBOLIC_BENCHMARKS = (
    "presentation_duration",   # 1. convergence / duration sweep
    "sequence_length",         # 2. capacity vs list length (+ feedback modes)
    "multiple_sequences",      # 3. catastrophic forgetting (T=2 special case)
    "continual_chain",         # 3b. retention matrix: load axis + acquisition under load
    "paired_associate",        # 3c. AB/AC shared-cue overwrite (+ disjoint-cue control)
    "noise_invariance",        # 4. retrieval-cue Gaussian noise sweep
    "cue_masking",             # 4b. cue COMPLETENESS: graded feature masking
    "cue_availability",        # 4c. cue AVAILABILITY: a whole modality absent (audio x symbolic)
    "semantic_similarity",     # 5. semantic interference
    "symbolic_disambiguation", # 5b. overlapping sequences: graded discriminability + load
    "schema_consistency",      # 6. acquisition rate vs consistency with prior knowledge
    "cognitive_phenomena",     # 6b. L0 behavioural read-outs (Kahana 2020 sec 4); descriptive, unscored
    "interval_retention",      # 7. Serial order 5.4 -- the gap AS THE CUE
    "interval_generation",     # 8. Serial order 5.5 -- the gap AS THE OUTPUT
)

# Sections 7 and 8 are gated on a DECLARED capability and are skipped, without a
# metric, for any arm that does not hold it (`TemporallyClocked` and the strictly
# stronger `TimingPredictive` respectively). The scorer then reports those rows
# NOT APPLICABLE rather than zero -- an arm whose fit_sequence takes **kwargs
# would swallow `intervals=` and look like a failure instead of an abstention.
# Today only DTSESNSequenceNetwork declares either. See
# memval/benchmarks/interval_timing.py.

# Removed 2026-09-01 (docs/capacities/capacity_coverage_audit.md D5/D6):
#   letter_noise     -- scored the vocabulary's edit-distance geometry, not the
#                       model. Because SymbolicEncoder embeddings carry no
#                       orthographic structure, corrupting a surface form and
#                       snapping back via edit_distance_nearest returns either
#                       the original word (no perturbation) or a different
#                       vocabulary word (a wholly different random vector), so
#                       the sweep measures P(cued with the wrong item). Measured
#                       across-model spread over 7 architectures: 0.000 for
#                       `insert` and `transpose`, <=0.111 for the rest, against
#                       0.711 for the sigma sweep it duplicated.
#   interval_decay   -- the batch suite has no time axis; the only interval it
#                       can express is count of interposed items, i.e.
#                       interference, already covered by multiple_sequences and
#                       the retention matrix. Elapsed-time intent moves to the
#                       online suite (isi_tolerance / interval_encoding).

def _get_model_kwargs(base_kwargs: Dict[str, Any], target_epochs: int) -> Dict[str, Any]:
    """Helper to copy model kwargs and override epoch parameters."""
    kwargs = dict(base_kwargs)
    kwargs["n_epochs"] = target_epochs
    kwargs["epochs"] = target_epochs
    return kwargs

def load_vocab() -> Dict[str, str]:
    """Loads vocabulary from the external json file or falls back to defaults."""
    try:
        # Resolve path relative to this file
        vocab_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "vocab.json")
        )
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load external vocab: {e}")
    
    # Standard fallback vocab
    return {
        'apple': 'fruit', 'banana': 'fruit', 'orange': 'fruit', 'grape': 'fruit', 'pear': 'fruit',
        'peach': 'fruit', 'plum': 'fruit', 'cherry': 'fruit', 'kiwi': 'fruit', 'mango': 'fruit',
        'lemon': 'fruit', 'lime': 'fruit',
        'cat': 'animal', 'dog': 'animal', 'cow': 'animal', 'horse': 'animal', 'sheep': 'animal',
        'pig': 'animal', 'lion': 'animal',
        'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle',
        'hammer': 'tool', 'wrench': 'tool',
        'red': 'color', 'blue': 'color', 'green': 'color',
        'one': 'number', 'two': 'number', 'three': 'number'
    }

# ==========================================
# Metric Helper Functions
# ==========================================
def supports_batched_probe(network: Any) -> bool:
    """Whether this arm can settle many independent cues in one call.

    Opt-in, and deliberately nominal rather than a `hasattr` on the method: the
    batched path is only equivalent for an arm whose `predict_next` is PURE (the
    `HippocampalModel` contract) and whose per-sample states do not interact. An
    arm that carries state across probes must not declare it -- the batched
    rollout advances every trial in lockstep and calls `reset_context` once for
    the whole batch rather than once per trial.

    Everything without the flag takes the original loop, unchanged, so adding
    this cannot move any existing arm's numbers.
    """
    return bool(getattr(network, "batched_probe", False))


def _probe_batch(network: Any, cues: np.ndarray, context: Optional[np.ndarray]) -> np.ndarray:
    """One prediction per row of `cues`, batched where the arm allows it."""
    if supports_batched_probe(network):
        return np.asarray(network.predict_next_batch(cues, current_context=context))
    return np.asarray([network.predict_next(c, current_context=context) for c in cues])


def measure_recall_associative(
    network: Any,
    words: List[str],
    encoder: SymbolicEncoder,
    decoder: SymbolicDecoder,
    context_vec: Optional[np.ndarray] = None,
    n_trials: int = 1,
    noise_scale: float = 0.0,
    cue_transform: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
) -> np.ndarray:
    """One-step cued recall: each position probed independently with a noisy cue.

    ``cue_transform(vec, position) -> vec`` is applied to the cue *after* the
    Gaussian noise, so it composes rather than replaces: pass ``noise_scale=0``
    for a pure structural manipulation, or leave it to cross corruption with
    whatever the transform does. Defaults to ``None``, which is bit-for-bit
    today's behaviour. `cue_masking` uses it to mask cue dimensions.
    """
    success_counts = np.zeros(len(words))
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])

    if supports_batched_probe(network):
        # Same probes, same order, one settle. Every (trial, position) pair here
        # is independent by construction -- this is the one-step cued protocol --
        # so collecting the cues first and predicting once is not an
        # approximation. The noise is drawn in the IDENTICAL nested order, so an
        # arm that declares the flag sees the same cues it would have seen.
        cues, positions = [], []
        for _ in range(n_trials):
            for i in range(len(words) - 1):
                clean_evt = encoder.encode([words[i]])[0]
                noisy_evt = clean_evt + np.random.normal(0, noise_scale,
                                                         encoder.embedding_dim)
                if cue_transform is not None:
                    noisy_evt = cue_transform(noisy_evt, i)
                cues.append(noisy_evt)
                positions.append(i)
        network.current_t = positions[-1] if positions else 0
        preds = _probe_batch(network, np.asarray(cues), context)
        for i, pred in zip(positions, preds):
            if decoder.decode(pred, top_k=1)[0] == words[i + 1]:
                success_counts[i + 1] += 1
        success_counts[0] = n_trials
        return success_counts / n_trials

    # Disable noise or do standard run based on model requirements
    for _ in range(n_trials):
        for i in range(len(words) - 1):
            network.current_t = i
            # Encode and add noise
            clean_evt = encoder.encode([words[i]])[0]
            noisy_evt = clean_evt + np.random.normal(0, noise_scale, encoder.embedding_dim)
            if cue_transform is not None:
                noisy_evt = cue_transform(noisy_evt, i)
            
            # Predict
            pred = network.predict_next(noisy_evt, current_context=context)
            
            # Decode and check match
            decoded_word = decoder.decode(pred, top_k=1)[0]
            if decoded_word == words[i + 1]:
                success_counts[i + 1] += 1
                
    success_counts[0] = n_trials
    return success_counts / n_trials

#: Rollout ownership for the autoregressive probes.
#: "model" -> the arm's own `recall()` runs the rollout under the protocol it
#: declares in `rollout_mode`. Everything else is the HARNESS taking the rollout
#: over and imposing its own feedback projection on every arm alike.
ROLLOUT_MODES = ("model", "raw", "l2", "quantized")
HARNESS_CONTROLLED_MODES = ("raw", "l2", "quantized")


def rollout_owner_label(feedback_mode: str) -> str:
    """Short human-readable tag naming who ran the rollout.

    Any plot or metric built from a harness-controlled mode must carry this, so
    a reader can tell an arm's own behaviour from a protocol the harness forced
    on it.
    """
    if feedback_mode == "model":
        return "model-owned rollout"
    return f"harness-controlled rollout (feedback={feedback_mode})"


def _measure_recall_model_owned(
    network: Any,
    words: List[str],
    encoder: SymbolicEncoder,
    decoder: SymbolicDecoder,
    context_vec: Optional[np.ndarray] = None,
    n_trials: int = 1,
    noise_scale: float = 0.0,
) -> np.ndarray:
    """Serial-position curve with the ROLLOUT OWNED BY THE MODEL.

    One `recall()` call per trial: the arm decides what it feeds back and what
    state it carries. An arm that reduces the prompt to its last row behaves
    exactly as under `feedback_mode="raw"` (verified in
    tests/test_rollout_mode.py), so this changes nothing for the memoryless
    arms and lets latent-state arms roll forward as designed.
    """
    success_counts = np.zeros(len(words))
    recall_len = len(words) - 1

    for _ in range(n_trials):
        if hasattr(network, "reset_context"):
            network.reset_context()

        cue = encoder.encode([words[0]])[0] + np.random.normal(
            0, noise_scale, encoder.embedding_dim
        )

        kwargs = {}
        if context_vec is not None:
            # Tile the single context across the recall horizon; arms that index
            # `prompt_context[t]` need one row per step, and arms that ignore
            # context are unaffected.
            kwargs["prompt_context"] = np.tile(
                np.asarray(context_vec).reshape(1, -1), (recall_len, 1)
            )

        try:
            recalled = np.asarray(network.recall(cue, length=recall_len, **kwargs))
        except TypeError:
            # A few arms declare `recall(self, prompt_event, length)` with no
            # context parameter at all (original_eqprop, eq_prop, hopfield).
            recalled = np.asarray(network.recall(cue, length=recall_len))

        for i in range(min(recall_len, len(recalled))):
            if decoder.decode(recalled[i], top_k=1)[0] == words[i + 1]:
                success_counts[i + 1] += 1

    success_counts[0] = n_trials
    return success_counts / n_trials


def rollout_recall_and_margin(
    network: Any,
    words: List[str],
    encoder: SymbolicEncoder,
    n_trials: int = 1,
    noise_scale: float = 0.0,
    feedback_mode: str = "l2",
) -> Tuple[np.ndarray, np.ndarray]:
    """Autoregressive rollout from a corrupted INITIAL cue, with the margin.

    Returns ``(recall, margin)``, each of length ``len(words) - 1`` (one entry
    per predicted position). ``margin[k]`` is ``cos(pred, target) - max
    cos(pred, other)`` over the whole vocabulary at rollout step k, averaged
    over trials -- the graded quantity behind the boolean recall, which keeps
    resolving after recall has floored. Only item 0's cue carries the noise;
    every later input is the model's own prediction under ``feedback_mode``
    ("raw", "l2" or "quantized", harness-controlled).
    """
    E = np.asarray(encoder.embeddings, dtype=float)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    ctx = np.array([1.0])
    n_steps = len(words) - 1
    hits = np.zeros(n_steps)
    margins = np.zeros(n_steps)
    for _ in range(n_trials):
        if hasattr(network, "reset_context"):
            network.reset_context()
        current = encoder.encode([words[0]])[0] + np.random.normal(
            0, noise_scale, encoder.embedding_dim)
        for i in range(n_steps):
            network.current_t = i
            pred = np.asarray(network.predict_next(current, current_context=ctx), dtype=float)
            pv = pred / (np.linalg.norm(pred) + 1e-12)
            sims = E @ pv
            ti = encoder.word_to_idx[words[i + 1]]
            others = np.delete(sims, ti)
            margins[i] += float(sims[ti] - (others.max() if others.size else -1.0))
            hits[i] += float(int(np.argmax(sims)) == ti)
            if feedback_mode == "quantized":
                current = encoder.embeddings[int(np.argmax(sims))]
            elif feedback_mode == "l2":
                n = float(np.linalg.norm(pred))
                current = pred / n if n > 1e-12 else pred
            else:
                current = pred
    return hits / n_trials, margins / n_trials


def measure_recall_autoregressive(
    network: Any,
    words: List[str],
    encoder: SymbolicEncoder,
    decoder: SymbolicDecoder,
    context_vec: Optional[np.ndarray] = None,
    n_trials: int = 1,
    noise_scale: float = 0.0,
    feedback_mode: str = "raw",
    cue_transform: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
) -> np.ndarray:
    """Chained autoregressive recall loop: inject noise for the first item, then
    feed each prediction back as the next input.

    ``feedback_mode`` selects **who owns the rollout**.

      - ``"model"`` (default): the model owns it. One ``model.recall(cue, length)``
                         call; the arm applies whatever protocol it declares in
                         ``rollout_mode``, so a latent-state arm rolls forward in
                         latent space instead of being forced through
                         observation-space feedback it never uses. This is the
                         default because the alternative silently imposes the
                         harness's protocol on every arm.

    The remaining modes are **harness-controlled**: the harness takes the rollout
    over and projects the prediction back onto the encoder's input manifold
    itself, overriding the arm's own protocol. They exist to isolate error drift,
    and any figure or metric built from them must say so (see
    ``HARNESS_CONTROLLED_MODES`` and ``rollout_owner_label``).

      - ``"raw"``:       feed the prediction back verbatim (original behaviour).
                         Output magnitude is unconstrained, so it drifts off the
                         encoder's unit-norm manifold and errors compound.
      - ``"l2"``:        L2-renormalize the prediction to the unit sphere, matching
                         the encoder's unit-norm embeddings. Fixes magnitude drift.
      - ``"quantized"``: snap to the model's own decoded symbol and feed back its
                         clean embedding (codebook feedback). Drift-free upper bound
                         for a discrete/symbolic domain; hides sub-threshold
                         representational degradation.

    Decoding is always cosine-based (scale-invariant); only the fed-back vector
    changes between modes.
    """
    if feedback_mode not in ROLLOUT_MODES:
        raise ValueError(f"Unknown feedback_mode: {feedback_mode!r}")

    if feedback_mode == "model":
        return _measure_recall_model_owned(
            network, words, encoder, decoder, context_vec, n_trials, noise_scale
        )

    success_counts = np.zeros(len(words))
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])

    if supports_batched_probe(network):
        # Trials are independent chains, so they can be advanced in lockstep:
        # step i of every trial is one batched settle. Noise enters only at
        # i == 0, and is drawn per trial in the SAME order as the serial loop
        # below, so the cues are identical.
        #
        # `reset_context` runs once for the batch rather than once per trial,
        # which is why `batched_probe` must only be declared by an arm whose
        # `predict_next` is pure (see `supports_batched_probe`).
        if hasattr(network, "reset_context"):
            network.reset_context()
        clean_first = encoder.encode([words[0]])[0]
        current = []
        for _ in range(n_trials):
            evt = clean_first + np.random.normal(0, noise_scale, encoder.embedding_dim)
            if cue_transform is not None:
                evt = cue_transform(evt, 0)
            current.append(evt)
        current = np.asarray(current)

        for i in range(len(words) - 1):
            network.current_t = i
            preds = _probe_batch(network, current, context)
            nxt = np.empty_like(current)
            for t, pred in enumerate(preds):
                decoded_word = decoder.decode(pred, top_k=1)[0]
                if decoded_word == words[i + 1]:
                    success_counts[i + 1] += 1
                if feedback_mode == "raw":
                    nxt[t] = pred
                elif feedback_mode == "l2":
                    norm = np.linalg.norm(pred)
                    nxt[t] = pred / norm if norm > 1e-8 else pred
                else:  # "quantized"
                    nxt[t] = encoder.encode([decoded_word])[0]
            current = nxt

        success_counts[0] = n_trials
        return success_counts / n_trials

    for _ in range(n_trials):
        if hasattr(network, "reset_context"):
            network.reset_context()
        current_evt = None
        for i in range(len(words) - 1):
            network.current_t = i
            if i == 0:
                # Encode and add noise for the first item only
                clean_evt = encoder.encode([words[0]])[0]
                current_evt = clean_evt + np.random.normal(0, noise_scale, encoder.embedding_dim)
                # Same hook as the associative probe, and the same reason it is
                # applied at i == 0 only: this protocol manipulates the INITIAL
                # cue and lets the rest of the chain show what propagates.
                if cue_transform is not None:
                    current_evt = cue_transform(current_evt, 0)

            # Predict
            pred = network.predict_next(current_evt, current_context=context)

            # Decode and check match
            decoded_word = decoder.decode(pred, top_k=1)[0]
            if decoded_word == words[i + 1]:
                success_counts[i + 1] += 1

            # Project the prediction back onto the input manifold per feedback_mode
            if feedback_mode == "raw":
                current_evt = pred
            elif feedback_mode == "l2":
                norm = np.linalg.norm(pred)
                current_evt = pred / norm if norm > 1e-8 else pred
            else:  # "quantized"
                current_evt = encoder.encode([decoded_word])[0]

    success_counts[0] = n_trials
    return success_counts / n_trials

def mean_recall_rate(recall_curve: np.ndarray) -> float:
    """Computes Mean Recall Rate (MRR), excluding position 0 (always correct cue)."""
    if len(recall_curve) <= 1:
        return 0.0
    return float(np.mean(recall_curve[1:]))

def memory_span(recall_curve: np.ndarray, threshold: float = 0.75) -> int:
    """Computes Memory Span: the longest prefix recalled reliably above threshold."""
    span = 0
    for r in recall_curve[1:]:
        if r >= threshold:
            span += 1
        else:
            break
    return span

def recall_fidelity(
    network: Any,
    words: List[str],
    encoder: SymbolicEncoder,
    n_trials: int = 1,
    noise_scale: float = 0.0
) -> float:
    """Average cosine similarity between predicted embedding and ground-truth target."""
    gt_embs = encoder.encode(words)
    total_sim, count = 0.0, 0
    for _ in range(n_trials):
        network.current_t = 0
        for i in range(len(words) - 1):
            clean_evt = encoder.encode([words[i]])[0]
            noisy = clean_evt + np.random.normal(0, noise_scale, encoder.embedding_dim)
            pred = network.predict_next(noisy, current_context=np.array([1.0]))
            
            norm_pred = np.linalg.norm(pred)
            norm_gt = np.linalg.norm(gt_embs[i + 1])
            if norm_pred > 0 and norm_gt > 0:
                sim = float(np.dot(pred, gt_embs[i + 1])) / (norm_pred * norm_gt)
            else:
                sim = 0.0
            total_sim += sim
            count += 1
    return total_sim / count if count > 0 else 0.0


def run_to_convergence(
    model_class: Type[Any],
    model_kwargs: Dict[str, Any],
    words: List[str],
    encoder: SymbolicEncoder,
    decoder: SymbolicDecoder,
    mrr_threshold: float = 0.95,
    max_epochs: int = 500,
    n_trials: int = 1,
    noise_scale: float = 0.0,
    convergence_patience: int = 4
) -> Dict[str, Any]:
    """
    Train a fresh model instance using exponential epoch checkpointing until
    MRR >= mrr_threshold on ``convergence_patience`` checkpoints or the epoch
    budget is exhausted.

    Checkpoint schedule: powers of two (1, 2, 4, ...) up to and including
    max_epochs, e.g. max_epochs=2048 -> 1, 2, ..., 1024, 2048 and
    max_epochs=500 -> 1, 2, ..., 256, 500.

    ``convergence_patience`` is the number of checkpoints that must cross
    ``mrr_threshold`` before the sweep stops. It defaults to 4 so the trajectory
    keeps sampling past the first crossing and the asymptotic tail is visible:
    a one-shot associator often crosses at epoch 1, which would otherwise
    collapse the curve to a single point. ``convergence_epochs`` still reports
    the *first* crossing (the true convergence point).

    Returns:
        {
          "convergence_mrr": float,       # MRR at convergence (or best achieved)
          "convergence_epochs": int | None,  # epoch where threshold first crossed, None = did not converge
          "convergence_span": int,        # memory_span at convergence
          "converged": bool,              # whether threshold was met
          "mrr_curve": [{"epoch": int, "mrr": float}, ...] # learning trajectory
        }
    """
    # Exponential checkpoint schedule (powers of two) up to and including the
    # budget, so any max_epochs (e.g. 2048) is honoured rather than capped at a
    # hardcoded ceiling.
    checkpoints = []
    ep = 1
    while ep < max_epochs:
        checkpoints.append(ep)
        ep *= 2
    checkpoints.append(max_epochs)
    best_mrr = 0.0
    best_span = 0
    convergence_epochs = None
    crossings = 0
    mrr_curve = []

    for ep in checkpoints:
        try:
            run_kwargs = _get_model_kwargs(model_kwargs, ep)
            if 'encoder' in model_class.__init__.__code__.co_varnames:
                model = model_class(encoder=encoder, n_features=encoder.embedding_dim, **run_kwargs)
            else:
                model = model_class(n_features=encoder.embedding_dim, **run_kwargs)

            word_list_encoded = encoder.encode(words)
            model.fit_sequence(word_list_encoded, epochs=ep)

            # Evaluate cued recall
            model.reset_context()
            curve = measure_recall_associative(
                model, words, encoder, decoder, n_trials=n_trials, noise_scale=noise_scale
            )
            mrr = mean_recall_rate(curve)

            # Evaluate autoregressive recall to calculate span
            curve_auto = measure_recall_autoregressive(
                model, words, encoder, decoder, n_trials=n_trials, noise_scale=noise_scale
            )
            span = memory_span(curve_auto)

            mrr_curve.append({"epoch": ep, "mrr": mrr})

            if mrr > best_mrr:
                best_mrr = mrr
                best_span = span

            # Record the first threshold crossing as the convergence epoch, but
            # keep sampling until `convergence_patience` checkpoints have crossed
            # so the asymptotic tail of the curve is captured.
            if mrr >= mrr_threshold:
                if convergence_epochs is None:
                    convergence_epochs = ep
                crossings += 1
                if crossings >= convergence_patience:
                    break
        except Exception as e:
            print(f"Warning: Convergence trial failed for {ep} epochs: {e}")
            mrr_curve.append({"epoch": ep, "mrr": 0.0})

    return {
        "convergence_mrr": best_mrr,
        "convergence_span": best_span,
        "convergence_epochs": convergence_epochs,
        "converged": convergence_epochs is not None,
        "mrr_curve": mrr_curve
    }


#: Embedding width for the two metric-time sections. Narrower than the suite's
#: 100 because both streams are tiny (6-item), and it matches the demos these
#: sections were promoted from so a discrepancy between them is a real one.
INTERVAL_EMBED_DIM = 64


def _build_interval_model(model_class, model_kwargs):
    """A fresh arm for one metric-time run.

    `n_epochs` is stripped: these sections drive exposure themselves through
    `n_reps` (how many times each stream is presented), and leaving the registry
    value in would multiply the two, which is the same class of mistake as the
    call-level `epochs=` being dropped.
    """
    kw = {k: v for k, v in (model_kwargs or {}).items()
          if k not in ("n_epochs", "epochs")}
    return model_class(n_features=INTERVAL_EMBED_DIM, **kw)


def _plot_interval_retention(out, plots_dir):
    """The sweep, with the trained gaps marked.

    The figure to look at is the step: a model that honours the interval switches
    once, near the trained gaps. One that has merely absorbed slow drift wanders.
    """
    sw = out["series"]["interval_sweep"]
    gaps, branch = sw["gaps"], sw["branch"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.step(gaps, branch, where="mid", color="#2e6296", lw=2.2, marker="o")
    for g in sw["trained_gaps"]:
        ax.axvline(g, color="#AD3F6B", ls=(0, (4, 3)), lw=1.4)
        ax.annotate(f"trained {g:g}s", xy=(g, 1.12), ha="center", fontsize=8.5,
                    color="#AD3F6B")
    xg = out["metrics"].get("interval_crossing_gap")
    if xg == xg:
        ax.axvline(xg, color="#3F7A4F", lw=1.6)
        ax.annotate(f"crossing {xg:.2f}s", xy=(xg, -0.22), ha="center",
                    fontsize=8.5, color="#3F7A4F")
    ax.set_xscale("log")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["short branch", "long branch"])
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlabel("elapsed gap after the shared prefix (s, log scale)")
    ax.set_title(
        "5.4 interval-as-cue: the gap is the only thing telling the branches apart\n"
        f"discrimination {out['metrics']['interval_discrimination_acc']:.2f}  |  "
        f"same stream, gaps removed (control) {out['metrics']['interval_control_acc']:.2f}",
        fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "interval_retention.png"), dpi=150)
    plt.close(fig)


def _plot_interval_generation(out, plots_dir):
    """Generated gaps against trained gaps: tempo on the left, rhythm on the right."""
    ser = out["series"]["interval_generation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    for row, colour in zip(ser["tempo"], ("#2e6296", "#B07A16")):
        xs = np.arange(1, len(row["gaps"]) + 1)
        axes[0].plot(xs, row["gaps"], "-o", color=colour,
                     label=f"{row['label']} (generated)")
        axes[0].axhline(row["true_tempo"], color=colour, ls=(0, (4, 3)), lw=1.2)
    axes[0].set_xlabel("generated step")
    axes[0].set_ylabel("inter-item gap (s)")
    axes[0].set_title(
        f"tempo — mean relative error "
        f"{out['metrics']['tempo_reproduction_error']:.3f}\n"
        "dashed = trained tempo", fontsize=10, loc="left")
    axes[0].legend(fontsize=8.5, frameon=False)

    g, t = ser["rhythm_gaps"], ser["rhythm_true_gaps"]
    xs = np.arange(1, len(g) + 1)
    axes[1].bar(xs - 0.18, t, width=0.34, color="#c9c9c9", label="trained")
    axes[1].bar(xs + 0.18, g, width=0.34, color="#3F7A4F", label="generated")
    axes[1].set_xticks(xs)
    axes[1].set_xlabel("generated step")
    axes[1].set_title(
        f"rhythm — pause in the right place: "
        f"{'yes' if out['metrics']['rhythm_pause_position_acc'] >= 0.5 else 'no'}",
        fontsize=10, loc="left")
    axes[1].legend(fontsize=8.5, frameon=False)

    fig.suptitle("5.5 interval-as-output: the model produces what AND when",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "interval_generation.png"), dpi=150)
    plt.close(fig)


def run_symbolic_pipeline(
    model_class: Type[Any],
    model_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: str = "./results",
    n_trials: int = 30,
    mrr_threshold: float = 0.95,
    max_epochs: int = 500,
    benchmarks: Optional[List[str]] = None,
    benchmark_args: Optional[Dict[str, Dict[str, Any]]] = None,
    epochs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Runs the symbolic benchmarking suite against any model class that conforms
    to the HippocampalModel interface.

    Args:
        model_class: The class of the model to instantiate.
        model_kwargs: Dictionary of arguments to pass to the model constructor.
        output_dir: Directory where results and plots will be saved.
        n_trials: Number of evaluation trials (default: 30).
        mrr_threshold: MRR threshold for convergence (default: 0.95).
        max_epochs: Max epochs for convergence (default: 500).
        benchmarks: Subset of SYMBOLIC_BENCHMARKS to run (default: all). Lets you
            skip the expensive sections while iterating on one.
        benchmark_args: Per-benchmark parameter overrides, keyed by section name
            (e.g. ``{"sequence_length": {"epochs": 500}}``). The trainable
            sections honour an ``epochs`` key (default 300).
            "schema_consistency" instead honours ``branching``,
            ``features_per_node``, ``schema_trials``, ``new_item_trials``,
            ``n_probe_trials``, ``criterion_frac``, ``seed`` and
            ``epochs_per_trial``; it does not read ``epochs``, because its
            dependent variable is trials-to-criterion at one presentation per
            trial.
            "symbolic_disambiguation" honours ``n_episodes``, ``total_length``,
            ``shared_fraction``, ``shared_position``, ``zone_fraction``,
            ``embedding_dim``, ``n_discriminator_dims``, ``disc_scale``,
            ``middle_position`` and ``seed``, plus the usual exposure keys.
            ``shared_position`` is 0.0 for the main sweeps (shared PREFIX, so the
            discriminator is the only thing identifying an episode);
            ``middle_position`` is where the shared-MIDDLE condition puts its
            corridor, leaving a unique prefix before it. Its cost is three sweeps over the same generator, so
            shorten the grids via ``n_episodes``/``total_length`` rather than
            ``epochs``.
            "cue_masking" honours ``branching``, ``features_per_node``,
            ``seq_len``, ``list_scope``, ``n_draws``, ``renormalize``,
            ``feedback_mode`` and ``seed``, plus the usual exposure keys. Its
            cost is dominated by the sweep rather than by training — the grid is
            one point per cue feature, run in three masking modes under two
            probes — so ``features_per_node`` and ``n_draws`` are the knobs to
            turn when it is too slow, not ``epochs``.
        epochs: Global training-epoch override for the fixed-epoch sections. The
            resolved value follows the precedence ``--benchmark-args`` (per
            section) > this ``epochs`` > model default > the section's own
            default. Does not affect the ``presentation_duration`` convergence
            sweep (governed by ``max_epochs``).

    Returns:
        Dict: Dictionary of calculated metrics.
    """
    model_kwargs = model_kwargs or {}
    selected = resolve_benchmarks(benchmarks, SYMBOLIC_BENCHMARKS, "symbolic")
    benchmark_args = resolve_benchmark_args(benchmark_args, SYMBOLIC_BENCHMARKS, "symbolic")

    def section_epochs(section: str, fallback: int) -> int:
        """Resolve a section's epoch count: benchmark_args > --epochs > model
        default > the section fallback."""
        return int(benchmark_arg(
            benchmark_args, section, "epochs",
            resolve_epochs_default(epochs, model_kwargs, fallback)
        ))

    def exposure_policy(section: str, fallback: int, force_criterion: bool = False) -> Dict[str, Any]:
        """Control sections take the registry budget (model_kwargs["n_epochs"]);
        result sections stay on the criterion ladder. Precedence in resolve_exposure."""
        return resolve_exposure(benchmark_args, section, benchmark_arg, fallback,
                                global_epochs=epochs,
                                model_fixed_epochs=(model_kwargs or {}).get("n_epochs"),
                                force_criterion=force_criterion)

    def train_at_exposure(policy, build, fit, score):
        """Train under `policy` and report the exposure it actually took.

        Returns ``(model, record)`` where record carries `epochs`, `reached` and
        `mode`, which the section writes out as `<name>_epochs_to_criterion` and
        `<name>_criterion_reached`. Under a pinned budget `reached` reports
        whether the criterion was met at that budget, so a pinned run is still
        readable against the criterion-referenced one.
        """
        if policy["mode"] == "fixed":
            m = build(policy["epochs"])
            fit(m, policy["epochs"])
            sc = float(score(m))
            return m, {"epochs": policy["epochs"], "reached": sc >= policy["criterion"],
                       "mode": "fixed", "source": policy.get("source"), "score": sc}
        res = epochs_to_criterion(
            lambda: build(policy["max_epochs"]), fit, score,
            criterion=policy["criterion"], max_epochs=policy["max_epochs"])
        return res["model"], {"epochs": res["epochs"], "reached": res["reached"],
                              "mode": "criterion", "source": policy.get("source"), "score": res["score"]}

    # Establish subdirectories
    model_name = model_class.__name__
    run_dir = os.path.join(output_dir, model_name, "symbolic")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style="darkgrid")
    vocab = load_vocab()
    
    # Structure metrics output
    results = {
        "metadata": {
            "model_name": model_name,
            "suite": "symbolic",
            "n_trials": n_trials,
            "model_kwargs": {k: str(v) for k, v in model_kwargs.items()},
            # The `recall()` rollout protocol this arm hardcodes. Recorded so a
            # cross-arm comparison can see when two arms were not scored under
            # the same protocol (see docs/sections/rollout_protocol.md).
            "rollout_mode": getattr(
                rollout_mode_of(model_class), "value", None),
            # How each section probed: clean single cue (the default), or a
            # declared degradation sweep. See memval/benchmarks/probe.py.
            "probe_protocols": {
                sec: probe_record(sec, model_class, n_trials)
                for sec in SYMBOLIC_BENCHMARKS if sec in SECTION_PROBE_PROTOCOL
            },
        },
        "metrics": {},
        "series": {}
    }

    print(f"[symbolic] running benchmarks: {selected}")

    # Shared fixtures used by more than one section (defined unconditionally so a
    # selected section never depends on an earlier, unselected one having run).
    fruit_words = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']

    # ==========================================
    # Exposure baseline and deviation guard
    # ==========================================
    # Every criterion-referenced section settles its own exposure on its own
    # material, which keeps arms comparable WITHIN a section: each is measured at
    # the same functional point (criterion 0.95) rather than at the same epoch
    # count, and MODEL_REGISTRY's defaults span 1 to 300 across the taxonomy.
    #
    # What that leaves unreported is the other axis: for ONE arm, how far each
    # section's exposure sits from a common reference. Without it a cross-section
    # claim like "this arm handles retention but fails disambiguation" silently
    # compares two differently-trained models, and nothing in the results file
    # says so.
    #
    # The reference is the same 7-word list `presentation_duration` uses, settled
    # by the same staircase on the same criterion. It is a REFERENCE, not a
    # budget: sections are not pinned to it, because "converged" is a property of
    # (model, material) and not of the model alone. Pinning would start a section
    # whose material is heavier on an undertrained arm, and every failure there
    # would then be ambiguous between the manipulation and the undertraining.
    # This just makes the spread visible so a reader can judge it.
    _baseline_cache: Dict[str, Any] = {}

    def exposure_baseline() -> Dict[str, Any]:
        """Reference exposure for this arm on the shared fruit list. Cached."""
        if _baseline_cache:
            return _baseline_cache
        try:
            base_vocab = {w: c for w, c in vocab.items() if c == "fruit"}
            enc_b = SymbolicEncoder(base_vocab, embedding_dim=100,
                                    category_variance=0.2, seed=42)
            dec_b = SymbolicDecoder(enc_b)
            emb_b = enc_b.encode(fruit_words)

            def _mk_b(ep, _e=enc_b):
                kw = _get_model_kwargs(model_kwargs, ep)
                if 'encoder' in model_class.__init__.__code__.co_varnames:
                    return model_class(encoder=_e, n_features=_e.embedding_dim, **kw)
                return model_class(n_features=_e.embedding_dim, **kw)

            def _fit_b(m, ep, _x=emb_b):
                if hasattr(m, "n_epochs"):
                    m.n_epochs = ep
                m.fit_sequence(_x, epochs=ep)

            def _score_b(m, _w=fruit_words, _e=enc_b, _d=dec_b):
                m.reset_context()
                return mean_recall_rate(measure_recall_associative(
                    m, _w, _e, _d, n_trials=n_trials, noise_scale=0.0))

            # Same criterion and cap as every section's own staircase, so the
            # reference is measured the way the sections are.
            res = epochs_to_criterion(
                lambda: _mk_b(DEFAULT_MAX_EPOCHS), _fit_b, _score_b,
                criterion=DEFAULT_CRITERION, max_epochs=DEFAULT_MAX_EPOCHS)
            _baseline_cache.update({"epochs": int(res["epochs"]),
                                    "reached": bool(res["reached"]),
                                    "score": float(res["score"])})
        except Exception as e:
            print(f"Warning: exposure baseline failed for {model_name}: {e}")
            _baseline_cache.update({"epochs": None, "reached": False,
                                    "score": float('nan')})
        return _baseline_cache

    # ==========================================
    # 1. Presentation Duration (Convergence-based)
    # ==========================================
    if "presentation_duration" in selected:
        _pn, _ps = resolve_probe("presentation_duration", model_class, n_trials)  # probe protocol
        print(f"Running Presentation Duration convergence benchmarks (threshold: {mrr_threshold}, max epochs: {max_epochs})...")

        # Create default encoder for fruits list (using only fruit category to match notebook)
        fruit_vocab = {w: c for w, c in vocab.items() if c == "fruit"}
        enc_duration = SymbolicEncoder(fruit_vocab, embedding_dim=100, category_variance=0.2, seed=42)
        dec_duration = SymbolicDecoder(enc_duration)

        # Same legacy-global-stream pin as semantic_similarity above. Scope
        # note: the ask was to seed semantic_similarity, but this section was the
        # only symbolic metric still moving across identical runs afterwards
        # (`convergence_span` flipped 6/5/6 over three AHN runs), and it is the
        # same one-line mechanism. Leaving it would mean the suite is
        # reproducible except in one place, which is the hardest state to reason
        # about. Every other symbolic section either seeds already
        # (cue_masking, paired_associate, continual_chain, schema_consistency)
        # or was measured stable over three runs at n_trials=30.
        dur_seed = int(benchmark_arg(benchmark_args, "presentation_duration",
                                     "seed", 42))
        np.random.seed(dur_seed)

        conv = run_to_convergence(
            model_class=model_class,
            model_kwargs=model_kwargs,
            words=fruit_words,
            encoder=enc_duration,
            decoder=dec_duration,
            mrr_threshold=mrr_threshold,
            max_epochs=max_epochs,
            n_trials=_pn,
            noise_scale=_ps
        )

        results["metrics"]["convergence_mrr"] = conv["convergence_mrr"]
        results["metrics"]["convergence_span"] = conv["convergence_span"]
        results["metrics"]["convergence_epochs"] = conv["convergence_epochs"]
        results["metrics"]["converged"] = conv["converged"]
        results["series"]["convergence_curve"] = conv["mrr_curve"]

        # Save Convergence Plot
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
        fig, ax = plt.subplots(figsize=(7, 5))
        epochs_plotted = [point["epoch"] for point in conv["mrr_curve"]]
        mrrs_plotted = [point["mrr"] for point in conv["mrr_curve"]]
        ax.plot(epochs_plotted, mrrs_plotted, '-o', color='purple', label='MRR')
        ax.axhline(y=mrr_threshold, color='red', linestyle='--', label=f'Threshold ({mrr_threshold})')
        if conv["converged"]:
            ax.axvline(x=conv["convergence_epochs"], color='green', linestyle=':', label=f'Converged ({conv["convergence_epochs"]} ep)')

        # Log x-axis (checkpoints grow as powers of two), linear y-axis (MRR). Show
        # every checkpoint as an integer tick spanning the lowest to the highest
        # epoch, and suppress the default log minor ticks (which otherwise render
        # spurious scientific-notation labels like "3e10").
        ax.set_xscale('log', base=2)
        if epochs_plotted:
            ax.set_xlim(epochs_plotted[0] * 0.9, epochs_plotted[-1] * 1.1)
        ax.xaxis.set_major_locator(FixedLocator(epochs_plotted))
        ax.xaxis.set_major_formatter(FixedFormatter([str(e) for e in epochs_plotted]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Epochs (log2 scale)')
        ax.set_ylabel('Mean Recall Rate (MRR)')
        ax.set_title('Convergence Trajectory')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "convergence_curve.png"), dpi=150)
        plt.close(fig)

    # ==========================================
    # 2. Sequence Length
    # ==========================================
    if "sequence_length" in selected:
        _pn, _ps = resolve_probe("sequence_length", model_class, n_trials)  # probe protocol
        seqlen_policy = exposure_policy("sequence_length", 300)
        print(f"Running Sequence Length benchmarks... (exposure: {seqlen_policy['mode']}"
              + (f", criterion={seqlen_policy['criterion']}" if seqlen_policy["mode"] == "criterion"
                 else f", epochs={seqlen_policy['epochs']}") + ")")
        lengths = [3, 5, 7, 9, 11]
        length_mrr = []
        # Exposure is a result here, not a parameter: each length is trained until
        # CUED recall reaches criterion, so every span below is read at the point
        # where "the associations are present" is true rather than at whatever
        # epoch count this arm's registry entry happens to carry.
        length_exposure, length_reached = [], []
        # Compare three autoregressive-feedback modes to separate genuine capacity
        # limits from error drift caused by feeding un-normalized states back.
        feedback_modes = ["raw", "l2", "quantized"]
        length_span = {mode: [] for mode in feedback_modes}

        # Vocabulary pool for fruit
        all_fruits = [word for word, cat in vocab.items() if cat == "fruit"]

        for L in lengths:
            try:
                # Sample words of length L
                test_words = all_fruits[:L] if L <= len(all_fruits) else all_fruits
                # Fill with animal fallback if list length is larger than fruit count
                if len(test_words) < L:
                    animals = [w for w, c in vocab.items() if c == "animal"]
                    test_words += animals[:L - len(test_words)]

                len_vocab = {w: vocab[w] for w in test_words}
                enc_len = SymbolicEncoder(len_vocab, embedding_dim=100, category_variance=0.2, seed=42)
                dec_len = SymbolicDecoder(enc_len)

                encoded_words = enc_len.encode(test_words)

                def _build(ep, _e=enc_len):
                    kw = _get_model_kwargs(model_kwargs, ep)
                    if 'encoder' in model_class.__init__.__code__.co_varnames:
                        return model_class(encoder=_e, n_features=_e.embedding_dim, **kw)
                    return model_class(n_features=_e.embedding_dim, **kw)

                def _fit(m, ep, _x=encoded_words):
                    if hasattr(m, "n_epochs"):
                        m.n_epochs = ep
                    m.fit_sequence(_x, epochs=ep)

                def _score(m, _w=test_words, _e=enc_len, _d=dec_len):
                    m.reset_context()
                    return mean_recall_rate(measure_recall_associative(
                        m, _w, _e, _d, n_trials=_pn, noise_scale=_ps))

                model, exp_rec = train_at_exposure(seqlen_policy, _build, _fit, _score)
                length_exposure.append(exp_rec["epochs"])
                length_reached.append(bool(exp_rec["reached"]))

                model.reset_context()
                curve = measure_recall_associative(
                    model, test_words, enc_len, dec_len, n_trials=_pn, noise_scale=_ps
                )
                length_mrr.append(mean_recall_rate(curve))
                length_span.setdefault("cued", []).append(memory_span(curve))
                for mode in feedback_modes:
                    curve_auto = measure_recall_autoregressive(
                        model, test_words, enc_len, dec_len, n_trials=_pn,
                        noise_scale=_ps, feedback_mode=mode
                    )
                    length_span[mode].append(memory_span(curve_auto))
            except Exception as e:
                print(f"Warning: Length benchmark failed for L={L}: {e}")
                length_mrr.append(0.0)
                length_exposure.append(0)
                length_reached.append(False)
                length_span.setdefault("cued", []).append(0)
                for mode in feedback_modes:
                    length_span[mode].append(0)

        # max_memory_span keeps its original meaning (raw feedback) for backward
        # compatibility; the normalized/quantized variants are reported alongside.
        results["metrics"]["max_memory_span"] = int(np.max(length_span["raw"])) if length_span["raw"] else 0
        # Exposure as a result. `_criterion_reached` gates everything above it:
        # a span read at a censored exposure is a lower bound, not a capacity.
        results["metrics"]["seqlen_epochs_to_criterion"] = (
            float(np.mean(length_exposure)) if length_exposure else float("nan"))
        results["metrics"]["seqlen_max_epochs_to_criterion"] = (
            int(np.max(length_exposure)) if length_exposure else 0)
        results["metrics"]["seqlen_criterion_reached"] = bool(
            length_reached and all(length_reached))
        results["metrics"]["seqlen_exposure_mode"] = seqlen_policy["mode"]
        results["metrics"]["seqlen_exposure_source"] = seqlen_policy.get("source")
        # The cued span at that exposure -- the other half of unrolling_gap, which
        # the audit notes is "one line away and unmade".
        cued_spans = length_span.get("cued", [])
        if cued_spans:
            gaps = [1.0 - (r / c) for r, c in zip(length_span["raw"], cued_spans) if c]
            results["metrics"]["unrolling_gap"] = (
                float(np.mean(gaps)) if gaps else float("nan"))
        results["metrics"]["max_memory_span_l2"] = int(np.max(length_span["l2"])) if length_span["l2"] else 0
        results["metrics"]["max_memory_span_quantized"] = int(np.max(length_span["quantized"])) if length_span["quantized"] else 0
        results["series"]["length_sweep"] = {
            "lengths": lengths,
            "mrr": length_mrr,
            "span": length_span["raw"],  # backward-compatible alias for raw feedback
            "span_raw": length_span["raw"],
            "span_l2": length_span["l2"],
            "span_quantized": length_span["quantized"],
            "span_cued": length_span.get("cued", []),
            "epochs_to_criterion": length_exposure,
            "criterion_reached": length_reached,
        }

        # Save Length Plot
        fig, ax1 = plt.subplots(figsize=(7, 5))
        color = 'tab:blue'
        ax1.set_xlabel('List Length')
        ax1.set_ylabel('Mean Recall Rate (MRR)', color=color)
        ax1.plot(lengths, length_mrr, '-o', color=color, label='MRR (associative)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Memory Span (threshold = 0.75)')
        span_styles = {
            "raw": ('-s', 'tab:red', 'Span (raw feedback)'),
            "l2": ('-^', 'tab:green', 'Span (L2-normalized)'),
            "quantized": ('-D', 'tab:purple', 'Span (quantized)'),
        }
        for mode in feedback_modes:
            marker, mcolor, mlabel = span_styles[mode]
            ax2.plot(lengths, length_span[mode], marker, color=mcolor, label=mlabel)
        ax2.set_ylim(-0.5, max(lengths) + 0.5)

        # Merge both axes' handles into a single legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)

        plt.title('Capacity vs. Sequence Length (autoregressive feedback modes)')
        # Every curve here is a HARNESS-CONTROLLED rollout: the harness overrides
        # each arm's own `rollout_mode` and imposes the same observation-space
        # feedback on all of them. Say so on the figure -- otherwise these read
        # as the arms' own behaviour, which is what the default (model-owned)
        # path measures.
        plt.figtext(0.5, 0.005,
                    "harness-controlled rollout: feedback imposed by the harness, "
                    "overriding each arm's own rollout_mode",
                    ha="center", fontsize=8, style="italic")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "length_curves.png"), dpi=150)
        plt.close()

    # ==========================================
    # 3. Multiple Sequences (Catastrophic Forgetting)
    # ==========================================
    if "multiple_sequences" in selected:
        _pn, _ps = resolve_probe("multiple_sequences", model_class, n_trials)  # probe protocol
        multi_policy = exposure_policy("multiple_sequences", 300)
        # Interleaved A+B is acquisition-under-load -- a RESULT, so it keeps the
        # ladder even when A and B alone run on the fixed budget.
        multi_policy_result = exposure_policy("multiple_sequences", 300, force_criterion=True)
        print(f"Running Multiple Sequences interference benchmarks... "
              f"(exposure: {multi_policy['mode']})")
        # Replication axis for the serial-position curve: LIST PAIRS, not cue
        # noise. Under a clean single probe each position is a 0/1 hit, so the
        # only honest way to put a probability back on the y-axis is to average
        # over material. Pairs rotate through the categories with >= 7 words,
        # disjoint within a pair (B must not share items with A -- that is the
        # interference being measured); items may recur across pairs. Pair 0 is
        # the canonical fruit->animal pair every scalar metric below refers to,
        # so n_lists=1 reproduces the old section exactly. Seeds would not do
        # this job: hopfield and theta start from W=0 and are seed-invariant.
        n_lists = int(benchmark_arg(benchmark_args, "multiple_sequences", "n_lists", 5))
        _cats = [c for c in ("fruit", "animal", "number", "vehicle", "color")
                 if sum(1 for w in vocab.values() if w == c) >= 7]
        _lists = {c: [w for w, cc in vocab.items() if cc == c][:7] for c in _cats}
        _pairs = [(_cats[i], _cats[(i + 1) % len(_cats)]) for i in range(len(_cats))][:n_lists]
        # canonical pair first, verbatim
        list_A = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']
        list_B = ['cat', 'dog', 'cow', 'horse', 'sheep', 'pig', 'lion']
        _pairs[0] = ("fruit", "animal"); _lists["fruit"], _lists["animal"] = list_A, list_B

        try:
            multi_vocab = {w: vocab[w] for w in list_A + list_B}
            enc_multi = SymbolicEncoder(multi_vocab, embedding_dim=100, category_variance=0.2, seed=42)
            dec_multi = SymbolicDecoder(enc_multi)

            emb_A = enc_multi.encode(list_A)
            emb_B = enc_multi.encode(list_B)

            def _mk(ep, _e=enc_multi):
                kw = _get_model_kwargs(model_kwargs, ep)
                if 'encoder' in model_class.__init__.__code__.co_varnames:
                    return model_class(encoder=_e, n_features=_e.embedding_dim, **kw)
                return model_class(n_features=_e.embedding_dim, **kw)

            def _consolidate(m, emb):
                if not hasattr(m, "consolidate"):
                    return
                import inspect
                if "sequence_data" in inspect.signature(m.consolidate).parameters:
                    m.consolidate(emb, context_data=np.tile(np.array([1.0]), (len(emb), 1)))
                else:
                    m.consolidate()

            def _fit_on(emb):
                def _f(m, ep, _emb=emb):
                    if hasattr(m, "n_epochs"):
                        m.n_epochs = ep
                    m.fit_sequence(_emb, epochs=ep)
                return _f

            def _score_on(words):
                def _s(m, _w=words):
                    m.reset_context()
                    return mean_recall_rate(measure_recall_associative(
                        m, _w, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps))
                return _s

            # 1. Train on list A to criterion. Equalising acquisition BEFORE the
            # interference is the point: a forgetting delta measured after a fixed
            # budget confounds how much A was damaged with how much A was ever
            # learned, and the two arms of that confound move in opposite
            # directions across the taxonomy.
            model, exp_A = train_at_exposure(multi_policy, _mk, _fit_on(emb_A),
                                             _score_on(list_A))
            _consolidate(model, emb_A)
            model_canon_before = model

            # Test A before interference
            model.reset_context()
            curve_A_before = measure_recall_associative(
                model, list_A, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps
            )
            mrr_before = mean_recall_rate(curve_A_before)
            # Deterministic graded companion to the (now binary) clean accuracy.
            _canon_marg_before = measure_recall_margin(model, list_A, enc_multi)
            margin_before = mean_margin(_canon_marg_before)
            recall_before = measure_recall_autoregressive(
                model, list_A, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps
            )
            span_before = memory_span(recall_before)

            # 2. Train on list B to ITS OWN criterion, continuing from the
            # A-trained weights. The staircase rebuilds at each checkpoint, so the
            # A phase is replayed at its settled exposure rather than assumed
            # incremental -- correct for arms whose fit replaces weights.
            def _mk_after_A(_ep, _E=exp_A["epochs"]):
                m = _mk(_E)
                _fit_on(emb_A)(m, _E)
                _consolidate(m, emb_A)
                return m

            model, exp_B = train_at_exposure(multi_policy, _mk_after_A,
                                             _fit_on(emb_B), _score_on(list_B))
            _consolidate(model, emb_B)

            # Test A after interference
            model.reset_context()
            curve_A_after = measure_recall_associative(
                model, list_A, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps
            )
            recall_after = measure_recall_autoregressive(
                model, list_A, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps
            )
            span_after = memory_span(recall_after)
            mrr_after = mean_recall_rate(curve_A_after)
            margin_after = mean_margin(measure_recall_margin(model, list_A, enc_multi))
            delta_mrr = mrr_after - mrr_before

            # 3. INTERLEAVED ingestion condition (added 2026-09-04). The blocked
            # protocol above -- A to criterion, then B to criterion -- is one
            # ingestion order, and for some arms it is the wrong one: an arm
            # that sizes every assembly by how often its item occurs in the
            # stream gives a list presented alone the whole network, and the
            # next list re-labels it (the retired Vieth STDP arm did exactly
            # this). This condition trains A and B as ONE stream, blocks in a
            # seeded random order with each list getting exactly `ep` passes
            # (so per-list exposure matches the blocked condition), to a
            # criterion on the WORSE of the two lists, then reads both. The
            # contrast `mrr_after(blocked) - mrr_A(interleaved)` is the cost of
            # blocking. It is a condition inside the section, not a suite of
            # its own, and the blocked keys above are untouched.
            ms_ingestion = benchmark_arg(benchmark_args, "multiple_sequences",
                                         "ingestion", "both")
            if ms_ingestion not in ("both", "blocked", "interleaved"):
                raise ValueError("multiple_sequences:ingestion must be both|blocked|interleaved")
            if ms_ingestion != "blocked":
                def _fit_interleaved(m, ep):
                    # Fresh generator per call so every staircase checkpoint
                    # sees the same block order for the same exposure.
                    order_rng = np.random.default_rng(42)
                    if hasattr(m, "n_epochs"):
                        m.n_epochs = 1
                    for _ in range(int(ep)):
                        blocks = [emb_A, emb_B]
                        for i in order_rng.permutation(2):
                            m.fit_sequence(blocks[i], epochs=1)

                def _score_both(m):
                    m.reset_context()
                    a = mean_recall_rate(measure_recall_associative(
                        m, list_A, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps))
                    m.reset_context()
                    b = mean_recall_rate(measure_recall_associative(
                        m, list_B, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps))
                    return min(a, b)

                model_i, exp_I = train_at_exposure(multi_policy_result, _mk, _fit_interleaved,
                                                   _score_both)
                _consolidate(model_i, np.vstack([emb_A, emb_B]))
                model_i.reset_context()
                curve_A_inter = measure_recall_associative(
                    model_i, list_A, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps)
                model_i.reset_context()
                curve_B_inter = measure_recall_associative(
                    model_i, list_B, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps)
                mrr_A_inter = mean_recall_rate(curve_A_inter)
                mrr_B_inter = mean_recall_rate(curve_B_inter)
                span_A_inter = memory_span(measure_recall_autoregressive(
                    model_i, list_A, enc_multi, dec_multi, n_trials=_pn, noise_scale=_ps))
                results["metrics"]["multiple_seq_epochs_interleaved"] = float(exp_I["epochs"])
                results["metrics"]["multiple_seq_interleaved_criterion_reached"] = bool(exp_I["reached"])
                results["metrics"]["multiple_seq_interleaved_mrr_A"] = mrr_A_inter
                results["metrics"]["multiple_seq_interleaved_mrr_B"] = mrr_B_inter
                results["metrics"]["multiple_seq_interleaved_span_A"] = int(span_A_inter)
                # Blocked-after minus interleaved: negative = blocking cost A's recall.
                results["metrics"]["multiple_seq_blocking_cost"] = mrr_after - mrr_A_inter
                results["series"]["multiple_sequences_interleaved"] = {
                    "mrr_A": mrr_A_inter, "mrr_B": mrr_B_inter,
                    "span_A": int(span_A_inter),
                    "curve_A": curve_A_inter.tolist(), "curve_B": curve_B_inter.tolist(),
                    "epochs": float(exp_I["epochs"]), "reached": bool(exp_I["reached"]),
                }
                results["metrics"]["multiple_seq_ingestion"] = ms_ingestion
        except Exception as e:
            print(f"Warning: Multiple sequences benchmark failed: {e}")
            mrr_before = 0.0
            mrr_after = 0.0
            delta_mrr = 0.0
            span_before = 0
            span_after = 0
            curve_A_before = np.zeros(len(list_A))
            curve_A_after = np.zeros(len(list_A))

        # ---- Remaining list pairs (pair 0 = the canonical run above) --------
        # Same blocked protocol, same budget policy; each pair gets its own
        # encoder over its own vocabulary, exactly as the canonical pair does.
        # Nothing here touches the canonical scalars or the interleaved condition.
        _pos_hit_before = [np.asarray(curve_A_before, float)]
        _pos_hit_after = [np.asarray(curve_A_after, float)]
        _pos_marg_before = [_canon_marg_before]
        _pos_marg_after = [measure_recall_margin(model, list_A, enc_multi)]
        _pair_records = [{"A": "fruit", "B": "animal", "mrr_before": mrr_before, "mrr_after": mrr_after,
                          "epochs_A": float(exp_A["epochs"]), "epochs_B": float(exp_B["epochs"])}]
        def _run_pair(_wA, _wB, b_mult=1):
            """Blocked A->B on its own encoder. Returns (hit_before, hit_after,
            margin_before, margin_after, exp_A, exp_B, epochs_B_effective, in_span).

            `in_span` is the model-free overlap covariate: the mean fraction of each
            B input's norm lying inside span(A). For a linear associator the damage a
            B step does to A is lr * err_B * (x_B . x_A), so this number -- a property
            of the MATERIAL, computable from the encoder alone -- is what a forgetting
            delta should be read against. `b_mult` scales B's budget (plasticity
            asymmetry); the arm is rebuilt at A's settled exposure and B is then fit
            once for b_mult x its own budget, so replace-style fits are handled the
            same way the staircase handles them.
            """
            _enc = SymbolicEncoder({w: vocab[w] for w in _wA + _wB}, embedding_dim=100,
                                   category_variance=0.2, seed=42)
            _dec = SymbolicDecoder(_enc)
            _eA, _eB = _enc.encode(_wA), _enc.encode(_wB)
            _Q, _ = np.linalg.qr(_eA.T)
            in_span = float(np.mean(np.linalg.norm(_Q.T @ _eB.T, axis=0) / np.linalg.norm(_eB, axis=1)))
            def _mk_p(ep, _e=_enc):
                kw = _get_model_kwargs(model_kwargs, ep)
                if 'encoder' in model_class.__init__.__code__.co_varnames:
                    return model_class(encoder=_e, n_features=_e.embedding_dim, **kw)
                return model_class(n_features=_e.embedding_dim, **kw)
            def _fit_p(emb):
                def _f(m, ep, _emb=emb):
                    if hasattr(m, "n_epochs"): m.n_epochs = ep
                    m.fit_sequence(_emb, epochs=ep)
                return _f
            def _score_p(words, _e=_enc, _d=_dec):
                def _s(m, _w=words):
                    m.reset_context()
                    return mean_recall_rate(measure_recall_associative(
                        m, _w, _e, _d, n_trials=_pn, noise_scale=_ps))
                return _s
            _m, _xA = train_at_exposure(multi_policy, _mk_p, _fit_p(_eA), _score_p(_wA))
            _consolidate(_m, _eA)
            _m.reset_context()
            _cb = measure_recall_associative(_m, _wA, _enc, _dec, n_trials=_pn, noise_scale=_ps)
            _mb = measure_recall_margin(_m, _wA, _enc)
            def _mk_after(_ep, _E=_xA["epochs"], _mk0=_mk_p, _f0=_fit_p(_eA), _e0=_eA):
                mm = _mk0(_E); _f0(mm, _E); _consolidate(mm, _e0); return mm
            _m, _xB = train_at_exposure(multi_policy, _mk_after, _fit_p(_eB), _score_p(_wB))
            _epB_eff = int(_xB["epochs"])
            if int(b_mult) != 1:
                _epB_eff = int(b_mult) * int(_xB["epochs"])
                _m = _mk_after(0); _fit_p(_eB)(_m, _epB_eff)
            _consolidate(_m, _eB)
            _m.reset_context()
            _ca = measure_recall_associative(_m, _wA, _enc, _dec, n_trials=_pn, noise_scale=_ps)
            _ma = measure_recall_margin(_m, _wA, _enc)
            return (np.asarray(_cb, float), np.asarray(_ca, float), _mb, _ma, _xA, _xB, _epB_eff, in_span)

        for (_cA, _cB) in _pairs[1:]:
            try:
                _cb, _ca, _mb, _ma, _xA, _xB, _epB_eff, _isp = _run_pair(_lists[_cA], _lists[_cB])
                _pos_hit_before.append(_cb); _pos_hit_after.append(_ca)
                _pos_marg_before.append(_mb); _pos_marg_after.append(_ma)
                _pair_records.append({"A": _cA, "B": _cB, "mrr_before": mean_recall_rate(_cb),
                                      "mrr_after": mean_recall_rate(_ca), "in_span": _isp,
                                      "epochs_A": float(_xA["epochs"]), "epochs_B": float(_xB["epochs"])})
            except Exception as e:
                print(f"Warning: multiple_sequences pair {_cA}->{_cB} failed: {e}")
        _n_done = len(_pos_hit_before)
        # Per-position probability over lists (the y-axis), and per-position margin.
        prob_before = np.mean(np.vstack(_pos_hit_before), axis=0)
        prob_after = np.mean(np.vstack(_pos_hit_after), axis=0)
        # Position 0 is the cue (NaN by construction); aggregate positions 1: and
        # put the NaN back, so numpy does not warn on an all-NaN column.
        def _agg(rows, fn):
            M = np.vstack(rows)[:, 1:]
            return np.concatenate([[np.nan], fn(M, axis=0)])
        marg_before = _agg(_pos_marg_before, np.nanmean); marg_after = _agg(_pos_marg_after, np.nanmean)
        marg_before_lo = _agg(_pos_marg_before, np.nanmin); marg_before_hi = _agg(_pos_marg_before, np.nanmax)
        marg_after_lo = _agg(_pos_marg_after, np.nanmin); marg_after_hi = _agg(_pos_marg_after, np.nanmax)
        results["metrics"]["multiple_seq_n_lists"] = int(_n_done)
        # All-pairs margin scalars (the plot's right-hand delta). The un-suffixed
        # `multiple_seq_margin_*` / `delta_margin_forgetting` keys remain the
        # canonical fruit->animal pair, for continuity with earlier results files.
        results["metrics"]["multiple_seq_margin_before_lists"] = float(np.nanmean(marg_before[1:]))
        results["metrics"]["multiple_seq_margin_after_lists"] = float(np.nanmean(marg_after[1:]))
        results["metrics"]["delta_margin_forgetting_lists"] = float(np.nanmean(marg_after[1:]) - np.nanmean(marg_before[1:]))
        results["metrics"]["multiple_seq_prob_before"] = float(np.mean(prob_before[1:]))
        results["metrics"]["multiple_seq_prob_after"] = float(np.mean(prob_after[1:]))
        results["metrics"]["delta_prob_forgetting"] = float(np.mean(prob_after[1:]) - np.mean(prob_before[1:]))
        results["series"]["multiple_sequences_lists"] = {
            "n_lists": int(_n_done), "pairs": _pair_records,
            "prob_before": prob_before.tolist(), "prob_after": prob_after.tolist(),
            "margin_before": marg_before.tolist(), "margin_after": marg_after.tolist(),
        }

        # ---- OVERLAP condition (added 2026-09-06) ---------------------------
        # The canonical pair is fruit->animal: near-orthogonal in a 100-d random
        # embedding (in_span ~0.23), so a linear associator's B updates land in
        # weight-space A never used and P(A) stays at ceiling (shown by the
        # retired standalone_overlap_demo.py, at tag archive/pre-cleanup).
        # This block holds everything fixed except the
        # DIRECTION of B relative to A: B drawn from A's own category (in_span
        # ~0.37) against a disjoint-category control, both at L=6 so the two
        # conditions share list length (only fruit and animal have 12 words).
        # `b_exposure_multiplier` (default 1) is the plasticity-asymmetry knob;
        # the demo needed 10x to move accuracy, and margin moves well before that.
        ms_overlap = benchmark_arg(benchmark_args, "multiple_sequences", "overlap", "both")
        ms_bmult = int(benchmark_arg(benchmark_args, "multiple_sequences", "b_exposure_multiplier", 1))
        if ms_overlap not in ("off", "both"):
            raise ValueError("multiple_sequences:overlap must be off|both")
        if ms_overlap == "both":
            _ov_lists = {c: [w for w, cc in vocab.items() if cc == c] for c in ("fruit", "animal", "number")}
            _ov_pairs = {
                "disjoint":      [(_ov_lists["fruit"][:6], _ov_lists["animal"][:6]),
                                  (_ov_lists["animal"][:6], _ov_lists["number"][:6])],
                "same_category": [(_ov_lists["fruit"][:6], _ov_lists["fruit"][6:12]),
                                  (_ov_lists["animal"][:6], _ov_lists["animal"][6:12])],
            }
            _ov = {}
            for _cond, _pp in _ov_pairs.items():
                _hb, _ha, _gb, _ga, _isp, _rec = [], [], [], [], [], []
                for (_wA, _wB) in _pp:
                    try:
                        _cb, _ca, _mb, _ma, _xA, _xB, _epB_eff, _s_ = _run_pair(_wA, _wB, b_mult=ms_bmult)
                        _hb.append(_cb); _ha.append(_ca); _gb.append(_mb); _ga.append(_ma); _isp.append(_s_)
                        _rec.append({"A": _wA, "B": _wB, "in_span": _s_, "mrr_before": mean_recall_rate(_cb),
                                     "mrr_after": mean_recall_rate(_ca), "epochs_A": float(_xA["epochs"]),
                                     "epochs_B": float(_xB["epochs"]), "epochs_B_effective": int(_epB_eff)})
                    except Exception as e:
                        print(f"Warning: multiple_sequences overlap pair {_wA[0]}->{_wB[0]} failed: {e}")
                if not _hb:
                    continue
                _pb, _pa = np.mean(np.vstack(_hb), axis=0), np.mean(np.vstack(_ha), axis=0)
                _gmb, _gma = _agg(_gb, np.nanmean), _agg(_ga, np.nanmean)
                _ov[_cond] = dict(in_span=float(np.mean(_isp)), prob_before=_pb, prob_after=_pa,
                                  margin_before=_gmb, margin_after=_gma, pairs=_rec)
                for _k, _v in (("in_span", float(np.mean(_isp))),
                               ("prob_before", float(np.mean(_pb[1:]))), ("prob_after", float(np.mean(_pa[1:]))),
                               ("delta_prob", float(np.mean(_pa[1:]) - np.mean(_pb[1:]))),
                               ("margin_before", float(np.nanmean(_gmb[1:]))), ("margin_after", float(np.nanmean(_gma[1:]))),
                               ("delta_margin", float(np.nanmean(_gma[1:]) - np.nanmean(_gmb[1:])))):
                    results["metrics"][f"multiple_seq_overlap_{_cond}_{_k}"] = _v
            if "disjoint" in _ov and "same_category" in _ov:
                # The overlap cost: extra forgetting attributable to shared input
                # directions alone (same length, same budgets, same variance).
                results["metrics"]["multiple_seq_overlap_cost_prob"] = (
                    results["metrics"]["multiple_seq_overlap_same_category_delta_prob"]
                    - results["metrics"]["multiple_seq_overlap_disjoint_delta_prob"])
                results["metrics"]["multiple_seq_overlap_cost_margin"] = (
                    results["metrics"]["multiple_seq_overlap_same_category_delta_margin"]
                    - results["metrics"]["multiple_seq_overlap_disjoint_delta_margin"])
            results["metrics"]["multiple_seq_overlap_b_exposure_multiplier"] = ms_bmult
            results["series"]["multiple_sequences_overlap"] = {
                _c: {"in_span": _d["in_span"], "pairs": _d["pairs"],
                     "prob_before": _d["prob_before"].tolist(), "prob_after": _d["prob_after"].tolist(),
                     "margin_before": _d["margin_before"].tolist(), "margin_after": _d["margin_after"].tolist()}
                for _c, _d in _ov.items()}
            results["series"]["multiple_sequences_overlap"]["b_exposure_multiplier"] = ms_bmult

            # Figure: the two conditions side by side, before dashed / after solid.
            fig, (oL, oR) = plt.subplots(1, 2, figsize=(12, 5))
            _col = {"disjoint": "tab:blue", "same_category": "tab:red"}
            for _c, _d in _ov.items():
                _x = np.arange(len(_d["prob_after"]))
                oL.plot(_x, _d["prob_before"], "--", color=_col[_c], alpha=0.6)
                oL.plot(_x, _d["prob_after"], "-o", color=_col[_c], label=f'{_c} (in-span {_d["in_span"]:.2f}): after B')
                oR.plot(_x, _d["margin_before"], "--", color=_col[_c], alpha=0.6)
                oR.plot(_x, _d["margin_after"], "-o", color=_col[_c], label=f'{_c}: after B')
            oL.set_ylim(-0.05, 1.05); oL.set_xlabel("Serial Position"); oL.set_ylabel("P(A recalled | clean cue) over pairs")
            oL.set_title(f"Overlap condition, L=6, B budget x{ms_bmult}  (dashed = before B)\n"
                         f"overlap cost, prob: {results['metrics'].get('multiple_seq_overlap_cost_prob', float('nan')):+.2f}", fontsize=10)
            oL.legend(fontsize=8, loc="lower left")
            oR.axhline(0, color="k", ls=":", lw=0.8); oR.set_xlabel("Serial Position"); oR.set_ylabel("margin(A) = cos(pred,target) - best competitor")
            _L = max(len(_d["prob_after"]) for _d in _ov.values()); oL.set_xticks(range(_L)); oR.set_xticks(range(1, _L))
            oR.plot([], [], "--", color="gray", label="dashed = before B")
            oR.set_title(f"overlap cost, margin: {results['metrics'].get('multiple_seq_overlap_cost_margin', float('nan')):+.3f}", fontsize=10)
            oR.legend(fontsize=8)
            fig.text(0.5, 0.005, "same encoder, variance, length and budgets in both conditions; only B's direction relative to A differs "
                                 "(in_span = fraction of B inside span(A), model-free)", ha="center", fontsize=8, style="italic")
            plt.tight_layout(rect=(0, 0.03, 1, 1))
            plt.savefig(os.path.join(plots_dir, "multiple_seq_overlap.png"), dpi=150)
            plt.close()

        # Exposure per phase, so a reader can see that A and B were each taken to
        # the same functional point before the delta was read.
        try:
            results["metrics"]["multiple_seq_epochs_A"] = float(exp_A["epochs"])
            results["metrics"]["multiple_seq_epochs_B"] = float(exp_B["epochs"])
            results["metrics"]["multiple_seq_criterion_reached"] = bool(
                exp_A["reached"] and exp_B["reached"])
            results["metrics"]["multiple_seq_exposure_mode"] = multi_policy["mode"]
            results["metrics"]["multiple_seq_exposure_source"] = multi_policy.get("source")
        except NameError:
            results["metrics"]["multiple_seq_criterion_reached"] = False
        results["metrics"]["multiple_seq_mrr_before"] = mrr_before
        results["metrics"]["multiple_seq_mrr_after"] = mrr_after
        results["metrics"]["delta_mrr_forgetting"] = delta_mrr
        # Margin resolves forgetting below the accuracy floor: two arms at MRR 0
        # after list B can still differ in how far list A's targets fell.
        results["metrics"]["multiple_seq_margin_before"] = margin_before
        results["metrics"]["multiple_seq_margin_after"] = margin_after
        results["metrics"]["delta_margin_forgetting"] = margin_after - margin_before

        results["series"]["multiple_sequences"] = {
            "mrr_before": mrr_before,
            "mrr_after": mrr_after,
            "delta_mrr": delta_mrr,
            "span_before": span_before,
            "span_after": span_after,
            "delta_span": span_after - span_before,
            "curve_before": curve_A_before.tolist(),
            "curve_after": curve_A_after.tolist()
        }

        # Save Multiple Sequences Plot -- two panels.
        # Left: per-position recall PROBABILITY, i.e. the fraction of list pairs
        # on which that position was recalled from a clean cue (n_lists=1 makes
        # this a 0/1 hit indicator, and the label says so). Right: per-position
        # MARGIN (target cosine minus best competitor), mean over pairs with the
        # min-max band -- the graded companion that resolves forgetting the
        # probability axis cannot see once an arm is at ceiling.
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
        _xs = np.arange(len(prob_before))
        axL.plot(_xs, prob_before, '-o', color='blue', label=f'Before List B (mean {np.mean(prob_before[1:]):.2f})')
        axL.plot(_xs, prob_after, '-x', color='red', label=f'After List B (mean {np.mean(prob_after[1:]):.2f})')
        _ms_i = results["series"].get("multiple_sequences_interleaved")
        if _ms_i:
            axL.plot(_ms_i["curve_A"], '--s', color='green', alpha=0.7,
                     label=f'A interleaved with B, canonical pair (MRR {_ms_i["mrr_A"]:.2f})')
        axL.set_xlabel('Serial Position')
        axL.set_ylabel(f'P(recalled | clean cue) over {_n_done} list pairs' if _n_done > 1
                       else 'recalled from clean cue (0/1), single list pair')
        axL.set_title(f'Catastrophic Forgetting  (delta P: {np.mean(prob_after[1:]) - np.mean(prob_before[1:]):+.2f}; '
                      f'canonical delta MRR: {delta_mrr:+.2f})', fontsize=10)
        axL.set_ylim(-0.05, 1.05); axL.legend(fontsize=8, loc='center left')
        axR.plot(_xs, marg_before, '-o', color='blue', label='Before List B')
        axR.fill_between(_xs, marg_before_lo, marg_before_hi, color='blue', alpha=0.12)
        axR.plot(_xs, marg_after, '-x', color='red', label='After List B')
        axR.fill_between(_xs, marg_after_lo, marg_after_hi, color='red', alpha=0.12)
        axR.axhline(0.0, color='k', lw=0.8, ls=':')
        axR.set_xlabel('Serial Position'); axR.set_ylabel('margin = cos(pred, target) - best competitor')
        axR.set_title(f'Per-position margin, mean over {_n_done} pairs (band = min..max)  '
                      f'delta_margin_forgetting_lists: {np.nanmean(marg_after[1:]) - np.nanmean(marg_before[1:]):+.3f}', fontsize=10)
        axR.legend(fontsize=8)
        fig.text(0.5, 0.005, "clean single probe; replication over LIST PAIRS, not cue noise (docs/sections/probe_protocol.md)",
                 ha="center", fontsize=8, style="italic")
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        plt.savefig(os.path.join(plots_dir, "multiple_seq_forgetting.png"), dpi=150)
        plt.close()

    # ==========================================
    # 3b. Continual chain (retention matrix)
    # ==========================================
    # The load axis and the plasticity axis, from one N-task chain.
    # `multiple_sequences` above is this section's T=2 corner: one interposed
    # list, one number, and no way to see either the decay gradient or the
    # diagonal. See memval/benchmarks/continual_chain.py for why the diagonal is
    # the half that matters -- every other stability read-out in the suite is
    # maximised by an arm that simply stops learning.
    if "continual_chain" in selected:
        _pn, _ps = resolve_probe("continual_chain", model_class, n_trials)  # probe protocol
        from memval.benchmarks.continual_chain import ContinualChainBenchmark

        chain_epochs = section_epochs("continual_chain", 300)
        n_chain_tasks = int(benchmark_arg(benchmark_args, "continual_chain",
                                          "n_tasks", 6))
        chain_seq_len = int(benchmark_arg(benchmark_args, "continual_chain",
                                          "seq_len", 5))
        chain_rehearse = bool(int(benchmark_arg(benchmark_args, "continual_chain",
                                                "rehearse", 1)))
        n_rehearsed = benchmark_arg(benchmark_args, "continual_chain",
                                    "n_rehearsed", None)
        print(f"Running Continual Chain... (tasks={n_chain_tasks}x{chain_seq_len}, "
              f"epochs={chain_epochs}, rehearse={chain_rehearse})")

        try:
            # One category per task, so tasks are near-orthogonal across and
            # similar within: the load axis is isolated from the overlap axis,
            # which `semantic_similarity` and `tmaze_disambiguation` own.
            # Both overlap dials are exposed as section args so the chain can
            # be run at a chosen geometry; the defaults reproduce every saved
            # run (sigma 0.2, quasi-orthogonal categories).
            from memval.benchmarks.continual_chain import build_chain_material
            chain_sigma = float(benchmark_arg(benchmark_args, "continual_chain",
                                              "category_variance", 0.2))
            _rb = benchmark_arg(benchmark_args, "continual_chain",
                                "between_category_cosine", None)
            chain_rho_b = (None if _rb is None or str(_rb).strip().lower() in ("", "none")
                           else float(_rb))
            material = build_chain_material(
                vocab, n_chain_tasks, chain_seq_len, embedding_dim=100,
                category_variance=chain_sigma, between_category_cosine=chain_rho_b,
                seed=42)
            chain_sequences = material["sequences"]
            chain_vocab = material["vocab"]
            enc_chain = material["encoder"]
            dec_chain = material["decoder"]
            chain_embeddings = material["embeddings"]

            chain_kwargs = _get_model_kwargs(model_kwargs, chain_epochs)
            if 'encoder' in model_class.__init__.__code__.co_varnames:
                chain_model = model_class(encoder=enc_chain,
                                          n_features=enc_chain.embedding_dim,
                                          **chain_kwargs)
            else:
                chain_model = model_class(n_features=enc_chain.embedding_dim,
                                          **chain_kwargs)

            chain_names = list(chain_sequences.keys())

            def _score_chain_task(i: int) -> float:
                chain_model.reset_context()
                return mean_recall_rate(measure_recall_associative(
                    chain_model, chain_sequences[chain_names[i]],
                    enc_chain, dec_chain, n_trials=_pn, noise_scale=_ps))

            # Chance for a decode over the chain's own vocabulary. Only the
            # ratio-based statistics use it; ACC, forgetting and intransigence
            # are differences and stay on the raw scale.
            chain_chance = 1.0 / len(chain_vocab)

            chain_out = ContinualChainBenchmark(
                n_trials=_pn, noise_scale=_ps,
                rehearse=chain_rehearse,
                n_rehearsed=None if n_rehearsed is None else int(n_rehearsed),
                # Staircase ceiling, `--benchmark-args continual_chain:max_epochs=N`.
                # Default unchanged (512); only an arm that would have run past N
                # is affected, and it already reports `chain_acquired` False.
                max_epochs=int(benchmark_arg(benchmark_args, "continual_chain",
                                             "max_epochs", 512)),
            ).evaluate(
                model=chain_model,
                datasets={
                    "sequences": chain_sequences,
                    "embeddings": chain_embeddings,
                    "score_fn": _score_chain_task,
                },
                epochs=chain_epochs,
                chance_level=chain_chance,
            )
            results["metrics"].update(chain_out["metrics"])
            results["series"]["continual_chain"] = chain_out["series"]
            # The stimulus geometry the matrix was measured on, beside the laws.
            results["series"]["continual_chain"]["material"] = material["geometry"]

            # --- plots ---
            from memval.metrics.retention import plot_retention_matrix
            R_chain = np.array(chain_out["series"]["retention_matrix"], dtype=float)
            plot_retention_matrix(
                R_chain,
                os.path.join(plots_dir, "continual_chain_retention.png"),
                title=f"Retention matrix ({model_name})",
                value_label="cued recall (mean recall rate)",
                task_labels=chain_names,
                chance_level=chain_chance,
            )

            # Three axes from the stored series (replottable with
            # bin/replot_continual_chain.py): load curve, diagonal vs final row,
            # and epochs-to-criterion per chain position -- the exposure x
            # retention pairing One-shot learning promises.
            from memval.benchmarks.continual_chain import plot_chain_axes
            plot_chain_axes(
                chain_out["series"], chain_out["metrics"],
                os.path.join(plots_dir, "continual_chain_axes.png"),
                baseline_epochs=results["metrics"].get("convergence_epochs"),
            )

            sel = chain_out["series"].get("selective_retention")
            if sel:
                fig, ax = plt.subplots(figsize=(7, 4.2))
                colors = ["tab:green" if i in sel["rehearsed"] else "tab:grey"
                          for i in range(len(sel["delta"]))]
                ax.bar(range(len(sel["delta"])), sel["delta"], color=colors)
                ax.set_xticks(range(len(sel["task_labels"])))
                ax.set_xticklabels(sel["task_labels"], rotation=45, ha="right", fontsize=8)
                ax.axhline(0.0, color="black", lw=0.8)
                ax.set_ylabel("Change in recall after rehearsal")
                ax.set_title(
                    "Selective retention: green = still relevant (re-presented)\n"
                    f"selectivity={chain_out['metrics']['select_selectivity']:+.3f}  "
                    f"under_pressure={chain_out['metrics']['select_under_pressure']}")
                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, "continual_chain_selectivity.png"),
                            dpi=150)
                plt.close(fig)

        except Exception as e:
            print(f"Warning: Continual chain benchmark failed: {e}")
            results["series"]["continual_chain"] = {}

    # ==========================================
    # 3c. Paired associate (AB/AC shared-cue overwrite)
    # ==========================================
    # `multiple_sequences` is the disjoint-cue control of this section, run
    # without its experimental arm. Here the same cues are re-paired with new
    # targets, so the old response becomes an error and forgetting is scored as
    # success -- the symbolic twin of tmaze_reversal, in the modality where the
    # rest of the interference measurement lives.
    if "paired_associate" in selected:
        _pn, _ps = resolve_probe("paired_associate", model_class, n_trials)  # probe protocol
        from memval.benchmarks.paired_associate import PairedAssociateBenchmark

        pa_epochs = section_epochs("paired_associate", 100)
        pa_n_pairs = int(benchmark_arg(benchmark_args, "paired_associate", "n_pairs", 5))
        pa_ab_trials = int(benchmark_arg(benchmark_args, "paired_associate",
                                         "ab_trials", 3))
        pa_p2_trials = int(benchmark_arg(benchmark_args, "paired_associate",
                                         "phase2_trials", 6))
        print(f"Running Paired Associate AB/AC... (pairs={pa_n_pairs}, "
              f"AB trials={pa_ab_trials}, phase-2 trials={pa_p2_trials}, "
              f"epochs={pa_epochs})")

        try:
            # Five disjoint roles, five disjoint categories: cues and targets do
            # not share category structure, so any competition measured is
            # competition for the cue and not residual semantic overlap.
            by_cat: Dict[str, List[str]] = {}
            for word, cat in vocab.items():
                by_cat.setdefault(cat, []).append(word)
            roles = ["fruit", "animal", "vehicle", "color", "number"]
            missing = [r for r in roles
                       if len(by_cat.get(r, [])) < pa_n_pairs]
            if missing:
                raise ValueError(f"categories too small for {pa_n_pairs} pairs: {missing}")
            cues_A, targets_B, targets_C, cues_D, targets_E = (
                sorted(by_cat[r])[:pa_n_pairs] for r in roles)

            pa_words = cues_A + targets_B + targets_C + cues_D + targets_E
            pa_vocab = {w: vocab[w] for w in pa_words}
            enc_pa = SymbolicEncoder(pa_vocab, embedding_dim=100,
                                     category_variance=0.2, seed=42)
            dec_pa = SymbolicDecoder(enc_pa)

            pa_datasets = {
                "cues_A": cues_A, "targets_B": targets_B, "targets_C": targets_C,
                "cues_D": cues_D, "targets_E": targets_E,
                "encoder": enc_pa, "decoder": dec_pa,
            }

            pa_kwargs = _get_model_kwargs(model_kwargs, pa_epochs)
            bench_pa = PairedAssociateBenchmark(n_trials=_pn, noise_scale=_ps)
            pa_series: Dict[str, Any] = {}
            for cond in ("abac", "control"):
                # Each condition is an independent run from a fresh model: the
                # two phase 2s are mutually contaminating.
                if 'encoder' in model_class.__init__.__code__.co_varnames:
                    pa_model = model_class(encoder=enc_pa,
                                           n_features=enc_pa.embedding_dim,
                                           **pa_kwargs)
                else:
                    pa_model = model_class(n_features=enc_pa.embedding_dim,
                                           **pa_kwargs)
                out = bench_pa.evaluate(
                    model=pa_model, datasets=pa_datasets, condition=cond,
                    ab_trials=pa_ab_trials, phase2_trials=pa_p2_trials,
                    epochs=pa_epochs,
                )
                results["metrics"].update(out["metrics"])
                pa_series[cond] = out["series"]

            # The contrast the section exists for: damage under a shared cue,
            # over and above damage from an equal quantity of disjoint-cue
            # learning. Emitted here rather than derived downstream because both
            # halves are only ever produced together.
            ret_abac = results["metrics"].get("pa_abac_ab_retention")
            ret_ctrl = results["metrics"].get("pa_control_ab_retention")
            if (ret_abac is not None and ret_ctrl is not None
                    and not np.isnan(ret_abac) and not np.isnan(ret_ctrl)):
                results["metrics"]["pa_cue_competition_cost"] = float(ret_ctrl - ret_abac)
            results["series"]["paired_associate"] = pa_series

            # --- plot ---
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
            for ax, cond in zip(axes, ("abac", "control")):
                s = pa_series[cond]
                ax.plot(s["trials"], s["old_rate"], '-o', color="tab:red",
                        label="old target (B)")
                ax.plot(s["trials"], s["new_rate"], '-o', color="tab:green",
                        label="new target (C)")
                ax.plot(s["trials"], s["other_rate"], '-o', color="tab:grey",
                        label="neither")
                ax.axvline(0, ls=":", color="black", lw=0.8)
                ax.set_xlabel("Phase-2 trials")
                ax.set_ylim(-0.05, 1.05)
                ax.set_title("AB/AC (shared cue)" if cond == "abac"
                             else "AB/DE control (disjoint cue)")
                ax.legend(fontsize=8)
            axes[0].set_ylabel("Response rate to cue A")
            cost = results["metrics"].get("pa_cue_competition_cost")
            fig.suptitle("Paired-associate interference"
                         + (f"  —  cue-competition cost = {cost:+.2f}"
                            if cost is not None else ""))
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "paired_associate_abac.png"), dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Paired associate benchmark failed: {e}")
            results["series"]["paired_associate"] = {}

    # ==========================================
    # 4. Noise Invariance
    # ==========================================
    if "noise_invariance" in selected:
        # Train one model on the fruit list for the retrieval-cue noise sweep.
        # enc_noise stays None if setup fails, which the sweep checks before
        # using the model.
        enc_noise = None
        noise_policy = exposure_policy("noise_invariance", 300)
        try:
            fruit_vocab = {w: c for w, c in vocab.items() if c == "fruit"}
            enc_noise = SymbolicEncoder(fruit_vocab, embedding_dim=100, category_variance=0.2, seed=42)
            dec_noise = SymbolicDecoder(enc_noise)

            word_list_encoded = enc_noise.encode(fruit_words)

            def _mk_noise(ep, _e=enc_noise):
                kw = _get_model_kwargs(model_kwargs, ep)
                if 'encoder' in model_class.__init__.__code__.co_varnames:
                    return model_class(encoder=_e, n_features=_e.embedding_dim, **kw)
                return model_class(n_features=_e.embedding_dim, **kw)

            def _fit_noise(m, ep, _x=word_list_encoded):
                if hasattr(m, "n_epochs"):
                    m.n_epochs = ep
                m.fit_sequence(_x, epochs=ep)

            # The criterion is read at sigma = 0. The sweep manipulates the CUE,
            # so training must be held at a fixed functional point or a shallow
            # tolerance curve cannot be told from an undertrained model.
            def _score_noise(m):
                m.reset_context()
                return mean_recall_rate(measure_recall_associative(
                    m, fruit_words, enc_noise, dec_noise, n_trials=n_trials,
                    noise_scale=0.0))

            model, noise_exp = train_at_exposure(noise_policy, _mk_noise,
                                                 _fit_noise, _score_noise)
        except Exception as e:
            print(f"Warning: Noise-invariance model setup failed: {e}")
            enc_noise = None

    if "noise_invariance" in selected:
        print(f"Running Noise Invariance benchmarks... "
              f"(exposure: {noise_policy['mode']})")
        noise_levels = np.linspace(0.0, 1.0, 11)
        noise_mrr = []

        try:
            if enc_noise is None:
                raise RuntimeError("shared noise-invariance model unavailable")
            for sigma in noise_levels:
                # Seeded per level (2026-09-06): the probe noise used to come from
                # wherever the global stream happened to be, so a section-subset
                # run and a full-suite run disagreed by up to one grid step.
                np.random.seed(int(2000 + round(sigma * 100)))
                model.reset_context()
                curve = measure_recall_associative(
                    model, fruit_words, enc_noise, dec_noise, n_trials=n_trials, noise_scale=sigma
                )
                noise_mrr.append(mean_recall_rate(curve))

            # Extract tolerance threshold (sigma where MRR first drops below 0.5)
            tolerance_thresh = 1.0
            for sigma, mrr in zip(noise_levels, noise_mrr):
                if mrr < 0.5:
                    tolerance_thresh = float(sigma)
                    break
        except Exception as e:
            print(f"Warning: Noise sweep failed: {e}")
            noise_mrr = [0.0] * len(noise_levels)
            tolerance_thresh = 0.0

        # The corruption half of P17's second probe: autoregressive rollout
        # with the noise on the INITIAL cue only, every later step fed the
        # model's own prediction, verbatim ("raw" feedback): no projection
        # back onto the encoder's unit-norm manifold, so what is measured is
        # the arm's own free-running dynamics from a corrupted start.
        #
        # One feedback mode, deliberately (2026-09-26). "quantized" was dropped
        # first: snapping every step to a clean codebook item makes each step a
        # clean cued probe, so it restated the cued sweep. "l2" was dropped
        # next: across the six zoo arms it coincides with raw exactly for AHN,
        # theta, tPC and Chen (any positively-homogeneous arm: relu(W(cx)) =
        # c relu(Wx) under a cosine decode) and moves DTS-ESN (0.37 vs 0.31)
        # and EP (0.15 vs 0.16) in opposite directions -- a harness
        # intervention, not a property the section is asking about.
        #
        # Diagnostic, not scored: the LEVEL is set by free-running ability,
        # which Serial order / `unrolling` already scores; what this adds is
        # the shape against sigma.
        noise_rollout: List[float] = []
        # Margin under the same rollout: at the corrupted first step and at the
        # later, self-driven steps separately, so the sweep shows whether the
        # corruption degrades the margin and whether that degradation PROPAGATES
        # or the chain recovers once it is on its own output.
        noise_rollout_margin: Dict[str, List[float]] = {"first": [], "later": [], "mean": []}
        try:
            if enc_noise is None:
                raise RuntimeError("shared noise-invariance model unavailable")
            for sigma in noise_levels:
                np.random.seed(int(1000 + round(sigma * 100)))
                rec, mar = rollout_recall_and_margin(
                    model, fruit_words, enc_noise, n_trials=n_trials,
                    noise_scale=sigma, feedback_mode="raw")
                noise_rollout.append(float(np.mean(rec)))
                noise_rollout_margin["first"].append(float(mar[0]))
                noise_rollout_margin["later"].append(
                    float(np.mean(mar[1:])) if len(mar) > 1 else float("nan"))
                noise_rollout_margin["mean"].append(float(np.mean(mar)))
        except Exception as e:
            print(f"Warning: Noise rollout sweep failed: {e}")
            noise_rollout = [float("nan")] * len(noise_levels)
            noise_rollout_margin = {k: [float("nan")] * len(noise_levels)
                                    for k in ("first", "later", "mean")}
        rollout_thresh = 1.0
        for sigma, r in zip(noise_levels, noise_rollout):
            if not (r >= 0.5):
                rollout_thresh = float(sigma)
                break

        try:
            results["metrics"]["noise_epochs_to_criterion"] = float(noise_exp["epochs"])
            results["metrics"]["noise_criterion_reached"] = bool(noise_exp["reached"])
        except NameError:
            results["metrics"]["noise_criterion_reached"] = False
        results["metrics"]["noise_tolerance_threshold"] = tolerance_thresh
        results["metrics"]["noise_rollout_auc"] = float(np.nanmean(noise_rollout))
        results["metrics"]["noise_rollout_tolerance"] = rollout_thresh
        results["metrics"]["noise_completion_gap"] = float(
            np.mean(noise_mrr) - np.nanmean(noise_rollout))
        _mf, _ml = np.array(noise_rollout_margin["first"]), np.array(noise_rollout_margin["later"])
        results["metrics"]["noise_rollout_margin_auc"] = float(np.nanmean(noise_rollout_margin["mean"]))
        results["metrics"]["noise_rollout_margin_first_auc"] = float(np.nanmean(_mf))
        results["metrics"]["noise_rollout_margin_later_auc"] = float(np.nanmean(_ml))
        # Slope of the first-step margin against sigma: how fast corruption eats
        # the decision at the corrupted step. Propagation = later-step slope.
        _ok = np.isfinite(_mf) & np.isfinite(_ml)
        results["metrics"]["noise_rollout_margin_first_slope"] = (
            float(np.polyfit(noise_levels[_ok], _mf[_ok], 1)[0]) if _ok.sum() > 1 else float("nan"))
        results["metrics"]["noise_rollout_margin_later_slope"] = (
            float(np.polyfit(noise_levels[_ok], _ml[_ok], 1)[0]) if _ok.sum() > 1 else float("nan"))
        results["series"]["noise_sweep"] = {
            "noise_levels": noise_levels.tolist(),
            "mrr": noise_mrr,
            "rollout_raw": noise_rollout,
            "rollout_margin_first": noise_rollout_margin["first"],
            "rollout_margin_later": noise_rollout_margin["later"],
            "rollout_margin_mean": noise_rollout_margin["mean"],
        }

        # Save Noise Invariance Plot: cued (left) and rollout from a noisy
        # initial cue (right), on the same sigma axis.
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
        ax = axes[0]
        ax.plot(noise_levels, noise_mrr, '-o', color='purple', label='cued recall')
        ax.axhline(0.5, linestyle='--', color='red', alpha=0.6, label='threshold (0.5)')
        ax.axvline(tolerance_thresh, linestyle=':', color='black', alpha=0.8,
                   label=f'tolerance limit: {tolerance_thresh:.2f}')
        ax.set_xlabel('cue noise sigma (every cue)')
        ax.set_ylabel('cued recall (MRR)')
        ax.set_title('Completion from a corrupted cue')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax = axes[1]
        ax.plot(noise_levels, noise_mrr, '-', color='purple', alpha=0.25, lw=1,
                label='cued (left panel, for reference)')
        ax.plot(noise_levels, noise_rollout, '-^', color='tab:red',
                label=f'rollout, {rollout_owner_label("raw")}')
        ax.axhline(0.5, linestyle='--', color='red', alpha=0.6)
        ax.axvline(rollout_thresh, linestyle=':', color='black', alpha=0.8,
                   label=f'rollout tolerance: {rollout_thresh:.2f}')
        ax.set_xlabel('noise sigma on the INITIAL cue only')
        ax.set_ylabel('autoregressive recall')
        ax.set_title(f'Rollout from a corrupted initial cue\n'
                     f'(propagation; gap vs cued = {results["metrics"]["noise_completion_gap"]:+.2f})',
                     fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7.5, loc='upper right')
        ax = axes[2]
        ax.plot(noise_levels, noise_rollout_margin["first"], '-^', color='tab:red',
                label='step 1: the corrupted cue\'s own prediction')
        ax.plot(noise_levels, noise_rollout_margin["later"], '--s', color='tab:orange',
                label='steps 2+: driven by the model\'s own output')
        ax.axhline(0.0, color='black', lw=0.8)
        ax.set_xlabel('noise sigma on the INITIAL cue only')
        ax.set_ylabel('margin: cos(target) − best competitor')
        _s1 = results["metrics"]["noise_rollout_margin_first_slope"]
        _s2 = results["metrics"]["noise_rollout_margin_later_slope"]
        ax.set_title(f'Rollout margin under corruption\n'
                     f'(slope vs sigma: step 1 {_s1:+.2f}, later steps {_s2:+.2f})',
                     fontsize=10)
        ax.legend(fontsize=7.5, loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "noise_invariance.png"), dpi=150)
        plt.close(fig)

    # ==========================================
    # 5. Semantic Similarity
    # ==========================================
    if "semantic_similarity" in selected:
        _pn, _ps = resolve_probe("semantic_similarity", model_class, n_trials)  # probe protocol
        sem_policy = exposure_policy("semantic_similarity", 300)
        # Probe noise is drawn from the LEGACY GLOBAL numpy stream inside
        # measure_recall_associative, so a section is only reproducible if it
        # pins that stream -- the same thing cue_masking, paired_associate,
        # continual_chain and schema_consistency already do. Seeded per rung
        # below rather than once here, so that adding or reordering a variance
        # rung cannot shift the draws every other rung sees.
        #
        # This section needed it more than most. Its scored read-out is
        # epochs-to-criterion, which is a STEP function of a noisy score: at the
        # hardest rung the criterion crossing sits right where the noise lives,
        # so a draw that would move mean MRR by a hundredth moves the reported
        # exposure by tens of epochs. Combined with the vocabulary-ordering bug
        # fixed above, seven runs of the same rule spanned 33-64.
        sem_seed = int(benchmark_arg(benchmark_args, "semantic_similarity",
                                     "seed", 42))
        print(f"Running Semantic Similarity benchmarks... "
              f"(exposure: {sem_policy['mode']}, seed={sem_seed})")

        def _sem_train(words, enc_s, dec_s):
            """Train one list to criterion and return (model, exposure record).

            Under high overlap an arm may never reach criterion; that censoring is
            the result, so it is returned rather than silently absorbed.
            """
            emb = enc_s.encode(words)

            def _mk(ep, _e=enc_s):
                kw = _get_model_kwargs(model_kwargs, ep)
                if 'encoder' in model_class.__init__.__code__.co_varnames:
                    return model_class(encoder=_e, n_features=_e.embedding_dim, **kw)
                return model_class(n_features=_e.embedding_dim, **kw)

            def _fit(m, ep, _x=emb):
                if hasattr(m, "n_epochs"):
                    m.n_epochs = ep
                m.fit_sequence(_x, epochs=ep)

            def _score(m, _w=words, _e=enc_s, _d=dec_s):
                m.reset_context()
                return mean_recall_rate(measure_recall_associative(
                    m, _w, _e, _d, n_trials=_pn, noise_scale=_ps))

            return train_at_exposure(sem_policy, _mk, _fit, _score)

        # 5a. Categorical Comparison
        list_high = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum'] # High Similarity
        list_low = ['apple', 'cat', 'car', 'hammer', 'red', 'one', 'blue'] # Low Similarity

        try:
            # High Similarity list
            # dict.fromkeys, NOT set(): SymbolicEncoder draws one RNG sample
            # per word in the mapping's ITERATION order, so an order-unstable
            # vocabulary hands each word a different vector on every process
            # even though the encoder itself is seeded. `set()` iterates in
            # PYTHONHASHSEED order, which is randomised per process, and that
            # made this section the only non-reproducible one in the suite:
            # seven runs of the SAME learning rule gave epochs-to-criterion of
            # 33/37/41/45/46/47/64 at the hardest rung (1.9x spread) with the
            # rung's own cosine moving 0.787-0.798. `similarity_exposure_cost`
            # feeds item_similarity, which is weight 0.40 of Sequence
            # disambiguation, so the spread was larger than most real effects.
            # dict.fromkeys dedups while preserving first-seen order.
            sim_vocab = {w: vocab[w] for w in dict.fromkeys(list_high + list_low)}
            enc_sim = SymbolicEncoder(sim_vocab, embedding_dim=100, category_variance=0.2, seed=42)
            dec_sim = SymbolicDecoder(enc_sim)

            np.random.seed(sem_seed)
            model_high, sem_exp_high = _sem_train(list_high, enc_sim, dec_sim)
            model_high.reset_context()
            np.random.seed(sem_seed)
            curve_high = measure_recall_associative(
                model_high, list_high, enc_sim, dec_sim, n_trials=_pn, noise_scale=_ps
            )
            mrr_high = mean_recall_rate(curve_high)

            # Low Similarity list
            np.random.seed(sem_seed)
            model_low, sem_exp_low = _sem_train(list_low, enc_sim, dec_sim)
            model_low.reset_context()
            np.random.seed(sem_seed)
            curve_low = measure_recall_associative(
                model_low, list_low, enc_sim, dec_sim, n_trials=_pn, noise_scale=_ps
            )
            mrr_low = mean_recall_rate(curve_low)
            similarity_drop = mrr_low - mrr_high
        except Exception as e:
            print(f"Warning: Semantic similarity comparison failed: {e}")
            mrr_high = 0.0
            mrr_low = 0.0
            similarity_drop = 0.0

        results["metrics"]["mrr_high_similarity"] = mrr_high
        results["metrics"]["mrr_low_similarity"] = mrr_low
        try:
            results["metrics"]["semantic_epochs_high"] = float(sem_exp_high["epochs"])
            results["metrics"]["semantic_epochs_low"] = float(sem_exp_low["epochs"])
            results["metrics"]["semantic_criterion_reached"] = bool(
                sem_exp_high["reached"] and sem_exp_low["reached"])
        except NameError:
            results["metrics"]["semantic_criterion_reached"] = False
        results["metrics"]["similarity_effect_mrr_drop"] = similarity_drop

        # 5b. Parametric Sweep
        variances = [0.05, 0.1, 0.2, 0.5, 1.0]
        cosine_sims = []
        parametric_mrrs = []
        sem_sweep_exposure, sem_sweep_reached = [], []

        for v in variances:
            try:
                # Recreate encoder with specific category variance
                enc_var = SymbolicEncoder(sim_vocab, embedding_dim=100, category_variance=v, seed=42)
                dec_var = SymbolicDecoder(enc_var)

                # Calculate mean pairwise cosine similarity of list_high under this category variance
                embs = enc_var.encode(list_high)
                pairwise_sims = []
                for i in range(len(embs)):
                    for j in range(i + 1, len(embs)):
                        dot = np.dot(embs[i], embs[j])
                        norm_i = np.linalg.norm(embs[i])
                        norm_j = np.linalg.norm(embs[j])
                        pairwise_sims.append(dot / (norm_i * norm_j))
                mean_sim = float(np.mean(pairwise_sims))

                # Each variance is trained to its OWN criterion, so a low MRR at
                # high overlap is a discriminability result rather than the same
                # budget being worth less as the items crowd together. Where the
                # criterion is unreachable the exposure is censored and recorded.
                # Per-rung seeding, not one stream across the sweep: the
                # staircase inside _sem_train draws a different number of
                # samples at every rung (it probes once per checkpoint), so a
                # shared stream would make each rung's cue noise depend on how
                # long the PREVIOUS rung took to converge. That is precisely the
                # coupling that turns "harder rung" into "different noise", and
                # it is the reason the exposure ladder looked jagged.
                np.random.seed(sem_seed)
                model_var, exp_var = _sem_train(list_high, enc_var, dec_var)
                sem_sweep_exposure.append(int(exp_var["epochs"]))
                sem_sweep_reached.append(bool(exp_var["reached"]))
                model_var.reset_context()
                np.random.seed(sem_seed)
                curve_var = measure_recall_associative(
                    model_var, list_high, enc_var, dec_var, n_trials=_pn, noise_scale=_ps
                )

                # Only append to both on complete success of the iteration
                cosine_sims.append(mean_sim)
                parametric_mrrs.append(mean_recall_rate(curve_var))
            except Exception as e:
                print(f"Warning: Parametric similarity sweep failed for variance={v}: {e}")
                cosine_sims.append(0.0)
                parametric_mrrs.append(0.0)

        results["series"]["similarity_sweep"] = {
            "category_variances": variances,
            "cosine_similarities": cosine_sims,
            "epochs_to_criterion": sem_sweep_exposure,
            "criterion_reached": sem_sweep_reached,
            "mrr": parametric_mrrs
        }

        # Save Semantic Similarity Plots
        plt.figure(figsize=(7, 5))
        plt.plot(cosine_sims, parametric_mrrs, '-s', color='orange')
        plt.xlabel('Mean Pairwise Cosine Similarity of List')
        plt.ylabel('Mean Recall Rate (MRR)')
        plt.title('Semantic Interference vs. Pairwise Similarity')
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "semantic_similarity.png"), dpi=150)
        plt.close()

    # ==========================================
    # 5b. Symbolic disambiguation (overlapping sequences, Agster paradigm)
    # ==========================================
    if "symbolic_disambiguation" in selected:
        _pn, _ps = resolve_probe("symbolic_disambiguation", model_class, n_trials)  # probe protocol
        # N item sequences sharing a middle stretch, discriminated by a context
        # block. This is the capacity's symbolic arm, and it exists because the
        # spatial section cannot express the two axes that matter:
        #
        #   axis B  BifurcatingRouteGenerator grades discriminator similarity by
        #           rotating a one-hot vector, which uses two dimensions however
        #           many are allocated and is capped at TWO discriminators.
        #           `category_variance` grades it generatively instead, so a
        #           whole family of confusable discriminators is available.
        #   axis C  N > 2 confusable episodes then cost nothing extra. The load
        #           finding on the symbolic chain was that load rather than
        #           overlap separates arms, and axis C is its analogue here.
        #
        # NOT the same manipulation as `semantic_similarity`. That grades the
        # similarity of the ITEMS, where transitions are bijective and no cue
        # ever demands two successors, so it reads 0 with both sides at ceiling.
        # The ambiguity has to exist first -- that is what the shared stretch is
        # for -- and only then does discriminator similarity mean anything.
        from memval.benchmarks.symbolic_disambiguation import SymbolicDisambiguationBenchmark
        from memval.generators.symbolic_overlap import SymbolicOverlapGenerator

        sd_policy = exposure_policy("symbolic_disambiguation", 100)
        sd = {k: benchmark_arg(benchmark_args, "symbolic_disambiguation", k, d)
              for k, d in (("n_episodes", 4), ("total_length", 16),
                           ("shared_fraction", 0.5), ("shared_position", 0.0),
                           ("zone_fraction", 1.0),
                           ("embedding_dim", 64), ("n_discriminator_dims", 32),
                           ("disc_scale", 2.0), ("seed", 42))}
        print(f"Running Symbolic Disambiguation... "
              f"(exposure: {sd_policy['mode']})")
        try:
            sd_gen = SymbolicOverlapGenerator(seed=int(sd["seed"]))
            sd_bench = SymbolicDisambiguationBenchmark()

            def _sd_base(**over):
                cfg = dict(n_episodes=int(sd["n_episodes"]),
                           total_length=int(sd["total_length"]),
                           shared_fraction=float(sd["shared_fraction"]),
                           shared_position=float(sd["shared_position"]),
                           zone_fraction=float(sd["zone_fraction"]),
                           embedding_dim=int(sd["embedding_dim"]),
                           n_discriminator_dims=int(sd["n_discriminator_dims"]),
                           disc_scale=float(sd["disc_scale"]))
                cfg.update(over)
                return cfg

            # Exposure is settled on ONE-STEP PREDICTION ALONG AN EPISODE, which
            # sits upstream of the scored divergence choice. Settling it on the
            # scored quantity would pin the report to its own threshold.
            ref_set = sd_gen.generate(seed=int(sd["seed"]),
                                      **_sd_base(category_variance=0.5))
            sd_feat = ref_set["inputs"][0].shape[1]
            sd_nc = ref_set["n_content"]

            def _mk_sd(ep, _n=sd_feat):
                kw = _get_model_kwargs(model_kwargs, ep)
                return model_class(n_features=_n, **kw)

            def _fit_sd(m, ep, _X=ref_set["inputs"][0]):
                # ONE episode, not all N. The criterion establishes that the arm
                # can represent an episode of this length at all, which is
                # strictly upstream of whether it can tell N of them apart.
                # Training the criterion on all N folds the ambiguity into it:
                # the divergence step then has N legal successors, so the score
                # is capped near (L-1)/L by construction, the staircase never
                # converges, and it runs to max_epochs -- handing the scored
                # sweep a one-shot associator saturated by 512 passes and
                # reporting its wreckage as the model's capability.
                if hasattr(m, "n_epochs"):
                    m.n_epochs = ep
                m.fit_sequence(_X, epochs=ep)

            def _score_sd(m, _X=ref_set["inputs"][0], _nc=sd_nc):
                """One-step next-item accuracy along the episode, nearest among
                the episode's own items.

                Discrete, not a cosine: a mean cosine to the target saturates
                well below 0.95 on near-orthogonal embeddings, so a cosine
                criterion never converges. Scored on the SAME single episode
                that `_fit_sd` trains, so it is reachable and measures only
                whether the material was learned."""
                m.reset_context()
                cands = _X[:, :_nc]
                hits = tot = 0
                for t in range(len(_X) - 1):
                    p = np.asarray(m.predict_next(_X[t]), dtype=float)[:_nc]
                    pn = np.linalg.norm(p)
                    if pn < 1e-12:
                        tot += 1
                        continue
                    cn = np.linalg.norm(cands, axis=1)
                    cn[cn == 0] = 1.0
                    sims = (cands @ p) / (cn * pn)
                    hits += int(np.argmax(sims) == t + 1)
                    tot += 1
                return hits / tot if tot else 0.0

            _, sd_exp = train_at_exposure(sd_policy, _mk_sd, _fit_sd, _score_sd)
            sd_epochs = int(sd_exp["epochs"])
            print(f"  [symbolic_disambiguation] exposure to criterion: "
                  f"{sd_epochs} epochs (reached={sd_exp['reached']})")

            sd_mk = dict(model_kwargs)

            # --- axis B: graded discriminator similarity, cue AT the decision --
            # zone_fraction=1.0 leaves the discriminator on at the divergence
            # step, so this rung isolates DISCRIMINABILITY: the cue is present
            # and the only question is whether the arm can tell two contexts
            # apart. Any failure here is not a memory failure.
            variances = [0.05, 0.1, 0.2, 0.5, 1.0]
            var_rows = sd_bench.sweep(
                model_class, sd_mk,
                [_sd_base(category_variance=v, zone_fraction=1.0)
                 for v in variances],
                n_trials=1, fit_epochs=sd_epochs, seed=int(sd["seed"]))

            # --- axis C: load, at a fixed mid discriminability -----------------
            episode_counts = [2, 3, 4, 6, 8]
            load_rows = sd_bench.sweep(
                model_class, sd_mk,
                [_sd_base(n_episodes=n, category_variance=0.5, zone_fraction=1.0)
                 for n in episode_counts],
                n_trials=1, fit_epochs=sd_epochs, seed=int(sd["seed"]))

            # --- axis A: withdrawal, for comparability with the spatial suite --
            zone_fracs = [1.0, 0.75, 0.5, 0.25]
            delay_rows = sd_bench.sweep(
                model_class, sd_mk,
                [_sd_base(category_variance=0.5, zone_fraction=zf)
                 for zf in zone_fracs],
                n_trials=1, fit_epochs=sd_epochs, seed=int(sd["seed"]))

            # --- axis A', decoupled: support and delay are NOT one knob --------
            # Sweeping zone_fraction alone moves both at once, because the zone
            # starts at the corridor entrance: a shorter cue is also an earlier
            # withdrawal. Every "how much cue does it need" number read off such
            # a sweep is confounded with "how long can it hold one".
            # zone_params_for solves the two apart.
            from memval.generators.overlap import zone_params_for

            def _dd_row(duration, delay, **over):
                zp = zone_params_for(int(sd["total_length"]),
                                     float(sd["shared_fraction"]),
                                     float(sd["shared_position"]),
                                     duration, delay)
                return _sd_base(category_variance=0.5,
                                zone_fraction=zp["zone_fraction"],
                                zone_offset=zp["zone_offset"], **over)

            # Constant support (2 steps), growing gap: how long can it HOLD one?
            hold_delays = [0, 1, 2, 4, 6]
            hold_rows = sd_bench.sweep(
                model_class, sd_mk,
                [_dd_row(2, d) for d in hold_delays],
                n_trials=1, fit_epochs=sd_epochs, seed=int(sd["seed"]))

            # Constant gap (2 steps), growing support: how much does it NEED?
            support_durations = [1, 2, 4, 6]
            support_rows = sd_bench.sweep(
                model_class, sd_mk,
                [_dd_row(d, 2) for d in support_durations],
                n_trials=1, fit_epochs=sd_epochs, seed=int(sd["seed"]))

            # --- axis D: LENGTH of the shared stretch, varied independently ----
            # P22's fourth manipulation. The suffix is held fixed and the shared
            # stretch grown, under two support regimes: PERSISTENT (the
            # discriminator is on throughout, so the only thing that changes is
            # how many shared transitions sit between entry and fork) and
            # ONSET-ONLY (a 2-step cue at the corridor entrance, so the gap to
            # the fork grows with the stretch -- the "most complex" case, where
            # whatever separates the episodes must be carried internally).
            sd_suffix = int(sd["total_length"]) - int(
                round(float(sd["shared_fraction"]) * int(sd["total_length"])))
            shared_lengths = [2, 4, 6, 8, 10]

            def _len_row(sl, onset_only, **over):
                total = sl + sd_suffix
                frac = sl / total
                cfg = dict(total_length=total, shared_fraction=frac,
                           shared_position=0.0, category_variance=0.5)
                if onset_only:
                    zp = zone_params_for(total, frac, 0.0, 2, sl - 2)
                    cfg.update(zone_fraction=zp["zone_fraction"],
                               zone_offset=zp["zone_offset"])
                else:
                    cfg.update(zone_fraction=1.0)
                cfg.update(over)
                return _sd_base(**cfg)

            len_persist_rows = sd_bench.sweep(
                model_class, sd_mk, [_len_row(sl, False) for sl in shared_lengths],
                n_trials=1, fit_epochs=sd_epochs, seed=int(sd["seed"]))
            len_onset_rows = sd_bench.sweep(
                model_class, sd_mk, [_len_row(sl, True) for sl in shared_lengths],
                n_trials=1, fit_epochs=sd_epochs, seed=int(sd["seed"]))

            # --- axis C control: orthogonal discriminators ---------------------
            # Maximum separability, so discriminability is removed as a limiting
            # factor. Anything that still degrades with N here is ORDINARY
            # CAPACITY, not a disambiguation failure. Without this control a
            # falling load curve cannot be attributed to either.
            load_ctrl_rows = sd_bench.sweep(
                model_class, sd_mk,
                [_sd_base(n_episodes=n, discriminator_mode="orthogonal",
                          zone_fraction=1.0)
                 for n in episode_counts],
                n_trials=1, fit_epochs=sd_epochs, seed=int(sd["seed"]))

            # --- shared MIDDLE: the Agster paradigm proper ---------------------
            # Everything above shares a PREFIX (shared_position=0), so the
            # discriminator is the only thing that identifies an episode. With a
            # shared middle each episode has a unique prefix first, which means
            # identity is recoverable WITHOUT modality B at all -- if the arm can
            # carry it. Two conditions:
            #   informative discriminators DIFFER per sequence, as usual
            #   endogenous  every episode gets the SAME discriminator, so
            #               modality B carries zero information and the unique
            #               prefix is the only thing left. This is the hardest
            #               rung in the capacity and it is meaningful only for a
            #               StatePrimeable arm: with no carried state the probe
            #               input is identical across episodes and the floor is
            #               a protocol artefact.
            mid_pos = float(benchmark_arg(benchmark_args, "symbolic_disambiguation",
                                          "middle_position", 0.33))
            # zone_fraction=1.0 in BOTH rungs, so the cue spans the whole middle
            # and is present AT the decision. That is deliberate: it isolates
            # "can it reach the right arm when the middle is shared" from "can
            # it hold a cue across a gap", which axis A already owns. Under the
            # `identical` rung the cue is present but uninformative, so the two
            # rungs differ ONLY in whether modality B carries information.
            mid_rows = sd_bench.sweep(
                model_class, sd_mk,
                [_sd_base(shared_position=mid_pos, category_variance=0.5,
                          zone_fraction=1.0, discriminator_mode=m)
                 for m in ("category", "identical")],   # informative | endogenous
                n_trials=1, fit_epochs=sd_epochs, seed=int(sd["seed"]))
            mid_informative, mid_endo = mid_rows[0], mid_rows[1]

            # ---- headline scalars -------------------------------------------
            cued = var_rows[-1]           # most separable discriminators, cue on
            withdrawn = delay_rows[-1]    # longest cue-free delay
            biggest = load_rows[-1]

            results["metrics"]["symdis_cued_divergence_accuracy"] = \
                float(cued["divergence_accuracy"])
            results["metrics"]["symdis_cued_divergence_margin"] = \
                float(cued["divergence_margin"])
            results["metrics"]["symdis_withdrawn_divergence_accuracy"] = \
                float(withdrawn["divergence_accuracy"])
            results["metrics"]["symdis_withdrawn_divergence_margin"] = \
                float(withdrawn["divergence_margin"])
            results["metrics"]["symdis_shared_stretch_accuracy"] = \
                float(cued["shared_stretch_accuracy"])
            results["metrics"]["symdis_state_primed"] = bool(cued["state_primed"])

            # Axis B headline: the realised discriminator similarity at which
            # the arm drops below halfway between chance and ceiling. Reported
            # against REALISED cosine, not the category_variance dial, because
            # that mapping is nonlinear and saturating.
            sd_chance = float(cued["chance_level"])
            sd_half = sd_chance + 0.5 * (1.0 - sd_chance)
            sims_b = [float(r["discriminator_similarity"]) for r in var_rows]
            accs_b = [float(r["divergence_accuracy"]) for r in var_rows]
            thresh = float("nan")
            for sim, acc in sorted(zip(sims_b, accs_b)):
                if acc < sd_half:
                    thresh = sim
                    break
            results["metrics"]["symdis_similarity_tolerance_threshold"] = thresh

            # Axis C headline: the largest episode count still above chance.
            above = [int(r["n_episodes"]) for r in load_rows
                     if float(r["divergence_accuracy"]) > float(r["chance_level"])]
            results["metrics"]["symdis_max_episodes_above_chance"] = \
                float(max(above)) if above else 0.0
            results["metrics"]["symdis_context_graded_confusion_index"] = \
                float(biggest["context_graded_confusion_index"])

            # Decoupled support/delay. Half-way between chance and ceiling is
            # the crossing used throughout, so the two read against each other.
            def _first_where(rows, key, ok):
                for r in rows:
                    if ok(float(r["divergence_accuracy"])):
                        return float(r[key])
                return float("nan")

            # Smallest cue duration that works, at a FIXED 2-step gap.
            results["metrics"]["symdis_support_needed"] = _first_where(
                support_rows, "cue_duration", lambda a: a >= sd_half)
            # Longest gap survived, at a FIXED 2-step cue.
            _held = [float(r["delay"]) for r in hold_rows
                     if float(r["divergence_accuracy"]) >= sd_half]
            results["metrics"]["symdis_max_delay_at_fixed_support"] = (
                max(_held) if _held else float("nan"))

            # Load attribution. The orthogonal control removes discriminability
            # as a factor, so the gap separates "cannot tell them apart" from
            # "ran out of capacity".
            ctrl_big = load_ctrl_rows[-1]
            results["metrics"]["symdis_load_orthogonal_accuracy"] = \
                float(ctrl_big["divergence_accuracy"])
            results["metrics"]["symdis_load_disambiguation_cost"] = \
                float(ctrl_big["divergence_accuracy"]) - float(biggest["divergence_accuracy"])

            # Shared-stretch length: longest stretch still above the halfway
            # crossing, persistent cue (scored) and onset-only cue (protocol-
            # limited: a memoryless arm cannot carry the cue across any gap).
            _lp = [int(r["shared_len"]) for r in len_persist_rows
                   if float(r["divergence_accuracy"]) >= sd_half]
            _lo = [int(r["shared_len"]) for r in len_onset_rows
                   if float(r["divergence_accuracy"]) >= sd_half]
            results["metrics"]["symdis_max_shared_len_persistent"] = (
                float(max(_lp)) if _lp else 0.0)
            results["metrics"]["symdis_max_shared_len_onset"] = (
                float(max(_lo)) if _lo else 0.0)

            # Shared middle: unique prefix, then a shared corridor.
            results["metrics"]["symdis_middle_informative_accuracy"] = \
                float(mid_informative["divergence_accuracy"])
            results["metrics"]["symdis_middle_endogenous_accuracy"] = \
                float(mid_endo["divergence_accuracy"])
            results["metrics"]["symdis_middle_prefix_len"] = float(mid_informative["prefix_len"])

            results["metrics"]["symdis_epochs_to_criterion"] = float(sd_exp["epochs"])
            results["metrics"]["symdis_criterion_reached"] = bool(sd_exp["reached"])
            results["metrics"]["symdis_exposure_mode"] = sd_policy["mode"]
            results["metrics"]["symdis_exposure_source"] = sd_policy.get("source")

            results["series"]["symbolic_disambiguation"] = {
                "category_variances": [float(r["category_variance"]) for r in var_rows],
                "discriminator_similarities": sims_b,
                "variance_accuracy": accs_b,
                "variance_margin": [float(r["divergence_margin"]) for r in var_rows],
                "episode_counts": [int(r["n_episodes"]) for r in load_rows],
                "load_accuracy": [float(r["divergence_accuracy"]) for r in load_rows],
                "load_chance": [float(r["chance_level"]) for r in load_rows],
                "load_margin": [float(r["divergence_margin"]) for r in load_rows],
                "delays": [int(r["delay"]) for r in delay_rows],
                "delay_accuracy": [float(r["divergence_accuracy"]) for r in delay_rows],
                "delay_margin": [float(r["divergence_margin"]) for r in delay_rows],
                "confusion_matrix_largest_n": biggest["confusion_matrix"],
                "state_primed": bool(cued["state_primed"]),
                # decoupled support / delay
                "hold_delays": [int(r["delay"]) for r in hold_rows],
                "hold_accuracy": [float(r["divergence_accuracy"]) for r in hold_rows],
                "hold_cue_duration": [int(r["cue_duration"]) for r in hold_rows],
                "support_durations": [int(r["cue_duration"]) for r in support_rows],
                "support_accuracy": [float(r["divergence_accuracy"]) for r in support_rows],
                "support_delay": [int(r["delay"]) for r in support_rows],
                # orthogonal-discriminator load control
                "load_orthogonal_accuracy": [float(r["divergence_accuracy"])
                                             for r in load_ctrl_rows],
                # shared-stretch length, two support regimes
                "shared_lengths": [int(r["shared_len"]) for r in len_persist_rows],
                "shared_len_persistent_accuracy": [float(r["divergence_accuracy"])
                                                   for r in len_persist_rows],
                "shared_len_persistent_margin": [float(r["divergence_margin"])
                                                 for r in len_persist_rows],
                "shared_len_onset_accuracy": [float(r["divergence_accuracy"])
                                              for r in len_onset_rows],
                "shared_len_onset_margin": [float(r["divergence_margin"])
                                            for r in len_onset_rows],
                "shared_len_onset_delay": [int(r["delay"]) for r in len_onset_rows],
                # shared middle
                "middle_accuracy": [float(r["divergence_accuracy"]) for r in mid_rows],
                "middle_modes": [str(r["discriminator_mode"]) for r in mid_rows],
                "middle_endogenous": [bool(r["endogenous"]) for r in mid_rows],
                "middle_prefix_len": int(mid_informative["prefix_len"]),
            }

            fig, axes = plt.subplots(1, 6, figsize=(30, 4.6))
            axes[0].plot(sims_b, accs_b, '-o', color='tab:purple')
            axes[0].axhline(sd_chance, color='red', ls=':', lw=0.8,
                            label=f'chance ({sd_chance:.2f})')
            axes[0].set_xlabel('realised mean pairwise cosine of discriminators')
            axes[0].set_ylabel('divergence accuracy')
            axes[0].set_title('B: discriminability (cue ON at decision)')

            axes[1].plot([r["n_episodes"] for r in load_rows],
                         [r["divergence_accuracy"] for r in load_rows],
                         '-o', color='tab:blue', label='accuracy')
            axes[1].plot([r["n_episodes"] for r in load_rows],
                         [r["chance_level"] for r in load_rows],
                         ':', color='red', lw=0.8, label='chance (1/N)')
            axes[1].set_xlabel('N confusable episodes sharing the stretch')
            axes[1].set_ylabel('divergence accuracy')
            axes[1].set_title('C: load')

            axes[2].plot([r["delay"] for r in delay_rows],
                         [r["divergence_accuracy"] for r in delay_rows],
                         '-o', color='tab:green', label='accuracy')
            axes[2].axhline(sd_chance, color='red', ls=':', lw=0.8, label='chance')
            axes[2].set_xlabel('delay = shared steps with no discriminator')
            axes[2].set_ylabel('divergence accuracy')
            axes[2].set_title('A: withdrawal')

            axes[1].plot([r["n_episodes"] for r in load_ctrl_rows],
                         [r["divergence_accuracy"] for r in load_ctrl_rows],
                         '--^', color='tab:cyan',
                         label='orthogonal control')

            axes[3].plot([r["delay"] for r in hold_rows],
                         [r["divergence_accuracy"] for r in hold_rows],
                         '-o', color='tab:orange', label='delay @ 2-step cue')
            axes[3].plot([r["cue_duration"] for r in support_rows],
                         [r["divergence_accuracy"] for r in support_rows],
                         '-s', color='tab:brown', label='cue duration @ 2-step gap')
            axes[3].axhline(sd_chance, color='red', ls=':', lw=0.8, label='chance')
            axes[3].set_xlabel('steps')
            axes[3].set_ylabel('divergence accuracy')
            axes[3].set_title("A': support and delay, decoupled")

            # NB 'informative', not 'cued': in this suite "cued recall" names a
            # PROBE (one-step, supplied cue) as against a rollout. These two
            # rungs use the identical probe and differ only in whether modality
            # B carries information.
            axes[4].bar(['informative', 'endogenous'],
                        [mid_informative["divergence_accuracy"],
                         mid_endo["divergence_accuracy"]],
                        color=['tab:green', 'tab:red'])
            axes[4].axhline(sd_chance, color='red', ls=':', lw=0.8, label='chance')
            axes[4].set_ylabel('divergence accuracy')
            axes[4].set_title(f'shared MIDDLE (prefix={mid_informative["prefix_len"]} steps)')

            axes[5].plot(shared_lengths,
                         [r["divergence_accuracy"] for r in len_persist_rows],
                         '-o', color='tab:green', label='cue persistent through the stretch')
            axes[5].plot(shared_lengths,
                         [r["divergence_accuracy"] for r in len_onset_rows],
                         '--s', color='tab:red', label='cue onset-only (2 steps at entry)')
            axes[5].axhline(sd_chance, color='red', ls=':', lw=0.8, label='chance')
            axes[5].set_xlabel('length of the shared stretch (steps; suffix fixed)')
            axes[5].set_ylabel('divergence accuracy')
            axes[5].set_title('D: shared-stretch length, two support regimes')
            for ax in axes:
                ax.set_ylim(-0.05, 1.05)
                ax.legend(fontsize=8)
            fig.suptitle(f'Symbolic disambiguation - {model_name} '
                         f'(state_primed={bool(cued["state_primed"])})')
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "symbolic_disambiguation.png"),
                        dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Symbolic disambiguation failed for {model_name}: {e}")
            for k in ("symdis_cued_divergence_accuracy",
                      "symdis_cued_divergence_margin",
                      "symdis_withdrawn_divergence_accuracy",
                      "symdis_withdrawn_divergence_margin",
                      "symdis_shared_stretch_accuracy",
                      "symdis_similarity_tolerance_threshold",
                      "symdis_max_episodes_above_chance",
                      "symdis_context_graded_confusion_index",
                      "symdis_support_needed",
                      "symdis_max_delay_at_fixed_support",
                      "symdis_load_orthogonal_accuracy",
                      "symdis_load_disambiguation_cost",
                      "symdis_middle_informative_accuracy",
                      "symdis_middle_endogenous_accuracy",
                      "symdis_middle_prefix_len",
                      "symdis_epochs_to_criterion"):
                results["metrics"][k] = float('nan')
            results["metrics"]["symdis_criterion_reached"] = False
            results["metrics"]["symdis_state_primed"] = False
            results["metrics"]["symdis_exposure_mode"] = sd_policy["mode"]
            results["metrics"]["symdis_exposure_source"] = sd_policy.get("source")
            results["series"]["symbolic_disambiguation"] = {}

    # ==========================================
    # 6. Schema consistency (acquisition rate vs prior knowledge)
    # ==========================================
    if "cue_masking" in selected:
        # Structural half of pattern completion (docs/capacities/capacities.md P17): a
        # graded fragment of the cue, against noise_invariance's graded
        # corruption of all of it. Deliberately NOT built on SymbolicEncoder --
        # its dimensions are a random basis, so masking them would score the
        # embedding geometry rather than the arm, which is what got letter_noise
        # deleted (D5). See memval/benchmarks/cue_masking.py.
        from memval.benchmarks.cue_masking import run_cue_masking

        mask_policy = exposure_policy("cue_masking", 300)
        print(f"Running Cue Masking benchmark... "
              f"(exposure: {mask_policy['mode']})")
        mask_kwargs = {
            k: benchmark_arg(benchmark_args, "cue_masking", k, default)
            for k, default in (
                # Two levels, not schema_consistency's three: the block contrast
                # needs equally-sized shared/identity blocks. See build_hierarchy.
                ("branching", (2, 12)),
                ("features_per_node", 6),
                ("seq_len", 8),
                # "across" spans categories so both feature blocks discriminate;
                # under "within" the shared block is constant across the list and
                # the block contrast is settled by the stimulus. See study_list.
                ("list_scope", "across"),
                ("n_draws", 8),
                ("renormalize", True),
                ("feedback_mode", "l2"),
                ("seed", 0),
            )
        }
        try:
            mask = run_cue_masking(
                model_class=model_class,
                model_kwargs=model_kwargs,
                exposure=mask_policy,
                run_dir=run_dir,
                metrics_filename="cue_masking_metrics.json",
                run_name=model_name,
                **mask_kwargs,
            )
            # Keys are already `mask_`-prefixed by the section itself.
            results["metrics"].update(mask["metrics"])
            results["series"].update(mask["series"])
        except Exception as e:
            print(f"Warning: Cue masking failed for {model_name}: {e}")

    if "cue_availability" in selected:
        # Third probe property (paper figure B): a whole MODALITY is absent from
        # the cue. Two-block input (symbolic | the item's own pure tone, one tone
        # per item since 2026-09-26); cue with both, audio removed, or symbol
        # removed; cued on every item and rolled out from each sequence's first
        # item, scored on exact next-item recall. Criterion-referenced
        # on the full cue; round-robin ingestion in the multi condition so that
        # blocked-list forgetting (Continual retention's axis) stays out of it.
        from memval.benchmarks.cue_availability import run_cue_availability
        _pn, _ps = resolve_probe("cue_availability", model_class, n_trials)  # clean single
        avail_policy = exposure_policy("cue_availability", 300)
        print(f"Running Cue Availability benchmark... (exposure: {avail_policy['mode']})")
        avail_kwargs = {
            k: benchmark_arg(benchmark_args, "cue_availability", k, default)
            for k, default in (
                # 10 per sequence since 2026-09-26: at 4 every readout but EP's
                # was at ceiling. Categories fruit / animal / number (see
                # cue_availability.DEFAULT_MATERIAL).
                ("seq_len", 10),
                ("n_sequences", 3),
                # 100 bins at sigma 4 Hz keep the 30 item tones at pairwise
                # cosine <= 0.06; see MultimodalEncoder.
                ("n_bins", 100),
                ("tone_sigma", 4.0),
                ("audio_gain", 1.0),
                # raw since 2026-09-26, matching noise_invariance: l2 is a harness
                # intervention (a no-op for positively-homogeneous arms, a shift for
                # DTS-ESN and EP), not something this section is asking about.
                ("feedback_mode", "raw"),
                ("renormalize", True),
                ("seed", 42),
            )
        }
        try:
            avail = run_cue_availability(
                model_class=model_class, model_kwargs=model_kwargs, vocab=vocab,
                exposure=avail_policy, run_dir=run_dir,
                metrics_filename="cue_availability_metrics.json",
                run_name=model_name, **avail_kwargs)
            results["metrics"].update(avail["metrics"])
            results["series"].update(avail["series"])
        except Exception as e:
            print(f"Warning: Cue availability failed for {model_name}: {e}")

    if "schema_consistency" in selected:
        _pn, _ps = resolve_probe("schema_consistency", model_class, n_trials)  # probe protocol
        # Imported here, not at module scope: schema_consistency imports the
        # recall helpers from this module, so a top-level import is circular.
        from memval.benchmarks.schema_consistency import run_schema_consistency

        print("Running Schema Consistency benchmark...")
        schema_kwargs = {
            k: benchmark_arg(benchmark_args, "schema_consistency", k, default)
            for k, default in (
                ("branching", (2, 2, 6)),
                ("features_per_node", 2),
                ("schema_trials", 40),          # CEILING for the acquisition staircase
                ("schema_criterion", 0.75),     # base-list MRR that counts as acquired
                ("n_seeds", 5),                 # replicates; mean under the key, <key>_sd
                ("new_item_trials", 15),
                ("n_probe_trials", 20),
                ("criterion_frac", 0.75),
                ("seed", 0),
                # "extended" (default) trains the host list + the new item;
                # "focused" trains the new pair alone. Only "focused" yields a
                # valid interference read-out -- under "extended" the host
                # category is rehearsed while the new item is learned. See
                # INTERFERENCE_PROTOCOLS in schema_consistency.py.
                ("interference_protocol", "extended"),
            )
        }
        # Probe protocol for this section (memval/benchmarks/probe.py): clean single
        # cue. `schema_kwargs` already carries the section's benchmark_arg
        # defaults for these two keys, so override there rather than passing them
        # twice -- the section's own probe used to draw 20 noisy cues at sigma=0.05
        # independently of the rest of the suite.
        schema_kwargs["n_probe_trials"] = _pn
        schema_kwargs["noise_scale"] = _ps
        schema = run_schema_consistency(
            model_class=model_class,
            model_kwargs=model_kwargs,
            # Deliberately NOT section_epochs(): this section measures
            # trials-to-criterion at one presentation per trial. Inheriting the
            # global --epochs would saturate every rung on trial 1 and erase the
            # effect (docs/sections/schema_benchmark_design.md S7, fix 2). Override it
            # only per-section, and knowingly.
            epochs_per_trial=benchmark_arg(
                benchmark_args, "schema_consistency", "epochs_per_trial", 1),
            run_dir=run_dir,
            metrics_filename="schema_consistency_metrics.json",
            run_name=model_name,
            **schema_kwargs,
        )
        # Keys are already `schema_`-prefixed by the section itself.
        results["metrics"].update(schema["metrics"])
        results["series"].update(schema["series"])

    # ==========================================
    # 7 & 8. Serial order, metric time (5.4, 5.5)
    # ==========================================
    # Half of Serial order's weight, and unmeasured until 2026-09-04. Both are
    # gated on the arm's DECLARED capability rather than on a signature check:
    # `AsymmetricHopfieldNetwork.fit_sequence` takes **kwargs and would accept
    # `intervals=[...]` while remaining clocked by ordinal, so duck-typing would
    # report a capability the arm does not have, silently and in the direction
    # that inflates the score.
    # ==========================================
    # 6b. Cognitive phenomena -- L0 behavioural read-outs
    # ==========================================
    # The shape of recall (serial position, list length, presentation rate,
    # prior-list intrusion recency, semantic clustering) reported against the
    # human reference band (Kahana 2020 sec 4). DESCRIPTIVE BY CONTRACT: the
    # section returns no scored metric, writes its own
    # cognitive_phenomena_metrics.json, and only its summary enters
    # results["series"]. See docs/sections/cognitive_phenomena_design.md.
    if "cognitive_phenomena" in selected:
        _pn, _ps = resolve_probe("cognitive_phenomena", model_class, n_trials)  # clean single
        from memval.benchmarks.cognitive_phenomena import run_cognitive_phenomena
        cog_policy = exposure_policy("cognitive_phenomena", 300)
        print(f"Running Cognitive Phenomena (L0)... (exposure: {cog_policy['mode']})")
        cog_kwargs = {
            k: benchmark_arg(benchmark_args, "cognitive_phenomena", k, default)
            for k, default in (
                ("lengths", (5, 7, 10)),
                ("headline_length", 7),
                ("categories", ("fruit", "animal", "number")),
                ("ladder", (1, 2, 4, 8)),
                # Replicate unit is (category, encoder seed): three categories
                # alone left every band index sign-unstable. 3 x 5 = 15.
                ("n_seeds", 5),
                ("pli_n_tasks", 6),
                ("pli_seq_len", 5),
                ("pli_between_cosines", (0.0, 0.5, 0.8)),
                ("semantic_variances", (0.1, 0.2, 1.0, 3.0)),
                ("semantic_ladder", (1, 2, 4)),
                ("feedback_mode", "l2"),
                ("seed", 42),
            )
        }
        try:
            cog = run_cognitive_phenomena(
                model_class=model_class, model_kwargs=model_kwargs, vocab=vocab,
                exposure=cog_policy, run_dir=run_dir,
                metrics_filename="cognitive_phenomena_metrics.json",
                run_name=model_name, **cog_kwargs)
            assert not cog["metrics"], "cognitive_phenomena must not emit scored metrics"
            results["series"].update(cog["series"])
        except Exception as e:
            print(f"Warning: Cognitive phenomena failed for {model_name}: {e}")

    _interval_ok = interval_sections_applicable(model_class)

    if "interval_retention" in selected:
        _pn, _ps = resolve_probe("interval_retention", model_class, n_trials)  # probe protocol
        if not _interval_ok["interval_retention"]:
            print(f"[SKIP] interval_retention: {model_name} does not declare "
                  f"TemporallyClocked, so elapsed time cannot change its state. "
                  f"Scored not-applicable, never zero.")
        else:
            iv_reps = int(benchmark_arg(benchmark_args, "interval_retention",
                                        "n_reps", 40))
            print(f"Running Interval Retention (5.4)... "
                  f"(gap is the only discriminator, n_reps={iv_reps})")
            try:
                out = run_interval_retention(
                    lambda: _build_interval_model(model_class, model_kwargs),
                    n_reps=iv_reps, n_trials=_pn, seed=42)
                results["metrics"].update(out["metrics"])
                results["series"].update(out["series"])
                _plot_interval_retention(out, plots_dir)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: interval_retention failed: {e}")

    if "interval_generation" in selected:
        _pn, _ps = resolve_probe("interval_generation", model_class, n_trials)  # probe protocol
        if not _interval_ok["interval_generation"]:
            print(f"[SKIP] interval_generation: {model_name} does not declare "
                  f"TimingPredictive, so it has no read-out that emits a gap. "
                  f"Scored not-applicable, never zero.")
        else:
            gen_reps = int(benchmark_arg(benchmark_args, "interval_generation",
                                         "n_reps", 40))
            print(f"Running Interval Generation (5.5)... "
                  f"(tempo + rhythm under autonomous rollout, n_reps={gen_reps})")
            try:
                out = run_interval_generation(
                    lambda: _build_interval_model(model_class, model_kwargs),
                    n_reps=gen_reps, seed=42)
                results["metrics"].update(out["metrics"])
                results["series"].update(out["series"])
                _plot_interval_generation(out, plots_dir)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: interval_generation failed: {e}")

    # ---- exposure deviation guard ---------------------------------------
    # Derived from whatever `*_epochs_to_criterion` keys the run produced, so a
    # section added later is covered without touching this block.
    # Sections do not all spell their exposure key the same way: most use
    # `<prefix>_epochs_to_criterion`, but `multiple_sequences` reports one per
    # list (`_epochs_A` / `_epochs_B`) and `semantic_similarity` one per overlap
    # condition (`_epochs_high` / `_epochs_low`). Matching only the common
    # spelling would silently omit exactly the sections that train twice.
    _EXPOSURE_SUFFIXES = ("_epochs_to_criterion", "_epochs_A", "_epochs_B",
                          "_epochs_high", "_epochs_low", "_epochs_interleaved")
    _exposure_keys = sorted(
        k for k in results["metrics"]
        if any(k.endswith(sfx) for sfx in _EXPOSURE_SUFFIXES))
    if _exposure_keys:
        _base = exposure_baseline()
        results["metrics"]["exposure_baseline_epochs"] = (
            float(_base["epochs"]) if _base["epochs"] is not None else float('nan'))
        results["metrics"]["exposure_baseline_reached"] = bool(_base["reached"])

        _ratios = {}
        for k in _exposure_keys:
            v = results["metrics"].get(k)
            sfx = next(x for x in _EXPOSURE_SUFFIXES if k.endswith(x))
            prefix = k[: -len(sfx)] + ("" if sfx == "_epochs_to_criterion"
                                       else sfx.replace("_epochs", ""))
            ratio = float('nan')
            if (_base["epochs"] and isinstance(v, (int, float))
                    and v == v and v > 0):
                ratio = float(v) / float(_base["epochs"])
            results["metrics"][f"{prefix}_exposure_vs_baseline"] = ratio
            if ratio == ratio:
                _ratios[prefix] = ratio

        _band = EXPOSURE_DEVIATION_BAND
        _worst, _worst_at = 1.0, None
        for prefix, r in _ratios.items():
            d = max(r, 1.0 / r) if r > 0 else float('inf')
            if d > _worst:
                _worst, _worst_at = d, prefix
        results["metrics"]["exposure_deviation_max"] = float(_worst)
        results["metrics"]["exposure_deviation_section"] = str(_worst_at or "")
        results["metrics"]["exposure_within_band"] = bool(_worst <= _band)
        if _worst > _band:
            print(f"  [exposure] WARNING: {_worst_at} trained at {_worst:.1f}x the "
                  f"baseline ({_base['epochs']} epochs). Cross-section comparison "
                  f"with it is not like-for-like.")
        results["series"]["exposure_vs_baseline"] = {
            "baseline_epochs": _base["epochs"],
            "sections": sorted(_ratios),
            "ratios": [_ratios[p] for p in sorted(_ratios)],
            "band": _band,
        }

    # Write metrics.json
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Symbolic pipeline finished successfully for {model_name}. Results saved to {run_dir}")
    return results["metrics"]
