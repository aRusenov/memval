"""Continual chain: the retention matrix as a benchmark section.

Three things the suite could not previously ask, all read off one N-task chain:

1. **Acquisition under load** (the diagonal). Every stability read-out in the
   suite can be maximised by not learning: an arm frozen after task 0 has a
   perfect final row, ACC = 1, BWT = 0 and ``avg_forgetting`` = 0. The diagonal
   ``R[i, i]`` is that arm's tell, and it costs nothing to read because the
   matrix already contains it. ``memval.metrics.retention`` turns it into
   ``avg_learning`` / ``intransigence`` and pairs it with stability in
   ``stability_plasticity_index``.

2. **The load axis** (``forgetting_by_gap``). ``multiple_sequences`` is the
   T = 2 special case of this section — one interposed list, one number. The
   chain gives the retained fraction at every interposition gap 1 … T-1, i.e.
   the forgetting gradient rather than a single post-interference point.

3. **Selective retention** (the rehearsal phase, ``rehearse=True``).
   ``paper/capacities.md`` argues that forgetting under a finite substrate is
   functional rather than a failure, but every other section treats all stored
   material as worth keeping, so the claim had no instrument. Here a designated
   subset stays relevant — it is re-presented — and the rest is not. The
   question is not *how much* survives but *whether what survives is the part
   that is still in use*.

   The rehearsed subset is the **oldest** tasks, deliberately: they are the most
   decayed (so a gain has headroom to be visible) and they are the ones recency
   works against (so selectivity cannot be recency wearing a different hat). The
   read-out is a difference-in-differences — each task against its own
   pre-rehearsal score — which removes the serial-position baseline entirely.

   Read it only under capacity pressure. If the chain never saturated the
   substrate, nothing had to be discarded, and a selectivity of zero means the
   question was not posed rather than that the arm failed it; ``pressure`` and
   ``under_pressure`` are emitted as the guard.

The protocol lives here; the matrix and every statistic derived from it come
from ``memval.metrics.retention``, so this section, ``bin/continual_chain_
experiment.py`` and anything else adopting a retention read-out report the
identical quantity.

Scope: symbolic (an encoder/decoder pair and a word-list per task). The metric
module underneath is modality-agnostic, so a spatial chain is a second caller
of it rather than a change here.
"""
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..encoders.symbolic import SymbolicDecoder, SymbolicEncoder
from ..metrics.retention import retention_matrix, retention_summary
from .base import Benchmark


def build_chain_material(
    vocab: Dict[str, str],
    n_tasks: int,
    seq_len: int,
    *,
    embedding_dim: int = 100,
    category_variance: float = 0.2,
    between_category_cosine: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """The chain's stimulus: one category per task, and the geometry it has.

    This is the construction the ``continual_chain`` section has always used,
    factored out so the section, ``bin/continual_chain_overlap_sweep.py`` and
    any other caller build the identical material from the identical arguments.
    The first ``n_tasks`` categories (sorted) with at least ``seq_len`` words
    become the tasks; each task is that category's first ``seq_len`` words in
    sorted order.

    Two overlap dials, both on ``SymbolicEncoder``:

    - ``category_variance`` (sigma) sets the *within*-task cosine
      ``rho = 1 / (1 + d sigma^2)``: how confusable the items of one list are.
    - ``between_category_cosine`` (rho_b) sets the cosine between task
      prototypes exactly, and so the *between*-task word cosine
      ``rho * rho_b``. This is the forgetting lever -- task B disturbs task A
      in proportion to ``x_B . x_A`` -- and it is the axis the section's
      default (``None``: quasi-orthogonal, ~0 +- 1/sqrt(d)) holds at zero.

    Returns a dict with ``sequences`` (task -> words), ``vocab`` (word ->
    task, order-stable), ``encoder``, ``decoder``, ``embeddings`` (task ->
    array) and ``geometry``: the measured within / between cosines next to the
    laws they should follow, so a report can print the stimulus it actually
    had rather than the parameters it asked for.
    """
    by_category: Dict[str, List[str]] = {}
    for word, cat in vocab.items():
        by_category.setdefault(cat, []).append(word)
    usable = [c for c in sorted(by_category) if len(by_category[c]) >= seq_len]
    if len(usable) < n_tasks:
        raise ValueError(
            f"need {n_tasks} categories with >= {seq_len} words, "
            f"vocab has {len(usable)}: {usable}")
    cats = usable[:n_tasks]
    sequences = {c: sorted(by_category[c])[:seq_len] for c in cats}
    chain_vocab = {w: c for c, ws in sequences.items() for w in ws}

    encoder = SymbolicEncoder(chain_vocab, embedding_dim=embedding_dim,
                              category_variance=category_variance, seed=seed,
                              between_category_cosine=between_category_cosine)
    decoder = SymbolicDecoder(encoder)
    embeddings = {c: encoder.encode(ws) for c, ws in sequences.items()}

    # Measured geometry, beside the laws (docs/encoder_design.md section 6.4).
    E = encoder.embeddings
    task_of = np.array([chain_vocab[w] for w in encoder.idx_to_word])
    G = E @ E.T
    same = task_of[:, None] == task_of[None, :]
    off = ~np.eye(len(task_of), dtype=bool)
    P = np.stack([encoder.category_vectors[c] for c in cats])
    PC = (P @ P.T)[~np.eye(len(cats), dtype=bool)]
    rho_law = 1.0 / (1.0 + embedding_dim * float(category_variance) ** 2)
    rho_b = between_category_cosine
    geometry = {
        "embedding_dim": int(embedding_dim),
        "category_variance": float(category_variance),
        "between_category_cosine": None if rho_b is None else float(rho_b),
        "within_cos_law": rho_law,
        "within_cos_measured": float(G[same & off].mean()),
        "between_word_cos_law": None if rho_b is None else rho_law * float(rho_b),
        "between_word_cos_measured": float(G[~same].mean()),
        "between_word_cos_sd": float(G[~same].std()),
        "prototype_cos_min": float(PC.min()),
        "prototype_cos_max": float(PC.max()),
    }
    return {"sequences": sequences, "vocab": chain_vocab, "encoder": encoder,
            "decoder": decoder, "embeddings": embeddings, "geometry": geometry}


def consolidate_if_supported(model: Any, embeddings: np.ndarray) -> None:
    """Call ``model.consolidate`` with whatever signature the arm declares.

    Arms that anchor weights (EWC and friends) need the sequence they just
    learned; arms that do not take no argument; arms with no consolidation step
    are skipped. Mirrors the helper in ``spatial_disambiguation`` rather than
    importing it, because that one is typed for spatial trajectories.
    """
    if not hasattr(model, "consolidate"):
        return
    if "sequence_data" in inspect.signature(model.consolidate).parameters:
        context = np.tile(np.array([1.0]), (len(embeddings), 1))
        model.consolidate(embeddings, context_data=context)
    else:
        model.consolidate()


#: Outcome classes for a failed probe, by where the decoded word came from.
INTRUSION_CLASSES = ("correct", "within_task", "cross_earlier", "cross_later")
#: Where the best competitor of the target sits, for every probe (failed or not).
COMPETITOR_CLASSES = ("same_task", "earlier", "later")


def probe_intrusions(
    model: Any,
    sequences: Dict[str, List[str]],
    encoder: Any,
    decoder: Any,
    context_vec: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """What the model recalls *instead*, and who the runner-up is, per probe.

    The retention matrix scores each probe correct / incorrect and keeps
    nothing else, so any claim that a rung "confuses across tasks" or "within
    a task" is an inference from the stimulus geometry. This turns it into a
    measurement. One clean cue per position, exactly the section's probe
    (``reset_context`` per task, ``current_t`` set, ``predict_next`` on the
    clean embedding, nearest-neighbour decode over the chain vocabulary), and
    for every probe it records:

    - ``outcome`` -- ``correct``, or the *source* of the intruding word:
      ``within_task`` (same list; ``offset`` is its position relative to the
      target), ``cross_earlier`` (a task trained before this one) or
      ``cross_later`` (trained after: retroactive interference).
    - ``competitor`` -- the best non-target item by cosine, classed the same
      way (``same_task`` / ``earlier`` / ``later``), and ``margin`` = cos to
      target minus cos to that competitor. This resolves above the accuracy
      floor: the runner-up can switch from a list-mate to a later task while
      the matrix still reads 1.00, which is the direction of pressure before
      any error occurs.

    Meant to run on the model's final state (after the last stage, or after
    rehearsal if that ran). Returns the per-probe ``records`` and the counts a
    figure needs.
    """
    names = list(sequences.keys())
    where: Dict[str, tuple] = {}
    for ti, name in enumerate(names):
        for pos, w in enumerate(sequences[name]):
            where[w] = (ti, pos)
    vocab_words = list(encoder.idx_to_word)
    E = np.asarray(encoder.embeddings, dtype=float)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    idx_of = {w: k for k, w in enumerate(vocab_words)}
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])

    def _rel(task_idx: int, other_task: int, same_label: str) -> str:
        if other_task == task_idx:
            return same_label
        return "earlier" if other_task < task_idx else "later"

    records: List[Dict[str, Any]] = []
    for ti, name in enumerate(names):
        words = sequences[name]
        if hasattr(model, "reset_context"):
            model.reset_context()
        for t in range(len(words) - 1):
            model.current_t = t
            cue = encoder.encode([words[t]])[0]
            pred = np.asarray(model.predict_next(cue, current_context=context), dtype=float).ravel()
            target = words[t + 1]
            decoded = decoder.decode(pred, top_k=1)[0]
            pn = np.linalg.norm(pred)
            sims = En @ (pred / pn) if pn > 1e-12 else np.zeros(len(vocab_words))
            ti_target = idx_of[target]
            others = np.delete(np.arange(len(vocab_words)), ti_target)
            comp_idx = int(others[np.argmax(sims[others])])
            comp_word = vocab_words[comp_idx]
            comp_task, comp_pos = where[comp_word]
            rec = {
                "task": name, "task_index": ti, "position": t + 1, "cue": words[t],
                "target": target, "decoded": decoded,
                "margin": float(sims[ti_target] - sims[comp_idx]),
                "competitor": comp_word,
                "competitor_class": _rel(ti, comp_task, "same_task"),
                "competitor_offset": (comp_pos - (t + 1)) if comp_task == ti else None,
            }
            if decoded == target:
                rec["outcome"] = "correct"
                rec["offset"] = None
            else:
                d_task, d_pos = where[decoded]
                cls = _rel(ti, d_task, "within_task")
                rec["outcome"] = {"earlier": "cross_earlier", "later": "cross_later"}.get(cls, cls)
                rec["offset"] = (d_pos - (t + 1)) if d_task == ti else None
            records.append(rec)

    outcome_counts = {c: sum(r["outcome"] == c for r in records) for c in INTRUSION_CLASSES}
    competitor_counts = {c: sum(r["competitor_class"] == c for r in records)
                         for c in COMPETITOR_CLASSES}
    margins = np.array([r["margin"] for r in records], dtype=float)
    by_comp = {c: float(np.mean([r["margin"] for r in records if r["competitor_class"] == c]))
               if competitor_counts[c] else float("nan") for c in COMPETITOR_CLASSES}
    offsets = [r["offset"] for r in records if r["offset"] is not None]
    within_offsets = {str(k): int(sum(o == k for o in offsets)) for k in sorted(set(offsets))}
    return {
        "n_probes": len(records),
        "outcome_counts": outcome_counts,
        "competitor_counts": competitor_counts,
        "margin_mean": float(margins.mean()) if margins.size else float("nan"),
        "margin_by_competitor": by_comp,
        "within_offsets": within_offsets,
        "records": records,
    }


def _avg_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (1-based), numpy only."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y) -> float:
    """Spearman rank correlation, NaN if either side is constant."""
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def usage_phase(
    model: Any,
    sequences: Dict[str, List[str]],
    embeddings: Dict[str, np.ndarray],
    encoder: Any,
    *,
    rates: Optional[Sequence[float]] = None,
    schedule: Optional[np.ndarray] = None,
    n_blocks: int = 16,
    rng: Optional[np.random.Generator] = None,
    epochs_per_presentation: int = 1,
    score_fn: Optional[Callable[[int], float]] = None,
) -> Dict[str, Any]:
    """A use-structured stream after the chain: does retention track future use?

    The environmental-statistics operationalisation of "useful" (Anderson &
    Schooler 1991): a memory's utility is how often the world re-presents it,
    and an ideal allocator retains in proportion to that. No reward channel,
    nothing the model has to read -- every arm can be tested on it.

    After the chain has loaded the substrate, run ``n_blocks`` blocks. In each
    block task *i* is re-presented (one ``fit_sequence`` call of
    ``epochs_per_presentation`` passes) with probability ``rates[i]``, or
    exactly as ``schedule[block, i]`` says when an explicit schedule is given
    (the recency condition: equal counts, different placement). Tasks scheduled
    in the same block are presented in random order. After every block every
    task is probed with the clean single-cue **margin** (target minus best
    competitor, ``measure_recall_margin``), because accuracy saturates on the
    frequently-used tasks and the ranking lives above that floor.

    Returns the margin trajectory ``(n_blocks + 1, n_tasks)`` (row 0 = before
    the stream), the realised presentation counts, and the allocation scores:

    - ``spearman_margin_vs_use`` -- rank correlation between final margin and
      realised count. 1.0 is the ideal allocator; the sign is the finding.
    - ``misallocation`` -- mean final margin of never-used tasks minus mean
      final margin of the most-used tasks. Positive means the arm is keeping
      what the world stopped presenting at the expense of what it keeps
      presenting -- the over-stable failure.
    - ``used_below_criterion`` -- how many used tasks end below 0.95 cued
      recall (needs ``score_fn``), the plasticity cost of that stability.

    Forgetting here is interference-driven: an unvisited trace degrades only
    through overlap with the visited ones. Run it at a between-task cosine
    that produces pressure, and run the same ladder with exactly orthogonal
    tasks as the disuse control -- a flat never-used trace there means the arm
    has no passive forgetting at all, which is a roster property to state,
    not stability.
    """
    from .probe import mean_margin, measure_recall_margin

    names = list(sequences.keys())
    T = len(names)
    rng = rng if rng is not None else np.random.default_rng(0)
    if (rates is None) == (schedule is None):
        raise ValueError("give exactly one of rates (frequency) or schedule (explicit)")
    if rates is not None:
        rates = np.asarray(rates, dtype=float)
        if rates.shape != (T,):
            raise ValueError(f"rates must have one entry per task ({T}), got {rates.shape}")
        sched = rng.random((n_blocks, T)) < rates[None, :]
        sched[:, rates >= 1.0] = True
        sched[:, rates <= 0.0] = False
    else:
        sched = np.asarray(schedule, dtype=bool)
        if sched.ndim != 2 or sched.shape[1] != T:
            raise ValueError(f"schedule must be (n_blocks, {T}), got {sched.shape}")
        n_blocks = sched.shape[0]

    def probe_margin(i: int) -> float:
        return mean_margin(measure_recall_margin(model, sequences[names[i]], encoder))

    margins = np.full((n_blocks + 1, T), np.nan)
    margins[0] = [probe_margin(i) for i in range(T)]
    for b in range(n_blocks):
        todo = np.flatnonzero(sched[b])
        rng.shuffle(todo)
        for i in todo:
            emb = embeddings[names[i]]
            model.fit_sequence(emb, epochs=int(epochs_per_presentation))
            consolidate_if_supported(model, emb)
        margins[b + 1] = [probe_margin(i) for i in range(T)]

    counts = sched.sum(axis=0).astype(int)
    final = margins[-1]
    used = counts > 0
    never = ~used
    top = counts == counts.max()
    misallocation = (float(final[never].mean() - final[top].mean())
                     if never.any() and top.any() else float("nan"))
    accuracy = ([float(score_fn(i)) for i in range(T)] if score_fn is not None else None)
    below = (int(sum(1 for i in range(T) if used[i] and accuracy[i] < 0.95))
             if accuracy is not None else None)

    return {
        "task_labels": names,
        "n_blocks": int(n_blocks),
        "schedule": sched.tolist(),
        "counts": counts.tolist(),
        "rates": None if rates is None else rates.tolist(),
        "margins": margins.tolist(),
        "final_margin": final.tolist(),
        "final_accuracy": accuracy,
        "delta_margin": (final - margins[0]).tolist(),
        "spearman_margin_vs_use": spearman(final, counts),
        "misallocation": misallocation,
        "used_below_criterion": below,
    }


class ContinualChainBenchmark(Benchmark):
    """Train one model on a chain of N sequences, scoring every one seen so far.

    A **stage is one sequence**, trained with a single ``fit_sequence`` call at
    a fixed epoch budget, with **no weight reset between stages** — that is what
    makes the chain continual. Scoring never trains: ``score_fn`` is expected to
    clear transient state (``reset_context``) and leave weights untouched.

    Evaluation noise is seeded per evaluated task and held fixed across stages
    (``eval_noise_base + i``), so a change down a column of the matrix is the
    model moving, not the probe.
    """

    def __init__(
        self,
        n_trials: int = 1,
        noise_scale: float = 0.0,
        eval_noise_base: int = 10_000,
        rehearse: bool = True,
        n_rehearsed: Optional[int] = None,
        rehearsal_epochs: Optional[int] = None,
        criterion: Optional[float] = 0.95,
        max_epochs: int = 512,
        epochs_per_step: int = 1,
        pressure_threshold: float = 0.05,
    ):
        """
        Args:
            n_trials: cued-recall trials per measurement.
            noise_scale: Gaussian cue noise at probe time.
            eval_noise_base: base seed for per-task evaluation noise.
            rehearse: run the selective-retention phase after the chain.
            n_rehearsed: how many of the oldest tasks stay relevant. Defaults to
                half the chain, rounded down, and is clamped to ``1 …  T-1`` so
                both buckets are always non-empty.
            rehearsal_epochs: epoch budget for the re-presentation. Defaults to
                the chain's own per-stage budget, so rehearsal is one more
                ordinary presentation rather than a privileged one.
            pressure_threshold: ``avg_forgetting`` above which the chain counts
                as having saturated the substrate. Below it, selectivity is
                reported but flagged unscorable — nothing needed discarding.
        """
        self.n_trials = int(n_trials)
        self.noise_scale = float(noise_scale)
        self.eval_noise_base = int(eval_noise_base)
        self.rehearse = bool(rehearse)
        self.n_rehearsed = n_rehearsed
        self.rehearsal_epochs = rehearsal_epochs
        # Criterion-referenced exposure. `criterion=None` restores the fixed-budget
        # behaviour, which is how an older number is reproduced.
        self.criterion = criterion
        self.max_epochs = int(max_epochs)
        self.epochs_per_step = int(epochs_per_step)
        self.pressure_threshold = float(pressure_threshold)

    # -- selective retention -------------------------------------------------

    def _rehearsed_indices(self, n_tasks: int) -> List[int]:
        k = self.n_rehearsed if self.n_rehearsed is not None else n_tasks // 2
        k = int(max(1, min(n_tasks - 1, k)))
        return list(range(k))          # the oldest tasks: see the module note

    # -- main ---------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        datasets: Dict[str, Any],
        epochs: int = 300,
        chance_level: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the chain (and, if enabled, the rehearsal phase) on one model.

        Args:
            model: an already-constructed arm. It is trained in place and is
                **not** reset between tasks.
            datasets: ``{"sequences": {name: [word, ...]}, "embeddings":
                {name: ndarray}, "score_fn": callable}``. ``score_fn(i)`` scores
                task *i* and returns a scalar in [0, 1]; the benchmark owns the
                evaluation seed and calls it inside a fixed-seed block.
            epochs: per-stage training budget.
            chance_level: chance score for ``score_fn``, forwarded to
                ``retention_summary`` for the ratio-based statistics only.

        Returns:
            ``{"metrics": {...}, "series": {...}}`` — flat scalars for the
            scorecard, matrix and curves for the plots.
        """
        names: List[str] = list(datasets["sequences"].keys())
        embeddings: Dict[str, np.ndarray] = datasets["embeddings"]
        score_fn: Callable[[int], float] = datasets["score_fn"]
        n_tasks = len(names)
        if n_tasks < 2:
            raise ValueError(f"a chain needs at least 2 tasks, got {n_tasks}")

        exposures: List[Dict[str, Any]] = []

        def train_task(j: int) -> None:
            """Train task j to criterion on its own recall, not for a fixed budget.

            The chain's diagonal is "acquisition under load", and ACC / BWT /
            forgetting are all differences against it. Under a fixed budget the
            diagonal varies with how fast the arm happens to learn, so those
            differences mix "how much was forgotten" with "how much was ever
            acquired". Taking every task to the same criterion removes that
            confound and makes the matrix comparable across arms whose registry
            epoch defaults differ by 300x.
            """
            emb = embeddings[names[j]]
            if self.criterion is None:
                model.fit_sequence(emb, epochs=epochs)
                consolidate_if_supported(model, emb)
                exposures.append({"task": names[j], "epochs": int(epochs),
                                  "reached": None, "mode": "fixed"})
                return
            used, reached = 0, False
            while used < self.max_epochs:
                step = min(self.epochs_per_step, self.max_epochs - used)
                model.fit_sequence(emb, epochs=step)
                used += step
                if float(score_fn(j)) >= self.criterion:
                    reached = True
                    break
            consolidate_if_supported(model, emb)
            exposures.append({"task": names[j], "epochs": int(used),
                              "reached": bool(reached), "mode": "criterion"})

        def scored(i: int) -> float:
            np.random.seed(self.eval_noise_base + i)
            return float(score_fn(i))

        R = retention_matrix(n_tasks, train_task, lambda i, j: scored(i))
        summary = retention_summary(R, chance_level=chance_level)

        metrics: Dict[str, Any] = {
            "chain_n_tasks": n_tasks,
            "chain_epochs_to_criterion": (float(np.mean([e["epochs"] for e in exposures]))
                                          if exposures else float("nan")),
            "chain_criterion_reached": bool(
                exposures and all(e["reached"] is not False for e in exposures)),
            "chain_exposure_mode": ("criterion" if self.criterion is not None else "fixed"),
            "chain_avg_accuracy": summary["avg_accuracy"],
            "chain_avg_forgetting": summary["avg_forgetting"],
            "chain_backward_transfer": summary["backward_transfer"],
            "chain_avg_learning": summary["avg_learning"],
            "chain_intransigence": summary["intransigence"],
            "chain_learning_slope": summary["learning_slope"],
            "chain_retention_ratio": summary["retention_ratio"],
            "chain_stability_plasticity_index": summary["stability_plasticity_index"],
            # Guard: did the chain acquire anything at all? Every ratio-based
            # statistic above is vacuous if the diagonal is at chance, and a
            # vacuous SPI reads as a middling one rather than as missing.
            "chain_acquired": bool(
                summary["avg_learning"] > chance_level + 0.1 * (1.0 - chance_level)
            ),
            "chain_chance_level": float(chance_level),
        }
        series: Dict[str, Any] = {
            "retention_matrix": R.tolist(),
            "task_labels": names,
            "per_task_learned": summary["per_task_learned"],
            "per_task_final": summary["per_task_final"],
            "forgetting_by_gap": summary["forgetting_by_gap"],
            "gaps": list(range(1, n_tasks)),
            "exposures": exposures,
            "max_epochs": int(self.max_epochs) if self.criterion is not None else None,
        }

        if self.rehearse:
            metrics_sel, series_sel = self._run_rehearsal(
                model, names, embeddings, scored, R, summary,
                epochs=epochs,
            )
            metrics.update(metrics_sel)
            series.update(series_sel)

        return {"metrics": metrics, "series": series}

    def _run_rehearsal(
        self,
        model: Any,
        names: Sequence[str],
        embeddings: Dict[str, np.ndarray],
        scored: Callable[[int], float],
        R: np.ndarray,
        summary: Dict[str, Any],
        epochs: int,
    ) -> Any:
        """Re-present the still-relevant subset, then re-score every task.

        The comparison is a difference-in-differences: each task is scored
        against **its own** pre-rehearsal value, so the serial-position profile
        of the chain cancels and what is left is the effect of relevance alone.
        """
        n_tasks = len(names)
        rehearsed = self._rehearsed_indices(n_tasks)
        rehearsed_set = set(rehearsed)
        unrehearsed = [i for i in range(n_tasks) if i not in rehearsed_set]
        reh_epochs = int(self.rehearsal_epochs if self.rehearsal_epochs is not None
                         else epochs)

        before = np.array([R[n_tasks - 1, i] for i in range(n_tasks)])

        for i in rehearsed:
            emb = embeddings[names[i]]
            model.fit_sequence(emb, epochs=reh_epochs)
            consolidate_if_supported(model, emb)

        after = np.array([scored(i) for i in range(n_tasks)])
        delta = after - before

        gain = float(np.mean(delta[rehearsed]))
        drift = float(np.mean(delta[unrehearsed]))
        pressure = float(summary["avg_forgetting"])
        under_pressure = bool(pressure > self.pressure_threshold)

        metrics = {
            # Difference-in-differences. Positive = the substrate was
            # reallocated toward the material still in use.
            "select_selectivity": gain - drift,
            "select_rehearsed_gain": gain,
            "select_unrehearsed_drift": drift,
            "select_rehearsed_recall": float(np.mean(after[rehearsed])),
            "select_unrehearsed_recall": float(np.mean(after[unrehearsed])),
            "select_n_rehearsed": len(rehearsed),
            "select_rehearsal_epochs": reh_epochs,
            # Guards.
            "select_pressure": pressure,
            "select_under_pressure": under_pressure,
            # If re-presentation moves nothing, the arm had no plasticity left
            # to allocate and a zero selectivity says nothing about *selection*.
            "select_rehearsal_effective": bool(abs(gain) > 1e-6),
        }
        series = {
            "selective_retention": {
                "rehearsed": rehearsed,
                "unrehearsed": unrehearsed,
                "before": before.tolist(),
                "after": after.tolist(),
                "delta": delta.tolist(),
                "task_labels": list(names),
            }
        }
        return metrics, series


def plot_chain_axes(
    series: Dict[str, Any],
    metrics: Dict[str, Any],
    path: str,
    baseline_epochs: Optional[float] = None,
    max_epochs: Optional[int] = None,
) -> None:
    """The chain's three axes from its stored series, so a saved run can be replotted.

    Left: forgetting gradient against interposed load. Middle: the diagonal
    (acquired at criterion, with everything earlier stored) against the final
    row. Right: what that acquisition **cost** -- epochs to criterion at every
    chain position, which is the exposure-against-retention pairing the One-shot
    capacity promises: a one-shot arm keeps a flat line at 1 while the diagonal
    holds. Censored tasks (never reached criterion inside the budget) are drawn
    open, at the budget. ``baseline_epochs`` is the fresh-model exposure from
    ``presentation_duration`` (``convergence_epochs``), drawn as a reference;
    ``None`` means that section was itself censored.
    """
    import matplotlib.pyplot as plt

    names = list(series.get("task_labels") or [])
    gaps = series.get("gaps") or []
    exposures = series.get("exposures") or []
    idx = list(range(len(names)))
    n_panels = 3 if exposures else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2))

    axes[0].plot(gaps, series.get("forgetting_by_gap") or [], '-o', color="crimson")
    axes[0].set_xlabel("Interposed tasks (gap g)")
    axes[0].set_ylabel("Retained fraction R[i+g, i] / R[i, i]")
    axes[0].set_title("Forgetting gradient (load axis)")
    axes[0].set_xticks(gaps)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(1.0, ls=":", color="grey", lw=0.8)

    axes[1].plot(idx, series.get("per_task_learned") or [], '-o',
                 color="tab:blue", label="learned (diagonal)")
    axes[1].plot(idx, series.get("per_task_final") or [], '-x',
                 color="tab:red", label="final (bottom row)")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("cued recall (mean recall rate)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title(
        f"Plasticity vs stability  "
        f"(intrans={metrics.get('chain_intransigence', float('nan')):+.2f}, "
        f"SPI={metrics.get('chain_stability_plasticity_index', float('nan')):.2f})")
    axes[1].legend(fontsize=8)

    if exposures:
        ax = axes[2]
        ep = [float(e["epochs"]) for e in exposures]
        reached = [e.get("reached") is not False for e in exposures]
        budget = max_epochs or series.get("max_epochs") or (max(ep) if not all(reached) else None)
        ax.plot(idx, ep, '-', color="tab:purple", lw=1.2, zorder=1)
        hit = [i for i in idx if reached[i]]
        miss = [i for i in idx if not reached[i]]
        if hit:
            ax.plot(hit, [ep[i] for i in hit], 'o', color="tab:purple",
                    label="reached criterion", zorder=2)
        if miss:
            ax.plot(miss, [ep[i] for i in miss], 'o', mfc="white", mec="tab:purple",
                    label="censored at budget", zorder=2)
        if baseline_epochs is not None:
            ax.axhline(float(baseline_epochs), ls=":", color="grey", lw=0.9,
                       label=f"fresh list ({float(baseline_epochs):g})")
        if budget is not None:
            ax.axhline(float(budget), ls="--", color="lightgrey", lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(idx)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("epochs to criterion (log)")
        mode = str(metrics.get("chain_exposure_mode", exposures[0].get("mode", "")))
        mean_ep = metrics.get("chain_epochs_to_criterion", float(np.mean(ep)))
        ax.set_title(f"Exposure under load ({mode}; mean={mean_ep:.1f})")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
