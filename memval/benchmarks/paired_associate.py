"""AB/AC paired-associate interference — the symbolic twin of ``tmaze_reversal``.

``multiple_sequences`` trains list A (fruit) then list B (animals): disjoint
cues, disjoint targets, disjoint categories. Whatever damage it finds is generic
weight interference from an equal quantity of new material — the substrate
filling up. It cannot produce the failure the continual-learning literature is
actually named after, because nothing in list B ever competes for a cue that
list A already owns.

This section supplies the missing paradigm (McCloskey & Cohen 1989, after the
AB/AC transfer design of Barnes & Underwood 1959). The same cues are re-paired
with new targets:

    phase 1   a_i -> b_i        the original association
    phase 2   a_i -> c_i        the same cue, a new target

After phase 2, ``b_i`` is no longer the right answer. Returning it is an
**intrusion**, not retention — which makes this section a read-out on both
halves of stability–plasticity at once, from one protocol:

    plasticity   did ``a_i -> c_i`` get in, and how fast?
    stability    what happened to ``a_i -> b_i`` while it did?

and, because the correct behaviour here is to *stop* emitting a learned
response, it is one of the two places in the suite where forgetting is scored as
success rather than as damage (the other is ``tmaze_reversal``).

The control condition
---------------------
Phase 2 trains ``d_i -> e_i`` instead: fresh cues, fresh targets, identical pair
count, identical trial count, identical epoch budget. Damage to AB under the
control is generic interference; damage under AB/AC is generic interference
*plus* cue competition. The difference isolates the second, and it is the
quantity ``multiple_sequences`` structurally cannot measure — that section is
this control condition, with no experimental arm to compare it against.

Response coding
---------------
Every probe is scored into one of three buckets — the old target, the new
target, or neither — because "AB recall fell" is ambiguous between the two
failures that matter: the cue now retrieves ``c_i`` (a clean overwrite, which is
correct after phase 2) or it retrieves noise (the association was destroyed
rather than updated). ``other_rate`` separates them.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import Benchmark
from .continual_chain import consolidate_if_supported

#: Response buckets, in the order they are reported.
RESPONSE_BUCKETS = ("old", "new", "other")


class PairedAssociateBenchmark(Benchmark):
    """AB/AC (and AB/DE control) paired-associate interference.

    A **pair is a two-element sequence** and a **trial is one pass over the
    whole pair list**, in a fixed order, one ``fit_sequence`` call per pair.
    Pairs are trained separately rather than concatenated into one long list:
    concatenation would additionally teach ``b_i -> a_{i+1}``, which is a serial
    list, not a set of independent associations.

    Probes never train. Each cue is presented with the same Gaussian noise the
    rest of the symbolic suite uses, and the prediction is decoded to a word.
    """

    def __init__(
        self,
        n_trials: int = 1,
        noise_scale: float = 0.0,
        criterion: float = 0.75,
        eval_seed: int = 20_000,
    ):
        """
        Args:
            n_trials: probe repetitions per measurement.
            noise_scale: Gaussian cue noise at probe time.
            criterion: new-association recall counted as "acquired", for
                trials-to-criterion.
            eval_seed: fixed probe-noise seed, re-applied before every
                measurement so a change across trials is the model moving.
        """
        self.n_trials = int(n_trials)
        self.noise_scale = float(noise_scale)
        self.criterion = float(criterion)
        self.eval_seed = int(eval_seed)

    # -- probing -------------------------------------------------------------

    def _probe(
        self,
        model: Any,
        cues: Sequence[str],
        old_targets: Sequence[str],
        new_targets: Optional[Sequence[str]],
        encoder: Any,
        decoder: Any,
    ) -> Dict[str, float]:
        """Cue each pair and bucket the decoded response.

        Returns rates for ``old`` / ``new`` / ``other``, which sum to 1.0.
        ``new`` is 0.0 when ``new_targets`` is None (before phase 2 defines them).
        """
        np.random.seed(self.eval_seed)
        context = np.array([1.0])
        counts = {b: 0 for b in RESPONSE_BUCKETS}
        total = 0

        for _ in range(self.n_trials):
            for k, cue in enumerate(cues):
                if hasattr(model, "reset_context"):
                    model.reset_context()
                model.current_t = 0
                clean = encoder.encode([cue])[0]
                noisy = clean + np.random.normal(0, self.noise_scale,
                                                 encoder.embedding_dim)
                pred = model.predict_next(noisy, current_context=context)
                word = decoder.decode(pred, top_k=1)[0]
                if word == old_targets[k]:
                    counts["old"] += 1
                elif new_targets is not None and word == new_targets[k]:
                    counts["new"] += 1
                else:
                    counts["other"] += 1
                total += 1

        return {b: counts[b] / total if total else 0.0 for b in RESPONSE_BUCKETS}

    # -- training ------------------------------------------------------------

    def _train_pass(
        self,
        model: Any,
        pair_embeddings: List[np.ndarray],
        epochs: int,
    ) -> None:
        """One pass over the pair list: one ``fit_sequence`` call per pair."""
        for emb in pair_embeddings:
            if hasattr(model, "reset_context"):
                model.reset_context()
            model.fit_sequence(emb, epochs=epochs)
            consolidate_if_supported(model, emb)

    @staticmethod
    def _pair_embeddings(cues: Sequence[str], targets: Sequence[str],
                         encoder: Any) -> List[np.ndarray]:
        return [encoder.encode([c, t]) for c, t in zip(cues, targets)]

    def _trials_to_criterion(self, curve: Sequence[float]) -> Tuple[float, bool]:
        """First trial index (1-based) at or above criterion; NaN if never."""
        for t, v in enumerate(curve):
            if v >= self.criterion:
                return float(t + 1), True
        return float("nan"), False

    # -- main ----------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        datasets: Dict[str, Any],
        condition: str = "abac",
        ab_trials: int = 3,
        phase2_trials: int = 6,
        epochs: int = 100,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run one condition end to end on a fresh model.

        Args:
            model: an already-constructed arm, trained in place. Each condition
                needs its **own** instance — phase 1 is common to both, but the
                two phase 2s are mutually contaminating.
            datasets: ``{"cues_A", "targets_B", "targets_C", "cues_D",
                "targets_E", "encoder", "decoder"}``.
            condition: ``"abac"`` (phase 2 = A–C, shared cue) or ``"control"``
                (phase 2 = D–E, disjoint cue).
            ab_trials: passes over the AB list in phase 1.
            phase2_trials: passes over the phase-2 list, measured after each.
            epochs: epochs per ``fit_sequence`` call.

        Returns:
            Flat metrics plus per-trial curves. Keys are prefixed
            ``pa_<condition>_``.
        """
        if condition not in ("abac", "control"):
            raise ValueError(f"unknown condition {condition!r}; "
                             f"expected 'abac' or 'control'")

        enc, dec = datasets["encoder"], datasets["decoder"]
        cues_A: List[str] = list(datasets["cues_A"])
        targets_B: List[str] = list(datasets["targets_B"])
        targets_C: List[str] = list(datasets["targets_C"])

        if condition == "abac":
            p2_cues, p2_targets = cues_A, targets_C
        else:
            p2_cues, p2_targets = list(datasets["cues_D"]), list(datasets["targets_E"])

        ab_pairs = self._pair_embeddings(cues_A, targets_B, enc)
        p2_pairs = self._pair_embeddings(p2_cues, p2_targets, enc)

        # `new_targets` is always the C list, in both conditions. Under the
        # control the C words are never trained, so the `new` bucket there is a
        # floor check on the decoder rather than a measurement -- it should stay
        # at ~0, and a non-zero value means the vocabulary geometry, not the
        # model, is producing the AB/AC "overwrite".
        def probe() -> Dict[str, float]:
            return self._probe(model, cues_A, targets_B, targets_C, enc, dec)

        # --- phase 1: A -> B --------------------------------------------------
        for _ in range(ab_trials):
            self._train_pass(model, ab_pairs, epochs)
        baseline = probe()

        # --- phase 2: A -> C, or D -> E under the control ---------------------
        curves: Dict[str, List[float]] = {b: [baseline[b]] for b in RESPONSE_BUCKETS}
        for _ in range(phase2_trials):
            self._train_pass(model, p2_pairs, epochs)
            r = probe()
            for b in RESPONSE_BUCKETS:
                curves[b].append(r[b])

        final = {b: curves[b][-1] for b in RESPONSE_BUCKETS}
        # Curves include the pre-phase-2 point at index 0; trials-to-criterion
        # counts phase-2 trials, so it is measured on the tail.
        ttc, reached = self._trials_to_criterion(curves["new"][1:])

        p = f"pa_{condition}"
        metrics: Dict[str, Any] = {
            f"{p}_ab_recall_baseline": baseline["old"],
            f"{p}_ab_recall_final": final["old"],
            f"{p}_ab_retention": (final["old"] / baseline["old"]
                                  if baseline["old"] > 0 else float("nan")),
            f"{p}_ac_recall_final": final["new"],
            f"{p}_ac_trials_to_criterion": ttc,
            f"{p}_ac_criterion_reached": reached,
            f"{p}_other_rate_final": final["other"],
            # Guard: was there an AB association to interfere with at all?
            f"{p}_ab_acquired": bool(baseline["old"] >= self.criterion),
            f"{p}_phase2_trials": int(phase2_trials),
            f"{p}_ab_trials": int(ab_trials),
            f"{p}_epochs": int(epochs),
        }
        series = {
            "condition": condition,
            "trials": list(range(phase2_trials + 1)),
            **{f"{b}_rate": curves[b] for b in RESPONSE_BUCKETS},
        }
        return {"metrics": metrics, "series": series}
