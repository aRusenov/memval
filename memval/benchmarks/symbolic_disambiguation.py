"""Scoring for symbolic overlapping sequences.

Implements the metric fixes of `docs/disambiguation_design.md` sec 5 for the
symbolic modality, including the two that the spatial section still cannot
produce because it is capped at two episodes:

  5.1  Score at the DIVERGENCE POINT, not over the episode. During the shared
       stretch every episode predicts the same successor, so an ambiguous
       representation scores *correct* there -- and that free fraction grows
       with shared-stretch length, making the metric easier exactly as axis C
       makes the task harder. Shared-stretch accuracy is reported as a control
       and never folded into the headline.
  5.2  MARGIN, not nearest-of-two. Target similarity minus best-competitor
       similarity keeps resolving after accuracy has floored at chance.
  5.3  Confusion STRUCTURE, not confusion rate. With N > 2 confusable episodes,
       *which* wrong episode the model falls into is informative: falling into
       the contextually nearest neighbour is evidence about context, falling
       into a random one is not. Reported as a confusion matrix plus a
       context-graded confusion index.

PROBE PROTOCOL -- and it differs from the spatial section, deliberately.

This is a **single-step probe, not an autoregressive rollout.** The prefix is
delivered through `observe` (state moves, plasticity does not) for arms that
declare `StatePrimeable`, then exactly ONE `predict_next` is taken at the
divergence step and scored against the N candidate successors. Nothing is fed
back; no rollout happens.

`spatial_disambiguation.py` instead rolls forward from the zone to the end of
the route, feeding each prediction back as the next input. Its
`divergence_accuracy` is therefore read at the end of a chain of self-generated
inputs, so it is confounded with ordinary rollout drift; that is what its
`shared_stretch_error` control exists to isolate. The single-step probe here has
no such confound, because there is nothing to drift.

Consequence: **do not compare divergence accuracy across the two modalities as
if it were one number.** They answer different questions. The symbolic one is
"given the history, is the branch decision right"; the spatial one is "given the
history, is the branch decision right AND does the trajectory survive being
generated". Within a modality both are comparable across arms.

The single-step form is the one sec 5.1 asks for -- score at the divergence
point, and keep the recovery profile separate rather than folded in.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.base import HippocampalModel
from ..models.capabilities import supports_priming
from .base import Benchmark


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0


class SymbolicDisambiguationBenchmark(Benchmark):
    """Forced choice among N episodes' successors at the divergence step.

    Note on ``n_trials``: nothing in the probe is stochastic -- no cue noise, no
    sampling -- so for a deterministic arm every trial repeats the same
    computation and the reported proportions are unchanged. It scales the
    confusion counts and the cost, not the precision. Raise it only for an arm
    whose ``predict_next`` is itself stochastic.
    """

    def evaluate(
        self,
        model: HippocampalModel,
        episode_set: Dict[str, Any],
        n_trials: int = 1,
        fit_epochs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        inputs: List[np.ndarray] = episode_set["inputs"]
        n_ep = episode_set["n_episodes"]
        n_content = episode_set["n_content"]
        se = episode_set["shared_end"]
        ss = episode_set["shared_start"]
        ze = episode_set["zone_end"]
        div = episode_set["divergence_step"]      # == se - 1

        # The N candidate successors: episode i's first divergent observation.
        # Compared in the CONTENT channel only -- the discriminator block is
        # zero at se under withdrawal, and including it would let a model score
        # by reproducing a cue it was handed rather than by choosing a branch.
        successors = np.array([X[se, :n_content] for X in inputs], dtype=float)

        primed = supports_priming(model)

        # ---- train ----------------------------------------------------------
        model.reset_context()
        for X in inputs:
            if fit_epochs is not None:
                model.fit_sequence(X, epochs=fit_epochs)
            else:
                model.fit_sequence(X)

        # ---- probe at the divergence step ------------------------------------
        confusion = np.zeros((n_ep, n_ep), dtype=float)
        margins, correct_flags = [], []
        for trial in range(max(1, n_trials)):
            for e, X in enumerate(inputs):
                model.reset_context()
                # Deliver the history. Without this an arm sees only X[div],
                # which is IDENTICAL across episodes under withdrawal, so the
                # forced choice is at chance by construction.
                if primed and div > 0:
                    model.observe_sequence(X[:div])
                pred = np.asarray(model.predict_next(X[div]), dtype=float)
                pred_content = pred[:n_content]

                sims = np.array([_cos(pred_content, s) for s in successors])
                chosen = int(np.argmax(sims))
                confusion[e, chosen] += 1.0

                # Target minus BEST COMPETITOR, the readout that resolves below
                # the accuracy floor. Ties count as failures: argmax would
                # silently award a tie to the lower index.
                competitors = np.delete(sims, e)
                margins.append(float(sims[e] - competitors.max()))
                correct_flags.append(bool(sims[e] > competitors.max()))

        n_probes = max(1, n_trials) * n_ep
        confusion /= max(1.0, confusion.sum(axis=1, keepdims=True).max())
        row_tot = confusion.sum(axis=1, keepdims=True)
        row_tot[row_tot == 0] = 1.0
        confusion_norm = confusion / row_tot

        # ---- shared-stretch control ------------------------------------------
        # Teacher-forced one-step accuracy where every episode agrees. Near
        # ceiling for any arm that learned the material at all; a low value here
        # voids the divergence numbers, because the arm never learned the
        # corridor it is supposed to be carrying a cue across.
        shared_hits, shared_tot = 0, 0
        if se - ss > 1:
            X0 = inputs[0]
            model.reset_context()
            for t in range(ss, se - 1):
                p = np.asarray(model.predict_next(X0[t]), dtype=float)[:n_content]
                tgt = X0[t + 1, :n_content]
                # nearest among the shared stretch's own successors
                cands = np.array([X0[u, :n_content] for u in range(ss + 1, se)])
                sims = np.array([_cos(p, c) for c in cands])
                shared_hits += int(np.argmax(sims) == (t + 1 - (ss + 1)))
                shared_tot += 1
        shared_acc = shared_hits / shared_tot if shared_tot else float("nan")

        # ---- context-graded confusion index (sec 5.3) -------------------------
        # Correlate P(fall into j | cued with i) against cos(disc_i, disc_j)
        # over the off-diagonal. Positive means errors go to the CONTEXTUALLY
        # NEAREST neighbour, which is evidence the discriminator is represented
        # at all -- a very different result from uniform confusion. Undefined
        # for N == 2 (a single off-diagonal pair has no variance to correlate).
        cgci = float("nan")
        disc_enc = episode_set.get("discriminator_encoder")
        if n_ep > 2 and disc_enc is not None:
            D = np.array([disc_enc.embeddings[i] for i in range(n_ep)], dtype=float)
            errs, sims_d = [], []
            for i in range(n_ep):
                for j in range(n_ep):
                    if i == j:
                        continue
                    errs.append(confusion_norm[i, j])
                    sims_d.append(_cos(D[i], D[j]))
            if np.std(errs) > 1e-12 and np.std(sims_d) > 1e-12:
                cgci = float(np.corrcoef(errs, sims_d)[0, 1])

        acc = float(np.mean(correct_flags))
        return {
            "divergence_accuracy": acc,
            "divergence_margin": float(np.mean(margins)),
            "shared_stretch_accuracy": shared_acc,
            "confusion_matrix": confusion_norm.tolist(),
            "context_graded_confusion_index": cgci,
            "chance_level": 1.0 / n_ep,
            "n_episodes": n_ep,
            "delay": episode_set["delay"],
            "category_variance": episode_set.get("category_variance"),
            "discriminator_similarity":
                episode_set.get("discriminator_similarity", {}).get("mean"),
            "state_primed": bool(primed),
            "n_probes": n_probes,
            # Self-describing protocol, so a row is never read against the
            # spatial section's autoregressive number by accident.
            "probe": "single_step",
        }

    def sweep(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        grid: List[Dict[str, Any]],
        n_trials: int = 1,
        fit_epochs: Optional[int] = None,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Run `evaluate` over a grid of generator settings, one fresh arm each.

        Each grid entry is a kwargs dict for `SymbolicOverlapGenerator.generate`.
        A fresh model per row is deliberate: carrying one arm across rows would
        make every row after the first a continual-learning result, which is a
        different capacity.
        """
        from ..generators.symbolic_overlap import SymbolicOverlapGenerator

        gen = SymbolicOverlapGenerator(seed=seed)
        records: List[Dict[str, Any]] = []
        for entry in grid:
            episode_set = gen.generate(seed=seed, **entry)
            kw = dict(model_kwargs)
            kw["n_features"] = episode_set["inputs"][0].shape[1]
            model = model_class(**kw)
            row = self.evaluate(model, episode_set, n_trials=n_trials,
                                fit_epochs=fit_epochs)
            row.update({k: v for k, v in entry.items()
                        if k in ("n_episodes", "category_variance",
                                 "zone_fraction", "zone_offset",
                                 "shared_fraction", "shared_position",
                                 "total_length", "discriminator_mode")})
            # Realised geometry, so a row is labelled by what it actually was
            # rather than by the fractions that were asked for.
            row["cue_duration"] = int(episode_set["zone_end"]
                                      - episode_set["zone_start"])
            row["prefix_len"] = int(episode_set["shared_start"])
            row["shared_len"] = int(episode_set["shared_end"] - episode_set["shared_start"])
            row["endogenous"] = bool(episode_set.get("endogenous", False))
            records.append(row)
        return records
