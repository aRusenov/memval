"""Cue AVAILABILITY: pattern completion when a whole modality is absent from the cue.

The third probe property of the paper's figure, beside noise (`noise_invariance`)
and fragment (`cue_masking`). Each studied item is a two-block vector
(`MultimodalEncoder`): its symbolic embedding and its OWN pure tone -- every
studied item has a distinct tone, so the audio block is a second, independent
code for the same item. Two conditions:

  single   one 10-item sequence.
  multi    K disjoint 10-item sequences (one category each). No overlap on
           either block -- overlap is Sequence disambiguation's axis.

Three cues, every probe taken with the same trained model:

  both       the full [symbol | audio] vector: the reference.
  symbolic   audio removed (audio block zeroed).
  audio      symbol removed (symbolic block zeroed).

Each cue is probed two ways, as P17 promises:

  cued       every item of every sequence (except the last, which has no
             successor) is cued in turn; the read-out is exact next-item
             recall, averaged over all item-audio pairs.
  rollout    each sequence's FIRST item is cued, then the arm runs on its own
             output, fed back verbatim ("raw" feedback, as in `noise_invariance`;
             the cue is manipulated at step 0 only); the read-out is
             exact next-item recall at every step, averaged over steps and
             sequences.

The next item is always decoded from the prediction's SYMBOLIC block against the
whole suite vocabulary (chance 1/|vocab|), whatever the cue carried, so the three
cues are scored on the same read-out and an audio-only cue has to cross
modalities to succeed. As a diagnostic the prediction's AUDIO block is also
decoded against the studied items' tones (`<cue>_audio_recall`, chance
1/n_studied): for an item-only cue that is completion of the missing modality.
Margins (target minus best competitor) are reported beside every cued accuracy
because they resolve below the accuracy floor.

Until 2026-09-26 the tone belonged to the SEQUENCE. A tone-only cue then carried
no position, so it could only be scored on membership, and a rollout launched
from it was a fixed point that repeated one member (membership 1.00, transition
validity 0.00 on AHN) -- not comparable with the item cues' exact recall.

Exposure: the model is trained to criterion on FULL-cue recall across all
sequences (`avail_<cond>_epochs_to_criterion`), ingesting the sequences
round-robin (one epoch of each in turn) so that the multi condition does not
import blocked-ingestion forgetting, which is Continual retention's axis.

Every metric is `avail_<condition>_<cue>_<readout>`; guards and config carry the
`avail_` prefix without a condition.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from memval.encoders.multimodal import MODALITIES, MultimodalEncoder
from memval.encoders.symbolic import SymbolicEncoder
from memval.benchmarks.exposure import epochs_to_criterion

CONDITIONS = ("single", "multi")
#: (category, tone Hz) for the multi condition, in order; `single` uses the first.
#: Categories of the multi condition, in order; `single` uses the first. 10 items
#: per sequence since 2026-09-26 (were 4, at ceiling on every arm); `color` was
#: replaced by `number` then, because data/vocab.json has only 7 colours.
DEFAULT_MATERIAL = ("fruit", "animal", "number")


def _make_model(model_class: Type, enc: MultimodalEncoder,
                model_kwargs: Dict[str, Any], epochs: int):
    kwargs = dict(model_kwargs)
    kwargs["n_epochs"] = epochs
    kwargs["epochs"] = epochs
    if "encoder" in model_class.__init__.__code__.co_varnames:
        return model_class(encoder=enc, n_features=enc.embedding_dim, **kwargs)
    return model_class(n_features=enc.embedding_dim, **kwargs)


def _fit_round_robin(model: Any, encoded: Sequence[np.ndarray], epochs: int) -> None:
    """One epoch of every sequence in turn, `epochs` times.

    The EP family ignores a call-level ``epochs=`` and loops its own
    ``n_epochs``, so the count is pinned on the instance for every call.
    """
    for _ in range(int(epochs)):
        for X in encoded:
            if hasattr(model, "n_epochs"):
                model.n_epochs = 1
            model.fit_sequence(X, epochs=1)


def _margin(sims: np.ndarray, target_idx: Sequence[int]) -> float:
    """Best target-set cosine minus best cosine outside the target set."""
    mask = np.zeros(len(sims), dtype=bool)
    mask[list(target_idx)] = True
    inside = sims[mask].max() if mask.any() else -1.0
    outside = sims[~mask].max() if (~mask).any() else -1.0
    return float(inside - outside)


def _predict(model: Any, cue: np.ndarray, t: int) -> np.ndarray:
    model.current_t = t
    return np.asarray(model.predict_next(cue, current_context=np.array([1.0])), dtype=float)


def _feedback(pred: np.ndarray, enc: MultimodalEncoder, mode: str) -> np.ndarray:
    if mode == "quantized":
        return enc.embeddings[enc.word_to_idx[enc.decode_item(pred)]]
    if mode == "l2":
        n = float(np.linalg.norm(pred))
        return pred / n if n > 1e-12 else pred
    return pred


def probe_condition(model: Any, enc: MultimodalEncoder, feedback_mode: str = "raw",
                    renormalize: bool = True) -> Dict[str, float]:
    """All read-outs for one trained model over its sequences.

    Cued: every item except each sequence's last, averaged over all of them.
    Rollout: each sequence's first item only, averaged over steps and sequences.
    """
    out: Dict[str, List[float]] = {}

    def push(key: str, val: float) -> None:
        out.setdefault(key, []).append(float(val))

    def score(m: str, pred: np.ndarray, target: str, prefix: str) -> None:
        push(f"{m}_{prefix}", enc.decode_item(pred) == target)
        if prefix == "recall":
            push(f"{m}_margin", _margin(enc.item_similarities(pred),
                                        [enc.word_to_idx[target]]))
            push(f"{m}_audio_recall", enc.tone_of_prediction(pred) == target)

    for words in enc.sequences.values():
        L = len(words)
        for m in MODALITIES:
            # ---- cued: every item-audio pair that has a successor ----------
            for i in range(L - 1):
                if hasattr(model, "reset_context"):
                    model.reset_context()
                pred = _predict(model, enc.cue(words[i], m, renormalize), i)
                score(m, pred, words[i + 1], "recall")
            # ---- rollout: the first item only, then the arm's own output ---
            if hasattr(model, "reset_context"):
                model.reset_context()
            current = enc.cue(words[0], m, renormalize)
            for i in range(L - 1):
                pred = _predict(model, current, i)
                score(m, pred, words[i + 1], "rollout")
                current = _feedback(pred, enc, feedback_mode)
    return {k: float(np.mean(v)) for k, v in out.items()}


def build_material(vocab: Dict[str, str], n_sequences: int, seq_len: int,
                   material: Sequence[str] = DEFAULT_MATERIAL,
                   embedding_dim: int = 100, category_variance: float = 0.2,
                   n_bins: int = 100, tone_sigma: float = 4.0,
                   min_freq: float = 200.0, max_freq: float = 600.0,
                   audio_gain: float = 1.0, seed: int = 42) -> MultimodalEncoder:
    """Disjoint sequences, one category each, every studied item with its own tone.

    Tones are laid out for the FULL material (len(material) x seq_len items)
    whatever ``n_sequences`` is, so the single condition uses exactly the tones
    its items carry in the multi condition. Frequencies are evenly spaced,
    centred in their slots so none sits on the band edge, and assigned to items
    in a seeded shuffled order: neighbouring frequencies (the only pairs with any
    tone overlap) then land on unrelated items rather than on sequence
    neighbours, where they would make some transitions easier than others.
    """
    if n_sequences > len(material):
        raise ValueError(f"at most {len(material)} sequences are configured, "
                         f"got n_sequences={n_sequences}")
    sym = SymbolicEncoder(vocab, embedding_dim=embedding_dim,
                          category_variance=category_variance, seed=seed)
    all_seqs: Dict[str, List[str]] = {}
    for cat in material:
        words = [w for w, c in vocab.items() if c == cat][:seq_len]
        if len(words) < seq_len:
            raise ValueError(f"category {cat!r} has only {len(words)} words, need {seq_len}")
        all_seqs[cat] = words
    items = [w for ws in all_seqs.values() for w in ws]
    n = len(items)
    step = (max_freq - min_freq) / n
    freqs = min_freq + step * (np.arange(n) + 0.5)
    order = np.random.default_rng(seed).permutation(n)
    item_tones = {w: float(freqs[order[j]]) for j, w in enumerate(items)}
    seqs = dict(list(all_seqs.items())[:n_sequences])
    tones = {w: item_tones[w] for ws in seqs.values() for w in ws}
    return MultimodalEncoder(sym, tones, seqs, n_bins=n_bins, tone_sigma=tone_sigma,
                             min_freq=min_freq, max_freq=max_freq, audio_gain=audio_gain)


def run_cue_availability(
    model_class: Type,
    model_kwargs: Optional[Dict[str, Any]] = None,
    vocab: Optional[Dict[str, str]] = None,
    seq_len: int = 10,
    n_sequences: int = 3,
    n_bins: int = 100,
    tone_sigma: float = 4.0,
    audio_gain: float = 1.0,
    feedback_mode: str = "raw",
    renormalize: bool = True,
    exposure: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    run_dir: Optional[str] = None,
    metrics_filename: str = "cue_availability_metrics.json",
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    model_kwargs = dict(model_kwargs or {})
    if vocab is None:
        from memval.benchmarks.symbolic_pipeline import load_vocab
        vocab = load_vocab()
    policy = exposure or {"mode": "criterion", "epochs": None, "criterion": 0.95,
                          "max_epochs": 512, "fallback_epochs": 300}
    metrics: Dict[str, Any] = {
        "avail_seq_len": int(seq_len), "avail_n_sequences": int(n_sequences),
        "avail_audio_gain": float(audio_gain), "avail_feedback_mode": feedback_mode,
        "avail_n_bins": int(n_bins), "avail_tone_sigma": float(tone_sigma),
        "avail_chance_item": 1.0 / len(vocab),
        "avail_exposure_mode": policy["mode"],
    }
    series: Dict[str, Any] = {"conditions": {}}

    for cond in CONDITIONS:
        K = 1 if cond == "single" else int(n_sequences)
        np.random.seed(seed)
        enc = build_material(vocab, K, seq_len, n_bins=n_bins, tone_sigma=tone_sigma,
                             audio_gain=audio_gain, seed=seed)
        encoded = [enc.encode(w) for w in enc.sequences.values()]

        def _score_full(m, _enc=enc) -> float:
            return probe_condition(m, _enc, feedback_mode, renormalize)["both_recall"]

        if policy["mode"] == "fixed":
            n_ep = int(policy["epochs"])
            model = _make_model(model_class, enc, model_kwargs, n_ep)
            _fit_round_robin(model, encoded, n_ep)
            exp_rec = {"epochs": n_ep, "reached": _score_full(model) >= policy["criterion"]}
        else:
            res = epochs_to_criterion(
                lambda _e=enc: _make_model(model_class, _e, model_kwargs,
                                           int(policy["max_epochs"])),
                lambda m, ep, _x=encoded: _fit_round_robin(m, _x, ep),
                _score_full,
                criterion=float(policy["criterion"]),
                max_epochs=int(policy["max_epochs"]))
            model, exp_rec = res["model"], {"epochs": res["epochs"], "reached": res["reached"]}

        r = probe_condition(model, enc, feedback_mode, renormalize)
        for k, v in r.items():
            metrics[f"avail_{cond}_{k}"] = v
        metrics[f"avail_{cond}_epochs_to_criterion"] = float(exp_rec["epochs"])
        metrics[f"avail_{cond}_criterion_reached"] = bool(exp_rec["reached"])
        # Chance for the audio-block diagnostic: 1 / number of studied tones.
        metrics[f"avail_{cond}_chance_tone"] = 1.0 / len(enc.tone_names)
        metrics[f"avail_{cond}_tone_cosine_max"] = enc.describe()["tone_cosine_max"]
        series["conditions"][cond] = {
            "readouts": r, "sequences": enc.sequences,
            "tones": dict(enc.item_tones),
            "geometry": enc.describe(),
            "exposure": exp_rec,
        }
        print(f"  [{cond}] epochs={exp_rec['epochs']} reached={exp_rec['reached']}  "
              + "  ".join(f"{k}={v:.2f}" for k, v in r.items()))

    # Guard: the section poses its question only if the full cue was acquired.
    metrics["avail_acquired"] = bool(all(
        metrics[f"avail_{c}_both_recall"] >= 0.5 for c in CONDITIONS))
    metrics["avail_epochs_to_criterion"] = float(np.mean(
        [metrics[f"avail_{c}_epochs_to_criterion"] for c in CONDITIONS]))
    metrics["avail_criterion_reached"] = bool(all(
        metrics[f"avail_{c}_criterion_reached"] for c in CONDITIONS))

    if run_dir is not None:
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
        _plot(series, metrics, run_name or model_class.__name__, run_dir)
        with open(os.path.join(run_dir, metrics_filename), "w") as f:
            json.dump({"metadata": {"section": "cue_availability", "model": run_name,
                                    "seed": seed},
                       "metrics": metrics, "series": series}, f, indent=2, default=float)
    return {"metrics": metrics, "series": {"cue_availability": series}}


def _plot(series: Dict[str, Any], metrics: Dict[str, Any], model_name: str,
          run_dir: str) -> None:
    import matplotlib.pyplot as plt

    labels = {"both": "both", "symbolic": "symbolic only\n(audio removed)",
              "audio": "audio only\n(symbol removed)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, cond in zip(axes, CONDITIONS):
        r = series["conditions"][cond]["readouts"]
        K = int(metrics["avail_n_sequences"]) if cond == "multi" else 1
        x = np.arange(len(MODALITIES))
        w = 0.36
        cued = [r.get(f"{m}_recall", np.nan) for m in MODALITIES]
        roll = [r.get(f"{m}_rollout", np.nan) for m in MODALITIES]
        ax.bar(x - w / 2, cued, w, color="tab:blue",
               label="cued recall (every item, averaged)")
        ax.bar(x + w / 2, roll, w, color="tab:red",
               label=f"rollout from each sequence's first item "
                     f"({metrics.get('avail_feedback_mode', 'l2')} feedback)")
        ax.axhline(metrics["avail_chance_item"], color="grey", ls=":", lw=1)
        for xi, m in zip(x, MODALITIES):
            mg, ar = r.get(f"{m}_margin"), r.get(f"{m}_audio_recall")
            txt = []
            if mg is not None and np.isfinite(mg):
                txt.append(f"margin {mg:+.2f}")
            if ar is not None and np.isfinite(ar):
                txt.append(f"next tone {ar:.2f}")
            ax.text(xi, -0.21, "\n".join(txt), ha="center", va="top", fontsize=7.5,
                    color="tab:blue", transform=ax.get_xaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in MODALITIES], fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("exact next-item recall")
        ex = series["conditions"][cond]["exposure"]
        ax.set_title(f"{cond}: {K} sequence{'s' if K > 1 else ''} x "
                     f"{metrics['avail_seq_len']} items, one tone per item\n"
                     f"(to criterion on the full cue: {ex['epochs']} epochs, "
                     f"reached={ex['reached']})", fontsize=9.5)
    fig.suptitle(f"Cue availability — {model_name}: completion when a modality is absent "
                 f"(dotted: chance 1/|vocab|; 'next tone' = the prediction's audio block "
                 f"names the next item's tone)", fontsize=9.5)
    # One figure-level legend below the panels: inside an axis it covers a bar,
    # since every bar can reach the top.
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(run_dir, "plots", "cue_availability.png"), dpi=150)
    plt.close(fig)
