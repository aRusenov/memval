"""Probe protocol: what cue a section presents, and how many times.

Every retrieval probe in the suite used to draw ``n_trials=30`` Gaussian
perturbations of the cue at a fixed ``noise_scale=0.05``, in every section,
for every arm. Measured on 2026-09-03:

* for a well-trained arm all 30 draws agree at every position -- the trials
  collapse to one and buy nothing;
* for an arm near a decision boundary they produce a graded fraction, but that
  fraction is a Monte-Carlo estimate of the deterministic **margin**
  (Spearman 0.76, verdict agreement 89%), at 30x the cost and with an
  arbitrary sigma baked in;
* the 30 draws share model, weights, sequence and encoder, so n=30 is
  pseudo-replication for any claim about *capability*. The unit that varies
  capability is the training seed.

So the protocol is now declared per section and recorded in the results:

* ``CLEAN_SINGLE`` -- the default for every capability section. One clean cue,
  one probe; report accuracy AND margin. Repeats only if the arm declares
  ``stochastic_forward`` (then the repeats sample the *model's* noise, with the
  cue held fixed -- a different estimand, and it must say so).
* ``NOISE_SWEEP`` / ``MASK_SWEEP`` -- sections whose capacity is *defined* by
  cue degradation (noise invariance, cue masking / pattern completion). There
  sigma or the mask fraction is the independent variable and is swept, and
  repeated draws at each level are the estimand P(correct | level).
* ``NOISE_TRIALS`` -- the legacy protocol, kept only so an old results file can
  be described. No shipped section uses it.

What "clean" does NOT remove: rollout drift under OBSERVATION rollout (that is
the arm's own error and is what autoregressive sections measure), and material
similarity (``category_variance``, overlap fraction) which is a property of the
material, not the probe.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models.capabilities import is_stochastic_forward


class ProbeProtocol(str, Enum):
    CLEAN_SINGLE = "clean_single"
    NOISE_SWEEP = "noise_sweep"
    MASK_SWEEP = "mask_sweep"
    NOISE_TRIALS = "noise_trials"   # legacy; describes pre-refactor results only


#: Per-section declaration. Any section absent here is an error at pipeline
#: start -- an undeclared probe is the silent default this module removes.
SECTION_PROBE_PROTOCOL: Dict[str, ProbeProtocol] = {
    # --- symbolic suite ---
    "presentation_duration":   ProbeProtocol.CLEAN_SINGLE,
    "sequence_length":         ProbeProtocol.CLEAN_SINGLE,
    "multiple_sequences":      ProbeProtocol.CLEAN_SINGLE,
    "continual_chain":         ProbeProtocol.CLEAN_SINGLE,
    "paired_associate":        ProbeProtocol.CLEAN_SINGLE,
    "noise_invariance":        ProbeProtocol.NOISE_SWEEP,
    "cue_masking":             ProbeProtocol.MASK_SWEEP,
    "cue_availability":        ProbeProtocol.CLEAN_SINGLE,   # categorical manipulation, no sweep
    "semantic_similarity":     ProbeProtocol.CLEAN_SINGLE,
    "symbolic_disambiguation": ProbeProtocol.CLEAN_SINGLE,
    "schema_consistency":      ProbeProtocol.CLEAN_SINGLE,
    "cognitive_phenomena":     ProbeProtocol.CLEAN_SINGLE,   # L0 read-outs; replication over material
    "interval_retention":      ProbeProtocol.CLEAN_SINGLE,
    "interval_generation":     ProbeProtocol.CLEAN_SINGLE,
    # --- online symbolic suite ---
    "online_convergence":      ProbeProtocol.CLEAN_SINGLE,
    "isi_tolerance":           ProbeProtocol.CLEAN_SINGLE,
}

LEGACY_SIGMA = 0.05


def resolve_probe(
    section: str,
    model_or_class: Any,
    n_trials_requested: int,
    sweep_level: Optional[float] = None,
) -> Tuple[int, float]:
    """(n_trials, noise_scale) a section must pass to the probe functions.

    ``sweep_level`` is the current sigma for a NOISE_SWEEP section and is
    ignored otherwise. Raises on an undeclared section.
    """
    try:
        proto = SECTION_PROBE_PROTOCOL[section]
    except KeyError:
        raise KeyError(
            f"section {section!r} has no entry in SECTION_PROBE_PROTOCOL; every "
            f"section must declare how it probes") from None

    if proto is ProbeProtocol.CLEAN_SINGLE:
        # Deterministic arm + clean cue => every repeat is identical. Repeat only
        # when the arm's own forward pass is stochastic, and then the repeats
        # measure the model's noise, not the cue's.
        n = int(n_trials_requested) if is_stochastic_forward(model_or_class) else 1
        return n, 0.0
    if proto is ProbeProtocol.NOISE_SWEEP:
        if sweep_level is None:
            raise ValueError("NOISE_SWEEP sections must pass sweep_level=sigma")
        return int(n_trials_requested), float(sweep_level)
    if proto is ProbeProtocol.MASK_SWEEP:
        return int(n_trials_requested), 0.0
    return int(n_trials_requested), LEGACY_SIGMA


def probe_record(section: str, model_or_class: Any, n_trials_requested: int) -> Dict[str, Any]:
    """What to write into results metadata for this section."""
    proto = SECTION_PROBE_PROTOCOL[section]
    if proto in (ProbeProtocol.CLEAN_SINGLE, ProbeProtocol.MASK_SWEEP):
        n, s = resolve_probe(section, model_or_class, n_trials_requested)
    else:
        n, s = int(n_trials_requested), None   # sigma is the swept variable
    return {
        "protocol": proto.value,
        "n_trials": n,
        "noise_scale": s,
        "stochastic_forward": bool(is_stochastic_forward(model_or_class)),
    }


# ---------------------------------------------------------------------------
# The deterministic graded readout that replaces the noise trials
# ---------------------------------------------------------------------------

def measure_recall_margin(
    network: Any,
    words: List[str],
    encoder: Any,
    context_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-position margin from ONE clean probe: cos(pred, target) minus the
    best competitor's cosine. Position 0 (the cue itself) is NaN.

    Positive means the target wins the decode; the magnitude is how decisively.
    Ties (margin == 0) are failures under the accuracy readout, per the
    model-free-reference house rule. This is what the 30 noise trials were
    estimating stochastically -- and it resolves below the accuracy floor,
    where accuracy is 0 for every position and the trials were too.
    """
    X = np.asarray(encoder.encode(words), dtype=float)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])

    margins = np.full(len(words), np.nan)
    if hasattr(network, "reset_context"):
        network.reset_context()
    for i in range(len(words) - 1):
        network.current_t = i
        pred = np.asarray(network.predict_next(X[i].copy(), current_context=context), dtype=float).ravel()
        pn = np.linalg.norm(pred)
        if pn <= 1e-12:
            margins[i + 1] = -1.0     # a null prediction loses to everything
            continue
        sims = Xn @ (pred / pn)
        target = sims[i + 1]
        competitor = np.max(np.delete(sims, i + 1))
        margins[i + 1] = float(target - competitor)
    return margins


def mean_margin(margins: np.ndarray) -> float:
    """Mean over recalled positions (position 0 excluded), NaN-safe."""
    m = np.asarray(margins, dtype=float)[1:]
    return float(np.nanmean(m)) if m.size and not np.all(np.isnan(m)) else float("nan")
