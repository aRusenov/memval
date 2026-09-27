"""Weight-space plasticity metrics — the efficiency index.

NOT WIRED INTO ANY PIPELINE. Kept as a standalone, tested utility (the same call
taken for the since-retired text-noise utility at capacity_coverage_audit.md C3), not as a
benchmark read-out. It was drafted for ``schema_consistency`` and withdrawn on
2026-09-02: its utility is unproven and it is **not comparable across the model
taxonomy**. Four problems, all verified:

1. *Checkpoint granularity.* EI is computed per training trial, so cancellation
   *within* a trial is invisible. Holding total training fixed and varying only
   ``epochs_per_trial``, Original EP reads 0.948 / 0.977 / 0.985 / 1.000 at
   1 / 5 / 20 / 60 epochs per checkpoint — and 1.000 is definitional at one
   checkpoint. Arms differ in ``n_epochs`` by design (AHN 1, EP 100).
2. *Discovery is wrong for several arms.* ``DTSESNSequenceNetwork`` exposes a
   fixed random reservoir, time constants and the RLS inverse-correlation
   matrix ``P`` (optimiser state, not a synapse) alongside the one learned matrix ``W_out``; ``ThetaPhaseSequenceNetwork``
   exposes four phase/gating arrays alongside ``W``.
3. *Architectural non-commensurability.* EI measures whether updates cancel, and
   a Hebbian outer-product rule in a near-orthogonal regime cannot cancel: AHN
   returns 0.9953-0.9956 on every rung including `random`. The dynamic range is a
   property of the learning rule, not of the material — so a cross-arm ranking
   would rank rules, not schema sensitivity.
4. *Stochastic rules are penalised.* Sign-flipping updates (e.g. EP's
   ``random_flip_beta``) inflate the denominator for reasons unrelated to
   consistency.

Only a *within-arm, across-condition* comparison at fixed ``epochs_per_trial`` is
defensible — which is how McClelland uses it (one network, all parameters held
identical, NPA vs NM). Anything cross-arm is not. Read all of that before
re-adopting this.

Implements the coherence-of-learning measure from McClelland (2013), *Incorporating
Rapid Neocortical Learning of New Schema-Consistent Information Into Complementary
Learning Systems Theory*, JEP:General 142, 1190–1210 (the paper McClelland,
McNaughton & Lampinen 2020 defers to as ``[18]``). The analysis is attributed
there to Ken Norman.

The efficiency index is

    TC = sum_ij |w_ij(end) - w_ij(start)|            total *net* change
    SC = sum_trials sum_ij |w_ij(after) - w_ij(before)|   summed per-trial change
    EI = TC / SC

Read it as: did the updates this material induced **accumulate**, or did they
fight each other? Perfectly coherent updates all point the same way, so the net
change equals the sum of the individual changes and EI = 1. Mutually
contradictory updates make weights "vacillate back and forth from item to item"
and cancel, driving EI toward 0. McClelland reports EI = 0.93 for new
schema-consistent material against EI = 0.27 for schema-inconsistent material,
at identical learning rates.

The original motivation was that it is measured on the *weights*, not on recall,
so it keeps range when a behavioural read-out is pinned at ceiling — as several
arms are across every condition of the schema section. That motivation did not
survive the fairness check above: the range it keeps is not comparable between
arms. The section is scored on MRR-derived read-outs instead.

By the triangle inequality TC <= SC, so EI is already normalised to [0, 1]; no
scaling is needed before scoring it.

Note the L1 norm: the paper is explicit that "the measure of change used is the
sum ... of the absolute value of the difference". (Its Figure 6 change *index*,
a different quantity, uses squared differences.)

Usage::

    tracker = PlasticityTracker(model)
    tracker.begin()
    for _ in range(n_trials):
        model.fit_sequence(...)
        tracker.checkpoint()
    result = tracker.result()      # {"efficiency_index": ..., "by_param": {...}}

``by_param`` matters for multi-layer arms: McClelland's central mechanistic
finding is that consistent and inconsistent material differ dramatically in the
*item-specific* input weights (which structurally cannot cause interference)
while differing far less in the *shared* downstream weights (which can). A
single scalar EI averages that distinction away.
"""
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


#: Attribute names that hold transient per-step state rather than learned
#: parameters. Underscore-prefixed attributes are skipped separately.
_TRANSIENT = frozenset({"current_state", "current_context", "state", "context"})


def discover_params(model: Any) -> List[str]:
    """Names of a model's learned-parameter arrays, best effort.

    Resolution order:

    1. ``model.plasticity_params`` if the arm declares it — always preferred,
       and the escape hatch for any arm this heuristic gets wrong.
    2. Public float ndarray attributes, minus the known transient-state names.

    Biases are deliberately included: the paper counts them ("including bias
    weights as well as connections from other layers"). Fixed, non-learned
    projections (e.g. a frozen DG matrix) are harmless if they slip through —
    they contribute 0 to both TC and SC and so cannot move the overall ratio,
    though they will show as nan in ``by_param``.
    """
    declared = getattr(model, "plasticity_params", None)
    if declared:
        return [n for n in declared if isinstance(getattr(model, n, None), np.ndarray)]
    return sorted(
        name for name, val in vars(model).items()
        if isinstance(val, np.ndarray)
        and val.dtype.kind == "f"
        and not name.startswith("_")
        and name not in _TRANSIENT
    )


class PlasticityTracker:
    """Accumulates net and per-trial weight change to compute an efficiency index.

    Args:
        model: any arm exposing ndarray parameters.
        params: explicit attribute names to track. Defaults to
            :func:`discover_params`.

    The tracker holds a reference to the model, not a copy, so it reads whatever
    the parameters are at each call. ``begin()`` may be called again to restart
    the measurement window — useful to exclude a schema-acquisition phase from
    the measurement of new learning, which is what the paper does.
    """

    def __init__(self, model: Any, params: Optional[Iterable[str]] = None):
        self.model = model
        self.params: List[str] = list(params) if params is not None else discover_params(model)
        self._base: Dict[str, np.ndarray] = {}
        self._prev: Dict[str, np.ndarray] = {}
        self._summed: Dict[str, float] = {}
        self._n = 0

    def _snapshot(self) -> Dict[str, np.ndarray]:
        out = {}
        for name in self.params:
            val = getattr(self.model, name, None)
            if isinstance(val, np.ndarray):
                out[name] = np.array(val, dtype=float, copy=True)
        return out

    def begin(self) -> "PlasticityTracker":
        """Mark the start of the measurement window. Resets any prior state."""
        self._base = self._snapshot()
        self._prev = {k: v.copy() for k, v in self._base.items()}
        self._summed = {k: 0.0 for k in self._base}
        self._n = 0
        return self

    def checkpoint(self) -> None:
        """Record one training trial's worth of change. Call after each trial."""
        if not self._base:
            raise RuntimeError("PlasticityTracker.begin() must be called first.")
        cur = self._snapshot()
        for name, arr in cur.items():
            prev = self._prev.get(name)
            # A parameter that changed shape mid-window (e.g. a grown layer) is
            # not comparable; drop it rather than silently misreporting.
            if prev is None or prev.shape != arr.shape:
                self._summed.pop(name, None)
                self._base.pop(name, None)
                continue
            self._summed[name] += float(np.abs(arr - prev).sum())
        self._prev = cur
        self._n += 1

    def result(self) -> Dict[str, Any]:
        """Efficiency index over the window, overall and per parameter.

        Returns a dict with ``efficiency_index``, ``total_change`` (TC),
        ``summed_change`` (SC), ``n_checkpoints``, ``params``, and ``by_param``
        mapping each parameter name to its own ``{ei, total, summed}``.

        ``efficiency_index`` is nan when no weight moved at all (SC = 0) — that
        is "nothing was learned", which is not the same as incoherent learning
        and must not be reported as EI = 0.
        """
        by_param: Dict[str, Dict[str, float]] = {}
        tc_all = sc_all = 0.0
        for name in sorted(self._summed):
            base, prev = self._base.get(name), self._prev.get(name)
            if base is None or prev is None or base.shape != prev.shape:
                continue
            tc = float(np.abs(prev - base).sum())
            sc = self._summed[name]
            by_param[name] = {"ei": (tc / sc) if sc > 0 else float("nan"),
                              "total": tc, "summed": sc}
            tc_all += tc
            sc_all += sc
        return {
            "efficiency_index": (tc_all / sc_all) if sc_all > 0 else float("nan"),
            "total_change": tc_all,
            "summed_change": sc_all,
            "n_checkpoints": self._n,
            "params": list(self.params),
            "by_param": by_param,
        }
