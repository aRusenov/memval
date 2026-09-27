#!/usr/bin/env python
"""Capacity scorecard: map every emitted benchmark metric onto its manipulated
dimension and the capacity it evidences, normalise it to [0, 1], and roll the
result up with a two-level weighted sum.

    metric  --normalise-->  n_m in [0,1]
    dimension score   D_d = sum_m(v_m * n_m) / sum_m(v_m)
    capacity score    C   = sum_d(w_d * D_d) / sum_d(w_d)     [available d only]
    capacity coverage cov = sum_{d available}(w_d) / sum_{d intended}(w_d)

Every key written to ``metrics.json`` is classified. Keys carrying a role other
than ``score`` (``config``, ``guard``, ``control``, ``diagnostic``) are reported
but never enter a sum, and any key missing from the spec is raised loudly rather
than silently dropped.

Taxonomy sources:
  * capacities              -- paper/capacities.md (five, post serial-order merge)
  * material / probe split  -- docs/capacity_coverage_audit.md + manipulation taxonomy
  * exclusion rationale     -- docs/capacity_coverage_audit.md sec 0 (C1, C2) and sec 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Normalisers.  Each maps a raw metric onto [0, 1] where 1 is "ideal".
# --------------------------------------------------------------------------

def hi(x, **k):
    """Already on [0,1], higher is better."""
    return _clip(x)

def lo(x, **k):
    """Already on [0,1], lower is better."""
    return _clip(1.0 - x)

def neg_corr(x, **k):
    """A signed correlation whose *predicted* value is negative, mapped to [0,1].

    ``hi``/``lo`` both assume [0,1] inputs and mis-score a correlation: ``hi``
    clips a theory-confirming -1 to 0.0 and awards a theory-contradicting +1 the
    full 1.0. Here -1 -> 1.0, 0 -> 0.0, +1 -> 0.0.
    """
    return _clip(-x)

def signed_drop(x, **k):
    """A signed change where 0 (or better) is ideal and -1 is total loss.

    For a metric emitted as ``after - before``: 0.0 means nothing was lost,
    -1.0 means everything was. ``lo`` cannot score this -- it computes
    ``1 - x``, so a catastrophic -0.606 becomes 1.606 and CLIPS TO 1.0, awarding
    the worst possible result full marks. ``hi`` cannot either: it would score
    perfect retention (0.0) as 0.0. Hence: -1 -> 0.0, 0 -> 1.0, and a positive
    change (recall improved after interference) is clipped to 1.0 rather than
    rewarded above it.
    """
    return _clip(1.0 + x)

def chance(x, chance_level=0.5, **k):
    """Chance-corrected accuracy: (x - c) / (1 - c), floored at 0."""
    return _clip((x - chance_level) / (1.0 - chance_level))

def ratio(x, ref=1.0, **k):
    """Fraction of an attainable maximum, higher is better."""
    return _clip(x / ref) if ref else float("nan")

def inv_count(x, **k):
    """Trials/epochs to criterion, 1 -> 1.0, n -> 1/n. Lower is better."""
    return _clip(1.0 / x) if x and x > 0 else 0.0

def budget(x, max_trials=12.0, **k):
    """Trials to criterion against the protocol's own trial budget."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return _clip((max_trials - x) / (max_trials - 1.0))

def log_ratio(x, **k):
    """A ratio >= 1 where 1 is ideal and cost grows multiplicatively. Scored on a
    log scale because the quantity being compared (exposure) is itself log-ish:
    1 -> 1.0, 2 -> 0.50, 8 -> 0.25, 32 -> 0.17. An infinite (censored) ratio is 0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(v):
        return float("nan")
    if math.isinf(v) or v <= 0:
        return 0.0
    return _clip(1.0 / (1.0 + math.log2(max(v, 1.0))))


def boolean(x, **k):
    return 1.0 if bool(x) else 0.0

def _clip(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(v):
        return float("nan")
    return max(0.0, min(1.0, v))


# --------------------------------------------------------------------------
# Metric specification.
#
# role:  score      -- enters the weighted sum
#        control    -- an ablation/floor whose low value is the intended result
#        guard      -- validity check on its own section
#        config     -- a protocol setting, not a measurement
#        diagnostic -- measured, but excluded from scoring with a stated reason
#
# v      -- within-dimension weight (2 = primary readout, 1 = secondary)
# --------------------------------------------------------------------------

M = lambda **kw: kw

SPEC: Dict[str, Dict[str, Any]] = {}

def add(key, capacity, dimension, kind, section, suite, norm, v=1,
        role="score", note="", **norm_kw):
    SPEC[key] = M(capacity=capacity, dimension=dimension, kind=kind, section=section,
                  suite=suite, norm=norm, v=v, role=role, note=note, norm_kw=norm_kw)

# ======================= CONTINUAL RETENTION ==============================
# Dimension: load (material) -- how many other sequences intervene.
add("multiple_seq_mrr_before", "Continual retention", "load", "material",
    "multiple_sequences", "symbolic", chance, v=1, role="diagnostic",
    note="Pre-interference baseline; the reference for delta, not a retention score.",
    chance_level=1/14)
add("multiple_seq_mrr_after", "Continual retention", "load", "material",
    "multiple_sequences", "symbolic", chance, v=2,
    note="Recall of list A after list B. Chance = 1/14 (multi_vocab size).",
    chance_level=1/14)
# NOTE THE SIGN. symbolic_pipeline.py emits `mrr_after - mrr_before`, i.e.
# NEGATIVE means forgetting -- the opposite of what this entry's note claimed
# until 2026-09-04. Paired with `lo` it scored -0.606 (EP's catastrophic
# interference) as a clipped 1.0, the same as AHN's 0.000. `signed_drop` gives
# 1.0 at no loss and 0.0 at total loss; EP now normalises to 0.394.
add("delta_mrr_forgetting", "Continual retention", "load", "material",
    "multiple_sequences", "symbolic", signed_drop, v=2,
    note="MRR(after) - MRR(before). 0 = no catastrophic interference at T=2; "
         "negative = forgetting, and -1 is total loss of list A.")
# INTERLEAVED ingestion condition (added 2026-09-04). Same lists, same per-list
# exposure, but A and B are trained as one stream in seeded random block order
# to a criterion on the worse list. The blocked keys above are one ingestion
# order; for an arm that sizes assemblies by item frequency (the retired Vieth STDP arm) it is
# the order under which a list presented alone takes the whole network. Its
# source paper only ever trains in random block order, so without this
# condition the section could not distinguish "cannot hold two lists" from
# "cannot hold them in THIS order".
# Diagnostic, NOT scored (decided 2026-09-05). Interleaving is the easier
# protocol for every arm -- there is no "before" to forget from, so a high value
# here demonstrates joint acquisition, not retention -- and scoring it would
# reward the easier condition and move every arm's `load`. The condition exists
# for the CONTRAST: multiple_seq_blocking_cost is what the section adds.
add("multiple_seq_interleaved_mrr_A", "Continual retention", "load", "material",
    "multiple_sequences", "symbolic", chance, role="diagnostic",
    note="Recall of list A after A and B were trained interleaved -- the "
         "ingestion-order control for multiple_seq_mrr_after. Chance = 1/14. "
         "Not scored: it measures joint acquisition, not retention.",
    chance_level=1/14)
add("multiple_seq_interleaved_mrr_B", "Continual retention", "load", "material",
    "multiple_sequences", "symbolic", chance, role="diagnostic",
    note="Recall of list B under the same interleaved training. Chance = 1/14. "
         "Not scored, for the same reason.",
    chance_level=1/14)
add("multiple_seq_interleaved_span_A", "Continual retention", "load", "material",
    "multiple_sequences", "symbolic", None, role="diagnostic",
    note="Autoregressive span of list A after interleaved training.")
add("multiple_seq_blocking_cost", "Continual retention", "load", "material",
    "multiple_sequences", "symbolic", None, role="diagnostic",
    note="multiple_seq_mrr_after - multiple_seq_interleaved_mrr_A: what presenting "
         "the lists one after another cost A's recall, relative to presenting them "
         "as one stream. Negative = blocking hurts. Reported, not scored: both "
         "terms are already scored above and this is their contrast.")
add("multiple_seq_interleaved_criterion_reached", "Continual retention", "load",
    "material", "multiple_sequences", "symbolic", boolean, role="guard",
    note="Did the interleaved stream reach criterion on the WORSE of the two "
         "lists inside the staircase budget? False means the interleaved "
         "read-outs are lower bounds at a censored exposure.")
# --- 2026-09-23: tmaze_disambiguation encoder provenance. CONFIG, never scored.
# The grid dropped 400 -> 100 place cells (120 features) because a finer grid makes
# the centre-of-mass decoder WORSE (clean-decode floor 48% of eps at 100 cells vs
# 77% at 400) while the route is a 1-D curve. See docs/model_table.md.
add("tmaze_pc_n_cells", "Pattern completion", "cue_point", "material",
    "tmaze_completion", "spatial", None, role="config",
    note="Place cells in the completion route's encoder. Fixed per suite at 400 "
         "(20x20) from 2026-09-23; was 100. Pass n_cells_per_dim=10 to reproduce.")
add("tmaze_disamb_n_place_cells", "Sequence disambiguation", "contextual_overlap",
    "material", "tmaze_disambiguation", "spatial", None, role="config",
    note="Place cells in the disambiguation route's encoder (default 100, a 10x10 "
         "grid shared with tmaze_completion). Pass n_place_cells=400 to reproduce a "
         "pre-2026-09-23 number.")
add("tmaze_disamb_n_features", "Sequence disambiguation", "contextual_overlap",
    "material", "tmaze_disambiguation", "spatial", None, role="config",
    note="Total input width of the disambiguation route: place cells + odour dims "
         "(default 100 + 20 = 120).")

add("multiple_seq_ingestion", "Continual retention", "load", "material",
    "multiple_sequences", "symbolic", None, role="config",
    note="Which ingestion conditions the section ran: both (default), blocked, "
         "or interleaved (--benchmark-args multiple_sequences:ingestion=...).")

# --- 2026-09-06: clean single probe, list-pair replication, margin readout, overlap condition.
# All DIAGNOSTIC / CONFIG on purpose: `delta_mrr_forgetting` / `multiple_seq_mrr_after` stay the
# scored keys, so this session's additions change no published capacity number. Switching the
# scored key to the graded `_lists` margin delta is a one-line decision left to the maintainer.
add("multiple_seq_n_lists", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='config',
    note='How many A->B list pairs the serial-position curve averages over (default 5; pair 0 is the canonical fruit->animal pair every un-suffixed scalar refers to).')
add("multiple_seq_prob_before", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="P(position recalled | clean cue) averaged over list pairs, before B. The honest replacement for the 30-noise-trial 'recall probability': replication over MATERIAL, not cue noise.")
add("multiple_seq_prob_after", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='Same, after B. At ceiling for a capable arm on disjoint-category pairs (in_span ~0.23): see the overlap keys for why.')
add("delta_prob_forgetting", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='prob_after - prob_before over list pairs. Negative = forgetting. 0 at ceiling is a true statement about recall, not a broken metric; read delta_margin_forgetting_lists alongside.')
add("multiple_seq_margin_before", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='Mean per-position margin (cos to target minus best competitor) on the CANONICAL pair before B. Positive <=> clean decode correct; magnitude = how decisively.')
add("multiple_seq_margin_after", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='Canonical-pair margin after B.')
add("delta_margin_forgetting", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='Canonical-pair margin delta. Resolves forgetting below the accuracy floor: theta -0.006 vs EP -0.108 at their budgets while both sit at P=1.00.')
add("multiple_seq_margin_before_lists", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="Margin before B, mean over all list pairs (the figure's right panel).")
add("multiple_seq_margin_after_lists", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='Margin after B, mean over all list pairs.')
add("delta_margin_forgetting_lists", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='All-pairs margin delta -- the graded forgetting number the figure shows. Candidate for the scored key.')
add("multiple_seq_overlap_b_exposure_multiplier", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='config',
    note='Plasticity-asymmetry knob for the overlap condition: B is refit for k x its budget (default 1).')
add("multiple_seq_overlap_disjoint_in_span", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="Model-free overlap covariate for the disjoint-category CONTROL (fruit->animal, animal->number), L=6: mean fraction of each B input's norm inside span(A). A linear associator's B step disturbs A by lr*err_B*(x_B.x_A), so this is what a forgetting delta must be read against (~0.22 disjoint, ~0.37 same-category).")
add("multiple_seq_overlap_disjoint_prob_before", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='P(A recalled | clean cue) before B, disjoint-category CONTROL (fruit->animal, animal->number), L=6.')
add("multiple_seq_overlap_disjoint_prob_after", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='P(A recalled | clean cue) after B, disjoint-category CONTROL (fruit->animal, animal->number), L=6.')
add("multiple_seq_overlap_disjoint_delta_prob", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='prob_after - prob_before, disjoint-category CONTROL (fruit->animal, animal->number), L=6.')
add("multiple_seq_overlap_disjoint_margin_before", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='Margin before B, disjoint-category CONTROL (fruit->animal, animal->number), L=6.')
add("multiple_seq_overlap_disjoint_margin_after", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='Margin after B, disjoint-category CONTROL (fruit->animal, animal->number), L=6.')
add("multiple_seq_overlap_disjoint_delta_margin", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='Margin delta, disjoint-category CONTROL (fruit->animal, animal->number), L=6.')
add("multiple_seq_overlap_same_category_in_span", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="Model-free overlap covariate for the B drawn from A's own category (fruit->fruit, animal->animal), L=6: mean fraction of each B input's norm inside span(A). A linear associator's B step disturbs A by lr*err_B*(x_B.x_A), so this is what a forgetting delta must be read against (~0.22 disjoint, ~0.37 same-category).")
add("multiple_seq_overlap_same_category_prob_before", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="P(A recalled | clean cue) before B, B drawn from A's own category (fruit->fruit, animal->animal), L=6.")
add("multiple_seq_overlap_same_category_prob_after", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="P(A recalled | clean cue) after B, B drawn from A's own category (fruit->fruit, animal->animal), L=6.")
add("multiple_seq_overlap_same_category_delta_prob", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="prob_after - prob_before, B drawn from A's own category (fruit->fruit, animal->animal), L=6.")
add("multiple_seq_overlap_same_category_margin_before", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="Margin before B, B drawn from A's own category (fruit->fruit, animal->animal), L=6.")
add("multiple_seq_overlap_same_category_margin_after", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="Margin after B, B drawn from A's own category (fruit->fruit, animal->animal), L=6.")
add("multiple_seq_overlap_same_category_delta_margin", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note="Margin delta, B drawn from A's own category (fruit->fruit, animal->animal), L=6.")
add("multiple_seq_overlap_cost_prob", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='same_category delta_prob minus disjoint delta_prob: extra forgetting attributable to shared input directions alone (same length, budgets, variance). theta at B x10: -0.40.')
add("multiple_seq_overlap_cost_margin", 'Continual retention', 'load', 'material',
    'multiple_sequences', 'symbolic', None, role='diagnostic',
    note='same_category delta_margin minus disjoint delta_margin. Moves at B x1 where prob does not.')

# Dimension: contingency (material) -- a learned association is invalidated.
for proto in ("direct", "extinction"):
    add(f"reversal_{proto}_reversal_final_arm_accuracy", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", chance, v=2,
        note="Post-reversal choice of the newly rewarded arm. Chance = 0.5 (2AFC).",
        chance_level=0.5)
    add(f"reversal_{proto}_reversal_final_perseveration", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", lo, v=2,
        note="Residual pull to the old, now-invalid arm. This is the plasticity "
             "side of stability-plasticity: lower is better.")
    add(f"reversal_{proto}_reversal_trials_to_criterion", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", budget, v=2,
        note="Trials to relearn the reversed contingency, against the 12-trial budget.",
        max_trials=12.0)
    add(f"reversal_{proto}_acquisition_trials_to_criterion", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", budget, v=1,
        note="Original acquisition speed; the reference for the reversal cost.",
        max_trials=12.0)
    add(f"reversal_{proto}_recovery_after_interference_arm_accuracy",
        "Continual retention", "contingency", "material", "tmaze_reversal",
        "spatial", chance, v=2,
        note="Does the reversed mapping survive its own subsequent interference.",
        chance_level=0.5)
    add(f"reversal_{proto}_recovery_after_interference_perseveration",
        "Continual retention", "contingency", "material", "tmaze_reversal",
        "spatial", lo, v=1, note="Perseveration measured after the interference probe.")
    add(f"reversal_{proto}_recovery_after_interference_delta", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", lo, v=1,
        note="Accuracy lost across the interference probe.")
    add(f"reversal_{proto}_reversal_criterion_reached", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", boolean, v=1,
        role="guard", note="Did the reversal stage converge inside the budget.")
    add(f"reversal_{proto}_acquisition_criterion_reached", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", boolean, v=1,
        role="guard", note="Did acquisition converge inside the budget.")
    add(f"reversal_{proto}_acquisition_final_arm_accuracy", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", chance, v=1,
        role="guard", note="Was anything learned to reverse in the first place.",
        chance_level=0.5)
    add(f"reversal_{proto}_max_trials", "Continual retention", "contingency",
        "material", "tmaze_reversal", "spatial", None, role="config",
        note="Trial budget per stage.")
    # Reward / anticipation read-outs: informative shape, not a retention score.
    for stage in ("acquisition", "reversal", "extinction"):
        for tail, nte in (
            ("final_reward_pred", "Predicted reward at the goal; scale set by reward_gain."),
            ("final_goal_identity_margin", "Margin between the two goal identities."),
            ("final_reward_pre_goal", "Reward predicted one step before the goal."),
            ("final_anticipation_lead", "Steps ahead the reward is anticipated."),
            ("max_anticipation_lead", "Best anticipation lead over the stage."),
        ):
            add(f"reversal_{proto}_{stage}_{tail}", "Continual retention",
                "contingency", "material", "tmaze_reversal", "spatial", None,
                role="diagnostic", note=nte + " Unbounded / gain-scaled: shape read-out.")
    add(f"reversal_{proto}_{'extinction'}_criterion_reached", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", boolean, v=1,
        role="guard", note="Did the extinction stage converge.")
    add(f"reversal_{proto}_extinction_trials_to_criterion", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", budget, v=1,
        note="Trials to extinguish the acquired association.", max_trials=12.0)
    add(f"reversal_{proto}_extinction_final_arm_accuracy", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", None,
        role="diagnostic",
        note="Arm choice during extinction: the arm mapping is unchanged here, "
             "so 1.0 is expected and carries no retention information.")
    add(f"reversal_{proto}_extinction_final_perseveration", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", lo, v=1,
        note="Perseveration measured at the end of extinction.")
    add(f"reversal_{proto}_acquisition_final_perseveration", "Continual retention",
        "contingency", "material", "tmaze_reversal", "spatial", lo, v=1,
        role="guard",
        note="Perseveration at the end of original acquisition. A baseline for the "
             "reversal-stage value, not itself a retention score.")
add("reversal_extinction_extinction_reward_drop", "Continual retention",
    "contingency", "material", "tmaze_reversal", "spatial", hi, v=2,
    note="Collapse of the reward prediction under extinction, normalised by the "
         "reward prediction actually acquired (see derived key).")
add("reversal_reward_gain", "Continual retention", "contingency", "material",
    "tmaze_reversal", "spatial", None, role="config",
    note="Teaching-signal strength. A protocol choice, explicitly not a result "
         "(see spatial_pipeline.py:505).")
add("reversal_epochs_per_trial", "Continual retention", "contingency", "material",
    "tmaze_reversal", "spatial", None, role="config", note="Training budget per trial.")

# --- Dimension: load, continued -- the chain. --------------------------------
# `multiple_sequences` above is the T=2 corner of this section: one interposed
# list, measured once. The chain measures the same thing at every interposition
# gap and, crucially, reads the diagonal as well (see plasticity_under_load).
add("chain_avg_accuracy", "Continual retention", "load", "material",
    "continual_chain", "symbolic", chance, v=2,
    note="ACC: mean recall of every task after the whole chain. Chance = 1/30 "
         "(6 tasks x 5 words). Shifts if n_tasks/seq_len are overridden.",
    chance_level=1/30)
add("chain_avg_forgetting", "Continual retention", "load", "material",
    "continual_chain", "symbolic", lo, v=2,
    note="Mean drop from each task's peak to its final score. Can go negative "
         "under backward transfer, which `lo` clips to 1.0 -- read "
         "chain_backward_transfer when it does.")
add("chain_retention_ratio", "Continual retention", "load", "material",
    "continual_chain", "symbolic", hi, v=2,
    note="Chance-corrected ACC / LA: of what was actually acquired, how much "
         "survived. THE un-gameable stability number -- unlike avg_forgetting "
         "it gives no credit for material the arm never learned, so an arm "
         "cannot raise it by freezing.")
add("chain_backward_transfer", "Continual retention", "load", "material",
    "continual_chain", "symbolic", None, role="diagnostic",
    note="Signed BWT. Reported rather than scored: it shares a scale with no "
         "other metric here and its sign is the information.")
add("chain_n_tasks", "Continual retention", "load", "material",
    "continual_chain", "symbolic", None, role="config",
    note="Chain length. Sets the range of the forgetting gradient.")
add("chain_chance_level", "Continual retention", "load", "material",
    "continual_chain", "symbolic", None, role="config",
    note="Chance recall for the chain vocabulary; used by the ratio statistics.")
add("chain_acquired", "Continual retention", "load", "material",
    "continual_chain", "symbolic", boolean, role="guard",
    note="Did the chain acquire anything at all. Every ratio statistic is "
         "vacuous below this, and a vacuous SPI reads as a middling one.")

# --- Dimension: plasticity_under_load (material) -----------------------------
# The other half of stability-plasticity, and the half every other section in
# the suite is blind to. Read down the DIAGONAL of the same matrix: how well was
# each task acquired at the moment it was trained, with everything earlier
# already in the substrate? An arm that stops learning after task 0 maximises
# every stability read-out in this capacity -- perfect final row, zero
# forgetting, zero BWT -- and is caught only here.
add("chain_avg_learning", "Continual retention", "plasticity_under_load",
    "material", "continual_chain", "symbolic", chance, v=2,
    note="LA: mean of the retention-matrix diagonal. Acquisition under load. "
         "Chance = 1/30, as chain_avg_accuracy.", chance_level=1/30)
add("chain_intransigence", "Continual retention", "plasticity_under_load",
    "material", "continual_chain", "symbolic", lo, v=2,
    note="R[0,0] minus the mean of the rest of the diagonal: how much "
         "acquisition degraded once the substrate was no longer empty. The "
         "reference is the arm's own first task, so this is within-run and is "
         "NOT comparable to Chaudhry's I in absolute terms. Negative (forward "
         "transfer) clips to 1.0.")
add("chain_learning_slope", "Continual retention", "plasticity_under_load",
    "material", "continual_chain", "symbolic", None, role="diagnostic",
    note="OLS slope of the diagonal against task index -- the shape "
         "intransigence averages away. A cliff at task 1 and a steady decline "
         "give the same mean and different slopes.")
add("chain_stability_plasticity_index", "Continual retention",
    "plasticity_under_load", "material", "continual_chain", "symbolic", None,
    role="diagnostic",
    note="THE JOINT READ-OUT: harmonic mean of chance-corrected LA and "
         "retention_ratio, so the frozen arm and the overwriting arm both score "
         "near 0 where an arithmetic mean would give each 0.5. Deliberately NOT "
         "scored: both of its components are already scored in their own "
         "dimensions, and adding it would double-count them. Report it beside "
         "the capacity score, and read it with the two coordinates -- it says "
         "how balanced an arm is, only the coordinates say which way it fails.")

# --- Dimension: relevance (material) -----------------------------------------
# paper/capacities.md argues that forgetting under a finite substrate is
# functional rather than a failure. Every other section treats all stored
# material as worth keeping, so that claim had no instrument. Here half the
# chain stays in use (it is re-presented) and half does not; the question is not
# how much survives but whether what survives is the part still being used.
add("select_selectivity", "Continual retention", "relevance", "material",
    "continual_chain", "symbolic", hi, v=2,
    note="Difference-in-differences: (gain on re-presented tasks) - (drift on "
         "the rest), each against its own pre-rehearsal score, so the chain's "
         "serial-position profile cancels. Positive = the substrate was "
         "reallocated toward the material still in use. An arm that keeps "
         "everything and an arm that keeps nothing both score 0; the two "
         "component read-outs separate them.")
add("select_rehearsed_gain", "Continual retention", "relevance", "material",
    "continual_chain", "symbolic", hi, v=1,
    note="Recovery of the re-presented (still-relevant) tasks. The rehearsed "
         "set is the OLDEST half of the chain: most decayed, so a gain has "
         "headroom, and recency-disadvantaged, so this cannot be recency.")
add("select_unrehearsed_drift", "Continual retention", "relevance", "material",
    "continual_chain", "symbolic", None, role="diagnostic",
    note="Change in the tasks that were not re-presented. A NEGATIVE value is "
         "the intended result under capacity pressure -- that is the functional "
         "forgetting the capacity prose claims -- so neither hi nor lo scores "
         "it correctly on its own; it enters via select_selectivity.")
for _k, _n in (("select_rehearsed_recall", "re-presented"),
               ("select_unrehearsed_recall", "not re-presented")):
    add(_k, "Continual retention", "relevance", "material", "continual_chain",
        "symbolic", None, role="diagnostic",
        note=f"Absolute post-rehearsal recall of the {_n} tasks. Levels, kept "
             f"so the difference-in-differences can be read back to them.")
add("select_pressure", "Continual retention", "relevance", "material",
    "continual_chain", "symbolic", None, role="guard",
    note="The chain's own avg_forgetting: how hard the substrate was pressed. "
         "Selectivity is only a question when something had to be discarded.")
add("select_under_pressure", "Continual retention", "relevance", "material",
    "continual_chain", "symbolic", boolean, role="guard",
    note="Did the chain saturate the substrate. False = the dimension is "
         "dropped: nothing needed discarding, so a zero selectivity means the "
         "question was not posed, not that the arm failed it.")
add("select_rehearsal_effective", "Continual retention", "relevance", "material",
    "continual_chain", "symbolic", boolean, role="guard",
    note="Did re-presentation move anything at all. If not, the arm had no "
         "plasticity left to allocate and selectivity says nothing about "
         "selection.")
for _k, _d in (("select_n_rehearsed", "How many of the oldest tasks stayed relevant."),
               ("select_rehearsal_epochs", "Epoch budget for the re-presentation.")):
    add(_k, "Continual retention", "relevance", "material", "continual_chain",
        "symbolic", None, role="config", note=_d)

# --- Dimension: contingency, continued -- the symbolic instrument ------------
# tmaze_reversal was the only place in the suite where an association is
# invalidated and overwriting is the correct answer, and it lives in one
# modality on one 2AFC contingency. AB/AC re-pairs the same cues with new
# targets, so the old response becomes an error: after phase 2, emitting B is an
# intrusion, not retention. multiple_sequences is this section's control
# condition, published without its experimental arm.
add("pa_abac_ac_recall_final", "Continual retention", "contingency", "material",
    "paired_associate", "symbolic", hi, v=2,
    note="Was the NEW association acquired -- the plasticity half. Cue A now "
         "returns target C.")
add("pa_abac_ac_trials_to_criterion", "Continual retention", "contingency",
    "material", "paired_associate", "symbolic", budget, v=2,
    note="Passes over the AC list before the new association reaches criterion, "
         "against the 6-trial phase-2 budget. The symbolic twin of "
         "reversal_trials_to_criterion.", max_trials=6.0)
add("pa_abac_ab_recall_final", "Continual retention", "contingency", "material",
    "paired_associate", "symbolic", lo, v=2,
    note="Rate at which cue A still returns the OLD target after AC training. "
         "Scored lower-is-better because after phase 2 this is an INTRUSION, "
         "not retention -- B is no longer the right answer. The symbolic "
         "counterpart of reversal_final_perseveration.")
add("pa_abac_other_rate_final", "Continual retention", "contingency", "material",
    "paired_associate", "symbolic", lo, v=1,
    note="Cue A returning neither target: the association was DESTROYED rather "
         "than updated. Separates a clean overwrite from a collapse, which a "
         "fall in AB recall alone cannot.")
add("pa_cue_competition_cost", "Continual retention", "contingency", "material",
    "paired_associate", "symbolic", hi, v=2,
    note="AB retention under the disjoint-cue control MINUS AB retention under "
         "AB/AC: damage attributable to competition for the cue, over and above "
         "generic interference from an equal quantity of new learning. Scored "
         "higher-is-better, which is counterintuitive and deliberate: a high "
         "value means forgetting was aimed at exactly the association that "
         "became invalid instead of spread over everything. Indiscriminate "
         "forgetting and total rigidity both give 0.")
add("pa_abac_ab_retention", "Continual retention", "contingency", "material",
    "paired_associate", "symbolic", None, role="diagnostic",
    note="AB final / AB baseline. The normalised form of ab_recall_final, kept "
         "because it is the input to pa_cue_competition_cost; not scored "
         "separately, which would double-count it.")
add("pa_abac_ab_recall_baseline", "Continual retention", "contingency",
    "material", "paired_associate", "symbolic", hi, role="guard",
    note="AB recall before phase 2. There must be an association to interfere "
         "with before any of this section means anything.")
add("pa_abac_ab_acquired", "Continual retention", "contingency", "material",
    "paired_associate", "symbolic", boolean, role="guard",
    note="Did phase 1 reach criterion on the AB list.")
add("pa_abac_ac_criterion_reached", "Continual retention", "contingency",
    "material", "paired_associate", "symbolic", boolean, role="guard",
    note="Did the new association converge inside the phase-2 budget; "
         "trials_to_criterion is NaN when it did not.")

# The disjoint-cue reference condition: identical pair count, trial count and
# epoch budget, fresh cues and fresh targets. This IS the multiple_sequences
# manipulation, run here so the AB/AC arm has something to be read against.
add("pa_control_ab_recall_final", "Continual retention", "contingency",
    "material", "paired_associate", "symbolic", None, role="reference",
    note="AB recall after an equal quantity of DISJOINT-cue learning. High is "
         "the expected result and is the baseline the AB/AC arm is measured "
         "against, not a score in itself.")
add("pa_control_ab_retention", "Continual retention", "contingency", "material",
    "paired_associate", "symbolic", None, role="reference",
    note="Normalised form of the above; the first term of "
         "pa_cue_competition_cost.")
add("pa_control_ac_recall_final", "Continual retention", "contingency",
    "material", "paired_associate", "symbolic", None, role="control",
    note="FLOOR CHECK. The C words are never trained under the control, so this "
         "must stay at ~0. A non-zero value means the vocabulary geometry, not "
         "the model, is producing the AB/AC overwrite.")
add("pa_control_other_rate_final", "Continual retention", "contingency",
    "material", "paired_associate", "symbolic", None, role="reference",
    note="Collapse rate under disjoint-cue interference.")
add("pa_control_ab_recall_baseline", "Continual retention", "contingency",
    "material", "paired_associate", "symbolic", hi, role="guard",
    note="Phase 1 is identical across conditions; this must match the AB/AC "
         "baseline, and a divergence means the two runs were not matched.")
add("pa_control_ab_acquired", "Continual retention", "contingency", "material",
    "paired_associate", "symbolic", boolean, role="guard",
    note="Did phase 1 reach criterion under the control run.")
add("pa_control_ac_trials_to_criterion", "Continual retention", "contingency",
    "material", "paired_associate", "symbolic", None, role="control",
    note="NaN by construction: the C list is never trained under the control.")
add("pa_control_ac_criterion_reached", "Continual retention", "contingency",
    "material", "paired_associate", "symbolic", None, role="control",
    note="False by construction, as above.")
for _cond in ("abac", "control"):
    for _key, _desc in (("phase2_trials", "Passes over the phase-2 list."),
                        ("ab_trials", "Passes over the AB list in phase 1."),
                        ("epochs", "Epochs per fit_sequence call.")):
        add(f"pa_{_cond}_{_key}", "Continual retention", "contingency", "material",
            "paired_associate", "symbolic", None, role="config", note=_desc)

# ======================= ONE-SHOT LEARNING =================================
add("convergence_epochs", "One-shot learning", "presentations", "material",
    "presentation_duration", "symbolic", inv_count, v=2,
    note="Epochs (presentations) to reach the MRR criterion. 1 -> 1.0.")
add("convergence_mrr", "One-shot learning", "presentations", "material",
    "presentation_duration", "symbolic", chance, v=2,
    note="Recall at the convergence checkpoint. Chance = 1/12 (fruit vocab).",
    chance_level=1/12)
add("convergence_span", "One-shot learning", "presentations", "material",
    "presentation_duration", "symbolic", ratio, v=1,
    note="Memory span at convergence, against the 6 attainable (7-item list).", ref=6.0)
add("converged", "One-shot learning", "presentations", "material",
    "presentation_duration", "symbolic", boolean, role="guard",
    note="Did the sweep reach the criterion at all.")

# Dimension: schema consistency (material). Guards decide whether it is scorable.
for rung in ("duplicate", "within", "across", "random"):
    add(f"schema_{rung}_relative_cost", "One-shot learning",
        "schema_consistency", "material", "schema_consistency", "symbolic",
        inv_count, v=2,
        note=f"Trials to acquire the {rung} rung / trials the SAME arm needed to "
             f"acquire the schema. The per-trial unit cancels, so this is the "
             f"cross-arm-comparable cost (McClelland Fig. 3 reads new-item error "
             f"against the arm's own end-of-acquisition line). <= 1 scores 1.0: "
             f"the new item cost no more than a known one. nan if either side "
             f"was censored.")
    add(f"schema_{rung}_trials_to_criterion", "One-shot learning",
        "schema_consistency", "material", "schema_consistency", "symbolic",
        inv_count, v=1,
        note=f"Trials to acquire the {rung} rung, in raw presentations. WITHIN-ARM "
             f"ONLY: one presentation is one pass, and one pass is ~6 LMS updates "
             f"for AHN but one averaged update for EP. Compare arms on "
             f"relative_cost, not this.")
    add(f"schema_{rung}_new_item_recall", "One-shot learning", "schema_consistency",
        "material", "schema_consistency", "symbolic", hi, v=2,
        note=f"Recall of the new {rung} transition after acquisition. This is "
             f"the acquisition read-out; unlike final_mrr it has no arithmetic "
             f"floor.")
    add(f"schema_{rung}_new_item_auc", "One-shot learning", "schema_consistency",
        "material", "schema_consistency", "symbolic", hi, v=2,
        note=f"Mean new-transition recall across all trials on the {rung} rung. "
             f"Continuous acquisition-speed measure; keeps ranging when "
             f"trials_to_criterion saturates at 1.")
    add(f"schema_{rung}_prefix_recall", "One-shot learning", "schema_consistency",
        "material", "schema_consistency", "symbolic", hi, v=1,
        note=f"Recall of the already-known transitions in the {rung} rung's own "
             f"training list -- within-sequence disruption, distinct from the "
             f"interference_* read-out on the other categories.")
    add(f"schema_{rung}_final_mrr", "One-shot learning", "schema_consistency",
        "material", "schema_consistency", "symbolic", hi, role="diagnostic",
        note=f"Sequence-mean MRR of the {rung} rung. DILUTED: (N-2)/(N-1) of "
             f"the scored transitions are already known, so this cannot fall "
             f"below schema_dilution_floor. Diagnostic only -- score "
             f"new_item_recall instead.")
    add(f"schema_{rung}_criterion_reached", "One-shot learning", "schema_consistency",
        "material", "schema_consistency", "symbolic", boolean, role="guard",
        note=f"Did the {rung} rung reach criterion.")
    add(f"schema_{rung}_base_mrr", "One-shot learning", "schema_consistency",
        "material", "schema_consistency", "symbolic", None, role="diagnostic",
        note="Pre-manipulation baseline for the rung.")
    add(f"schema_{rung}_criterion", "One-shot learning", "schema_consistency",
        "material", "schema_consistency", "symbolic", None, role="config",
        note="MRR criterion for the rung.")
    add(f"schema_{rung}_nearest_base_cosine", "One-shot learning", "schema_consistency",
        "material", "schema_consistency", "symbolic", None, role="diagnostic",
        note=f"Cosine from the {rung} item to its nearest base item, in the space "
             f"the decoder searches. The discriminability covariate: duplicate "
             f"~0.87, within ~0.72, across ~0.58, random ~0.29.")
    add(f"schema_{rung}_projection_ratio", "One-shot learning", "schema_consistency",
        "material", "schema_consistency", "symbolic", None, role="diagnostic",
        note="Model-free check that the rung ladder is structurally graded. A "
             "property of the encoder, not of the model.")
    # Interference read-out of the same manipulation -> Continual retention.
    for rel in ("same", "sibling", "far"):
        add(f"schema_{rung}_interference_{rel}", "Continual retention",
            "schema_consistency", "material", "schema_consistency", "symbolic",
            lo, v=1,
            note=f"Damage to {rel}-related prior material when the {rung} rung "
                 f"is learned. REQUIRES schema_interference_valid: under "
                 f"interference_protocol='extended' the new list is the host "
                 f"list plus one item, so learning it rehearses the host "
                 f"category and the `same` bucket measures rehearsal, not "
                 f"interference. Note DIMENSION_GUARDS gates this dimension on "
                 f"schema_resolved (an acquisition guard) only -- check "
                 f"schema_interference_valid by hand before quoting this.")
add("schema_consistency_speed_corr", "One-shot learning", "schema_consistency",
    "material", "schema_consistency", "symbolic", neg_corr, v=2,
    note="Correlation between rung consistency and acquisition speed over "
         "within/across/random ONLY (schema_corr_rungs) -- the headline number "
         "of the section. `duplicate` is excluded: it is bit-identical to a "
         "base item (nearest cosine 0.87) so its cost is decoder "
         "discriminability, not consistency; it is reported separately as "
         "schema_duplicate_excess_trials. Predicted NEGATIVE (more consistent -> "
         "fewer trials), hence neg_corr. Saturates to nan when every rung is "
         "one-shot; read schema_consistency_auc_corr then.")
add("schema_consistency_auc_corr", "One-shot learning", "schema_consistency",
    "material", "schema_consistency", "symbolic", hi, v=2,
    note="Correlation between rung consistency and mean new-transition recall, "
         "over within/across/random only (see speed_corr). Continuous "
         "counterpart of speed_corr; still ranges when every rung reaches "
         "criterion on trial 1.")
for k, nte in (
    ("schema_acquired", "Did the model acquire the schema at all."),
    ("schema_at_ceiling", "Are all rungs saturated (manipulation vacuous)."),
    ("schema_at_floor", "Are all rungs at floor (manipulation vacuous)."),
    ("schema_resolved", "Master guard: is the consistency manipulation resolvable."),
    ("schema_ladder_ordered", "Is the rung ordering monotone."),
    ("schema_acquisition_reached",
     "Did schema acquisition reach schema_criterion within the schema_trials "
     "ceiling. False => every relative_cost is nan."),
    ("schema_interference_valid",
     "Was the new item trained WITHOUT rehearsing the host category, so the "
     "interference_* columns measure interference rather than rehearsal. False "
     "under interference_protocol='extended'."),
    ("schema_sequence_identity_available",
     "Could this arm be told which sequence a training pair belongs to (needs "
     "n_context > 0 AND a fit_sequence accepting context_data). False for every "
     "arm today; no protocol passes an identifier either way. Reported so the "
     "extended/focused distinction is never read as something the model was "
     "told -- see audit pending item 18."),
    ("schema_auc_resolved", "Do the rungs separate on the continuous AUC read-out."),
    ("schema_criterion_below_dilution_floor",
     "Would a run scored on sequence-mean MRR be degenerate by arithmetic."),
    ("schema_probe_item_accuracy", "Item-level probe of schema knowledge."),
    ("schema_probe_category_accuracy", "Category-level probe of schema knowledge."),
):
    add(k, "One-shot learning", "schema_consistency", "material",
        "schema_consistency", "symbolic",
        hi if k.startswith("schema_probe_") else boolean,
        role="guard", note=nte)
add("schema_duplicate_excess_trials", "One-shot learning", "schema_consistency",
    "material", "schema_consistency", "symbolic", None, role="diagnostic",
    note="Trials the exact-copy rung needed beyond the mean of within/across/"
         "random. Positive = the item identical to a known one was HARDER to "
         "acquire: a discriminability cost of the same-space predict_next "
         "read-out (the paper's localist architecture never pays it). Reported "
         "here, kept out of the correlations.")
add("schema_acquisition_trials", "One-shot learning", "schema_consistency",
    "material", "schema_consistency", "symbolic", None, role="diagnostic",
    note="Trials (staircased) for base-list MRR to hold at schema_criterion. The "
         "denominator of every relative_cost; the ceiling value when unmet.")
add("schema_corr_rungs", "One-shot learning", "schema_consistency",
    "material", "schema_consistency", "symbolic", None, role="config",
    note="Rung set the headline correlations are computed over "
         "(within/across/random; `duplicate` reported separately).")
add("schema_acquisition_criterion", "One-shot learning", "schema_consistency",
    "material", "schema_consistency", "symbolic", None, role="config",
    note="Absolute base-list MRR the schema must hold to count as acquired.")
add("schema_dilution_floor", "One-shot learning", "schema_consistency",
    "material", "schema_consistency", "symbolic", None, role="config",
    note="(N-2)/(N-1) for the new list -- the value sequence-mean MRR cannot "
         "fall below. A criterion under this makes final_mrr unscorable.")
for k in ("schema_probe_item_chance", "schema_probe_category_chance"):
    add(k, "One-shot learning", "schema_consistency", "material",
        "schema_consistency", "symbolic", None, role="config",
        note="Chance level for the corresponding probe.")

# Dimension: presentations under the streamed regime (online arms only).
for k, nte in (
    ("online_mrr_1shot", "Recall after a single streamed presentation."),
    ("online_mrr_5pass", "Recall after 5 streamed presentations."),
    ("online_mrr_20pass", "Recall after 20 streamed presentations."),
):
    add(k, "One-shot learning", "presentations_streamed", "material",
        "online_convergence", "online_symbolic", hi, v=2, note=nte)
for k, nte in (
    ("online_span_1shot", "Span after a single streamed presentation."),
    ("online_span_5pass", "Span after 5 streamed presentations."),
    ("online_span_20pass", "Span after 20 streamed presentations."),
):
    add(k, "One-shot learning", "presentations_streamed", "material",
        "online_convergence", "online_symbolic", ratio, v=1, note=nte, ref=6.0)
add("isi_tolerance_mrr_sweep", "Continual retention", "interval", "material",
    "isi_tolerance", "online_symbolic", None, role="diagnostic",
    note="Recall across interposed-noise intervals (series-valued).")
add("isi_tolerance_span_sweep", "Continual retention", "interval", "material",
    "isi_tolerance", "online_symbolic", None, role="diagnostic",
    note="Span across interposed-noise intervals (series-valued).")

# ======================= PATTERN COMPLETION ================================
add("noise_tolerance_threshold", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", ratio, v=2,
    note="Gaussian sigma at which cued recall first drops below 0.5, over a "
         "sweep bounded at sigma = 1.0.", ref=1.0)
# The corruption half of P17's rollout probe (added 2026-09-06): noise on the
# initial cue only, then free-run. Diagnostics, never scored, for the same
# reason as `mask_completion_gap`: the level is free-running ability, which
# Serial order / `unrolling` already scores; the shape against sigma is new.
add("noise_rollout_auc", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="Mean autoregressive recall across the sigma sweep with the noise on "
         "the initial cue only, raw feedback: each prediction fed back verbatim. "
         "Runs before 2026-09-26 computed this under l2 feedback; identical for "
         "AHN, theta, tPC and Chen, not for DTS-ESN or EP.")
# LEGACY keys below: emitted by noise_invariance for part of 2026-09-26 and
# before, while the section still ran more than one feedback mode. Kept only so
# results files written then still score -- the scorer refuses unregistered keys.
add("noise_rollout_raw_auc", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="LEGACY (2026-09-26, briefly): the raw-feedback sweep while l2 was still "
         "the primary. Raw is now the only mode and is reported as noise_rollout_auc.")
add("noise_rollout_quantized_auc", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="LEGACY (pre-2026-09-26 runs): the sweep under quantized feedback. "
         "Dropped because snapping every step to a clean item makes each step a "
         "clean cued probe, so it restated the cued curve.")
add("noise_rollout_tolerance", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="Sigma at which rollout recall from a corrupted initial cue first drops "
         "below 0.5; the rollout twin of noise_tolerance_threshold.")
add("noise_completion_gap", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="Cued sweep mean minus rollout (raw feedback) sweep mean: what a corrupted "
         "initial cue costs once the chain runs on its own. P17's "
         "error-propagation clause for the corruption axis; read within-arm.")
add("noise_rollout_margin_auc", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="Mean rollout margin (target minus best competitor) across the sigma sweep, "
         "raw feedback, all steps.")
add("noise_rollout_margin_first_auc", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="Rollout margin at step 1 -- the corrupted cue's own prediction -- averaged "
         "over sigma.")
add("noise_rollout_margin_later_auc", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="Rollout margin at steps 2+, where the input is the model's own output: "
         "does the corruption propagate or does the chain recover.")
add("noise_rollout_margin_first_slope", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="Linear slope of the step-1 rollout margin against sigma (margin units per "
         "unit sigma; more negative = corruption eats the decision faster).")
add("noise_rollout_margin_later_slope", "Pattern completion", "cue_corruption", "probe",
    "noise_invariance", "symbolic", None, role="diagnostic",
    note="Linear slope of the steps-2+ rollout margin against sigma: the propagation "
         "of an initial corruption into the self-driven chain.")

# Dimension: cue_completeness, third probe property (2026-09-06): a whole
# MODALITY absent from the cue (memval/benchmarks/cue_availability.py). Reported
# as diagnostics under cue_completeness -- whether availability earns its own
# scored dimension (and what weight) is a paper decision not yet taken.
for _c in ("single", "multi"):
    for _cue, _what in (("both", "the full [symbol | audio] cue: the reference"),
                        ("symbolic", "the symbol alone, audio removed"),
                        ("audio", "the audio alone, symbol removed")):
        for _k, _nte in (
            ("recall", f"Exact next-item recall cued with {_what}; every item-audio "
                       f"pair with a successor, averaged. Chance 1/|vocab|."),
            ("rollout", f"Exact next-item recall along a rollout launched from each "
                        f"sequence's first item cued with {_what}; averaged over steps "
                        f"and sequences."),
            ("margin", f"Cued margin (target minus best competitor), {_what}."),
            ("audio_recall", f"Cued with {_what}: the prediction's AUDIO block names the "
                             f"next item's tone (chance 1/n_studied). For the "
                             f"symbol-only cue this is completion of the missing modality."),
        ):
            add(f"avail_{_c}_{_cue}_{_k}", "Pattern completion", "cue_completeness",
                "probe", "cue_availability", "symbolic", None, role="diagnostic",
                note=f"[{_c}] {_nte}")
    add(f"avail_{_c}_epochs_to_criterion", "Pattern completion", "cue_completeness",
        "probe", "cue_availability", "symbolic", None, role="config",
        note=f"[{_c}] Round-robin epochs until full-cue recall reached criterion.")
    add(f"avail_{_c}_criterion_reached", "Pattern completion", "cue_completeness",
        "probe", "cue_availability", "symbolic", boolean, role="guard",
        note=f"[{_c}] Full-cue criterion reached inside the budget.")
    add(f"avail_{_c}_chance_tone", "Pattern completion", "cue_completeness",
        "probe", "cue_availability", "symbolic", None, role="config",
        note=f"[{_c}] Chance for the audio-block read-out: 1 / number of studied tones.")
    add(f"avail_{_c}_tone_cosine_max", "Pattern completion", "cue_completeness",
        "probe", "cue_availability", "symbolic", None, role="config",
        note=f"[{_c}] Largest cosine between two items' tone codes (model-free).")
    # LEGACY: the sequence-level-tone design (2026-09-06 .. 2026-09-26). Kept only
    # so results files written then still score -- the scorer refuses unknown keys.
    for _k in ("symbolic_tone_cos", "symbolic_tone_recovery", "audio_membership",
               "audio_seq_id", "audio_rollout_membership",
               "audio_rollout_transition_validity", "chance_membership",
               "chance_seq_id", "exposure_vs_baseline"):
        add(f"avail_{_c}_{_k}", "Pattern completion", "cue_completeness", "probe",
            "cue_availability", "symbolic", None, role="diagnostic",
            note=f"[{_c}] LEGACY (tone per SEQUENCE, before 2026-09-26).")
for _k, _nte in (("avail_seq_len", "Items per sequence."), ("avail_n_sequences", "K in multi."),
                 ("avail_audio_gain", "Audio block weight before joint renormalisation."),
                 ("avail_n_bins", "Frequency bins of the audio block."),
                 ("avail_tone_sigma", "Tone receptive-field width (Hz)."),
                 ("avail_feedback_mode", "Rollout feedback (harness-controlled)."),
                 ("avail_chance_item", "1/|vocab|: chance for exact next-item recall."),
                 ("avail_epochs_to_criterion", "Mean over conditions.")):
    add(_k, "Pattern completion", "cue_completeness", "probe", "cue_availability",
        "symbolic", None, role="config", note=_nte)
add("avail_acquired", "Pattern completion", "cue_completeness", "probe",
    "cue_availability", "symbolic", boolean, role="guard",
    note="Full-cue recall >= 0.5 in both conditions; below that the availability "
         "read-outs have no association to complete from.")
add("tmaze_pc_coverage", "Pattern completion", "cue_point", "probe",
    "tmaze_completion", "spatial", hi, v=2,
    note="Fraction of ground-truth route points matched within eps by some "
         "recalled point. Timing-free and non-saturating (audit C1).")
add("tmaze_pc_divergence_step", "Pattern completion", "cue_point", "probe",
    "tmaze_completion", "spatial", ratio, v=2,
    note="First step whose aligned error exceeds eps, over the recall length. "
         "Continuous-space analogue of memory span; also scored under Serial order.",
    ref=11.0)
add("tmaze_pc_mse", "Pattern completion", "cue_point", "probe", "tmaze_completion",
    "spatial", None, role="diagnostic",
    note="EXCLUDED by docs/capacity_coverage_audit.md C1: centre-of-mass decoding "
         "pulls a lost readout to the arena middle, so MSE saturates in the "
         "failure regime and cannot separate 'somewhat wrong' from 'lost'.")
add("tmaze_pc_recall_len", "Pattern completion", "cue_point", "probe",
    "tmaze_completion", "spatial", None, role="config",
    note="Length of the recalled suffix; the denominator for divergence_step.")
add("tmaze_pc_eps", "Pattern completion", "cue_point", "probe", "tmaze_completion",
    "spatial", None, role="config", note="Arena tolerance for coverage / divergence.")

# Dimension: cue_completeness (probe) -- the STRUCTURAL half of P17, built
# 2026-09-02. A swept fraction of the cue's feature dimensions is removed and the
# rest left exact, against noise_invariance's graded corruption of all of them.
# Built on HierarchicalEncoder, not SymbolicEncoder: the latter's dimensions are
# a random basis, so masking them scores the embedding geometry rather than the
# arm -- the failure mode that got letter_noise deleted (D5).
#
# Every scored metric here is an IN-BOUND rate: it is read only where the
# model-free reference (`cue_identifiability`) says the masked cue still names
# its own item. Past that point several studied items share one input and the
# full-grid AUC picks up which colliding successor the arm happens to favour.
# The full-grid AUCs are kept as diagnostics so the two are never confused.
for _m, _v, _what in (
    ("random", 2, "a uniformly random fraction of the cue's features -- P17's "
                  "literal manipulation"),
    ("shared", 1, "the category features first, identity last: can the arm "
                  "complete from identity alone"),
    ("identity", 1, "the identity features first, category last: can it complete "
                    "from category context alone"),
):
    add(f"mask_{_m}_recall_in_bound", "Pattern completion", "cue_completeness",
        "probe", "cue_masking", "symbolic", hi, v=_v,
        note=f"Cued recall under masking of {_what}. Averaged only over masked "
             f"fractions at which the cue still identifies its item, so it "
             f"cannot be inflated by cue collisions. Read "
             f"mask_{_m}_in_bound_points for how many of the grid that was.")
    add(f"mask_{_m}_auc", "Pattern completion", "cue_completeness", "probe",
        "cue_masking", "symbolic", None, role="diagnostic",
        note=f"Full-grid mean of the same curve. DIAGNOSTIC: past "
             f"mask_{_m}_identifiable_to the masked cues of several studied "
             f"items are identical, so this rewards collision luck. Its scored "
             f"companion is mask_{_m}_recall_in_bound.")
    add(f"mask_{_m}_rollout_auc", "Pattern completion", "cue_completeness",
        "probe", "cue_masking", "symbolic", None, role="diagnostic",
        note=f"P17's second probe: autoregressive recall from a masked INITIAL "
             f"cue only. Measured and reported, deliberately not scored -- its "
             f"level is set by the arm's free-running ability, which Serial "
             f"order/`unrolling` already scores, so scoring it here would count "
             f"the same weakness twice. What this section adds is its shape "
             f"across the masking axis.")
    add(f"mask_{_m}_completion_advantage", "Pattern completion",
        "cue_completeness", "probe", "cue_masking", "symbolic", None,
        role="diagnostic",
        note=f"Recall minus the matched-filter reference on the same raw cue, "
             f"over the full grid. Positive = the arm recovers targets the "
             f"degraded cue no longer specifies, i.e. completion rather than cue "
             f"matching. A contrast, so reported rather than scored.")
    add(f"mask_{_m}_identifiable_to", "Pattern completion", "cue_completeness",
        "probe", "cue_masking", "symbolic", None, role="config",
        note=f"Largest masked fraction at which the {_m}-masked cue still names "
             f"its own item, model-free. The window every in-bound metric uses.")
    add(f"mask_{_m}_in_bound_points", "Pattern completion", "cue_completeness",
        "probe", "cue_masking", "symbolic", None, role="config",
        note="Grid points inside that window; the denominator of the in-bound rate.")
    add(f"mask_{_m}_margin_auc", "Pattern completion", "cue_completeness",
        "probe", "cue_masking", "symbolic", None, role="diagnostic",
        note="Full-grid margin; the in-bound version is the scored one.")
add("mask_random_margin_in_bound", "Pattern completion", "cue_completeness",
    "probe", "cue_masking", "symbolic", hi, v=1,
    note="Target-minus-best-competitor cosine under random masking, in bound. "
         "Kept beside the recall rate because it keeps resolving after recall "
         "has floored, which is where a masking curve gets interesting.")
for _m in ("shared", "identity"):
    add(f"mask_{_m}_margin_in_bound", "Pattern completion", "cue_completeness",
        "probe", "cue_masking", "symbolic", None, role="diagnostic",
        note="Margin under block masking. Diagnostic: the random-mode margin is "
             "the scored one, and scoring all three would triple-count one "
             "quantity across three correlated views of it.")
add("mask_random_tolerance", "Pattern completion", "cue_completeness", "probe",
    "cue_masking", "symbolic", ratio, v=2,
    note="Largest fraction of the cue that can be discarded with recall still "
         "at or above 0.5. The structural counterpart of "
         "noise_tolerance_threshold, and the same sign convention: higher is "
         "more robust.", ref=1.0)
for _m in ("shared", "identity"):
    add(f"mask_{_m}_tolerance", "Pattern completion", "cue_completeness", "probe",
        "cue_masking", "symbolic", None, role="diagnostic",
        note="Block-masking tolerance. Diagnostic because it is capped by the "
             "block's own size: once a block is exhausted the mode spills into "
             "the other and the number stops being about that block.")
add("mask_block_asymmetry", "Pattern completion", "cue_completeness", "probe",
    "cue_masking", "symbolic", None, role="diagnostic",
    note="Recall under identity-masking minus recall under shared-masking, in "
         "the window where both remove the SAME NUMBER of features and both "
         "cues are still identifiable. An architectural fingerprint, never a "
         "quality, so it is reported and not scored: >0 leans on category "
         "structure, <0 on identity features, ~0 is the holistic-retrieval null "
         "(the cue is a bag of dimensions to this arm).")
add("mask_completion_gap", "Pattern completion", "cue_completeness", "probe",
    "cue_masking", "symbolic", None, role="diagnostic",
    note="Cued AUC minus rollout AUC, meaned over modes: what an initially "
         "fragmentary cue costs once the chain has to run on its own. P17's "
         "error-propagation clause. Confounded with free-running ability, so "
         "read it as a within-arm contrast.")
add("mask_identifiability_auc", "Pattern completion", "cue_completeness",
    "probe", "cue_masking", "symbolic", None, role="control",
    note="The model-free reference curve itself, averaged. A property of the "
         "stimulus, identical for every arm at a given configuration; present "
         "so a scorecard reader can see what the arms were measured against.")
add("mask_resolved", "Pattern completion", "cue_completeness", "probe",
    "cue_masking", "symbolic", boolean, role="guard",
    note="Did the section pose its question? False if the list was never "
         "acquired, if masking never moved recall, or if the cue stopped naming "
         "its item almost immediately. Read mask_at_floor / mask_at_ceiling / "
         "mask_identifiable_region for which.")
for _k, _n in (
    ("mask_acquired", "Was the clean-cue list learned at all. Every masked "
                      "read-out is vacuous below this."),
    ("mask_at_ceiling", "In-bound recall is >= 0.95 in every mode: the arm "
                        "completes wherever the cue still names its item, so the "
                        "scored recall metrics carry no information about it. "
                        "Not a failure -- read mask_random_tolerance and "
                        "mask_random_margin_in_bound, which still range."),
    ("mask_margin_ranges", "Does the margin still separate at that ceiling. One "
                           "of the two clauses that keep mask_resolved True when "
                           "recall is saturated."),
    ("mask_at_floor", "Clean-cue recall never rose above chance."),
    ("mask_identifiable_region", "Was there a usable window at all "
                                 "(mask_identifiable_to >= 0.25)."),
):
    add(_k, "Pattern completion", "cue_completeness", "probe", "cue_masking",
        "symbolic", None, role="guard", note=_n)
for _k, _n in (
    ("mask_seq_len", "List length; with mask_n_features it sets the load."),
    ("mask_n_features", "Embedding dimension of the hierarchy."),
    ("mask_alpha", "P/N. Kept under the Hopfield-class knee ~0.138 so a "
                   "substrate-capacity ceiling is not read as a masking result."),
    ("mask_active_features", "Active features per cue; the sweep's granularity."),
    ("mask_shared_features", "Size of the shared (category) block."),
    ("mask_identity_features", "Size of the identity block."),
    ("mask_block_comparable_frac", "Fraction up to which the two block modes "
                                   "remove the same number of features."),
    ("mask_block_window_frac", "Upper fraction of the asymmetry window."),
    ("mask_block_window_points", "Grid points in the asymmetry window."),
    ("mask_identifiable_to", "Largest fraction at which a randomly-masked cue "
                             "still names its item."),
    ("mask_chance_level", "1 / vocabulary size."),
    ("mask_clean_recall", "Recall at fraction 0; the sweep's own baseline."),
    ("mask_list_scope", "'across' (categories span the list) or 'within'."),
    ("mask_renormalize", "Whether masked cues are re-L2-normalised. False "
                         "reintroduces the magnitude confound on purpose."),
    ("mask_feedback_mode", "Rollout feedback projection."),
    ("mask_n_items", "Vocabulary size of the hierarchy."),
):
    add(_k, "Pattern completion", "cue_completeness", "probe", "cue_masking",
        "symbolic", None, role="config", note=_n)

# ======================= SEQUENCE DISAMBIGUATION ===========================
# Dimension: contextual overlap x external support (material).
add("tmaze_disamb_full_branch_acc", "Sequence disambiguation", "contextual_overlap",
    "material", "tmaze_disambiguation", "spatial", chance, v=1,
    note="Legacy fully-cued block (odour on every step). Chance = 0.5.",
    chance_level=0.5)
add("tmaze_disamb_full_confusion", "Sequence disambiguation", "contextual_overlap",
    "material", "tmaze_disambiguation", "spatial", lo, v=1,
    note="Rate of entering the other route's arm in the fully-cued block.")
add("tmaze_disamb_mec_only_branch_acc", "Sequence disambiguation",
    "contextual_overlap", "material", "tmaze_disambiguation", "spatial", None,
    role="control",
    note="Odour channel ablated: place code alone cannot separate identical stems, "
         "so 0.5 is the intended floor, not a model failure.")
add("tmaze_disamb_mec_only_confusion", "Sequence disambiguation",
    "contextual_overlap", "material", "tmaze_disambiguation", "spatial", None,
    role="control", note="Confusion under the ablated control; 0.5 is intended.")

_GRADED_ROLE = {"concurrent": "score"}          # withdrawn rungs handled below
for tag in ("concurrent", "delay0", "delay2", "delay4", "delay6", "delay7"):
    prot = tag != "concurrent"
    role = "protocol_limited" if prot else "score"
    nte_suffix = ("" if not prot else
                  " EXCLUDED from the capacity score by docs/capacity_coverage_audit.md "
                  "C2: with the odour withdrawn, nothing in the memoryless predict_next "
                  "probe can carry the discriminator across the cue-free stretch, so a "
                  "floor here is a protocol finding, not a model ranking.")
    add(f"tmaze_disamb_graded_{tag}_divergence_accuracy", "Sequence disambiguation",
        "contextual_overlap", "material", "tmaze_disambiguation", "spatial",
        chance, v=2, role=role, chance_level=0.5,
        note="Correct branch at the divergence point (episode-averaging removed)." + nte_suffix)
    add(f"tmaze_disamb_graded_{tag}_branch_accuracy", "Sequence disambiguation",
        "contextual_overlap", "material", "tmaze_disambiguation", "spatial",
        chance, v=1, role=role, chance_level=0.5,
        note="Correct arm averaged over the post-divergence suffix." + nte_suffix)
    add(f"tmaze_disamb_graded_{tag}_divergence_margin", "Sequence disambiguation",
        "contextual_overlap", "material", "tmaze_disambiguation", "spatial",
        hi, v=2, role=role,
        note="Normalised margin toward the correct branch at divergence. Keeps "
             "resolving after accuracy has floored." + nte_suffix)
    add(f"tmaze_disamb_graded_{tag}_branch_margin", "Sequence disambiguation",
        "contextual_overlap", "material", "tmaze_disambiguation", "spatial",
        hi, v=1, role=role, note="Normalised margin over the suffix." + nte_suffix)
    add(f"tmaze_disamb_graded_{tag}_shared_stretch_error", "Sequence disambiguation",
        "contextual_overlap", "material", "tmaze_disambiguation", "spatial",
        None, role="control",
        note="Trajectory error on the shared stem, where both routes agree. Isolates "
             "ordinary rollout drift from cue loss; a control, not a score. NaN when "
             "the rung leaves no cue-free shared steps.")

# Guidance-withdrawal sweep (2026-09-08, docs/disambiguation_design.md S3.4):
# material fixed (odour on throughout), the probe walks the arm up the stem
# with observe() to a cue point and hands the rollout to the arm's own
# recall(), unclamped, d steps before the fork. Recall is unclamped, so an arm
# that outputs the odour block re-predicts it and carries the cue in its own
# loop -- a memoryless arm scores 1.0 here. That is a finding about cue
# carriage in the loop, not latent state, so the rows are diagnostic and
# unscored; the odour-withdrawn rungs above remain the latent-state question.
for d in range(8):
    _g = f"tmaze_disamb_guided_d{d}"
    add(f"{_g}_arm_accuracy", "Sequence disambiguation", "contextual_overlap",
        "material", "tmaze_disambiguation", "spatial", chance, role="diagnostic",
        chance_level=0.5,
        note=f"Recall starts {d} step(s) before the fork; nearest arm at any position, "
             "ties fail. Mean over 5 model seeds.")
    add(f"{_g}_arm_accuracy_sd", "Sequence disambiguation", "contextual_overlap",
        "material", "tmaze_disambiguation", "spatial", None, role="control",
        note="s.d. over model seeds of the arm accuracy.")
    add(f"{_g}_divergence_margin", "Sequence disambiguation", "contextual_overlap",
        "material", "tmaze_disambiguation", "spatial", hi, role="diagnostic",
        note=f"Normalised margin at the fork step, recall from {d} step(s) before it.")
    add(f"{_g}_divergence_margin_sd", "Sequence disambiguation", "contextual_overlap",
        "material", "tmaze_disambiguation", "spatial", None, role="control",
        note="s.d. over model seeds of the fork-step margin.")
    add(f"{_g}_branch_margin", "Sequence disambiguation", "contextual_overlap",
        "material", "tmaze_disambiguation", "spatial", hi, role="diagnostic",
        note="Normalised margin averaged over the arm.")
    add(f"{_g}_positional_error", "Sequence disambiguation", "contextual_overlap",
        "material", "tmaze_disambiguation", "spatial", None, role="diagnostic",
        note="Mean decoded distance to the step-matched target along the arm.")
    add(f"{_g}_positional_error_sd", "Sequence disambiguation", "contextual_overlap",
        "material", "tmaze_disambiguation", "spatial", None, role="control",
        note="s.d. over model seeds of the positional error.")
    add(f"{_g}_stem_error", "Sequence disambiguation", "contextual_overlap",
        "material", "tmaze_disambiguation", "spatial", None, role="control",
        note="Decoded error on the free-run stem stretch before the fork; NaN at d=0.")
add("tmaze_disamb_guided_n_seeds", "Sequence disambiguation", "contextual_overlap",
    "material", "tmaze_disambiguation", "spatial", None, role="control",
    note="Model seeds in the guidance sweep (1 for arms without a seed kwarg).")
add("tmaze_disamb_guided_rollout_mode", "Sequence disambiguation", "contextual_overlap",
    "material", "tmaze_disambiguation", "spatial", None, role="control",
    note="The arm's declared recall() rollout mode used by the guidance sweep.")

# Dimension: item similarity (material).
add("mrr_high_similarity", "Sequence disambiguation", "item_similarity", "material",
    "semantic_similarity", "symbolic", None, role="diagnostic",
    note="Within-category list recall. SUPERSEDED as a score: since 2026-09-02 this\n         list is trained to an MRR criterion, so the reported MRR is pinned just\n         above that criterion by construction and carries no dynamic range. What\n         varies with overlap is the EXPOSURE it took -- see similarity_exposure_cost.")
add("mrr_low_similarity", "Sequence disambiguation", "item_similarity", "material",
    "semantic_similarity", "symbolic", None, role="diagnostic",
    note="Across-category list recall. SUPERSEDED as a score: since 2026-09-02 this\n         list is trained to an MRR criterion, so the reported MRR is pinned just\n         above that criterion by construction and carries no dynamic range. What\n         varies with overlap is the EXPOSURE it took -- see similarity_exposure_cost.")
add("similarity_effect_mrr_drop", "Sequence disambiguation", "item_similarity",
    "material", "semantic_similarity", "symbolic", None, role="diagnostic",
    note="MRR(low) - MRR(high) at the single fixed variance. Reads 0 whenever both "
         "sides are at ceiling, which is exactly this arm's case: the graded sweep "
         "is where the effect lives.")

# Dimension: contextual overlap, symbolic arm (material).
# The symbolic section is where axes B and C actually live: the spatial
# generator grades discriminability by rotating a one-hot vector, which is
# capped at two discriminators, while category_variance grades it generatively
# and supports N > 2. See docs/disambiguation_design.md sec 3.
add("symdis_cued_divergence_accuracy", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    chance, v=2, chance_level=0.25,
    note="Forced choice among N episodes' successors at the divergence step, "
         "with the discriminator still present AT the decision. Isolates "
         "discriminability from memory: a failure here is not a memory failure. "
         "Chance is 1/N and defaults to N=4.")
add("symdis_cued_divergence_margin", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    hi, v=1,
    note="Target similarity minus best competitor at the divergence step. Keeps "
         "resolving after accuracy has floored; ties count as failures.")
add("symdis_withdrawn_divergence_accuracy", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    chance, v=2, role="protocol_limited", chance_level=0.25,
    note="Same choice with the discriminator withdrawn before the decision. "
         "EXCLUDED from the score unless the arm declares StatePrimeable: with "
         "no carried state the probe cannot deliver the discriminator across the "
         "cue-free stretch, so a floor is a protocol finding. Read alongside "
         "symdis_state_primed.")
add("symdis_withdrawn_divergence_margin", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    hi, v=1, role="protocol_limited",
    note="Margin under withdrawal. Same protocol caveat as the accuracy.")
add("symdis_similarity_tolerance_threshold", "Sequence disambiguation",
    "item_similarity", "material", "symbolic_disambiguation", "symbolic",
    lo, v=2,
    note="Axis B headline: the REALISED mean pairwise cosine of the "
         "discriminators at which accuracy first drops below halfway between "
         "chance and ceiling. Lower is better (tolerates more confusable "
         "contexts). Reported against realised cosine rather than the "
         "category_variance dial, because that mapping is nonlinear and "
         "saturating. This is what semantic_similarity was supposed to measure "
         "and could not: there, transitions are bijective so no cue ever demands "
         "two successors and both conditions sit at ceiling.")
add("symdis_max_episodes_above_chance", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    ratio, v=2, ref=8.0,
    note="Axis C headline: the largest N confusable episodes sharing one stretch "
         "still scored above 1/N. The disambiguation analogue of the load sweep, "
         "and unavailable spatially -- a 2-D rotation cannot place N "
         "discriminators at controlled mutual similarity. CENSORED at the "
         "grid's largest N: a value equal to the grid maximum is a lower bound, "
         "not a capacity. Extend the grid via --benchmark-args before reading "
         "it as one.")
add("symdis_context_graded_confusion_index", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    hi, v=1, role="diagnostic",
    note="Correlation between P(confusing i with j) and cos(disc_i, disc_j). "
         "Positive means errors go to the contextually NEAREST neighbour, which "
         "is evidence the discriminator is represented at all; uniform confusion "
         "is a different and weaker result. NaN for N == 2, where a single "
         "off-diagonal pair has no variance to correlate.")
add("symdis_support_needed", "Sequence disambiguation", "contextual_overlap",
    "material", "symbolic_disambiguation", "symbolic", lo, v=2,
    note="Smallest cue DURATION reaching halfway between chance and ceiling, at "
         "a FIXED 2-step gap to the decision. Lower is better. Sweeping "
         "zone_fraction alone cannot produce this: the zone starts at the "
         "corridor entrance, so a shorter cue is also an earlier withdrawal and "
         "duration is confounded with delay. generators/overlap.zone_params_for "
         "solves the two apart. NaN when no duration in the grid reaches the "
         "crossing.")
add("symdis_max_delay_at_fixed_support", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    hi, v=2, role="protocol_limited",
    note="Longest gap between cue offset and decision still reaching halfway, at "
         "a FIXED 2-step cue. The retention half of the pair above. "
         "protocol_limited for the same reason as the withdrawn rows: an arm "
         "with no carried state cannot express any delay > 0, so read it "
         "alongside symdis_state_primed.")
add("symdis_max_shared_len_persistent", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    ratio, v=2, ref=10.0,
    note="Longest shared stretch (steps, suffix fixed) at which the fork is "
         "still chosen above halfway with the discriminator ON throughout. "
         "P22's 'length of the shared stretch, varied independently'. CENSORED "
         "at the grid maximum (10): a value equal to it is a lower bound.")
add("symdis_max_shared_len_onset", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    ratio, v=2, ref=10.0, role="protocol_limited",
    note="Same sweep with a 2-step cue at the corridor entrance only, so the "
         "gap to the fork grows with the stretch -- the case where the "
         "separation must be carried internally. protocol_limited: read with "
         "symdis_state_primed, as for the withdrawn rows.")
add("symdis_load_orthogonal_accuracy", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    None, role="control", chance_level=0.125,
    note="The load sweep repeated with EXACTLY ORTHONORMAL discriminators, at "
         "the grid's largest N. Maximum separability, so discriminability is "
         "removed as a limiting factor. A control, not a score.")
add("symdis_load_disambiguation_cost", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    lo, v=2,
    note="Orthogonal-control accuracy minus category accuracy at the largest N. "
         "This is the ATTRIBUTION metric: without it a falling load curve cannot "
         "be told apart from ordinary capacity running out. ~0 means the "
         "degradation is capacity and would happen with any discriminators; "
         "large means it is genuinely a disambiguation failure. Can go negative "
         "by noise when both sides sit at the floor.")
add("symdis_middle_informative_accuracy", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    chance, v=2, chance_level=0.25,
    note="Shared MIDDLE rather than shared prefix: each episode has a unique "
         "prefix, then the shared corridor, with DIFFERING discriminators in the "
         "zone. Named 'informative', not 'cued', because in this suite cued "
         "recall names a PROBE; this rung and the endogenous one use the "
         "identical probe and differ only in whether modality B carries "
         "information. "
         "The Agster paradigm proper. Everything else in this section pins "
         "shared_position=0, which makes the discriminator the only thing that "
         "identifies an episode.")
add("symdis_middle_endogenous_accuracy", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    chance, v=2, role="protocol_limited", chance_level=0.25,
    note="Shared middle with EVERY episode given the same discriminator, so "
         "modality B carries zero information and the unique prefix is the only "
         "thing that distinguishes them. The hardest rung in the capacity. "
         "Meaningful ONLY for a StatePrimeable arm: with no carried state the "
         "probe input is identical across episodes and the floor is a protocol "
         "artefact, not a model result.")
add("symdis_middle_prefix_len", "Sequence disambiguation", "contextual_overlap",
    "material", "symbolic_disambiguation", "symbolic", None, role="config",
    note="Length of the unique prefix in the shared-middle condition, in steps. "
         "How much evidence the endogenous condition actually supplies.")
add("symdis_shared_stretch_accuracy", "Sequence disambiguation",
    "contextual_overlap", "material", "symbolic_disambiguation", "symbolic",
    None, role="guard",
    note="One-step accuracy over the shared stretch, where every episode agrees. "
         "A control, never a score: episode-averaged scoring gets EASIER as the "
         "shared stretch lengthens, so this is reported separately. A low value "
         "voids the divergence numbers -- the arm never learned the corridor it "
         "is supposed to carry a cue across.")
add("symdis_state_primed", "Sequence disambiguation", "contextual_overlap",
    "material", "symbolic_disambiguation", "symbolic", None, role="config",
    note="Whether the arm declares StatePrimeable, so the shared stretch was "
         "delivered through observe(). False means the withdrawn rows rest on "
         "the last cued step alone and are not comparable with a primed row.")

# ======================= EXPOSURE BASELINE (cross-cutting) =================
# Not a capacity claim: provenance for every other number in the run. Each
# criterion-referenced section settles its own exposure on its own material,
# which keeps ARMS comparable within a section. These record the other axis --
# how far each section sits from one reference exposure for the same arm -- so a
# cross-section claim is not made silently between two differently-trained
# models.
add("exposure_baseline_epochs", "Continual retention", "load", "material",
    "presentation_duration", "symbolic", None, role="config",
    note="Reference exposure for this arm: epochs to criterion on the shared "
         "7-word list, same staircase and criterion every section uses. A "
         "REFERENCE, not a budget -- sections are not pinned to it, because "
         "'converged' is a property of (model, material), not of the model "
         "alone. Pinning would start a heavier section undertrained and make its "
         "failures ambiguous between the manipulation and the undertraining.")
add("exposure_baseline_reached", "Continual retention", "load", "material",
    "presentation_duration", "symbolic", None, role="guard",
    note="Did the reference itself reach criterion? False makes every ratio "
         "below a lower bound.")
add("exposure_deviation_max", "Continual retention", "load", "material",
    "presentation_duration", "symbolic", None, role="diagnostic",
    note="Largest ratio between any section's exposure and the reference, taken "
         "in whichever direction is larger. 1.0 means every section trained the "
         "arm at the same point.")
add("exposure_deviation_section", "Continual retention", "load", "material",
    "presentation_duration", "symbolic", None, role="config",
    note="Which section produced exposure_deviation_max.")
add("exposure_within_band", "Continual retention", "load", "material",
    "presentation_duration", "symbolic", None, role="guard",
    note="Whether every section's exposure sits inside "
         "symbolic_pipeline.EXPOSURE_DEVIATION_BAND (8x) of the reference. "
         "False does NOT invalidate a section on its own terms -- each is still "
         "measured at its own criterion -- but it does mean cross-section "
         "comparison for this arm is not like-for-like, and the run prints a "
         "warning naming the section.")

# ======================= SERIAL ORDER ======================================
add("max_memory_span", "Serial order", "unrolling", "material", "sequence_length",
    "symbolic", ratio, v=2, ref=10.0,
    note="Best span under raw autoregressive feedback, over the 10 attainable at "
         "the grid's longest list (L = 11).")
add("max_memory_span_l2", "Serial order", "unrolling", "probe", "sequence_length",
    "symbolic", ratio, v=1, ref=10.0,
    note="Span with the fed-back state L2-renormalised: removes magnitude drift.")
add("max_memory_span_quantized", "Serial order", "unrolling", "probe",
    "sequence_length", "symbolic", ratio, v=1, ref=10.0,
    note="Span with codebook feedback: the drift-free upper bound. The gap to raw "
         "separates representational degradation from readout drift.")

# The promised decomposition of "associations present but unrolling collapses"
# (audit sec 5, Serial order). Not yet emitted: needs S-O2 plus the cued span
# added to series.length_sweep. Registered here so they score the moment they land.
add("unrolling_gap", "Serial order", "unrolling", "probe", "sequence_length",
    "symbolic", lo, v=2,
    note="1 - span(rollout, raw) / span(cued), meaned over the length grid and "
         "measured at each length's own CUED criterion. The literal paragraph-11 "
         "clause, read where its antecedent holds: the associations are present by "
         "construction at that point. 0 = unrolling costs nothing. Measuring it at a "
         "fixed epoch count instead would compare arms at different points on their "
         "learning curves -- MODEL_REGISTRY's defaults span 1 to 300 epochs.")
add("unrolling_exposure_ratio", "Serial order", "unrolling", "material",
    "sequence_length", "symbolic", log_ratio, v=2,
    note="Exposure for ROLLOUT to reach criterion divided by exposure for CUED "
         "recall to reach it, meaned over the length grid. Self-referenced, so it is "
         "comparable across arms that converge at wildly different rates. 1.0 = "
         "unrolling comes free with the associations; large = a sample-efficiency "
         "gap rather than an architectural one; censored = a genuine collapse. "
         "Scored as 1/(1+log2(ratio)) because exposure is a log quantity.")
add("cascade_recovery_rate", "Serial order", "unrolling", "probe", "sequence_length",
    "symbolic", hi, v=2,
    note="P(correct at step k+1 | error at step k) under free rollout. This is "
         "`error_cascade`: 0 means an error is terminal and the chain never "
         "re-enters.")
add("cascade_conditional_ratio", "Serial order", "unrolling", "probe",
    "sequence_length", "symbolic", None, role="diagnostic",
    note="P(correct k+1 | correct k) / P(correct k+1 | error k). Magnitude of "
         "propagation; unbounded above, so read it, do not score it.")
add("rollout_margin_at_break", "Serial order", "unrolling", "probe",
    "sequence_length", "symbolic", hi, v=1,
    note="Target-minus-best-competitor cosine at the step where the span ends. "
         "Resolves below the accuracy floor, where span is already 0 and flat.")

# ---- establishment: was order established at all, separately from association.
# Needs S-O2 (decoded identities out of measure_recall_associative). Doubles as a
# read-before guard: at chance here, the unrolling numbers are uninterpretable.
add("order_given_item", "Serial order", "establishment", "probe", "sequence_length",
    "symbolic", chance, v=2, chance_level=1 / 29,
    note="P(decode is the correct next item | decode is a studied item), meaned "
         "over the length grid. Chance = 1/(L-1); 1/29 is the grid's longest list "
         "(L = 30; grid 10/20/30 since 2026-09-25, was 3..11 with chance 0.1). "
         "High membership with chance-level order here is 'order never "
         "established' as distinct from 'nothing stored'.")
add("list_membership_rate", "Serial order", "establishment", "probe",
    "sequence_length", "symbolic", hi, v=1,
    note="Fraction of cued predictions decoding to ANY studied item. The "
         "denominator for order_given_item; on its own it is a storage read, not "
         "an order read.")
add("establishment_break_length", "Serial order", "establishment", "material",
    "sequence_length", "symbolic", ratio, v=1, ref=30.0,
    note="Longest list length at which order_given_item still holds at >= 0.75, "
         "over the grid's longest list.")

# ---- binding_ordinal: items recalled, but out of sequence. Needs S-O2.
# d = (study position of the decode) - (true target position).
add("order_error_fraction", "Serial order", "binding_ordinal", "probe",
    "sequence_length", "symbolic", hi, v=2,
    note="Of all cued failures, the fraction that are in-list transpositions "
         "rather than omissions or intrusions. Exactly the separation memory_span "
         "cannot make. Higher = the failure is one of order, not of storage.")
add("transposition_locality", "Serial order", "binding_ordinal", "probe",
    "sequence_length", "symbolic", hi, v=2,
    note="Of transposition errors, the fraction at |d| = 1. Graded locality: "
         "1.0 = order errors only ever swap neighbours.")
add("transposition_asymmetry", "Serial order", "binding_ordinal", "probe",
    "sequence_length", "symbolic", None, role="diagnostic",
    note="Forward share, count(d=+1) / [count(d=+1) + count(d=-1)]. 0.5 = a "
         "symmetric pairwise associator; above = the forward bias the STDP "
         "argument of paragraph 10 predicts. An architectural fingerprint, not a "
         "quality: never scored.")
add("intrusion_rate", "Serial order", "binding_ordinal", "probe", "sequence_length",
    "symbolic", lo, v=1,
    note="Decode falls outside the studied list. Lower is better. Reads a trivial "
         "0.00 when the cued curve is perfect, which is why it sits behind the "
         "failures_observed guard with the rest of the dimension rather than "
         "scoring on its own.")


# ---- serial-order probe (bin/probe_serial_order.py). These are the audit's
# specified metrics computed off decoded identities; until S-O2 lands in
# `measure_recall_associative` they come from the probe rather than the suite,
# and every row says so.
add("rollout_criterion_reached", "Serial order", "unrolling", "probe",
    "sequence_length", "symbolic", boolean, role="guard",
    note="Did rollout reach criterion at every list length inside the staircase "
         "budget? False means unrolling_exposure_ratio is censored at some length "
         "and the mean understates the cost.")
for k, nte in (
    ("failures_observed",
     "Was any cued failure observed at all? Gates the whole binding_ordinal "
     "dimension, whose metrics are all conditionals on an error."),
):
    add(k, "Serial order", "binding_ordinal", "probe", "sequence_length", "symbolic",
        boolean, role="guard", note=nte)
for k, nte in (
    ("n_cued_failures", "Cued failures observed across the length grid."),
    ("n_cued_probes", "Cued probes across the length grid; the denominator."),
):
    add(k, "Serial order", "binding_ordinal", "probe", "sequence_length", "symbolic",
        None, role="config", note=nte)


# ---- Serial order, metric time. Two dimensions, together half the capacity.
# Neither is a wired section yet; both are gated on a declared capability, so an
# ordinal-clocked arm reports NOT APPLICABLE rather than a zero.
for k, v_, nte in (
    ("interval_discrimination_acc", 2,
     "Accuracy at choosing the right continuation when the elapsed gap is the only "
     "thing telling two otherwise identical prefixes apart. Chance = 0.5 for the "
     "two-branch design in examples/dts_esn_interval_demo.py."),
    ("interval_switch_sharpness", 1,
     "How abruptly the read-out switches branch across an interval sweep. A model "
     "that honours the interval switches near the trained gap; one that has merely "
     "absorbed some slow drift crosses over gradually."),
    ("interval_encoded", 1,
     "Guard-style summary: did the interval change the prediction at all, against a "
     "same-stream control run with the gaps removed."),
):
    add(k, "Serial order", "interval_retention", "material", "interval_retention",
        "symbolic", chance if k.endswith("_acc") else hi, v=v_,
        chance_level=0.5 if k.endswith("_acc") else 0.0, note=nte)
for k, v_, nte in (
    ("tempo_reproduction_error", 2,
     "Absolute error between generated and trained inter-item gap, normalised by "
     "the trained gap, under autonomous rollout. Lower is better."),
    ("rhythm_pause_position_acc", 2,
     "Does the longest generated gap fall at the position the trained sequence "
     "paused? The rhythm test of examples/dts_esn_generation_demo.py."),
    ("peak_time_error", 1,
     "Error in the predicted time-to-next-event. The MacDonald 2011 time-cell "
     "read-out named in the audit."),
    ("weber_slope", 1,
     "Slope of timing variability against interval length. Scalar-timing "
     "signature; diagnostic of the mechanism rather than of quality."),
):
    add(k, "Serial order", "interval_generation", "material", "interval_generation",
        "symbolic", lo if k.endswith("_error") else hi, v=v_,
        role="diagnostic" if k == "weber_slope" else "score", note=nte)


# ---- supporting read-outs of the two metric-time sections ----------------
# Controls and configuration, never scores. The two controls are the reason the
# dimensions can be believed at all, so they are reported rather than dropped.
add("interval_control_acc", "Serial order", "interval_retention", "probe",
    "interval_retention", "symbolic", None, role="control",
    note="Discrimination accuracy of the SAME arm trained on the SAME event "
         "stream with the intervals removed. This is the control that makes "
         "interval_discrimination_acc mean anything: if the ordinal-clocked "
         "twin also discriminates, the design leaked something other than time. "
         "Expected at or below chance (0.5).")
add("interval_crossing_gap", "Serial order", "interval_retention", "probe",
    "interval_retention", "symbolic", None, role="diagnostic",
    note="Gap, in seconds, at which the read-out switches continuation "
         "(geometric midpoint of the bracketing sweep points). Meaningful only "
         "against the two trained gaps it should fall between.")
add("interval_ambiguous_gap_collapse", "Serial order", "interval_generation",
    "probe", "interval_generation", "symbolic", None, role="control",
    note="The timing head run on interval-as-CUE's ambiguous stream, where the "
         "gap is not determined by the items. 1.0 = the predicted gap sits at "
         "the mean of the two trained gaps, i.e. it collapsed as it should. "
         "This is the evidence that 5.4 and 5.5 need different streams and are "
         "two sections rather than two read-outs of one.")
for _k, _dim in (("interval_n_reps", "interval_retention"),
                 ("generation_n_reps", "interval_generation")):
    add(_k, "Serial order", _dim, "material", _dim, "symbolic", None,
        role="config",
        note="Presentations of each stream. These two sections use a FIXED "
             "exposure rather than a criterion staircase -- see the section "
             "module's docstring. Defensible only while one arm can run them; "
             "the moment a second time-clocked arm exists this must become "
             "criterion-referenced like every other section.")


# ---- criterion-referenced exposure (memval/benchmarks/exposure.py).
# Exposure is now a RESULT rather than a parameter: each section trains until it
# reaches a stated criterion on a read-out upstream of the one it scores, and
# reports what that cost. Two roles, deliberately:
#   *_criterion_reached  GUARD. A metric measured at a censored exposure is a
#                        lower bound, not a capacity.
#   *_epochs_to_criterion DIAGNOSTIC in its home capacity. Sample efficiency is
#                        already scored once, by One-shot learning's
#                        `presentations` dimension; scoring each section's
#                        exposure again would count the same property five times.
_EXPOSURE = (
    ("seqlen", "Serial order", "unrolling", "sequence_length", "symbolic",
     "cued recall reaching criterion at each list length"),
    ("multiple_seq", "Continual retention", "load", "multiple_sequences", "symbolic",
     "each list reaching criterion before the interference delta is read"),
    ("noise", "Pattern completion", "cue_corruption", "noise_invariance", "symbolic",
     "clean-cue recall reaching criterion before the sigma sweep"),
    ("semantic", "Sequence disambiguation", "item_similarity", "semantic_similarity",
     "symbolic", "each list reaching criterion at its own overlap"),
    ("chain", "Continual retention", "load", "continual_chain", "symbolic",
     "every task in the chain reaching criterion as it is learned"),
    ("tmaze_pc", "Pattern completion", "cue_point", "tmaze_completion", "spatial",
     "teacher-forced one-step prediction along the route reaching criterion"),
    ("mask", "Pattern completion", "cue_completeness", "cue_masking", "symbolic",
     "clean-cue recall reaching criterion before the masking sweep"),
    ("avail", "Pattern completion", "cue_completeness", "cue_availability", "symbolic",
     "full-cue recall reaching criterion (round-robin over sequences) before the "
     "modality-availability probes"),
    ("tmaze_disamb", "Sequence disambiguation", "contextual_overlap",
     "tmaze_disambiguation", "spatial",
     "one-step prediction on the route reaching criterion"),
    ("symdis", "Sequence disambiguation", "contextual_overlap",
     "symbolic_disambiguation", "symbolic",
     "one-step prediction along an episode reaching criterion"),
)
for pre, cap, dim, section, suite, what in _EXPOSURE:
    add(f"{pre}_criterion_reached", cap, dim, "material", section, suite, boolean,
        role="guard",
        note=f"Did {what} succeed inside the staircase budget? False means every "
             f"metric in this section was read at a censored exposure and is a "
             f"lower bound.")
    add(f"{pre}_exposure_mode", cap, dim, "material", section, suite, None,
        role="config",
        note="'criterion' (the default) or 'fixed' when a budget was pinned with "
             "--benchmark-args <section>:epochs=N.")
    add(f"{pre}_exposure_source", cap, dim, "material", section, suite, None,
        role="config",
        note="Where the exposure came from: cli:section > cli:global > registry (the arm's fixed budget, control sections only) > None (criterion ladder). See docs/paper_methods_protocol.md.")
# Ratio of each section's settled exposure to the arm's reference exposure.
# Generated from _EXPOSURE so a section added there is covered automatically.
# The derived spellings are the sections that train TWICE and so report one
# exposure per condition rather than one per section.
_RATIO_EXTRA = {"multiple_seq": ("multiple_seq_A", "multiple_seq_B",
                                 "multiple_seq_interleaved"),
                "semantic": ("semantic_high", "semantic_low"),
                "seqlen": ("seqlen", "seqlen_max"),
                "avail": ("avail", "avail_single", "avail_multi")}
for _pre, _cap, _dim, _sec, _suite, _what in _EXPOSURE:
    if _suite != "symbolic":
        continue                      # the guard lives in the symbolic pipeline
    for _name in _RATIO_EXTRA.get(_pre, (_pre,)):
        add(f"{_name}_exposure_vs_baseline", _cap, _dim, "material", _sec,
            "symbolic", None, role="diagnostic",
            note="This section's settled exposure divided by the arm's reference "
                 "exposure (exposure_baseline_epochs). 1.0 means the section "
                 "trained the arm at the same point as the reference. Provenance, "
                 "not a score: a section is still valid at its own criterion, but "
                 "a large ratio means a CROSS-SECTION comparison for this arm is "
                 "not like-for-like. Read with exposure_within_band.")
    add(f"{pre}_exposure_source", cap, dim, "material", section, suite, None,
        role="config",
        note="Where the exposure came from: cli:section > cli:global > registry (the arm's fixed budget, control sections only) > None (criterion ladder). See docs/paper_methods_protocol.md.")

for pre, extra in (("seqlen", ("epochs_to_criterion", "max_epochs_to_criterion")),
                   ("multiple_seq", ("epochs_A", "epochs_B", "epochs_interleaved")),
                   ("noise", ("epochs_to_criterion",)),
                   ("semantic", ("epochs_high", "epochs_low")),
                   ("chain", ("epochs_to_criterion",)),
                   ("tmaze_pc", ("epochs_to_criterion",)),
                   ("tmaze_disamb", ("epochs_to_criterion",)),
                   ("symdis", ("epochs_to_criterion",)),
                   ("mask", ("epochs_to_criterion",))):
    cap, dim, section, suite = next(
        (c, d, sec, su) for p, c, d, sec, su, _ in _EXPOSURE if p == pre)
    for tail in extra:
        add(f"{pre}_{tail}", cap, dim, "material", section, suite, None,
            role="diagnostic",
            note="Exposure the section actually needed, in passes over the "
                 "material. Reported rather than chosen: MODEL_REGISTRY's epoch "
                 "defaults span 1 to 300 across the taxonomy, so a metric read at "
                 "'the default' is read at a different point on every arm's "
                 "learning curve. Diagnostic here; sample efficiency is scored "
                 "once, under One-shot learning.")


# --------------------------------------------------------------------------
# Intended dimension weights per capacity.  Dimensions listed but never scored
# still count toward the intended denominator, so `coverage` reports how much of
# each capacity the suite could actually measure for this arm.
# --------------------------------------------------------------------------
DIMENSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Rebalanced 2026-09-02 when the chain and AB/AC sections landed. The
    # capacity previously had three dimensions, all but one of them scoring
    # stability, and the one plasticity dimension came from a single 2AFC
    # protocol in a single modality. Now:
    #   load                  stability under interposed material (T=2 + the chain)
    #   plasticity_under_load can new material still get in at all
    #   contingency           overwriting an association that became invalid,
    #                         with an instrument in EACH modality
    #   relevance             is what survives the part still in use
    # `load` and `contingency` give weight to the two new dimensions rather than
    # the capacity growing an inflated denominator; contingency stays the
    # heaviest because it is now the only dimension with two independent
    # instruments.
    "Continual retention": {
        "load": 0.25,
        "plasticity_under_load": 0.20,
        "contingency": 0.35,
        "relevance": 0.10,
        "schema_consistency": 0.10,
    },
    "One-shot learning": {
        "presentations": 0.50,
        "schema_consistency": 0.25,
        "presentations_streamed": 0.25,
    },
    "Pattern completion": {
        "cue_corruption": 0.40,
        "cue_point": 0.35,
        "cue_completeness": 0.25,
    },
    "Sequence disambiguation": {
        "contextual_overlap": 0.60,
        "item_similarity": 0.40,
    },
    # Rebalanced 2026-09-02 when temporal contiguity was removed as a promise and
    # the capacity was re-cut onto the four paragraph-11 rows exactly. `establishment`
    # is new: "order never established" previously had no dimension at all, only the
    # sequence_length MRR proxy, which conflates order with association. It is light
    # because it is one conditional read-out off an existing curve, and because it
    # also serves as a read-before guard for `unrolling`.
    # Rebalanced 2026-09-02. `paper/capacities.md` para 11 puts metric time on an
    # equal footing with ordinal order -- "representing time is a matter of
    # preserving order AND the temporal distance between items" -- so the two
    # interval read-outs together carry HALF the capacity. The ordinal three keep
    # their previous proportions inside the remaining half, rounded.
    "Serial order": {
        "unrolling": 0.20,
        "binding_ordinal": 0.20,
        "establishment": 0.10,
        "interval_retention": 0.25,
        "interval_generation": 0.25,
    },
}

# Dimensions with no instrument in the shipped suites at all.
UNBUILT_DIMENSIONS = {
    # `cue_completeness` was here until 2026-09-02. It is now the `cue_masking`
    # section of the symbolic suite: a swept fraction of the cue's feature
    # dimensions removed, on a hierarchical encoder whose columns mean something
    # (memval/benchmarks/cue_masking.py).
    # interval_retention and interval_generation were here until 2026-09-04.
    # Both are now wired sections (memval/benchmarks/interval_timing.py, run from
    # the symbolic pipeline), so they are scored where the arm declares the
    # capability and reported NOT APPLICABLE where it does not -- which is the
    # INTERFACE_GATED_DIMENSIONS branch above, checked first. Serial order's
    # metric-time half is no longer structurally unmeasurable.
}

CAPACITY_ORDER = [
    "Continual retention", "One-shot learning", "Pattern completion",
    "Sequence disambiguation", "Serial order",
]

# Dimensions gated on a validity guard: if the guard fails, every metric under
# the dimension is vacuous and the whole dimension is dropped from the score.
# Dimensions that require a model capability the arm may simply not declare.
#
# LEGACY FALLBACK. Results files written after the OnlineTrainable capability
# landed carry `metadata.status == "not_applicable"` and omit the metrics
# entirely -- the pipeline now refuses to run the sweep instead of running it
# and writing zeros. This table (and `_probe_interfaces`) exists only to score
# results files generated BEFORE that change, which still contain the zeros.
# Do not add new entries: gate new capabilities at `memval.benchmarks.ingest`,
# where the regime is chosen.
INTERFACE_GATED_DIMENSIONS = {
    "interval_retention": (
        "TemporallyClocked",
        "The arm does not declare memval.models.capabilities.TemporallyClocked, so "
        "elapsed time cannot change its state and the interval cannot be a cue for "
        "it. Not a low score: the benchmark has no meaning for an ordinal-clocked "
        "arm, which sees two contradictory transitions out of the shared prefix and "
        "can only pick one. The section is also not wired yet -- but the capability "
        "gap is the binding constraint, so it is reported as the reason.",
    ),
    "interval_generation": (
        "TimingPredictive",
        "The arm does not declare memval.models.capabilities.TimingPredictive: it "
        "has no read-out that emits a gap, so it cannot reproduce a tempo or a "
        "rhythm however well it reproduces the item order. Strictly stronger than "
        "TemporallyClocked -- absorbing elapsed time is not the same as producing "
        "it.",
    ),
    "presentations_streamed": (
        "fit_event",
        "The arm does not declare OnlineTrainable, so no streamed (event-by-event) "
        "regime exists for it. Older results files ran the online_symbolic pipeline "
        "anyway and wrote 0.0 at every sweep point; those zeros are a capability "
        "gap, not a recall failure, and are excluded. Note that MODEL_REGISTRY "
        "still lists 'online_symbolic' among this model's modalities.",
    ),
}

# Keys are ("<capacity>", "<dimension>") for a guard that applies to one
# capacity's use of a section, or "<dimension>" for one that applies to every
# use. `schema_consistency` needs both forms: it feeds two capacities from one
# protocol run, and the two halves are invalidated by different things.
DIMENSION_GUARDS = {
    "schema_consistency": (
        "schema_resolved",
        "schema_resolved is False: the rung ladder does not separate on "
        "trials-to-criterion, so the consistency manipulation has no dynamic "
        "range and the rung comparison is vacuous. Read schema_at_ceiling / "
        "schema_at_floor for which end it collapsed to, and "
        "schema_auc_resolved for whether the continuous readout still ranges.",
    ),
    "plasticity_under_load": (
        "chain_acquired",
        "chain_acquired is False: the chain never acquired anything, so the "
        "diagonal is at chance and intransigence measures nothing. Read "
        "chain_avg_learning against chain_chance_level.",
    ),
    "relevance": (
        "select_under_pressure",
        "select_under_pressure is False: the chain did not saturate the "
        "substrate, so nothing had to be discarded and selectivity is vacuous. "
        "A zero here means the question was not posed, NOT that the arm failed "
        "it. Raise the load (--benchmark-args continual_chain:n_tasks) or read "
        "select_pressure to see how far off saturation the arm was.",
    ),
    "cue_completeness": (
        "mask_resolved",
        "mask_resolved is False: the masking sweep did not pose its question. "
        "Read mask_at_floor (the list was never acquired, so every masked "
        "read-out below it is vacuous), mask_at_ceiling (recall never moved, so "
        "the manipulation has no range on this arm -- read "
        "mask_random_margin_in_bound, which still resolves) and "
        "mask_identifiable_region (the masked cue stopped naming its own item "
        "almost immediately, which is a property of the encoder configuration "
        "and not of the arm -- raise features_per_node).",
    ),
    ("Serial order", "binding_ordinal"): (
        "failures_observed",
        "No cued failure was observed to classify. Every binding_ordinal metric is "
        "a conditional on an error — order_error_fraction divides by the failure "
        "count, transposition_locality and _asymmetry divide by the transposition "
        "count — so with a perfect cued curve they are undefined, not zero. The arm "
        "sits above the instrument's range rather than failing it; widen the length "
        "grid or raise the load before reading this dimension.",
    ),
    ("Continual retention", "schema_consistency"): (
        "schema_interference_valid",
        "schema_interference_valid is False: the section ran under "
        "interference_protocol='extended', where the new list IS the host list "
        "plus one item, so learning the new item rehearses every base transition "
        "of the host category. The `same` bucket then measures rehearsal rather "
        "than damage and reads at or below zero -- which would score as "
        "'no forgetting'. Re-run the section with "
        "interference_protocol='focused' and pass it via --schema-focused.",
    ),
}


# --------------------------------------------------------------------------
# Derived metrics: computed from `series`, which carries signal the scalar
# summaries in `metrics` throw away.
# --------------------------------------------------------------------------

def derive(sp: Dict[str, Any], sy: Dict[str, Any], on: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def emit(key, value, capacity, dimension, kind, section, suite, n, note,
             role="score", v=1, source=""):
        out.append(dict(key=key, raw=value, capacity=capacity, dimension=dimension,
                        kind=kind, section=section, suite=suite, n=n, note=note,
                        role=role, v=v, derived=True, source=source))

    sym_series, sp_series = sy.get("series", {}), sp.get("series", {})
    sym_m, sp_m = sy.get("metrics", {}), sp.get("metrics", {})

    # -- Pattern completion: whole-sweep area, not just the crossing point.
    ns = sym_series.get("noise_sweep")
    if ns:
        auc = float(np.mean(ns["mrr"]))
        emit("noise_sweep_mean_mrr", auc, "Pattern completion", "cue_corruption",
             "probe", "noise_invariance", "symbolic", _clip(auc), v=2,
             source="series.noise_sweep.mrr",
             note="Mean cued-recall MRR across the whole sigma sweep [0, 1]. The "
                  "threshold metric reports only where the curve crosses 0.5; this "
                  "scores the shape of the decay.")

    # -- Sequence disambiguation: the graded similarity sweep, which the scalar
    #    high/low pair cannot see because both of its points sit at ceiling.
    ss = sym_series.get("similarity_sweep")
    if ss:
        auc = float(np.mean(ss["mrr"]))
        emit("similarity_sweep_mean_mrr", auc, "Sequence disambiguation",
             "item_similarity", "material", "semantic_similarity", "symbolic",
             float("nan"), role="diagnostic", source="series.similarity_sweep.mrr",
             note="Mean MRR across the category-variance sweep. DIAGNOSTIC since "
                  "2026-09-02: every rung is now trained to criterion, so this sits "
                  "just above the criterion whatever the overlap. Read "
                  "similarity_exposure_cost instead.")
        eps_ = ss.get("epochs_to_criterion") or []
        reached_ = ss.get("criterion_reached") or []
        if eps_ and len(eps_) == len(ss["cosine_similarities"]):
            order = sorted(range(len(eps_)), key=lambda i: ss["cosine_similarities"][i])
            easiest, hardest = eps_[order[0]], eps_[order[-1]]
            cost = (hardest / easiest) if easiest else float("inf")
            if not all(reached_):
                cost = float("inf")          # censored: the criterion was unreachable
            emit("similarity_exposure_cost", cost, "Sequence disambiguation",
                 "item_similarity", "material", "semantic_similarity", "symbolic",
                 log_ratio(cost), v=2, source="series.similarity_sweep",
                 note="Exposure to reach criterion at the HARDEST overlap divided by "
                      "the exposure at the easiest. Since each variance is now trained "
                      "to its own criterion, recall no longer separates the rungs — the "
                      "cost of getting there does. Self-referenced, so it is comparable "
                      "across arms; infinite (scoring 0) when the criterion is "
                      "unreachable at some overlap, which is the real failure case.")

    # -- Serial order: how much of the span is lost to readout drift alone.
    ls = sym_series.get("length_sweep")
    if ls:
        raw, quant, lengths = ls["span_raw"], ls["span_quantized"], ls["lengths"]
        frac = [r / (L - 1) for r, L in zip(raw, lengths)]
        emit("span_fraction_mean_raw", float(np.mean(frac)), "Serial order",
             "unrolling", "material", "sequence_length", "symbolic",
             _clip(float(np.mean(frac))), v=2, source="series.length_sweep",
             note="Mean of span/(L-1) across the length grid under raw feedback: "
                  "capacity as a fraction of what each list length allows.")
        rq = [(r / q) if q else 0.0 for r, q in zip(raw, quant)]
        emit("span_raw_over_quantized", float(np.mean(rq)), "Serial order",
             "unrolling", "probe", "sequence_length", "symbolic",
             _clip(float(np.mean(rq))), v=2, source="series.length_sweep",
             note="Mean raw-span / quantized-span across the grid. 1.0 = no readout "
                  "drift; below 1.0 is span lost purely to feeding an unnormalised "
                  "state back, not to a capacity limit.")

    # -- Serial order: the spatial unrolling read-out, mirrored from completion.
    if "tmaze_pc_divergence_step" in sp_m and sp_m.get("tmaze_pc_recall_len"):
        ds = sp_m["tmaze_pc_divergence_step"] / sp_m["tmaze_pc_recall_len"]
        emit("tmaze_pc_divergence_fraction", float(ds), "Serial order", "unrolling",
             "material", "tmaze_completion", "spatial", _clip(ds), v=2,
             source="metrics.tmaze_pc_divergence_step / tmaze_pc_recall_len",
             note="Fraction of the recalled route held within eps before the first "
                  "divergence. The spatial sections recall open-loop, so this is a "
                  "joint completion-and-unrolling read-out; it is scored under both "
                  "capacities by design (audit C1).")

    # -- Continual retention: extinction depth against what was acquired.
    drop = sp_m.get("reversal_extinction_extinction_reward_drop")
    base = sp_m.get("reversal_extinction_acquisition_final_reward_pred")
    if drop is not None and base:
        emit("extinction_reward_drop_fraction", float(drop / base), "Continual retention",
             "contingency", "material", "tmaze_reversal", "spatial",
             _clip(drop / base), v=2,
             source="reversal_extinction_extinction_reward_drop / "
                    "reversal_extinction_acquisition_final_reward_pred",
             note="Reward prediction destroyed by extinction as a fraction of the "
                  "reward prediction acquired. Puts the gain-scaled raw drop on [0,1].")

    # -- Continual retention: relearning cost.
    acq = sp_m.get("reversal_direct_acquisition_trials_to_criterion")
    rev = sp_m.get("reversal_direct_reversal_trials_to_criterion")
    if acq and rev:
        emit("reversal_cost_ratio", float(rev / acq), "Continual retention",
             "contingency", "material", "tmaze_reversal", "spatial", float("nan"),
             role="diagnostic", source="reversal / acquisition trials_to_criterion",
             note="Trials to reverse divided by trials to acquire. > 1 means the old "
                  "association resists; reported, not scored (no principled ceiling).")

    # -- Disambiguation: the shape of the availability collapse, and the bias it hides.
    gr = sp_series.get("tmaze_disamb_graded")
    if gr:
        withdrawn = [r for r in gr if r.get("availability") == "withdrawn"]
        if withdrawn:
            emit("disamb_withdrawn_mean_divergence_accuracy",
                 float(np.mean([r["divergence_accuracy"] for r in withdrawn])),
                 "Sequence disambiguation", "contextual_overlap", "material",
                 "tmaze_disambiguation", "spatial",
                 _clip((np.mean([r["divergence_accuracy"] for r in withdrawn]) - 0.5) / 0.5),
                 role="protocol_limited", v=2, source="series.tmaze_disamb_graded",
                 note="Mean divergence accuracy over every odour-withdrawn rung. "
                      "Protocol-limited (audit C2): the memoryless predict_next probe "
                      "cannot carry a discriminator across a cue-free stretch.")
            worst = min(withdrawn, key=lambda r: r["delay"])
            bias = abs(worst["route_A_branch_accuracy"] - worst["route_B_branch_accuracy"])
            emit("disamb_delay0_route_bias", float(bias), "Sequence disambiguation",
                 "contextual_overlap", "material", "tmaze_disambiguation", "spatial",
                 float("nan"), role="diagnostic", source="series.tmaze_disamb_graded",
                 note="|route A - route B| branch accuracy at the first withdrawn rung. "
                      "A large value means the averaged accuracy hides a collapse onto "
                      "one route rather than genuine partial performance.")
    return out


# --------------------------------------------------------------------------
# Rollup
# --------------------------------------------------------------------------

def classify(suites: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn every emitted metric into a spec-annotated record. Unknown keys raise."""
    rows, unknown = [], []
    for suite, payload in suites.items():
        for key, raw in sorted(payload.get("metrics", {}).items()):
            spec = SPEC.get(key)
            if spec is None:
                # Seed-replication companions (schema_consistency and any
                # section using its convention): `<key>_sd` is the s.d. over
                # seeds of a registered numeric metric, `<key>_frac` the
                # fraction of seeds a registered boolean guard passed. Neither
                # is scored on its own -- the mean / all() lives under the bare
                # key -- so each is REGISTERED here as a diagnostic under its
                # parent's capacity/dimension (norm None), not rejected as
                # unknown. Registering (rather than only appending a row) is
                # what lets every later SPEC[key] lookup -- overlay_focused,
                # the suite status check -- treat it like any other metric.
                base = next((key[:-len(sfx)] for sfx in ("_sd", "_frac")
                             if key.endswith(sfx) and key[:-len(sfx)] in SPEC), None)
                if base is None:
                    unknown.append(f"{suite}:{key}")
                    continue
                parent = SPEC[base]
                add(key, parent["capacity"], parent["dimension"], parent["kind"],
                    parent["section"], parent["suite"], None, v=parent["v"],
                    role="diagnostic",
                    note=(("s.d. over seeds of " if key.endswith("_sd")
                           else "fraction of seeds passing ") + base))
                spec = SPEC[key]
            n = float("nan")
            if spec["norm"] is not None:
                n = spec["norm"](raw, **spec["norm_kw"])
            rows.append(dict(key=key, raw=raw, n=n, derived=False, source=f"{suite}/metrics.json",
                             **{k: spec[k] for k in
                                ("capacity", "dimension", "kind", "section", "suite", "v", "role", "note")}))
    if unknown:
        raise SystemExit(f"Unclassified metric keys (add them to SPEC): {unknown}")
    return rows


#: Keys whose value depends on which interference protocol the section ran
#: under, and which are only meaningful under 'focused'. Everything else the
#: section emits is an acquisition read-out and is taken from the suite run.
FOCUSED_ONLY_KEYS = ("_interference_", "_prefix_recall")


def overlay_focused(rows: List[Dict[str, Any]], focused: Dict[str, Any]) -> int:
    """Replace the damage read-outs with values from an
    ``interference_protocol='focused'`` run of the same section.

    The section emits two read-outs from one protocol, and they need different
    training data. Acquisition (trials-to-criterion, new_item_recall/auc) is
    valid under either protocol and is taken from the suite's own run, so the
    headline acquisition numbers stay comparable with everything else in the
    report. Damage (interference_*, prefix_recall) is valid only under
    'focused', because 'extended' rehearses the host category while the new item
    is learned. Rather than dropping the damage half, we source it from a
    dedicated focused run and say so on every row it touches.
    """
    fm = focused.get("metrics", {})
    if not fm:
        return 0
    n = 0
    for r in rows:
        if r.get("derived") or not r["key"].startswith("schema"):
            continue
        if not any(t in r["key"] for t in FOCUSED_ONLY_KEYS):
            continue
        if r["key"] not in fm:
            continue
        spec = SPEC[r["key"]]
        r["raw"] = fm[r["key"]]
        r["n"] = (spec["norm"](r["raw"], **spec["norm_kw"])
                  if spec["norm"] is not None else float("nan"))
        r["source"] = "schema_consistency (interference_protocol='focused')"
        r["protocol"] = "focused"
        r["note"] = r["note"] + (" Sourced from a dedicated "
                                 "interference_protocol='focused' run, the only "
                                 "protocol under which this number measures damage.")
        n += 1
    return n


def gate_saturated_mask_recall(rows: List[Dict[str, Any]],
                               guards: Dict[str, Any]) -> int:
    """Demote the masking recall read-outs when the section reports them saturated.

    `mask_at_ceiling` means in-bound recall is >= 0.95 in every mode: the arm
    completes correctly at every masked fraction where the cue still names its
    item. The section's own note says the dimension is then carried by
    `mask_random_tolerance` and `mask_random_margin_in_bound`. Scoring three
    near-1.00 recall values it has just declared uninformative would let a
    saturated instrument inflate the dimension it failed to resolve.
    """
    if not guards.get("mask_at_ceiling", False):
        return 0
    n = 0
    for r in rows:
        if r["role"] == "score" and r["key"].endswith("_recall_in_bound"):
            r["role"] = "diagnostic"
            r["note"] = r["note"] + (" EXCLUDED: mask_at_ceiling is True, so this "
                                     "read-out is saturated and the section itself "
                                     "reports it as carrying no information about "
                                     "this arm. Raising cue_masking's seq_len pushes "
                                     "it off the ceiling.")
            n += 1
    return n


def gate_damage_readouts(rows: List[Dict[str, Any]], guards: Dict[str, Any]) -> int:
    """Demote the schema section's damage read-outs when no valid protocol run
    backs them.

    `interference_*` and `prefix_recall` measure what learning the new item cost
    the material already stored. Under interference_protocol='extended' the new
    list is the host list plus one item, so learning it *rehearses* every host
    transition: interference reads ~0 and prefix_recall reads ~1.0, both of which
    normalise to a near-perfect score for a run that never tested the thing. This
    is capacity-agnostic on purpose -- it must hold wherever these keys are
    filed, not only under the capacity whose dimension guard catches it.
    """
    if guards.get("schema_interference_valid", False):
        return 0
    n = 0
    for r in rows:
        if not r["key"].startswith("schema") or r["role"] != "score":
            continue
        if not any(t in r["key"] for t in FOCUSED_ONLY_KEYS):
            continue
        r["role"] = "protocol_limited"
        r["note"] = r["note"] + (" EXCLUDED: schema_interference_valid is False, so "
                                 "this ran under interference_protocol='extended', "
                                 "where the host category is rehearsed while the new "
                                 "item is learned. Supply a focused run with "
                                 "--schema-focused to score it.")
        n += 1
    return n


#: Chain read-outs that award credit for stability and are therefore vacuous
#: on material the arm never acquired. `chain_avg_accuracy` is NOT here: it is
#: chance-normalised and correctly reads ~0 on an unacquired chain.
CHAIN_STABILITY_KEYS = ("chain_avg_forgetting", "chain_retention_ratio")


def gate_chain_readouts(rows: List[Dict[str, Any]], guards: Dict[str, Any]) -> int:
    """Demote the chain's stability read-outs when the chain was never acquired.

    `chain_avg_forgetting` is lower-is-better and `chain_retention_ratio` is a
    ratio of what survived to what was learned. On an arm whose chain accuracy
    never left chance, forgetting is ~0 because there was nothing to forget, and
    that normalised to a near-perfect stability score: the (retired) Vieth STDP arm
    (2026-09-04) scored 0.41 on `load` with `chain_acquired` False and chain
    ACC 0.042 against chance 0.033. The same class of defect as the sign
    inversion of 2026-09-04 (a normaliser for forgetting is untestable on an arm
    that does not forget) -- this is its mirror, an arm that never learns.

    Metric-level rather than a dimension guard on purpose: `load` also carries
    the `multiple_sequences` section, which has its own before/after and stands
    on its own. Mirrors `gate_damage_readouts`.
    """
    if guards.get("chain_acquired", True):
        return 0
    n = 0
    for r in rows:
        if r["key"] in CHAIN_STABILITY_KEYS and r["role"] == "score":
            r["role"] = "protocol_limited"
            r["note"] = r["note"] + (" EXCLUDED: chain_acquired is False, so the chain "
                                     "never left chance and there was nothing to "
                                     "forget; a stability credit here would reward "
                                     "not learning. Read chain_avg_accuracy.")
            n += 1
    return n


def rollup(rows: List[Dict[str, Any]], guards: Dict[str, Any],
           inclusive: bool = False) -> Dict[str, Any]:
    """Two-level weighted sum. `inclusive=True` also admits protocol-limited
    metrics and guard-failed dimensions, to show what the exclusions cost."""
    accept = {"score", "protocol_limited"} if inclusive else {"score"}
    out: Dict[str, Any] = {}

    for capacity in CAPACITY_ORDER:
        weights = DIMENSION_WEIGHTS[capacity]
        dims: Dict[str, Any] = {}
        for dim, w in weights.items():
            metrics = [r for r in rows if r["capacity"] == capacity
                       and r["dimension"] == dim and r["role"] in accept
                       and not math.isnan(r["n"])]
            status, reason, D = "scored", "", float("nan")

            # The capability gate is checked FIRST. When an arm cannot run a
            # dimension at all, that is the more specific and more useful reason
            # than "no section is wired": wiring the section would not change this
            # arm's row. Where a dimension is both, the note says so.
            if dim in INTERFACE_GATED_DIMENSIONS and not guards.get(
                    "iface:" + INTERFACE_GATED_DIMENSIONS[dim][0], True):
                status, reason = "not_applicable", INTERFACE_GATED_DIMENSIONS[dim][1]
                metrics = []
            elif dim in UNBUILT_DIMENSIONS:
                status, reason = "unbuilt", UNBUILT_DIMENSIONS[dim]
            elif not inclusive:
                spec_guard = (DIMENSION_GUARDS.get((capacity, dim))
                              or DIMENSION_GUARDS.get(dim))
                if spec_guard is not None:
                    gkey, greason = spec_guard
                    if not guards.get(gkey, False):
                        status, reason, metrics = "unresolved", greason, []
                # The shared guard still applies on top of a capacity-specific one.
                shared = DIMENSION_GUARDS.get(dim)
                if (status == "scored" and shared is not None
                        and (capacity, dim) in DIMENSION_GUARDS
                        and not guards.get(shared[0], False)):
                    status, reason, metrics = "unresolved", shared[1], []
            if status != "scored":
                metrics = []          # never let a stray metric revive a gated dim
            if status == "scored" and not metrics:
                status = "unrunnable"
                reason = (
                    "No finite scoring metric was produced. For the streamed "
                    "regime this means the arm does not implement fit_event; "
                    "otherwise the dimension's section(s) were not run, or every "
                    "metric they emitted normalised to NaN."
                    if dim == "presentations_streamed" else
                    "No finite scoring metric was produced: the section(s) "
                    "feeding this dimension were not run, or every metric they "
                    "emitted normalised to NaN. Check that the suite run "
                    "included them before reading this as a model result.")
            if metrics:
                num = sum(r["v"] * r["n"] for r in metrics)
                den = sum(r["v"] for r in metrics)
                D = num / den
                status = "scored"

            dims[dim] = dict(weight=w, score=D, status=status, reason=reason,
                             n_metrics=len(metrics),
                             metrics=[r["key"] for r in metrics])

        avail = {d: v for d, v in dims.items() if v["status"] == "scored"}
        wsum = sum(v["weight"] for v in avail.values())
        score = (sum(v["weight"] * v["score"] for v in avail.values()) / wsum) if wsum else float("nan")
        out[capacity] = dict(
            score=score,
            coverage=wsum / sum(weights.values()),
            dimensions=dims,
        )
    return out


def _md(text: str) -> str:
    """Escape a cell so embedded pipes/newlines cannot break a markdown table."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{nd}f}"
    if isinstance(x, list):
        return "[" + ", ".join(_fmt(i, nd) for i in x) + "]"
    return str(x)


ROLE_LABEL = {
    "score": "SCORED",
    "protocol_limited": "PROTOCOL-LIMITED",
    "control": "CONTROL",
    # A comparison condition rather than a floor: its value is the baseline the
    # experimental arm is read against (e.g. the disjoint-cue run of
    # paired_associate), so a HIGH value is expected and it is still never summed.
    "reference": "REFERENCE",
    "guard": "GUARD",
    "config": "CONFIG",
    "diagnostic": "DIAGNOSTIC",
}


def write_markdown(path, model, rows, primary, inclusive, guards, suite_meta, series):
    L: List[str] = []
    w = L.append
    w(f"# Capacity scorecard — {model}")
    w("")
    w("Generated by `bin/score_capacities.py` from the `metrics.json` written by "
      "`bin/run_benchmark.py`. Every key emitted by every suite appears below exactly once.")
    w("")
    w("## Run provenance")
    w("")
    w("| Suite | Sections | n_trials | Model kwargs | Status |")
    w("|---|---|---|---|---|")
    for suite, meta in suite_meta.items():
        w(f"| `{suite}` | {meta['sections']} | {meta['n_trials']} | "
          f"`{meta['kwargs']}` | {_md(meta['status'])} |")
    w("")

    w("## 0. Validity guards")
    w("")
    w("These are read before any score below them. A failed guard voids its whole "
      "section: the manipulation had no dynamic range, so the numbers under it "
      "carry no information about the model.")
    w("")
    w("| Guard | Value | Consequence |")
    w("|---|---|---|")
    for k, v in sorted(guards.items()):
        if k.startswith("iface:"):
            ok = bool(v)
            w(f"| `{k}` (model interface) | {ok} | "
              + ("Implemented; the streamed regime is scorable." if ok else
                 "**Not implemented** — the streamed sections are excluded, not scored 0.")
              + " |")
            continue
        note = _md(SPEC.get(k, {}).get("note", ""))
        w(f"| `{k}` | {_fmt(v)} | {note} |")
    w("")
    w("## 1. Capacity scores")
    w("")
    w("`score` uses validated metrics only. `score (inclusive)` re-admits the "
      "protocol-limited and guard-failed metrics, so the gap between the two columns "
      "is exactly what the exclusions are worth. `coverage` is the fraction of each "
      "capacity's intended dimension weight that produced a score at all.")
    w("")
    w("| Capacity | Score | Coverage | Score × coverage | Score (inclusive) | Dimensions scored |")
    w("|---|---:|---:|---:|---:|---|")
    for c in CAPACITY_ORDER:
        p, i = primary[c], inclusive[c]
        scored = [d for d, v in p["dimensions"].items() if v["status"] == "scored"]
        w(f"| **{c}** | {_fmt(p['score'])} | {_fmt(p['coverage'], 2)} | "
          f"{_fmt(p['score'] * p['coverage'])} | {_fmt(i['score'])} | "
          f"{', '.join('`'+s+'`' for s in scored) or '—'} |")
    w("")

    w("## 2. Dimension scores")
    w("")
    w("| Capacity | Dimension | Kind | Weight | Score | Status | Metrics | Note |")
    w("|---|---|---|---:|---:|---|---:|---|")
    for c in CAPACITY_ORDER:
        for dim, v in primary[c]["dimensions"].items():
            kinds = {r["kind"] for r in rows if r["capacity"] == c and r["dimension"] == dim}
            kind = "/".join(sorted(kinds)) or "—"
            w(f"| {c} | `{dim}` | {kind} | {v['weight']:.2f} | {_fmt(v['score'])} | "
              f"{v['status'].upper()} | {v['n_metrics']} | {_md(v['reason'] or '')} |")
    w("")

    w("## 3. Every metric, with its dimension and capacity")
    w("")
    w("`role` decides whether a metric enters the sum. `v` is its within-dimension "
      "weight (2 = primary readout). `n` is the value after normalisation to [0,1].")
    w("")
    for c in CAPACITY_ORDER:
        crows = [r for r in rows if r["capacity"] == c]
        if not crows:
            continue
        w(f"### {c}")
        w("")
        for dim in DIMENSION_WEIGHTS[c]:
            drows = [r for r in crows if r["dimension"] == dim]
            dv = primary[c]["dimensions"][dim]
            w(f"#### `{dim}` — {dv['status'].upper()}"
              + (f" (score {_fmt(dv['score'])}, weight {dv['weight']:.2f})"
                 if dv["status"] == "scored" else ""))
            w("")
            if dv["reason"]:
                w(f"> {_md(dv['reason'])}")
                w("")
            if not drows:
                w("_No metric is emitted for this dimension._")
                w("")
                continue
            w("| Metric | Suite / section | Raw | n | v | Role | Meaning |")
            w("|---|---|---:|---:|---:|---|---|")
            for r in sorted(drows, key=lambda r: (r["role"] != "score", r["key"])):
                src = f"{r['suite']} / `{r['section']}`"
                star = " *(derived)*" if r.get("derived") else ""
                w(f"| `{r['key']}`{star} | {src} | {_fmt(r['raw'], 4)} | {_fmt(r['n'])} | "
                  f"{r['v'] if r['role'] in ('score','protocol_limited') else '—'} | "
                  f"{ROLE_LABEL[r['role']]} | {_md(r['note'])} |")
            w("")

    w("## 4. Scoring formula")
    w("")
    w("```")
    w("  n_m  = normalise(raw_m)                       in [0, 1], 1 = ideal")
    w("  D_d  = SUM_m (v_m * n_m) / SUM_m (v_m)        dimension score")
    w("  C    = SUM_d (w_d * D_d) / SUM_d (w_d)        over AVAILABLE dimensions")
    w("  cov  = SUM_{d available} w_d / SUM_{d intended} w_d")
    w("```")
    w("")
    w("Normalisers used: identity (higher-better on [0,1]); `1-x` (lower-better on "
      "[0,1]); chance correction `(x-c)/(1-c)`; `x/ref` against a stated attainable "
      "maximum; `1/n` for counts-to-criterion; `(budget-n)/(budget-1)` for "
      "trials-to-criterion against a protocol budget.")
    w("")

    w("## 5. Series behind the derived metrics")
    w("")
    for name, s in series.items():
        w(f"- **`{name}`** — `{json.dumps(s)[:400]}`")
    w("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _wrap_note(notes, width=104):
    """Join footnote sentences and hard-wrap to a fixed column, for a figure
    caption that must not depend on the renderer's text metrics."""
    import textwrap
    return "\n".join(textwrap.wrap(" ".join(notes), width=width))


def write_radar(path, model, primary, inclusive):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = CAPACITY_ORDER
    N = len(labels)
    ang = [n / N * 2 * math.pi for n in range(N)] + [0.0]

    def series(d):
        v = [0.0 if math.isnan(d[c]["score"]) else d[c]["score"] for c in labels]
        return v + v[:1]

    prim, incl = series(primary), series(inclusive)
    disc = [(0.0 if math.isnan(primary[c]["score"]) else primary[c]["score"])
            * primary[c]["coverage"] for c in labels]
    disc += disc[:1]

    fig = plt.figure(figsize=(11.0, 11.6), facecolor="white")
    ax = fig.add_axes([0.235, 0.255, 0.53, 0.53], polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")

    theta = np.linspace(0, 2 * math.pi, 240)
    for r in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.plot(theta, [r] * 240, color="#dedede", lw=0.7, zorder=0)
    for a in ang[:-1]:
        ax.plot([a, a], [0, 1.0], color="#e6e6e6", lw=0.7, zorder=0)

    ax.plot(ang, incl, color="#B07A16", lw=1.6, ls=(0, (5, 3)), zorder=2,
            label="inclusive — protocol-limited metrics re-admitted")
    ax.fill(ang, incl, color="#B07A16", alpha=0.11, zorder=1)

    ax.plot(ang, prim, color="#2e6296", lw=2.6, zorder=4,
            label="capacity score — validated metrics only")
    ax.fill(ang, prim, color="#2e6296", alpha=0.19, zorder=3)
    ax.scatter(ang[:-1], prim[:-1], s=90, color="#2e6296", zorder=6,
               edgecolor="white", linewidth=1.6)

    ax.plot(ang, disc, color="#AD3F6B", lw=1.8, ls=(0, (1.6, 2.2)), zorder=5,
            marker="s", ms=6, markeredgecolor="white", markeredgewidth=1.0,
            label="score x coverage — discounted by what the suite can measure")

    ax.set_xticks([])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8,
                       color="#a8a8a8")
    ax.set_rlabel_position(36)
    ax.grid(False)
    ax.spines["polar"].set_color("#d2d2d2")

    # Vertex annotations, stacked in SCREEN space so the order is always
    # name / score / coverage regardless of which side of the circle we are on.
    for a, c in zip(ang[:-1], labels):
        p = primary[c]
        ux, uy = math.sin(a), math.cos(a)          # outward direction on screen
        ha = "center" if abs(ux) < 1e-6 else ("left" if ux > 0 else "right")
        ox, oy = 34 * ux, 34 * uy
        cov = p["coverage"]
        col = "#AD3F6B" if cov < 0.6 else ("#B07A16" if cov < 0.85 else "#3F7A4F")
        for dy, txt, kw in (
            (17, c, dict(fontsize=13, fontweight="bold", color="#1b1b1b")),
            (-3, _fmt(p["score"], 2), dict(fontsize=17, fontweight="bold",
                                           color="#2e6296")),
            (-21, f"coverage {cov:.0%}", dict(fontsize=9.5, color=col)),
        ):
            ax.annotate(txt, xy=(a, 1.0), xycoords="data",
                        xytext=(ox, oy + dy), textcoords="offset points",
                        ha=ha, va="center", **kw)

    fig.text(0.5, 0.955, "MemVal capacity profile", ha="center",
             fontsize=19, fontweight="bold", color="#1b1b1b")
    fig.text(0.5, 0.928, model, ha="center", fontsize=13, color="#5a5a5a")
    fig.text(0.5, 0.906,
             "spatial + symbolic suites   |   weighted sums over normalised "
             "metrics, grouped by manipulated dimension",
             ha="center", fontsize=9.5, color="#8a8a8a")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.135), fontsize=10,
              frameon=False, ncol=1, handlelength=2.8, labelspacing=0.6)

    # The footnote is DERIVED, not fixed. It used to assert three arm-specific
    # facts as if they were properties of the suite -- including "the
    # odour-withdrawn rungs floor for a protocol reason under the memoryless
    # predict_next probe", which is simply untrue of an arm that declares
    # StatePrimeable and does hold the discriminator across a delay. A caption
    # that is wrong for the one arm the row was designed to distinguish is worse
    # than no caption.
    notes = ["coverage = share of a capacity's intended dimension weight that "
             "produced a score. Low coverage means the suite lacks the "
             "instrument or the arm lacks the capability, not that the model "
             "failed."]
    for cap, dims in (("Serial order", ("interval_retention", "interval_generation")),
                      ("Pattern completion", ("cue_completeness",)),
                      ("One-shot learning", ("presentations_streamed",))):
        info = primary.get(cap, {}).get("dimensions", {})
        unbuilt = [d for d in dims if info.get(d, {}).get("status") == "unbuilt"]
        na = [d for d in dims if info.get(d, {}).get("status") == "not_applicable"]
        if unbuilt:
            notes.append(f"{cap}: {', '.join(unbuilt)} has no section wired in "
                         f"MemVal yet — a gap in the SUITE, identical for every arm.")
        if na:
            notes.append(f"{cap}: {', '.join(na)} is not applicable — this arm "
                         f"does not declare the capability the row needs.")
    gap = {c: (0.0 if math.isnan(primary[c]["score"]) else primary[c]["score"])
              - (0.0 if math.isnan(inclusive[c]["score"]) else inclusive[c]["score"])
           for c in labels}
    worst = max(gap, key=gap.get)
    if gap[worst] > 0.05:
        notes.append(f"{worst}: the inclusive line sits {gap[worst]:.2f} BELOW "
                     f"the score, so the excluded metrics were dragging it down.")
    elif min(gap.values()) < -0.05:
        best = min(gap, key=gap.get)
        notes.append(f"{best}: the inclusive line sits {-gap[best]:.2f} ABOVE "
                     f"the score — re-admitting the excluded metrics would "
                     f"raise it, so read which ones and why in the scorecard.")
    fig.text(0.5, 0.012, _wrap_note(notes, width=124),
             ha="center", va="bottom", fontsize=8.4, color="#7a7a7a", linespacing=1.7)

    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def _probe_interfaces(model_class_name: Optional[str]) -> Dict[str, bool]:
    """Ask the registry whether this arm actually implements the optional model
    interfaces the suites depend on. Missing/unresolvable -> assume implemented,
    so the gate only ever fires on positive evidence of absence."""
    out: Dict[str, bool] = {}
    if not model_class_name:
        return out
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_memval_registry",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_benchmark.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memval.models.base import HippocampalModel
        cls = next((e["class"] for e in mod.MODEL_REGISTRY.values()
                    if e["class"].__name__ == model_class_name), None)
        if cls is None:
            return out
        # The capability declaration is authoritative. The getattr comparison is
        # the pre-OnlineTrainable fallback, kept only so this script can still
        # score results files produced by older checkouts.
        try:
            from memval.models.capabilities import (
                supports_online, supports_intervals, predicts_timing)
            out["iface:fit_event"] = supports_online(cls)
            # Nominal, not duck-typed: an arm whose fit_sequence takes **kwargs
            # will accept `intervals` and drop them, so a signature check would
            # report a capability it does not have.
            out["iface:TemporallyClocked"] = supports_intervals(cls)
            out["iface:TimingPredictive"] = predicts_timing(cls)
        except ImportError:
            out["iface:fit_event"] = (
                getattr(cls, "fit_event", None)
                is not getattr(HippocampalModel, "fit_event", None))
    except Exception as e:  # noqa: BLE001 - the probe must never break scoring
        print(f"[warn] interface probe skipped: {e}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="Directory holding <suite>/metrics.json for one model.")
    ap.add_argument("--model", default=None, help="Display name (default: dir name).")
    ap.add_argument("--out-prefix", default=None,
                    help="Output path prefix (default: <results-dir>/capacity_scorecard).")
    ap.add_argument("--schema-focused", default=None,
                    help="Results dir of a schema_consistency run made with "
                         "interference_protocol='focused'. Its interference_* and "
                         "prefix_recall values replace the suite run's, which are "
                         "invalid under the default 'extended' protocol. Acquisition "
                         "metrics are always taken from the suite run.")
    args = ap.parse_args()

    root = os.path.abspath(args.results_dir)
    model = args.model or os.path.basename(root)
    prefix = args.out_prefix or os.path.join(root, "capacity_scorecard")

    suites, suite_meta = {}, {}
    for suite in ("spatial", "symbolic", "online_symbolic"):
        p = os.path.join(root, suite, "metrics.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            payload = json.load(f)
        suites[suite] = payload
        m = payload.get("metadata", {})
        scoring = [k for k in payload.get("metrics", {})
                   if SPEC.get(k, {}).get("role") == "score"]
        finite = [k for k in scoring
                  if SPEC[k]["norm"] and not math.isnan(
                      SPEC[k]["norm"](payload["metrics"][k], **SPEC[k]["norm_kw"]))]
        status = ("ran" if finite else
                  "RAN BUT PRODUCED NO USABLE METRIC — see the notes")
        if suite == "online_symbolic":
            declared = m.get("status")
            if declared == "not_applicable":
                # Written by the pipeline itself, at the capability gate.
                status = "NOT APPLICABLE — " + m.get(
                    "status_detail", "the arm does not declare OnlineTrainable.")
            elif declared != "ran" and not _probe_interfaces(m.get("model_name")).get(
                    "iface:fit_event", True):
                # Legacy results file: no status field, and the arm has no
                # streaming capability, so the zeros on disk are a capability gap.
                status = ("NOT APPLICABLE — the arm does not declare OnlineTrainable, "
                          "so every sweep point in this (pre-capability) results file "
                          "was written as 0.0. These zeros are excluded from scoring.")
        suite_meta[suite] = dict(sections=len({SPEC[k]["section"] for k in payload["metrics"]
                                               if k in SPEC}),
                                 n_trials=m.get("n_trials"),
                                 kwargs=json.dumps(m.get("model_kwargs", {})),
                                 status=status)

    model_class_name = next(
        (p.get("metadata", {}).get("model_name") for p in suites.values()
         if p.get("metadata", {}).get("model_name")), None)

    rows = classify(suites)
    rows += derive(suites.get("spatial", {}), suites.get("symbolic", {}),
                   suites.get("online_symbolic", {}))

    guards = {}
    guards.update(_probe_interfaces(model_class_name))
    for payload in suites.values():
        for k, v in payload.get("metrics", {}).items():
            if SPEC.get(k, {}).get("role") == "guard":
                guards[k] = v

    so_probe, n_so = {}, 0
    sp = os.path.join(root, "serial_order_probe.json")
    if os.path.exists(sp):
        with open(sp) as f:
            so_probe = json.load(f)
        for k, v in (so_probe.get("summary") or {}).items():
            spec = SPEC.get(k)
            if spec is None:
                continue
            n = (spec["norm"](v, **spec["norm_kw"])
                 if spec["norm"] is not None else float("nan"))
            rows.append(dict(
                key=k, raw=v, n=n, derived=False,
                source="bin/probe_serial_order.py (not yet a suite metric — audit S-O2)",
                note=spec["note"] + " Computed by the serial-order probe; it becomes a "
                     "suite metric once measure_recall_associative returns decoded "
                     "identities.",
                **{f: spec[f] for f in ("capacity", "dimension", "kind", "section",
                                        "suite", "v", "role")}))
            n_so += 1

    for k, v in (so_probe.get("summary") or {}).items():
        if SPEC.get(k, {}).get("role") == "guard":
            guards[k] = v
    if n_so:
        suite_meta["serial_order (probe)"] = dict(
            sections=1, n_trials=so_probe.get("n_trials"),
            kwargs=json.dumps({"epochs": so_probe.get("epochs")}),
            status=f"ran — supplies {n_so} specified metrics not yet emitted by the "
                   f"suite (audit S-O2)")

    # Bound unconditionally: it is read below whether or not a focused run was
    # supplied, and without this any invocation without --schema-focused dies
    # with UnboundLocalError before writing the scorecard.
    focused: Dict[str, Any] = {}
    n_overlaid = 0
    if args.schema_focused:
        fp = os.path.join(os.path.abspath(args.schema_focused), "symbolic",
                          "metrics.json")
        if not os.path.exists(fp):
            fp = os.path.join(os.path.abspath(args.schema_focused),
                              "schema_consistency_metrics.json")
        with open(fp) as f:
            focused = json.load(f)
        proto = focused.get("metadata", {}).get("interference_protocol")
        if proto and proto != "focused":
            raise SystemExit(
                f"--schema-focused points at a run with "
                f"interference_protocol={proto!r}; it must be 'focused'.")
        n_overlaid = overlay_focused(rows, focused)
        # The overlay changes which protocol the damage read-outs came from, so
        # the validity guard must come from the same place.
        for k, v in focused.get("metrics", {}).items():
            if k == "schema_interference_valid":
                guards[k] = v
        suite_meta["schema_consistency (focused)"] = dict(
            sections=1, n_trials=focused.get("metadata", {}).get("n_probe_trials"),
            kwargs=json.dumps({"interference_protocol": "focused"}),
            status=f"ran — supplies {n_overlaid} damage read-outs "
                   f"(interference_*, prefix_recall)")

    n_sat = gate_saturated_mask_recall(rows, guards)
    if n_sat:
        print(f"[warn] {n_sat} cue-masking recall read-outs demoted: mask_at_ceiling "
              f"is True, so they carry no information about this arm.")
    n_demoted = gate_damage_readouts(rows, guards)
    n_demoted += gate_chain_readouts(rows, guards)
    if n_demoted:
        print(f"[warn] {n_demoted} schema damage read-outs demoted: no "
              f"interference_protocol='focused' run supplied (--schema-focused).")

    primary = rollup(rows, guards, inclusive=False)
    incl = rollup(rows, guards, inclusive=True)

    series = {}
    for suite in ("spatial", "symbolic"):
        for k, v in suites.get(suite, {}).get("series", {}).items():
            if k in ("noise_sweep", "similarity_sweep", "length_sweep",
                     "convergence_curve", "multiple_sequences",
                     "duplicate", "within", "across", "random"):
                series[k] = v
    for k, v in focused.get("series", {}).items():
        series[f"focused_{k}"] = v

    # ---- ALTERNATE scoring: forgetting read from margin, not accuracy ---------
    # `delta_mrr_forgetting` is quantised to 1/P under the clean single probe and
    # sits at 0 for every arm at ceiling (theta -0.006 vs EP -0.108 in margin,
    # both 0.000 in MRR). This recomputes the profile with Continual retention's
    # load dimension scored from `delta_margin_forgetting_lists`, normalised as
    # the FRACTION OF MARGIN RETAINED over list pairs:
    #     n = clip(1 + delta_margin_lists / margin_before_lists, 0, 1)
    # so 0 change -> 1.0, total loss of margin -> 0.0, self-referenced and hence
    # comparable across arms whose absolute margins differ. Written as a second
    # radar and a `capacities_margin_scored` block; the published profile is
    # untouched. Switching the scored key for real is a one-line SPEC change.
    primary_margin = incl_margin = None
    _by = {r["key"]: r for r in rows}
    _dm, _mb = _by.get("delta_margin_forgetting_lists"), _by.get("multiple_seq_margin_before_lists")
    if _dm and _mb and isinstance(_mb.get("raw"), (int, float)) and _mb["raw"] > 1e-9 \
            and isinstance(_dm.get("raw"), (int, float)):
        n_alt = max(0.0, min(1.0, 1.0 + _dm["raw"] / _mb["raw"]))
        rows_alt = []
        for r in rows:
            r2 = dict(r)
            if r2["key"] == "delta_mrr_forgetting":
                r2["role"] = "diagnostic"
            elif r2["key"] == "delta_margin_forgetting_lists":
                r2["role"], r2["n"] = "score", n_alt
            rows_alt.append(r2)
        primary_margin = rollup(rows_alt, guards)
        incl_margin = rollup(rows_alt, guards, inclusive=True)

    payload = dict(
        model=model, results_dir=root,
        formula={
            "metric_to_dimension": "D_d = sum_m(v_m * n_m) / sum_m(v_m)",
            "dimension_to_capacity": "C = sum_d(w_d * D_d) / sum_d(w_d) over available d",
            "coverage": "sum_{d available} w_d / sum_{d intended} w_d",
        },
        dimension_weights=DIMENSION_WEIGHTS,
        schema_focused_overlay=dict(
            applied=bool(n_overlaid), n_metrics=n_overlaid,
            source=args.schema_focused,
            keys=list(FOCUSED_ONLY_KEYS)),
        suite_metadata=suite_meta,
        capacities=primary,
        capacities_inclusive=incl,
        capacities_margin_scored=primary_margin,
        margin_scored_key="delta_margin_forgetting_lists",
        metrics=sorted(rows, key=lambda r: (r["capacity"], r["dimension"], r["key"])),
        guards=guards,
    )
    with open(prefix + ".json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
    write_markdown(prefix + ".md", model, rows, primary, incl, guards, suite_meta, series)
    write_radar(prefix + "_radar.png", model, primary, incl)
    if primary_margin is not None:
        write_radar(prefix + "_radar_margin.png", model + "  [forgetting scored by margin]",
                    primary_margin, incl_margin)

    print(f"\nCapacity profile — {model}\n" + "=" * 64)
    if primary_margin is not None:
        _c = "Continual retention"
        print(f"  [alternate] {_c} scored by margin: {primary_margin[_c]['score']:.3f}  "
              f"(published, by MRR: {primary[_c]['score']:.3f}) -> *_radar_margin.png")
    for c in CAPACITY_ORDER:
        p = primary[c]
        print(f"  {c:<26} {_fmt(p['score'])}   coverage {p['coverage']:.0%}   "
              f"(inclusive {_fmt(incl[c]['score'])})")
    print("=" * 64)
    print(f"  {len(rows)} metrics classified "
          f"({sum(1 for r in rows if r['role'] == 'score')} scored, "
          f"{sum(1 for r in rows if r['role'] == 'protocol_limited')} protocol-limited, "
          f"{sum(1 for r in rows if r['role'] in ('guard','config','control','diagnostic'))} "
          f"non-scoring)")
    for p in (prefix + ".json", prefix + ".md", prefix + "_radar.png"):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
