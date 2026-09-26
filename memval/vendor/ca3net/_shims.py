"""Stand-ins for the upstream analysis/plotting modules we deliberately do not vendor.

Upstream's `spw_network.py` and `stdp.py` import `plots`, `detect_replay` and
`detect_oscillations` at module level, but use them only inside `analyse_results()`
and their `__main__` blocks. We need `run_simulation()` and `learning()`, not the
oscillation/replay analysis, so vendoring those modules would drag in `pywt`,
`seaborn` and `pandas` for code MemVal never calls.

Rather than delete the import lines (which would make the vendored sources diverge
from upstream in a way that is hard to audit), the import lines are repointed here
and every symbol raises on use. Same tactic as `_artemis.py` in
`memval/vendor/spiking_eqprop/`.

Why those modules are not vendored
----------------------------------
`bayesian_decoding.py` / `detect_replay.py` decode a **1-D scalar position** from
per-neuron tuning curves and fit a straight line through the posterior. That ties
the arm to 1-D spatial tasks. MemVal instead treats the PC population as a generic
assembly substrate and decodes with `memval/models/baselines/_ca3net_io.py` — see
`docs/ca3net_port_plan.md` §2.
"""


def _unavailable(name, module):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            "ca3net.%s() is upstream analysis code that MemVal does not vendor "
            "(it lived in `%s.py`). MemVal decodes with "
            "memval/models/baselines/_ca3net_io.py instead -- see "
            "docs/ca3net_port_plan.md section 2. The pristine upstream copy is at "
            "scratch/ca3net_repo/ if you need it." % (name, module)
        )
    _raise.__name__ = name
    return _raise


# --- from plots.py -----------------------------------------------------------
for _n in ("plot_STDP_rule", "plot_wmx", "plot_wmx_avg", "plot_w_distr",
           "save_selected_w", "plot_weights", "plot_raster",
           "plot_posterior_trajectory", "plot_PSD", "plot_TFR", "plot_zoomed",
           "plot_detailed", "plot_LFP"):
    globals()[_n] = _unavailable(_n, "plots")

# --- from detect_replay.py ---------------------------------------------------
for _n in ("replay_circular", "slice_high_activity", "replay_linear"):
    globals()[_n] = _unavailable(_n, "detect_replay")

# --- from detect_oscillations.py ---------------------------------------------
for _n in ("analyse_rate", "ripple_AC", "ripple", "gamma", "calc_TFR",
           "analyse_estimated_LFP"):
    globals()[_n] = _unavailable(_n, "detect_oscillations")

del _n
