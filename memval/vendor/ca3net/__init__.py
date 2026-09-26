"""Vendored core of `KaliLab/ca3net`.

Reference implementation for:

    A. Ecker, B. Bagi, E. Vertes, O. Steinbach-Nemeth, M. Kerekes, E. Papp,
    I. Papp, S. Kali, T. Freund, N. Hajos, A. Gulyas (2022).
    "Hippocampal sharp wave-ripples and the associated sequence replay emerge
    from structured synaptic interactions in a network model of area CA3."
    eLife 11:e71850.  https://github.com/KaliLab/ca3net

Upstream commit: dc0cf27b863f6b00e93be6a4f932e334dd508e29 (2026-04-13)
Pristine reference clone: ``scratch/ca3net_repo/``

What is vendored, and what is not
---------------------------------
Vendored (4 of 16 upstream scripts): ``poisson_proc``, ``helper``, ``stdp``,
``spw_network`` -- the minimal closure needed to run ``stdp.learning()`` (STDP over
presented spike trains) and ``spw_network.run_simulation()`` (the recurrent CA3
network, spontaneous or cued).

NOT vendored: ``bayesian_decoding``, ``detect_replay``, ``detect_oscillations``,
``plots``, ``analyse_movement``, ``generate_spike_train``, ``modify_wmx``,
``gamma_network``, ``optimization/``. The first four are repointed at ``_shims``;
see that module for why. ``generate_spike_train`` is replaced by
``memval/models/baselines/_ca3net_io.py``, which encodes MemVal sequences instead
of simulated maze running -- see ``docs/ca3net_port_plan.md`` section 2.

Edits to the vendored sources
-----------------------------
All are marked ``# VENDOR EDIT`` inline. In summary:

1. Import lines repointed (``from helper`` -> ``from .helper``; ``plots`` /
   ``detect_*`` -> ``._shims``); ``pywt`` guarded (analysis-only).
2. ``stdp.py`` no longer calls ``set_device("cpp_standalone")`` at import time --
   it globally switched the Brian2 device and made it impossible to run learning
   and recall in one process. Use :func:`use_standalone` explicitly.
3. ``spw_network.py`` defaults to the ``cython`` codegen target rather than
   ``numpy`` (a 10 s run: ~12 min -> ~20 s).
4. ``run(10000*ms)`` -> ``run(SIM_DURATION_MS*ms)``.

The len_sim landmine
--------------------
``helper._avg_rate()`` maps the population-rate array onto a time axis of length
``helper.len_sim`` *regardless of the actual simulation duration*, and upstream
callers convert bin indices to ms using it. A run whose duration differs from
``len_sim`` silently reports mis-scaled event times. :func:`configure` is the only
supported way to change duration or population size, because it keeps
``helper.len_sim``, ``spw_network.SIM_DURATION_MS`` and the ``nPCs``/``nBCs``
constants in the three modules in lockstep.

Licence
-------
Upstream ships a verbatim MIT ``LICENSE`` whose copyright line reads
``Copyright (c) 2015 yaringal`` -- an unrelated party, seven years before the
paper. GitHub reports "MIT" on text match alone. Redistribution status is
therefore **unresolved**; see ``docs/ca3net_port_plan.md`` Phase 0. This is the
same open question as `tpc-arm-added` and `spiking-eqprop-arm-added`.
"""

import importlib

__all__ = ["configure", "use_standalone", "helper", "stdp", "spw_network",
           "poisson_proc"]

_SUBMODULES = {"helper", "stdp", "spw_network", "poisson_proc", "_shims"}


def __getattr__(name):
    """Lazy submodule import: touching `ca3net` must not pull in Brian2."""
    if name in _SUBMODULES:
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def use_standalone(directory=None):
    """Switch Brian2 into C++ standalone mode (what upstream's `stdp.py` did at
    import time). ~20x faster for the learning stage, but it is a *global,
    one-way* switch: the recall network cannot then run in the same process.
    Call it in a dedicated learning process only.
    """
    from brian2 import set_device
    set_device("cpp_standalone", directory=directory)


def configure(n_pcs=None, n_bcs=None, sim_duration_ms=None):
    """Set population sizes and simulation duration across every vendored module.

    Use this rather than assigning the module constants directly: `len_sim` and
    `SIM_DURATION_MS` must agree or the rate-binning helpers mis-scale time (see
    the module docstring).
    """
    helper = __getattr__("helper")
    spw = __getattr__("spw_network")
    stdp = __getattr__("stdp")

    if n_pcs is not None:
        for m in (helper, spw, stdp):
            setattr(m, "nPCs", int(n_pcs))
    if n_bcs is not None:
        for m in (helper, spw):
            setattr(m, "nBCs", int(n_bcs))
    if sim_duration_ms is not None:
        d = float(sim_duration_ms)
        helper.len_sim = d
        spw.SIM_DURATION_MS = d

    return {"nPCs": helper.nPCs, "nBCs": helper.nBCs,
            "len_sim": helper.len_sim, "SIM_DURATION_MS": spw.SIM_DURATION_MS}
