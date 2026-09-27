# MemVal: notes for agents

MemVal is a **benchmark, not a model**. An arm that floors a section is a
result to report, not a bug to tune away. Overview: `README.md`. Docs map:
`docs/index.md`. CLI: `docs/running.md`.

## Roster

Six arms, registered in `MODEL_REGISTRY` in `bin/run_benchmark.py`:
- `ahn` (AsymmetricHopfieldNetwork; `hopfield` is an alias)
- `theta` (ThetaPhaseSequenceNetwork)
- `original_eqprop` (OriginalEqPropSequenceNetwork)
- `temporal_pc` (MultilayerTemporalPCNetwork)
- `predictive_recirculation` (PredictiveRecirculationNetwork)
- `dts_esn` (DTSESNSequenceNetwork)

The classes are in `memval/models/baselines/`. Settings and provenance are in
`docs/models/model_table.md`.

Retired arms (spiking, DG/EWC EqProp, GPT-2, KNN, HiCL, ca3net) and the old
notebooks, standalone scripts and dashboard are at git tag `archive/pre-cleanup`.
Recover a file with `git show archive/pre-cleanup:<path>`. Do not reintroduce
them without being asked.

## Contracts every arm must keep

- **`predict_next` is a pure function of the cue.** Probes are independent, so
  a stateful arm infers its state from the cue without mutating what it carries
  forward. `recall` threads its state itself.
- **`fit_sequence` trains exactly the `epochs=` it is passed.** Don't loop over
  `self.n_epochs` and ignore the kwarg. `tests/test_epochs_kwarg_contract.py`
  enforces this across the registry.
- **Capabilities are declared, not probed.** `memval/models/capabilities.py`
  defines `OnlineTrainable`, `rollout_mode`, `TemporallyClocked` and the rest.
  A section an arm cannot run is scored NOT APPLICABLE, never zero.

## Protocol rules

- Exposure is criterion-referenced by default: sections train to a criterion
  and report what it cost (`memval/benchmarks/exposure.py`).
- Paper-derived presentation and exposure may be set per arm. Choosing
  parameters because they improve scores may not.
- New sections must be reproducible: seed every random draw, including probe
  noise, and never rely on hash order.

## Working here

- Tests: `pytest tests/` (~420 tests, about a minute). No torch or other extras
  are needed.
- **Run one numpy-heavy process at a time.** Concurrent runs oversubscribe BLAS
  threads and slow down 10–30×.
- `results/` is committed. `zoo_capacity_run/` is the main comparison; rebuild
  its `_compare/` and `index.html` with the builders listed in `docs/running.md`
  after changing any scorecard.
- `scratch/` is git-ignored and holds local clones and throwaway runs. Never make
  committed code depend on it.
