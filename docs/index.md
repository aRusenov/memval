# Docs map

Start with `README.md` (the project) and `running.md` (the CLI).

## `capacities/`: what is measured
- `capacities.md`: the five capacities, in the paper's prose. Cited by the scorer.
- `capacity_questions.md`: each capacity as questions, dimensions and capability tags.
- `capacity_coverage_audit.md`: what the suites actually measure, with the exclusion rules (C1/C2) and pending items.

## `sections/`: how each benchmark section works
- `encoder_design.md`: the symbolic stimulus space and its encoders.
- `probe_protocol.md`: the cue each section presents, and how many times.
- `rollout_protocol.md`: what `recall()` feeds back; the rollout modes.
- `disambiguation_design.md`: sequence disambiguation (spatial and symbolic).
- `schema_benchmark_design.md`: schema consistency (acquisition and interference).
- `spatial_reversal_design.md`: the T-maze reversal section.
- `cognitive_phenomena_design.md`: descriptive L0 read-outs, never scored.

## `models/`: the zoo
- `model_table.md`: the six arms as run, with budgets, provenance audit and taxonomy.

## `reports/`: results write-ups
- `capacity_report_ahn.md`: the AHN reference run and the per-arm runbook.
- `capacity_report_zoo.md`: the zoo capacity run, its cross-arm findings and each arm.

## `paper/`: drafts
- `paper_intro.md`: introduction draft.
- `paper_methods_protocol.md`: methods, the recall protocol.
- `paper_models_roster.md`: methods, the model roster. Needs a pass for the six-arm roster.
- `results_intro_capacities.md`: results intro, grounding the capacities.

## `proposals/`: designed, not built
- `nonlinearity_benchmark_design.md`: non-linearity sections plus a linear oracle.
- `interval_encoding_design.md`: interval encoding.
- `free_recall_design.md`: free recall (demoted).
- `true_online_learning.md`: the raw-stream ingestion regime.
- `ep_theta_phase_exploration.md`: theta rhythm as the orchestrator of EP's two phases.

## `background/`: context and history
- `benchmark_landscape.md`: survey of continual-learning and sequence-memory benchmarks.
- `ep_continual_debugging_case_study.md`: diagnosing EP's forgetting, stage by stage.
- `online_continual_benchmark.md`: design of the removed online-continual suite.

## `figures/`
Figure sources (`*.py`, `*.tex`) and their rendered outputs.

Material that has been removed from the tree (retired arms, their port notes and
reports, notebooks, the dashboard) is at git tag `archive/pre-cleanup`.
