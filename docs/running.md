# Running MemVal

All commands assume the project root as the working directory and the virtual
environment active:

```bash
source .venv/bin/activate
pip install -e ".[dev]"   # once
```

---

## Entry point

Every suite runs through one script:

```
python bin/run_benchmark.py --model <model> [--suite <suite>] [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | — (required) | Arm to run. See [Models](#models). |
| `--suite` | `all` | `spatial`, `symbolic`, `online_symbolic`, or `all`. |
| `--benchmarks` | all | Comma-separated subset of sections. Needs a single `--suite`. |
| `--benchmark-args` | — | Per-section overrides, `section:key=value,...` (e.g. `sequence_length:epochs=300`). |
| `--list-benchmarks` | | Print the sections of each suite and exit. |
| `--output-dir` | `./results` | Where results JSON and plots go. |
| `--n-trials` | `30` | Evaluation trials per metric. |
| `--epochs` | — | Global override for the fixed-epoch sections. Precedence: `--benchmark-args` > `--epochs` > model default > section default. |
| `--max-epochs` | per arm | Ceiling of the `presentation_duration` sweep. |
| `--mrr-threshold` | `0.95` | MRR convergence threshold (symbolic). |
| `--noise-scale` | `0.05` | σ of the Gaussian ISI filler in the online suite. |

---

## Models

The registry is `MODEL_REGISTRY` in `bin/run_benchmark.py`. Settings, budgets and
their provenance are in `docs/models/model_table.md`.

| `--model` | class | spatial | symbolic | online | notes |
|---|---|:---:|:---:|:---:|---|
| `ahn` | `AsymmetricHopfieldNetwork` | ✅ | ✅ | skipped | Reference arm. Fast. Batch-only, so the online sections skip it. The old key `hopfield` still resolves. |
| `theta` | `ThetaPhaseSequenceNetwork` | ✅ | ✅ | ✅ | Hasselmo 2002; controlled pair for AHN. |
| `original_eqprop` | `OriginalEqPropSequenceNetwork` | ✅ | ✅ | ✅ | Scellier & Bengio 2017. Slowest arm. |
| `temporal_pc` | `MultilayerTemporalPCNetwork` | ✅ | ✅ | ✅ | Tang et al. 2023. |
| `predictive_recirculation` | `PredictiveRecirculationNetwork` | ✅ | ✅ | ✅ | Chen et al. 2024, local rule; controlled pair for tPC. |
| `dts_esn` | `DTSESNSequenceNetwork` | ✅ | ✅ | ✅ | Tanaka et al. 2022; the only arm with interval timing. |

Retired arms (spiking, DG/EWC EqProp, GPT-2, KNN, HiCL, ...) live at git tag
`archive/pre-cleanup`.

---

## Suites

`python bin/run_benchmark.py --list-benchmarks` prints the authoritative list.
Design notes for each section are in `docs/sections/`.

### `spatial`

1. **T-Maze Pattern Completion** — place-cell encoded route recovery from a cue
   point. Reports `coverage` and `divergence_step` alongside MSE, because MSE
   saturates in the failure regime (centre-of-mass decoding pulls a lost readout
   to the arena middle, so a collapsed recall plateaus rather than diverging).
2. **T-Maze Odour Disambiguation** — two routes sharing a stem, told apart by an
   odour cue. Ships two blocks: the fully-cued corner (odour present through the
   arms) and a **graded availability sweep** that withdraws the odour
   progressively earlier, leaving `delay = shared_end - zone_end` shared steps
   with no discriminator. Withdrawn rows are expected at chance under the
   memoryless `predict_next` probe — a protocol finding, not a model ranking.
3. **T-Maze Reversal** — how fast an invalidated place→reward association is
   overwritten, with and without an explicit extinction stage.

### `symbolic`

| section | what it measures |
|---|---|
| `presentation_duration` | Convergence: epochs to reach the MRR threshold. |
| `sequence_length` | Capacity vs list length (+ feedback modes). |
| `multiple_sequences` | Catastrophic forgetting, the T=2 corner of the chain, with an interleaved-ingestion condition. |
| `continual_chain` | Retention matrix over a task chain: load axis + acquisition under load. |
| `paired_associate` | AB/AC shared-cue overwrite, with a disjoint-cue control. |
| `noise_invariance` | Gaussian noise on the retrieval cue. |
| `cue_masking` | Cue *completeness*: graded feature masking. |
| `cue_availability` | Cue *availability*: a whole modality absent (audio × symbolic). |
| `semantic_similarity` | Semantic interference across category variance. |
| `symbolic_disambiguation` | Overlapping sequences: graded discriminability + load. |
| `schema_consistency` | Acquisition rate vs consistency with prior knowledge. |
| `cognitive_phenomena` | Descriptive L0 read-outs (Kahana 2020 §4); never scored. |
| `interval_retention` | Serial order 5.4, the gap as the cue. Needs `TemporallyClocked`. |
| `interval_generation` | Serial order 5.5, the gap as the output. Needs `TimingPredictive`. |

The two interval sections are gated on a declared capability. An arm without it
is skipped and scored NOT APPLICABLE, never zero; today only `dts_esn` qualifies.

Notes on selected sections:

3b. **Continual Chain** — a 6-task chain scored as a retention matrix. Reports
   stability (ACC, forgetting, and the retained fraction at every interposition
   gap), plasticity (the diagonal: acquisition under load, and intransigence),
   and the joint index. Ends with a selective-retention phase: the oldest half
   of the chain is re-presented and everything is re-scored, so what survives
   can be checked against what is still in use. Args: `n_tasks`, `seq_len`,
   `rehearse`, `n_rehearsed`, `epochs`.
3c. **Paired Associate (AB/AC)** — the same cues re-paired with new targets, so
   the old response becomes an error. Runs a disjoint-cue control at an equal
   budget, and reports the difference as the cue-competition cost. Args:
   `n_pairs`, `ab_trials`, `phase2_trials`, `epochs`.
4. **Noise Invariance** — Gaussian cue noise sweep.
5. **Semantic Similarity** — category variance sweep.
5b. **Symbolic Disambiguation** — N sequences sharing a stretch, discriminated
   by a context block. Five sweeps. Axis B
   grades how confusable the discriminators are via `category_variance` with the
   cue present at the decision, axis C raises N, and axis A withdraws the cue
   before the decision. This is where discriminability is actually gradeable:
   `semantic_similarity` varies the similarity of the ITEMS, where transitions
   are bijective and no cue ever demands two successors, so it reads ~0 with
   both sides at ceiling. Args: `n_episodes`, `total_length`, `shared_fraction`,
   `zone_fraction`, `shared_position`, `middle_position`, `embedding_dim`,
   `n_discriminator_dims`, `disc_scale`, `seed`. Two further sweeps separate cue
   DURATION from cue-free DELAY, which a `zone_fraction` sweep alone confounds
   because the zone starts at the corridor entrance. The load sweep carries an
   orthogonal-discriminator control, so a falling curve can be attributed to
   disambiguation rather than to ordinary capacity. A shared-MIDDLE condition
   gives each sequence a unique prefix, with an endogenous rung in which every
   sequence gets the same discriminator so the prefix is all that remains.

**Exposure baseline and deviation guard.** Every criterion-referenced section
settles its own exposure on its own material, which keeps *arms* comparable
within a section: each is measured at the same functional point rather than the
same epoch count, and the registry defaults span 1 to 300 epochs across the
taxonomy. What that leaves unreported is the other axis, so the run also settles
one reference exposure per arm on the shared 7-word list and reports every
section's ratio to it (`<prefix>_exposure_vs_baseline`), the worst deviation and
the section that produced it. Sections are *not* pinned to the reference:
"converged" is a property of model and material together, so pinning would start
a heavier section undertrained and make its failures ambiguous between the
manipulation and the undertraining. The guard just makes the spread visible. A
run whose worst ratio exceeds `EXPOSURE_DEVIATION_BAND` (8x) prints a warning and
sets `exposure_within_band` false, meaning cross-section comparison for that arm
is not like-for-like.

### `online_symbolic`

| section | what it measures |
|---|---|
| `online_convergence` | One-shot and multi-pass recall: MRR at 1, 5 and 20 presentations. |
| `isi_tolerance` | Recall across inter-stimulus interval durations. |

Only arms that declare `OnlineTrainable` run these sections.

---

## Examples

```bash
# Quick sanity check: spatial suite on the reference arm
python bin/run_benchmark.py --model ahn --suite spatial --n-trials 5

# One section of one suite
python bin/run_benchmark.py --model original_eqprop --suite symbolic \
    --benchmarks continual_chain --benchmark-args continual_chain:n_tasks=3

# Everything, to a separate output directory
python bin/run_benchmark.py --model dts_esn --output-dir ./results/experiment_01
```

---

## A full capacity run for one arm

`bin/run_capacity_arm.sh` runs the whole runbook for one arm: both suites, the
streamed regime, the focused-protocol schema re-run, the serial-order probe and
the sparse-code capacity curves. It then scores and renders the arm.

```bash
bin/run_capacity_arm.sh <model> <ClassName> <out-dir> [n-trials] [extra args...]
bin/run_capacity_arm.sh theta ThetaPhaseSequenceNetwork results/zoo_capacity_run 30
```

Its header explains how to cap exposure on a slow arm: use `max_epochs=N`, not
`epochs=N`. The procedure it automates is `docs/reports/capacity_report_ahn.md`
§1. Then, across arms:

```bash
# scorecard for one arm (run_capacity_arm.sh does this)
Z=results/zoo_capacity_run; C=ThetaPhaseSequenceNetwork
python bin/score_capacities.py --results-dir $Z/$C --schema-focused $Z/_schema_focused/$C --model $C
python bin/build_scorecard_page.py --scorecard $Z/$C/capacity_scorecard.json --out $Z/$C/capacity_scorecard.html

# cross-arm comparison and the browsable index
python bin/compare_capacity_arms.py \
    --scorecard "AHN=$Z/AsymmetricHopfieldNetwork/capacity_scorecard.json" \
    --scorecard "theta=$Z/ThetaPhaseSequenceNetwork/capacity_scorecard.json" \
    --scorecard "tPC=$Z/MultilayerTemporalPCNetwork/capacity_scorecard.json" \
    --scorecard "recirc=$Z/PredictiveRecirculationNetwork/capacity_scorecard.json" \
    --scorecard "EP=$Z/OriginalEqPropSequenceNetwork/capacity_scorecard.json" \
    --scorecard "DTS-ESN=$Z/DTSESNSequenceNetwork/capacity_scorecard.json" \
    --out-dir $Z/_compare
python bin/plot_capacity_overview.py
python bin/build_results_index.py --results-dir $Z
```

Run these one at a time. Concurrent numpy-heavy processes oversubscribe BLAS
threads and slow each other down 10–30×.

To browse `results/zoo_capacity_run/index.html`, serve the repo over HTTP
(`python -m http.server 8010`, also the `results` entry in `.claude/launch.json`).

---

## Output structure

Results are written to `--output-dir/<ClassName>/<suite>/`:

```
results/
└── AsymmetricHopfieldNetwork/
    ├── spatial/
    │   ├── metrics.json
    │   └── plots/
    ├── symbolic/
    │   ├── metrics.json
    │   ├── cue_masking_metrics.json          # cue_masking's own record
    │   ├── schema_consistency_metrics.json   # schema_consistency's own record
    │   └── plots/
    └── online_symbolic/
        ├── metrics.json
        └── plots/
```

`metrics.json` has two top-level keys:
- `metrics`: a flat dict of scalar summaries, also printed at the end of a run.
- `series`: the full data series for each sweep.

The capacity-run layout (scorecards, `_compare/`, `index.html`) is in
`docs/reports/capacity_report_zoo.md` §6.

---

## Running tests

```bash
pytest tests/                          # ~420 tests, about a minute
pytest tests/test_pipelines.py -v      # end-to-end pipeline smoke tests
```

---

## Exposure policy — criterion-referenced by default

Since 2026-09-02 every trainable section trains until it reaches a **criterion**
rather than for a fixed number of epochs, and reports what that cost:

```
<section>_epochs_to_criterion   what the exposure turned out to be
<section>_criterion_reached     False = censored; every metric in the section
                                is then a lower bound, not a capacity
<section>_exposure_mode         "criterion" (default) or "fixed"
```

**Why.** Exposure was resolved `CLI --epochs > model default > section fallback`,
and the model defaults in `MODEL_REGISTRY` span **1 to 300** across the taxonomy
(`ahn` 1, EP arms 100–300 at the time). Every span, MRR and forgetting delta was
therefore measured at a different point on each arm's learning curve, and the
difference was invisible in `metrics.json` — two arms reported the same metric
name from incomparable protocols.

**The criterion read-out sits upstream of the scored one**, so a section cannot
guarantee its own result: `sequence_length` settles on cued recall and scores
rollout span; `noise_invariance` settles at σ = 0 and scores the σ sweep;
`cue_masking` settles on the unmasked cue and scores the masking sweep;
`tmaze_completion` settles on teacher-forced one-step prediction and scores
open-loop coverage. Where the two would coincide, the exposure itself is the
informative quantity and the section says so.

**Pinning a budget** (to reproduce an older number, or to hold a section at a set
exposure) is still available per section:

```bash
python bin/run_benchmark.py --model ahn --suite symbolic \
    --benchmark-args sequence_length:epochs=300
```

`criterion` and `max_epochs` are tunable the same way. See
`memval/benchmarks/exposure.py`.
