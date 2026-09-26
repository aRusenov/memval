# Running MemVal Pipelines from the CLI

## Prerequisites

All commands assume you are in the project root (`memval/`) with the virtual environment activated:

```bash
source .venv/bin/activate
```

To install the project in editable mode (required once):

```bash
pip install -e .
```

---

## Entry Point

All benchmark pipelines are driven by a single script:

```
bin/run_benchmark.py
```

### Full Usage

```
python bin/run_benchmark.py --model <model> [--suite <suite>] [options]
```

---

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model` | ✅ | — | Model to benchmark. See [Available Models](#available-models). |
| `--suite` | | `all` | Suite to run: `spatial`, `symbolic`, `online_symbolic`, or `all`. |
| `--output-dir` | | `./results` | Directory for results JSON and plots. |
| `--n-trials` | | `30` | Number of evaluation trials per metric. |
| `--mrr-threshold` | | `0.95` | MRR convergence threshold (symbolic suite). |
| `--max-epochs` | | `500` | Max training epochs for convergence sweep (symbolic suite). |
| `--noise-scale` | | `0.05` | Gaussian noise σ for cued recall trials (online symbolic suite). |

---

## Available Models

| `--model` | Spatial | Symbolic | Online Symbolic | Notes |
|---|:---:|:---:|:---:|---|
| `hopfield` | ✅ | ✅ | ✅ | Fast. Good baseline. |
| `dg_eqprop` | ✅ | ✅ | | Dentate Gyrus + Equilibrium Propagation. |
| `ewc_dg_eqprop` | ✅ | ✅ | | DG EqProp + Elastic Weight Consolidation. |
| `gpt2` | | ✅ | | Requires `torch`. Slow. |

---

## Suite Descriptions

### `spatial`
Runs the spatial benchmarking suite:
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

To run the standalone T-Maze Odour Disambiguation demo directly:
```bash
python demos/tmaze_disambiguation_demo.py --model hopfield --fit-epochs 100
```

### `symbolic`
Runs the symbolic benchmarking suite on word lists:
1. **Convergence Trajectory** — epochs to reach MRR threshold.
2. **Sequence Length Sweep** — capacity limits vs. list length.
3. **Catastrophic Forgetting** — MRR before/after learning a second list.
   The T=2 corner of the chain below.
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
6. **Schema Consistency** — acquisition rate and interference as a function of
   how consistent new material is with an acquired schema.

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
Runs the online (event-by-event) symbolic suite:
1. **One-shot and multi-pass recall** — MRR at 1, 5, 20 presentations.
2. **ISI Tolerance** — recall across inter-stimulus interval durations.

---

## Examples

### Run all suites with the DG EqProp model

```bash
python bin/run_benchmark.py --model dg_eqprop
```

### Run only the spatial suite with Hopfield

```bash
python bin/run_benchmark.py --model hopfield --suite spatial
```

### Run the symbolic suite with more trials and a custom output directory

```bash
python bin/run_benchmark.py \
    --model hopfield \
    --suite symbolic \
    --n-trials 50 \
    --output-dir ./results/experiment_01
```

### Run the symbolic suite with EWC, relaxed convergence threshold

```bash
python bin/run_benchmark.py \
    --model ewc_dg_eqprop \
    --suite symbolic \
    --mrr-threshold 0.80 \
    --max-epochs 1000
```

### Run spatial only with fewer trials for a quick sanity check

```bash
python bin/run_benchmark.py --model hopfield --suite spatial --n-trials 5
```

---

## Output Structure

Results are written to `--output-dir/<ModelClassName>/<suite>/`:

```
results/
└── HopfieldSequenceNetwork/
    ├── spatial/
    │   ├── metrics.json
    │   └── plots/
    │       ├── tmaze_completion.png
    │       ├── spatial_seq_completion.png
    │       └── tmaze_disambiguation.png
    ├── symbolic/
    │   ├── metrics.json
    │   ├── cue_masking_metrics.json          # section 4b's own record
    │   ├── schema_consistency_metrics.json   # section 6's own record
    │   └── plots/
    │       ├── convergence_curve.png
    │       ├── length_curves.png
    │       ├── multiple_seq_forgetting.png
    │       ├── noise_invariance.png
    │       ├── cue_masking.png
    │       ├── semantic_similarity.png
    │       ├── symbolic_disambiguation.png
    │       └── schema_consistency.png
    └── online_symbolic/
        ├── metrics.json
        └── plots/
            ├── online_convergence.png
            └── isi_tolerance.png
```

`metrics.json` contains two top-level keys:
- `metrics` — flat dict of scalar summary values (printed to stdout at end of run).
- `series` — full data series for each sweep (e.g., sweep records, recall curves).

---

## Running the Dashboard

The `dashboard/` directory contains a static web app that visualizes benchmark
results. It loads the `results/` metrics and plots produced by
`bin/run_benchmark.py`, so generate results first (see [Examples](#examples))
before opening it.

The dashboard uses `fetch()` to load `../results/<model>/<suite>/metrics.json`
and its Markdown docs, which browsers block over `file://`. **You must serve it
over HTTP from the project root** — do not open `dashboard/index.html` directly.

From the project root:

```bash
python -m http.server 8000
```

Then open:

```
http://localhost:8000/dashboard/
```

Notes:
- Serve from the project root (not from inside `dashboard/`) so the app can
  resolve `../results/`.
- An internet connection is needed on first load — fonts, icons, and the
  Markdown parser are loaded from CDNs.
- Models or suites without a `metrics.json` are simply skipped (a warning is
  logged to the browser console); run the corresponding benchmark to populate
  them.

---

## Running Tests

```bash
# All tests
pytest tests/

# Only spatial disambiguation tests
pytest tests/test_spatial_disambiguation.py -v

# Only pipeline integration tests
pytest tests/test_pipelines.py -v
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
(`hopfield` 1, EP arms 100–300). Every span, MRR and forgetting delta was
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
python bin/run_benchmark.py --model hopfield --suite symbolic \
    --benchmark-args sequence_length:epochs=300
```

`criterion` and `max_epochs` are tunable the same way. See
`memval/benchmarks/exposure.py`.
