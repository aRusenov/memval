#!/usr/bin/env python
import os
import sys
import argparse
from typing import Dict, Any, Type, Optional

# Add the project root to sys.path so we can import memval
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memval.models.baselines import AsymmetricHopfieldNetwork
from memval.models.baselines import OriginalEqPropSequenceNetwork
from memval.models.baselines import MultilayerTemporalPCNetwork
from memval.models.baselines import PredictiveRecirculationNetwork
from memval.models.baselines import DTSESNSequenceNetwork
from memval.models.baselines import ThetaPhaseSequenceNetwork
from memval.benchmarks.spatial_pipeline import run_spatial_pipeline, SPATIAL_BENCHMARKS
from memval.benchmarks.symbolic_pipeline import run_symbolic_pipeline, SYMBOLIC_BENCHMARKS
from memval.benchmarks.online_symbolic_pipeline import (
    run_online_symbolic_pipeline,
    ONLINE_SYMBOLIC_BENCHMARKS,
)

# Benchmark sections available per suite, for --benchmarks subsetting.
SUITE_BENCHMARKS = {
    "spatial": SPATIAL_BENCHMARKS,
    "symbolic": SYMBOLIC_BENCHMARKS,
    "online_symbolic": ONLINE_SYMBOLIC_BENCHMARKS,
}

# Central Registry of Models
MODEL_REGISTRY = {
    "ahn": {
        "class": AsymmetricHopfieldNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 1/1 seeds reached, epochs [1], mean 1.0
            # spatial:  1/1 seeds reached, epochs [2], mean 2.0 (400-cell grid)
            # Zero-init: seed-invariant, so 1 seed not 5 (sd=0 is a property of
            # the arm, never a finding).
            "n_epochs": 2,
            "learning_rate": 0.1,
            "activation": "relu",
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 4}}
    },
    "original_eqprop": {
        "class": OriginalEqPropSequenceNetwork,
        # online_symbolic added 2026-09-03. The arm has declared OnlineTrainable
        # all along, so capacity question 2.3 (one-shot from a single streamed
        # pass) is answerable for it; omitting the modality reported that as a
        # missing instrument rather than as a result. The pipeline itself already
        # gates on supports_online() and writes status: not_applicable, so the
        # modality list was the only thing withholding the run.
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            # Paper values (Scellier & Bengio 2017, Table 2, 784-500-10 row), set
            # 2026-09-06 after the provenance audit (docs/model_table.md). Not
            # paper-matchable here: the paper's hard sigmoid rho(s)=0 v s ^ 1 (the
            # class offers tanh|sigmoid; kept tanh) and its asymmetric 20 free / 4
            # clamped iterations (one budget serves both phases; set to the free
            # count). dt=0.5 and learning_rate=0.1 already matched (eps, alpha_1).
            "n_hidden": 500,
            "n_settle_steps": 20,
            "learning_rate": 0.1,
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 5/5 seeds reached, epochs [19, 30, 17, 22, 19], mean 21.4
            # spatial:  5/5 seeds reached, epochs [40, 47, 43, 42, 40], mean 42.4 (400-cell grid)
            "n_epochs": 43,
            "beta": 1.0,
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 85}}
    },
    # --- non-EP arm: temporal predictive coding (Tang, Barron & Bogacz 2023) --
    # Registered 2026-09-03. Its absence was an oversight, not a decision:
    # models/baselines/__init__.py records tPC and DTS-ESN as "a crucial part of
    # the taxonomy ... compared normally", yet neither was reachable from this
    # registry, so no pipeline could run them.
    #
    # This is the ONLY registered arm that declares StatePrimeable, which makes
    # it the only arm for which the graded disambiguation sweep's `observe`
    # priming path fires at all. Every other entry here reports
    # state_primed=False, so their delay > 0 rows are protocol floors rather
    # than model results (docs/disambiguation_design.md sec 3, C2).
    #
    # kwargs follow bin/continual_chain_experiment.py's "tpc2" precedent
    # (n_hidden=128, learning_rate=0.05). inf_iters is the relaxation budget per
    # timestep and dominates runtime; the reference default is 100.
    "temporal_pc": {
        "class": MultilayerTemporalPCNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            # Paper ratio (Tang et al. 2023, Table 1: latent 480 for 784-d, 630 for
            # 1024-d, i.e. ~0.6x input) applied to the 100-d embedding, and the
            # paper's 100 inference iterations at step 1e-2 (the class default).
            # Set 2026-09-06; was 128 / 50. learning_rate has no paper value to
            # inherit (the paper's span 1e-5..5e-1) and stays 0.05.
            "n_hidden": 60,
            "learning_rate": 0.05,
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 5/5 seeds reached, epochs [39, 34, 48, 41, 44], mean 41.2
            # spatial:  5/5 seeds reached, epochs [32, 42, 12, 35, 14], mean 27.0 (400-cell grid)
            "n_epochs": 83,
            "inf_iters": 100,
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 54}}
    },
    # --- non-EP arm: predictive recirculation (Chen, Zhang, Cameron & Sejnowski 2024)
    # Registered 2026-09-08 as the CONTROLLED PAIR for `temporal_pc`: same family
    # (rate x online, prediction-error driven, tanh latent carried across
    # events, local outer-product updates), differing on exactly two design
    # decisions -- the latent is a forward pass rather than a relaxation, and
    # the error reaches the hidden layer through the forward input weights U
    # rather than W_out^T. It is also the only arm carrying anatomy beyond a
    # motif (EC/DG input, CA3 recurrent, CA1 error units).
    # Ported from the paper's equations (the repo has no license); only the
    # LOCAL rule is ported, not the BPTT-trained headline model.
    # n_hidden / learning_rate copy the tPC entry so the pair differs only where
    # the equations differ. n_epochs sized 2026-09-08 by the docs/model_table.md
    # rule (2x epochs-to-criterion on the canonical material, larger of the two
    # modalities): 7-item list reaches at 7 epochs, the 15-step T-maze at 64
    # (48 fails, 64 passes; ladder 1..64), so 2 x 64 = 128. Fast to criterion on
    # short material, but the 24-item cue_masking list and the length sweep are
    # NOT acquired at 512 epochs -- the paper's own long-sequence caveat.
    "predictive_recirculation": {
        "class": PredictiveRecirculationNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            "n_hidden": 60,
            "learning_rate": 0.05,
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 4/5 seeds reached, epochs [10, 512, 18, 40, 18], mean 21.5
            # spatial:  1/5 seeds reached, epochs [512, 4, 512, 512, 512], mean 4.0 (400-cell grid)
            "n_epochs": 43,
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 8}}
    },
    # --- non-EP arm: theta-phase encode/retrieve (Hasselmo, Bodelon & Wyble 2002) -
    # Registered 2026-09-03 for the zoo capacity run. The arm was implemented and
    # demoed (examples/theta_phase_demo.py) but never reachable from this
    # registry, so no suite could run it.
    #
    # It is the CONTROLLED PAIR for `ahn`: at the paper's optimal phases
    # (phi_EC = phi_LTP = 0, phi_CA3 = pi, X = 1) the theta cycle integral
    # reduces exactly to the delta rule, so this arm takes the *identical* weight
    # step as AsymmetricHopfieldNetwork at the same learning_rate. The one
    # remaining difference is the readout -- AHN rectifies, Hasselmo eq 2.4 is
    # linear -- so `rectify_output=True` makes the pair exact on predictions too.
    # That is the point of running it: same rule, error derived by oscillatory
    # phase separation instead of explicit subtraction. Any divergence from the
    # AHN scorecard is a bug or a readout effect, not a model difference.
    #
    # n_epochs=1 matches ahn: the arm is naturally online and one pass is the
    # honest default. Sections that need more staircase up to it.
    "theta": {
        "class": ThetaPhaseSequenceNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 1/1 seeds reached, epochs [1], mean 1.0
            # spatial:  1/1 seeds reached, epochs [2], mean 2.0 (400-cell grid)
            # Zero-init: seed-invariant, so 1 seed not 5 (sd=0 is a property of
            # the arm, never a finding).
            "n_epochs": 2,
            "learning_rate": 0.1,
            "rectify_output": True,
            "modulation_depth": 1.0,
            "seed": 42
        },
        "modality_kwargs": {"spatial": {"n_epochs": 4}}
    },
    # The only arm clocked by elapsed TIME rather than event ordinal: it declares
    # TemporallyClocked and TimingPredictive (memval/models/capabilities.py), so
    # it is the only one that can answer Serial order's interval questions
    # (docs/capacity_questions.md 5.4, 5.5 -- half that capacity's weight). It
    # also declares OnlineTrainable. kwargs follow examples/dts_esn_*_demo.py;
    # predict_timing=True enables the timing head that 5.5 needs. The reservoir
    # is fixed; the RLS readout keeps updating across passes, and fit_sequence
    # loops `epochs` times (resetting context each pass). n_epochs=1 is the
    # ordinal-clocked baseline exposure, not a statement that more is inert.
    "dts_esn": {
        "class": DTSESNSequenceNetwork,
        "modalities": ["spatial", "symbolic", "online_symbolic"],
        "default_kwargs": {
            "n_units": 400,
            "tau_min": 0.1,
            "tau_max": 20.0,
            # Kept at 0.9 DELIBERATELY (audit 2026-09-06). The paper tunes rho per
            # system -- 1.0 for three of them, 0.1 for the N=400 Lorenz setting we
            # take the size from -- and the class requires rho < 1 for a stable rest
            # state (gaps must decay). n_units=400, log-uniform leak rates and
            # connectivity=0.1 ARE the paper's. RLS readout is ours (paper: ridge).
            "spectral_radius": 0.9,
            "dt": 0.05,
            "predict_timing": True,
            # Budget sized 2026-09-23 by bin/size_budgets.py: 2x the MEAN epochs to
            # criterion over the seeds that REACHED, on one 10-item list / one
            # 10-transition route, eps relative to the route step. Per suite;
            # online inherits symbolic.
            # symbolic: 5/5 seeds reached, epochs [1, 1, 1, 1, 1], mean 1.0
            # spatial:  criterion UNREACHED on every seed (400-cell grid) -> no
            # spatial budget; the suite runs at this budget with
            # criterion_reached=False recorded.
            "n_epochs": 2,
            "seed": 42
        }
    }
}

# Old registry names that still resolve, so pre-rename commands keep working.
MODEL_ALIASES = {"hopfield": "ahn"}


def resolve_model_name(name: str) -> str:
    """argparse ``type=`` for ``--model``: map a legacy alias to its registry key."""
    return MODEL_ALIASES.get(name, name)

# EP (Equilibrium Propagation) models settle iteratively and need a far larger
# epoch budget to reach the convergence threshold than one-shot associators, so
# the symbolic convergence sweep runs them out further by default.
EP_SYMBOLIC_MAX_EPOCHS = 2048
DEFAULT_SYMBOLIC_MAX_EPOCHS = 500

def resolve_symbolic_max_epochs(model_class: Type[Any], cli_max_epochs: Optional[int]) -> int:
    """Resolve the symbolic convergence epoch budget: an explicit CLI value wins;
    otherwise EP models get EP_SYMBOLIC_MAX_EPOCHS and everything else the default."""
    if cli_max_epochs is not None:
        return cli_max_epochs
    if "EqProp" in model_class.__name__:
        return EP_SYMBOLIC_MAX_EPOCHS
    return DEFAULT_SYMBOLIC_MAX_EPOCHS

def _coerce_arg_value(raw: str) -> Any:
    """Best-effort typing of a CLI override value: bool, then int, then float,
    else the original string."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_benchmark_args(spec: str, available) -> Dict[str, Dict[str, Any]]:
    """Parse a ``--benchmark-args`` spec into ``{benchmark: {key: value}}``.

    Format: comma-separated ``benchmark:key=value`` entries. Values are typed via
    ``_coerce_arg_value``. Raises ``ValueError`` on malformed entries or unknown
    benchmark names (validated against ``available``).
    """
    result: Dict[str, Dict[str, Any]] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        bench, sep, rest = token.partition(":")
        if not sep or "=" not in rest:
            raise ValueError(
                f"Malformed --benchmark-args entry {token!r}; "
                f"expected 'benchmark:key=value'."
            )
        key, _, value = rest.partition("=")
        bench, key = bench.strip(), key.strip()
        if not bench or not key:
            raise ValueError(
                f"Malformed --benchmark-args entry {token!r}; "
                f"expected 'benchmark:key=value'."
            )
        result.setdefault(bench, {})[key] = _coerce_arg_value(value.strip())
    unknown = [b for b in result if b not in available]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s) in --benchmark-args: {unknown}. "
            f"Available: {list(available)}"
        )
    return result


def print_summary_table(metrics: Dict[str, Any], title: str):
    """Prints a clean summary table of calculated metrics in the CLI."""
    print("\n" + "=" * 50)
    print(f" {title} METRICS SUMMARY ")
    print("=" * 50)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:<35}: {v:.5f}")
        else:
            print(f"  {k:<35}: {v}")
    print("=" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run MemVal sequence memory benchmark pipelines.")
    parser.add_argument(
        "--model",
        type=resolve_model_name,
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Name of the model to benchmark (required unless --list-benchmarks)."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["spatial", "symbolic", "online_symbolic", "all"],
        help="Benchmark suite to run (default: all)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory where results and plots will be saved (default: ./results)."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of trials for evaluation (default: 30)."
    )
    parser.add_argument(
        "--mrr-threshold",
        type=float,
        default=0.95,
        help="MRR threshold for convergence (default: 0.95)."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max epochs for symbolic convergence sweep. Defaults per model: "
             f"{EP_SYMBOLIC_MAX_EPOCHS} for EP (Equilibrium Propagation) models, "
             f"{DEFAULT_SYMBOLIC_MAX_EPOCHS} otherwise. Pass a value to override."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Global training-epoch override for the fixed-epoch benchmark "
             "sections (spatial and symbolic). Precedence: --benchmark-args "
             "(per section) > --epochs > model default > each section's own "
             "default. Does not affect the epoch sweeps (symbolic "
             "'presentation_duration', governed by --max-epochs, and spatial "
             "'tmaze_reversal')."
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.05,
        help="Standard deviation of the Gaussian ISI filler events streamed between items "
             "in the online suite (default: 0.05). Retrieval cues are clean by protocol; "
             "cue degradation is swept only in noise_invariance / cue_masking -- see docs/probe_protocol.md."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated subset of benchmark sections to run within the "
             "chosen suite (default: all). Requires a single --suite (not 'all') "
             "since section names are suite-specific. Use --list-benchmarks to "
             "see the available names per suite."
    )
    parser.add_argument(
        "--benchmark-args",
        type=str,
        default=None,
        help="Per-benchmark parameter overrides, as a comma-separated list of "
             "'benchmark:key=value' entries (e.g. 'sequence_length:epochs=500'). "
             "Values are parsed as bool/int/float when possible, else kept as "
             "strings. Requires a single --suite (not 'all'). The trainable "
             "sections honour an 'epochs' key."
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="Print the benchmark sections available in each suite and exit."
    )

    args = parser.parse_args()

    if args.list_benchmarks:
        print("Available benchmark sections per suite:")
        for suite, names in SUITE_BENCHMARKS.items():
            print(f"  {suite}:")
            for name in names:
                print(f"    - {name}")
        sys.exit(0)

    if args.model is None:
        parser.error("--model is required (unless --list-benchmarks).")

    # Parse and validate the --benchmarks subset. It is suite-specific, so a
    # single concrete suite must be chosen (not "all").
    selected_benchmarks = None
    if args.benchmarks is not None:
        if args.suite == "all":
            parser.error("--benchmarks requires a single --suite (not 'all'), "
                         "because section names are specific to each suite.")
        selected_benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        available = SUITE_BENCHMARKS[args.suite]
        unknown = [b for b in selected_benchmarks if b not in available]
        if unknown:
            parser.error(f"Unknown {args.suite} benchmark(s): {unknown}. "
                         f"Available: {list(available)}")

    # Parse and validate the --benchmark-args overrides. Like --benchmarks, the
    # section names are suite-specific, so a single concrete suite is required.
    benchmark_args = None
    if args.benchmark_args is not None:
        if args.suite == "all":
            parser.error("--benchmark-args requires a single --suite (not 'all'), "
                         "because section names are specific to each suite.")
        try:
            benchmark_args = parse_benchmark_args(
                args.benchmark_args, SUITE_BENCHMARKS[args.suite]
            )
        except ValueError as e:
            parser.error(str(e))

    model_entry = MODEL_REGISTRY[args.model]
    model_class = model_entry["class"]
    model_kwargs = model_entry["default_kwargs"]
    # Per-modality overrides (input mapping, budget) layered on the defaults.
    _mod_kw = model_entry.get("modality_kwargs", {})
    def kwargs_for(modality):
        return {**model_kwargs, **_mod_kw.get(modality, {})}
    supported_modalities = model_entry["modalities"]

    print(f"Starting benchmark run for model: {args.model}")
    print(f"Supported modalities: {supported_modalities}")
    print(f"Target suite: {args.suite}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of trials: {args.n_trials}")
    print("-" * 50)

    # 1. Spatial Suite Execution
    if args.suite in ["spatial", "all"]:
        if "spatial" not in supported_modalities:
            print(f"\n[SKIP] Modality 'spatial' is not supported by model '{args.model}'.")
            if args.suite == "spatial":
                sys.exit(1)
        else:
            print("\nExecuting Spatial Benchmarks...")
            spatial_metrics = run_spatial_pipeline(
                model_class=model_class,
                model_kwargs=kwargs_for("spatial"),
                output_dir=args.output_dir,
                n_trials=args.n_trials,
                benchmarks=selected_benchmarks,
                benchmark_args=benchmark_args,
                epochs=args.epochs
            )
            print_summary_table(spatial_metrics, f"{args.model.upper()} SPATIAL")

    # 2. Symbolic Suite Execution
    if args.suite in ["symbolic", "all"]:
        if "symbolic" not in supported_modalities:
            print(f"\n[SKIP] Modality 'symbolic' is not supported by model '{args.model}'.")
            if args.suite == "symbolic":
                sys.exit(1)
        else:
            symbolic_max_epochs = resolve_symbolic_max_epochs(model_class, args.max_epochs)
            print(f"\nExecuting Symbolic Benchmarks (max convergence epochs: {symbolic_max_epochs})...")
            symbolic_metrics = run_symbolic_pipeline(
                model_class=model_class,
                model_kwargs=kwargs_for("symbolic"),
                output_dir=args.output_dir,
                n_trials=args.n_trials,
                mrr_threshold=args.mrr_threshold,
                max_epochs=symbolic_max_epochs,
                benchmarks=selected_benchmarks,
                benchmark_args=benchmark_args,
                epochs=args.epochs
            )
            print_summary_table(symbolic_metrics, f"{args.model.upper()} SYMBOLIC")

    # 3. Online Symbolic Suite Execution
    if args.suite in ["online_symbolic", "all"]:
        if "online_symbolic" not in supported_modalities:
            print(f"\n[SKIP] Modality 'online_symbolic' is not supported by model '{args.model}'.")
            if args.suite == "online_symbolic":
                sys.exit(1)
        else:
            print("\nExecuting Online Symbolic Benchmarks...")
            online_metrics = run_online_symbolic_pipeline(
                model_class=model_class,
                model_kwargs=kwargs_for("online_symbolic"),
                output_dir=args.output_dir,
                n_trials=args.n_trials,
                noise_scale=args.noise_scale,
                benchmarks=selected_benchmarks,
                benchmark_args=benchmark_args
            )
            print_summary_table(online_metrics, f"{args.model.upper()} ONLINE SYMBOLIC")

    print("Benchmarking run completed.")

if __name__ == "__main__":
    main()
