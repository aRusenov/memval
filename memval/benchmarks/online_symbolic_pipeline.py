import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Type, Optional

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.models.capabilities import (
    UnsupportedRegime,
    rollout_mode_of,
    supports_online,
)
from memval.benchmarks.ingest import ingest
from memval.benchmarks.probe import resolve_probe, probe_record
from memval.benchmarks._selection import resolve_benchmarks, resolve_benchmark_args
from memval.benchmarks.symbolic_pipeline import (
    measure_recall_associative,
    measure_recall_autoregressive,
    load_vocab,
    mean_recall_rate,
    memory_span,
)

# Canonical, ordered list of benchmark sections in the online-symbolic suite.
# Pass any subset as ``benchmarks=`` to run_online_symbolic_pipeline.
ONLINE_SYMBOLIC_BENCHMARKS = (
    "online_convergence",   # 1. presentations / repetitions sweep
    "isi_tolerance",        # 2. inter-stimulus-interval noise sweep
)


def stream_sequence(model: Any, sequence_encoded: np.ndarray, n_presentations: int = 1, isi_steps: int = 0, noise_scale: float = 0.05):
    """
    Stream a sequence to the model event-by-event via the ``ingest`` seam.
    Handles repetitions (n_presentations) and inserts Gaussian noise ISI steps.

    Raises ``UnsupportedRegime`` (from ``ingest``) if the arm does not declare
    ``OnlineTrainable``. Callers must let that propagate rather than scoring it.
    """
    n_features = sequence_encoded.shape[1]
    for _presentation in range(n_presentations):
        if isi_steps:
            # Interleave noise events between items: build the padded stream and
            # hand it to `ingest` as one fenced presentation, so the ISI steps
            # sit inside the sequence rather than around it.
            stream = []
            for t in range(len(sequence_encoded)):
                stream.append(sequence_encoded[t])
                if t < len(sequence_encoded) - 1:
                    stream.extend(
                        np.random.normal(0.0, noise_scale, n_features) for _ in range(isi_steps)
                    )
            ingest(model, np.asarray(stream), regime="streamed")
        else:
            ingest(model, sequence_encoded, regime="streamed")


def run_online_symbolic_pipeline(
    model_class: Type[Any],
    model_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: str = "./results",
    n_trials: int = 30,
    run_name: Optional[str] = None,
    noise_scale: float = 0.05,
    benchmarks: Optional[List[str]] = None,
    benchmark_args: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Runs the online continuous learning symbolic benchmark suite.

    Sweeps:
    1. Online Convergence (number of sequence presentations / repetitions)
    2. ISI Tolerance (number of Gaussian noise steps injected between items)

    ``benchmarks`` selects a subset of ONLINE_SYMBOLIC_BENCHMARKS (default: all).
    ``benchmark_args`` (per-section parameter overrides) is accepted and validated
    for interface parity with the other suites; these sections sweep their own
    parameter grids, so no override keys are currently honoured.
    """
    model_kwargs = model_kwargs or {}
    selected = resolve_benchmarks(benchmarks, ONLINE_SYMBOLIC_BENCHMARKS, "online_symbolic")
    resolve_benchmark_args(benchmark_args, ONLINE_SYMBOLIC_BENCHMARKS, "online_symbolic")
    model_name = run_name or model_class.__name__
    
    # Establish subdirectories
    run_dir = os.path.join(output_dir, model_name, "online_symbolic")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style="darkgrid")
    vocab = load_vocab()
    
    fruit_words = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']
    fruit_vocab = {w: c for w, c in vocab.items() if c == "fruit"}
    encoder = SymbolicEncoder(fruit_vocab, embedding_dim=100, category_variance=0.2, seed=42)
    decoder = SymbolicDecoder(encoder)
    word_list_encoded = encoder.encode(fruit_words)

    # Initialize results structure
    # The whole suite is the streamed ingestion regime, so an arm that does not
    # declare OnlineTrainable has no result here -- not a bad one. Record that
    # as a status and skip, rather than running the sweeps and writing zeros.
    online = supports_online(model_class)

    results = {
        "metadata": {
            "model_name": model_name,
            "suite": "online_symbolic",
            "n_trials": n_trials,
            "noise_scale": noise_scale,
            "model_kwargs": {k: str(v) for k, v in model_kwargs.items()},
            "status": "ran" if online else "not_applicable",
            "requires_capability": "OnlineTrainable",
            "online_equivalent": bool(getattr(model_class, "online_equivalent", False)),
            # The `recall()` rollout protocol this arm hardcodes. Recorded so a
            # cross-arm comparison can see when two arms were not scored under
            # the same protocol (see docs/rollout_protocol.md).
            "rollout_mode": getattr(
                rollout_mode_of(model_class), "value", None),
            "probe_protocols": {
                sec: probe_record(sec, model_class, n_trials)
                for sec in ONLINE_SYMBOLIC_BENCHMARKS
            },
        },
        "metrics": {},
        "series": {}
    }

    if not online:
        print(f"[{model_name}] NOT APPLICABLE: does not declare OnlineTrainable; "
              f"the streamed regime does not exist for this arm. Writing no metrics.")
        results["metadata"]["status_detail"] = (
            f"{model_class.__name__} does not declare memval.models.capabilities."
            f"OnlineTrainable, so it has no event-by-event ingestion path. The suite "
            f"was skipped; absent metrics here are a capability gap, not a recall "
            f"failure, and must not be scored as 0.0."
        )
        selected = ()

    # ==========================================
    # 1. Online Convergence (Duration Sweep)
    # ==========================================
    if "online_convergence" in selected:
        _pn, _ps = resolve_probe("online_convergence", model_class, n_trials)  # probe protocol
        print(f"[{model_name}] Running Online Convergence Sweep (presentations)...")
        presentation_values = [1, 2, 5, 10, 20]
        conv_mrr_vals = []
        conv_span_vals = []

        for n_pres in presentation_values:
            try:
                # Instantiate a fresh model instance
                if 'encoder' in model_class.__init__.__code__.co_varnames:
                    model = model_class(encoder=encoder, n_features=encoder.embedding_dim, **model_kwargs)
                else:
                    model = model_class(n_features=encoder.embedding_dim, **model_kwargs)

                # Stream sequence online
                stream_sequence(model, word_list_encoded, n_presentations=n_pres, isi_steps=0, noise_scale=noise_scale)

                # Evaluate
                model.reset_context()
                curve_assoc = measure_recall_associative(
                    model, fruit_words, encoder, decoder, n_trials=_pn, noise_scale=_ps
                )
                mrr = mean_recall_rate(curve_assoc)
                conv_mrr_vals.append(mrr)

                curve_auto = measure_recall_autoregressive(
                    model, fruit_words, encoder, decoder, n_trials=_pn, noise_scale=_ps
                )
                span = memory_span(curve_auto)
                conv_span_vals.append(span)
            except UnsupportedRegime:
                # Capability gap, not a recall failure. `None` keeps it out of
                # the metrics; a 0.0 here would be scored as a performance floor.
                conv_mrr_vals.append(None)
                conv_span_vals.append(None)
            except Exception as e:
                print(f"Warning: Online convergence failed for {n_pres} presentations: {e}")
                conv_mrr_vals.append(0.0)
                conv_span_vals.append(0)

        # Save convergence series
        results["series"]["online_convergence"] = {
            "presentations": presentation_values,
            "mrr": conv_mrr_vals,
            "span": conv_span_vals
        }

        # Record scalar metrics. A None sweep point is a capability gap; omit the
        # metric entirely rather than coercing it to a number.
        for key, vals, idx in (
            ("online_mrr_1shot", conv_mrr_vals, 0), ("online_mrr_5pass", conv_mrr_vals, 2),
            ("online_mrr_20pass", conv_mrr_vals, 4), ("online_span_1shot", conv_span_vals, 0),
            ("online_span_5pass", conv_span_vals, 2), ("online_span_20pass", conv_span_vals, 4),
        ):
            if vals[idx] is not None:
                results["metrics"][key] = vals[idx]

        # Plot Convergence Sweep
        fig, ax1 = plt.subplots(figsize=(7, 5))
        color = 'tab:purple'
        ax1.set_xlabel('Presentations (Repetitions)')
        ax1.set_ylabel('Mean Recall Rate (MRR)', color=color)
        ax1.plot(presentation_values, conv_mrr_vals, '-o', color=color, linewidth=2, label='MRR')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Memory Span (threshold = 0.75)', color=color)
        ax2.plot(presentation_values, conv_span_vals, '-s', color=color, linewidth=2, label='Span')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.5, len(fruit_words) + 0.5)

        plt.title(f'Online Learning Convergence - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "online_convergence.png"), dpi=150)
        plt.close()

    # ==========================================
    # 2. ISI Tolerance (Interference Sweep)
    # ==========================================
    if "isi_tolerance" in selected:
        _pn, _ps = resolve_probe("isi_tolerance", model_class, n_trials)  # probe protocol
        print(f"[{model_name}] Running ISI Tolerance Sweep...")
        isi_values = [0, 1, 3, 5, 10]
        isi_mrr_vals = []
        isi_span_vals = []

        for isi_steps in isi_values:
            try:
                # Instantiate a fresh model instance
                if 'encoder' in model_class.__init__.__code__.co_varnames:
                    model = model_class(encoder=encoder, n_features=encoder.embedding_dim, **model_kwargs)
                else:
                    model = model_class(n_features=encoder.embedding_dim, **model_kwargs)

                # Stream sequence with n_presentations=5 and specified ISI steps
                stream_sequence(model, word_list_encoded, n_presentations=5, isi_steps=isi_steps, noise_scale=noise_scale)

                # Evaluate
                model.reset_context()
                curve_assoc = measure_recall_associative(
                    model, fruit_words, encoder, decoder, n_trials=_pn, noise_scale=_ps
                )
                mrr = mean_recall_rate(curve_assoc)
                isi_mrr_vals.append(mrr)

                curve_auto = measure_recall_autoregressive(
                    model, fruit_words, encoder, decoder, n_trials=_pn, noise_scale=_ps
                )
                span = memory_span(curve_auto)
                isi_span_vals.append(span)
            except UnsupportedRegime:
                # Capability gap, not a recall failure. `None` keeps it out of
                # the metrics; a 0.0 here would be scored as a performance floor.
                isi_mrr_vals.append(None)
                isi_span_vals.append(None)
            except Exception as e:
                print(f"Warning: ISI tolerance sweep failed for ISI={isi_steps}: {e}")
                isi_mrr_vals.append(0.0)
                isi_span_vals.append(0)

        # Save ISI series
        results["series"]["isi_tolerance"] = {
            "isi_steps": isi_values,
            "mrr": isi_mrr_vals,
            "span": isi_span_vals
        }

        results["metrics"]["isi_tolerance_mrr_sweep"] = isi_mrr_vals
        results["metrics"]["isi_tolerance_span_sweep"] = isi_span_vals

        # Plot ISI Sweep
        fig, ax1 = plt.subplots(figsize=(7, 5))
        color = 'tab:blue'
        ax1.set_xlabel('ISI Gaussian Noise Steps')
        ax1.set_ylabel('Mean Recall Rate (MRR)', color=color)
        ax1.plot(isi_values, isi_mrr_vals, '-o', color=color, linewidth=2, label='MRR')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Memory Span (threshold = 0.75)', color=color)
        ax2.plot(isi_values, isi_span_vals, '-s', color=color, linewidth=2, label='Span')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.5, len(fruit_words) + 0.5)

        plt.title(f'ISI Gaussian Noise Decay/Tolerance - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "isi_tolerance.png"), dpi=150)
        plt.close()

    # Save metrics.json
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Online symbolic pipeline finished successfully for {model_name}. Results saved to {run_dir}")
    return results["metrics"]


if __name__ == "__main__":
    from memval.models.baselines import AsymmetricHopfieldNetwork

    run_online_symbolic_pipeline(
        model_class=AsymmetricHopfieldNetwork,
        model_kwargs={"learning_rate": 0.1, "activation": "relu", "n_epochs": 2},
        run_name="AsymmetricHopfieldNetwork_online",
    )
