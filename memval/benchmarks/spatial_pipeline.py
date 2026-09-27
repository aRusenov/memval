import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, Type, Optional

from memval.generators.t_maze import TMazeGenerator
from memval.models.capabilities import rollout_mode_of
from memval.benchmarks.symbolic_pipeline import rollout_owner_label
from memval.encoders.spatial import PlaceCellEncoder, decode_floor, relative_eps, route_step
from memval.benchmarks._selection import resolve_benchmarks, resolve_benchmark_args, benchmark_arg, resolve_epochs_default
from memval.benchmarks.exposure import epochs_to_criterion, resolve_exposure

# Canonical, ordered list of benchmark sections in the spatial suite. Pass any
# subset as ``benchmarks=`` to run_spatial_pipeline to skip the rest.
SPATIAL_BENCHMARKS = (
    "tmaze_completion",       # 1. T-maze pattern completion (place-cell)
    "tmaze_disambiguation",   # 2. T-maze odour disambiguation, graded availability
    "tmaze_reversal",         # 3. T-maze reversal (place->reward overwriting)
)

# Removed 2026-09-01 (docs/capacities/capacity_coverage_audit.md D1/D3/D4):
#   spatial_sequence    -- duplicated tmaze_completion as a completion test. Its
#                          non-saturating readouts (coverage, divergence_step)
#                          moved onto tmaze_completion rather than being lost.
#   anchoring_few_shot  -- measures drift correction, a capacity out of scope for
#                          now. Note its anchor period k was never swept, so the
#                          drift question was never actually asked.
#   object_arena        -- never implemented; emitted literal "TODO" strings.

def _get_model_kwargs(base_kwargs: Dict[str, Any], target_epochs: int) -> Dict[str, Any]:
    """Helper to copy model kwargs and override epoch parameters."""
    kwargs = dict(base_kwargs)
    kwargs["n_epochs"] = target_epochs
    kwargs["epochs"] = target_epochs
    return kwargs


def _rollout_recall(
    model: Any,
    true_encoded: np.ndarray,
    start_idx: int,
    recall_len: int,
    anchor_period: int = 0,
    anchor_gain: float = 1.0,
    obs_noise: float = 0.0,
    anchor_dims: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Autoregressive spatial recall with optional periodic sensory anchoring.

    Seeds from the last prompt vector (``true_encoded[start_idx - 1]``) and rolls
    the model forward ``recall_len`` steps, feeding each prediction back as the
    next input. Prediction ``t`` targets position ``start_idx + t``.

    With ``anchor_period > 0`` the loop is closed: every ``anchor_period``-th step
    the fed-back vector is corrected toward the true observation at the position
    just predicted — the "sense where you actually are" step, i.e. EC sensory
    input reconciled by a CA1-style blend::

        current <- pred + anchor_gain * (obs - pred)

    ``anchor_gain=1.0`` is a hard sensory reset (teacher forcing); ``0 < gain < 1``
    is a partial comparator. ``obs_noise`` adds Gaussian sensory noise to the
    observation. ``anchor_dims`` (a slice / index array) restricts the correction
    to a subset of feature dims — the modality-partial case, matching the odour
    clamp used in the disambiguation task. ``anchor_period=0`` recovers pure
    open-loop replay.

    Returns the ``(recall_len, n_features)`` array of predicted encodings (the raw
    predictions, recorded before any anchoring correction is applied).
    """
    n_features = true_encoded.shape[1]

    if not anchor_period:
        # MODEL-OWNED ROLLOUT (the default). No per-step intervention is needed,
        # so hand the whole prompt to the arm and let it roll forward under the
        # protocol it declares in `rollout_mode`. The prefix matters: a
        # prompt_conditioned arm builds its carried state from it, and the
        # memoryless arms reduce it to the last row, reproducing the loop below
        # exactly (tests/test_rollout_mode.py pins that equivalence).
        prompt = true_encoded[:start_idx]
        recalled = np.asarray(model.recall(prompt, length=recall_len))
        out = np.zeros((recall_len, n_features))
        out[: len(recalled)] = recalled[:recall_len]
        return out

    # HARNESS-CONTROLLED ROLLOUT. Periodic sensory anchoring is a per-step
    # intervention the `recall` signature cannot express, so the harness takes
    # the rollout over and imposes observation-space feedback on every arm
    # alike. Anything reported from this path must be labelled as such --
    # see `rollout_owner_label`.
    recon = np.zeros((recall_len, n_features))
    current = true_encoded[start_idx - 1].copy()
    for t in range(recall_len):
        pred = model.predict_next(current)
        recon[t] = pred
        true_idx = start_idx + t
        is_anchor = (
            anchor_period
            and (t + 1) % anchor_period == 0
            and true_idx < len(true_encoded)
        )
        if is_anchor:
            obs = true_encoded[true_idx].copy()
            if obs_noise:
                draw = rng.normal if rng is not None else np.random.normal
                obs = obs + draw(0.0, obs_noise, n_features)
            nxt = pred + anchor_gain * (obs - pred)
            if anchor_dims is not None:
                corrected = pred.copy()
                corrected[anchor_dims] = nxt[anchor_dims]
                nxt = corrected
            current = nxt
        else:
            current = pred
    return recon


# Fidelity metrics reported for every open-loop spatial rollout.
_RECALL_METRICS = ("mse", "rmse", "coverage", "divergence_step")


def _recall_metrics(true_xy: np.ndarray, decoded_xy: np.ndarray, eps: Optional[float] = None) -> Dict[str, float]:
    """Several complementary views of how close a decoded recall path is to GT.

    - ``mse`` / ``rmse``: index-aligned per-step squared / root error. Pointwise
      and timing-locked; bounded by the arena because center-of-mass decoding
      pulls a lost readout toward the middle, so these *saturate* in the failure
      regime (a collapsed recall plateaus rather than exploding).
    - ``coverage``: fraction of GT points that have *some* recon point within
      ``eps`` (nearest-neighbour, timing-free). Matches the "% visual overlap"
      intuition and keeps its full 0..1 range even when recall collapses.
    - ``divergence_step``: first step whose index-aligned error exceeds ``eps``
      (0..recall_len). The continuous-space analogue of the symbolic memory span.
    """
    if eps is None:
        # No absolute default: the T-maze keeps a fixed physical extent, so its
        # step scales as 1/L and a constant eps silently changes difficulty with
        # route length. Derive it from the route instead (memval/encoders/spatial).
        eps = relative_eps(true_xy, k=1.05)
    err = np.linalg.norm(decoded_xy - true_xy, axis=1)
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    over = np.where(err > eps)[0]
    divergence_step = int(over[0]) if len(over) else int(len(err))
    # Nearest-neighbour distance from each GT point to the recon path.
    pairwise = np.linalg.norm(true_xy[:, None, :] - decoded_xy[None, :, :], axis=2)
    coverage = float(np.mean(pairwise.min(axis=1) <= eps))
    return {"mse": mse, "rmse": rmse, "coverage": coverage, "divergence_step": divergence_step}


def probe_next_from_history(model: Any, X: np.ndarray, t: int) -> np.ndarray:
    """One-step prediction of X[t+1], giving the arm the kind of input it trained on.

    A criterion probe must not strip an arm of the mechanism it runs on. An
    ordinal arm's state IS the current vector, so a bare cue loses nothing. A
    stateful arm carries the prefix, and probing it with a bare cue rebuilds it
    from rest -- the DTS-ESN scores 0.36 on the one-step criterion cold and 1.00
    warm, for the same weights. So arms that declare ``StatePrimeable`` are
    primed through the declared, non-learning ``observe`` path with the true
    prefix, then probed with the bare cue, exactly as
    ``symbolic_disambiguation`` already does. Nothing here passes a trajectory
    into ``predict_next``: not every arm accepts one, and the declaration is
    about ``observe``, not about cue shape.
    """
    model.reset_context()
    if t > 0:
        # Base-class no-op on arms without carried state (2026-09-08): the
        # same call on every arm, no capability check at the probe.
        model.observe_sequence(X[:t])
    if hasattr(model, "current_t"):
        model.current_t = t
    return np.asarray(model.predict_next(X[t]), dtype=float)


def run_spatial_pipeline(
    model_class: Type[Any],
    model_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: str = "./results",
    n_trials: int = 30,
    benchmarks: Optional[list] = None,
    benchmark_args: Optional[Dict[str, Dict[str, Any]]] = None,
    epochs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Runs the spatial benchmarking suite against any model class that conforms
    to the HippocampalModel interface.

    Args:
        model_class: The class of the model to instantiate.
        model_kwargs: Dictionary of arguments to pass to the model constructor.
        output_dir: Directory where results and plots will be saved.
        n_trials: Number of evaluation trials (default: 30).
        benchmarks: Subset of SPATIAL_BENCHMARKS to run (default: all).
        benchmark_args: Per-benchmark parameter overrides, keyed by section name
            (e.g. ``{"tmaze_completion": {"epochs": 200}}``). The trainable
            sections honour an ``epochs`` key; see each section for its default.
        epochs: Global training-epoch override for the fixed-epoch sections. The
            resolved value follows the precedence ``--benchmark-args`` (per
            section) > this ``epochs`` > model default > the section's own
            default.

    Returns:
        Dict: Dictionary of calculated metrics.
    """
    model_kwargs = model_kwargs or {}
    selected = resolve_benchmarks(benchmarks, SPATIAL_BENCHMARKS, "spatial")
    benchmark_args = resolve_benchmark_args(benchmark_args, SPATIAL_BENCHMARKS, "spatial")

    def exposure_policy(section: str, fallback: int, force_criterion: bool = False) -> Dict[str, Any]:
        """Control sections take the registry budget (model_kwargs["n_epochs"]);
        result sections stay on the criterion ladder. Precedence in resolve_exposure."""
        return resolve_exposure(benchmark_args, section, benchmark_arg, fallback,
                                global_epochs=epochs,
                                model_fixed_epochs=(model_kwargs or {}).get("n_epochs"),
                                force_criterion=force_criterion)

    def train_at_exposure(policy, build, fit, score):
        """Train under `policy`; returns (model, {epochs, reached, mode, score})."""
        if policy["mode"] == "fixed":
            m = build(policy["epochs"])
            fit(m, policy["epochs"])
            sc = float(score(m))
            return m, {"epochs": policy["epochs"], "reached": sc >= policy["criterion"],
                       "mode": "fixed", "source": policy.get("source"), "score": sc}
        res = epochs_to_criterion(
            lambda: build(policy["max_epochs"]), fit, score,
            criterion=policy["criterion"], max_epochs=policy["max_epochs"])
        return res["model"], {"epochs": res["epochs"], "reached": res["reached"],
                              "mode": "criterion", "source": policy.get("source"), "score": res["score"]}

    def section_epochs(section: str, fallback: int) -> int:
        """Resolve a section's epoch count: benchmark_args > --epochs > model
        default > the section fallback."""
        return int(benchmark_arg(
            benchmark_args, section, "epochs",
            resolve_epochs_default(epochs, model_kwargs, fallback)
        ))

    # Establish subdirectories
    model_name = model_class.__name__
    run_dir = os.path.join(output_dir, model_name, "spatial")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style="darkgrid")
    
    # Structure metrics output
    results = {
        "metadata": {
            "model_name": model_name,
            "suite": "spatial",
            "n_trials": n_trials,
            "model_kwargs": {k: str(v) for k, v in model_kwargs.items()},
            # The `recall()` rollout protocol this arm hardcodes. Recorded so a
            # cross-arm comparison can see when two arms were not scored under
            # the same protocol (see docs/sections/rollout_protocol.md).
            "rollout_mode": getattr(
                rollout_mode_of(model_class), "value", None),
        },
        "metrics": {},
        "series": {}
    }

    # ==========================================
    # 1. T-Maze Pattern Completion (Raw vs PC)
    # ==========================================
    if "tmaze_completion" in selected:
        tmaze_policy = exposure_policy("tmaze_completion", 100)
        # Tolerance for coverage / divergence_step, in arena units. Stated as a
        # parameter because both metrics are defined relative to it.
        # Tolerance is RELATIVE to the route's step (memval/encoders/spatial.py):
        # eps was absolute at 0.15 while the T-maze's physical extent is fixed, so
        # the step scales as 1/L and a change of route length silently changed the
        # criterion's difficulty. `eps_k` (default 1.05) reproduces the historical
        # 0.15 at the shipped 15-point route. Pass `eps=` to pin an absolute value
        # for reproducing an old number.
        _eps_abs = benchmark_arg(benchmark_args, "tmaze_completion", "eps", None)
        _eps_k = float(benchmark_arg(benchmark_args, "tmaze_completion", "eps_k", 1.05))
        print(f"Running T-Maze benchmarks... "
              f"(exposure: {tmaze_policy['mode']}, eps={'abs ' + str(_eps_abs) if _eps_abs is not None else f'{_eps_k}x route step'})")
        tmaze_gen = TMazeGenerator(seed=42)
        seq_len = 15
        prompt_len = int(seq_len * 0.3) # 4 steps
        recall_len = seq_len - prompt_len # 21 steps

        tmaze_trajectories = tmaze_gen.generate(n_sequences=1, sequence_length=seq_len, turn_direction='right')
        tmaze_seq = tmaze_trajectories[0]
        prompt = tmaze_seq[:prompt_len]
        true_suffix = tmaze_seq[prompt_len:]

        # --- 1. Place Cell Encoded T-Maze ---
        try:
            # Place-cell grid FIXED PER SUITE at 20x20 = 400 cells (2026-09-23), so
            # every spatial section shares one encoder geometry; the sections that
            # need a cue channel add 20 dims on top (disambiguation odour, reversal
            # reward), giving 400 here and 420 there. Was 10x10 = 100.
            #
            # Measured on this route when the change was made: the 400-cell grid has a
            # WORSE clean-decode floor (0.116 = 77% of eps, vs 0.072 = 48% at 100
            # cells, because field width is sigma_scale x route step and is not tied
            # to the grid) and yet the arms score BETTER on it -- hopfield 1.00/1.00,
            # theta 1.00/1.00, tPC 0.93 -> 1.00, EP 0.86 -> 0.93. The redundancy of
            # the finer code makes the transition easier to reproduce than the coarser
            # floor makes it to decode. So the floor bounds how precisely a position
            # can be READ, not how well an arm scores; report it, do not tune on it.
            _tmc_cells = int(benchmark_arg(benchmark_args, "tmaze_completion",
                                           "n_cells_per_dim", 20))
            encoder = PlaceCellEncoder(seed=42, n_cells_per_dim=_tmc_cells, sigma_scale=1.0)
            tmaze_seq_encoded = encoder.encode(tmaze_seq)
            # eps needs the encoder (for the reported floor) and the route.
            recall_eps = (float(_eps_abs) if _eps_abs is not None
                          else relative_eps(tmaze_seq, k=_eps_k))
            _floor = decode_floor(encoder, tmaze_seq)
            prompt_encoded = tmaze_seq_encoded[:prompt_len]

            def _mk_tm(ep):
                kw = _get_model_kwargs(model_kwargs, ep)
                # 1024 is this section's width for arms that do not declare one
                # (place-cell input is 100-dim and the route is long, so a
                # narrow hidden layer under-fits it). setdefault, not a literal
                # kwarg: an arm whose MODEL_REGISTRY entry declares n_hidden has
                # made an explicit per-arm choice, and passing both raised
                # TypeError("got multiple values for keyword argument
                # 'n_hidden'") -- which the surrounding try/except turned into a
                # silent section-wide nan. Arms without the kwarg are unaffected.
                kw.setdefault("n_hidden", 1024)
                return model_class(n_features=encoder.n_cells, **kw)

            def _fit_tm(m, ep, _x=tmaze_seq_encoded):
                if hasattr(m, "n_epochs"):
                    m.n_epochs = ep
                m.fit_sequence(_x, epochs=ep)

            # The criterion read-out must sit UPSTREAM of the scored one, or the
            # section guarantees its own result: training until coverage >= 0.95
            # and then reporting coverage is circular. So exposure is settled on
            # teacher-forced one-step prediction along the route ("are the
            # transitions learned"), and coverage / divergence_step are then
            # measured open-loop from the cue point. Same split as the symbolic
            # sections: criterion on cued, score on rollout.
            def _score_tm(m, _x=tmaze_seq_encoded):
                hits = 0
                for t in range(len(_x) - 1):
                    pred = probe_next_from_history(m, _x, t)
                    xy = encoder.decode(np.asarray([pred]))[0]
                    hits += float(np.linalg.norm(xy - tmaze_seq[t + 1]) <= recall_eps)
                return hits / max(len(_x) - 1, 1)

            model_pc, tmaze_exp = train_at_exposure(tmaze_policy, _mk_tm, _fit_tm, _score_tm)
            model_pc.reset_context()

            # Autoregressive recall (open loop; anchor_period=0 = pure replay)
            recon_encoded = _rollout_recall(
                model_pc, tmaze_seq_encoded, start_idx=prompt_len, recall_len=recall_len
            )
            reconstructed_pc = encoder.decode(recon_encoded)
            # MSE alone saturates in the failure regime (see _recall_metrics), so
            # the non-saturating companions are reported beside it. They moved
            # here from the removed spatial_sequence section rather than being
            # lost with it: `coverage` keeps its full 0..1 range when recall
            # collapses, and `divergence_step` is the continuous-space analogue
            # of memory span -- the error-propagation readout the completion
            # prose promises.
            results["metrics"]["tmaze_pc_epochs_to_criterion"] = float(tmaze_exp["epochs"])
            results["metrics"]["tmaze_pc_criterion_reached"] = bool(tmaze_exp["reached"])
            results["metrics"]["tmaze_pc_exposure_mode"] = tmaze_policy["mode"]
            results["metrics"]["tmaze_pc_exposure_source"] = tmaze_policy.get("source")
            tm_metrics = _recall_metrics(true_suffix, reconstructed_pc, eps=recall_eps)
            pc_mse = tm_metrics["mse"]
        except Exception as e:
            print(f"Warning: Place Cell T-Maze failed for {model_name}: {e}")
            tm_metrics = {m: float('nan') for m in _RECALL_METRICS}
            pc_mse = float('nan')
            reconstructed_pc = np.zeros((recall_len, 2))

        results["metrics"]["tmaze_pc_mse"] = pc_mse
        results["metrics"]["tmaze_pc_coverage"] = tm_metrics["coverage"]
        results["metrics"]["tmaze_pc_divergence_step"] = tm_metrics["divergence_step"]
        results["metrics"]["tmaze_pc_recall_len"] = int(recall_len)
        results["metrics"]["tmaze_pc_eps"] = recall_eps
        # Provenance + the decoder's own error floor, so a reader can see how much of
        # the tolerance was spent before the model contributed anything.
        results["metrics"]["tmaze_pc_n_cells"] = int(encoder.n_cells)
        results["metrics"]["tmaze_pc_eps_mode"] = "absolute" if _eps_abs is not None else "relative"
        results["metrics"]["tmaze_pc_eps_k"] = _eps_k
        results["metrics"]["tmaze_pc_route_step"] = route_step(tmaze_seq)
        results["metrics"]["tmaze_pc_decode_floor_max"] = float(np.max(_floor))
        results["metrics"]["tmaze_pc_decode_floor_frac_of_eps"] = float(np.max(_floor) / recall_eps) if recall_eps else float("nan")

        results["series"]["tmaze_true"] = tmaze_seq.tolist()
        results["series"]["tmaze_prompt"] = prompt.tolist()
        results["series"]["tmaze_reconstructed_pc"] = reconstructed_pc.tolist()

        # Save T-Maze Plot
        plt.figure(figsize=(6, 6))
        plt.plot(tmaze_seq[:, 0], tmaze_seq[:, 1], '--', color='gray', alpha=0.5, label='True T-Maze Path')
        plt.plot(prompt[:, 0], prompt[:, 1], '-o', color='green', label='Prompt Cue (First 30%)')
        if not np.isnan(pc_mse):
            plt.plot(reconstructed_pc[:, 0], reconstructed_pc[:, 1], '-b^', alpha=0.8,
                     label=f'PC Model (MSE {pc_mse:.4f}, cov {tm_metrics["coverage"]:.2f}, '
                           f'div step {tm_metrics["divergence_step"]}/{recall_len})')
        plt.title('T-Maze Pattern Completion')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "tmaze_completion.png"), dpi=150)
        plt.close()

    # ==========================================
    # 2. T-Maze Odour Disambiguation (graded odour availability)
    # ==========================================
    if "tmaze_disambiguation" in selected:
        disamb_policy = exposure_policy("tmaze_disambiguation", 100)
        disamb_epochs = (disamb_policy["epochs"] if disamb_policy["mode"] == "fixed"
                         else None)
        disamb_exp = None
        print(f"Running T-Maze Odour Disambiguation... "
              f"(exposure: {disamb_policy['mode']})")
        from memval.generators.tmaze_disambiguation import TMazeDisambiguationGenerator
        from memval.benchmarks.spatial_disambiguation import SpatialDisambiguationBenchmark, consolidate_if_supported

        # Place-cell grid: 400 (20x20). Briefly reduced to 100 on 2026-09-23 and
        # REVERTED the same day, to keep the feature width FIXED PER SUITE rather
        # than varying per section (tmaze_reversal needs 400 for its finer-stepping
        # route, and a suite-uniform width is worth more than a per-section optimum).
        # For the record, at this route's step the 100-cell grid decodes BETTER
        # (clean-decode floor 48% of eps vs 77% at 400) because field width is
        # sigma_scale x route step and is not tied to the grid; branch accuracy was
        # 1.000 either way and margin 0.914 -> 0.892. See docs/models/model_table.md and
        # bin/size_budgets.py. Pass `n_place_cells=100` to get the tighter decoder.
        _disamb_npc = int(benchmark_arg(benchmark_args, "tmaze_disambiguation",
                                        "n_place_cells", 400))
        _disamb_nod = int(benchmark_arg(benchmark_args, "tmaze_disambiguation",
                                        "n_odour_dims", 20))
        tmaze_disamb_gen = TMazeDisambiguationGenerator(seed=42)
        tmaze_route_pair = tmaze_disamb_gen.generate(
            sequence_length=15,
            stem_fraction=0.50,
            n_place_cells=_disamb_npc,
            n_odour_dims=_disamb_nod,
            seed=42,
            sigma_scale=1.0,
            odour_on_arms=True,      # odour persists through the arms, not just the stem
            balance_modalities=True, # place/odour matched to a common per-timestep norm
            odour_scale=2.0          # odour weighted x2 so the cue is decisive for a linear map
        )

        try:
            tmaze_disamb_bench = SpatialDisambiguationBenchmark()
            tmaze_disamb_kwargs = dict(model_kwargs)
            tmaze_disamb_kwargs["n_features"] = _disamb_npc + _disamb_nod

            if disamb_epochs is None:
                inA = tmaze_route_pair["input_A"]

                def _mk_dis(ep, _n=inA.shape[1]):
                    kw = _get_model_kwargs(model_kwargs, ep)
                    return model_class(n_features=_n, **kw)

                def _fit_dis(m, ep, _x=inA):
                    if hasattr(m, "n_epochs"):
                        m.n_epochs = ep
                    m.fit_sequence(_x, epochs=ep)

                # Upstream of the scored read-out: one-step prediction fidelity on
                # the route, not branch accuracy (which is what the section reports
                # and so cannot also settle its own exposure).
                def _score_dis(m, _x=inA):
                    num = den = 0.0
                    for t in range(len(_x) - 1):
                        p = probe_next_from_history(m, _x, t)
                        tgt = _x[t + 1]
                        d = np.linalg.norm(p) * np.linalg.norm(tgt)
                        num += float(p @ tgt / d) if d > 1e-12 else 0.0
                        den += 1.0
                    return num / den if den else 0.0

                _, disamb_exp = train_at_exposure(disamb_policy, _mk_dis, _fit_dis,
                                                  _score_dis)
                disamb_epochs = int(disamb_exp["epochs"])
                print(f"  [tmaze_disambiguation] exposure to criterion: "
                      f"{disamb_epochs} epochs (reached={disamb_exp['reached']})")

            for cond in ["full", "mec_only"]:
                metrics = tmaze_disamb_bench.evaluate(
                    model=model_class(**tmaze_disamb_kwargs),
                    route_pair=tmaze_route_pair,
                    condition=cond,
                    n_trials=n_trials,
                    fit_epochs=disamb_epochs
                )
                results["metrics"][f"tmaze_disamb_{cond}_branch_acc"] = metrics["branch_accuracy"]
                results["metrics"][f"tmaze_disamb_{cond}_confusion"]  = metrics["confusion_rate"]
            # Encoder provenance, written UNCONDITIONALLY: the exposure keys below
            # only exist on the criterion path, and these must land on the fixed path too.
            results["metrics"]["tmaze_disamb_n_place_cells"] = _disamb_npc
            results["metrics"]["tmaze_disamb_n_features"] = _disamb_npc + _disamb_nod
            if disamb_exp is not None:
                results["metrics"]["tmaze_disamb_epochs_to_criterion"] = float(disamb_exp["epochs"])
                results["metrics"]["tmaze_disamb_criterion_reached"] = bool(disamb_exp["reached"])

            # Plot a single trial reconstruction from the T-maze example
            encoder = tmaze_route_pair["encoder"]
            n_pc = encoder.n_cells
            n_od = tmaze_route_pair["input_A"].shape[-1] - n_pc
            traj_A = encoder.decode(tmaze_route_pair["input_A"][:, :n_pc])
            traj_B = encoder.decode(tmaze_route_pair["input_B"][:, :n_pc])
            se = tmaze_route_pair["shared_end"]

            # Train a fresh model specifically for the plot trial
            plot_model = model_class(**tmaze_disamb_kwargs)
            plot_model.reset_context()
            plot_model.fit_sequence(tmaze_route_pair["input_A"], epochs=disamb_epochs)
            consolidate_if_supported(plot_model, tmaze_route_pair["input_A"])
            plot_model.fit_sequence(tmaze_route_pair["input_B"], epochs=disamb_epochs)
            consolidate_if_supported(plot_model, tmaze_route_pair["input_B"])

            def get_single_recall(input_seq):
                # Same protocol as the scorer (2026-09-08): stem via observe()
                # (no-op on stateless arms), then the arm's own recall() from
                # the cue [place_{se-1}, odour], nothing clamped; clip decode.
                plot_model.reset_context()
                if se > 1:
                    plot_model.observe_sequence(input_seq[: se - 1])
                L = len(input_seq)
                rec = np.asarray(plot_model.recall(input_seq[se - 1], length=L - se), dtype=float)
                return encoder.decode(np.clip(rec[:, :n_pc], 0.0, None))

            rec_A = get_single_recall(tmaze_route_pair["input_A"])
            rec_B = get_single_recall(tmaze_route_pair["input_B"])

            plt.figure(figsize=(6, 6))
            plt.plot(traj_A[:, 0], traj_A[:, 1], '--', color='blue', alpha=0.3, label='Target Left')
            plt.plot(traj_B[:, 0], traj_B[:, 1], '--', color='red', alpha=0.3, label='Target Right')
            plt.plot(traj_A[:se, 0], traj_A[:se, 1], '-', color='green', linewidth=3, label='Stem Cue (with Odour)')
            plt.plot(rec_A[:, 0], rec_A[:, 1], '-b^', label='Recall Left')
            plt.plot(rec_B[:, 0], rec_B[:, 1], '-rx', label='Recall Right')
            plt.title('T-Maze Odour Disambiguation')
            plt.xlim(-1.1, 1.1)
            plt.ylim(-0.1, 1.1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "tmaze_disambiguation.png"), dpi=150)
            plt.close()

        except Exception as e:
            print(f"Warning: T-Maze Odour Disambiguation example failed: {e}")
            for cond in ["full", "mec_only"]:
                results["metrics"][f"tmaze_disamb_{cond}_branch_acc"] = float('nan')
                results["metrics"][f"tmaze_disamb_{cond}_confusion"]  = float('nan')

        # ------------------------------------------------------------------
        # 2b. Graded odour availability (axis A of docs/sections/disambiguation_design.md)
        # ------------------------------------------------------------------
        # The block above is the *fully-cued* corner: the odour is present on
        # every step including the arms, so the probe re-supplies the
        # discriminating cue at the decision itself. This sweep places that
        # corner on a continuum by withdrawing the odour progressively earlier
        # along the shared corridor, leaving
        #
        #     delay = shared_end - zone_end
        #
        # steps over which no discriminator is available. The `concurrent` row
        # keeps the odour on the arms and is the reference; every `withdrawn`
        # row removes it, so the branch must be chosen from what the model
        # carried out of the zone. Rollout (2026-09-08): the arm's own recall()
        # from the zone-end cue, nothing clamped -- the withdrawn material has
        # zero odour after the zone, so the arm predicts ~0 there itself; see
        # SpatialDisambiguationBenchmark._run_trials.
        #
        # READ THIS BEFORE QUOTING ANY ROW -- but the protocol block is now
        # lifted. The shared zone is ingested through `observe` (the non-learning
        # state update of memval.models.capabilities.StatePrimeable) before the
        # rollout, so history IS delivered to any arm that declares it can carry
        # state. Each row records `state_primed`:
        #
        #   state_primed=True  -- the arm received the stem and had to use it.
        #                         A withdrawn-row floor is now a MODEL result.
        #   state_primed=False -- the arm declares no carried state. observe()
        #                         is still called (2026-09-08: base-class no-op,
        #                         no capability check), so the branch decision
        #                         rests on the last cued step alone. A floor
        #                         here is a statement about the arm, not the
        #                         harness, but it is not comparable with a
        #                         primed row.
        #
        # Read branch_margin alongside branch_accuracy: at chance, accuracy alone
        # cannot distinguish "carried nothing" from "carried a weak bias".
        try:
            from memval.generators.bifurcating_route import BifurcatingRouteGenerator

            grad_total = int(benchmark_arg(benchmark_args, "tmaze_disambiguation",
                                           "graded_total_length", 16))
            grad_shared = float(benchmark_arg(benchmark_args, "tmaze_disambiguation",
                                              "graded_shared_fraction", 0.5))
            # Odour always starts at stem entry (zone_offset=0) and is withdrawn
            # progressively earlier; the last rung is a single onset step.
            zone_fractions = [1.0, 0.75, 0.5, 0.25, 0.125]
            # Same 10x10 grid as the block above (see the note there).
            n_pc_g, n_od_g = _disamb_npc, _disamb_nod

            grad_gen = BifurcatingRouteGenerator(seed=42)
            grad_bench = SpatialDisambiguationBenchmark()
            grad_kwargs = dict(model_kwargs)
            grad_kwargs["n_features"] = n_pc_g + n_od_g

            def _graded_row(zone_fraction, on_arms):
                route = grad_gen.generate(
                    total_length=grad_total, shared_fraction=grad_shared,
                    shared_position=0.0, zone_fraction=zone_fraction, zone_offset=0.0,
                    n_place_cells=n_pc_g, n_encounter_dims=n_od_g,
                    balance_modalities=True, odour_scale=2.0,
                    odour_on_suffix=on_arms, seed=42,
                )
                out = grad_bench.evaluate(
                    model=model_class(**grad_kwargs), route_pair=route,
                    condition="full", n_trials=n_trials, fit_epochs=disamb_epochs,
                )
                out["zone_fraction"] = zone_fraction
                out["availability"] = "concurrent" if on_arms else "withdrawn"
                return out

            grad_records = [_graded_row(1.0, True)]
            grad_records += [_graded_row(zf, False) for zf in zone_fractions]

            for rec in grad_records:
                tag = ("concurrent" if rec["availability"] == "concurrent"
                       else f"delay{rec['delay']}")
                for key in ("branch_accuracy", "divergence_accuracy",
                            "divergence_margin", "branch_margin",
                            "shared_stretch_error"):
                    results["metrics"][f"tmaze_disamb_graded_{tag}_{key}"] = rec[key]
            results["series"]["tmaze_disamb_graded"] = grad_records

            # ------------------------------------------------------------------
            # 2c. Guidance withdrawal (2026-09-08). Material = the concurrent
            # row (odour through stem and arms, both routes, blocked fit) and
            # is the SAME on every rung; only the probe moves. The stem is
            # delivered to every arm through observe() (base-class no-op on
            # arms without carried state) up to cue point k, then the arm's
            # own recall() rolls forward from [place_k, odour] with nothing
            # clamped. Recall starts d = shared_end - k steps before the fork:
            # "guided this far, does it recover the correct arm". Multi-seed
            # (model seed), band = +-1 s.d. over seeds.
            # ------------------------------------------------------------------
            guided = None
            try:
                guided_seeds = int(benchmark_arg(benchmark_args, "tmaze_disambiguation",
                                                 "guided_seeds", 5))
                guided_route = grad_gen.generate(
                    total_length=grad_total, shared_fraction=grad_shared,
                    shared_position=0.0, zone_fraction=1.0, zone_offset=0.0,
                    n_place_cells=n_pc_g, n_encounter_dims=n_od_g,
                    balance_modalities=True, odour_scale=2.0,
                    odour_on_suffix=True, seed=42,
                )
                _seeded = "seed" in grad_kwargs
                _base_seed = int(grad_kwargs.get("seed", 42))

                def _guided_factory(s):
                    kw = dict(grad_kwargs)
                    if _seeded:
                        kw["seed"] = int(s)
                    return model_class(**kw)

                guided = grad_bench.guidance_sweep(
                    _guided_factory, guided_route, fit_epochs=disamb_epochs,
                    seeds=[_base_seed + i for i in range(guided_seeds if _seeded else 1)],
                )
                for rec in guided["records"]:
                    d = rec["recall_starts_before_fork"]
                    for key in ("arm_accuracy", "positional_error", "stem_error",
                                "divergence_margin", "branch_margin"):
                        results["metrics"][f"tmaze_disamb_guided_d{d}_{key}"] = rec[key]
                    for key in ("arm_accuracy", "positional_error", "divergence_margin"):
                        results["metrics"][f"tmaze_disamb_guided_d{d}_{key}_sd"] = rec[key + "_sd"]
                results["metrics"]["tmaze_disamb_guided_n_seeds"] = guided["n_seeds"]
                results["metrics"]["tmaze_disamb_guided_rollout_mode"] = guided["rollout_mode"]
                results["series"]["tmaze_disamb_guided"] = guided["records"]
                print(f"  [tmaze_disambiguation] guidance sweep: {guided['n_seeds']} seed(s), "
                      + ", ".join(f"d{r['recall_starts_before_fork']}:acc {r['arm_accuracy']:.2f}"
                                  for r in guided["records"]))
            except Exception as e:
                print(f"Warning: Guidance-withdrawal sweep failed: {e}")
                results["series"]["tmaze_disamb_guided"] = []

            withdrawn = [r for r in grad_records if r["availability"] == "withdrawn"]
            concurrent = grad_records[0]

            n_rows = 2 if guided is not None and guided["records"] else 1
            fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.9 * n_rows), squeeze=False)
            # The concurrent row (odour on the arms too) is the first point of
            # the same curve, not a reference line: every withdrawn rung has
            # already lost the odour on the arms, so the drop from "arms on" to
            # delay 0 is the first withdrawal step and belongs on the line.
            top_rows = [concurrent] + withdrawn
            top_x = [-1] + [r["delay"] for r in withdrawn]
            top_ticklabels = ["arms\non"] + [str(r["delay"]) for r in withdrawn]
            axes[0][0].plot(top_x, [r["branch_accuracy"] for r in top_rows], '-o',
                            label='branch accuracy (suffix mean)')
            axes[0][0].plot(top_x, [r["divergence_accuracy"] for r in top_rows], '-s',
                            label='divergence accuracy (branch step)')
            axes[0][0].axhline(0.5, color='red', linestyle=':', alpha=0.7, label='chance')
            axes[0][0].set_ylim(-0.05, 1.05)
            axes[0][0].set_ylabel('accuracy')
            axes[0][0].set_title('Branch choice vs cue-free delay\n(odour withdrawn during training + probe)')

            axes[0][1].plot(top_x, [r["divergence_margin"] for r in top_rows], '-s',
                            color='tab:purple', label='divergence margin')
            axes[0][1].axhline(0.0, color='red', linestyle=':', alpha=0.7, label='no preference')
            axes[0][1].set_ylabel('(d_wrong - d_correct) / arm separation')
            axes[0][1].set_title('Commitment margin vs cue-free delay\n(odour withdrawn during training + probe)')

            for ax in axes[0]:
                ax.set_xticks(top_x)
                ax.set_xticklabels(top_ticklabels)
                ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('odour on the arms  |  delay = shared steps with no odour available')
                ax.legend(fontsize=8)

            if n_rows == 2:
                gr = guided["records"]
                ds = [r["recall_starts_before_fork"] for r in gr]

                def _band(ax, key, color, label, style='-o'):
                    m = np.array([r[key] for r in gr], dtype=float)
                    sd = np.nan_to_num(np.array([r[key + "_sd"] for r in gr], dtype=float))
                    ax.plot(ds, m, style, color=color, label=label)
                    ax.fill_between(ds, m - sd, m + sd, color=color, alpha=0.15)

                ax = axes[1][0]
                _band(ax, "divergence_margin", "tab:purple", "divergence margin (fork step)")
                _band(ax, "branch_margin", "tab:gray", "branch margin (arm mean)", '-s')
                ax.axhline(0.0, color='red', linestyle=':', alpha=0.7, label='no preference')
                ax.set_ylabel('(d_wrong - d_correct) / arm separation')
                ax.set_title('Commitment vs guidance withdrawn\n(during probe alone: odour on in training, recall unclamped)')
                ax.legend(fontsize=8)

                ax = axes[1][1]
                _band(ax, "arm_accuracy", "tab:blue", "arm accuracy (nearest arm, any position)")
                ax.axhline(0.5, color='red', linestyle=':', alpha=0.7, label='chance')
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel('arm accuracy')
                ax2 = ax.twinx()
                _band(ax2, "positional_error", "tab:orange", "positional error along the arm", '-^')
                ax2.set_ylabel('mean decoded distance to target')
                ax2.set_ylim(bottom=0.0)
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right')
                ax.set_title('Arm choice and position vs guidance withdrawn\n(during probe alone)')

                for ax in axes[1]:
                    ax.set_xlabel('d = recall starts d steps before the fork (guided up to there)')

            # The title must track `state_primed`, not assert a floor: for an
            # arm that declares StatePrimeable the withdrawn rows ARE a model
            # result, and a figure claiming otherwise is exactly what gets
            # quoted back wrongly. `_primed` is computed just below; hoist it.
            _primed = any(r.get("state_primed") for r in grad_records)
            _protocol_note = ('carried state declared; withdrawn rows are a MODEL result'
                              if _primed else
                              'no carried state declared; observe() is a no-op, '
                              'withdrawn rows are a floor of the arm')
            fig.suptitle(f'Graded odour availability - {model_name}\n({_protocol_note})',
                         fontsize=11)
            # Both rows are the arm's own recall() from a cue, nothing clamped;
            # they differ in the material (odour withdrawn vs on) and in where
            # the cue sits (zone end vs cue point k).
            _rm = getattr(rollout_mode_of(model_class), "value", None)
            _foot = (f"top: {rollout_owner_label('model')} (rollout_mode={_rm}) from the "
                     f"zone-end cue; odour withdrawn from the material, not clamped; "
                     f"state_primed={_primed}")
            if n_rows == 2:
                _foot += (f"\nbottom: {rollout_owner_label('model')} "
                          f"(rollout_mode={guided['rollout_mode']}) from [place_k, odour]; "
                          f"stem via observe() (no-op on stateless arms); "
                          f"{guided['n_seeds']} seed(s), band = +-1 s.d.")
            fig.text(0.5, 0.005, _foot, ha="center", fontsize=8, style="italic")
            fig.tight_layout(rect=(0, 0.03 if n_rows == 2 else 0.02, 1, 0.95))
            fig.savefig(os.path.join(plots_dir, "tmaze_disambiguation_graded.png"), dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Graded odour-availability sweep failed: {e}")
            results["series"]["tmaze_disamb_graded"] = []

    # ==========================================
    # 3. T-Maze Reversal (overwriting an invalid association)
    # ==========================================
    # The complement of the retention matrix: how fast does the model overwrite a
    # place->reward association that has become invalid? Runs the protocol twice,
    # with and without an explicit extinction stage, so the value of extinction is
    # measured rather than assumed. See docs/sections/spatial_reversal_design.md.
    if "tmaze_reversal" in selected:
        from memval.generators.t_maze_reversal import TMazeReversalGenerator
        from memval.benchmarks.spatial_reversal import SpatialReversalBenchmark

        # This section's training unit is a *presentation*, not an epoch: one
        # trial is one fit_sequence call, and a stage runs max_trials of them.
        # So it deliberately does not inherit the model's own n_epochs default
        # (typically 100) the way the other sections do -- that would be 100
        # epochs *per trial*. --epochs and --benchmark-args still apply.
        # The default (3) is set so the *reversal* stage has dynamic range for a
        # delta-rule arm at lr≈0.05: crank it up and every arm reverses on trial
        # 1, hiding the measurement. It interacts with learning rate, so state it
        # on every figure and prefer comparing arms at matched total exposure.
        rev_epochs = int(benchmark_arg(
            benchmark_args, "tmaze_reversal", "epochs",
            epochs if epochs is not None else 3
        ))
        max_trials = int(benchmark_arg(benchmark_args, "tmaze_reversal", "max_trials", 12))
        # reward_gain is the teaching-signal strength and is a choice, not a
        # default: sweep it and report the plateau. 0.0 gives the dead-channel
        # control (block present, always zero), isolating the effect of the added
        # dimensionality from the effect of the reward signal.
        dead_channel = bool(int(benchmark_arg(
            benchmark_args, "tmaze_reversal", "control_dead_channel", 0)))
        reward_gain = 0.0 if dead_channel else float(
            benchmark_arg(benchmark_args, "tmaze_reversal", "reward_gain", 1.0))
        measure_savings = bool(int(benchmark_arg(
            benchmark_args, "tmaze_reversal", "measure_savings", 0)))
        print(f"Running T-Maze Reversal... "
              f"(epochs/trial={rev_epochs}, max_trials={max_trials}, reward_gain={reward_gain})")

        n_rev_place, n_rev_reward = 400, 20
        rev_gen = TMazeReversalGenerator(seed=42)
        rev_task = rev_gen.generate(
            sequence_length=16,
            stem_fraction=0.5,
            n_place_cells=n_rev_place,
            n_reward_dims=n_rev_reward,
            reward_gain=reward_gain,
            reward_zone_fraction=0.4,
            sigma_scale=1.0,
            balance_modalities=True,
            seed=42,
        )

        rev_conditions = ["direct", "extinction"]
        rev_results: Dict[str, Any] = {}
        try:
            rev_kwargs = _get_model_kwargs(model_kwargs, rev_epochs)
            rev_kwargs["n_features"] = n_rev_place + n_rev_reward

            bench = SpatialReversalBenchmark()
            for cond in rev_conditions:
                # Each condition is an independent run from a fresh model.
                out = bench.evaluate(
                    model=model_class(**rev_kwargs),
                    task=rev_task,
                    condition=cond,
                    max_trials=max_trials,
                    measure_savings=measure_savings,
                    generator=rev_gen,
                )
                rev_results[cond] = out
                for key, val in out.items():
                    if key in ("curves", "condition"):
                        continue
                    results["metrics"][f"reversal_{cond}_{key}"] = val
                results["series"][f"reversal_{cond}_curves"] = out["curves"]

            results["metrics"]["reversal_reward_gain"] = reward_gain
            results["metrics"]["reversal_epochs_per_trial"] = rev_epochs

            # Plot the extinction protocol's stage-by-stage curves, plus the
            # reversal-stage comparison between the two protocols.
            ext = rev_results["extinction"]["curves"]
            stage_order = [s for s in ("acquisition", "extinction", "reversal", "rereversal")
                           if s in ext]
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            offset = 0
            boundaries = []
            for stage in stage_order:
                xs = np.arange(offset + 1, offset + 1 + len(ext[stage]["arm_accuracy"]))
                axes[0].plot(xs, ext[stage]["arm_accuracy"], '-o', label=f'{stage}')
                axes[1].plot(xs, ext[stage]["reward_pred"], '-o', label=f'{stage}')
                axes[2].plot(xs, ext[stage]["anticipation_lead"], '-o', label=f'{stage}')
                offset = xs[-1]
                boundaries.append(offset + 0.5)
            axes[0].axhline(bench.criterion, color='red', linestyle='--', alpha=0.5,
                            label='criterion')
            for ax in axes:
                for b in boundaries[:-1]:
                    ax.axvline(b, color='gray', linestyle=':', alpha=0.7)
                ax.set_xlabel('presentation (trial)')
                ax.legend(fontsize=8)
            axes[0].set_ylabel('arm accuracy (fraction of steps)')
            axes[0].set_title('Branch: which arm is predicted')
            axes[1].set_ylabel('predicted reward / reward_gain')
            axes[1].set_title('Reward channel (extinction is scored here)')
            axes[2].set_ylabel('steps before goal onset')
            axes[2].set_title('Anticipation lead (measured, not injected)')
            fig.suptitle(f'T-Maze Reversal — {model_name} (extinction protocol)')
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "tmaze_reversal.png"), dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"Warning: T-Maze Reversal failed for {model_name}: {e}")
            for cond in rev_conditions:
                results["metrics"][f"reversal_{cond}_reversal_trials_to_criterion"] = float('nan')
                results["metrics"][f"reversal_{cond}_reversal_final_arm_accuracy"] = float('nan')

    # Write metrics.json
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Spatial pipeline finished successfully for {model_name}. Results saved to {run_dir}")
    return results["metrics"]
