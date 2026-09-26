import inspect
import numpy as np
from typing import Dict, Any, List, Type
from ..models.base import HippocampalModel
from ..models.capabilities import is_stochastic_forward, supports_priming, rollout_mode_of
from .base import Benchmark
from .overlap_config import OverlapConfig
from ..generators.bifurcating_route import BifurcatingRouteGenerator


def consolidate_if_supported(model, sequence_data) -> None:
    """Call ``model.consolidate`` after a ``fit_sequence``, if the model defines it.

    Models use two consolidate signatures: EWC-style needs the just-fitted
    ``sequence_data`` (to accumulate Fisher information and re-anchor), while
    CLS-style takes no sequence argument. Dispatch on the signature so both work.
    """
    consolidate = getattr(model, "consolidate", None)
    if consolidate is None:
        return
    if "sequence_data" in inspect.signature(consolidate).parameters:
        consolidate(sequence_data)
    else:
        consolidate()


class SpatialDisambiguationBenchmark(Benchmark):
    """
    Evaluates a hippocampal model on multi-modal sequence disambiguation.
    Two routes share a central corridor and diverge into distinct suffixes,
    distinguished by an odour/encounter signal carried on a separate input
    block alongside the place code.

    Note on what this currently measures: the odour is present on *every* step
    of the shared zone, including the last one before the branch, and it is
    delivered as extra input dimensions rather than through the model's
    ``context`` channel. So this scores concurrent binding of a co-presented
    "what"-stream signal to a branch choice, not maintenance of a cue across
    the corridor. Gating the odour off partway up the stem is what would turn
    the corridor into a genuine delay.
    """

    def __init__(self, metric: str = "mse"):
        self.metric = metric

    def evaluate(
        self,
        model: HippocampalModel,
        route_pair: Dict[str, Any],
        condition: str = "full",
        n_trials: int = 20,
        fit_epochs: int = 200,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model under a specific ablation condition.
        
        Args:
            model: The integrated HippocampalModel under test.
            route_pair: Output dictionary from BifurcatingRouteGenerator.generate().
            condition: "full" | "mec_only" | "lec_only" | "conflicting"
            n_trials: Number of independent retrieval trials.
            fit_epochs: Epochs to train the model on each sequence.
            
        Returns:
            Dict containing disambiguation metrics.
        """
        input_A = route_pair["input_A"]
        input_B = route_pair["input_B"]
        zs = route_pair["zone_start"]
        ze = route_pair["zone_end"]
        se = route_pair["shared_end"]
        L = len(input_A)
        n_features = input_A.shape[-1]
        
        encoder = route_pair["encoder"]
        n_place_cells = encoder.n_cells
        n_encounter_dims = n_features - n_place_cells
        
        # Probe protocol (docs/probe_protocol.md). Nothing inside a trial is
        # random: test_seq_A/B are built once below and the rollout is a
        # deterministic function of them, so for a deterministic arm every
        # trial is an identical recomputation. Repeat only when the arm's own
        # forward pass is stochastic; then the repeats sample the model's noise.
        n_trials_requested = int(n_trials)
        n_trials = n_trials_requested if is_stochastic_forward(model) else 1

        # 1. Train the model on both sequences
        model.reset_context()
        model.fit_sequence(input_A, epochs=fit_epochs, **kwargs)
        consolidate_if_supported(model, input_A)

        model.fit_sequence(input_B, epochs=fit_epochs, **kwargs)
        consolidate_if_supported(model, input_B)

        # Build condition-masked test sequences
        # Route A test sequence
        test_seq_A = input_A.copy()
        if condition == "mec_only":
            test_seq_A[:, n_place_cells:] = 0.0
        elif condition == "lec_only":
            test_seq_A[:, :n_place_cells] = 0.0
        elif condition == "conflicting":
            # MEC from A, LEC from B
            test_seq_A[:, n_place_cells:] = input_B[:, n_place_cells:]

        # Route B test sequence
        test_seq_B = input_B.copy()
        if condition == "mec_only":
            test_seq_B[:, n_place_cells:] = 0.0
        elif condition == "lec_only":
            test_seq_B[:, :n_place_cells] = 0.0
        elif condition == "conflicting":
            # MEC from B, LEC from A
            test_seq_B[:, n_place_cells:] = input_A[:, n_place_cells:]

        # Run trials for Route A and Route B
        results_A = self._run_trials(model, test_seq_A, input_A, route_pair, "A", ze, L, n_place_cells, n_encounter_dims, n_trials)
        results_B = self._run_trials(model, test_seq_B, input_B, route_pair, "B", ze, L, n_place_cells, n_encounter_dims, n_trials)

        # Average metrics across routes
        def _mean(key):
            vals = [v for v in (results_A[key], results_B[key]) if not np.isnan(v)]
            # All-NaN is legitimate: shared_stretch_error is undefined at
            # delay = 0, where the rollout begins at the branch and there is no
            # shared stretch to score. Return NaN rather than warning on it.
            return float(np.mean(vals)) if vals else float("nan")

        rec_A, rec_B = results_A["recovery_profile"], results_B["recovery_profile"]
        recovery = (np.nanmean([rec_A, rec_B], axis=0).tolist()
                    if len(rec_A) and len(rec_A) == len(rec_B) else [])

        return {
            "branch_accuracy": _mean("branch_accuracy"),
            "confusion_rate": _mean("confusion_rate"),
            "suffix_mse": _mean("suffix_mse"),
            "divergence_accuracy": _mean("divergence_accuracy"),
            "branch_margin": _mean("branch_margin"),
            "divergence_margin": _mean("divergence_margin"),
            "shared_stretch_error": _mean("shared_stretch_error"),
            "recovery_profile": recovery,
            "route_A_branch_accuracy": results_A["branch_accuracy"],
            "route_B_branch_accuracy": results_B["branch_accuracy"],
            # delay = shared_end - zone_end: shared steps with no odour available.
            # 0 reproduces the fully-cued corner; > 0 requires carried state.
            "delay": int(se - ze),
            # Protocol provenance, propagated from the trial runner. See the note
            # on `state_primed` in _run_trials.
            "state_primed": bool(results_A["state_primed"]),
            "probe_protocol": "clean_single",
            "rollout": "model_owned_recall",
            "n_trials_requested": n_trials_requested,
            "n_trials_effective": int(n_trials),
            "stochastic_forward": bool(is_stochastic_forward(model)),
        }

    def _run_trials(
        self,
        model: HippocampalModel,
        test_seq: np.ndarray,
        gt_seq: np.ndarray,
        route_pair: Dict[str, Any],
        target_route: str,
        ze: int,
        L: int,
        n_place_cells: int,
        n_encounter_dims: int,
        n_trials: int
    ) -> Dict[str, float]:
        recall_len = L - ze
        se = route_pair["shared_end"]
        encoder = route_pair["encoder"]
        
        gt_coords_A = encoder.decode(route_pair["input_A"][:, :n_place_cells])
        gt_coords_B = encoder.decode(route_pair["input_B"][:, :n_place_cells])

        trial_branch_accs = []
        trial_confusion_rates = []
        trial_suffix_mses = []
        trial_divergence_accs = []
        trial_recovery = []
        # Whether this arm can be state-primed over the shared zone. Constant
        # across trials; recorded in the result so every row says which
        # protocol produced it.
        primed = supports_priming(model)

        trial_margins = []
        trial_divergence_margins = []
        trial_shared_errors = []

        for _ in range(n_trials):
            # 1. Reset, then deliver the stem up to the cue through
            #    `observe_sequence` (non-learning state update; the base-class
            #    no-op on arms without carried state, so the same call on every
            #    arm and no capability check). `primed` records whether the arm
            #    declares carried state (StatePrimeable): provenance, not a gate.
            model.reset_context()
            if ze > 1:
                model.observe_sequence(test_seq[: ze - 1])

            # 2. The arm's own rollout from the cue (2026-09-08; previously a
            #    harness-driven predict_next loop with the odour clamped from
            #    the test sequence at every step). `recall` rolls forward under
            #    the arm's declared `rollout_mode` with nothing clamped: place
            #    and odour after the cue are whatever the arm feeds itself. In
            #    the withdrawn material the odour is zero after the zone during
            #    training, so a trained arm predicts ~0 there on its own; in the
            #    concurrent material it re-predicts the cue and carries it in
            #    its loop. Under `mec_only` / `conflicting` the masking applies
            #    to the observed prefix and the cue, which is all the arm sees.
            rec = np.asarray(model.recall(test_seq[ze - 1], length=recall_len), dtype=float)
            recalled_steps = np.zeros((recall_len, test_seq.shape[1]))
            n_got = min(len(rec), recall_len)
            # Clip at zero for decoding and MSE (the decoder is a centre of
            # mass over non-negative activations). Until 2026-09-08 this was an
            # affine remap (0.5p + 0.5) applied whenever any output was
            # negative -- right only for a tanh-coded bump with a -1
            # background, which no arm on the roster produces; it put a 0.5
            # pedestal on every cell and read ~0.005 margins for arms that
            # separate the routes at ~0.8.
            recalled_steps[:n_got] = np.clip(rec[:n_got], 0.0, None)

            # Evaluate metrics for suffix portion (se to L)
            # Find recall step indices corresponding to suffix: [se - ze : L - ze]
            suffix_start_idx = se - ze
            if suffix_start_idx < 0:
                suffix_start_idx = 0
            
            pred_suffix_pc = recalled_steps[suffix_start_idx:, :n_place_cells]
            gt_suffix_pc = gt_seq[se:, :n_place_cells]
            
            # 3. Compute suffix MSE
            suffix_mse = np.mean((gt_suffix_pc - pred_suffix_pc) ** 2)
            trial_suffix_mses.append(suffix_mse)

            # 4. Compute classification (branch accuracy) via 2D decoding
            correct_counts = 0
            confusion_counts = 0
            total_suffix_steps = L - se
            
            if total_suffix_steps > 0:
                decoded_coords = encoder.decode(pred_suffix_pc) # shape (total_suffix_steps, 2)
                
                step_correct = []
                step_margin = []
                for step_idx in range(total_suffix_steps):
                    t_global = se + step_idx
                    pred_c = decoded_coords[step_idx]

                    dist_to_A = np.linalg.norm(pred_c - gt_coords_A[t_global])
                    dist_to_B = np.linalg.norm(pred_c - gt_coords_B[t_global])

                    if target_route == "A":
                        d_correct, d_wrong = dist_to_A, dist_to_B
                    else:
                        d_correct, d_wrong = dist_to_B, dist_to_A

                    if d_correct < d_wrong:
                        correct_counts += 1
                        step_correct.append(1.0)
                    elif d_wrong < d_correct:
                        confusion_counts += 1
                        step_correct.append(0.0)
                    else:
                        step_correct.append(0.0)

                    # Arm separation at this step normalises the margin, so it is
                    # comparable across steps (the arms fan apart) and configs.
                    sep = float(np.linalg.norm(gt_coords_A[t_global] - gt_coords_B[t_global]))
                    step_margin.append((d_wrong - d_correct) / sep if sep > 1e-9 else float("nan"))
                    
                trial_branch_accs.append(correct_counts / total_suffix_steps)
                trial_confusion_rates.append(confusion_counts / total_suffix_steps)
                trial_divergence_accs.append(step_correct[0])
                trial_recovery.append(step_correct)
                trial_margins.append(float(np.mean(step_margin)))
                trial_divergence_margins.append(step_margin[0])
            else:
                trial_branch_accs.append(1.0)
                trial_confusion_rates.append(0.0)
                trial_divergence_accs.append(1.0)
                trial_recovery.append([])
                trial_margins.append(float("nan"))
                trial_divergence_margins.append(float("nan"))

            # Shared-stretch control (docs/disambiguation_design.md S5.1). The
            # rollout starts at the last cued step, so recalled steps
            # [0, se - ze) fall inside the shared corridor, where BOTH episodes
            # predict the same successor. Error here should be near zero for
            # every arm; it is reported as a control and never folded into the
            # headline number. Undefined (NaN) at delay = 0, where the rollout
            # begins at the branch.
            if suffix_start_idx > 0:
                shared_dec = encoder.decode(recalled_steps[:suffix_start_idx, :n_place_cells])
                shared_gt = gt_coords_A[ze:se]
                trial_shared_errors.append(
                    float(np.mean(np.linalg.norm(shared_dec - shared_gt, axis=1))))

        recovery = (np.mean(np.array(trial_recovery), axis=0).tolist()
                    if trial_recovery and len(trial_recovery[0]) else [])
        return {
            "branch_accuracy": float(np.mean(trial_branch_accs)),
            "confusion_rate": float(np.mean(trial_confusion_rates)),
            "suffix_mse": float(np.mean(trial_suffix_mses)),
            # Scored at the divergence step alone: during the shared stretch both
            # episodes predict the same successor, so an ambiguous representation
            # is scored *correct* there. Averaging over the suffix therefore gives
            # away a fraction of the score for free, and that fraction grows with
            # shared-stretch length -- the metric eases exactly as the task hardens.
            "divergence_accuracy": float(np.mean(trial_divergence_accs)),
            "recovery_profile": recovery,
            # Normalised margin: (d_wrong - d_correct) / arm separation. A thin
            # linear bias no longer scores identically to a decisive commitment.
            "branch_margin": float(np.nanmean(trial_margins)),
            "divergence_margin": float(np.nanmean(trial_divergence_margins)),
            "shared_stretch_error": (float(np.mean(trial_shared_errors))
                                     if trial_shared_errors else float("nan")),
            # Protocol provenance. False means the shared zone was NOT ingested
            # (the arm declares no StatePrimeable), so a withdrawn-cue floor is
            # the arm having no state to carry -- not the harness withholding
            # the history. True means the history was delivered and the arm
            # still had to use it. Any figure comparing rows must show this.
            "state_primed": bool(primed),
        }

    def guidance_sweep(
        self,
        model_factory,
        route_pair: Dict[str, Any],
        fit_epochs: int,
        seeds=(0,),
        min_cue_point: int = 1,
        **fit_kwargs,
    ) -> Dict[str, Any]:
        """Guidance-withdrawal sweep (2026-09-08).

        The material is the fully-cued pair (odour through stem AND arms on
        both routes) and is the same on every rung; only the probe moves. The
        harness walks the model up the stem with ``observe_sequence`` (the
        base-class no-op on arms without carried state) as far as cue point
        ``k``, then hands the rollout to the arm's own ``recall`` from the cue
        ``[place_k, odour]``. Nothing is clamped after that: place and odour
        are whatever the arm rolls forward under its declared ``rollout_mode``.
        Recall starts ``d = shared_end - k`` steps before the fork, so the
        x-axis reads "guided this far, does it recover the correct arm".

        Per rung and route, scored on the decoded (clipped) place block:

        * ``arm_accuracy`` -- fraction of arm steps whose decoded position is
          nearer the correct arm (any point on it) than the wrong arm. Credit
          for the choice regardless of position along the arm; ties fail.
        * ``positional_error`` -- mean decoded distance to the step-matched
          target over the arm; ``stem_error`` the same over the free-run
          stretch before the fork (NaN at d = 0).
        * ``divergence_margin`` / ``branch_margin`` -- the section's
          step-matched normalised margins, at the fork step and over the arm.

        ``model_factory(seed)`` returns a fresh arm; each seed is trained once
        (blocked: A then B, ``fit_epochs`` each) and probed at every rung.
        Means are over routes then seeds; ``*_sd`` is the s.d. over seeds of
        the route-mean.
        """
        input_A = route_pair["input_A"]
        input_B = route_pair["input_B"]
        se = route_pair["shared_end"]
        L = len(input_A)
        encoder = route_pair["encoder"]
        n_pc = encoder.n_cells
        gt = {"A": encoder.decode(input_A[:, :n_pc]),
              "B": encoder.decode(input_B[:, :n_pc])}
        arm_pts = {"A": gt["A"][se:], "B": gt["B"][se:]}
        cue_points = list(range(se, max(1, int(min_cue_point)) - 1, -1))
        keys = ("arm_accuracy", "positional_error", "stem_error",
                "divergence_margin", "branch_margin")

        def _score(rec: np.ndarray, k: int, tgt: str) -> Dict[str, float]:
            other = "B" if tgt == "A" else "A"
            pc = np.zeros((L - k, n_pc))
            n = min(len(rec), L - k)
            pc[:n] = np.clip(np.asarray(rec, dtype=float)[:n, :n_pc], 0.0, None)
            dec = encoder.decode(pc)
            i0 = se - k
            arm_dec = dec[i0:]
            pos_err = float(np.mean(np.linalg.norm(arm_dec - gt[tgt][se:], axis=1)))
            stem_err = (float(np.mean(np.linalg.norm(dec[:i0] - gt[tgt][k:se], axis=1)))
                        if i0 > 0 else float("nan"))
            corr, margins = [], []
            for j, pt in enumerate(arm_dec):
                t = se + j
                d_c_any = float(np.min(np.linalg.norm(arm_pts[tgt] - pt, axis=1)))
                d_w_any = float(np.min(np.linalg.norm(arm_pts[other] - pt, axis=1)))
                corr.append(1.0 if d_c_any < d_w_any else 0.0)
                d_c = float(np.linalg.norm(pt - gt[tgt][t]))
                d_w = float(np.linalg.norm(pt - gt[other][t]))
                sep = float(np.linalg.norm(gt["A"][t] - gt["B"][t]))
                margins.append((d_w - d_c) / sep if sep > 1e-9 else float("nan"))
            return {
                "arm_accuracy": float(np.mean(corr)),
                "positional_error": pos_err,
                "stem_error": stem_err,
                "divergence_margin": float(margins[0]),
                "branch_margin": float(np.nanmean(margins)),
            }

        per = {k: {"A": [], "B": []} for k in cue_points}
        rollout = None
        primed_decl = None
        for s in seeds:
            model = model_factory(s)
            rollout = getattr(rollout_mode_of(model), "value", None)
            primed_decl = supports_priming(model)
            if hasattr(model, "n_epochs"):
                model.n_epochs = int(fit_epochs)
            model.reset_context()
            model.fit_sequence(input_A, epochs=fit_epochs, **fit_kwargs)
            consolidate_if_supported(model, input_A)
            model.fit_sequence(input_B, epochs=fit_epochs, **fit_kwargs)
            consolidate_if_supported(model, input_B)
            for k in cue_points:
                for tgt, X in (("A", input_A), ("B", input_B)):
                    model.reset_context()
                    if k > 1:
                        model.observe_sequence(X[: k - 1])
                    rec = model.recall(X[k - 1], length=L - k)
                    per[k][tgt].append(_score(np.asarray(rec, dtype=float), k, tgt))

        records = []
        for k in cue_points:
            rec = {"cue_point": int(k), "recall_starts_before_fork": int(se - k)}
            for key in keys:
                a = np.array([r[key] for r in per[k]["A"]], dtype=float)
                b = np.array([r[key] for r in per[k]["B"]], dtype=float)
                seed_means = np.nanmean(np.stack([a, b]), axis=0)
                rec[key] = float(np.nanmean(seed_means))
                rec[key + "_sd"] = (float(np.nanstd(seed_means)) if len(seed_means) > 1
                                    else float("nan"))
                rec["route_A_" + key] = float(np.nanmean(a))
                rec["route_B_" + key] = float(np.nanmean(b))
            records.append(rec)
        return {
            "records": records,
            "n_seeds": int(len(list(seeds))),
            "rollout_mode": rollout,
            "state_primed": bool(primed_decl),
            "probe_protocol": "observe_then_recall",
        }

    def sweep(
        self,
        model_class: Type[HippocampalModel],
        model_kwargs: Dict[str, Any],
        configs: List[OverlapConfig],
        conditions: List[str],
        n_trials: int = 20,
        fit_epochs: int = 200,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Runs evaluate() across every config × condition pair.
        """
        records = []
        generator = BifurcatingRouteGenerator(seed=42)
        
        for config in configs:
            # Generate the specific environment/route geometry
            geom = config.resolve()
            
            # Instantiate model
            # Note: n_features must be n_place_cells + n_encounter_dims
            n_place_cells = 400
            n_encounter_dims = 50
            total_features = n_place_cells + n_encounter_dims
            
            # Prepare constructor args
            run_kwargs = dict(model_kwargs)
            # Ensure proper n_features or equivalent
            if "n_features" not in run_kwargs:
                run_kwargs["n_features"] = total_features
            
            # Some models require specific init kwargs (e.g. n_epochs)
            if "n_epochs" in run_kwargs:
                run_kwargs["n_epochs"] = fit_epochs
            if "epochs" in run_kwargs:
                run_kwargs["epochs"] = fit_epochs

            route_pair = generator.generate(
                total_length=config.total_length,
                shared_fraction=config.shared_fraction,
                shared_position=config.shared_position,
                zone_fraction=config.zone_fraction,
                zone_offset=config.zone_offset,
                encounter_similarity=config.encounter_similarity,
                n_place_cells=n_place_cells,
                n_encounter_dims=n_encounter_dims,
                seed=42
            )
            
            for cond in conditions:
                # Instantiate fresh model
                model = model_class(**run_kwargs)
                
                metrics = self.evaluate(
                    model=model,
                    route_pair=route_pair,
                    condition=cond,
                    n_trials=n_trials,
                    fit_epochs=fit_epochs,
                    **kwargs
                )
                
                # Create record
                record = {
                    "total_length": config.total_length,
                    "shared_fraction": config.shared_fraction,
                    "shared_position": config.shared_position,
                    "zone_fraction": config.zone_fraction,
                    "zone_offset": config.zone_offset,
                    "encounter_similarity": config.encounter_similarity,
                    "condition": cond,
                    **metrics
                }
                records.append(record)
                
        return records
