# noise_invariance before the 2026-09-26 rerun

Per-arm copies of `symbolic/metrics.json`, `plots/noise_invariance.png` and the
capacity scorecards as they stood before `noise_invariance` was rerun to replace
panel 2's quantized rollout with a raw rollout.

The rerun pinned each arm to its stored `noise_epochs_to_criterion` (AHN 4,
theta 4, DTS-ESN 1, tPC 86, Chen 128, EP 220), and every stored cued and l2
rollout value reproduced exactly. The only differences are: `rollout_raw` added,
`rollout_quantized` / `noise_rollout_quantized_auc` removed. Raw section runs:
`../../_noise_raw_2026-09-26/noise_invariance/`. Logs:
`../../_logs/<model>_noise_raw_2026-09-26.log`.

Underscore directories are skipped by `bin/build_results_index.py`.
