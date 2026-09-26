# Re-score preview, 2026-09-26 — NOT published

What the six rate arms' scorecards become when re-scored today, produced as a
side effect of merging the `noise_invariance` rerun (quantized rollout -> raw).
The published scorecards in `../<Class>/` were restored to their previous state
because this re-score also moves scores that have nothing to do with that rerun:

- **Serial order / `establishment`** drops on all six arms (e.g. DTS-ESN
  1.00 -> 0.84, EP 0.99 -> 0.83) although DTS-ESN's inputs are unchanged since
  09-07, so the scorer's establishment logic has changed since these scorecards
  were written.
- **AHN Serial order / `binding_ordinal`** goes unresolved -> scored (0.59;
  coverage 0.3 -> 0.5), because `AsymmetricHopfieldNetwork/serial_order_probe.json`
  was rewritten 09-25 22:48.
- **One-shot / `schema_consistency`** moves on AHN and theta (0.613 -> 0.674).

Continual retention and Pattern completion are identical. Promote these by
re-running `bin/score_capacities.py` (with `--schema-focused`) and
`bin/build_scorecard_page.py` per arm once the above is reviewed.
