# Superseded: AHN sparse_capacity at 4 passes per ingestion

Archived 2026-09-24. These are the files that stood in
`AsymmetricHopfieldNetwork/sparse_capacity/` before the re-run at the current
registry budget.

- `epochs` = 4. The run took it from `MODEL_REGISTRY["hopfield"]["n_epochs"]`
  as it stood at the time (`--epochs` was not passed).
- AHN's registry `n_epochs` was re-sized 4 -> 2 on 2026-09-23, so the same
  command now produces a different curve. The replacement is at 2.
- What moved (condition B, raw unroll): 0.267 -> 0.222 at 10 items,
  0.202 -> 0.182 at 100 items. Condition A and cued recall are effectively
  unchanged.
- Reproduce these exact files with `--epochs 4`.

The replacement also carries per-position readouts (`unroll_curve`,
`unroll_prefix`, `sparse_capacity_positions.png`) that did not exist when this
was generated, so the JSON schemas differ.
