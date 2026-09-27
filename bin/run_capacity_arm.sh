#!/usr/bin/env bash
# Run the full capacity runbook (docs/reports/capacity_report_ahn.md sec 1) for ONE arm.
#
#   bin/run_capacity_arm.sh <model> <ClassName> <out-dir> [n-trials] [extra args...]
#   SKIP_SPATIAL=1 bin/run_capacity_arm.sh ...   # symbolic side only
#
# Steps, in order: the two suites, the streamed regime, the FOCUSED-protocol
# schema re-run (a separate output dir, because the suite default 'extended' is
# valid for acquisition only), the serial-order probe, the standalone sparse-code
# capacity curves (not a scored section), then score and render.
set -o pipefail

MODEL="$1"; CLASS="$2"; OUT="$3"; NT="${4:-30}"; shift 4 2>/dev/null || shift 3
EXTRA=("$@")

# ---------------------------------------------------------------------------
# Capping exposure on a slow arm
# ---------------------------------------------------------------------------
# Extra args are forwarded to the two suite runs, so a slow arm can be given a
# lower staircase ceiling, e.g.
#
#   bin/run_capacity_arm.sh original_eqprop OriginalEqPropSequenceNetwork <out> 30 \
#       --benchmark-args symbolic_disambiguation:max_epochs=128
#
# Use `max_epochs=N`, NOT `epochs=N`. They are different knobs and only one of
# them is safe here:
#
#   max_epochs=N  lowers the CEILING of the criterion staircase. An arm that
#                 reaches criterion below N is bit-identical to an uncapped run
#                 (ahn reaches symbolic_disambiguation at 3 epochs, dts_esn
#                 at 1, temporal_pc at 37); only an arm that would have run past
#                 N is truncated, and it was already reporting
#                 `criterion_reached: False`.
#   epochs=N      pins a FIXED budget and switches `<s>_exposure_mode` to
#                 "fixed", so the staircase does not run at all and every arm
#                 trains exactly N epochs -- including the ones that currently
#                 stop early. That silently changes results across the board and
#                 is for reproducing an old fixed-budget number, not for speed.
#
# Pick N above what the arms you still want measured actually need:
# original_eqprop reaches symbolic_disambiguation's criterion at 347 epochs, so
# a cap of 128 would censor a measurement that currently succeeds. It is a
# per-arm decision, which is why this is an argument and not a default.
#
# Cost, for scale: symbolic_disambiguation sweeps ~30 points across seven axes
# and trains every one to criterion. On an arm that never reaches it (the
# retired spiking EP arm), that section alone took 34 minutes of a 45-minute
# symbolic suite.

LOG="$OUT/_logs"; mkdir -p "$LOG"
say() { echo "[$(date +%H:%M:%S)] $MODEL: $*"; }
fail=0
step() {  # step <name> <cmd...>
  local name="$1"; shift
  say "$name ..."
  local t0=$SECONDS
  if "$@" > "$LOG/${MODEL}_${name}.log" 2>&1; then
    say "$name done in $((SECONDS-t0))s"
  else
    say "$name FAILED (exit $?) -- see $LOG/${MODEL}_${name}.log"
    tail -20 "$LOG/${MODEL}_${name}.log"
    fail=1
  fi
}

# SKIP_SPATIAL=1 runs the symbolic side only, leaving whatever spatial
# metrics.json is already in the tree for the score step to read. Use it for a
# symbolic-only re-run; the resulting scorecard then MIXES a fresh symbolic run
# with an older spatial one, which the merged metadata does not record -- say so
# when reporting such a run.
if [ "${SKIP_SPATIAL:-0}" = "1" ]; then
  say "SKIPPING spatial suite (SKIP_SPATIAL=1); score will read the stored spatial metrics"
else
  step spatial  python -u bin/run_benchmark.py --model "$MODEL" --suite spatial \
      --output-dir "$OUT" --n-trials "$NT" ${EXTRA[@]+"${EXTRA[@]}"}
fi
step symbolic python -u bin/run_benchmark.py --model "$MODEL" --suite symbolic \
    --output-dir "$OUT" --n-trials "$NT" ${EXTRA[@]+"${EXTRA[@]}"}
step online   python -u bin/run_benchmark.py --model "$MODEL" --suite online_symbolic \
    --output-dir "$OUT" --n-trials "$NT"
step schema_focused python -u bin/run_benchmark.py --model "$MODEL" --suite symbolic \
    --benchmarks schema_consistency \
    --benchmark-args schema_consistency:interference_protocol=focused \
    --output-dir "$OUT/_schema_focused" --n-trials "$NT" ${EXTRA[@]+"${EXTRA[@]}"}
step serial_order python -u bin/probe_serial_order.py --model "$MODEL" \
    --out "$OUT/$CLASS/serial_order_probe.json"
step sparse_capacity python -u bin/sparse_capacity_benchmark.py --model "$MODEL" \
    --out "$OUT"
step score python -u bin/score_capacities.py \
    --results-dir "$OUT/$CLASS" \
    --schema-focused "$OUT/_schema_focused/$CLASS" \
    --model "$CLASS"
step page python -u bin/build_scorecard_page.py \
    --scorecard "$OUT/$CLASS/capacity_scorecard.json" \
    --out "$OUT/$CLASS/capacity_scorecard.html"

say "ALL STEPS ATTEMPTED (fail=$fail)"
exit $fail
