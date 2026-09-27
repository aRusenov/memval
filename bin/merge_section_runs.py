#!/usr/bin/env python
"""Union per-section benchmark runs into one suite ``metrics.json``.

**The existing destination file is part of the union.** If ``<out>/<cls>/<suite>/
metrics.json`` already exists, its ``metrics`` and ``series`` are loaded first
and the section runs are layered on top (a section run wins on a duplicate key,
which is reported). Before 2026-09-08 this script rebuilt the destination from
the section runs alone, so merging ONE re-run section into a directory holding a
full-suite run silently replaced thirteen sections with one -- which is exactly
what happened to OriginalEqPropSequenceNetwork's zoo symbolic file that night.
Pass ``--replace`` to get the old behaviour deliberately.

``run_benchmark.py --benchmarks <section>`` rewrites the suite's whole
``metrics.json``, so a suite run one section at a time (to review each before
paying for the next) has to be merged before ``score_capacities.py`` can read
it. Each section run lives in its own ``--output-dir``; this script unions
their ``metrics`` and ``series`` dicts (later runs win on a duplicate key,
which is reported), copies side files (``cue_masking_metrics.json``,
``schema_consistency_metrics.json``, plots) and writes the merged suite file.

    python bin/merge_section_runs.py --class OriginalEqPropSequenceNetwork --suite symbolic \
        --sections-root results/zoo_capacity_run/_ep_sections \
        --out results/zoo_capacity_run

Section-subset runs are not bit-identical to a full-suite run (probe noise
draws from the global RNG; <= 0.03 MRR on noisy-probe arms), which is the
same for every section here and is recorded in the merged metadata.
"""
import argparse
import glob
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--sections-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replace", action="store_true",
                    help="Rebuild the destination from the section runs ALONE, "
                         "discarding whatever metrics.json already holds. Default "
                         "is to union with it.")
    args = ap.parse_args()

    merged = {"metadata": None, "metrics": {}, "series": {}, "sections_merged": []}
    dest = os.path.join(args.out, args.cls, args.suite)
    existing = os.path.join(dest, "metrics.json")
    if os.path.exists(existing) and not args.replace:
        with open(existing) as f:
            prior = json.load(f)
        merged["metadata"] = prior.get("metadata")
        merged["metrics"] = dict(prior.get("metrics", {}))
        merged["series"] = dict(prior.get("series", {}))
        merged["sections_merged"] = list(
            (prior.get("metadata") or {}).get("merged_from_section_runs") or [])
        print(f"starting from existing {existing}: {len(merged['metrics'])} metrics, "
              f"{len(merged['series'])} series (section runs layer on top)")
    os.makedirs(os.path.join(dest, "plots"), exist_ok=True)
    runs = sorted(glob.glob(os.path.join(args.sections_root, "*", args.cls, args.suite, "metrics.json")))
    for path in runs:
        section = path.split(os.sep)[-4]
        with open(path) as f:
            r = json.load(f)
        if merged["metadata"] is None:
            merged["metadata"] = r.get("metadata", {})
        for key in ("metrics", "series"):
            for k, v in r.get(key, {}).items():
                if k in merged[key]:
                    print(f"  duplicate {key} key {k!r} from {section}: keeping the later run")
                merged[key][k] = v
        merged["sections_merged"].append(section)
        src = os.path.dirname(path)
        for side in glob.glob(os.path.join(src, "*_metrics.json")):
            shutil.copy(side, dest)
        for plot in glob.glob(os.path.join(src, "plots", "*")):
            shutil.copy(plot, os.path.join(dest, "plots"))
        print(f"merged {section}: {len(r.get('metrics', {}))} metrics, {len(r.get('series', {}))} series")
    if merged["metadata"] is not None:
        merged["metadata"]["merged_from_section_runs"] = merged["sections_merged"]
    with open(os.path.join(dest, "metrics.json"), "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {dest}/metrics.json with {len(merged['metrics'])} metrics from {len(runs)} section runs")


if __name__ == "__main__":
    main()
