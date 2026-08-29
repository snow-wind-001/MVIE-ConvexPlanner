"""Run unseen-seed forest robustness tests without replacing UI demo data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forest_simulator.generate_forest_scenarios import (  # noqa: E402
    DENSITIES,
    run_scenario,
)


def timing_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(records, density=None):
    selected = records if density is None else [
        record for record in records if record["density"] == density
    ]
    successes = sum(record["reactive_success"] for record in selected)
    safe = sum(record["reactive_safe"] for record in selected)
    deadline = sum(record["deadline_met"] for record in selected)
    clearance = [
        record["reactive_min_clearance_m"]
        for record in selected
        if record["reactive_min_clearance_m"] is not None
    ]
    overhead = [
        record["length_overhead_percent"]
        for record in selected
        if record["length_overhead_percent"] is not None
    ]
    return {
        "scenarios": len(selected),
        "full_safe_count": sum(record["full_safe"] for record in selected),
        "reactive_success_count": successes,
        "reactive_safe_count": safe,
        "deadline_met_count": deadline,
        "reactive_success_rate": successes / len(selected),
        "reactive_success_wilson_95": wilson_interval(successes, len(selected)),
        "planning_timing": timing_summary(
            [record["reactive_planning_ms"] for record in selected]
        ),
        "validation_timing": timing_summary(
            [record["reactive_validation_ms"] for record in selected]
        ),
        "guide_timing": timing_summary(
            [record["reactive_guide_ms"] for record in selected]
        ),
        "full_timing": timing_summary(
            [record["full_planning_ms"] for record in selected]
        ),
        "minimum_clearance_m": None if not clearance else float(min(clearance)),
        "median_clearance_m": None if not clearance else float(np.median(clearance)),
        "median_length_overhead_percent": None
        if not overhead
        else float(np.median(overhead)),
        "failed_seeds": [
            record["seed"] for record in selected if not record["reactive_success"]
        ],
        "unsafe_output_seeds": [
            record["seed"]
            for record in selected
            if record["reactive_success"] and not record["reactive_safe"]
        ],
        "deadline_miss_seeds": [
            record["seed"] for record in selected if not record["deadline_met"]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument(
        "--densities", nargs="+", choices=DENSITIES, default=list(DENSITIES)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "ccfa_analysis/forest_extended_benchmark.json",
    )
    args = parser.parse_args()

    records = []
    failure_cases = []
    for density in args.densities:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            scenario = run_scenario(density, seed)
            metrics = scenario["metrics"]
            records.append(
                {
                    "scenario_id": scenario["id"],
                    "density": density,
                    "seed": seed,
                    **metrics,
                }
            )
            if not metrics["reactive_success"] or not metrics["reactive_safe"]:
                failure_cases.append(scenario)
            print(
                f"{scenario['id']}: safe={metrics['reactive_safe']} "
                f"deadline={metrics['deadline_met']} "
                f"realtime={metrics['reactive_planning_ms']:.2f} ms",
                flush=True,
            )

    payload = {
        "protocol": {
            "seed_start": args.seed_start,
            "seeds_per_density": args.seeds,
            "densities": args.densities,
            "seen_demo_seeds_excluded": args.seed_start >= 10,
            "test": (
                "full FIRI reference, <=4 m resampling, one new trunk on route, "
                "one bounded 20 ms realtime repair, independent exact verification"
            ),
        },
        "overall": summarize(records),
        "summaries": {
            density: summarize(records, density) for density in args.densities
        },
        "records": records,
        "failure_cases": failure_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["summaries"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
