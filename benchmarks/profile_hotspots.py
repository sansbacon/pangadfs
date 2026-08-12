# -*- coding: utf-8 -*-
"""Call-level benchmark profiler for pangadfs.

Produces cProfile .prof artifacts and readable text summaries for:
- OptimizeDefault flow
- OptimizeMultilineup flow

Usage:
    python benchmarks/profile_hotspots.py
    python benchmarks/profile_hotspots.py --population-size 1500 --generations 20 --top 40
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
from pathlib import Path
from typing import Dict, Tuple

from stevedore import driver, named

from pangadfs.ga import GeneticAlgorithm


def _build_context(population_size: int, n_generations: int, optimize_driver: str) -> Dict:
    return {
        "ga_settings": {
            "n_generations": n_generations,
            "population_size": population_size,
            "stop_criteria": max(3, n_generations // 4),
            "points_column": "proj",
            "salary_column": "salary",
            "position_column": "pos",
            "csvpth": Path("tests/test_pool.csv"),
            "target_lineups": 10,
            "diversity_weight": 0.2,
            "min_overlap_threshold": 0.4,
            "diversity_method": "jaccard",
            "enable_profiling": True,
            "verbose": False,
        },
        "site_settings": {
            "salary_cap": 50000,
            "posmap": {"DST": 1, "QB": 1, "TE": 1, "RB": 2, "WR": 3, "FLEX": 1},
            "lineup_size": 9,
            "posfilter": {"QB": 14, "RB": 8, "WR": 8, "TE": 5, "DST": 4, "FLEX": 8},
            "flex_positions": ("RB", "WR", "TE"),
            "optimize_driver": optimize_driver,
        },
    }


def _build_plugin_managers(optimize_driver: str) -> Tuple[Dict, Dict]:
    mapping = {
        "pool": "pool_default",
        "pospool": "pospool_default",
        "populate": "populate_default",
        "fitness": "fitness_default",
        "optimize": optimize_driver,
        "select": "select_default",
        "crossover": "crossover_default",
        "mutate": "mutate_default",
    }

    dms = {
        k: driver.DriverManager(namespace=f"pangadfs.{k}", name=v, invoke_on_load=True)
        for k, v in mapping.items()
    }

    ems = {
        "validate": named.NamedExtensionManager(
            namespace="pangadfs.validate",
            names=["validate_salary", "validate_duplicates"],
            invoke_on_load=True,
            name_order=True,
        )
    }
    return dms, ems


def _build_ga(population_size: int, n_generations: int, optimize_driver: str) -> GeneticAlgorithm:
    ctx = _build_context(population_size=population_size, n_generations=n_generations, optimize_driver=optimize_driver)
    dms, ems = _build_plugin_managers(optimize_driver=optimize_driver)
    return GeneticAlgorithm(ctx=ctx, driver_managers=dms, extension_managers=ems)


def _run_profile(population_size: int, n_generations: int, optimize_driver: str, output_prefix: Path, top: int) -> Dict:
    ga = _build_ga(population_size=population_size, n_generations=n_generations, optimize_driver=optimize_driver)

    profiler = cProfile.Profile()
    profiler.enable()
    result = ga.optimize()
    profiler.disable()

    prof_path = output_prefix.with_suffix(".prof")
    txt_path = output_prefix.with_suffix(".txt")

    profiler.dump_stats(str(prof_path))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(top)
    txt_path.write_text(stream.getvalue(), encoding="utf-8")

    profile_data = result.get("profiling", {})
    return {
        "optimize_driver": optimize_driver,
        "population_size": population_size,
        "generations": n_generations,
        "total_profiled_s": float(profile_data.get("total_time", 0.0)),
        "setup_s": float(profile_data.get("setup_time", 0.0)),
        "avg_generation_s": float(profile_data.get("avg_generation_time", 0.0)),
        "prof_file": str(prof_path),
        "text_report": str(txt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cProfile hotspot profiling for pangadfs optimization")
    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--top", type=int, default=30, help="Number of top functions by cumulative time")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    default_summary = _run_profile(
        population_size=args.population_size,
        n_generations=args.generations,
        optimize_driver="optimize_default",
        output_prefix=args.output_dir / "hotspots_optimize_default",
        top=args.top,
    )

    multilineup_summary = _run_profile(
        population_size=args.population_size,
        n_generations=args.generations,
        optimize_driver="optimize_multilineup",
        output_prefix=args.output_dir / "hotspots_optimize_multilineup",
        top=args.top,
    )

    summary = {
        "default": default_summary,
        "multilineup": multilineup_summary,
    }

    summary_path = args.output_dir / "hotspots_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Hotspot profiling completed")
    print(f"Summary: {summary_path}")
    print(f"Default profile: {default_summary['prof_file']}")
    print(f"Multilineup profile: {multilineup_summary['prof_file']}")


if __name__ == "__main__":
    main()
