# -*- coding: utf-8 -*-
"""Lightweight benchmark suite for pangadfs optimization hot paths.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --repeats 2 --population-sizes 500 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

from stevedore import driver, named

from pangadfs.ga import GeneticAlgorithm
from pangadfs.optimize import OptimizeMultilineup


def _build_context(population_size: int, n_generations: int, target_lineups: int = 10) -> Dict:
    return {
        "ga_settings": {
            "n_generations": n_generations,
            "population_size": population_size,
            "stop_criteria": max(3, n_generations // 4),
            "points_column": "proj",
            "salary_column": "salary",
            "position_column": "pos",
            "csvpth": Path("tests/test_pool.csv"),
            "target_lineups": target_lineups,
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
        },
    }


def _build_plugin_managers() -> tuple[Dict, Dict]:
    plugins = [ns for ns in GeneticAlgorithm.PLUGIN_NAMESPACES if ns != "validate"]
    mapping = {p: f"{p}_default" for p in plugins}
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


def _build_ga(population_size: int, n_generations: int, target_lineups: int = 10) -> GeneticAlgorithm:
    ctx = _build_context(population_size=population_size, n_generations=n_generations, target_lineups=target_lineups)
    dms, ems = _build_plugin_managers()
    return GeneticAlgorithm(ctx=ctx, driver_managers=dms, extension_managers=ems)


def _run_optimize_once(population_size: int, n_generations: int) -> Dict[str, float]:
    ga = _build_ga(population_size=population_size, n_generations=n_generations)

    start = time.perf_counter()
    result = ga.optimize()
    total_wall = time.perf_counter() - start

    profiling = result.get("profiling", {})
    operations = profiling.get("operations", {})

    return {
        "population_size": float(population_size),
        "total_wall_s": total_wall,
        "total_profiled_s": float(profiling.get("total_time", 0.0)),
        "setup_s": float(profiling.get("setup_time", 0.0)),
        "avg_generation_s": float(profiling.get("avg_generation_time", 0.0)),
        "population_build_s": float(operations.get("Initial Population", {}).get("total_time", 0.0)),
        "validation_total_s": float(operations.get("Validation", {}).get("total_time", 0.0)),
        "validation_avg_s": float(operations.get("Validation", {}).get("avg_time", 0.0)),
        "fitness_total_s": float(operations.get("Fitness Evaluation", {}).get("total_time", 0.0)),
        "fitness_avg_s": float(operations.get("Fitness Evaluation", {}).get("avg_time", 0.0)),
        "selection_total_s": float(operations.get("Selection", {}).get("total_time", 0.0)),
    }


def benchmark_generation_and_validation(population_sizes: List[int], n_generations: int, repeats: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for size in population_sizes:
        samples = [_run_optimize_once(population_size=size, n_generations=n_generations) for _ in range(repeats)]
        keys = samples[0].keys()
        row = {k: mean([s[k] for s in samples]) for k in keys}
        rows.append(row)
    return rows


def benchmark_diversity_selection(population_size: int, n_generations: int, repeats: int, target_lineups: int) -> Dict[str, float]:
    ga = _build_ga(population_size=population_size, n_generations=n_generations, target_lineups=target_lineups)
    base_result = ga.optimize()
    population = base_result["population"]
    fitness = base_result["fitness"]

    optimizer = OptimizeMultilineup()
    timings: List[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        optimizer._select_diverse_lineups(
            population=population,
            fitness=fitness,
            target_count=target_lineups,
            diversity_weight=ga.ctx["ga_settings"].get("diversity_weight", 0.2),
            min_overlap_threshold=ga.ctx["ga_settings"].get("min_overlap_threshold", 0.4),
            diversity_method=ga.ctx["ga_settings"].get("diversity_method", "jaccard"),
        )
        timings.append(time.perf_counter() - start)

    return {
        "population_size": float(population_size),
        "target_lineups": float(target_lineups),
        "diversity_avg_s": mean(timings),
        "diversity_min_s": min(timings),
        "diversity_max_s": max(timings),
    }


def _render_markdown(gen_rows: List[Dict[str, float]], diversity: Dict[str, float]) -> str:
    lines = [
        "# pangadfs Benchmark Report",
        "",
        "## Generation and Validation Timings",
        "",
        "| population_size | total_wall_s | setup_s | population_build_s | avg_generation_s | validation_total_s | validation_avg_s | fitness_total_s | fitness_avg_s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in gen_rows:
        lines.append(
            "| {population_size:.0f} | {total_wall_s:.4f} | {setup_s:.4f} | {population_build_s:.4f} | {avg_generation_s:.4f} | {validation_total_s:.4f} | {validation_avg_s:.6f} | {fitness_total_s:.4f} | {fitness_avg_s:.6f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Diversity Selection Timings",
            "",
            "| population_size | target_lineups | diversity_avg_s | diversity_min_s | diversity_max_s |",
            "|---:|---:|---:|---:|---:|",
            "| {population_size:.0f} | {target_lineups:.0f} | {diversity_avg_s:.6f} | {diversity_min_s:.6f} | {diversity_max_s:.6f} |".format(
                **diversity
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pangadfs performance benchmarks")
    parser.add_argument("--population-sizes", type=int, nargs="+", default=[500, 1000])
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--diversity-population", type=int, default=1000)
    parser.add_argument("--diversity-target-lineups", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/last_benchmark_report.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("benchmarks/last_benchmark_report.md"))
    args = parser.parse_args()

    generation_rows = benchmark_generation_and_validation(
        population_sizes=args.population_sizes,
        n_generations=args.generations,
        repeats=args.repeats,
    )

    diversity_row = benchmark_diversity_selection(
        population_size=args.diversity_population,
        n_generations=args.generations,
        repeats=args.repeats,
        target_lineups=args.diversity_target_lineups,
    )

    report = {
        "generation_validation": generation_rows,
        "diversity_selection": diversity_row,
        "config": {
            "population_sizes": args.population_sizes,
            "generations": args.generations,
            "repeats": args.repeats,
            "diversity_population": args.diversity_population,
            "diversity_target_lineups": args.diversity_target_lineups,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_markdown.write_text(_render_markdown(generation_rows, diversity_row), encoding="utf-8")

    print("Benchmark completed")
    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.output_markdown}")


if __name__ == "__main__":
    main()
