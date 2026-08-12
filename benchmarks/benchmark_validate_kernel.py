# -*- coding: utf-8 -*-
"""Benchmark the PositionValidate kernel (Numba vs Python fallback).

Usage:
    python benchmarks/benchmark_validate_kernel.py
    python benchmarks/benchmark_validate_kernel.py --population-size 5000 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pangadfs.validate import (
    HAS_NUMBA,
    _validate_lineups_position_kernel_numba,
    _validate_lineups_position_kernel_py,
)


def _build_pool(n_players: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    positions = np.array(["QB", "RB", "WR", "TE", "DST"])
    probs = np.array([0.1, 0.25, 0.35, 0.15, 0.15])

    pos = rng.choice(positions, size=n_players, p=probs)
    salary = rng.integers(3000, 9500, size=n_players)
    proj = rng.uniform(3.0, 30.0, size=n_players)

    return pd.DataFrame(
        {
            "player": [f"P{i}" for i in range(n_players)],
            "team": ["T"] * n_players,
            "pos": pos,
            "salary": salary,
            "proj": proj,
        }
    )


def _build_population(pool: pd.DataFrame, population_size: int, lineup_size: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    player_ids = pool.index.to_numpy()
    pop = np.empty((population_size, lineup_size), dtype=np.int64)
    for i in range(population_size):
        pop[i] = rng.choice(player_ids, size=lineup_size, replace=False)
    return pop


def _prepare_kernel_inputs(
    population: np.ndarray,
    pool: pd.DataFrame,
    posmap: Dict[str, int],
    position_column: str,
    flex_positions: Tuple[str, ...],
):
    position_names = tuple(dict.fromkeys(pool[position_column].tolist()))
    position_to_code = {pos: idx for idx, pos in enumerate(position_names)}
    max_id = max(int(pool.index.max()), int(population.max())) + 1
    player_position_code = np.full(max_id, -1, dtype=np.int16)

    for player_id, pos in zip(pool.index.to_numpy(), pool[position_column].to_numpy()):
        player_position_code[player_id] = position_to_code.get(pos, -1)

    non_flex_items = [(pos, count) for pos, count in posmap.items() if pos != "FLEX"]
    required_codes = np.asarray([position_to_code.get(pos, -1) for pos, _ in non_flex_items], dtype=np.int16)
    required_counts = np.asarray([count for _, count in non_flex_items], dtype=np.int16)
    flex_required = int(posmap.get("FLEX", 0))

    flex_codes = np.asarray(
        [position_to_code.get(pos, -1) for pos in flex_positions if position_to_code.get(pos, -1) >= 0],
        dtype=np.int16,
    )

    return (
        player_position_code,
        required_codes,
        required_counts,
        flex_codes,
        flex_required,
        len(position_names),
    )


def _time_kernel(fn, repeats: int, *args) -> Dict[str, float]:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        mask = fn(*args)
        timings.append(time.perf_counter() - start)
        _ = int(mask.sum())

    return {
        "avg_s": mean(timings),
        "min_s": min(timings),
        "max_s": max(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PositionValidate kernel")
    parser.add_argument("--population-size", type=int, default=5000)
    parser.add_argument("--lineup-size", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/validate_kernel_benchmark.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("benchmarks/validate_kernel_benchmark.md"))
    args = parser.parse_args()

    posmap = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1, "FLEX": 1}
    flex_positions = ("RB", "WR", "TE")

    pool = _build_pool()
    population = _build_population(pool, args.population_size, args.lineup_size)

    (
        player_position_code,
        required_codes,
        required_counts,
        flex_codes,
        flex_required,
        n_positions,
    ) = _prepare_kernel_inputs(
        population=population,
        pool=pool,
        posmap=posmap,
        position_column="pos",
        flex_positions=flex_positions,
    )

    kernel_args = (
        population,
        player_position_code,
        required_codes,
        required_counts,
        flex_codes,
        flex_required,
        n_positions,
    )

    python_stats = _time_kernel(_validate_lineups_position_kernel_py, args.repeats, *kernel_args)

    numba_stats = None
    speedup = None
    if HAS_NUMBA:
        _ = _validate_lineups_position_kernel_numba(*kernel_args)
        numba_stats = _time_kernel(_validate_lineups_position_kernel_numba, args.repeats, *kernel_args)
        if numba_stats["avg_s"] > 0:
            speedup = python_stats["avg_s"] / numba_stats["avg_s"]

    report = {
        "config": {
            "population_size": args.population_size,
            "lineup_size": args.lineup_size,
            "repeats": args.repeats,
        },
        "python_kernel": python_stats,
        "numba_available": HAS_NUMBA,
        "numba_kernel": numba_stats,
        "speedup_x": speedup,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# PositionValidate Kernel Benchmark",
        "",
        "## Config",
        "",
        f"- population_size: {args.population_size}",
        f"- lineup_size: {args.lineup_size}",
        f"- repeats: {args.repeats}",
        "",
        "## Results",
        "",
        "| kernel | avg_s | min_s | max_s |",
        "|---|---:|---:|---:|",
        f"| python | {python_stats['avg_s']:.6f} | {python_stats['min_s']:.6f} | {python_stats['max_s']:.6f} |",
    ]

    if numba_stats is not None:
        md_lines.append(
            f"| numba | {numba_stats['avg_s']:.6f} | {numba_stats['min_s']:.6f} | {numba_stats['max_s']:.6f} |"
        )
        md_lines.append("")
        md_lines.append(f"Estimated speedup: {speedup:.2f}x")
    else:
        md_lines.append("")
        md_lines.append("Numba not available; only Python fallback was benchmarked.")

    args.output_markdown.write_text("\n".join(md_lines), encoding="utf-8")

    print("Validate kernel benchmark completed")
    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.output_markdown}")


if __name__ == "__main__":
    main()
