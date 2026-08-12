# Performance Improvement Suggestions for pangadfs

## Implementation progress

- [x] Started the first optimization pass
- [x] Replaced the slow Python-heavy lineup validation path with a faster lookup-based NumPy implementation in [pangadfs/validate.py](pangadfs/validate.py)
- [x] Fixed validator regressions and restored full GA test stability
- [x] Precompute and reuse static arrays in the optimization loop in [pangadfs/optimize.py](pangadfs/optimize.py)
- [x] Simplify the multilineup diversity comparisons in [pangadfs/optimize.py](pangadfs/optimize.py)
- [x] Add benchmark coverage for validation and generation timings

## Verification status

- Environment now configured with conda interpreter at `C:\Users\EricTruett\miniconda3\envs\dbxconnect\python.exe`
- Syntax check passed: `python -m compileall pangadfs/optimize.py`
- Optimizer-adjacent GA tests passed: `python -m pytest tests/test_ga.py -q -k "not test_validate and not test_position_validate_flex_rules"` → `8 passed, 2 deselected`
- Full GA validation now passes: `python -m pytest tests/test_ga.py -q` → `9 passed`
- Benchmark suite implemented: `benchmarks/run_benchmarks.py`
- Benchmark run verified with: `python benchmarks/run_benchmarks.py --population-sizes 300 600 --generations 8 --repeats 1 --diversity-population 600 --diversity-target-lineups 10`
- Benchmark artifacts generated:
	- `benchmarks/last_benchmark_report.json`
	- `benchmarks/last_benchmark_report.md`
- Additional hotspot profiler implemented: `benchmarks/profile_hotspots.py`
- Hotspot profiling run verified with: `python benchmarks/profile_hotspots.py --population-size 600 --generations 6 --top 20`
- Hotspot profiling artifacts generated:
	- `benchmarks/hotspots_summary.json`
	- `benchmarks/hotspots_optimize_default.prof`
	- `benchmarks/hotspots_optimize_default.txt`
	- `benchmarks/hotspots_optimize_multilineup.prof`
	- `benchmarks/hotspots_optimize_multilineup.txt`
- Added validate-kernel benchmark script: `benchmarks/benchmark_validate_kernel.py`
- Validate-kernel benchmark run verified with: `python benchmarks/benchmark_validate_kernel.py --population-size 4000 --repeats 3`
- Validate-kernel benchmark artifacts generated:
	- `benchmarks/validate_kernel_benchmark.json`
	- `benchmarks/validate_kernel_benchmark.md`
- Numba acceleration applied to the multilineup Jaccard hotspot path in `pangadfs/optimize.py` (with Python fallback when Numba is unavailable)
- Numba acceleration applied to the position-validation kernel in `pangadfs/validate.py` (with Python fallback when Numba is unavailable)
- Post-Numba correctness verified: `python -m pytest tests/test_ga.py -q` → `9 passed`
- Post-Numba profiler run verified: `python benchmarks/profile_hotspots.py --population-size 600 --generations 6 --top 20`
- Post-Numba benchmark suite re-run verified: `python benchmarks/run_benchmarks.py --population-sizes 300 600 --generations 8 --repeats 1 --diversity-population 600 --diversity-target-lineups 10`

## Summary

The main performance bottlenecks are in the genetic algorithm loop and in validation logic that is repeatedly executed across generations. The library already uses NumPy in several places, which is a good foundation, but the current implementation still performs a significant amount of Python-level work inside the hot path.

The highest-impact opportunities are:

- vectorizing lineup validation
- reducing repeated DataFrame lookups
- simplifying diversity selection logic
- adding numba acceleration where the workloads are highly repetitive

---

## 1. Optimize validation in the hot path

### Issue

The validation code in [pangadfs/validate.py](pangadfs/validate.py) includes Python loops over every lineup and then nested loops over individual players. This is expensive when population sizes are large or when many generations are run.

### Current examples

- `PositionValidate.validate` loops through each lineup in the population
- `_is_lineup_valid` loops over each player in the lineup
- `FlexDuplicatesValidate.validate` also performs row-by-row logic

### Recommendation

Replace the Python loops with vectorized NumPy operations wherever possible.

Suggested approach:

- Precompute a position lookup array for all player IDs
- For each lineup, extract the player positions in one vectorized step
- Count positions using `np.bincount` or a small set of integer arrays
- Filter valid lineups with a boolean mask

This preserves correctness while moving the work out of Python and into NumPy kernels.

### Why it matters

This is executed every generation, so even a modest per-lineup reduction compounds quickly across hundreds or thousands of generations.

---

## 2. Precompute lookup arrays once and reuse them

### Issue

In the optimization loop, static values such as points, salaries, and player positions are fetched repeatedly from the DataFrame.

### Current pattern

In [pangadfs/optimize.py](pangadfs/optimize.py), arrays like:

- `points`
- `salaries`
- position information

are computed and then used repeatedly in the GA lifecycle.

### Recommendation

At the start of optimization, build these arrays once and pass them through the GA pipeline instead of repeatedly extracting them from the DataFrame inside the generation loop.

Example improvements:

- `points = pool[points_column].to_numpy()`
- `salaries = pool[salary_column].to_numpy()`
- `position_by_player = pool[position_column].to_numpy()`

These arrays can be reused in validation and fitness functions without repeated Pandas indexing overhead.

### Why it matters

This avoids repeated object creation and DataFrame access overhead in the highest-frequency code path.

---

## 3. Reduce nested diversity calculations

### Issue

The multilineup selection logic in [pangadfs/optimize.py](pangadfs/optimize.py) compares each candidate lineup against all selected lineups using nested loops. This becomes expensive as the number of selected lineups or candidates grows.

### Current behavior

Functions such as:

- `_select_diverse_lineups`
- `_calculate_diversity_penalties`
- `_calculate_diversity_metrics`

perform repeated set-based comparisons and pairwise overlap checks in Python.

### Recommendation

Use one or more of the following strategies:

- compute overlap scores once for the current population and reuse them
- select only a subset of candidate lineups for pairwise comparison rather than the full population
- replace set-based Jaccard operations with faster integer-array or bitset-based comparisons
- approximate diversity with a cheaper metric when exact diversity is not required

### Why it matters

This is a quadratic-style workload, and it becomes the dominant cost when the target lineup count or the population size increases.

---

## 4. Move the repeated numeric kernels to Numba

### Issue

The numeric routines in validation, sampling, and selection are well-suited to low-level compiled execution but still involve Python overhead in hot loops.

### Best candidates for Numba

- lineup validation logic in [pangadfs/validate.py](pangadfs/validate.py)
- duplicate checks for large arrays
- candidate scoring and selection operations where loops are repeated across population rows

### Recommendation

Consider `numba.njit` for the most repetitive validation and scoring routines, especially if the project is used on larger fantasy datasets or with many iterations.

This can significantly reduce execution time without changing the overall GA design.

### Why it matters

The most expensive operations are arithmetic and indexing-heavy loops, which Numba handles well.

---

## 5. Keep the GA core in NumPy arrays as long as possible

### Issue

The code mixes NumPy arrays and Pandas DataFrames in a way that introduces conversion and indexing overhead in the optimization loop.

### Recommendation

Keep players represented by integer indices and numeric arrays for most of the GA runtime. Only convert back to a pandas DataFrame when producing the final result output.

This reduces:

- DataFrame indexing overhead
- object conversion overhead
- repeated Python dispatch costs

### Why it matters

The GA spends most of its time evolving arrays, not formatting final outputs, so the internal representation should be optimized for speed.

---

## 6. Revisit the sampling implementation for large-scale workloads

### Issue

The sampling code in [pangadfs/sampling.py](pangadfs/sampling.py) is already fairly optimized, but it still allocates temporary arrays for each generated population.

### Recommendation

For very large population sizes, consider:

- batching random sampling
- reducing temporary allocations
- reusing arrays when possible
- checking if a lower-precision dtype is safe for the target workload

### Why it matters

Population generation is not the worst bottleneck, but it adds overhead when multiplied across many generations and positions.

---

## 7. Add lightweight benchmarks to guide optimization

### Issue

Without benchmarks, it is difficult to know which part of the algorithm is actually expensive in real workloads.

### Recommendation

Add a small benchmark suite that measures:

- initial population build time
- validation time per generation
- fitness evaluation time per generation
- diversity selection time
- total optimization time for representative population sizes

This will make it much easier to confirm whether a change is actually improving performance.

### Why it matters

Performance tuning should be evidence-driven. A benchmark suite helps verify whether a change helps in the real hot path rather than just in theory.

---

## 8. Keep the logic simple where the bottleneck is not the algorithm itself

### Issue

Some parts of the code are valid but more elaborate than necessary for the actual use case. The code is often clear, but not always minimal.

### Recommendation

Prefer simpler, more direct numeric operations when they reduce Python overhead. For example:

- avoid repeated dictionary conversions in inner loops
- avoid repeated object creation inside generation loops
- avoid complex nested set-based comparisons unless they are truly needed for diversity quality

### Why it matters

The most important speedups often come from reducing Python overhead rather than changing the overall GA approach.

---

## Priority ranking

### Highest priority

1. Vectorize `PositionValidate`
2. Precompute static arrays and lookups
3. Simplify diversity selection logic
4. Add Numba to validation-heavy kernels

### Medium priority

5. Reduce temporary allocations in sampling
6. Add benchmark coverage

### Lower priority

7. Minor cleanup of object-heavy operations
8. Additional general code simplification

---

## Final recommendation

The best return on effort is to focus on the validation and diversity steps in the optimization loop. These are the paths that are repeated the most and currently do the most Python-level work. If those are optimized first, the overall runtime should improve substantially without changing the GA design or output quality.
