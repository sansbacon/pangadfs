# pangadfs Refactoring Plan

Tracking document for simplification, bug fixes, and performance improvements.

---

## 1. Bugs (Critical)

- [x] **`misc.py:97` — `population.flatten` missing parentheses**
  - `flat = population.flatten` assigns the method object, not the result.
  - Fix: `flat = population.flatten()`

- [x] **`ga.py:116` — `crossover()` calls `ext.obj.mutate()`**
  - Should be `ext.obj.crossover(**params, **kwargs)`

- [x] **`ga.py:299` — `populate()` iterates wrong extension manager**
  - Uses `self.extension_managers['crossover']` — should be `'populate'`

- [x] **`ga.py:259` — `agg: bool = 'False'` (string default)**
  - The string `'False'` is truthy. Fix: `agg: bool = False`

- [x] **`misc.py:158` — `@njit` used unconditionally after conditional import**
  - `from numba import njit` is in a `try/except` that passes on `ModuleNotFoundError`
  - But `@njit` on `_generate_shifted_indices` will raise `NameError` if numba is absent
  - Fix: wrap the decorated function in a conditional, or provide a fallback

- [x] **`crossover.py:30-32` — `_diverse()` shape mismatch**
  - `np.where(choice, diversity_matrix, cxidx)` — `diversity_matrix` is (N,N), `cxidx` is (N,)
  - This method likely never produces correct output
  - Fix: rethink the diverse crossover logic entirely

- [x] **`optimize_multioptimizer_field_ownership.py` — imports nonexistent modules**
  - `PopulateMultilineupSetsOptimized`, `CrossoverMultilineupSetsOptimized`, `MutateMultilineupSetsOptimized`
  - These modules don't exist in the repo — will crash at runtime
  - Fix: implement them or remove the optimizer

---

## 2. Performance

- [x] **Vectorize `PositionValidate`**
  - Replaced loop-based + "Optimized" dual classes with single vectorized implementation
  - Uses integer-encoded position array + bincount for O(1) per-player lookup, fully vectorized mask

- [x] **Vectorize `FitnessMultiOptimizerFieldOwnership._calculate_diversity`**
  - Current: nested Python loops over all lineup pairs — O(n²) with set operations
  - Action: use one-hot encoded overlap matrix (pattern already in `diversity_optimized`)

- [x] **Consolidate `multidimensional_shifting` to single fast version**
  - Three variants exist: original, `_fast`, `_numba`
  - Original calls `.to_numpy()` each time and uses float64
  - Action: make `_fast` (float32, keepdims) the default; gate numba behind availability check

- [x] **Replace `numpy_indexed.unique` in `DuplicatesValidate`**
  - `np.unique(arr, axis=0)` does the same thing natively (NumPy ≥ 1.13)
  - Removes external dependency

- [x] **`diversity()` in misc.py — use `diversity_optimized` instead**
  - Original creates a boolean 3D broadcast array; optimized uses `np.add.at` + dot product
  - Action: remove `diversity()`, rename `diversity_optimized` → `diversity`

---

## 3. Simplification & Refactoring

- [x] **Collapse `base.py` boilerplate**
  - Added `PluginBase` parent class with shared `__init__` (logging NullHandler configured once)
  - 9 subclasses now just declare their `@abstractmethod` — no duplicated `__init__`
  - Fixed incorrect docstring on `MutateBase` (was "crossover plugins")

- [x] **Remove `locals().copy()` pattern from `ga.py`**
  - Used in every method (pool, pospool, populate, fitness, crossover, mutate, select, validate)
  - Fragile: must remember to pop `self` and `kwargs`; leaks internal state if missed
  - Replace with explicit keyword forwarding

- [x] **Replace bare `except:` clauses**
  - At least 12 occurrences across `ga.py`
  - Swallows `TypeError`, `KeyError`, `KeyboardInterrupt` silently
  - Fix: `except Exception as e:` with `logging.exception(...)` or at minimum `logging.warning(...)`

- [x] **Remove `penalty.py::HighOwnershipPenalty`**
  - Body is `pass` with a TODO comment — dead code

- [x] **Consider replacing stevedore plugin system**
  - Adds complexity (entry_points in setup.py, DriverManager/NamedExtensionManager indirection)
  - For a single-developer project, plain dependency injection (pass strategy callables to `__init__`) is simpler
  - Eliminates `setup.py` entry-point registration and simplifies testing
  - Implemented hybrid approach: direct DI plugin dispatch now supported; stevedore remains backward-compatible fallback

---

## 4. Structural Improvements

- [x] **Extract `misc.py` into focused modules**
  - Created `sampling.py` — `multidimensional_shifting` (consolidated fast+numba), `parents`
  - Created `metrics.py` — `diversity`, `exposure`, `calculate_jaccard_diversity`
  - `misc.py` is now a thin re-export shim for backward compatibility

- [x] **Extract generation loop from `OptimizeDefault.optimize`**
  - Method is ~180 lines of procedural logic
  - Extract `_run_generation()` helper for readability and testability

- [x] **Populate `__init__.py` with public API**
  - Currently empty
  - Add `__all__` and re-export: `from pangadfs.ga import GeneticAlgorithm` etc.
  - Consumers should `from pangadfs import GeneticAlgorithm`

- [x] **Normalize line endings**
  - `ga.py` uses `\r\n`, other files use `\n`
  - Add `.gitattributes` with `* text=auto`
  - Verified tracked `.py/.md/.yml/.txt` files contain no CRLF

---

## 5. Minor / Style

- [x] Add return type annotations to all public methods
  - Added to all 9 abstract methods in `base.py` (np.ndarray, pd.DataFrame, Dict as appropriate)
  - Fixed missing annotation on `FitnessDefault.fitness`
  - All other implementations already had annotations
- [x] Remove unused import `numpy.testing as npt` in `misc.py`
  - `misc.py` is now a 2-line re-export shim — no direct imports remain
- [x] Remove unused import `scipy.stats.chisquare` in `misc.py`
  - Same — removed when misc.py was rewritten
- [x] Fix docstring in `MutateBase` (says "Base class for crossover plugins")
  - Fixed to "Base class for mutate plugins" in the earlier base.py refactoring
- [x] Remove commented-out code in `penalty.py` line 111
  - Removed when `HighOwnershipPenalty` was deleted

---

## Implementation Order (suggested)

1. Fix all bugs in §1 (low effort, high impact — things are currently broken)
2. Replace bare `except:` clauses (unlocks ability to debug everything else)
3. Performance items (vectorize validators, consolidate sampling)
4. Structural refactoring (split misc.py, collapse base.py)
5. Architectural decisions (stevedore replacement — only if desired)
