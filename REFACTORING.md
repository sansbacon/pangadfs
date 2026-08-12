# pangadfs Refactoring Plan

Tracking document for simplification, bug fixes, and performance improvements.

---

## 1. Bugs (Critical)

- [x] **`misc.py:97` — `population.flatten` missing parentheses**
  - `flat = population.flatten` assigns the method object, not the result.
  - Fix: `flat = population.flatten()`

- [ ] **`ga.py:116` — `crossover()` calls `ext.obj.mutate()`**
  - Should be `ext.obj.crossover(**params, **kwargs)`

- [ ] **`ga.py:299` — `populate()` iterates wrong extension manager**
  - Uses `self.extension_managers['crossover']` — should be `'populate'`

- [ ] **`ga.py:259` — `agg: bool = 'False'` (string default)**
  - The string `'False'` is truthy. Fix: `agg: bool = False`

- [ ] **`misc.py:158` — `@njit` used unconditionally after conditional import**
  - `from numba import njit` is in a `try/except` that passes on `ModuleNotFoundError`
  - But `@njit` on `_generate_shifted_indices` will raise `NameError` if numba is absent
  - Fix: wrap the decorated function in a conditional, or provide a fallback

- [ ] **`crossover.py:30-32` — `_diverse()` shape mismatch**
  - `np.where(choice, diversity_matrix, cxidx)` — `diversity_matrix` is (N,N), `cxidx` is (N,)
  - This method likely never produces correct output
  - Fix: rethink the diverse crossover logic entirely

- [ ] **`optimize_multioptimizer_field_ownership.py` — imports nonexistent modules**
  - `PopulateMultilineupSetsOptimized`, `CrossoverMultilineupSetsOptimized`, `MutateMultilineupSetsOptimized`
  - These modules don't exist in the repo — will crash at runtime
  - Fix: implement them or remove the optimizer

---

## 2. Performance

- [ ] **Vectorize `PositionValidate`**
  - Current: pure Python loop over every lineup in population
  - `PositionValidateOptimized` exists but isn't wired as default
  - Action: make vectorized version the only implementation, remove the loop-based one

- [ ] **Vectorize `FitnessMultiOptimizerFieldOwnership._calculate_diversity`**
  - Current: nested Python loops over all lineup pairs — O(n²) with set operations
  - Action: use one-hot encoded overlap matrix (pattern already in `diversity_optimized`)

- [ ] **Consolidate `multidimensional_shifting` to single fast version**
  - Three variants exist: original, `_fast`, `_numba`
  - Original calls `.to_numpy()` each time and uses float64
  - Action: make `_fast` (float32, keepdims) the default; gate numba behind availability check

- [ ] **Replace `numpy_indexed.unique` in `DuplicatesValidate`**
  - `np.unique(arr, axis=0)` does the same thing natively (NumPy ≥ 1.13)
  - Removes external dependency

- [ ] **`diversity()` in misc.py — use `diversity_optimized` instead**
  - Original creates a boolean 3D broadcast array; optimized uses `np.add.at` + dot product
  - Action: remove `diversity()`, rename `diversity_optimized` → `diversity`

---

## 3. Simplification & Refactoring

- [ ] **Collapse `base.py` boilerplate**
  - 9 classes that are identical except method name
  - Option A: Single `PluginBase` with `__init_subclass__` that auto-creates the abstract method
  - Option B: Use `typing.Protocol` (no inheritance required, duck-typing)
  - Either way, remove the duplicated `logging.getLogger(__name__).addHandler(logging.NullHandler())` × 9

- [ ] **Remove `locals().copy()` pattern from `ga.py`**
  - Used in every method (pool, pospool, populate, fitness, crossover, mutate, select, validate)
  - Fragile: must remember to pop `self` and `kwargs`; leaks internal state if missed
  - Replace with explicit keyword forwarding

- [ ] **Replace bare `except:` clauses**
  - At least 12 occurrences across `ga.py`
  - Swallows `TypeError`, `KeyError`, `KeyboardInterrupt` silently
  - Fix: `except Exception as e:` with `logging.exception(...)` or at minimum `logging.warning(...)`

- [ ] **Remove `penalty.py::HighOwnershipPenalty`**
  - Body is `pass` with a TODO comment — dead code

- [ ] **Consider replacing stevedore plugin system**
  - Adds complexity (entry_points in setup.py, DriverManager/NamedExtensionManager indirection)
  - For a single-developer project, plain dependency injection (pass strategy callables to `__init__`) is simpler
  - Eliminates `setup.py` entry-point registration and simplifies testing
  - This is a larger architectural decision — evaluate ROI before committing

---

## 4. Structural Improvements

- [ ] **Extract `misc.py` into focused modules**
  - `sampling.py` — `multidimensional_shifting`, `parents`
  - `metrics.py` — `diversity`, `exposure`, `calculate_jaccard_diversity`
  - `misc.py` is currently a grab-bag

- [ ] **Extract generation loop from `OptimizeDefault.optimize`**
  - Method is ~180 lines of procedural logic
  - Extract `_run_generation()` helper for readability and testability

- [ ] **Populate `__init__.py` with public API**
  - Currently empty
  - Add `__all__` and re-export: `from pangadfs.ga import GeneticAlgorithm` etc.
  - Consumers should `from pangadfs import GeneticAlgorithm`

- [ ] **Normalize line endings**
  - `ga.py` uses `\r\n`, other files use `\n`
  - Add `.gitattributes` with `* text=auto`

---

## 5. Minor / Style

- [ ] Add return type annotations to all public methods
- [ ] Remove unused import `numpy.testing as npt` in `misc.py`
- [ ] Remove unused import `scipy.stats.chisquare` in `misc.py`
- [ ] Fix docstring in `MutateBase` (says "Base class for crossover plugins")
- [ ] Remove commented-out code in `penalty.py` line 111

---

## Implementation Order (suggested)

1. Fix all bugs in §1 (low effort, high impact — things are currently broken)
2. Replace bare `except:` clauses (unlocks ability to debug everything else)
3. Performance items (vectorize validators, consolidate sampling)
4. Structural refactoring (split misc.py, collapse base.py)
5. Architectural decisions (stevedore replacement — only if desired)
