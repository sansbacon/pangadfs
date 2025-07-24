# Release Notes

## 0.3.0 (Upcoming)

**Major Refactoring:**

* **Multilineup Optimization Streamlined**: Removed inferior multilineup approaches and focused on the proven `OptimizeMultilineup` post-processing method
  - Removed `OptimizeMultilineupSets`, `OptimizePoolBased`, and `OptimizeMultiObjective` classes
  - Enhanced `OptimizeMultilineup` with aggressive diversity selection for better quality/diversity balance
  - Simplified configuration with fewer parameters to tune
  - ~50% reduction in multilineup-related code complexity

* **Validation Module Consolidated**: Merged position validation functionality into main validate module
  - Moved `PositionValidate` and `PositionValidateOptimized` from `validate_positions.py` to `validate.py`
  - Removed `validate_positions.py` file
  - All validation classes now available from single `pangadfs.validate` import
  - Improved code organization and developer experience

**Performance Improvements:**

* **Optimized Multilineup Selection**: Enhanced diversity algorithm with progressive threshold relaxation
* **Streamlined Codebase**: Removed unused optimization approaches reducing overhead
* **Better Memory Usage**: Simplified algorithms use less memory for large lineup sets

**API Changes:**

* **Simplified Imports**: All validation classes now imported from `pangadfs.validate`
* **Enhanced OptimizeMultilineup**: New parameters for better diversity control
  - `min_overlap_threshold`: More aggressive diversity requirements (default: 0.4)
  - Improved diversity metrics and reporting
* **Removed Classes**: `OptimizeMultilineupSets`, `OptimizePoolBased`, `OptimizeMultiObjective` no longer available

**Documentation Updates:**

* Updated README.md with multilineup optimization examples
* Enhanced optimize-reference.md with detailed OptimizeMultilineup documentation
* Updated validate-reference.md to reflect module consolidation
* Added migration notes for validation imports

**Breaking Changes:**

* `OptimizeMultilineupSets`, `OptimizePoolBased`, and `OptimizeMultiObjective` classes removed
* `pangadfs.validate_positions` module removed - use `pangadfs.validate` instead
* Some multilineup configuration parameters changed for better defaults

**Migration Guide:**

```python
# Old validation imports (no longer work)
from pangadfs.validate_positions import PositionValidate

# New validation imports
from pangadfs.validate import PositionValidate

# Old multilineup optimization (no longer available)
from pangadfs.optimize import OptimizeMultilineupSets

# New multilineup optimization (recommended)
from pangadfs.optimize import OptimizeMultilineup
```

## 0.2.0

**Major Changes:**

* **GUI Separation**: The GUI application has been moved to a separate repository ([pangadfs-gui](https://github.com/sansbacon/pangadfs-gui)) for better modularity and maintenance
* **Reduced Dependencies**: Core library now has minimal dependencies, with GUI-specific dependencies moved to the separate package
* **Focused Core**: The main pangadfs package now focuses purely on the genetic algorithm optimization engine

**Breaking Changes:**

* Removed `pangadfs-gui` console command from core package
* Removed GUI-related dependencies from core requirements
* GUI functionality now requires separate installation: `pip install pangadfs-gui`

**Installation Changes:**

* Core library: `pip install pangadfs`
* GUI application: `pip install pangadfs-gui` (requires core library)

## 0.1.1

Updated documentation structure

## 0.1.0

* Feature Description

Working version of basic application
