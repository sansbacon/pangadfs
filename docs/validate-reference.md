# Validate Reference

The validate module provides validation classes to ensure lineups meet various constraints and requirements.

## Available Validators

### General Validation

#### DuplicatesValidate
Removes lineups that contain duplicate players.

**Use case**: Ensure no player appears multiple times in a lineup.

**Key features**:
- Efficient duplicate detection using sorted arrays
- Handles both internal duplicates (within lineup) and external duplicates (between lineups)
- Optimized for large populations

#### FlexDuplicatesValidate
Validates that FLEX positions don't duplicate players already used in other positions.

**Use case**: Ensure FLEX players are truly additional and don't overlap with required positions.

**Key features**:
- Vectorized approach for efficiency
- Handles complex position mapping scenarios
- Prevents invalid lineups where FLEX duplicates required positions

#### SalaryValidate
Ensures all lineups meet salary cap constraints.

**Use case**: Filter out lineups that exceed the salary cap.

**Key features**:
- Fast salary calculation using numpy operations
- Configurable salary cap
- Efficient boolean indexing for filtering

### Position Validation

#### PositionValidate
Validates that lineups meet position requirements (QB, RB, WR, TE, etc.).

**Use case**: Ensure lineups have the correct number of players at each position.

**Key features**:
- Flexible position mapping support
- FLEX position handling
- Comprehensive position requirement checking

**Parameters**:
- `posmap`: Position requirements (e.g., {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1})
- `position_column`: Column name for positions in player pool
- `flex_positions`: Positions that can fill FLEX slots (default: ('RB', 'WR', 'TE'))

#### PositionValidateOptimized
Optimized version of position validation using vectorized operations.

**Use case**: Same as PositionValidate but with better performance for large populations.

**Key features**:
- Vectorized validation for better performance
- Pre-computed position arrays for fast lookup
- Same functionality as PositionValidate with speed improvements

## Usage Examples

### Basic Validation Setup
```python
from pangadfs.validate import DuplicatesValidate, SalaryValidate, PositionValidate

# Set up validators
validators = [
    DuplicatesValidate(),
    SalaryValidate(),
    PositionValidate()
]

# Apply validation in sequence
for validator in validators:
    population = validator.validate(
        population=population,
        salaries=salaries,
        salary_cap=50000,
        pool=player_pool,
        posmap={'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1},
        position_column='pos',
        flex_positions=('RB', 'WR', 'TE')
    )
```

### Using with GeneticAlgorithm
```python
from stevedore.named import NamedExtensionManager

# Set up validation extension manager
emgrs = {
    'validate': NamedExtensionManager(
        namespace='pangadfs.validate',
        names=['validate_salary', 'validate_duplicates', 'validate_positions'],
        invoke_on_load=True,
        name_order=True
    )
}

ga = GeneticAlgorithm(ctx=ctx, extension_managers=emgrs)
```

## Module Consolidation

**Note**: All validation classes are now consolidated in the single `pangadfs.validate` module. Previously, position validation classes were in a separate `validate_positions` module, but they have been moved for better organization and easier imports.

**Migration**: If you were previously importing from `pangadfs.validate_positions`, simply change your imports to use `pangadfs.validate`:

```python
# Old (no longer works)
from pangadfs.validate_positions import PositionValidate

# New (current)
from pangadfs.validate import PositionValidate
```

## API Reference

::: pangadfs.validate
