
::: pangadfs.optimize

## OptimizeMultiOptimizerFieldOwnership

The `OptimizeMultiOptimizerFieldOwnership` class is an advanced multi-optimizer that balances three key objectives:
-   **Score optimization**: Prioritizes both the best-performing lineups and the overall quality of the entire lineup set.
-   **Diversity**: Ensures that the generated lineups are different from each other, reducing overlap and increasing the chances of a unique lineup winning.
-   **Field ownership differentiation**: Provides a strategic edge by considering the projected ownership of players in the field. This can be used to create contrarian lineups that avoid popular players or to leverage low-owned players with high upside.

### Configuration

The behavior of the optimizer can be controlled through the `ga_settings` in the context object.

| Setting                      | Description                                                                                                | Default        |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------- |
| `target_lineups`             | The number of lineups to generate.                                                                         | `10`           |
| `score_weight`               | The weight to give to the score component of the fitness function.                                         | `0.5`          |
| `diversity_weight`           | The weight to give to the diversity component of the fitness function.                                     | `0.3`          |
| `field_ownership_weight`     | The weight to give to the field ownership component of the fitness function.                               | `0.2`          |
| `field_ownership_strategy`   | The strategy to use for field ownership differentiation. Can be `contrarian`, `leverage`, or `balanced`.     | `'contrarian'` |
| `ownership_column`           | The name of the column in the player pool CSV that contains the ownership data.                            | `'ownership'`  |
| `top_k_focus`                | The number of top lineups to prioritize in the score component.                                            | `min(10, target_lineups)` |

### Example

```python
ctx = {
    'ga_settings': {
        'target_lineups': 20,
        'score_weight': 0.5,
        'diversity_weight': 0.3,
        'field_ownership_weight': 0.2,
        'field_ownership_strategy': 'contrarian',
        'ownership_column': 'ownership',
        'top_k_focus': 5,
        # ... other GA settings
    }
}
=======
# Optimize Reference

The optimize module provides different optimization strategies for lineup generation.

## Available Optimizers

### OptimizeDefault
The standard single-lineup optimizer that uses a genetic algorithm to find the best possible lineup.

**Use case**: When you need one optimal lineup.

**Key features**:
- Fast convergence to optimal solution
- Proven genetic algorithm implementation
- Suitable for most single-lineup scenarios

### OptimizeMultilineup
Advanced optimizer for generating multiple diverse lineups using a post-processing approach.

**Use case**: When you need multiple diverse lineups (50-150 lineups).

**Key features**:
- Quality-first approach: Runs standard GA first, then selects diverse lineups
- Aggressive diversity selection with configurable thresholds
- Scalable for large lineup sets
- Comprehensive diversity metrics

**Configuration parameters**:
- `target_lineups`: Number of lineups to generate (default: 1)
- `diversity_weight`: Weight for diversity vs quality (default: 0.2)
- `min_overlap_threshold`: Minimum diversity requirement (default: 0.4)
- `diversity_method`: 'jaccard' or 'hamming' similarity (default: 'jaccard')

**Example usage**:
```python
from pangadfs.optimize import OptimizeMultilineup

# Configure for multiple diverse lineups
ga_settings = {
    'target_lineups': 100,
    'diversity_weight': 0.25,
    'min_overlap_threshold': 0.4,
    'diversity_method': 'jaccard',
    'population_size': 1000,
    'n_generations': 150,
    # ... other GA settings
}

optimizer = OptimizeMultilineup()
ga = GeneticAlgorithm(ctx={'ga_settings': ga_settings, ...}, optimize=optimizer)
results = ga.optimize()

# Access multiple lineups
lineups = results['lineups']  # List of DataFrames
scores = results['scores']    # List of scores
diversity_metrics = results['diversity_metrics']  # Diversity statistics
```

## Removed Optimizers

The following optimizers have been removed to streamline the codebase:

- `OptimizeMultilineupSets` - Replaced by OptimizeMultilineup's superior post-processing approach
- `OptimizePoolBased` - Removed due to complexity without clear benefits
- `OptimizeMultiObjective` - Removed as academic approach not suitable for practical use

## API Reference

::: pangadfs.optimize
