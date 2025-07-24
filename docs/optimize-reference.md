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
