# Multilineup Optimization Plugin

This document describes the new `OptimizeMultilineup` plugin that extends pangadfs to support generating multiple diverse lineups (up to 150 or more) in a single optimization run.

## Overview

The `OptimizeMultilineup` plugin implements a multi-objective optimization approach that balances lineup quality (fitness) with diversity between lineups. It uses the same genetic algorithm framework as the original optimizer but adds a post-processing step to select diverse, high-scoring lineups from the final population.

## Key Features

- **Seamless Plugin Integration**: Drop-in replacement for the default optimizer
- **Backward Compatibility**: Works in single-lineup mode identical to the original optimizer
- **Configurable Diversity**: Adjustable diversity weights and thresholds
- **Multiple Diversity Methods**: Jaccard and Hamming similarity calculations
- **Scalable**: Supports 1 to 150+ lineups efficiently
- **Rich Metrics**: Provides detailed diversity statistics

## Installation

The plugin is automatically registered when you install pangadfs. No additional installation steps required.

## Usage

### Basic Multilineup Optimization

```python
from pangadfs.ga import GeneticAlgorithm
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager

# Configuration with multilineup settings
ctx = {
    'ga_settings': {
        # Standard GA settings
        'crossover_method': 'uniform',
        'csvpth': Path('your_pool.csv'),
        'n_generations': 20,
        'population_size': 30000,
        # ... other standard settings ...
        
        # Multilineup-specific settings
        'target_lineups': 150,           # Number of lineups to generate
        'diversity_weight': 0.3,         # Weight for diversity penalty (0-1)
        'min_overlap_threshold': 0.3,    # Minimum allowed overlap between lineups
        'diversity_method': 'jaccard',   # 'jaccard' or 'hamming'
    },
    'site_settings': {
        # Standard site settings
        # ... same as before ...
    }
}

# Set up driver managers - use optimize_multilineup instead of optimize_default
dmgrs = {}
for ns in GeneticAlgorithm.PLUGIN_NAMESPACES:
    if ns == 'optimize':
        dmgrs[ns] = DriverManager(
            namespace='pangadfs.optimize', 
            name='optimize_multilineup',  # Use multilineup optimizer
            invoke_on_load=True)
    else:
        dmgrs[ns] = DriverManager(
            namespace=f'pangadfs.{ns}', 
            name=f'{ns}_default', 
            invoke_on_load=True)

# Run optimization
ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
results = ga.optimize()

# Access results
print(f"Generated {len(results['lineups'])} lineups")
print(f"Best score: {results['best_score']}")
print(f"Average overlap: {results['diversity_metrics']['avg_overlap']:.3f}")

# Iterate through all lineups
for i, (lineup, score) in enumerate(zip(results['lineups'], results['scores'])):
    print(f"Lineup {i+1}: Score {score:.2f}")
    print(lineup)
```

### Single Lineup Mode (Backward Compatibility)

```python
# Set target_lineups to 1 for single lineup mode
ctx = {
    'ga_settings': {
        # ... standard settings ...
        'target_lineups': 1,  # Single lineup mode
    },
    # ... rest of config ...
}

# Results will be identical to OptimizeDefault
results = ga.optimize()
print(results['best_lineup'])  # Works exactly as before
print(results['best_score'])   # Works exactly as before
```

## Configuration Options

### Multilineup Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_lineups` | int | 1 | Number of lineups to generate (1-150+) |
| `diversity_weight` | float | 0.3 | Weight for diversity penalty (0.0-1.0) |
| `min_overlap_threshold` | float | 0.3 | Minimum allowed overlap between lineups (0.0-1.0) |
| `diversity_method` | str | 'jaccard' | Diversity calculation method ('jaccard' or 'hamming') |

### Parameter Guidelines

- **target_lineups**: Start with 10-50 for testing, scale up to 150 for production
- **diversity_weight**: 
  - 0.0 = Pure fitness optimization (may produce similar lineups)
  - 0.5 = Balanced fitness and diversity
  - 1.0 = Maximum diversity (may sacrifice lineup quality)
- **min_overlap_threshold**: 
  - 0.1 = Very diverse lineups (few shared players)
  - 0.5 = Moderate diversity
  - 0.8 = Allow high overlap between lineups

## Result Structure

The plugin returns an enhanced result dictionary:

```python
{
    # Backward compatibility (same as OptimizeDefault)
    'population': np.ndarray,        # Final GA population
    'fitness': np.ndarray,           # Population fitness scores
    'best_lineup': pd.DataFrame,     # Single best lineup
    'best_score': float,             # Best lineup score
    
    # New multilineup results
    'lineups': List[pd.DataFrame],   # All selected lineups
    'scores': List[float],           # Corresponding scores
    'diversity_metrics': {
        'avg_overlap': float,        # Average overlap between lineups
        'min_overlap': float,        # Minimum overlap between any two lineups
        'diversity_matrix': np.ndarray  # Pairwise similarity matrix
    },
    
    # Optional profiling data
    'profiling': dict               # If profiling enabled
}
```

## Algorithm Details

### Multi-Objective Selection Process

1. **Start with Best**: Select the highest-scoring lineup from the final GA population
2. **Iterative Selection**: For each subsequent lineup:
   - Calculate selection scores: `fitness - (diversity_weight × diversity_penalty)`
   - Select the lineup with the highest selection score
   - Add to selected set and repeat
3. **Diversity Penalties**: Calculated based on overlap with already-selected lineups
4. **Early Termination**: Stops if diversity thresholds cannot be met

### Diversity Calculation Methods

**Jaccard Similarity**:
```
similarity = |intersection| / |union|
```
- Good for measuring player overlap
- Range: 0.0 (no shared players) to 1.0 (identical lineups)

**Hamming Similarity**:
```
similarity = matching_positions / total_positions
```
- Good for positional lineup structure
- Range: 0.0 (no matching positions) to 1.0 (identical lineups)

## Performance Considerations

- **Time Complexity**: O(n²) for lineup selection, where n = target_lineups
- **Memory Usage**: Stores only selected lineups, not full diversity matrices
- **Scalability**: Tested up to 150 lineups with 30,000 population size
- **Optimization**: Uses numpy operations for efficient similarity calculations

## Examples

See the included example files:
- `pangadfs/app/multilineup_app.py` - Complete usage examples
- `test_multilineup.py` - Test script demonstrating functionality

## Troubleshooting

### Common Issues

**"Could not find N diverse lineups"**
- Reduce `target_lineups` or increase `min_overlap_threshold`
- Increase GA `population_size` for more diversity
- Reduce `diversity_weight` to allow more similar lineups

**Low diversity between lineups**
- Increase `diversity_weight` (try 0.5-0.8)
- Decrease `min_overlap_threshold`
- Check that your player pool has sufficient diversity

**Poor lineup quality**
- Decrease `diversity_weight` (try 0.1-0.3)
- Increase GA generations or population size
- Verify your fitness function and constraints

### Performance Optimization

For large-scale optimization (100+ lineups):
- Use larger `population_size` (50,000+)
- Consider reducing `n_generations` if diversity is more important than absolute optimization
- Monitor memory usage with very large target_lineups

## Integration with Existing Code

The plugin is designed for seamless integration:

```python
# Change this:
dmgrs['optimize'] = DriverManager(
    namespace='pangadfs.optimize', 
    name='optimize_default',
    invoke_on_load=True)

# To this:
dmgrs['optimize'] = DriverManager(
    namespace='pangadfs.optimize', 
    name='optimize_multilineup',
    invoke_on_load=True)

# Add multilineup settings to your context
ctx['ga_settings']['target_lineups'] = 150
ctx['ga_settings']['diversity_weight'] = 0.3
```

All existing code using `results['best_lineup']` and `results['best_score']` will continue to work unchanged.
