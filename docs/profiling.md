# Genetic Algorithm Profiling

The pangadfs genetic algorithm now includes comprehensive profiling capabilities to help you analyze performance and identify optimization bottlenecks.

## Overview

The profiling system tracks timing for each component of the genetic algorithm and provides detailed performance analysis including:

- **Component-level timing**: Individual timing for each GA operation (selection, crossover, mutation, etc.)
- **Generation-level metrics**: Per-generation timing and convergence tracking
- **Convergence analysis**: When the optimal solution was found during the optimization process
- **Performance insights**: Automatic analysis of time distribution and bottlenecks

## Enabling Profiling

To enable profiling, simply add `'enable_profiling': True` to your GA settings:

```python
ctx = {
    'ga_settings': {
        'crossover_method': 'uniform',
        'csvpth': Path('data/pool.csv'),
        'n_generations': 20,
        'population_size': 30000,
        'enable_profiling': True,  # Enable profiling
        # ... other settings
    },
    # ... site settings
}
```

## Using Profiling

### Basic Usage

```python
from pangadfs.ga import GeneticAlgorithm

# Create GA instance with profiling enabled
ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)

# Run optimization
results = ga.optimize()

# Display results
print(results['best_lineup'])
print(f'Lineup score: {results["best_score"]}')

# Display profiling results
if ga.profiler.enabled:
    ga.profiler.print_profiling_results()
```

### Programmatic Access

You can also access profiling data programmatically:

```python
# Run optimization
results = ga.optimize()

# Access profiling data from results
if 'profiling' in results:
    profiling_data = results['profiling']
    
    print(f"Total optimization time: {profiling_data['total_time']:.2f}s")
    print(f"Time to best solution: {profiling_data['time_to_best_solution']:.2f}s")
    print(f"Best solution found at generation: {profiling_data['best_solution_generation']}")
    
    # Access individual operation timings
    for op_name, op_data in profiling_data['operations'].items():
        print(f"{op_name}: {op_data['total_time']:.3f}s ({op_data['call_count']} calls)")
```

## Profiling Output

When profiling is enabled, you'll see detailed output like this:

```
============================================================
GENETIC ALGORITHM PROFILING RESULTS
============================================================

Total Optimization Time: 45.23 seconds
Setup Time: 3.45 seconds
Optimization Loop Time: 41.78 seconds
Time to Best Solution: 32.18 seconds (Generation 15)
Best Solution Found at: 71.2% of total runtime
Generations Completed: 20
Average Generation Time: 2.089 seconds

Component Timing Summary:
┌─────────────────────┬───────────┬─────────┬─────────┬─────────┬───────────┐
│ Operation           │ Total (s) │ Avg (s) │ Min (s) │ Max (s) │ Calls     │
├─────────────────────┼───────────┼─────────┼─────────┼─────────┼───────────┤
│ Fitness Evaluation  │     18.45 │   0.878 │   0.820 │   1.020 │        21 │
│ Selection           │      8.92 │   0.213 │   0.190 │   0.250 │        42 │
│ Crossover           │      6.78 │   0.339 │   0.310 │   0.380 │        20 │
│ Mutation            │      3.12 │   0.156 │   0.140 │   0.190 │        20 │
│ Pool Creation       │      2.45 │   2.450 │   2.450 │   2.450 │         1 │
│ Validation          │      1.41 │   0.067 │   0.060 │   0.090 │        21 │
│ Initial Population  │      3.21 │   3.210 │   3.210 │   3.210 │         1 │
│ Pospool             │      0.89 │   0.890 │   0.890 │   0.890 │         1 │
└─────────────────────┴───────────┴─────────┴─────────┴─────────┴───────────┘

Performance Insights:
• Fitness Evaluation consumed 40.8% of component time
• Selection consumed 19.7% of component time
• Best solution found at 71.2% of total runtime
• Fastest generation: 1.890s, Slowest: 2.340s
============================================================
```

## Profiling Data Structure

The profiling data returned in results contains:

```python
{
    'profiling_enabled': True,
    'total_time': 45.23,                    # Total optimization time
    'setup_time': 3.45,                     # Time for setup phase
    'optimization_time': 41.78,             # Time for optimization loop
    'time_to_best_solution': 32.18,         # Time when best solution found
    'best_solution_generation': 15,         # Generation when best found
    'generations_completed': 20,            # Total generations completed
    'avg_generation_time': 2.089,           # Average time per generation
    'generation_times': [2.1, 2.0, ...],   # Individual generation times
    'operations': {
        'Pool Creation': {
            'total_time': 2.45,
            'call_count': 1,
            'avg_time': 2.45,
            'min_time': 2.45,
            'max_time': 2.45,
            'times': [2.45]
        },
        # ... other operations
    }
}
```

## Performance Analysis

The profiling system helps you identify:

1. **Bottleneck Operations**: Which GA components take the most time
2. **Convergence Efficiency**: How quickly the algorithm finds good solutions
3. **Generation Consistency**: Whether generation times are stable
4. **Setup vs Optimization**: Time distribution between setup and optimization phases

## Disabling Profiling

Profiling is disabled by default. To explicitly disable it:

```python
ctx = {
    'ga_settings': {
        'enable_profiling': False,  # Explicitly disable
        # ... other settings
    }
}
```

When disabled, profiling has zero performance overhead.

## Best Practices

1. **Enable for Development**: Use profiling during development to understand performance characteristics
2. **Disable for Production**: Disable profiling in production environments for maximum performance
3. **Analyze Bottlenecks**: Focus optimization efforts on the most time-consuming operations
4. **Track Convergence**: Use time-to-best-solution metrics to evaluate algorithm efficiency
5. **Compare Configurations**: Use profiling to compare different GA parameter settings

## Example Use Cases

### Finding Optimal Population Size
```python
# Test different population sizes and compare setup vs optimization time
for pop_size in [1000, 5000, 10000, 30000]:
    ctx['ga_settings']['population_size'] = pop_size
    ctx['ga_settings']['enable_profiling'] = True
    
    ga = GeneticAlgorithm(ctx=ctx, ...)
    results = ga.optimize()
    
    profiling = results['profiling']
    print(f"Pop {pop_size}: Setup {profiling['setup_time']:.2f}s, "
          f"Optimization {profiling['optimization_time']:.2f}s")
```

### Comparing Selection Methods
```python
# Compare performance of different selection methods
for method in ['fittest', 'roulette', 'tournament']:
    ctx['ga_settings']['select_method'] = method
    ctx['ga_settings']['enable_profiling'] = True
    
    ga = GeneticAlgorithm(ctx=ctx, ...)
    results = ga.optimize()
    
    selection_time = results['profiling']['operations']['Selection']['total_time']
    print(f"{method}: {selection_time:.3f}s total selection time")
```

The profiling system provides comprehensive insights into your genetic algorithm's performance, helping you optimize both the algorithm parameters and the underlying implementation.
