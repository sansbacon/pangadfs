# Introduction

pangadfs is a pandas-based genetic algorithm for NFL daily fantasy optimization. It uses a plugin architecture to enable maximum flexibility while also providing a fully-functional implementation of a genetic algorithm.

Here is a simple example of using pangadfs to optimize a DraftKings NFL lineup:

```
from pathlib import Path
from pangadfs import GeneticAlgorithm


ctx = {
    'ga_settings': {
        'n_generations': 20,
        'population_size': 5000,
        'stop_criteria': 5,
        'points_column': 'proj',
        'salary_column': 'salary',
        'position_column': 'pos',
        'csvpth': Path(__file__).parent / 'pool.csv'
    },

    'site_settings': {
        'salary_cap': 50000,
        'posmap': {'DST': 1, 'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 7},
        'lineup_size': 9,
        'posthresh': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 6, 'DST': 4, 'FLEX': 8}
    }
}

# set up GeneticAlgorithm object
ga = GeneticAlgorithm(ctx=ctx)
ga.optimize(verbose=True)

```