# Plugin Namespaces

The plugin namespaces are specified using the [entry_points](https://packaging.python.org/specifications/entry-points/ "Entry Points") section in the setup.py file. Currently, there are entry points for pool, pospool, populate, fitness, crossover, mutate, and validate.

Pangadfs installs default plugins for each namespace. Other plugins can be installed and loaded by the application.

## pool

Pool plugins create a dataframe that must contain, at a minimum, columns for position (str), salary (int), and points (float) with no missing values. It can have as many other columns as desired.

The index must be unique. DefaultPool uses an RangeIndex, which works well, but a specific index type is not required. 

## pospool

Pospool plugins create a dict. The keys are positions ('QB', 'WR', etc.). The values are dataframes of all of the players that are eligible for the position. DefaultPospool maintains the index from DefaultPool and has columns for salary (int), points (float), and prob (float). Prob is the normalized probability of points per dollar, defined as (points / salary * 1000) / sum of points per dollar.

There is no requirment to use weighted probabilities but it is highly advisable. The genetic algorithm will converge on the optimal solution quicker when the initial population is fitter. 

## populate

Populate plugins create the initial population of lineups from the pospool. DefaultPopulate creates an ndarray of indices to rows in the pool. Each row in the 2D array has lineup_size elemens (indices for each player in a lineup, 9 for DK). The number of rows is set by the intial_size parameter.

## fitness

Fitness plugins
Creates 1D array of fitness scores.
Right now, can only run fitness once but can use any function to generate single value.


## crossover
Generates new population from the initial population.
Can run multiple crossover plugins
If pass agg=True, will aggregate the results into one population.
Otherwise, runs on crossed-over population


## mutate
Generates new population from the initial population.
Can run multiple mutate plugins
If pass agg=True, will aggregate the results into one population.
Otherwise, second call runs on first mutated population, and so forth       


## validate
Generates new population from the initial population.
Can run multiple validate plugins
Each subsequent call runs on prior validated population


