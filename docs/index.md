# Introduction

pangadfs is a pandas-based genetic algorithm for NFL daily fantasy optimization. It uses the [stevedore plugin system](https://docs.openstack.org/stevedore/latest/ "Stevedore plugins") to allow applications to customize one or more of the internal components. As recommended by [the stevedore documentation](https://docs.openstack.org/stevedore/latest/user/tutorial/creating_plugins.html#a-plugin-base-class "Stevedore documentation"), the [base module](base-reference.md) includes base classes to define each pluggable component. The [default module](default-reference.md) is the default implementation of each plugin, which provides a fully-functional implementation of a genetic algorithm.

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

## crossover


## mutate


## validate


# Default Plugins

This section describes the behavior of the default plugins. The architecture is designed to be flexible, so an application can use all of these plugins, none of them, or a combination of default and other plugins.

## DefaultPool
  
Reads in a csv file with columns: player, team, pos, salary, proj. Sorts players by position so indexes are sequential by position.

```
Usage:
    fn = Path.home() / 'projections.csv'
    pool = DefaultPool().pool(csvpth=fn)
```

## DefaultPospool

Creates prob (probabilities) column for weighted random sampling. Uses points per dollar to nudge initial selection to best options. While there are good reasons not to optimize solely on ppd, it is effective at creating a fit initial population.    

```
Usage:
    fn = Path.home() / 'projections.csv'
    pospool = DefaultPospool().pospool(
      pool = DefaultPool().pool(csvpth=fn)
      posfilter = {'QB': 12, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4}
    )
```

## DefaultPopulate

This creates the initial population.

## DefaultCrossover

The default approach is to take the top percentile (default is pctl = 50) of the population and divide in half into father and mother. Then generate a boolean array of the same shape as the father and mother.

Use the boolean array to generate two arrays of children:

* Child 1 takes from father on True and mother on False
* Child 2 takes from mother on True and father on False

```
Usage:
    c = DefaultCrossover()
    newpop = c.crossover(population=oldpop, population_fitness=oldpopfit, pctl=50)
```

## DefaultMutate

This mutates population acccording to the mutation rate.

## SalaryValidate

Ensures lineup does not exceed salary cap.

## DuplicatesValidate

Ensures lineup does not duplicate players in the flex.
