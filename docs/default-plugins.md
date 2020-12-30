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

Ensures lineup does not have duplicate players and that all lineups are unique.
