# Default Plugins

pangadfs is motivated by my experience with other optimziers, which typically have a monolithic design and don't make it easy to swap out components. pangadfs uses the [stevedore plugin system](https://docs.openstack.org/stevedore/latest/ "Stevedore plugins") to allow applications to customize one or more of the internal components. There are default implementations of each plugin namespace, which provides a fully-functional implementation of a genetic algorithm. Other plugins can replace any or all of the defaults.

As recommended by [the stevedore documentation](https://docs.openstack.org/stevedore/latest/user/tutorial/creating_plugins.html#a-plugin-base-class "Stevedore documentation"), the [base module](base-reference.md) includes base classes to define each pluggable component. Plugins may, but are not required to subclass these base classes.

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


## Default Optimize

Handles the basic optimization loop, coordinates pieces of genetic algorithm.