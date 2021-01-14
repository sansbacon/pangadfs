# pangadfs Features

pangadfs is a pandas-based genetic algorithm library for daily fantasy optimization (currently NFL only). pangadfs is extensible by design and is motivated by difficulties I encountered with other optimizers, which tend to have a monolithic design and don't make it easy to swap out components. 

This flexibility is made possible by the [stevedore plugin system](https://docs.openstack.org/stevedore/latest/ "Stevedore plugins"), which allows allow applications to customize one or more of the internal components. As recommended by [the stevedore documentation](https://docs.openstack.org/stevedore/latest/user/tutorial/creating_plugins.html#a-plugin-base-class "Stevedore documentation"), the [base module](base-reference.md) includes base classes to define each pluggable component. Each namespace has a default implementation (crossover, fitness, mutate, select, and so forth), which, collectively, provide a fully-functional implementation of a genetic algorithm.


## Optimizing with Genetic Algorithms

In principle, a population of individuals selected from the search space , often in a
random manner, serves as candidate solutions to optimize the problem [3]. The
individuals in this population are evaluated through ( "fitness" ) adaptation function.
A selection mechanism is then used to select individuals to be used as parents to
those of the next generation. These individuals will then be crossed and mutated to
form the new offspring. The next generation is finally formed by an alternative
mechanism between parents and their offspring [4]. This process is repeated until a
certain satisfaction condition.

Genetic algorithms maintain a population of candidate solutions, called individuals, for that given problem. These candidate solutions are iteratively evaluated and combined to create a new generation of solutions. Individuals with higher fitness (rated as better at solving the relevant problem) have a greater chance of being selected and passing their qualities to the next generation of candidate solutions This way, as generations go by, candidate solutions get better at solving the problem at hand.

Applied to the context of daily fantasy lineups, a genetic algorithm works as follows:

* Pool: load a pool of chromosomes (players) with position, salary, and points (projected if forward looking, actual points if historical).

* Pospool: segment the pool into positions. There will be duplicate genes (players) for multiposition eligibility / flex.

* Populate: create a population of individuals (here lineups) from a pool of genes (here player ids).

    * The initial populations are randomly created using weighted random sampling.

    * Individauls are encoded as an array of integer IDs. For example, a DK NFL individual is np.array([dst_id, qb_id, rb_id, rb_id, wr_id, wr_id, wr_id, te_id, flex_id]).

* Fitness: Assess the fitness of the population (is typically the sum of projected or actual points).

* Select: Discard low-performing individuals from the population.

* Crossover: create new individuals by randomly combining elements of the selected individuals.

* Mutate: randomly alter individuals by swapping out chromosomes.

* Validate: filter out invalid individuals (too much salary, duplicate chromosomes in individual, duplicate individuals in population, and so forth).

* Repeat for n generations (or until specified stop point, such as no improvement for 5 generations)

## Strengths and Weaknesses of Genetic Algorithms

* Individuals (lineups) are cheap
