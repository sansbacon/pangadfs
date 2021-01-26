# pangadfs/pangadfs/optimize.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Any, Dict

import pandas as pd
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager

from pangadfs.base import OptimizeBase
from pangadfs.ga import GeneticAlgorithm


class OptimizeDefault(OptimizeBase):

    def optimize(self, 
            *,
            ctx: Dict[str, Any],
            driver_managers: Dict[str, DriverManager] = None, 
            extension_managers: Dict[str, NamedExtensionManager] = None,
            **kwargs
            ) -> Dict[str, Any]:
        """Creates initial pool
        
        Args:

            **kwargs: keyword arguments for plugins
            
        Returns:
            Tuple[pd.DataFrame]

        """
        # set up GeneticAlgorithm object
        ga = GeneticAlgorithm(
            driver_managers=driver_managers, 
            extension_managers=extension_managers
        )
        
        # create pool and pospool
        # pospool used to generate initial population
        # is a dict of position_name: DataFrame
        pop_size = ctx['ga_settings']['population_size']
        pool = ga.pool(csvpth=ctx['ga_settings']['csvpth'])
        cmap = {'points': ctx['ga_settings']['points_column'],
                'position': ctx['ga_settings']['position_column'],
                'salary': ctx['ga_settings']['salary_column']}
        posfilter = ctx['site_settings']['posfilter']
        flex_positions = ctx['site_settings']['flex_positions']
        pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping=cmap, flex_positions=flex_positions)

        # create salary and points arrays
        # these match indices of pool
        cmap = {'points': ctx['ga_settings']['points_column'],
                'salary': ctx['ga_settings']['salary_column']}
        points = pool[cmap['points']].values
        salaries = pool[cmap['salary']].values
        
        # create initial population
        initial_population = ga.populate(
            pospool=pospool, 
            posmap=ctx['site_settings']['posmap'], 
            population_size=pop_size
        )

        # apply validators
        # default is to valdate duplicates and salary
        # can add other validators as desired
        initial_population = ga.validate(
            population=initial_population, 
            salaries=salaries,
            salary_cap=ctx['site_settings']['salary_cap']
        )

        # need fitness to determine best lineup
        # and also for selection when loop starts
        population_fitness = ga.fitness(
            population=initial_population, 
            points=points
        )

        # set overall_max based on initial population
        omidx = population_fitness.argmax()
        best_fitness = population_fitness[omidx]
        best_lineup = initial_population[omidx]

        # create new generations
        n_unimproved = 0
        population = initial_population.copy()

        for i in range(1, ctx['ga_settings']['n_generations'] + 1):

            # end program after n generations if not improving
            if n_unimproved == ctx['ga_settings']['stop_criteria']:
                break

            # display progress information with verbose parameter
            if ctx['ga_settings'].get('verbose'):
                logging.info(f'Starting generation {i}')
                logging.info(f'Best lineup score {best_fitness}')

            # select the population
            # here, we are holding back the fittest 20% to ensure
            # that crossover and mutation do not overwrite good individuals
            elite = ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population) // ctx['ga_settings'].get('elite_divisor', 5),
                method=ctx['ga_settings'].get('elite_method', 'fittest')
            )

            selected = ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population),
                method=ctx['ga_settings'].get('select_method', 'roulette')
            )

            # cross over the population
            # here, we use uniform crossover, which splits the population
            # and randomly exchanges 0 - all chromosomes
            crossed_over = ga.crossover(population=selected, method=ctx['ga_settings'].get('crossover_method', 'uniform'))

            # mutate the crossed over population (leave elite alone)
            # can use fixed rate or variable to reduce mutation over generations
            # here we use a variable rate that increases if no improvement is found
            mutation_rate = ctx['ga_settings'].get('mutation_rate', max(.05, n_unimproved / 50))
            mutated = ga.mutate(population=crossed_over, mutation_rate=mutation_rate)

            # validate the population (elite + mutated)
            population = ga.validate(
                population=np.vstack((elite, mutated)), 
                salaries=salaries, 
                salary_cap=ctx['site_settings']['salary_cap']
            )
            
            # assess fitness and get the best score
            population_fitness = ga.fitness(population=population, points=points)
            omidx = population_fitness.argmax()
            generation_max = population_fitness[omidx]
        
            # if new best score, then set n_unimproved to 0
            # and save the new best score and lineup
            # otherwise increment n_unimproved
            if generation_max > best_fitness:
                logging.info(f'Lineup improved to {generation_max}')
                best_fitness = generation_max
                best_lineup = population[omidx]
                n_unimproved = 0
            else:
                n_unimproved += 1
                logging.info(f'Lineup unimproved {n_unimproved} times')

        # FINALIZE RESULTS
        # show best score and lineup at conclusion
        # will break after n_generations or when stop_criteria reached
        return {
            'population': population,
            'fitness': population_fitness,
            'best_lineup': pool.loc[best_lineup, :],
            'best_score': best_fitness
        }
