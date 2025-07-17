# pangadfs/pangadfs/optimize.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Any, Dict

import numpy as np

from pangadfs.base import OptimizeBase
from pangadfs.ga import GeneticAlgorithm


class OptimizeDefault(OptimizeBase):

    def optimize(self, ga: GeneticAlgorithm, **kwargs) -> Dict[str, Any]:
        """Creates initial pool
        
        Args:
            ga (GeneticAlgorithm): the ga instance
            **kwargs: keyword arguments for plugins
            
        Returns:
            Dict
            'population': np.ndarray,
            'fitness': np.ndarray,
            'best_lineup': pd.DataFrame,
            'best_score': float

        """
        # Start profiling
        ga.profiler.start_optimization()
        # create pool and pospool
        # pospool used to generate initial population
        # is a dict of position_name: DataFrame
        pop_size = ga.ctx['ga_settings']['population_size']
        pool = ga.pool(csvpth=ga.ctx['ga_settings']['csvpth'])
        cmap = {'points': ga.ctx['ga_settings']['points_column'],
                'position': ga.ctx['ga_settings']['position_column'],
                'salary': ga.ctx['ga_settings']['salary_column']}
        posfilter = ga.ctx['site_settings']['posfilter']
        flex_positions = ga.ctx['site_settings']['flex_positions']
        pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping=cmap, flex_positions=flex_positions)

        # create salary and points arrays
        # these match indices of pool
        cmap = {'points': ga.ctx['ga_settings']['points_column'],
                'salary': ga.ctx['ga_settings']['salary_column']}
        points = pool[cmap['points']].values
        salaries = pool[cmap['salary']].values
        
        # create initial population
        initial_population = ga.populate(
            pospool=pospool, 
            posmap=ga.ctx['site_settings']['posmap'], 
            population_size=pop_size
        )

        # apply validators
        # default is to valdate duplicates and salary
        # can add other validators as desired
        initial_population = ga.validate(
            population=initial_population, 
            salaries=salaries,
            salary_cap=ga.ctx['site_settings']['salary_cap']
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

        # Mark setup phase complete
        ga.profiler.mark_setup_complete()
        
        # Mark initial best solution (generation 0)
        ga.profiler.mark_best_solution(0)

        # create new generations
        n_unimproved = 0
        population = initial_population.copy()

        for i in range(1, ga.ctx['ga_settings']['n_generations'] + 1):
            # Start generation timing
            ga.profiler.start_generation(i)

            # end program after n generations if not improving
            if n_unimproved == ga.ctx['ga_settings']['stop_criteria']:
                break

            # display progress information with verbose parameter
            if ga.ctx['ga_settings'].get('verbose'):
                logging.info(f'Starting generation {i}')
                logging.info(f'Best lineup score {best_fitness}')

            # select the population
            # here, we are holding back the fittest 20% to ensure
            # that crossover and mutation do not overwrite good individuals
            elite = ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population) // ga.ctx['ga_settings'].get('elite_divisor', 5),
                method=ga.ctx['ga_settings'].get('elite_method', 'fittest')
            )

            selected = ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population),
                method=ga.ctx['ga_settings'].get('select_method', 'roulette')
            )

            # cross over the population
            # here, we use uniform crossover, which splits the population
            # and randomly exchanges 0 - all chromosomes
            crossed_over = ga.crossover(population=selected, method=ga.ctx['ga_settings'].get('crossover_method', 'uniform'))

            # mutate the crossed over population (leave elite alone)
            # can use fixed rate or variable to reduce mutation over generations
            # here we use a variable rate that increases if no improvement is found
            mutation_rate = ga.ctx['ga_settings'].get('mutation_rate', max(.05, n_unimproved / 50))
            mutated = ga.mutate(population=crossed_over, mutation_rate=mutation_rate)

            # validate the population (elite + mutated)
            population = ga.validate(
                population=np.vstack((elite, mutated)), 
                salaries=salaries, 
                salary_cap=ga.ctx['site_settings']['salary_cap']
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
                # Mark when best solution was found
                ga.profiler.mark_best_solution(i)
            else:
                n_unimproved += 1
                logging.info(f'Lineup unimproved {n_unimproved} times')
            
            # End generation timing
            ga.profiler.end_generation()

        # End profiling
        ga.profiler.end_optimization()

        # FINALIZE RESULTS
        # will break after n_generations or when stop_criteria reached
        results = {
            'population': population,
            'fitness': population_fitness,
            'best_lineup': pool.loc[best_lineup, :],
            'best_score': best_fitness
        }
        
        # Add profiling data to results
        if ga.profiler.enabled:
            results['profiling'] = ga.profiler.export_to_dict()
        
        return results
