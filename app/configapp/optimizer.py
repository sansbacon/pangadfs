# gadfs/gadfs/optimizer.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging
from typing import Iterable

from pangadfs.ga import GeneticAlgorithm


class Optimizer:

    def __init__(self, ctx: dict):
        """Creates GeneticAlgorithm instance

        Args:
            ctx (dict): context dict with all relevant settings

        Returns:
            Optimizer: the optimizer instance

        """
        logging.getLogger(__name__).addHandler(logging.NullHandler())

        # ensure have necessary parameters
        self.ctx = ctx
        if errors := self._validate_ctx():
            raise ValueError('\n'.join(errors))

        # create ga
        self.ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)

    def _validate_ctx(self) -> Iterable[str]:
        """Ensures that ctx dict has all relevant settings
        
        Args:
            None
            
        Returns:
            Iterable[str]: the error messages

        """
        errors = []
        if 'ga_settings' not in self.ctx:
            errors.append('Missing ga_settings section')
        else:
            for k in ('n_generations', 'population_size', 'points_column', 'salary_column', 'csvpth'):
                if k not in self.ctx['ga_settings']:
                    errors.append(f'ga_settings missing {k}')
        if 'site_settings' not in self.ctx:
            errors.append('Missing site_settings section')
        else:
            for k in ('salary_cap', 'lineup_size', 'posfilter'):
                if k not in self.ctx['site_settings']:
                    errors.append(f'site_settings missing {k}')
        return errors

    def optimize(self, verbose: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Default routine for optimizing lineup
        
        Args:
            verbose (bool): controls logging of optimizer progress
            **kwargs: Keyword arguments for plugins (other than default)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: population and population fitness arrays                
        
        """
        pop_size = self.ctx['ga_settings']['population_size']
        pool = self.ga.pool(csvpth=self.ctx['ga_settings']['csvpth'])
        cmap = {'points': self.ctx['ga_settings']['points_column'],
			    'position': self.ctx['ga_settings']['position_column'],
			    'salary': self.ctx['ga_settings']['salary_column']}
        posfilter = self.ctx['site_settings']['posfilter']
        pospool = self.ga.pospool(pool=pool, posfilter=posfilter, column_mapping=cmap)

        # create dict of index and stat value
        # this will allow easy lookup later on
        cmap = {'points': self.ctx['ga_settings']['points_column'],
                'salary': self.ctx['ga_settings']['salary_column']}
        points = pool[cmap['points']].values
        salaries = pool[cmap['salary']].values
        
        # CREATE INITIAL POPULATION
        initial_population = self.ga.populate(
            pospool=pospool, 
            posmap=self.ctx['site_settings']['posmap'], 
            population_size=pop_size
        )

        # apply validators
        initial_population = self.ga.validate(
            population=initial_population, 
            salaries=salaries,
            salary_cap=self.ctx['site_settings']['salary_cap']
        )

        population_fitness = self.ga.fitness(
            population=initial_population, 
            points=points
        )

        # set overall_max based on initial population
        omidx = population_fitness.argmax()
        best_fitness = population_fitness[omidx]
        best_lineup = initial_population[omidx]

        # CREATE NEW GENERATIONS
        n_unimproved = 0
        for i in range(1, self.ctx['ga_settings']['n_generations'] + 1):
            if n_unimproved == self.ctx['ga_settings']['stop_criteria']:
                break
            if i == 1:
                population = initial_population

            if verbose:
                logging.info(f'Starting generation {i}')
                logging.info(f'Best lineup score {best_fitness}')

            elite = self.ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population) // 5,
                method='fittest'
            )

            selected = self.ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population),
                method='roulette'
            )

            # uniform crossover
            crossed_over = self.ga.crossover(population=selected, method='uniform')

            # reduce mutation over generations
            mutation_rate = max(.05, n_unimproved / 50)
            mutated = self.ga.mutate(population=crossed_over, mutation_rate=mutation_rate)

            population = self.ga.validate(
                population=np.vstack((elite, mutated)), 
                salaries=salaries, 
                salary_cap=self.ctx['site_settings']['salary_cap']
            )
            
            population_fitness = self.fitness(population=population, points=points)
            omidx = population_fitness.argmax()
            generation_max = population_fitness[omidx]
        
            if generation_max > best_fitness:
                logging.info(f'Lineup improved to {generation_max}')
                best_fitness = generation_max
                best_lineup = population[omidx]
                n_unimproved = 0
            else:
                n_unimproved += 1
                logging.info(f'Lineup unimproved {n_unimproved} times')

        # FINALIZE RESULTS
        if verbose:
            print(pool.loc[best_lineup, :])
            print(f'Lineup score: {best_fitness}')
        
        return population, population_fitness
