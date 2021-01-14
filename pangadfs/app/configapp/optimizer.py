# panggadfs/app/configapp/optimizer.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Any, Dict, Tuple

import numpy as np

from pangadfs.ga import GeneticAlgorithm
from gasettings import AppSetting


class Optimizer:

    def __init__(self, 
                 ctx: AppSetting, 
                 driver_managers: Dict[str, Any], 
                 extension_managers: Dict[str, Any]):
        """Creates GeneticAlgorithm instance

        Args:
            ctx (AppSetting): context dataclass with all relevant settings
            driver_managers (Dict[str, Any]): driver managers
            extension_managers (Dict[str, Any]): extension managers

        Returns:
            Optimizer: the optimizer instance

        """
        logging.getLogger(__name__).addHandler(logging.NullHandler())

        # app settings
        self.ctx = ctx

        # create ga
        self.ga = GeneticAlgorithm(
            driver_managers=driver_managers, 
            extension_managers=extension_managers
        )

    def optimize(self, verbose: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Default routine for optimizing lineup
        
        Args:
            verbose (bool): controls logging of optimizer progress
            **kwargs: Keyword arguments for plugins (other than default)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: population and population fitness arrays                
        
        """
        pop_size = self.ctx.ga_settings.population_size
        pool = self.ga.pool(csvpth=self.ctx.ga_settings.csvpth)
        cmap = {'points': self.ctx.ga_settings.points_column,
			    'position': self.ctx.ga_settings.position_column,
			    'salary': self.ctx.ga_settings.salary_column}
        posfilter = self.ctx.site_settings.posfilter
        pospool = self.ga.pospool(
            pool=pool, 
            posfilter=posfilter, 
            column_mapping=cmap, 
            flex_positions=self.ctx.site_settings.flex_positions
        )

        # create dict of index and stat value
        # this will allow easy lookup later on
        cmap = {'points': self.ctx.ga_settings.points_column,
                'salary': self.ctx.ga_settings.salary_column}
        points = pool[cmap['points']].values
        salaries = pool[cmap['salary']].values
        
        # CREATE INITIAL POPULATION
        initial_population = self.ga.populate(
            pospool=pospool, 
            posmap=self.ctx.site_settings.posmap, 
            population_size=pop_size
        )

        # apply validators
        initial_population = self.ga.validate(
            population=initial_population, 
            salaries=salaries,
            salary_cap=self.ctx.site_settings.salary_cap
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
        for i in range(1, self.ctx.ga_settings.n_generations + 1):
            if n_unimproved == self.ctx.ga_settings.stop_criteria:
                break
            if i == 1:
                population = initial_population

            if verbose:
                logging.info(f'Starting generation {i}')
                logging.info(f'Best lineup score {best_fitness}')

            elite = self.ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population) // self.ctx.ga_settings.elite_divisor,
                method=self.ctx.ga_settings.elite_method
            )

            selected = self.ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population),
                method=self.ctx.ga_settings.select_method
            )

            # uniform crossover
            crossed_over = self.ga.crossover(
                population=selected, 
                method=self.ctx.ga_settings.crossover_method
            )

            # reduce mutation over generations
            mutation_rate = max(self.ctx.ga_settings.mutation_rate, n_unimproved / 50)
            mutated = self.ga.mutate(population=crossed_over, mutation_rate=mutation_rate)

            population = self.ga.validate(
                population=np.vstack((elite, mutated)), 
                salaries=salaries, 
                salary_cap=self.ctx.site_settings.salary_cap
            )
            
            population_fitness = self.ga.fitness(population=population, points=points)
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
