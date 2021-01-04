# gadfs/gadfs/ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager


def exposure(population: np.ndarray = None) -> Dict[int, int]:
    """Returns dict of index: count of individuals

    Args:
        population (np.ndarray): the population

    Returns:
        Dict[int, int]: key is index, value is count of lineup

    """
    flat = population.flatten
    return {id: size for id, size in zip(flat, np.bincount(flat)[flat])}


def ownership_penalty(ownership, base=3, boost=2) -> np.ndarray:
    """Returns 1D array of penalties
    
    Args:
        ownership (np.ndarray): 1D array of ownership
        base (int): the logarithm base, default 3
        boost (int): the constant to boost low-owned players
        
    Returns:
        np.ndarray: 1D array of penalties
        
    """
    return 0 - np.log(ownership) / np.log(base) + boost
    

class GeneticAlgorithm:
    """Handles coordination of stevedore plugin managers."""

    PLUGIN_NAMESPACES = (
       'pool', 'pospool', 'populate', 'crossover', 
       'fitness', 'mutate', 'validate'
    )

    def __init__(self, 
                 ctx: dict, 
                 driver_managers: Dict[str, DriverManager] = None, 
                 extension_managers: Dict[str, NamedExtensionManager] = None):
        """Creates GeneticAlgorithm instance

        Args:
            ctx (dict): context dict with all relevant settings
            driver_managers (dict): key is namespace, value is DriverManager
            extension_managers (dict): key is namespace, value is NamedExtensionManager

        Returns:
            GeneticAlgorithm: the GA instance

        """
        logging.getLogger(__name__).addHandler(logging.NullHandler())

        # add driver/extension managers
        self.driver_managers = driver_managers if driver_managers else {}
        self.extension_managers = extension_managers if extension_managers else {}

        # ensure have necessary parameters
        self.ctx = ctx
        if errors := self._validate_ctx():
            raise ValueError('\n'.join(errors))
        
        # ensure plugin loaded for every namespace
        for ns in self.PLUGIN_NAMESPACES:
            if ns not in driver_managers and ns not in extension_managers:
                if ns == 'validate':
	                names = ['validate_salary', 'validate_duplicates']
	                self.emgrs[ns] = NamedExtensionManager(
                        namespace='pangadfs.validate', 
                        names=names, 
                        invoke_on_load=True, 
                        name_order=True
                    )
                else:
                    mgr = DriverManager(
                        namespace=f'pangadfs.{ns}', 
                        name=f'{ns}_default', 
                        invoke_on_load=True
                    )
                    self.driver_managers[ns] = mgr
        
    def _validate_ctx(self):
        """Ensures that ctx dict has all relevant settings
        
        Args:
            None
            
        Returns:
            List[str]: the error messages

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

    def crossover(self,
                  population: np.ndarray = None,
                  fitness: np.ndarray = None,
                  pctl: int = 50,
                  agg: bool = None,
                  **kwargs):
        """Crossover operation to generate new population

        Args:
            population (np.ndarray): the population to cross over, is 2D array
            fitness (np.ndarray): the fitness of the population to crossover, is 1D array
            pctl (int): the percentile fitness (1-99) to use in crossover, default 50 
            agg (bool): if True, then aggregate multiple crossovers, otherwise is sequential.          
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: the crossed-over population

        """
        # combine keyword arguments with **kwargs
        if population is not None:
            kwargs['population'] = population
        if fitness is not None:
            kwargs['fitness'] = fitness
        if pctl is not None:
            kwargs['pctl'] = pctl
        if agg:
            kwargs['agg'] = agg

        # if there is a driver, then use it and run once
        if mgr := self.driver_managers.get('crossover'):
            return mgr.driver.crossover(**kwargs)

        # if agg=True, then aggregate crossed over populations
        if kwargs.get('agg'):
            pops = []
            population = kwargs['population']
            for ext in self.extension_managers['crossover'].extensions:
                try:
                    kwargs['population'] = population
                    population = ext.obj.mutate(**kwargs)
                except:
                    continue
            return np.aggregate(pops)

        # otherwise, run crossover for each plugin
        # after first time, crosses over prior crossed-over population
        population = kwargs['population']
        for ext in self.extension_managers['crossover'].extensions:
            try:
                kwargs['population'] = population
                population = ext.obj.crossover(**kwargs)
            except:
                continue
        return population

    def fitness(self, 
                population: np.ndarray = None, 
                points_mapping: Dict[int, float] = None, 
                **kwargs):
        """Measures fitness of population

        Args:
            population (np.ndarray): the population to cross over, is 2D array
            points_mapping (Dict[int, float]): the fitness of the population to crossover, is 1D array
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: population fitness as 1D array of float

        """
        # combine keyword arguments with **kwargs
        if population is not None:
            kwargs['population'] = population
        if points_mapping is not None:
            kwargs['points_mapping'] = points_mapping

        # if there is a driver, then use it and run once
        if mgr := self.driver_managers.get('fitness'):
            return mgr.driver.fitness(**kwargs)

        # otherwise, return fitness for first valid plugin
        for ext in self.extension_managers['fitness'].extensions:
            try:
                return ext.obj.fitness(**kwargs)
            except:
                continue

    def mutate(self, 
               population: np.ndarray = None,
               mutation_rate: float = None,
               **kwargs):
        """Mutates population at frequency of mutation_rate

        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            mutation_rate (float): decimal value from 0 to 1, default .05
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: same shape and dtype as population

        """
        # combine keyword arguments with **kwargs
        if population is not None:
            kwargs['population'] = population
        if mutation_rate is not None:
            kwargs['mutation_rate'] = mutation_rate

        # if there is a driver, then use it and run once
        if mgr := self.driver_managers.get('mutate'):
            return mgr.driver.mutate(**kwargs)

        # otherwise, mutate with first valid plugin
        population = kwargs['population']
        for ext in self.extension_managers['mutate'].extensions:
            try:
                kwargs['population'] = population
                population = ext.obj.mutate(**kwargs)
            except:
                continue
        return population

    def optimize(self, verbose: bool = False, **kwargs):
        """Optimizes lineup
        
        Args:
            verbose (bool): controls logging of optimizer progress
            **kwargs: Keyword arguments for plugins (other than default)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: population and population fitness arrays                
        
        """
        pop_size = self.ctx['ga_settings']['population_size']
        pool = self.pool(csvpth=self.ctx['ga_settings']['csvpth'])
        cmap = {'points': self.ctx['ga_settings']['points_column'],
			    'position': self.ctx['ga_settings']['position_column'],
			    'salary': self.ctx['ga_settings']['salary_column']}
        posfilter = self.ctx['site_settings']['posfilter']
        pospool = self.pospool(pool=pool, posfilter=posfilter, column_mapping=cmap)

        # create dict of index and stat value
        # this will allow easy lookup later on
        cmap = {'points': self.ctx['ga_settings']['points_column'],
                'salary': self.ctx['ga_settings']['salary_column']}
        points_mapping = dict(zip(pool.index, pool[cmap['points']]))
        salary_mapping = dict(zip(pool.index, pool[cmap['salary']]))
        
        # CREATE NEW GENERATIONS
        n_unimproved = 0
        population = None
        for i in range(self.ctx['ga_settings']['n_generations'] + 1):
            if n_unimproved == self.ctx['ga_settings']['stop_criteria']:
                break
            if i == 0:
                # create initial population
                population = self.populate(
                    pospool=pospool, 
                    posmap=self.ctx['site_settings']['posmap'], 
                    population_size=pop_size
                )

                # apply validators
                population = self.validate(
                    population=population, 
                    salary_mapping=salary_mapping, 
                    salary_cap=self.ctx['site_settings']['salary_cap']
                )

            else:
                population_fitness = self.fitness(
                    population=population, 
                    points_mapping=points_mapping
                )
                
                oldmax = population_fitness.max()

                if verbose:
                    logging.info(f'Starting generation {i}')
                    logging.info(f'Current lineup score {oldmax}')

                population = self.crossover(
                    population=population, 
                    population_fitness=population_fitness
                )

                population = self.mutate(
                    population=population, 
                    mutation_rate=.1
                )

                population = self.validate(
                    population=population, 
                    salary_mapping=salary_mapping, 
                    salary_cap=self.ctx['site_settings']['salary_cap']
                )

                population_fitness = self.fitness(
                    population=population, 
                    points_mapping=points_mapping
                )

                pop_len = pop_size if len(population) > pop_size else len(population)
                
                # argpartition produces an unsorted rearrangement
                # use negative parameter and slice to get top n values
                # this ensures population does not get too large from concatenation
                population = population[np.argpartition(population_fitness, -pop_len, axis=0)][-pop_len:]
                newmax = population_fitness.max()

                if newmax > oldmax:
                    logging.info(f'Lineup improved to {newmax}')
                    oldmax = newmax
                    n_unimproved = 0
                else:
                    n_unimproved += 1
                    logging.info(f'Lineup unimproved {n_unimproved} times')

        population_fitness = self.fitness(population=population, points_mapping=points_mapping)

        if verbose:
            idx = population_fitness.argmax()
            print(pool.loc[population[idx], :])
            print(f'Lineup score: {population_fitness[idx]}')

        return population, population_fitness
 
    def pool(self, csvpth: Path = None, **kwargs):
        """Creates pool of players.

        Args:
            csvpth (Path): the path of the datafile
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            pd.DataFrame: initial pool of players
        
        """
        if csvpth is not None:
            kwargs['csvpth'] = csvpth

        if mgr := self.driver_managers.get('pool'):
            return mgr.driver.pool(**kwargs)   
        for ext in self.extension_managers['pool'].extensions:
            try:
                return ext.obj.pool(**kwargs)  
            except:
                logging.error(f'Could not load {ext}')

    def populate(self, 
                 pospool: Dict[str, pd.DataFrame] = None,
                 posmap: Dict[str, int] = None,
                 population_size: int = None,
                 probcol: str = None,
                 agg: bool = None,
                 **kwargs):
        """Creates initial population of specified size
        
        Args:
            pospool (Dict[str, pd.DataFrame]): pool segmented by position
            posmap (Dict[str, int]): positions & accompanying roster slots
            population_size (int): number of individuals to create
            probcol (str): the dataframe column with probabilities
            agg (bool): if True, then aggregate multiple crossovers, otherwise is sequential.
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: the population

        """
        # combine keyword arguments with **kwargs
        if pospool is not None:
            kwargs['pospool'] = pospool
        if posmap is not None:
            kwargs['posmap'] = posmap
        if population_size is not None:
            kwargs['population_size'] = population_size
        if probcol is not None:
            kwargs['probcol'] = probcol
        if agg:
            kwargs['agg'] = agg

        # if there is a driver, then use it and run once
        if mgr := self.driver_managers.get('populate'):
            return mgr.driver.populate(**kwargs)

        # if agg=True, then aggregate populations
        if kwargs.get('agg'):
            pops = []
            for ext in self.extension_managers['populate'].extensions:
                try:
                    pops.append(ext.obj.populate(**kwargs))  
                except:
                    continue
            return np.concatenate(pops)

        # otherwise, run populate using first valid plugin
        for ext in self.extension_managers['crossover'].extensions:
            try:
                return ext.obj.populate(**kwargs)
            except:
                continue

    def pospool(self, 
                pool: pd.DataFrame = None,
                posfilter: Dict[str, int] = None,
                column_mapping: Dict[str, str] = None,
                flex_positions: Iterable[str] = None,
                **kwargs):
        """Divides pool into positional buckets
        
        Args:   
            pool (pd.DataFrame):
            posfilter (Dict[str, int]): position name and points threshold
            column_mapping (Dict[str, str]): column names for player, position, salary, projection
            flex_positions (Iterable[str]): e.g. (WR, RB, TE)
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            Dict[str, pd.DataFrame] where keys == posfilter.keys

        """
        # combine keyword arguments with **kwargs
        if pool is not None:
            kwargs['pool'] = pool
        if posfilter is not None:
            kwargs['posfilter'] = posfilter
        if column_mapping is not None:
            kwargs['column_mapping'] = column_mapping
        if flex_positions is not None:
            kwargs['flex_positions'] = flex_positions

        # if there is a driver, then use it and run once
        # otherwise, run pospool using first valid plugin
        if mgr := self.driver_managers.get('pospool'): 
            return mgr.driver.pospool(**kwargs)
        for ext in self.extension_managers['pospool'].extensions:
            try:
                return ext.obj.pospool(**kwargs)  
            except:
                continue
        
    def validate(self, population: np.ndarray = None, **kwargs):
        """Validate lineup according to validate plugin criteria
        
        Args:
            population (np.ndarray): the population to validate.
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: same width and dtype as population. Likely less rows due to exclusions.
            
        """
        if population is not None:
            kwargs['population'] = population

        if mgr := self.driver_managers.get('validate'):
            return mgr.driver.validate(**kwargs)
        population = kwargs['population']
        for ext in self.extension_managers['validate'].extensions:
            kwargs['population'] = population
            population = ext.obj.validate(**kwargs)
        return population


if __name__ == '__main__':
    pass
