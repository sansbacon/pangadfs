# gadfs/gadfs/ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging
import numpy as np
from stevedore import driver


class GeneticAlgorithm:
    """"This class handles the coordination of all of the stevedore plugin managers."""

    PLUGIN_NAMESPACES = (
       'pool', 'pospool', 'populate', 'crossover', 
       'fitness', 'mutate', 'validate'
    )

    def __init__(self, 
                 ctx: dict, 
                 driver_managers: dict = None, 
                 extension_managers: dict = None):
        """Creates GeneticAlgorithm instance

        Args:
            ctx (dict): context dict with all relevant settings
            driver_managers (dict): key is namespace, value is DriverManager
            extension_managers (dict): key is namespace, value is NamedExtensionManager

        Returns:
            GeneticAlgorithm

        """
        logging.getLogger(__name__).addHandler(logging.NullHandler())

        # add driver/extension managers
        if not driver_managers and not extension_managers:
            raise ValueError('driver_managers and extension_managers cannot both be none')
        self.driver_managers = driver_managers
        self.extension_managers = extension_managers

        # ensure have necessary parameters
        self.ctx = ctx
        if errors := self._validate_ctx():
            raise ValueError('\n'.join(errors))
        
        # ensure plugin loaded for every namespace
        for ns in self.PLUGIN_NAMESPACES:
            if ns not in driver_managers and ns not in extension_managers:
                mgr = driver.DriverManager(
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
            List[str]

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
            for k in ('salary_cap', 'lineup_size', 'posthresh'):
                if k not in self.ctx['site_settings']:
                    errors.append(f'site_settings missing {k}')
        return errors

    def crossover(self, **kwargs):
        """Crossover operation to generate new population

           Keyword Args:
               agg (bool): if True, then aggregate multiple crossovers, otherwise is sequential.
               population (np.ndarray): the population to cross over, is 2D array
               population_fitness (np.ndarray): the fitness of the population to crossover, is 1D array
               pctl (int): the percentile fitness (1-99) to use in crossover, default 50 

            Returns:
                np.ndarray

        """
        # if there is a driver, then use it and run once
        if mgr := self.driver_managers.get('crossover'):
            return mgr.driver.crossover(**kwargs)

        # if agg=True, then aggregate mutated populations
        if kwargs.get('agg'):
            pops = []
            population = kwargs['population']
            for ext in self.extension_managers['mutate'].extensions:
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

    def fitness(self, **kwargs):
        """Measures fitness of population

           Keyword Args:
               population (np.ndarray): the population to cross over, is 2D array
               points_mapping (Dict[int, float]): the fitness of the population to crossover, is 1D array
    
            Returns:
                np.ndarray - 1D array of float

        """

        if mgr := self.driver_managers.get('fitness'):
            return mgr.driver.fitness(**kwargs)
        for ext in self.extension_managers['fitness'].extensions:
            try:
                return ext.obj.fitness(**kwargs)
            except:
                continue

    def mutate(self, **kwargs):
        """Mutates population

            Keyword Args:
                population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
                mutation_rate (float): decimal value from 0 to 1, default .05

            Returns:
                np.ndarray of same shape and dtype as population

        """
        if mgr := self.driver_managers.get('mutate'):
            return mgr.driver.mutate(**kwargs)
        population = kwargs['population']
        for ext in self.extension_managers['mutate'].extensions:
            try:
                kwargs['population'] = population
                population = ext.obj.mutate(**kwargs)
            except:
                continue
        return population

    def optimize(self, **kwargs):
        """Optimizes lineup
        
        Keyword Args:
            verbose (bool): controls logging of optimizer progress
                
        Returns:
            Tuple[np.ndarray, np.ndarray]                
        
        TODO: update to reflect current app structure
        """
        pool = self.pool(csvpth=self.ctx['ga_settings']['csvpth'])
        posfilter = self.ctx['ga_settings']['posfilter']
        pospool = self.pospool(
            pool=pool, 
            posfilter=posfilter, 
            column_mapping=self.ctx['ga_settings'].get('column_mapping', {})
        )

        # create dict of index and stat value
        # this will allow easy lookup later on
        points_mapping = dict(zip(pool.index, pool[self.ctx['ga_settings']['points_column']]))
        salary_mapping = dict(zip(pool.index, pool[self.ctx['ga_settings']['salary_column']]))
        
        # create initial population
        # choose 7 FLEX so always have unique value (6 required from 2 RB, 3 WR, 1 TE)
        population = self.populate(
          pospool=pospool, 
          posmap=self.ctx['site_settings']['posmap'], 
          population_size=self.ctx['ga_settings']['population_size']
        )
        
        population = self.validate(
          population=population, 
          salary_mapping=salary_mapping, 
          salary_cap=self.ctx['site_settings']['salary_cap']
        )
        
        population_fitness = self.fitness(population=population, points_mapping=points_mapping)
        oldmax = population_fitness.max()

        # CREATE NEW GENERATIONS
        for i in range(self.ctx['ga_settings']['n_generations']):
            if kwargs.get('verbose'):
                logging.info(f'Starting generation {i}')
            population = self.crossover(
              population=population, 
              population_fitness=population_fitness
            )
            
            population = self.validate(
              population=population, 
              salary_mapping=salary_mapping, 
              salary_cap=self.ctx['site_settings']['salary_cap']
            )
            
            population_fitness = self.fitness(population=population, points_mapping=points_mapping)
            thismax = population_fitness.max()
            if thismax > oldmax:
                oldmax = thismax
                if kwargs.get('verbose'):
                    logging.info(pool.loc[population[population_fitness.argmax()], :])
                    logging.info(f'Lineup score: {oldmax}')
        return population, population_fitness      
 
    def pool(self, **kwargs):
        """Creates pool of players.

            Keyword Args:
                csvpth (Path): the path of the datafile

            Returns:
                pd.DataFrame
        
        """
        if mgr := self.driver_managers.get('pool'):
            return mgr.driver.pool(**kwargs)   
        for ext in self.extension_managers['pool'].extensions:
            try:
                return ext.obj.pool(**kwargs)  
            except:
                logging.error(f'Could not load {ext}')

    def populate(self, **kwargs):
        """Creates initial population of specified size
        
            Keyword Args:
               pospool (Dict[str, pd.DataFrame]): pool segmented by position
               posmap (Dict[str, int]): positions & accompanying roster slots
               population_size (int): number of individuals to create
               probcol (str): the dataframe column with probabilities
               agg (bool): if True, then aggregate multiple crossovers, otherwise is sequential.

            Returns:
                np.ndarray

        """
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

    def pospool(self, **kwargs):
        """Divides pool into positional buckets
        
            Keyword Args:   
                pool (pd.DataFrame):
                posfilter (Dict[str, int]): position name and points threshold
                column_mapping (Dict[str, str]): column names for player, position, salary, projection
                flex_positions (Iterable[str]): e.g. (WR, RB, TE)
    
            Returns:
                Dict[str, pd.DataFrame] where keys == posfilter.keys

        """
        # if there is a driver, then use it and run once
        # otherwise, run pospool using first valid plugin
        if mgr := self.driver_managers.get('pospool'): 
            return mgr.driver.pospool(**kwargs)
        for ext in self.extension_managers['pospool'].extensions:
            try:
                return ext.obj.pospool(**kwargs)  
            except:
                continue
        
    def validate(self, **kwargs):
        """Validate lineup according to validate plugin criteria
        
            Keyword Args:
                population (np.ndarray): the population to validate.

            Returns:
                np.ndarray of same width and dtype as population.
                Likely has less rows because excluded invalid lineups.

        """
        if mgr := self.driver_managers.get('validate'):
            return mgr.driver.validate(**kwargs)
        population = kwargs['population']
        for ext in self.extension_managers['validate'].extensions:
            kwargs['population'] = population
            population = ext.obj.validate(**kwargs)
        return population


if __name__ == '__main__':
    pass
