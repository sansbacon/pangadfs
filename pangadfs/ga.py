# gadfs/gadfs/ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging
import numpy as np


class GeneticAlgorithm:

    """"
    This class handles the coordination of all of the plugin managers and namespaces.
    pangadfs uses stevedore for plugins, specificaly, the DriverManager and NamedExtensionManager.
    
    The simplest approach is for your app to use a DriverManager, and load one plugin from each namespace.
    You can also use a NamedExtensionManager, and you can load one or more plugin from each namespace.
    The most common use for this is in the validate namespace, where you want run the population
    through multiple validation checks.

    Plugin Namespaces

    pool: 
        Load the initial projections (or results) into standardized format
        Pool should have columns for position, salary, points/projection

    pospool: 
        Splits Pool into positional groups, including FLEX
        Adds weighting column for random selection (default is points-per-dollar)

    populate:
        Generates the initial population from the pool.
        Can run multiple populate plugins, will aggregate the results into one population.
    
    crossover:
        Generates new population from the initial population.
        Can run multiple crossover plugins
        If pass agg=True, will aggregate the results into one population.
        Otherwise, runs on crossed-over population

    fitness:
        Creates 1D array of fitness scores.
        Right now, can only run fitness once but can use any function to generate single value.

    mutate:
        Generates new population from the initial population.
        Can run multiple mutate plugins
        If pass agg=True, will aggregate the results into one population.
        Otherwise, second call runs on first mutated population, and so forth       

    validate:
        Generates new population from the initial population.
        Can run multiple validate plugins
        Each subsequent call runs on prior validated population

    """
    def __init__(self, driver_managers=None, extension_managers=None):
        logging.getLogger(__name__).addHandler(logging.NullHandler())
        if not driver_managers and not extension_managers:
            raise ValueError('driver_managers and extension_managers cannot both be none')
        self._pool = None
        self._pospool = None
        self.driver_managers = driver_managers
        self.extension_managers = extension_managers
        
    def crossover(self, **kwargs):
        """Can run crossover over multiple plugins as long as
           (1) plugin has a population parameter
           (2) plugin returns an ndarray

           Pass agg=True to aggregate results
           Otherwise, next crossover operates on result of prior crossover
        """
        # if there is a driver, then use it and run once
        if mgr := self.driver_managers.get('crossover'):
            return mgr.driver.crossover(**kwargs)

        # if agg=True, then aggregate mutated populations
        if kwargs.get('agg'):
            pops = []
            population = kwargs['population']
            for ext in self.extension_managers.get('mutate'):
                try:
                    kwargs['population'] = population
                    population = ext.obj.mutate(**kwargs)
                except:
                    continue
            return np.aggregate(pops)

        # otherwise, run crossover for each plugin
        # after first time, crosses over prior crossed-over population
        population = kwargs['population']
        for ext in self.extension_managers.get('crossover'):
            try:
                kwargs['population'] = population
                population = ext.obj.crossover(**kwargs)
            except:
                continue
        return population

    def fitness(self, **kwargs):
        """Can only run fitness over one plugin right now"""
        if mgr := self.driver_managers.get('fitness'):
            return mgr.driver.fitness(**kwargs)
        for ext in self.extension_managers.get('fitness'):
            try:
                return ext.obj.fitness(**kwargs)  
            except:
                continue

    def mutate(self, **kwargs):
        """Can run mutate over multiple plugins as long as
           (1) plugin has a population parameter
           (2) plugin returns an ndarray

        """
        if mgr := self.driver_managers.get('mutate'):
            return mgr.driver.mutate(**kwargs)
        population = kwargs['population']
        for ext in self.extension_managers.get('mutate'):
            try:
                kwargs['population'] = population
                population = ext.obj.mutate(**kwargs)
            except:
                continue
        return population

    def pool(self, **kwargs):
        """Right now only supports running one pool plugin"""
        if mgr := self.driver_managers.get('pool'):
            return mgr.driver.pool(**kwargs)   
        for ext in self.extension_managers.get('pool'):
            try:
                return ext.obj.pool(**kwargs)  
            except:
                continue

    def populate(self, **kwargs):
        """Aggregates populations generated from each plugin
           Only condition is that each element has same # of columns
        """
        if mgr := self.driver_managers.get('populate'):
            return mgr.driver.populate(**kwargs)
        pops = []
        for ext in self.extension_managers.get('populate'):
            try:
                pops.append(ext.obj.populate(**kwargs))  
            except:
                continue
        return np.concatenate(pops)

    def pospool(self, **kwargs):
        """Right now only supports running one pospool plugin"""
        if mgr := self.driver_managers.get('pospool'): 
            return mgr.driver.pospool(**kwargs)
        for ext in self.extension_managers.get('pospool'):
            try:
                return ext.obj.pospool(**kwargs)  
            except:
                continue
        
    def validate(self, **kwargs):
        """Can run mutate over multiple plugins as long as
           (1) plugin has a population parameter
           (2) plugin returns an ndarray

        """
        if mgr := self.driver_managers.get('validate'):
            return mgr.driver.validate(**kwargs)
        population = kwargs['population']
        for ext in self.extension_managers.get('validate'):
            kwargs['population'] = population
            population = ext.obj.validate(**kwargs)
        return population


if __name__ == '__main__':
    pass
