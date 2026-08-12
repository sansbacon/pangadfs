# pangadfs/pangadfs/ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import numpy as np
import pandas as pd

from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager
from pangadfs.profiler import GAProfiler


class GeneticAlgorithm:
    """Handles coordination of genetic algorithm plugins"""

    PLUGIN_NAMESPACES = (
       'pool', 'pospool', 'populate', 'fitness', 'optimize',
       'select', 'crossover', 'mutate', 'validate'
    )

    VALIDATE_PLUGINS = ('validate_salary', 'validate_duplicates', 'validate_positions')

    def __init__(self, 
                 ctx: Union[Dict, Any] = None,
                 driver_managers: Dict[str, DriverManager] = None, 
                 extension_managers: Dict[str, NamedExtensionManager] = None,
                 use_defaults: bool = False):
        """Creates GeneticAlgorithm instance

        Args:
            ctx (dict): the context dict, AppConfig object, or other configuration scheme
            driver_managers (dict): key is namespace, value is DriverManager
            extension_managers (dict): key is namespace, value is NamedExtensionManager
            use_defaults (bool): use default plugins

        Returns:
            GeneticAlgorithm: the GA instance

        """
        logging.getLogger(__name__).addHandler(logging.NullHandler())

        # add context
        self.ctx = ctx

        # add driver/extension managers
        self.driver_managers = driver_managers if driver_managers else {}
        self.extension_managers = extension_managers if extension_managers else {}

        # Initialize profiler based on context settings
        profiling_enabled = False
        if ctx and isinstance(ctx, dict):
            profiling_enabled = ctx.get('ga_settings', {}).get('enable_profiling', False)
        self.profiler = GAProfiler(enabled=profiling_enabled)

        # if use_defaults, then load default plugin(s) for missing namespaces
        if use_defaults:
            self._load_plugins()

    def _load_plugins(self):
        """Loads default plugins for any namespace that doesn't have a plugin"""
        for ns in self.PLUGIN_NAMESPACES:
            if ns not in self.driver_managers and ns not in self.extension_managers:
                if ns == 'validate':
                    self.extension_managers[ns] = NamedExtensionManager(
                        namespace='pangadfs.validate', 
                        names=self.VALIDATE_PLUGINS, 
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

    def crossover(self,
                  *,
                  population: np.ndarray = None,
                  agg: bool = None,
                  **kwargs) -> np.ndarray:
        """Crossover operation to generate new population

        Args:
            population (np.ndarray): the population to cross over, is 2D array
            agg (bool): if True, then aggregate multiple crossovers, otherwise is sequential.          
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: the crossed-over population

        """
        with self.profiler.time_operation('Crossover'):
            if mgr := self.driver_managers.get('crossover'):
                return mgr.driver.crossover(population=population, agg=agg, **kwargs)

            if agg:
                pops = []
                for ext in self.extension_managers['crossover'].extensions:
                    try:
                        pops.append(ext.obj.crossover(population=population, **kwargs))
                    except Exception as e:
                        logging.warning(f'Crossover plugin {ext} failed: {e}')
                        continue
                return np.concatenate(pops)

            # Sequential: each plugin crosses over the prior result
            for ext in self.extension_managers['crossover'].extensions:
                try:
                    population = ext.obj.crossover(population=population, **kwargs)
                except Exception as e:
                    logging.warning(f'Crossover plugin {ext} failed: {e}')
                    continue
            return population

    def fitness(self, 
                *,
                population: np.ndarray = None, 
                points: np.ndarray = None, 
                **kwargs) -> np.ndarray:
        """Measures fitness of population

        Args:
            population (np.ndarray): the population to cross over, is 2D array
            points (np.ndarray): the fitness of the population to crossover, is 1D array
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: population fitness as 1D array of float

        """
        with self.profiler.time_operation('Fitness Evaluation'):
            if mgr := self.driver_managers.get('fitness'):
                return mgr.driver.fitness(population=population, points=points, **kwargs)

            for ext in self.extension_managers['fitness'].extensions:
                try:
                    return ext.obj.fitness(population=population, points=points, **kwargs)
                except Exception as e:
                    logging.warning(f'Fitness plugin {ext} failed: {e}')
                    continue

    def mutate(self, 
               *,
               population: np.ndarray = None,
               mutation_rate: float = None,
               **kwargs) -> np.ndarray:
        """Mutates population at frequency of mutation_rate

        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            mutation_rate (float): decimal value from 0 to 1, default .05
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: same shape and dtype as population

        """
        with self.profiler.time_operation('Mutation'):
            if mgr := self.driver_managers.get('mutate'):
                return mgr.driver.mutate(population=population, mutation_rate=mutation_rate, **kwargs)

            for ext in self.extension_managers['mutate'].extensions:
                try:
                    return ext.obj.mutate(population=population, mutation_rate=mutation_rate, **kwargs)
                except Exception as e:
                    logging.warning(f'Mutate plugin {ext} failed: {e}')
                    continue

    def optimize(self, **kwargs) -> Dict[str, Any]:
        """Optimizes population

        Args:
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            dict

        """
        if mgr := self.driver_managers.get('optimize'):
            return mgr.driver.optimize(ga=self, **kwargs)

        for ext in self.extension_managers['optimize'].extensions:
            try:
                return ext.obj.optimize(ga=self, **kwargs)
            except Exception as e:
                logging.warning(f'Optimize plugin {ext} failed: {e}')
                continue
            
    def pool(self, *, csvpth: Path = None, **kwargs) -> pd.DataFrame:
        """Creates pool of players.

        Args:
            csvpth (Path): the path of the datafile
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            pd.DataFrame: initial pool of players
        
        """
        with self.profiler.time_operation('Pool Creation'):
            if mgr := self.driver_managers.get('pool'):
                return mgr.driver.pool(csvpth=csvpth, **kwargs)
            for ext in self.extension_managers['pool'].extensions:
                try:
                    return ext.obj.pool(csvpth=csvpth, **kwargs)
                except Exception as e:
                    logging.error(f'Pool plugin {ext} failed: {e}')

    def populate(self,
                 *,
                 pospool: Dict[str, pd.DataFrame] = None,
                 posmap: Dict[str, int] = None,
                 population_size: int = None,
                 probcol: str = 'prob',
                 agg: bool = False,
                 **kwargs) -> np.ndarray:
        """Creates initial population of specified size
        
        Args:
            pospool (Dict[str, pd.DataFrame]): pool segmented by position
            posmap (Dict[str, int]): positions & accompanying roster slots
            population_size (int): number of individuals to create
            probcol (str): the dataframe column with probabilities, default 'prob'
            agg (bool): default False. Aggregate multiple populate plugins if True.
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: the population

        """
        populate_kwargs = dict(
            pospool=pospool, posmap=posmap,
            population_size=population_size, probcol=probcol, **kwargs
        )

        with self.profiler.time_operation('Initial Population'):
            if mgr := self.driver_managers.get('populate'):
                return mgr.driver.populate(**populate_kwargs)

            if agg:
                pops = []
                for ext in self.extension_managers['populate'].extensions:
                    try:
                        pops.append(ext.obj.populate(**populate_kwargs))
                    except Exception as e:
                        logging.warning(f'Populate plugin {ext} failed: {e}')
                        continue
                return np.concatenate(pops)

            # Use first valid populate plugin
            for ext in self.extension_managers['populate'].extensions:
                try:
                    return ext.obj.populate(**populate_kwargs)
                except Exception as e:
                    logging.warning(f'Populate plugin {ext} failed: {e}')
                    continue

    def pospool(self, 
                *,
                pool: pd.DataFrame = None,
                posfilter: Dict[str, int] = None,
                column_mapping: Dict[str, str] = None,
                flex_positions: Iterable[str] = None,
                **kwargs) -> Dict[str, pd.DataFrame]:
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
        pospool_kwargs = dict(
            pool=pool, posfilter=posfilter,
            column_mapping=column_mapping, flex_positions=flex_positions, **kwargs
        )

        with self.profiler.time_operation('Pospool'):
            if mgr := self.driver_managers.get('pospool'):
                return mgr.driver.pospool(**pospool_kwargs)
            for ext in self.extension_managers['pospool'].extensions:
                try:
                    return ext.obj.pospool(**pospool_kwargs)
                except Exception as e:
                    logging.warning(f'Pospool plugin {ext} failed: {e}')
                    continue
        
    def select(self,
               *, 
               population: np.ndarray = None, 
               population_fitness: np.ndarray = None,
               n: int = None,
               method: str = 'fittest',
               **kwargs) -> np.ndarray:
        """Selects/filters population

        Args:
            population (np.ndarray): the population to cross over, is 2D array
            population_fitness (np.ndarray): 1D array of float
            n (int): number of individuals to select
            method (str): the selection method, default 'fittest'
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: selected population

        """
        select_kwargs = dict(
            population=population, population_fitness=population_fitness,
            n=n, method=method, **kwargs
        )

        with self.profiler.time_operation('Selection'):
            if mgr := self.driver_managers.get('select'):
                return mgr.driver.select(**select_kwargs)

            for ext in self.extension_managers['select'].extensions:
                try:
                    return ext.obj.select(**select_kwargs)
                except Exception as e:
                    logging.warning(f'Select plugin {ext} failed: {e}')
                    continue

    def validate(self,
                 *,
                 population: np.ndarray = None, 
                 salaries: np.ndarray = None, 
                 **kwargs) -> np.ndarray:
        """Validate lineup according to validate plugin criteria
        
        Args:
            population (np.ndarray): the population to validate.
            salaries (np.ndarray): the population salaries
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: same width and dtype as population. Likely less rows due to exclusions.
            
        """
        with self.profiler.time_operation('Validation'):
            if mgr := self.driver_managers.get('validate'):
                return mgr.driver.validate(population=population, salaries=salaries, **kwargs)

            # Chain validators: each filters the population in sequence
            for ext in self.extension_managers['validate'].extensions:
                population = ext.obj.validate(population=population, salaries=salaries, **kwargs)
            return population
