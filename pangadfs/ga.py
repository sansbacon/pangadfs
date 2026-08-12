# pangadfs/pangadfs/ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd

try:
    from stevedore.driver import DriverManager
    from stevedore.named import NamedExtensionManager
except ModuleNotFoundError:
    DriverManager = None
    NamedExtensionManager = None
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
                 driver_managers: Dict[str, Any] = None, 
                 extension_managers: Dict[str, Any] = None,
                 plugins: Dict[str, Any] = None,
                 use_defaults: bool = False):
        """Creates GeneticAlgorithm instance

        Args:
            ctx (dict): the context dict, AppConfig object, or other configuration scheme
            driver_managers (dict): key is namespace, value is DriverManager
            extension_managers (dict): key is namespace, value is NamedExtensionManager
            plugins (dict): dependency-injected plugins keyed by namespace. Values may be
                plugin instances/callables or lists for chained namespaces.
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
        self.plugins = plugins if plugins else {}

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
        if DriverManager is None or NamedExtensionManager is None:
            raise ImportError(
                'stevedore is required for use_defaults=True. '
                'Install stevedore or pass plugins/manager instances directly.'
            )

        for ns in self.PLUGIN_NAMESPACES:
            if ns in self.plugins:
                continue
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

    def _iter_plugins(self, namespace: str) -> List[Any]:
        """Returns plugins for a namespace in priority order.

        Priority:
        1) dependency-injected plugins
        2) stevedore DriverManager driver
        3) stevedore NamedExtensionManager extension objects
        """
        if namespace in self.plugins:
            plugin = self.plugins[namespace]
            if isinstance(plugin, (list, tuple)):
                return list(plugin)
            return [plugin]

        if mgr := self.driver_managers.get(namespace):
            return [mgr.driver]

        if ext_mgr := self.extension_managers.get(namespace):
            return [ext.obj for ext in ext_mgr.extensions]

        return []

    @staticmethod
    def _method(plugin: Any, method_name: str) -> Any:
        """Returns method_name method if available, otherwise plugin if callable."""
        method = getattr(plugin, method_name, None)
        if callable(method):
            return method
        if callable(plugin):
            return plugin
        return None

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
            plugins = self._iter_plugins('crossover')

            if agg:
                pops = []
                for plugin in plugins:
                    method = self._method(plugin, 'crossover')
                    if method is None:
                        continue
                    try:
                        pops.append(method(population=population, **kwargs))
                    except Exception as e:
                        logging.warning(f'Crossover plugin {plugin} failed: {e}')
                        continue
                return np.concatenate(pops)

            # Sequential: each plugin crosses over the prior result
            for plugin in plugins:
                method = self._method(plugin, 'crossover')
                if method is None:
                    continue
                try:
                    population = method(population=population, **kwargs)
                except Exception as e:
                    logging.warning(f'Crossover plugin {plugin} failed: {e}')
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
            for plugin in self._iter_plugins('fitness'):
                method = self._method(plugin, 'fitness')
                if method is None:
                    continue
                try:
                    return method(population=population, points=points, **kwargs)
                except Exception as e:
                    logging.warning(f'Fitness plugin {plugin} failed: {e}')
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
            for plugin in self._iter_plugins('mutate'):
                method = self._method(plugin, 'mutate')
                if method is None:
                    continue
                try:
                    return method(population=population, mutation_rate=mutation_rate, **kwargs)
                except Exception as e:
                    logging.warning(f'Mutate plugin {plugin} failed: {e}')
                    continue

    def optimize(self, **kwargs) -> Dict[str, Any]:
        """Optimizes population

        Args:
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            dict

        """
        for plugin in self._iter_plugins('optimize'):
            method = self._method(plugin, 'optimize')
            if method is None:
                continue
            try:
                return method(ga=self, **kwargs)
            except Exception as e:
                logging.warning(f'Optimize plugin {plugin} failed: {e}')
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
            for plugin in self._iter_plugins('pool'):
                method = self._method(plugin, 'pool')
                if method is None:
                    continue
                try:
                    return method(csvpth=csvpth, **kwargs)
                except Exception as e:
                    logging.error(f'Pool plugin {plugin} failed: {e}')

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
            plugins = self._iter_plugins('populate')

            if agg:
                pops = []
                for plugin in plugins:
                    method = self._method(plugin, 'populate')
                    if method is None:
                        continue
                    try:
                        pops.append(method(**populate_kwargs))
                    except Exception as e:
                        logging.warning(f'Populate plugin {plugin} failed: {e}')
                        continue
                return np.concatenate(pops)

            # Use first valid populate plugin
            for plugin in plugins:
                method = self._method(plugin, 'populate')
                if method is None:
                    continue
                try:
                    return method(**populate_kwargs)
                except Exception as e:
                    logging.warning(f'Populate plugin {plugin} failed: {e}')
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
            for plugin in self._iter_plugins('pospool'):
                method = self._method(plugin, 'pospool')
                if method is None:
                    continue
                try:
                    return method(**pospool_kwargs)
                except Exception as e:
                    logging.warning(f'Pospool plugin {plugin} failed: {e}')
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
            for plugin in self._iter_plugins('select'):
                method = self._method(plugin, 'select')
                if method is None:
                    continue
                try:
                    return method(**select_kwargs)
                except Exception as e:
                    logging.warning(f'Select plugin {plugin} failed: {e}')
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
            # Chain validators: each filters the population in sequence
            for plugin in self._iter_plugins('validate'):
                method = self._method(plugin, 'validate')
                if method is None:
                    continue
                population = method(population=population, salaries=salaries, **kwargs)
            return population
