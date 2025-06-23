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


class GeneticAlgorithm:
    """Handles coordination of genetic algorithm plugins"""

    PLUGIN_NAMESPACES = (
       'pool', 'pospool', 'populate', 'fitness', 'optimize',
       'select', 'crossover', 'mutate', 'validate'
    )

    REQUIRED_METHODS = {p: [p] for p in PLUGIN_NAMESPACES}

    VALIDATE_PLUGINS = ('validate_salary', 'validate_duplicates')

    # Plugin configuration mapping for more declarative and scalable plugin loading
    # This defines how each namespace should be loaded
    # Each namespace can specify:
    # - manager_type: 'driver' or 'named' to determine which manager to use
    # - Additional parameters specific to the manager type
    PLUGIN_CONFIG = {
        # Special configuration for validate namespace which uses NamedExtensionManager
        'validate': {
            'manager_type': 'named',
            # Use class attribute for plugin names to maintain a single source of truth
            'names': VALIDATE_PLUGINS,
            'invoke_on_load': True,
            'name_order': True
        },
        # Default configuration for other namespaces which use DriverManager
        'default': {
            'manager_type': 'driver',
            'name_template': '{ns}_default',
            'invoke_on_load': True
        }
    }


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

        # if use_defaults, then load default plugin(s) for missing namespaces
        if use_defaults:
            self._load_plugins()

    def _load_plugins(self):
        """Loads default plugins for any namespace that doesn't have a plugin
        
        Uses the PLUGIN_CONFIG mapping to determine how to load each namespace.
        This provides a more declarative and scalable approach to plugin loading.
        
        Raises:
            ImportError: If a plugin cannot be loaded
            AttributeError: If a plugin is missing required methods
            Exception: For any other plugin loading errors
        """
        for ns in self.PLUGIN_NAMESPACES:
            # Skip namespaces that already have managers
            if ns in self.driver_managers or ns in self.extension_managers:
                continue
                
            # Get the configuration for this namespace, or use the default if not specified
            config = self.PLUGIN_CONFIG.get(ns, self.PLUGIN_CONFIG['default']).copy()
            
            # Extract the manager type and remove it from the config
            manager_type = config.pop('manager_type', 'driver')
            
            try:
                # Create the appropriate manager based on the configuration
                if manager_type == 'named':
                    # For NamedExtensionManager, we need the names parameter
                    names = config.pop('names', [])
                    
                    # Create the manager with the namespace and remaining config parameters
                    manager = NamedExtensionManager(
                        namespace=f'pangadfs.{ns}',
                        names=names,
                        invoke_args=(self.ctx,),
                        **config
                    )
                    
                    # Validate each extension in the manager
                    for ext in manager.extensions:
                        self._validate_plugin(ns, ext.obj)
                    
                    self.extension_managers[ns] = manager
                    
                else:  # driver manager is the default
                    # For DriverManager, we need to format the name parameter if a template is provided
                    if 'name_template' in config:
                        config['name'] = config.pop('name_template').format(ns=ns)
                    
                    # Create the manager with the namespace and remaining config parameters
                    mgr = DriverManager(
                        namespace=f'pangadfs.{ns}',
                        invoke_args=(self.ctx,),
                        **config
                    )
                    
                    # Validate the driver
                    self._validate_plugin(ns, mgr.driver)
                    
                    self.driver_managers[ns] = mgr
                    
                logging.info(f"Successfully loaded plugin(s) for namespace '{ns}'")
                
            except ImportError as e:
                logging.error(f"Failed to import plugin for namespace '{ns}': {str(e)}")
                raise
            except AttributeError as e:
                logging.error(f"Plugin for namespace '{ns}' is missing required methods: {str(e)}")
                raise
            except Exception as e:
                logging.error(f"Error loading plugin for namespace '{ns}': {str(e)}")
                raise

    def _validate_plugin(self, namespace: str, plugin_obj: Any) -> None:
        """Validates that a plugin has the required methods for its namespace
        
        Args:
            namespace (str): The plugin namespace
            plugin_obj (Any): The plugin object to validate
            
        Raises:
            AttributeError: If the plugin is missing required methods
        """
        required_methods = self.REQUIRED_METHODS.get(namespace, [])
        missing_methods = []
        
        for method_name in required_methods:
            if not hasattr(plugin_obj, method_name):
                missing_methods.append(method_name)
            elif not callable(getattr(plugin_obj, method_name)):
                missing_methods.append(f"{method_name} (not callable)")
                
        if missing_methods:
            raise AttributeError(
                f"Plugin for namespace '{namespace}' is missing required methods: {', '.join(missing_methods)}"
            )
            
    def get_plugin(self, namespace: str) -> Any:
        """Get a plugin for the specified namespace
        
        This method provides a clean API for accessing plugins. It first checks
        if there's a driver manager for the namespace, and if not, it checks if
        there's an extension manager and returns the first extension.
        
        Args:
            namespace (str): The plugin namespace
            
        Returns:
            Any: The plugin object
            
        Raises:
            ValueError: If no plugin is found for the namespace
        """
        # Check if there's a driver manager for the namespace
        if mgr := self.driver_managers.get(namespace):
            return mgr.driver
            
        # Check if there's an extension manager for the namespace
        if mgr := self.extension_managers.get(namespace):
            if mgr.extensions:
                return mgr.extensions[0].obj
                
        # No plugin found
        raise ValueError(f"No plugin found for namespace '{namespace}'")
        
    def get_plugins(self, namespace: str) -> list:
        """Get all plugins for the specified namespace
        
        This method is useful for namespaces that can have multiple plugins,
        such as 'validate'.
        
        Args:
            namespace (str): The plugin namespace
            
        Returns:
            list: A list of plugin objects
            
        Raises:
            ValueError: If no plugins are found for the namespace
        """
        plugins = []
        
        # Check if there's a driver manager for the namespace
        if mgr := self.driver_managers.get(namespace):
            plugins.append(mgr.driver)
            
        # Check if there's an extension manager for the namespace
        if mgr := self.extension_managers.get(namespace):
            plugins.extend([ext.obj for ext in mgr.extensions])
            
        # No plugins found
        if not plugins:
            raise ValueError(f"No plugins found for namespace '{namespace}'")
            
        return plugins
    
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
        logging.debug('{} {}'.format(population, agg))

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            # Try to get a single plugin first
            plugin = self.get_plugin('crossover')
            return plugin.crossover(**params, **kwargs)
        except ValueError:
            # No single plugin found, try to get all plugins
            pass

        # if agg=True, then aggregate crossed over populations
        if kwargs.get('agg'):
            try:
                plugins = self.get_plugins('crossover')
                pops = []
                for plugin in plugins:
                    pops.append(plugin.crossover(**params, **kwargs))
                return np.concatenate(pops)
            except Exception as e:
                logging.error(f"Error in crossover plugins: {str(e)}")
                raise

        # otherwise, run crossover for each plugin sequentially
        # after first time, crosses over prior crossed-over population
        try:
            plugins = self.get_plugins('crossover')
            population = params['population']
            for plugin in plugins:
                params['population'] = population
                population = plugin.crossover(**params, **kwargs)
            return population
        except Exception as e:
            logging.error(f"Error in crossover plugins: {str(e)}")
            raise

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
        logging.debug('{} {}'.format(population, points))

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            # Get the fitness plugin and call its fitness method
            plugin = self.get_plugin('fitness')
            return plugin.fitness(**params, **kwargs)
        except Exception as e:
            logging.error(f"Error in fitness plugin: {str(e)}")
            raise

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
        logging.debug('{} {} {} {}'.format(population, mutation_rate))

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            # Get the mutate plugin and call its mutate method
            plugin = self.get_plugin('mutate')
            return plugin.mutate(**params, **kwargs)
        except Exception as e:
            logging.error(f"Error in mutate plugin: {str(e)}")
            raise

    def optimize(self, 
                 **kwargs) -> Dict[str, Any]:
        """Optimizes population

        Args:
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            dict

        """
        # combine keyword arguments with **kwargs
        # need to figure out best way to pass ga to optimize
        params = locals().copy()
        params['ga'] = params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            # Get the optimize plugin and call its optimize method
            plugin = self.get_plugin('optimize')
            return plugin.optimize(**params, **kwargs)
        except Exception as e:
            logging.error(f"Error in optimize plugin: {str(e)}")
            raise
            
    def pool(self, *, csvpth: Path = None, **kwargs) -> pd.DataFrame:
        """Creates pool of players.

        Args:
            csvpth (Path): the path of the datafile
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            pd.DataFrame: initial pool of players
        
        """
        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            # Get the pool plugin and call its pool method
            plugin = self.get_plugin('pool')
            return plugin.pool(**params, **kwargs)
        except Exception as e:
            logging.error(f"Error in pool plugin: {str(e)}")
            raise

    def populate(self,
                 *,
                 pospool: Dict[str, pd.DataFrame] = None,
                 posmap: Dict[str, int] = None,
                 population_size: int = None,
                 probcol: str = 'prob',
                 agg: bool = 'False',
                 **kwargs) -> np.ndarray:
        """Creates initial population of specified size
        
        Args:
            pospool (Dict[str, pd.DataFrame]): pool segmented by position
            posmap (Dict[str, int]): positions & accompanying roster slots
            population_size (int): number of individuals to create
            probcol (str): the dataframe column with probabilities, default 'probs'
            agg (bool): default False. Aggregate multiple crossovers if True.
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: the population

        """
        logging.debug('{} {} {} {}'.format(pospool, posmap, population_size, probcol))

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        agg = params.pop('agg')
        kwargs = params.pop('kwargs')

        try:
            # Try to get a single plugin first
            plugin = self.get_plugin('populate')
            return plugin.populate(**params, **kwargs)
        except ValueError:
            # No single plugin found, try to get all plugins
            pass

        # if agg=True, then aggregate populations
        if agg:
            try:
                plugins = self.get_plugins('populate')
                pops = []
                for plugin in plugins:
                    pops.append(plugin.populate(**params, **kwargs))
                return np.concatenate(pops)
            except Exception as e:
                logging.error(f"Error in populate plugins: {str(e)}")
                raise

        # otherwise, use the first plugin
        try:
            plugins = self.get_plugins('populate')
            return plugins[0].populate(**params, **kwargs)
        except Exception as e:
            logging.error(f"Error in populate plugin: {str(e)}")
            raise

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
        logging.debug('{} {} {} {}'.format(pool, posfilter, column_mapping, flex_positions))

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            # Get the pospool plugin and call its pospool method
            plugin = self.get_plugin('pospool')
            return plugin.pospool(**params, **kwargs)
        except Exception as e:
            logging.error(f"Error in pospool plugin: {str(e)}")
            raise
        
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
            method (str): the selection method, default roulette
            **kwargs: Keyword arguments for plugins (other than default)

        Returns:
            np.ndarray: population fitness as 1D array of float

        """
        logging.debug('Selection method {}, n is {}'.format(method, n))
        logging.debug('Pop size {}, fitness {}'.format(len(population), population_fitness.mean()))

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            # Get the select plugin and call its select method
            plugin = self.get_plugin('select')
            return plugin.select(**params, **kwargs)
        except Exception as e:
            logging.error(f"Error in select plugin: {str(e)}")
            raise

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
        logging.debug(f'Salaries {salaries}')

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            # Try to get a single plugin first
            plugin = self.get_plugin('validate')
            return plugin.validate(**params, **kwargs)
        except ValueError:
            # No single plugin found, try to get all plugins
            pass
            
        # For validate, we need to run all plugins sequentially
        try:
            plugins = self.get_plugins('validate')
            population = params['population']
            for plugin in plugins:
                params['population'] = population
                population = plugin.validate(**params, **kwargs)
            return population
        except Exception as e:
            logging.error(f"Error in validate plugins: {str(e)}")
            raise
            
    def run(self, population: list, generations: int):
        """Run the genetic algorithm for a specified number of generations
        
        This method demonstrates the use of the get_plugin method to get
        the plugins needed for the genetic algorithm.
        
        Args:
            population (list): The initial population
            generations (int): The number of generations to run
            
        Returns:
            list: The final population
        """
        fitness_fn = self.get_plugin("fitness")
        select_fn = self.get_plugin("select")
        mutate_fn = self.get_plugin("mutate")
        crossover_fn = self.get_plugin("crossover")
    
        for _ in range(generations):
            fitness_scores = [fitness_fn.evaluate(ind) for ind in population]
            selected = select_fn.select(population, fitness_scores)
            offspring = crossover_fn.crossover(selected)
            population = mutate_fn.mutate(offspring)
    
        return population
        
    @staticmethod
    def list_available_plugins(namespace: str):
        """List all available plugins for a given namespace
        
        Args:
            namespace (str): The plugin namespace
            
        Returns:
            list: A list of plugin names
        """
        from stevedore import ExtensionManager
        mgr = ExtensionManager(namespace=namespace, invoke_on_load=False)
        return [ext.name for ext in mgr.extensions]
