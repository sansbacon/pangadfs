# pangadfs/pangadfs/ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Union, Optional, List, Tuple
from functools import lru_cache
import time

import numpy as np
import pandas as pd

from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False


class GeneticAlgorithm:
    """Handles coordination of genetic algorithm plugins with performance optimizations"""

    PLUGIN_NAMESPACES = (
       'pool', 'pospool', 'populate', 'fitness', 'optimize',
       'select', 'crossover', 'mutate', 'validate'
    )

    REQUIRED_METHODS = {p: [p] for p in PLUGIN_NAMESPACES}

    VALIDATE_PLUGINS = ('validate_salary', 'validate_duplicates')

    # Plugin configuration mapping for more declarative and scalable plugin loading
    PLUGIN_CONFIG = {
        # Special configuration for validate namespace which uses NamedExtensionManager
        'validate': {
            'manager_type': 'named',
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
                 use_defaults: bool = False,
                 enable_caching: bool = True,
                 performance_monitoring: bool = False):
        """Creates GeneticAlgorithm instance with performance enhancements

        Args:
            ctx (dict): the context dict, AppConfig object, or other configuration scheme
            driver_managers (dict): key is namespace, value is DriverManager
            extension_managers (dict): key is namespace, value is NamedExtensionManager
            use_defaults (bool): use default plugins
            enable_caching (bool): enable result caching for expensive operations
            performance_monitoring (bool): enable performance monitoring and logging

        Returns:
            GeneticAlgorithm: the GA instance

        """
        logging.getLogger(__name__).addHandler(logging.NullHandler())

        # add context
        self.ctx = ctx

        # Performance enhancements
        self.enable_caching = enable_caching
        self.performance_monitoring = performance_monitoring
        self._operation_times = {} if performance_monitoring else None
        self._cache_hits = 0
        self._cache_misses = 0

        # add driver/extension managers
        self.driver_managers = driver_managers if driver_managers else {}
        self.extension_managers = extension_managers if extension_managers else {}

        # Cache for plugin objects to avoid repeated lookups
        self._plugin_cache = {}
        self._plugins_cache = {}

        # Pre-computed arrays cache
        self._array_cache = {}

        # if use_defaults, then load default plugin(s) for missing namespaces
        if use_defaults:
            self._load_plugins()

    def _time_operation(self, operation_name: str):
        """Decorator for timing operations when performance monitoring is enabled."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if self.performance_monitoring:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    
                    if operation_name not in self._operation_times:
                        self._operation_times[operation_name] = []
                    self._operation_times[operation_name].append(elapsed)
                    
                    # Log slow operations
                    if elapsed > 0.1:  # Log operations taking more than 100ms
                        logging.warning(f"Slow operation {operation_name}: {elapsed:.3f}s")
                    
                    return result
                else:
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def _load_plugins(self):
        """Loads default plugins for any namespace that doesn't have a plugin with optimizations"""
        for ns in self.PLUGIN_NAMESPACES:
            # Skip namespaces that already have managers
            if ns in self.driver_managers or ns in self.extension_managers:
                continue
                
            config = self.PLUGIN_CONFIG.get(ns, self.PLUGIN_CONFIG['default']).copy()
            manager_type = config.pop('manager_type', 'driver')
            
            try:
                if manager_type == 'named':
                    names = config.pop('names', [])
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
                    if 'name_template' in config:
                        config['name'] = config.pop('name_template').format(ns=ns)
                    
                    mgr = DriverManager(
                        namespace=f'pangadfs.{ns}',
                        invoke_args=(self.ctx,),
                        **config
                    )
                    
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
        """Validates that a plugin has the required methods for its namespace"""
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
        """Get a plugin for the specified namespace with caching"""
        # Check cache first
        if self.enable_caching and namespace in self._plugin_cache:
            self._cache_hits += 1
            return self._plugin_cache[namespace]
        
        if self.enable_caching:
            self._cache_misses += 1

        # Check if there's a driver manager for the namespace
        if mgr := self.driver_managers.get(namespace):
            plugin = mgr.driver
            if self.enable_caching:
                self._plugin_cache[namespace] = plugin
            return plugin
            
        # Check if there's an extension manager for the namespace
        if mgr := self.extension_managers.get(namespace):
            if mgr.extensions:
                plugin = mgr.extensions[0].obj
                if self.enable_caching:
                    self._plugin_cache[namespace] = plugin
                return plugin
                
        # No plugin found
        raise ValueError(f"No plugin found for namespace '{namespace}'")

    def get_plugins(self, namespace: str) -> Tuple[Any, ...]:
        """Get all plugins for the specified namespace with caching"""
        # Check cache first
        cache_key = f"{namespace}_all"
        if self.enable_caching and cache_key in self._plugins_cache:
            self._cache_hits += 1
            return self._plugins_cache[cache_key]
        
        if self.enable_caching:
            self._cache_misses += 1

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
        
        # Convert to tuple for caching (lists are not hashable)
        plugins_tuple = tuple(plugins)
        if self.enable_caching:
            self._plugins_cache[cache_key] = plugins_tuple
        return plugins_tuple

    def crossover(self,
                  *,
                  population: np.ndarray = None,
                  agg: bool = None,
                  **kwargs) -> np.ndarray:
        """Crossover operation with performance optimizations"""
        if self.performance_monitoring:
            start_time = time.time()

        # Validate inputs
        if population is None or len(population) == 0:
            raise ValueError("Population cannot be None or empty")

        # Ensure population is contiguous for better performance
        if not population.flags['C_CONTIGUOUS']:
            population = np.ascontiguousarray(population)

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            # Try to get a single plugin first
            plugin = self.get_plugin('crossover')
            result = plugin.crossover(**params, **kwargs)
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('crossover', elapsed)
            
            return result
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
                result = np.concatenate(pops)
                
                if self.performance_monitoring:
                    elapsed = time.time() - start_time
                    self._log_operation_time('crossover_agg', elapsed)
                
                return result
            except Exception as e:
                logging.error(f"Error in crossover plugins: {str(e)}")
                raise

        # otherwise, run crossover for each plugin sequentially
        try:
            plugins = self.get_plugins('crossover')
            population = params['population']
            for plugin in plugins:
                params['population'] = population
                population = plugin.crossover(**params, **kwargs)
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('crossover_sequential', elapsed)
            
            return population
        except Exception as e:
            logging.error(f"Error in crossover plugins: {str(e)}")
            raise

    def fitness(self, 
                *,
                population: np.ndarray = None, 
                points: np.ndarray = None, 
                **kwargs) -> np.ndarray:
        """Measures fitness of population with optimizations"""
        if self.performance_monitoring:
            start_time = time.time()

        # Validate inputs
        if population is None or points is None:
            raise ValueError("Population and points cannot be None")

        # Ensure arrays are contiguous for better performance
        if not population.flags['C_CONTIGUOUS']:
            population = np.ascontiguousarray(population)
        if not points.flags['C_CONTIGUOUS']:
            points = np.ascontiguousarray(points)

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            plugin = self.get_plugin('fitness')
            result = plugin.fitness(**params, **kwargs)
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('fitness', elapsed)
            
            return result
        except Exception as e:
            logging.error(f"Error in fitness plugin: {str(e)}")
            raise

    def mutate(self, 
               *,
               population: np.ndarray = None,
               mutation_rate: float = None,
               **kwargs) -> np.ndarray:
        """Mutates population with performance optimizations"""
        if self.performance_monitoring:
            start_time = time.time()

        # Validate inputs
        if population is None:
            raise ValueError("Population cannot be None")
        
        # Set default mutation rate if not provided
        if mutation_rate is None:
            mutation_rate = 0.05

        # Ensure population is contiguous for better performance
        if not population.flags['C_CONTIGUOUS']:
            population = np.ascontiguousarray(population)

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            plugin = self.get_plugin('mutate')
            result = plugin.mutate(**params, **kwargs)
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('mutate', elapsed)
            
            return result
        except Exception as e:
            logging.error(f"Error in mutate plugin: {str(e)}")
            raise

    def optimize(self, 
                 **kwargs) -> Dict[str, Any]:
        """Optimizes population with enhanced monitoring"""
        if self.performance_monitoring:
            start_time = time.time()

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params['ga'] = params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            plugin = self.get_plugin('optimize')
            result = plugin.optimize(**params, **kwargs)
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('optimize', elapsed)
                # Add performance stats to result
                result['ga_performance_stats'] = self.get_performance_stats()
            
            return result
        except Exception as e:
            logging.error(f"Error in optimize plugin: {str(e)}")
            raise
            
    def pool(self, *, csvpth: Path = None, **kwargs) -> pd.DataFrame:
        """Creates pool of players with caching"""
        if self.performance_monitoring:
            start_time = time.time()

        # Check cache for pool data
        cache_key = f"pool_{csvpth}_{hash(str(kwargs))}"
        if self.enable_caching and cache_key in self._array_cache:
            self._cache_hits += 1
            return self._array_cache[cache_key]

        if self.enable_caching:
            self._cache_misses += 1

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            plugin = self.get_plugin('pool')
            result = plugin.pool(**params, **kwargs)
            
            # Cache the result
            if self.enable_caching:
                self._array_cache[cache_key] = result
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('pool', elapsed)
            
            return result
        except Exception as e:
            logging.error(f"Error in pool plugin: {str(e)}")
            raise

    def populate(self,
                 *,
                 pospool: Dict[str, pd.DataFrame] = None,
                 posmap: Dict[str, int] = None,
                 population_size: int = None,
                 probcol: str = 'prob',
                 agg: bool = False,
                 **kwargs) -> np.ndarray:
        """Creates initial population with optimizations"""
        if self.performance_monitoring:
            start_time = time.time()

        # Validate inputs
        if pospool is None or posmap is None or population_size is None:
            raise ValueError("pospool, posmap, and population_size are required")

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        agg = params.pop('agg')
        kwargs = params.pop('kwargs')

        try:
            # Try to get a single plugin first
            plugin = self.get_plugin('populate')
            result = plugin.populate(**params, **kwargs)
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('populate', elapsed)
            
            return result
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
                result = np.concatenate(pops)
                
                if self.performance_monitoring:
                    elapsed = time.time() - start_time
                    self._log_operation_time('populate_agg', elapsed)
                
                return result
            except Exception as e:
                logging.error(f"Error in populate plugins: {str(e)}")
                raise

        # otherwise, use the first plugin
        try:
            plugins = self.get_plugins('populate')
            result = plugins[0].populate(**params, **kwargs)
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('populate_single', elapsed)
            
            return result
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
        """Divides pool into positional buckets with caching"""
        if self.performance_monitoring:
            start_time = time.time()

        # Check cache for pospool data
        cache_key = f"pospool_{id(pool)}_{hash(str(posfilter))}_{hash(str(column_mapping))}"
        if self.enable_caching and cache_key in self._array_cache:
            self._cache_hits += 1
            return self._array_cache[cache_key]

        if self.enable_caching:
            self._cache_misses += 1

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            plugin = self.get_plugin('pospool')
            result = plugin.pospool(**params, **kwargs)
            
            # Cache the result
            if self.enable_caching:
                self._array_cache[cache_key] = result
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('pospool', elapsed)
            
            return result
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
        """Selects/filters population with optimizations"""
        if self.performance_monitoring:
            start_time = time.time()

        # Validate inputs
        if population is None or population_fitness is None:
            raise ValueError("Population and population_fitness cannot be None")
        
        if n is None:
            n = len(population)

        # Ensure arrays are contiguous for better performance
        if not population.flags['C_CONTIGUOUS']:
            population = np.ascontiguousarray(population)
        if not population_fitness.flags['C_CONTIGUOUS']:
            population_fitness = np.ascontiguousarray(population_fitness)

        logging.debug('Selection method {}, n is {}'.format(method, n))
        logging.debug('Pop size {}, fitness {}'.format(len(population), population_fitness.mean()))

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        try:
            plugin = self.get_plugin('select')
            result = plugin.select(**params, **kwargs)
            
            if self.performance_monitoring:
                elapsed = time.time() - start_time
                self._log_operation_time('select', elapsed)
            
            return result
        except Exception as e:
            logging.error(f"Error in select plugin: {str(e)}")
            raise

    def validate(self,
                 *,
                 population: np.ndarray = None, 
                 salaries: np.ndarray = None, 
                 **kwargs) -> np.ndarray:
        """Validate lineup with performance optimizations and even population guarantee"""
        if self.performance_monitoring:
            start_time = time.time()

        # Validate inputs
        if population is None:
            raise ValueError("Population cannot be None")

        # Ensure arrays are contiguous for better performance
        if not population.flags['C_CONTIGUOUS']:
            population = np.ascontiguousarray(population)
        if salaries is not None and not salaries.flags['C_CONTIGUOUS']:
            salaries = np.ascontiguousarray(salaries)

        logging.debug(f'Salaries {salaries}')

        # combine keyword arguments with **kwargs
        params = locals().copy()
        params.pop('self', None)
        kwargs = params.pop('kwargs')

        validated_population = None

        try:
            # Try to get a single plugin first
            plugin = self.get_plugin('validate')
            validated_population = plugin.validate(**params, **kwargs)
            
        except ValueError:
            # No single plugin found, try to get all plugins
            try:
                plugins = self.get_plugins('validate')
                validated_population = params['population']
                for plugin in plugins:
                    params['population'] = validated_population
                    validated_population = plugin.validate(**params, **kwargs)
                    
            except Exception as e:
                logging.error(f"Error in validate plugins: {str(e)}")
                raise

        # Ensure even number of individuals for crossover compatibility
        if validated_population is not None and len(validated_population) % 2 != 0:
            if len(validated_population) > 1:
                # Remove the last individual to make it even
                validated_population = validated_population[:-1]
                logging.debug(f"Removed 1 individual to ensure even population size: {len(validated_population)}")
            else:
                # If only 1 individual left, duplicate it to make 2
                validated_population = np.vstack([validated_population, validated_population])
                logging.debug(f"Duplicated single individual to ensure even population size: {len(validated_population)}")

        if self.performance_monitoring:
            elapsed = time.time() - start_time
            self._log_operation_time('validate', elapsed)
        
        return validated_population

    def _log_operation_time(self, operation: str, elapsed: float):
        """Log operation time for performance monitoring"""
        if operation not in self._operation_times:
            self._operation_times[operation] = []
        self._operation_times[operation].append(elapsed)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.performance_monitoring:
            return {"performance_monitoring": False}

        stats = {
            "performance_monitoring": True,
            "cache_enabled": self.enable_caching,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "operation_stats": {}
        }

        for operation, times in self._operation_times.items():
            stats["operation_stats"][operation] = {
                "count": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            }

        return stats

    def clear_cache(self):
        """Clear all caches"""
        self._plugin_cache.clear()
        self._plugins_cache.clear()
        self._array_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def run(self, population: list, generations: int):
        """Run the genetic algorithm with performance monitoring"""
        if self.performance_monitoring:
            start_time = time.time()

        fitness_fn = self.get_plugin("fitness")
        select_fn = self.get_plugin("select")
        mutate_fn = self.get_plugin("mutate")
        crossover_fn = self.get_plugin("crossover")
    
        for generation in range(generations):
            fitness_scores = [fitness_fn.evaluate(ind) for ind in population]
            selected = select_fn.select(population, fitness_scores)
            offspring = crossover_fn.crossover(selected)
            population = mutate_fn.mutate(offspring)
    
        if self.performance_monitoring:
            elapsed = time.time() - start_time
            logging.info(f"GA run completed in {elapsed:.3f}s for {generations} generations")

        return population
        
    @staticmethod
    def list_available_plugins(namespace: str) -> List[str]:
        """List all available plugins for a given namespace"""
        from stevedore import ExtensionManager
        try:
            mgr = ExtensionManager(namespace=namespace, invoke_on_load=False)
            return [ext.name for ext in mgr.extensions]
        except Exception as e:
            logging.error(f"Error listing plugins for namespace {namespace}: {str(e)}")
            return []

    def benchmark_operations(self, iterations: int = 100) -> Dict[str, float]:
        """Benchmark key operations for performance analysis"""
        if not self.performance_monitoring:
            logging.warning("Performance monitoring must be enabled for benchmarking")
            return {}

        # Reset performance stats
        self._operation_times.clear()
        
        # Create sample data for benchmarking
        sample_population = np.random.randint(0, 100, size=(100, 10))
        sample_fitness = np.random.random(100)
        
        benchmark_results = {}
        
        # Benchmark plugin retrieval
        start_time = time.time()
        for _ in range(iterations):
            self.get_plugin('fitness')
        benchmark_results['plugin_retrieval'] = (time.time() - start_time) / iterations
        
        # Benchmark array operations
        start_time = time.time()
        for _ in range(iterations):
            np.ascontiguousarray(sample_population)
        benchmark_results['array_contiguous'] = (time.time() - start_time) / iterations
        
        return benchmark_results
