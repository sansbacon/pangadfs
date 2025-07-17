# pangadfs/pangadfs/duplicates.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

"""
Optimized duplicate detection and removal functions for genetic algorithms.

This module provides high-performance duplicate detection and removal functionality
that can be used throughout the genetic algorithm lifecycle, including:
- Initial population generation
- Post-crossover cleanup
- Post-mutation validation
- Between-generation maintenance
- Final population validation
"""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    logger.debug("Numba is available for duplicate detection JIT compilation")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.debug("Numba not available, using fallback implementations")
    # Fallback if Numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# JIT-compiled helper functions for performance
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _find_non_duplicate_flex_jit(main_lineups: np.ndarray, flex_options: np.ndarray) -> np.ndarray:
        """JIT-compiled FLEX duplicate removal for large populations"""
        n_lineups, lineup_size = main_lineups.shape
        n_flex_options = flex_options.shape[1]
        result = np.zeros(n_lineups, dtype=flex_options.dtype)
        
        for i in prange(n_lineups):
            lineup = main_lineups[i]
            flex_candidates = flex_options[i]
            
            # Find first FLEX player not in main lineup
            for j in range(n_flex_options):
                flex_player = flex_candidates[j]
                is_duplicate = False
                
                # Check if flex player is already in main lineup
                for k in range(lineup_size):
                    if lineup[k] == flex_player:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    result[i] = flex_player
                    break
            else:
                # If all FLEX players are duplicates, use the first one
                result[i] = flex_candidates[0]
        
        return result

    @njit(parallel=True, fastmath=True)
    def _remove_internal_duplicates_jit(population: np.ndarray) -> np.ndarray:
        """JIT-compiled internal duplicate removal"""
        n_individuals, n_positions = population.shape
        valid_mask = np.ones(n_individuals, dtype=np.bool_)
        
        for i in prange(n_individuals):
            lineup = population[i]
            # Check for internal duplicates
            for j in range(n_positions):
                for k in range(j + 1, n_positions):
                    if lineup[j] == lineup[k]:
                        valid_mask[i] = False
                        break
                if not valid_mask[i]:
                    break
        
        return population[valid_mask]

    @njit(parallel=True, fastmath=True)
    def _find_duplicate_lineup_indices_jit(population: np.ndarray) -> np.ndarray:
        """JIT-compiled duplicate lineup detection"""
        n_individuals = population.shape[0]
        duplicate_mask = np.zeros(n_individuals, dtype=np.bool_)
        
        for i in prange(n_individuals):
            if not duplicate_mask[i]:  # Only check if not already marked as duplicate
                for j in range(i + 1, n_individuals):
                    if not duplicate_mask[j]:  # Only check if not already marked
                        # Compare lineups element by element
                        is_same = True
                        for k in range(population.shape[1]):
                            if population[i, k] != population[j, k]:
                                is_same = False
                                break
                        if is_same:
                            duplicate_mask[j] = True
        
        return duplicate_mask


def find_non_duplicate_flex(main_lineups: np.ndarray, 
                           flex_options: np.ndarray,
                           use_jit: Optional[bool] = None) -> np.ndarray:
    """Find non-duplicate FLEX players for each lineup.
    
    Args:
        main_lineups (np.ndarray): Main lineup positions, shape (n_lineups, n_positions)
        flex_options (np.ndarray): FLEX player options, shape (n_lineups, n_flex_options)
        use_jit (bool, optional): Force JIT usage. If None, auto-selects based on data size.
        
    Returns:
        np.ndarray: Valid FLEX players, shape (n_lineups,)
    """
    if main_lineups.shape[0] != flex_options.shape[0]:
        raise ValueError("Number of lineups must match between main_lineups and flex_options")
    
    # Auto-select algorithm based on data size and availability
    if use_jit is None:
        use_jit = NUMBA_AVAILABLE and len(main_lineups) > 50
    
    if use_jit and NUMBA_AVAILABLE:
        return _find_non_duplicate_flex_jit(main_lineups, flex_options)
    else:
        return _find_non_duplicate_flex_numpy(main_lineups, flex_options)


def _find_non_duplicate_flex_numpy(main_lineups: np.ndarray, flex_options: np.ndarray) -> np.ndarray:
    """Numpy-based FLEX duplicate removal for smaller populations"""
    n_lineups = len(main_lineups)
    result = np.zeros(n_lineups, dtype=flex_options.dtype)
    
    for i in range(n_lineups):
        lineup = main_lineups[i]
        flex_candidates = flex_options[i]
        
        # Find first FLEX player not in main lineup using vectorized operations
        is_duplicate = np.isin(flex_candidates, lineup)
        non_duplicate_indices = np.where(~is_duplicate)[0]
        
        if len(non_duplicate_indices) > 0:
            result[i] = flex_candidates[non_duplicate_indices[0]]
        else:
            # If all FLEX players are duplicates, use the first one
            result[i] = flex_candidates[0]
    
    return result


def remove_internal_duplicates(population: np.ndarray, 
                              use_jit: Optional[bool] = None) -> np.ndarray:
    """Remove lineups that have duplicate players in different positions.
    
    Args:
        population (np.ndarray): Population to clean, shape (n_individuals, n_positions)
        use_jit (bool, optional): Force JIT usage. If None, auto-selects based on data size.
        
    Returns:
        np.ndarray: Population with internal duplicates removed
    """
    if population.size == 0:
        return population
    
    # Auto-select algorithm based on data size and availability
    if use_jit is None:
        use_jit = NUMBA_AVAILABLE and len(population) > 100
    
    if use_jit and NUMBA_AVAILABLE:
        return _remove_internal_duplicates_jit(population)
    else:
        return _remove_internal_duplicates_numpy(population)


def _remove_internal_duplicates_numpy(population: np.ndarray) -> np.ndarray:
    """Numpy-based internal duplicate removal"""
    if population.size == 0:
        return population
    
    # Check for internal duplicates using vectorized operations
    population_sorted = np.sort(population, axis=1)
    # Check if any adjacent elements in sorted rows are equal
    has_duplicates = (population_sorted[:, 1:] == population_sorted[:, :-1]).any(axis=1)
    
    return population[~has_duplicates]


def remove_duplicate_lineups(population: np.ndarray, 
                            use_jit: Optional[bool] = None,
                            return_indices: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Remove duplicate lineups from population.
    
    Args:
        population (np.ndarray): Population to clean, shape (n_individuals, n_positions)
        use_jit (bool, optional): Force JIT usage. If None, auto-selects based on data size.
        return_indices (bool): If True, also return indices of kept lineups
        
    Returns:
        np.ndarray or Tuple[np.ndarray, np.ndarray]: Unique population, optionally with indices
    """
    if population.size == 0:
        if return_indices:
            return population, np.array([], dtype=int)
        return population
    
    # Auto-select algorithm based on data size and availability
    if use_jit is None:
        use_jit = NUMBA_AVAILABLE and len(population) > 200
    
    if use_jit and NUMBA_AVAILABLE:
        duplicate_mask = _find_duplicate_lineup_indices_jit(population)
        unique_indices = np.where(~duplicate_mask)[0]
        unique_population = population[unique_indices]
    else:
        unique_population, unique_indices = _remove_duplicate_lineups_numpy(population)
    
    if return_indices:
        return unique_population, unique_indices
    return unique_population


def _remove_duplicate_lineups_numpy(population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy-based duplicate lineup removal"""
    # Sort each lineup to enable comparison
    sorted_population = np.sort(population, axis=1)
    
    # Find unique lineups
    unique_lineups, unique_indices = np.unique(sorted_population, axis=0, return_index=True)
    
    # Return original (unsorted) lineups at unique indices
    return population[unique_indices], unique_indices


def find_duplicate_indices(population: np.ndarray, 
                          use_jit: Optional[bool] = None) -> np.ndarray:
    """Find indices of duplicate lineups without removing them.
    
    Args:
        population (np.ndarray): Population to analyze, shape (n_individuals, n_positions)
        use_jit (bool, optional): Force JIT usage. If None, auto-selects based on data size.
        
    Returns:
        np.ndarray: Boolean mask where True indicates duplicate lineups
    """
    if population.size == 0:
        return np.array([], dtype=bool)
    
    # Auto-select algorithm based on data size and availability
    if use_jit is None:
        use_jit = NUMBA_AVAILABLE and len(population) > 200
    
    if use_jit and NUMBA_AVAILABLE:
        return _find_duplicate_lineup_indices_jit(population)
    else:
        return _find_duplicate_indices_numpy(population)


def _find_duplicate_indices_numpy(population: np.ndarray) -> np.ndarray:
    """Numpy-based duplicate index finding"""
    # Sort each lineup to enable comparison
    sorted_population = np.sort(population, axis=1)
    
    # Find unique lineups and their indices
    _, unique_indices = np.unique(sorted_population, axis=0, return_index=True)
    
    # Create mask for duplicates
    duplicate_mask = np.ones(len(population), dtype=bool)
    duplicate_mask[unique_indices] = False
    
    return duplicate_mask


def validate_flex_positions(population: np.ndarray, 
                           flex_column_index: int,
                           main_position_indices: Optional[list] = None) -> np.ndarray:
    """Validate that FLEX positions don't conflict with main positions.
    
    Args:
        population (np.ndarray): Population to validate
        flex_column_index (int): Column index of FLEX position
        main_position_indices (list, optional): Indices of main positions to check against
        
    Returns:
        np.ndarray: Boolean mask where True indicates valid lineups
    """
    if population.size == 0:
        return np.array([], dtype=bool)
    
    if main_position_indices is None:
        # Check against all other positions
        main_position_indices = [i for i in range(population.shape[1]) if i != flex_column_index]
    
    valid_mask = np.ones(len(population), dtype=bool)
    
    for i, lineup in enumerate(population):
        flex_player = lineup[flex_column_index]
        main_players = lineup[main_position_indices]
        
        # Check if FLEX player appears in main positions
        if flex_player in main_players:
            valid_mask[i] = False
    
    return valid_mask


class DuplicateRemover:
    """Class-based interface for duplicate removal with configuration options."""
    
    def __init__(self, 
                 use_jit: Optional[bool] = None,
                 batch_size: int = 1000,
                 auto_optimize: bool = True):
        """Initialize duplicate remover with configuration.
        
        Args:
            use_jit (bool, optional): Force JIT usage. If None, auto-selects.
            batch_size (int): Batch size for large population processing
            auto_optimize (bool): Automatically choose best algorithm based on data
        """
        self.use_jit = use_jit
        self.batch_size = batch_size
        self.auto_optimize = auto_optimize
        
    def remove_duplicates(self, 
                         population: np.ndarray,
                         remove_internal: bool = True,
                         remove_lineup_duplicates: bool = True) -> np.ndarray:
        """Remove duplicates from population with configurable options.
        
        Args:
            population (np.ndarray): Population to clean
            remove_internal (bool): Remove lineups with internal duplicates
            remove_lineup_duplicates (bool): Remove duplicate lineups
            
        Returns:
            np.ndarray: Cleaned population
        """
        if population.size == 0:
            return population
        
        result = population.copy()
        
        # Process in batches if population is large
        if len(population) > self.batch_size:
            return self._remove_duplicates_batched(result, remove_internal, remove_lineup_duplicates)
        
        # Remove internal duplicates first
        if remove_internal:
            result = remove_internal_duplicates(result, use_jit=self.use_jit)
        
        # Remove duplicate lineups
        if remove_lineup_duplicates:
            result = remove_duplicate_lineups(result, use_jit=self.use_jit)
        
        return result
    
    def _remove_duplicates_batched(self, 
                                  population: np.ndarray,
                                  remove_internal: bool,
                                  remove_lineup_duplicates: bool) -> np.ndarray:
        """Process large populations in batches."""
        cleaned_batches = []
        
        for i in range(0, len(population), self.batch_size):
            batch = population[i:i + self.batch_size]
            
            if remove_internal:
                batch = remove_internal_duplicates(batch, use_jit=self.use_jit)
            
            if remove_lineup_duplicates:
                batch = remove_duplicate_lineups(batch, use_jit=self.use_jit)
            
            cleaned_batches.append(batch)
        
        # Combine batches and do final duplicate removal
        result = np.vstack(cleaned_batches)
        
        if remove_lineup_duplicates:
            result = remove_duplicate_lineups(result, use_jit=self.use_jit)
        
        return result
    
    def validate_population(self, population: np.ndarray) -> Dict[str, Union[int, float]]:
        """Analyze population for duplicate statistics.
        
        Args:
            population (np.ndarray): Population to analyze
            
        Returns:
            dict: Statistics about duplicates in the population
        """
        if population.size == 0:
            return {
                'total_lineups': 0,
                'unique_lineups': 0,
                'duplicate_lineups': 0,
                'internal_duplicates': 0,
                'duplicate_percentage': 0.0
            }
        
        total_lineups = len(population)
        
        # Find lineup duplicates
        duplicate_mask = find_duplicate_indices(population, use_jit=self.use_jit)
        duplicate_lineups = np.sum(duplicate_mask)
        unique_lineups = total_lineups - duplicate_lineups
        
        # Find internal duplicates
        clean_internal = remove_internal_duplicates(population, use_jit=self.use_jit)
        internal_duplicates = total_lineups - len(clean_internal)
        
        return {
            'total_lineups': total_lineups,
            'unique_lineups': unique_lineups,
            'duplicate_lineups': duplicate_lineups,
            'internal_duplicates': internal_duplicates,
            'duplicate_percentage': (duplicate_lineups / total_lineups) * 100.0
        }


# Utility functions for performance analysis
def benchmark_duplicate_removal(population_sizes: list, 
                               iterations: int = 3,
                               duplicate_rate: float = 0.1) -> Dict:
    """Benchmark duplicate removal performance across different population sizes.
    
    Args:
        population_sizes (list): List of population sizes to test
        iterations (int): Number of iterations per size
        duplicate_rate (float): Approximate rate of duplicates to inject
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    results = {
        'population_sizes': population_sizes,
        'numpy_times': [],
        'jit_times': [] if NUMBA_AVAILABLE else None,
        'memory_estimates': []
    }
    
    for size in population_sizes:
        # Generate test population with some duplicates
        population = np.random.randint(0, 100, size=(size, 9))
        
        # Inject some duplicates
        n_duplicates = int(size * duplicate_rate)
        if n_duplicates > 0:
            duplicate_indices = np.random.choice(size, n_duplicates, replace=False)
            source_indices = np.random.choice(size, n_duplicates, replace=True)
            population[duplicate_indices] = population[source_indices]
        
        # Benchmark numpy implementation
        numpy_times = []
        for _ in range(iterations):
            start_time = time.time()
            _ = remove_duplicate_lineups(population, use_jit=False)
            numpy_times.append(time.time() - start_time)
        
        results['numpy_times'].append(np.mean(numpy_times))
        
        # Benchmark JIT implementation if available
        if NUMBA_AVAILABLE:
            jit_times = []
            for _ in range(iterations):
                start_time = time.time()
                _ = remove_duplicate_lineups(population, use_jit=True)
                jit_times.append(time.time() - start_time)
            
            results['jit_times'].append(np.mean(jit_times))
        
        # Estimate memory usage
        memory_estimate = population.nbytes
        results['memory_estimates'].append(memory_estimate)
    
    return results


# Export main functions and classes
__all__ = [
    'find_non_duplicate_flex',
    'remove_internal_duplicates', 
    'remove_duplicate_lineups',
    'find_duplicate_indices',
    'validate_flex_positions',
    'DuplicateRemover',
    'benchmark_duplicate_removal'
]
