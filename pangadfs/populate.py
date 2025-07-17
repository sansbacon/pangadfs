# pangadfs/pangadfs/populate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import functools
import logging
import time
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pangadfs.base import PopulateBase
from pangadfs.misc import multidimensional_shifting
from pangadfs.duplicates import find_non_duplicate_flex

# Set up logging
logger = logging.getLogger(__name__)


def profile_populate(func):
    """Optional decorator to profile population generation performance"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, '_profile', False):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Population generation took {execution_time:.4f} seconds")
            return result
        else:
            return func(self, *args, **kwargs)
    return wrapper


class PopulateDefault(PopulateBase):
    """Optimized population generation with enhanced performance."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._profile = False
        self._batch_size = 1000
        self._use_parallel = True

    @profile_populate
    def populate(self,
                 *, 
                 pospool, 
                 posmap: Dict[str, int], 
                 population_size: int, 
                 probcol: str = 'prob',
                 batch_size: Optional[int] = None,
                 use_parallel: bool = True,
                 profile: bool = False,
                 **kwargs) -> np.ndarray:
        """Creates individuals in population with optimized performance
        
        Args:
            pospool (Dict[str, DataFrame]): pool split into positions
            posmap (Dict[str, int]): positions & accompanying roster slots
            population_size (int): number of individuals to create
            probcol (str): the dataframe column with probabilities
            batch_size (int, optional): batch size for large populations
            use_parallel (bool): whether to use parallel processing for position sampling
            profile (bool): whether to enable performance profiling
            **kwargs: keyword arguments

        Returns:
            ndarray of size (population_size, sum(posmap.values()))

        """
        # Set instance options
        self._profile = profile
        if batch_size is not None:
            self._batch_size = batch_size
        self._use_parallel = use_parallel
        
        # Handle large populations with batch processing
        if population_size > self._batch_size:
            return self._generate_population_batched(pospool, posmap, population_size, probcol, **kwargs)
        else:
            return self._generate_population_single(pospool, posmap, population_size, probcol, **kwargs)

    def _generate_population_single(self, pospool, posmap, population_size, probcol, **kwargs) -> np.ndarray:
        """Generate population in a single batch"""
        
        # Generate position samples - use parallel processing if beneficial
        if self._use_parallel and len(posmap) > 2 and population_size > 100:
            pos_samples = self._generate_position_samples_parallel(pospool, posmap, population_size, probcol)
        else:
            pos_samples = self._generate_position_samples_sequential(pospool, posmap, population_size, probcol)
        
        # Handle FLEX position with optimized duplicate removal
        return self._assemble_population_with_flex(pos_samples, posmap)

    def _generate_population_batched(self, pospool, posmap, population_size, probcol, **kwargs) -> np.ndarray:
        """Generate large populations using batch processing"""
        
        populations = []
        remaining_size = population_size
        batch_count = 0
        
        while remaining_size > 0:
            current_batch_size = min(self._batch_size, remaining_size)
            
            batch_population = self._generate_population_single(
                pospool, posmap, current_batch_size, probcol, **kwargs)
            
            populations.append(batch_population)
            remaining_size -= current_batch_size
            batch_count += 1
            
            if self._profile:
                logger.debug(f"Generated batch {batch_count}, remaining: {remaining_size}")
        
        # Combine all batches
        return np.vstack(populations)

    def _generate_position_samples_parallel(self, pospool, posmap, population_size, probcol) -> Dict[str, np.ndarray]:
        """Generate position samples using parallel processing"""
        
        pos_samples = {}
        
        with ThreadPoolExecutor(max_workers=min(4, len(posmap))) as executor:
            futures = {}
            
            for pos, n in posmap.items():
                future = executor.submit(
                    multidimensional_shifting,
                    pospool[pos].index,
                    population_size,
                    n,
                    pospool[pos][probcol]
                )
                futures[pos] = future
            
            # Collect results
            for pos, future in futures.items():
                try:
                    pos_samples[pos] = future.result()
                except Exception as e:
                    logger.warning(f"Parallel generation failed for position {pos}: {str(e)}, falling back to sequential")
                    # Fallback to sequential generation
                    pos_samples[pos] = multidimensional_shifting(
                        pospool[pos].index, population_size, posmap[pos], pospool[pos][probcol])
        
        return pos_samples

    def _generate_position_samples_sequential(self, pospool, posmap, population_size, probcol) -> Dict[str, np.ndarray]:
        """Generate position samples sequentially"""
        
        return {
            pos: multidimensional_shifting(pospool[pos].index, population_size, n, pospool[pos][probcol])
            for pos, n in posmap.items()
        }

    def _assemble_population_with_flex(self, pos_samples: Dict[str, np.ndarray], posmap: Dict[str, int]) -> np.ndarray:
        """Assemble population with optimized FLEX handling"""
        
        # Concatenate non-FLEX positions
        non_flex_positions = [pos for pos in posmap if pos != 'FLEX']
        if non_flex_positions:
            pop = np.concatenate([pos_samples[pos] for pos in non_flex_positions], axis=1)
        else:
            # Handle edge case where only FLEX exists
            return pos_samples['FLEX']
        
        # Handle FLEX position if it exists
        if 'FLEX' in pos_samples:
            flex_samples = pos_samples['FLEX']
            
            # Use the optimized duplicate removal from duplicates module
            valid_flex = find_non_duplicate_flex(pop, flex_samples)
            valid_flex = valid_flex.reshape(-1, 1)  # Ensure 2D for column_stack
            
            # Combine with main population
            return np.column_stack((pop, valid_flex))
        else:
            return pop

    def set_batch_size(self, batch_size: int):
        """Set the batch size for large population generation"""
        self._batch_size = batch_size

    def set_parallel_processing(self, use_parallel: bool):
        """Enable or disable parallel processing for position sampling"""
        self._use_parallel = use_parallel

    def enable_profiling(self, enable: bool = True):
        """Enable or disable performance profiling"""
        self._profile = enable


# Utility functions for performance analysis
def benchmark_populate_performance(pospool, posmap, population_sizes, iterations=3):
    """Benchmark population generation performance across different sizes
    
    Args:
        pospool: Position pool data
        posmap: Position mapping
        population_sizes: List of population sizes to test
        iterations: Number of iterations per size
        
    Returns:
        dict: Benchmark results
    """
    populate = PopulateDefault()
    populate.enable_profiling(True)
    
    results = {
        'population_sizes': population_sizes,
        'execution_times': [],
        'memory_estimates': []
    }
    
    for size in population_sizes:
        times = []
        for _ in range(iterations):
            start_time = time.time()
            
            try:
                population = populate.populate(
                    pospool=pospool,
                    posmap=posmap,
                    population_size=size
                )
                execution_time = time.time() - start_time
                times.append(execution_time)
            except Exception as e:
                logger.error(f"Error in benchmark: {str(e)}")
                times.append(float('inf'))
        
        avg_time = np.mean(times) if times else float('inf')
        results['execution_times'].append(avg_time)
        
        # Estimate memory usage
        if times and times[0] != float('inf'):
            memory_estimate = size * sum(posmap.values()) * 8  # Rough estimate in bytes
            results['memory_estimates'].append(memory_estimate)
        else:
            results['memory_estimates'].append(0)
    
    return results


# Export classes and functions
__all__ = [
    'PopulateDefault',
    'benchmark_populate_performance',
    'profile_populate'
]
