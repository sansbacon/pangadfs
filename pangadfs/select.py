# pangadfs/pangadfs/select.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Optional, Tuple, Union
import numpy as np

from pangadfs.base import SelectBase
from pangadfs.misc import diversity

# Set up logging
logger = logging.getLogger(__name__)

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    logger.debug("Numba is available for selection JIT compilation")
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
    def _tournament_selection_jit(population_fitness: np.ndarray, 
                                 tournament_size: int, 
                                 n_tournaments: int) -> np.ndarray:
        """JIT-compiled tournament selection for large populations"""
        winners = np.empty(n_tournaments, dtype=np.int32)
        
        for i in prange(n_tournaments):
            # Random tournament participants
            tournament_indices = np.random.choice(len(population_fitness), tournament_size)
            
            # Find winner (highest fitness)
            best_idx = 0
            best_fitness = population_fitness[tournament_indices[0]]
            
            for j in range(1, tournament_size):
                current_fitness = population_fitness[tournament_indices[j]]
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_idx = j
            
            winners[i] = tournament_indices[best_idx]
        
        return winners

    @njit(parallel=True, fastmath=True)
    def _rank_weights_jit(fitness_length: int) -> np.ndarray:
        """JIT-compiled rank weight calculation"""
        ranks = np.arange(1, fitness_length + 1, dtype=np.float32)
        total = ranks.sum()
        return ranks / total

    @njit(fastmath=True)
    def _sus_selection_jit(fitness_cumsum: np.ndarray, 
                          fitness_sum: float, 
                          n: int) -> np.ndarray:
        """JIT-compiled stochastic universal sampling"""
        step = fitness_sum / n
        start = np.random.random() * step
        
        selectors = np.empty(n, dtype=np.float32)
        for i in range(n):
            selectors[i] = start + i * step
        
        # Find indices using binary search equivalent
        selected = np.empty(n, dtype=np.int32)
        for i in range(n):
            selector = selectors[i]
            # Binary search for insertion point
            left, right = 0, len(fitness_cumsum)
            while left < right:
                mid = (left + right) // 2
                if fitness_cumsum[mid] < selector:
                    left = mid + 1
                else:
                    right = mid
            selected[i] = left
        
        return selected


class SelectDefault(SelectBase):
    """Enhanced selection operations with performance optimizations and new methods."""
    
    def __init__(self, ctx=None):
        """Initialize the selection plugin with optional context.
        
        Args:
            ctx: Optional context dictionary containing plugin configuration.
        """
        super().__init__(ctx)
        self._use_jit_threshold = 100  # Use JIT for populations larger than this
        self._diversity_cache = {}  # Cache diversity calculations
        
    def _validate_inputs(self, population: np.ndarray, 
                        population_fitness: np.ndarray, 
                        n: int) -> None:
        """Validate input parameters for selection methods with enhanced error handling."""
        if population.size == 0:
            raise ValueError("Population cannot be empty")
        
        if len(population) != len(population_fitness):
            raise ValueError(f"Population and fitness arrays must have same length. "
                           f"Got population: {len(population)}, fitness: {len(population_fitness)}")
        
        if n <= 0:
            raise ValueError(f"Number of individuals to select must be positive, got {n}")
        
        # Make this a warning instead of an error - let the individual methods handle it
        if n > len(population):
            logger.warning(f"Requested to select {n} individuals but population only has {len(population)}. "
                          f"Methods will handle this appropriately.")
        
        if not np.isfinite(population_fitness).all():
            # Count and log non-finite values for debugging
            non_finite_count = (~np.isfinite(population_fitness)).sum()
            logger.error(f"Found {non_finite_count} non-finite fitness values out of {len(population_fitness)}")
            raise ValueError(f"Fitness values must be finite. Found {non_finite_count} non-finite values.")

    def _fittest(self,
                 *,
                 population: np.ndarray, 
                 population_fitness: np.ndarray,
                 n: int,
                 **kwargs) -> np.ndarray:
        """Enhanced n-fittest selection with better performance and robust error handling.
        
        Args:
            population (np.ndarray): the population to select from. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select
            **kwargs: keyword arguments for plugins

        Returns:
            np.ndarray: selected population

        """
        self._validate_inputs(population, population_fitness, n)
        
        pop_size = len(population)
        
        # Additional bounds checking to prevent argpartition errors
        if n <= 0:
            raise ValueError(f"Number of individuals to select must be positive, got {n}")
        
        if n > pop_size:
            logger.warning(f"Requested {n} individuals but population only has {pop_size}. Selecting all.")
            n = pop_size
        
        # Log debug information to help diagnose issues
        logger.debug(f"Selecting {n} fittest from population of {pop_size}")
        
        # Use argpartition for efficiency - O(n) instead of O(n log n)
        # But add bounds checking to prevent the kth out of bounds error
        if n < pop_size // 2 and n < pop_size:
            # Select top n directly - ensure we don't exceed bounds
            try:
                if n >= pop_size:
                    # If n equals population size, just sort all
                    sorted_indices = np.argsort(population_fitness)[::-1]
                    return population[sorted_indices]
                else:
                    # Safe to use argpartition
                    top_indices = np.argpartition(population_fitness, -n)[-n:]
                    # Sort the selected indices by fitness (descending)
                    sorted_indices = top_indices[np.argsort(population_fitness[top_indices])[::-1]]
                    return population[sorted_indices]
            except ValueError as e:
                logger.error(f"Error in argpartition with n={n}, pop_size={pop_size}: {e}")
                # Fallback to simple sorting
                sorted_indices = np.argsort(population_fitness)[::-1][:n]
                return population[sorted_indices]
        else:
            # More efficient to find bottom (len - n) and exclude them
            # But only if the calculation is valid
            bottom_count = pop_size - n
            
            if bottom_count <= 0:
                # Edge case: selecting all or more than all individuals
                sorted_indices = np.argsort(population_fitness)[::-1][:n]
                return population[sorted_indices]
            
            if bottom_count >= pop_size:
                # Edge case: this shouldn't happen due to earlier validation
                logger.error(f"Invalid bottom_count={bottom_count}, pop_size={pop_size}, n={n}")
                # Fallback to simple sorting
                sorted_indices = np.argsort(population_fitness)[::-1][:n]
                return population[sorted_indices]
            
            try:
                bottom_indices = np.argpartition(population_fitness, bottom_count)[:bottom_count]
                mask = np.ones(pop_size, dtype=bool)
                mask[bottom_indices] = False
                selected = population[mask]
                # Sort by fitness (descending)
                selected_fitness = population_fitness[mask]
                sorted_indices = np.argsort(selected_fitness)[::-1]
                return selected[sorted_indices]
            except ValueError as e:
                logger.error(f"Error in argpartition with bottom_count={bottom_count}, pop_size={pop_size}: {e}")
                # Fallback to simple sorting
                sorted_indices = np.argsort(population_fitness)[::-1][:n]
                return population[sorted_indices]

    def _rank(self, 
               *, 
               population: np.ndarray, 
               population_fitness: np.ndarray,
               n: int,
               selection_pressure: float = 1.0,
               **kwargs) -> np.ndarray:
        """Enhanced rank selection with configurable selection pressure.
        
        Args:
            population (np.ndarray): the population to select from. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select
            selection_pressure (float): selection pressure (1.0 = linear, >1.0 = more pressure)
            **kwargs: keyword arguments for plugins

        Returns:
            np.ndarray: selected population

        """
        self._validate_inputs(population, population_fitness, n)
        
        elements = len(population)
        
        # Use JIT compilation for large populations
        if NUMBA_AVAILABLE and elements > self._use_jit_threshold:
            # Get sorted indices (argsort is already optimized)
            temp = population_fitness.argsort()
            ranks = np.empty_like(temp, dtype=np.float32)
            ranks[temp] = np.arange(1, elements + 1, dtype=np.float32)
            
            # Apply selection pressure
            if selection_pressure != 1.0:
                ranks = ranks ** selection_pressure
            
            weights = ranks / ranks.sum()
        else:
            # Standard numpy implementation
            temp = population_fitness.argsort()
            ranks = np.empty_like(temp, dtype=np.float32)
            ranks[temp] = np.arange(1, elements + 1, dtype=np.float32)
            
            # Apply selection pressure
            if selection_pressure != 1.0:
                ranks = ranks ** selection_pressure
            
            weights = ranks / ranks.sum()
        
        # Use numpy's optimized choice function
        selected_indices = np.random.choice(elements, size=n, replace=False, p=weights)
        return population[selected_indices]

    def _roulette_wheel(self, 
                        *, 
                        population: np.ndarray, 
                        population_fitness: np.ndarray,
                        n: int,
                        **kwargs) -> np.ndarray:
        """Enhanced roulette wheel selection with better numerical stability.
        
        Args:
            population (np.ndarray): the population to select from. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select
            **kwargs: keyword arguments for plugins

        Returns:
            np.ndarray: selected population

        """
        self._validate_inputs(population, population_fitness, n)
        
        # Handle negative fitness values by shifting
        min_fitness = population_fitness.min()
        if min_fitness < 0:
            adjusted_fitness = population_fitness - min_fitness + 1e-10
        else:
            adjusted_fitness = population_fitness + 1e-10  # Avoid division by zero
        
        # Normalize weights for numerical stability
        weights = adjusted_fitness / adjusted_fitness.sum()
        
        # Use numpy's optimized choice function
        selected_indices = np.random.choice(len(population), size=n, replace=True, p=weights)
        return population[selected_indices]

    def _scaled(self, 
               *, 
               population: np.ndarray, 
               population_fitness: np.ndarray,
               n: int,
               scaling_factor: float = 1.0,
               **kwargs) -> np.ndarray:
        """Enhanced scaled selection with configurable scaling factor.
        
        Args:
            population (np.ndarray): the population to select from. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select
            scaling_factor (float): scaling factor for fitness transformation
            **kwargs: keyword arguments for plugins

        Returns:
            np.ndarray: selected population

        """
        self._validate_inputs(population, population_fitness, n)
        
        # Linear scaling with configurable factor
        fitness_min = population_fitness.min()
        fitness_max = population_fitness.max()
        
        if fitness_max == fitness_min:
            # All fitness values are equal, use uniform selection
            selected_indices = np.random.choice(len(population), size=n, replace=False)
            return population[selected_indices]
        
        # Apply scaling
        scaled = (population_fitness - fitness_min) / (fitness_max - fitness_min)
        
        # Apply scaling factor
        if scaling_factor != 1.0:
            scaled = scaled ** scaling_factor
        
        # Normalize and add small epsilon for numerical stability
        scaled = scaled + 1e-10
        weights = scaled / scaled.sum()
        
        selected_indices = np.random.choice(len(population), size=n, replace=False, p=weights)
        return population[selected_indices]

    def _sus(self, 
             *, 
             population: np.ndarray, 
             population_fitness: np.ndarray,
             n: int,
             **kwargs) -> np.ndarray:
        """Enhanced stochastic universal sampling with JIT optimization.
        
        Args:
            population (np.ndarray): the population to select from. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select
            **kwargs: keyword arguments for plugins

        Returns:
            np.ndarray: selected population

        """
        self._validate_inputs(population, population_fitness, n)
        
        # Handle negative fitness values
        min_fitness = population_fitness.min()
        if min_fitness < 0:
            adjusted_fitness = population_fitness - min_fitness + 1e-10
        else:
            adjusted_fitness = population_fitness + 1e-10
        
        # Create cumulative sum (the roulette wheel)
        fitness_cumsum = adjusted_fitness.cumsum()
        fitness_sum = fitness_cumsum[-1]
        
        # Use JIT compilation for large populations
        if NUMBA_AVAILABLE and len(population) > self._use_jit_threshold:
            selected_indices = _sus_selection_jit(fitness_cumsum, fitness_sum, n)
        else:
            # Standard implementation
            step = fitness_sum / n
            start = np.random.random() * step
            selectors = np.arange(start, fitness_sum, step)[:n]  # Ensure exactly n selectors
            selected_indices = np.searchsorted(fitness_cumsum, selectors)
        
        # Handle edge case where searchsorted returns len(population)
        selected_indices = np.clip(selected_indices, 0, len(population) - 1)
        
        return population[selected_indices]
     
    def _tournament(self, 
                    *, 
                    population: np.ndarray, 
                    population_fitness: np.ndarray,
                    tournament_size: int = 2,
                    n: Optional[int] = None,
                    **kwargs) -> np.ndarray:
        """Enhanced tournament selection with JIT optimization and flexible sizing.
        
        Args:
            population (np.ndarray): the population to select from. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            tournament_size (int): number of individuals to compete against each other
            n (int, optional): number of individuals to select. If None, selects len(population)//tournament_size
            **kwargs: keyword arguments for plugins

        Returns:
            np.ndarray: selected population

        """
        if n is None:
            n = len(population) // tournament_size
        
        self._validate_inputs(population, population_fitness, n)
        
        if tournament_size <= 0:
            raise ValueError("Tournament size must be positive")
        
        if tournament_size > len(population):
            raise ValueError("Tournament size cannot exceed population size")
        
        # Use JIT compilation for large populations or many tournaments
        if NUMBA_AVAILABLE and (len(population) > self._use_jit_threshold or n > 50):
            winner_indices = _tournament_selection_jit(population_fitness, tournament_size, n)
        else:
            # Standard implementation with vectorized operations where possible
            winner_indices = np.empty(n, dtype=int)
            
            for i in range(n):
                # Select random tournament participants
                tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                # Find winner (highest fitness)
                winner_idx = tournament_indices[np.argmax(population_fitness[tournament_indices])]
                winner_indices[i] = winner_idx
        
        return population[winner_indices]

    def _diversity_tournament(self,
                             *,
                             population: np.ndarray,
                             population_fitness: np.ndarray,
                             n: int,
                             tournament_size: int = 3,
                             diversity_weight: float = 0.3,
                             **kwargs) -> np.ndarray:
        """Tournament selection that balances fitness and diversity.
        
        Args:
            population (np.ndarray): the population to select from
            population_fitness (np.ndarray): the population fitness
            n (int): number of individuals to select
            tournament_size (int): size of each tournament
            diversity_weight (float): weight for diversity vs fitness (0.0 = pure fitness, 1.0 = pure diversity)
            **kwargs: keyword arguments
            
        Returns:
            np.ndarray: selected population
        """
        self._validate_inputs(population, population_fitness, n)
        
        # Calculate diversity matrix (cache if possible)
        pop_key = id(population)
        if pop_key not in self._diversity_cache:
            diversity_matrix = diversity(population)
            self._diversity_cache[pop_key] = diversity_matrix
        else:
            diversity_matrix = self._diversity_cache[pop_key]
        
        # Calculate diversity scores (lower overlap = higher diversity)
        max_overlap = diversity_matrix.max()
        diversity_scores = max_overlap - diversity_matrix.mean(axis=1)
        
        # Normalize scores
        fitness_norm = (population_fitness - population_fitness.min()) / (population_fitness.max() - population_fitness.min() + 1e-10)
        diversity_norm = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-10)
        
        # Combined score
        combined_scores = (1 - diversity_weight) * fitness_norm + diversity_weight * diversity_norm
        
        # Run tournament selection on combined scores
        return self._tournament(
            population=population,
            population_fitness=combined_scores,
            tournament_size=tournament_size,
            n=n
        )

    def _adaptive_pressure(self,
                          *,
                          population: np.ndarray,
                          population_fitness: np.ndarray,
                          n: int,
                          base_method: str = 'tournament',
                          **kwargs) -> np.ndarray:
        """Adaptive selection that adjusts pressure based on population diversity.
        
        Args:
            population (np.ndarray): the population to select from
            population_fitness (np.ndarray): the population fitness
            n (int): number of individuals to select
            base_method (str): base selection method to adapt
            **kwargs: keyword arguments
            
        Returns:
            np.ndarray: selected population
        """
        self._validate_inputs(population, population_fitness, n)
        
        # Calculate population diversity
        fitness_std = population_fitness.std()
        fitness_mean = population_fitness.mean()
        
        # Calculate coefficient of variation as diversity measure
        cv = fitness_std / (fitness_mean + 1e-10)
        
        # Adapt selection pressure based on diversity
        if cv < 0.1:  # Low diversity - reduce selection pressure
            if base_method == 'tournament':
                kwargs['tournament_size'] = max(2, kwargs.get('tournament_size', 3) - 1)
            elif base_method == 'rank':
                kwargs['selection_pressure'] = 0.7
        elif cv > 0.5:  # High diversity - increase selection pressure
            if base_method == 'tournament':
                kwargs['tournament_size'] = min(len(population) // 2, kwargs.get('tournament_size', 3) + 1)
            elif base_method == 'rank':
                kwargs['selection_pressure'] = 1.5
        
        # Call the base method with adapted parameters
        method_map = {
            'tournament': self._tournament,
            'rank': self._rank,
            'roulette': self._roulette_wheel,
            'sus': self._sus,
            'scaled': self._scaled
        }
        
        return method_map.get(base_method, self._tournament)(
            population=population,
            population_fitness=population_fitness,
            n=n,
            **kwargs
        )

    def _elite_preserving(self,
                         *,
                         population: np.ndarray,
                         population_fitness: np.ndarray,
                         n: int,
                         elite_ratio: float = 0.1,
                         base_method: str = 'tournament',
                         **kwargs) -> np.ndarray:
        """Selection that preserves elite individuals while using another method for the rest.
        
        Args:
            population (np.ndarray): the population to select from
            population_fitness (np.ndarray): the population fitness
            n (int): number of individuals to select
            elite_ratio (float): ratio of elite individuals to preserve
            base_method (str): selection method for non-elite individuals
            **kwargs: keyword arguments
            
        Returns:
            np.ndarray: selected population
        """
        self._validate_inputs(population, population_fitness, n)
        
        n_elite = max(1, int(n * elite_ratio))
        n_regular = n - n_elite
        
        # Select elite individuals
        elite_indices = np.argpartition(population_fitness, -n_elite)[-n_elite:]
        elite_population = population[elite_indices]
        
        if n_regular > 0:
            # Select remaining individuals using base method
            method_map = {
                'tournament': self._tournament,
                'rank': self._rank,
                'roulette': self._roulette_wheel,
                'sus': self._sus,
                'scaled': self._scaled,
                'fittest': self._fittest
            }
            
            regular_population = method_map.get(base_method, self._tournament)(
                population=population,
                population_fitness=population_fitness,
                n=n_regular,
                **kwargs
            )
            
            # Combine elite and regular selections
            return np.vstack([elite_population, regular_population])
        else:
            return elite_population

    def select(self, 
               *, 
               population: np.ndarray, 
               population_fitness: np.ndarray,
               n: int,
               method: str = 'roulette',
               **kwargs) -> np.ndarray:
        """Enhanced selection with multiple methods and optimizations.
        
        Args:
            population (np.ndarray): the population to select from. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            n (int): total number of individuals to select
            method (str): selection method - see dispatch dict for options
            **kwargs: keyword arguments for specific methods

        Returns:
            np.ndarray: selected population

        """
        dispatch = {
            'roulette': self._roulette_wheel,
            'sus': self._sus,
            'rank': self._rank,
            'tournament': self._tournament,
            'scaled': self._scaled,
            'fittest': self._fittest,
            'diversity_tournament': self._diversity_tournament,
            'adaptive': self._adaptive_pressure,
            'elite': self._elite_preserving
        }        

        params = {
            'population': population,
            'population_fitness': population_fitness,
            'n': n
        }

        if method not in dispatch:
            logger.warning(f"Unknown selection method: {method}. Using roulette wheel.")
            method = 'roulette'

        try:
            result = dispatch[method](**params, **kwargs)
            logger.debug(f"Selection complete using {method}, selected {len(result)} individuals")
            return result
        except Exception as e:
            logger.error(f"Selection failed with method {method}: {str(e)}")
            # Fallback to fittest selection
            logger.info("Falling back to fittest selection")
            return self._fittest(**params)

    def clear_cache(self):
        """Clear the diversity calculation cache."""
        self._diversity_cache.clear()

    def set_jit_threshold(self, threshold: int):
        """Set the population size threshold for using JIT compilation.
        
        Args:
            threshold (int): population size above which to use JIT compilation
        """
        self._use_jit_threshold = threshold

    def get_selection_stats(self, 
                           population: np.ndarray,
                           population_fitness: np.ndarray,
                           selected: np.ndarray) -> dict:
        """Calculate statistics about the selection process.
        
        Args:
            population (np.ndarray): original population
            population_fitness (np.ndarray): original fitness values
            selected (np.ndarray): selected individuals
            
        Returns:
            dict: selection statistics
        """
        # Find indices of selected individuals in original population
        selected_indices = []
        for sel_individual in selected:
            # Find matching individual in original population
            matches = np.where((population == sel_individual).all(axis=1))[0]
            if len(matches) > 0:
                selected_indices.append(matches[0])
        
        if not selected_indices:
            return {'error': 'Could not match selected individuals to original population'}
        
        selected_fitness = population_fitness[selected_indices]
        
        return {
            'selection_ratio': len(selected) / len(population),
            'fitness_improvement': {
                'mean_original': population_fitness.mean(),
                'mean_selected': selected_fitness.mean(),
                'improvement_ratio': selected_fitness.mean() / population_fitness.mean()
            },
            'diversity_metrics': {
                'original_std': population_fitness.std(),
                'selected_std': selected_fitness.std(),
                'diversity_retention': selected_fitness.std() / population_fitness.std()
            },
            'elite_capture': {
                'top_10_percent_captured': np.sum(selected_fitness >= np.percentile(population_fitness, 90)) / len(selected),
                'best_individual_selected': selected_fitness.max() == population_fitness.max()
            }
        }


# Export classes and functions
__all__ = [
    'SelectDefault'
]
