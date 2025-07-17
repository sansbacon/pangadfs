# pangadfs/pangadfs/mutate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
from typing import Optional, Union

from pangadfs.base import MutateBase

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False


class MutateDefault(MutateBase):
    
    def __init__(self, ctx=None):
        """Initialize the mutate plugin with optional context.
        
        Args:
            ctx: Optional context dictionary containing plugin configuration.
                 Can include 'mutation_rate' to override the default.
        """
        super().__init__(ctx)
        # Get default mutation rate from context if provided
        self.default_mutation_rate = .05
        if self.ctx and 'mutation_rate' in self.ctx:
            self.default_mutation_rate = self.ctx['mutation_rate']

    def mutate(self, *, population: np.ndarray, mutation_rate: float = None, 
               method: str = 'swap', **kwargs) -> np.ndarray:
        """Mutates individuals in population using the best available implementation.
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            mutation_rate (float): decimal value from 0 to 1. If None, uses the rate from context or default.
            method (str): mutation method - 'swap', 'uniform', 'gaussian', 'position_swap'
            **kwargs: Additional arguments for specific mutation methods

        Returns:
            np.ndarray: same shape as population

        """
        # Use provided mutation_rate, or fall back to the one from context/default
        if mutation_rate is None:
            mutation_rate = self.default_mutation_rate
            
        # Dispatch to appropriate method
        if method == 'swap':
            return self._mutate_swap(population, mutation_rate)
        elif method == 'uniform':
            return self._mutate_uniform(population, mutation_rate, **kwargs)
        elif method == 'gaussian':
            return self._mutate_gaussian(population, mutation_rate, **kwargs)
        elif method == 'position_swap':
            return self._mutate_position_swap(population, mutation_rate)
        else:
            raise ValueError(f"Unknown mutation method: {method}")

    def _mutate_swap(self, population: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Original swap mutation with optimizations."""
        if NUMBA_AVAILABLE:
            return self._mutate_swap_numba(population, mutation_rate)
        else:
            return self._mutate_swap_fast(population, mutation_rate)

    def _mutate_swap_fast(self, population: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Optimized swap mutation using fast numpy operations."""
        n_individuals, n_chromosomes = population.shape
        
        # More efficient mutation mask generation
        n_mutations = int(population.size * mutation_rate)
        if n_mutations == 0:
            return population.copy()
            
        # Create mutation indices more efficiently
        mutation_indices = np.random.choice(population.size, size=n_mutations, replace=False)
        mutation_mask = np.zeros(population.size, dtype=bool)
        mutation_mask[mutation_indices] = True
        mutation_mask = mutation_mask.reshape(population.shape)
        
        # Only create swap values where needed (memory optimization)
        if np.any(mutation_mask):
            # Generate random permutation indices for swapping
            swap_indices = np.random.permutation(n_individuals)
            swap_population = population[swap_indices]
            return np.where(mutation_mask, swap_population, population)
        
        return population.copy()

    def _mutate_uniform(self, population: np.ndarray, mutation_rate: float, 
                       min_val: int = 0, max_val: int = None) -> np.ndarray:
        """Uniform random mutation - replaces mutated genes with random values."""
        if max_val is None:
            max_val = np.max(population) + 1
            
        # Generate mutation mask
        mutation_mask = np.random.random(population.shape) < mutation_rate
        
        if np.any(mutation_mask):
            # Generate random values only where needed
            random_values = np.random.randint(min_val, max_val, size=np.sum(mutation_mask))
            result = population.copy()
            result[mutation_mask] = random_values
            return result
            
        return population.copy()

    def _mutate_gaussian(self, population: np.ndarray, mutation_rate: float,
                        sigma: float = 1.0) -> np.ndarray:
        """Gaussian mutation for continuous values."""
        mutation_mask = np.random.random(population.shape) < mutation_rate
        
        if np.any(mutation_mask):
            # Add Gaussian noise to mutated positions
            noise = np.random.normal(0, sigma, size=np.sum(mutation_mask))
            result = population.copy().astype(float)
            result[mutation_mask] += noise
            return result.astype(population.dtype)
            
        return population.copy()

    def _mutate_position_swap(self, population: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Position-aware swap mutation - swaps within the same position across individuals."""
        n_individuals, n_chromosomes = population.shape
        result = population.copy()
        
        # For each position (chromosome), decide whether to mutate
        for pos in range(n_chromosomes):
            if np.random.random() < mutation_rate:
                # Randomly permute this position across all individuals
                result[:, pos] = np.random.permutation(result[:, pos])
                
        return result

    # Numba-optimized implementations
    if NUMBA_AVAILABLE:
        @staticmethod
        @njit(parallel=True, fastmath=True)
        def _generate_mutation_mask_numba(shape: tuple, mutation_rate: float) -> np.ndarray:
            """Generate mutation mask using numba for better performance."""
            mask = np.zeros(shape, dtype=np.bool_)
            n_rows, n_cols = shape
            
            for i in prange(n_rows):
                for j in range(n_cols):
                    if np.random.random() < mutation_rate:
                        mask[i, j] = True
                        
            return mask

        def _mutate_swap_numba(self, population: np.ndarray, mutation_rate: float) -> np.ndarray:
            """Numba-optimized swap mutation."""
            n_individuals, n_chromosomes = population.shape
            
            # Generate mutation mask using numba
            mutation_mask = self._generate_mutation_mask_numba(population.shape, mutation_rate)
            
            if np.any(mutation_mask):
                # Generate swap indices
                swap_indices = np.random.permutation(n_individuals)
                swap_population = population[swap_indices]
                return np.where(mutation_mask, swap_population, population)
                
            return population.copy()
    else:
        def _mutate_swap_numba(self, population: np.ndarray, mutation_rate: float) -> np.ndarray:
            """Fallback when numba is not available."""
            return self._mutate_swap_fast(population, mutation_rate)

    def adaptive_mutate(self, *, population: np.ndarray, fitness: np.ndarray, 
                       base_rate: float = None, diversity_factor: float = 0.1) -> np.ndarray:
        """Adaptive mutation that adjusts rate based on population diversity and fitness.
        
        Args:
            population: Population to mutate
            fitness: Fitness values for each individual
            base_rate: Base mutation rate (uses default if None)
            diversity_factor: How much diversity affects mutation rate
            
        Returns:
            Mutated population
        """
        if base_rate is None:
            base_rate = self.default_mutation_rate
            
        # Calculate population diversity (simple measure)
        unique_individuals = len(np.unique(population.view(np.void), axis=0))
        diversity_ratio = unique_individuals / len(population)
        
        # Calculate fitness variance (normalized)
        fitness_var = np.var(fitness) / (np.mean(fitness) + 1e-8)
        
        # Adaptive mutation rate: higher when diversity is low or fitness variance is low
        adaptive_rate = base_rate * (1 + diversity_factor * (1 - diversity_ratio) + 
                                   diversity_factor * (1 / (1 + fitness_var)))
        adaptive_rate = min(adaptive_rate, 0.5)  # Cap at 50%
        
        return self.mutate(population=population, mutation_rate=adaptive_rate, method='swap')

    def multi_method_mutate(self, *, population: np.ndarray, mutation_rate: float = None,
                           methods: list = None, weights: list = None) -> np.ndarray:
        """Apply multiple mutation methods with specified weights.
        
        Args:
            population: Population to mutate
            mutation_rate: Overall mutation rate
            methods: List of mutation methods to use
            weights: Probability weights for each method
            
        Returns:
            Mutated population
        """
        if mutation_rate is None:
            mutation_rate = self.default_mutation_rate
            
        if methods is None:
            methods = ['swap', 'uniform', 'position_swap']
            
        if weights is None:
            weights = [0.5, 0.3, 0.2]
            
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Choose method for each individual
        method_choices = np.random.choice(methods, size=len(population), p=weights)
        
        result = population.copy()
        for method in methods:
            # Get individuals assigned to this method
            method_mask = method_choices == method
            if np.any(method_mask):
                method_population = population[method_mask]
                mutated = self.mutate(population=method_population, 
                                    mutation_rate=mutation_rate, method=method)
                result[method_mask] = mutated
                
        return result
