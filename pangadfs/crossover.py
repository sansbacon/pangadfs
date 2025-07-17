# pangadfs/pangadfs/crossover.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import List, Optional, Tuple, Union
import numpy as np

from pangadfs.base import CrossoverBase
from pangadfs.misc import diversity, parents

# Set up logging
logger = logging.getLogger(__name__)

try:
    from numba import njit
except ImportError:
    # Define a no-op decorator if Numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class CrossoverDefault(CrossoverBase):
    """Default implementation of crossover operations for genetic algorithms."""
    
    # Note: Numba-optimized functions are typically defined outside of classes
    # because Numba has limited support for class methods. However, we're including
    # this function in the class for organizational clarity.
    @staticmethod
    @njit
    def _uniform_crossover_impl(fathers: np.ndarray, mothers: np.ndarray, choice: np.ndarray) -> np.ndarray:
        """Numba-optimized implementation of uniform crossover"""
        child1 = np.where(choice, fathers, mothers)
        child2 = np.where(choice, mothers, fathers)
        return np.vstack((child1, child2))

    def _diverse(self, *, population: np.ndarray, **kwargs) -> np.ndarray:
        """Diverse crossover of individuals in population.
        
        Pairs individuals with the least common elements for crossover.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            **kwargs: Arbitrary keyword arguments

        Returns:
            np.ndarray: concatenation of two offspring

        """     
        # Calculate diversity matrix to find pairs with least common elements
        diversity_matrix = diversity(population)
        
        # For each individual, find the individual with the least overlap
        pairs = np.zeros(population.shape[0], dtype=int)
        for i in range(population.shape[0]):
            # Exclude self-pairing
            mask = np.ones(population.shape[0], dtype=bool)
            mask[i] = False
            # Find individual with minimum overlap
            pairs[i] = np.argmin(diversity_matrix[i, mask])
            
        # Perform crossover between each individual and its diverse pair
        fathers = population[::2]
        mothers = population[pairs[::2]]
        
        # Use uniform crossover for the diverse pairs
        choice = np.random.randint(2, size=fathers.size).reshape(fathers.shape).astype(bool)
        child1 = np.where(choice, fathers, mothers)
        child2 = np.where(choice, mothers, fathers)
        
        return np.vstack((child1, child2))

    def _one_point(self, *, population: np.ndarray, point: Optional[int] = None, **kwargs) -> np.ndarray:
        """Crosses over individuals in population at a single point.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            point (int, optional): the crossover point. If None, a random point is chosen.
            **kwargs: Arbitrary keyword arguments
        
        Returns:
            np.ndarray: concatenation of two offspring
            
        Raises:
            ValueError: If point is out of valid range

        """
        n_chromosomes = population.shape[1]
        
        if point is None:
            point = np.random.randint(1, n_chromosomes - 1)
        elif point <= 0 or point >= n_chromosomes:
            raise ValueError(f"Crossover point must be between 1 and {n_chromosomes-1}")
        
        logger.debug(f"One-point crossover at position {point}")
        
        fathers, mothers = parents(population)
        child1 = np.hstack((fathers[:, 0:point], mothers[:, point:]))
        child2 = np.hstack((mothers[:, 0:point], fathers[:, point:]))
        return np.vstack((child1, child2))

    def _two_point(self, *, population: np.ndarray, points: Optional[Tuple[int, int]] = None, **kwargs) -> np.ndarray:
        """Crosses over individuals in population at two points.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            points (Tuple[int, int], optional): the crossover points. If None, random points are chosen.
            **kwargs: Arbitrary keyword arguments    
    
        Returns:
            np.ndarray: concatenation of two offspring
            
        Raises:
            ValueError: If points are out of valid range or not in ascending order

        """
        n_chromosomes = population.shape[1]
        
        if points is None:
            # Generate two random distinct points and sort them
            points = sorted(np.random.choice(range(1, n_chromosomes), size=2, replace=False))
        else:
            point1, point2 = points
            if point1 <= 0 or point2 >= n_chromosomes or point1 >= point2:
                raise ValueError(f"Crossover points must be 0 < point1 < point2 < {n_chromosomes}")
            
        logger.debug(f"Two-point crossover at positions {points}")
        
        fathers, mothers = parents(population)
        point1, point2 = points
        child1 = np.hstack((fathers[:, 0:point1], 
                            mothers[:, point1:point2],
                            fathers[:, point2:]))
        child2 = np.hstack((mothers[:, 0:point1], 
                            fathers[:, point1:point2],
                            mothers[:, point2:]))
        return np.vstack((child1, child2))

    def _uniform(self, *, population: np.ndarray, **kwargs) -> np.ndarray:
        """Uniform crossover of individuals in population.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            **kwargs: Arbitrary keyword arguments

        Returns:
            np.ndarray: concatenation of two offspring

        """     
        fathers, mothers = parents(population)
        choice = np.random.randint(2, size=fathers.size).reshape(fathers.shape).astype(bool)
        
        # Use Numba-optimized implementation if available
        return self._uniform_crossover_impl(fathers, mothers, choice)

    def _ordered(self, *, population: np.ndarray, **kwargs) -> np.ndarray:
        """Ordered crossover (OX1) for permutation problems.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            **kwargs: Arbitrary keyword arguments

        Returns:
            np.ndarray: concatenation of two offspring
        """
        fathers, mothers = parents(population)
        n_chromosomes = fathers.shape[1]
        
        # Generate random crossover points
        points = sorted(np.random.choice(range(n_chromosomes), size=2, replace=False))
        logger.debug(f"Ordered crossover at positions {points}")
        
        # Initialize offspring
        child1 = np.full_like(fathers, -1)
        child2 = np.full_like(mothers, -1)
        
        # Copy segment from parents
        child1[:, points[0]:points[1]] = fathers[:, points[0]:points[1]]
        child2[:, points[0]:points[1]] = mothers[:, points[0]:points[1]]
        
        # Fill remaining positions
        for i in range(len(fathers)):
            # Create remaining elements lists
            remaining1 = [x for x in mothers[i] if x not in fathers[i, points[0]:points[1]]]
            remaining2 = [x for x in fathers[i] if x not in mothers[i, points[0]:points[1]]]
            
            # Fill positions after second point and before first point
            pos1 = points[1]
            pos2 = points[1]
            
            for j in range(n_chromosomes - (points[1] - points[0])):
                if pos1 == n_chromosomes:
                    pos1 = 0
                if pos2 == n_chromosomes:
                    pos2 = 0
                    
                child1[i, pos1] = remaining1[j]
                child2[i, pos2] = remaining2[j]
                pos1 += 1
                pos2 += 1
        
        return np.vstack((child1, child2))

    def _pmx(self, *, population: np.ndarray, **kwargs) -> np.ndarray:
        """Partially Mapped Crossover (PMX) for permutation problems.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            **kwargs: Arbitrary keyword arguments

        Returns:
            np.ndarray: concatenation of two offspring
        """
        fathers, mothers = parents(population)
        n_chromosomes = fathers.shape[1]
        
        # Generate random crossover points
        points = sorted(np.random.choice(range(n_chromosomes), size=2, replace=False))
        logger.debug(f"PMX crossover at positions {points}")
        
        # Initialize offspring
        child1 = np.full_like(fathers, -1)
        child2 = np.full_like(mothers, -1)
        
        # Copy segment from parents
        child1[:, points[0]:points[1]] = fathers[:, points[0]:points[1]]
        child2[:, points[0]:points[1]] = mothers[:, points[0]:points[1]]
        
        # Create mapping for each individual
        for i in range(len(fathers)):
            # Create mapping dictionaries
            map1 = {mothers[i, j]: fathers[i, j] for j in range(points[0], points[1])}
            map2 = {fathers[i, j]: mothers[i, j] for j in range(points[0], points[1])}
            
            # Fill remaining positions
            for j in range(n_chromosomes):
                if j < points[0] or j >= points[1]:
                    # For child1
                    value = mothers[i, j]
                    while value in map1:
                        value = map1[value]
                    child1[i, j] = value
                    
                    # For child2
                    value = fathers[i, j]
                    while value in map2:
                        value = map2[value]
                    child2[i, j] = value
        
        return np.vstack((child1, child2))

    def _cycle(self, *, population: np.ndarray, **kwargs) -> np.ndarray:
        """Cycle Crossover (CX) for permutation problems.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            **kwargs: Arbitrary keyword arguments

        Returns:
            np.ndarray: concatenation of two offspring
        """
        fathers, mothers = parents(population)
        n_chromosomes = fathers.shape[1]
        
        # Initialize offspring
        child1 = np.full_like(fathers, -1)
        child2 = np.full_like(mothers, -1)
        
        for i in range(len(fathers)):
            # Create position maps for quick lookup
            father_pos = {fathers[i, j]: j for j in range(n_chromosomes)}
            
            # Initialize cycle tracking
            cycle_start = 0
            cycle_no = 1
            cycle = [-1] * n_chromosomes  # -1 means not assigned to a cycle
            
            # Find cycles
            while -1 in cycle:
                # Find next unassigned position
                for j in range(n_chromosomes):
                    if cycle[j] == -1:
                        cycle_start = j
                        break
                
                # Mark positions in this cycle
                j = cycle_start
                while cycle[j] == -1:
                    cycle[j] = cycle_no
                    j = father_pos[mothers[i, j]]
                
                cycle_no += 1
            
            # Assign values based on cycles
            for j in range(n_chromosomes):
                if cycle[j] % 2 == 1:  # Odd cycles from father, even from mother
                    child1[i, j] = fathers[i, j]
                    child2[i, j] = mothers[i, j]
                else:
                    child1[i, j] = mothers[i, j]
                    child2[i, j] = fathers[i, j]
        
        return np.vstack((child1, child2))

    def _blx_alpha(self, *, population: np.ndarray, alpha: float = 0.5, **kwargs) -> np.ndarray:
        """Blend Crossover (BLX-α) for real-valued chromosomes.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            alpha (float): blending parameter, typically between 0 and 1.
            **kwargs: Arbitrary keyword arguments

        Returns:
            np.ndarray: concatenation of two offspring
        """
        fathers, mothers = parents(population)
        
        # Calculate range for each gene
        min_vals = np.minimum(fathers, mothers)
        max_vals = np.maximum(fathers, mothers)
        ranges = max_vals - min_vals
        
        # Extend range by alpha
        lower_bounds = min_vals - alpha * ranges
        upper_bounds = max_vals + alpha * ranges
        
        # Generate random values within extended ranges
        child1 = lower_bounds + np.random.random(fathers.shape) * (upper_bounds - lower_bounds)
        child2 = lower_bounds + np.random.random(fathers.shape) * (upper_bounds - lower_bounds)
        
        return np.vstack((child1, child2))

    def crossover(self, *, population: np.ndarray, method: str = 'uniform', **kwargs) -> np.ndarray:
        """Crossover individuals in population.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            method (str): crossover method, one of: 'uniform', 'diverse', 'one_point', 'two_point',
                         'ordered', 'pmx', 'cycle', 'blx_alpha'.
            **kwargs: Arbitrary keyword arguments

        Returns:
            np.ndarray: concatenation of two offspring
            
        Raises:
            ValueError: If population is empty or has odd number of individuals
            ValueError: If method is not recognized
        """
        if population.size == 0:
            raise ValueError("Population cannot be empty")
            
        if population.shape[0] % 2 != 0:
            raise ValueError("Population must have an even number of individuals for crossover")
        
        logger.debug(f"Performing {method} crossover on population of shape {population.shape}")
        
        dispatch = {
            'uniform': self._uniform,
            'diverse': self._diverse,
            'one_point': self._one_point,
            'two_point': self._two_point,
            'ordered': self._ordered,
            'pmx': self._pmx,
            'cycle': self._cycle,
            'blx_alpha': self._blx_alpha
        }
        
        if method not in dispatch:
            raise ValueError(f"Unknown crossover method: {method}. Available methods: {', '.join(dispatch.keys())}")
            
        result = dispatch[method](population=population, **kwargs)
        logger.debug(f"Crossover complete, resulting shape: {result.shape}")
        return result
