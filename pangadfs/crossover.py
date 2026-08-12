# pangadfs/pangadfs/crossover.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Tuple
import numpy as np

from pangadfs.base import CrossoverBase
from pangadfs.metrics import diversity
from pangadfs.sampling import parents


class CrossoverDefault(CrossoverBase):

    def _diverse(self, *, population: np.ndarray, **kwargs) -> np.ndarray:
        """Diverse crossover of individuals in population.
        
        Pairs each individual with its most-diverse partner (least overlap),
        then performs uniform crossover between the pairs.

        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            **kwargs: Arbitrary keyword arguments

        Returns:
            np.ndarray: offspring from diverse-pair uniform crossover

        """     
        # Get pairwise overlap matrix (lower = more diverse)
        overlap_matrix = diversity(population)
        
        # Zero out self-overlap so we don't pair with ourselves
        np.fill_diagonal(overlap_matrix, np.iinfo(overlap_matrix.dtype).max)
        
        # For each individual, find the most diverse partner (minimum overlap)
        # Randomize tie-breaking by adding small noise
        noise = np.random.random(overlap_matrix.shape) * 0.01
        partner_idx = np.argmin(overlap_matrix.astype(float) + noise, axis=1)
        
        # Uniform crossover between each individual and its diverse partner
        partners = population[partner_idx]
        choice = np.random.randint(2, size=population.size).reshape(population.shape).astype(bool)
        return np.where(choice, population, partners)

    def _one_point(self, *, population: np.ndarray, point: int = 3, **kwargs) -> np.ndarray:
        """Crosses over individuals in population at a single point.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            point (int): the crossover point, default 3
            **kwargs: Arbitrary keyword arguments
        
        Returns:
            np.ndarray: concatenation of two offspring

        """
        fathers, mothers = parents(population)
        child1 = np.hstack((fathers[:, 0:point], mothers[:, point:]))
        child2 = np.hstack((mothers[:, 0:point], fathers[:, point:]))
        return np.vstack((child1, child2))

    def _two_point(self, *, population: np.ndarray, points: Tuple[int] = (3, 7), **kwargs) -> np.ndarray:
        """Crosses over individuals in population at two points.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            points (Tuple[int]): the crossover points, default (3, 7)
            **kwargs: Arbitrary keyword arguments    
    
        Returns:
            np.ndarray: concatenation of two offspring

        """
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
        return np.vstack((np.where(choice, fathers, mothers), np.where(choice, mothers, fathers)))

    def crossover(self, *, population: np.ndarray, method: str = 'uniform', **kwargs) -> np.ndarray:
        """Crossover individuals in population.
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            method (str): crossover method, 'uniform', 'diverse', 'one_point', or 'two_point'.
            **kwargs: Arbitrary keyword arguments

        Returns:
            np.ndarray: concatenation of two offspring

        """     
        dispatch = {
            'uniform': self._uniform,
            'diverse': self._diverse,
            'one_point': self._one_point,
            'two_point': self._two_point
        }        

        return dispatch.get(method, self._uniform)(population=population, **kwargs)
