# pangadfs/pangadfs/mutate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np

from pangadfs.base import MutateBase


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

    def mutate(self, *, population: np.ndarray, mutation_rate: float = None) -> np.ndarray:
        """Mutates individuals in population
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            mutation_rate (float): decimal value from 0 to 1. If None, uses the rate from context or default.

        Returns:
            np.ndarray: same shape as population

        """
        # Use provided mutation_rate, or fall back to the one from context/default
        if mutation_rate is None:
            mutation_rate = self.default_mutation_rate
        # mutate is ndarray of same shape of population of dtype bool
        # if mutation_rate is .05, then 1 out of 20 values should be True
        # where mutate is true, swap randomly-selected player into population
        # ensures swap comes from same lineup slot, but does not prevent duplicates from other slots
        # so lineup positional allocation will stay valid, but duplicates are possible
        mutate = (
            np.random.binomial(n=1, p=mutation_rate, size=population.size)
            .reshape(population.shape)
            .astype(bool)
        )
        swap = population[np.random.choice(len(population), size=len(population), replace=False)]
        return np.where(mutate, swap, population)
