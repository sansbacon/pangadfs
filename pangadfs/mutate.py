# pangadfs/pangadfs/mutate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import numpy as np

from pangadfs.base import MutateBase


class MutateDefault(MutateBase):

    def mutate(self, *, population: np.ndarray, mutation_rate: float = .05) -> np.ndarray:
        """Mutates individuals in population
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            mutation_rate (float): decimal value from 0 to 1, default .05

        Returns:
            np.ndarray: same shape as population

        """
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
