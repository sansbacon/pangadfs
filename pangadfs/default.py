# gadfs/gadfs/default.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging

from .base import *


class DefaultCrossover(CrossoverBase):
    """Default crossover technique"""

    def crossover(population: Population, pool: Df):
        """Crosses over population"""
        logging.info('using default crossover')
        return population


class DefaultMutate(MutateBase):
    """Default mutate technique"""

    def mutate(population: Population, pool: Df, mutation_rate: float):
        """Mutates population"""
        logging.info('using default mutate')
        return population


class DefaultFitness(FitnessBase):
    """Default fitness technique."""

    def fitness(population: Population):
        """Implement fitness."""
        try:
            return float(len(population))
        except:
            return 1.0
        
        
class DefaultPopulate(PopulateBase):
    """Default populate technique"""

    def populate(pool, initial_size, n_chromosomes):
        """Creates individuals in population"""
        population = {}
        while len(list(population.keys())) < initial_size:
            mapping = {
                row.id: row.full_name
                for row in pool.sample(n_chromosomes).sort_values('id').itertuples()
            }
            population[tuple(mapping.keys())] = tuple(mapping.values())
        return population


class DefaultValidate(ValidateBase):
    """Default validate technique"""

    def validate(population):
        """Validates population"""
        return population is not None