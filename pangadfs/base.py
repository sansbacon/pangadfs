# gadfs/gadfs/base.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import abc
import logging
from typing import Dict, Tuple

from pandas.core.api import DataFrame


Population = Dict[Tuple, Tuple]
Df = DataFrame


class CrossoverBase(metaclass=abc.ABCMeta):
    """Base class for crossover plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def crossover(population: Population, pool: Df):
        """Implement crossover.

        Args:
            population (Population): the existing population of teams
            pool (Df): the entire player pool

        Returns:
            Population: new population

        """


class FitnessBase(metaclass=abc.ABCMeta):
    """Base class for fitness plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def fitness(population: Population):
        """Implement fitness.

        Args:
            population (Population): the existing population of teams

        Returns:
            float

        """

class MutateBase(metaclass=abc.ABCMeta):
    """Base class for crossover plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def mutate(population: Population, pool: Df, mutation_rate: float):
        """Mutates population at given mutation rate.

        Args:
            population (Population): the existing population of teams
            pool (Df): the entire player pool
            mutation_rate (float): between 0 and 1

        Returns:
            Population: new population

        """


class PopulateBase(metaclass=abc.ABCMeta):
    """Base class for populate plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def populate(pool: Df, initial_size: int, n_chromosomes: int):
        """Creates initial population from pool.

        Args:
            pool (Df): the entire player pool
            initial_size (int): initial size of population
            n_chromosomes (int): number of chromosomes for individual

        Returns:
            Population: new population

        """


class ValidateBase(metaclass=abc.ABCMeta):
    """Base class for validate plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def validate(population: Population):
        """Implement validate.

        Args:
            population (Population): the existing population of teams

        Returns:
            bool

        """

