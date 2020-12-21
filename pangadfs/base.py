# gadfs/gadfs/base.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import abc
import logging


class CrossoverBase(metaclass=abc.ABCMeta):
    """Base class for crossover plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def crossover(self, *args, **kwargs):
        """Implement crossover."""


class FitnessBase(metaclass=abc.ABCMeta):
    """Base class for fitness plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def fitness(self, *args, **kwargs):
        """Implement fitness."""


class MutateBase(metaclass=abc.ABCMeta):
    """Base class for crossover plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def mutate(self, *args, **kwargs):
        """Mutates population at given mutation rate."""


class PopulateBase(metaclass=abc.ABCMeta):
    """Base class for populate plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def populate(self, *args, **kwargs):
        """Creates initial population from pool."""


class PoolBase(metaclass=abc.ABCMeta):
    """Base class for pool plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def pool(self, *args, **kwargs):
        """Implement pool."""


class PospoolBase(metaclass=abc.ABCMeta):
    """Base class for pospool plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def pospool(self, *args, **kwargs):
        """Implement pospool."""


class ValidateBase(metaclass=abc.ABCMeta):
    """Base class for validate plugins."""

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    @abc.abstractmethod
    def validate(self, *args, **kwargs):
        """Implement validate."""
