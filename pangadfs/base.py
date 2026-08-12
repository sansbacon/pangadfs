# pangadfs/pangadfs/base.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import abc
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


class PluginBase(metaclass=abc.ABCMeta):
    """Common base for all pangadfs plugins.
    
    Subclasses only need to define a single @abstractmethod.
    Logging NullHandler is configured once here.
    """

    def __init__(self):
        logging.getLogger(__name__).addHandler(logging.NullHandler())


class CrossoverBase(PluginBase):
    """Base class for crossover plugins."""

    @abc.abstractmethod
    def crossover(self, *args, **kwargs):
        """Implement crossover."""


class FitnessBase(PluginBase):
    """Base class for fitness plugins."""

    @abc.abstractmethod
    def fitness(self, *args, **kwargs):
        """Implement fitness."""


class MutateBase(PluginBase):
    """Base class for mutate plugins."""

    @abc.abstractmethod
    def mutate(self, *args, **kwargs):
        """Mutates population at given mutation rate."""


class OptimizeBase(PluginBase):
    """Base class for optimize plugins."""

    @abc.abstractmethod
    def optimize(self, *args, **kwargs):
        """Implements optimize."""


class PenaltyBase(PluginBase):
    """Base class for penalty plugins."""

    @abc.abstractmethod
    def penalty(self, *args, **kwargs):
        """Calculates penalty for population fitness."""


class PopulateBase(PluginBase):
    """Base class for populate plugins."""

    @abc.abstractmethod
    def populate(self, *args, **kwargs):
        """Creates initial population from pool."""


class PoolBase(PluginBase):
    """Base class for pool plugins."""

    @abc.abstractmethod
    def pool(self, *args, **kwargs):
        """Implement pool."""


class PospoolBase(PluginBase):
    """Base class for pospool plugins."""

    @abc.abstractmethod
    def pospool(self, *args, **kwargs):
        """Implement pospool."""


class SelectBase(PluginBase):
    """Base class for select plugins."""

    @abc.abstractmethod
    def select(self, *args, **kwargs):
        """Implement select."""


class ValidateBase(PluginBase):
    """Base class for validate plugins."""

    @abc.abstractmethod
    def validate(self, *args, **kwargs):
        """Implement validate."""
