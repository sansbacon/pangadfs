# gadfs/gadfs/base.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import abc
import logging


class PluginBase(metaclass=abc.ABCMeta):
    """Base class for all plugins."""

    def __init__(self, ctx=None):
        """Initialize the plugin with optional context.
        
        Args:
            ctx: Optional context dictionary containing plugin configuration.
        """
        self.ctx = ctx
        logging.getLogger(__name__).addHandler(logging.NullHandler())


class CrossoverBase(PluginBase):
    """Base class for crossover plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def crossover(self, *args, **kwargs):
        """Implement crossover."""


class FitnessBase(PluginBase):
    """Base class for fitness plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def fitness(self, *args, **kwargs):
        """Implement fitness."""


class MutateBase(PluginBase):
    """Base class for mutate plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def mutate(self, *args, **kwargs):
        """Mutates population at given mutation rate."""


class OptimizeBase(PluginBase):
    """Base class for optimize plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def optimize(self, *args, **kwargs):
        """Implements optimize."""


class PenaltyBase(PluginBase):
    """Base class for penalty plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def penalty(self, *args, **kwargs):
        """Calculates penalty for population fitness."""


class PopulateBase(PluginBase):
    """Base class for populate plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def populate(self, *args, **kwargs):
        """Creates initial population from pool."""


class PoolBase(PluginBase):
    """Base class for pool plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def pool(self, *args, **kwargs):
        """Implement pool."""


class PospoolBase(PluginBase):
    """Base class for pospool plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def pospool(self, *args, **kwargs):
        """Implement pospool."""


class SelectBase(PluginBase):
    """Base class for select plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def select(self, *args, **kwargs):
        """Implement select."""


class ValidateBase(PluginBase):
    """Base class for validate plugins."""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @abc.abstractmethod
    def validate(self, *args, **kwargs):
        """Implement validate."""
