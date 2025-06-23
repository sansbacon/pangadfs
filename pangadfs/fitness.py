# pangadfs/pangadfs/fitness.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np

from pangadfs.base import FitnessBase


class FitnessDefault(FitnessBase):
    
    def __init__(self, ctx=None):
        """Initialize the fitness plugin with optional context.
        
        Args:
            ctx: Optional context dictionary containing plugin configuration.
                 Can include 'fitness_scale' to scale fitness values.
        """
        super().__init__(ctx)
        # Get fitness scale from context if provided
        self.fitness_scale = 1.0
        if self.ctx and 'fitness_scale' in self.ctx:
            self.fitness_scale = self.ctx['fitness_scale']

    def fitness(self,
                *, 
                population: np.ndarray, 
                points: np.ndarray,
                **kwargs):
        """Assesses population fitness using supplied mapping
        
        Args:
            population (np.ndarray): the population to assess fitness
            points (np.ndarray): 1D array of projected points in same order as pool
            **kwargs: Arbitrary keyword arguments
                - scale: Optional scale factor to apply to fitness values.
                         If provided, overrides the value from context.

        Returns:
            np.ndarray: 1D array of float

        """
        # Get scale factor from kwargs or use the one from context
        scale = kwargs.get('scale', self.fitness_scale)
        
        # Calculate fitness and apply scaling
        return np.sum(points[population], axis=1) * scale
