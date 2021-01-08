# pangadfs/pangadfs/fitness.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import numpy as np

from pangadfs.base import FitnessBase


class FitnessDefault(FitnessBase):

    def fitness(self,
                *, 
                population: np.ndarray, 
                points: np.ndarray):
        """Assesses population fitness using supplied mapping
        
        Args:
            population (np.ndarray): the population to assess fitness
            points (np.ndarray): 1D array of projected points in same order as pool

        Returns:
            np.ndarray: 1D array of float

        """
        return np.sum(points[population], axis=1)


