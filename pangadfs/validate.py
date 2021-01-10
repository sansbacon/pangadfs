# pangadfs/pangadfs/validate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import numpy as np
from numpy_indexed import unique

from pangadfs.base import ValidateBase


class DuplicatesValidate(ValidateBase):

    def validate(self,
                 *, 
                 population: np.ndarray, 
                 **kwargs) -> np.ndarray:
        """Removes duplicate individuals from population
        
            Args:
                population (np.ndarray): the population to validate
                **kwargs: keyword arguments for plugins

            Returns:
                np.ndarray: same width as population, likely has less rows

        """
        # the first part eliminates individuals with duplicate genes
        # the second part eliminates duplicate individuals
        population_sorted = np.sort(population, axis=-1)
        population = population[(population_sorted[...,1:] != population_sorted[..., :-1]).all(-1)]
        return unique(np.sort(population, axis=1))


class SalaryValidate(ValidateBase):

    def validate(self,
                 *, 
                 population: np.ndarray,
                 salaries: np.ndarray,
                 salary_cap: int, 
                 **kwargs) -> np.ndarray:
        """Ensures valid individuals in population
        
            Args:
                population (np.ndarray): the population to validate
                salaries (np.ndarray): 1D where indices are in same order as player indices
                salary_cap (int): the salary cap, e.g., 50000 or 60000
                **kwargs: keyword arguments for plugins

            Returns:
                np.ndarray: same width as population, likely has less rows

        """
        popsal = np.sum(salaries[population], axis=1)
        return population[popsal <= salary_cap]
