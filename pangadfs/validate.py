# pangadfs/pangadfs/validate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
from numpy_indexed import unique

from pangadfs.base import ValidateBase


class DuplicatesValidate(ValidateBase):

    def validate(self, *, population: np.ndarray, **kwargs) -> np.ndarray:
        if len(population) <= 1:
            return population
        
        # Sort once and reuse
        population_sorted = np.sort(population, axis=1)
        
        # Check for internal duplicates using the already sorted array
        has_internal_dups = (population_sorted[:, 1:] == population_sorted[:, :-1]).any(axis=1)
        population_clean = population[~has_internal_dups]
        
        if len(population_clean) <= 1:
            return population_clean
        
        # Use already sorted array for external duplicate removal
        population_clean_sorted = population_sorted[~has_internal_dups]
        return unique(population_clean_sorted)


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
        if len(population) == 0:
            return population
            
        # Use take for potentially faster indexing
        salary_matrix = np.take(salaries, population)
        popsal = np.sum(salary_matrix, axis=1)
        
        # Use nonzero for potentially faster boolean indexing
        valid_indices = np.nonzero(popsal <= salary_cap)[0]
        return population[valid_indices]