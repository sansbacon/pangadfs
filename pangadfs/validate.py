# pangadfs/pangadfs/validate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict

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


class FlexDuplicatesValidate(ValidateBase):
    """
    Validates that FLEX positions don't duplicate other positions.
    This replaces the expensive duplicate checking that was in PopulateDefault.
    Uses a more efficient vectorized approach.
    """

    def validate(self, *, population: np.ndarray, posmap: Dict[str, int] = None, **kwargs) -> np.ndarray:
        if len(population) <= 1 or not posmap or 'FLEX' not in posmap:
            return population
        
        # Calculate position boundaries
        pos_boundaries = {}
        start_idx = 0
        for pos, count in posmap.items():
            pos_boundaries[pos] = (start_idx, start_idx + count)
            start_idx += count
        
        # Get FLEX and non-FLEX position indices
        flex_start, flex_end = pos_boundaries['FLEX']
        
        # Extract FLEX and non-FLEX columns
        flex_players = population[:, flex_start:flex_end]
        non_flex_players = population[:, :flex_start]  # Only positions before FLEX
        
        # Use broadcasting to check for duplicates more efficiently
        # This replicates the original logic but in a more efficient way
        if non_flex_players.shape[1] > 0 and flex_players.shape[1] > 0:
            # Check if any FLEX player appears in non-FLEX positions
            dups = (flex_players[..., None] == non_flex_players[:, None, :]).any(-1)
            
            # Find valid FLEX players (those that don't duplicate)
            valid_flex_mask = ~dups
            
            # For each row, select the first valid FLEX player for each FLEX position
            valid_rows = []
            for i in range(len(population)):
                valid_flex_indices = np.where(valid_flex_mask[i])[0]
                if len(valid_flex_indices) >= flex_players.shape[1]:
                    # We have enough valid FLEX players
                    valid_rows.append(i)
            
            return population[valid_rows] if valid_rows else population[:0]  # Return empty with correct shape
        
        return population


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
