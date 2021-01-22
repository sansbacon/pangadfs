# pangadfs/pangadfs/penalty.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

"""
# Penalty framework

Idea is to replace optimizer rules with penalties. Penalties can be negative (bad) or positive (good).
* Advantages 
    * Doesn't throw away reasonable options (125% ownership arbitrary) and is flexible.
    * Does not require absurdly complex optimizer rules.
    * Can easily layer penalties on top of each other.
* Disadvantages
    * Takes some fiddling to get the parameters correct.

# Possible penalties

* Individual ownership penalty (global or just high-owned)
* Cumulative ownership penalty (global or just high-owned)
* Distances (too many similar lineups)
* Diversity (another way of measuring too many similar lineups)
* Position combinations (QB vs DST, WR + own DST, etc.)

"""

import numpy as np
from pangadfs.base import PenaltyBase


class DistancePenalty(PenaltyBase):

    def penalty(self, *, population: np.ndarray) -> np.ndarray:
        """Calculates distance penalty for overlapping lineups
        
        Args:
            population (np.ndarray): the population

        Returns:
            np.ndarray: 1D array of float

        TODO: add parameters for positional weighting
              that is, can prioritize distance at WR or other position
              conversely, can deprioritize distance at RB or other position
        """
        # one-hot encoded population
        # so, assume pool has ids 0, 1, 2, 3, 4
        # lineup is 1, 2
        # ohe would be [0, 1, 1, 0, 0] for that lineup
        ohe = np.sum((np.arange(population.max()) == population[...,None]-1).astype(int), axis=1)

        # now calculate distance between individuals in population
        # dist is a square matrix same length as population
        b = ohe.reshape(ohe.shape[0], 1, ohe.shape[1])
        dist = np.sqrt(np.einsum('ijk, ijk->ij', ohe-b, ohe-b))
        return 0 - ((dist - dist.mean()) / dist.std())


class DiversityPenalty(PenaltyBase):

    def penalty(self, *, population: np.ndarray) -> np.ndarray:
        """Calculates diversity penalty for overlapping lineups
        
        Args:
            population (np.ndarray): the population

        Returns:
            np.ndarray: 1D array of float

        """
        uniques = np.unique(population)
        a = (population[..., None] == uniques).sum(1)
        out = np.einsum('ij,kj->ik', a, a)
        diversity = np.sum(out, axis=1) / population.size
        return 0 - ((diversity - diversity.mean()) / diversity.std())


class OwnershipPenalty(PenaltyBase):

    def penalty(self, *, ownership: np.ndarray, base: float =3, boost: float = 2) -> np.ndarray:
        """Calculates penalties that are inverse to projected ownership
        
        Args:
            ownership (np.ndarray): 1D array of ownership
            base (int): the logarithm base, default 3
            boost (int): the constant to boost low-owned players
            
        Returns:
            np.ndarray: 1D array of penalties
            
        """
        return 0 - np.log(ownership) / np.log(base) + boost


class HighOwnershipPenalty(PenaltyBase):

    def penalty(self, *, ownership: np.ndarray, base: float =3, boost: float = 2) -> np.ndarray:
        """Calculates penalties that are inverse to projected ownership
        
        Args:
            ownership (np.ndarray): 1D array of ownership
            base (int): the logarithm base, default 3
            boost (int): the constant to boost low-owned players
            
        Returns:
            np.ndarray: 1D array of penalties
            
        TODO: implement this method
        """
        pass
        #return 0 - np.log(ownership) / np.log(base) + boost


