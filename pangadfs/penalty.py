# pangadfs/pangadfs/penalty.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np


def diversity_penalty(population: np.ndarray = None) -> np.ndarray:
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


def ownership_penalty(ownership, base=3, boost=2) -> np.ndarray:
    """Returns 1D array of penalties that are inverse to projected ownership
    
    Args:
        ownership (np.ndarray): 1D array of ownership
        base (int): the logarithm base, default 3
        boost (int): the constant to boost low-owned players
        
    Returns:
        np.ndarray: 1D array of penalties
        
    """
    return 0 - np.log(ownership) / np.log(base) + boost
    
