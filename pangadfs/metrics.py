# pangadfs/pangadfs/metrics.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

"""Population diversity and exposure metrics."""

from typing import Dict
import numpy as np


def diversity(population: np.ndarray) -> np.ndarray:
    """Calculates pairwise overlap between lineups (player IDs).

    Uses sparse count matrix + dot product for efficiency.

    Args:
        population (np.ndarray): shape (N, K), where each row is a lineup

    Returns:
        np.ndarray: shape (N, N), matrix of pairwise overlap scores
    """
    uniques, inverse = np.unique(population, return_inverse=True)
    N, K = population.shape
    U = len(uniques)

    # Construct count matrix a: shape (N, U)
    a = np.zeros((N, U), dtype=np.uint8)
    rows = np.repeat(np.arange(N), K)
    np.add.at(a, (rows, inverse), 1)

    # Pairwise dot product: overlap between lineups
    return a @ a.T


def calculate_jaccard_diversity(lineup1, lineup2) -> float:
    """Calculate Jaccard diversity between two lineups.

    Args:
        lineup1: First lineup (array-like of player IDs)
        lineup2: Second lineup (array-like of player IDs)

    Returns:
        float: Jaccard diversity (1 - Jaccard similarity)

    Examples:
        >>> calculate_jaccard_diversity([1, 2, 3, 4, 5], [1, 2, 6, 7, 8])
        0.667
    """
    set1 = set(lineup1)
    set2 = set(lineup2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    similarity = intersection / union if union > 0 else 0.0
    return 1.0 - similarity


def exposure(population: np.ndarray) -> Dict[int, int]:
    """Returns dict of player index to appearance count across all lineups.

    Args:
        population (np.ndarray): the population (2D array of player indices)

    Returns:
        Dict[int, int]: key is player index, value is count of appearances

    Examples:
        >>> fittest_population = population[np.where(fitness > np.percentile(fitness, 97))]
        >>> exp = exposure(fittest_population)
        >>> top_exposure = np.argpartition(np.array(list(exp.values())), -10)[-10:]
    """
    flat = population.flatten()
    return dict(zip(flat, np.bincount(flat)[flat]))
