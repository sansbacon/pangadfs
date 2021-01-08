# pangadfs/pangadfs/misc.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def diversity(population: np.ndarray) -> np.ndarray:
    """Calculates diversity of lineups
    
    Args:
        population (np.ndarray): the population

    Returns:
        np.ndarray: is square, shape len(population) x len(population)

    """
    uniques = np.unique(population)
    a = (population[..., None] == uniques).sum(1)
    return np.einsum('ij,kj->ik', a, a)


def multidimensional_shifting(elements: Iterable, num_samples: int, sample_size: int, probs: Iterable) -> np.ndarray:
    """Based on https://medium.com/ibm-watson/incredibly-fast-random-sampling-in-python-baf154bd836a
    
    Args:
        elements (iterable): iterable to sample from, typically a dataframe index
        num_samples (int): the number of rows (e.g. initial population size)
        sample_size (int): the number of columns (e.g. team size)
        probs (iterable): is same size as elements

    Returns:
        ndarray with shape (num_samples, sample_size)
        
    """
    replicated_probabilities = np.tile(probs, (num_samples, 1))
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    shifted_probabilities = random_shifts - replicated_probabilities
    samples = np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]
    return elements.to_numpy()[samples]


def parents(population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evenly splits population
    
    Args:
        population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: population split into half

    """
    fathers, mothers = np.array_split(population, 2)
    size = min(len(fathers), len(mothers))
    return fathers[:size], mothers[:size]
