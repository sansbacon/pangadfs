# pangadfs/pangadfs/postprocess.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

from typing import Dict
import numpy as np


def exposure(population: np.ndarray = None) -> Dict[int, int]:
    """Returns dict of index: count of individuals

    Args:
        population (np.ndarray): the population

    Returns:
        Dict[int, int]: key is index, value is count of lineup

    Examples:
        >>> fittest_population = population[np.where(fitness > np.percentile(fitness, 97))]
        >>> exposure = population_exposure(fittest_population)
        >>> top_exposure = np.argpartition(np.array(list(exposure.values())), -10)[-10:]
        >>> print([round(i, 3) for i in sorted(top_exposure / len(fittest_population), reverse=True)])            

    """
    flat = population.flatten
    return {id: size for id, size in zip(flat, np.bincount(flat)[flat])}


