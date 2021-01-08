# pangadfs/postprocess.py

from typing import Dict
import numpy as np


def exposure(population: np.ndarray = None) -> Dict[int, int]:
    """Returns dict of index: count of individuals

    Args:
        population (np.ndarray): the population

    Returns:
        Dict[int, int]: key is index, value is count of lineup

    """
    flat = population.flatten
    return {id: size for id, size in zip(flat, np.bincount(flat)[flat])}


