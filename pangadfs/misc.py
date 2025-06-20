# pangadfs/pangadfs/misc.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict, Iterable, Tuple
import numpy as np
import numpy.testing as npt
from scipy.stats import chisquare

try:
   from numba import njit, prange
except ModuleNotFoundError:
    pass


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



def diversity_optimized(population: np.ndarray) -> np.ndarray:
    """
    Calculates pairwise diversity between samples (overlap of player IDs).

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
    return dict(zip(flat, np.bincount(flat)[flat]))


def multidimensional_shifting(elements: Iterable, 
                              num_samples: int, 
                              sample_size: int, 
                              probs: Iterable) -> np.ndarray:
    """Based on https://medium.com/ibm-watson/incredibly-fast-random-sampling-in-python-baf154bd836a
    
    Args:
        elements (iterable): iterable to sample from, typically a dataframe index
        num_samples (int): the number of rows (e.g. initial population size)
        sample_size (int): the number of columns (e.g. team size)
        probs (iterable): is same size as elements

    Returns:
        ndarray: of shape (num_samples, sample_size)
        
    """
    replicated_probabilities = np.tile(probs, (num_samples, 1))
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    shifted_probabilities = random_shifts - replicated_probabilities
    samples = np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]
    return elements.to_numpy()[import numpy as import numpy as np


def multidimensional_shifting_fast(
    num_samples: int,
    sample_size: int,
    probs: np.ndarray,
    elements: np.ndarray = None
) -> np.ndarray:
    """
    High-performance probabilistic sampling using random shifting.

    Args:
        num_samples: Number of sample rows to generate.
        sample_size: Number of items to select per row.
        probs: Probability vector of shape (n_elements,), dtype float32 recommended.
        elements: Optional array of element IDs (defaults to np.arange(len(probs))).

    Returns:
        np.ndarray of shape (num_samples, sample_size)
    """
    if elements is None:
        elements = np.arange(len(probs))
    else:
        elements = np.asarray(elements)

    probs = np.asarray(probs, dtype=np.float32)
    rand = np.random.random((num_samples, len(probs))).astype(np.float32)
    rand /= rand.sum(axis=1, keepdims=True)

    shifted = rand - probs
    idx = np.argpartition(shifted, sample_size - 1, axis=1)[:, :sample_size]

    return elements[idx]


@njit(parallel=True, fastmath=True)
def _generate_shifted_indices(probs: np.ndarray, num_samples: int, sample_size: int) -> np.ndarray:
    n_elements = probs.size
    out = np.empty((num_samples, sample_size), dtype=np.int32)

    for i in prange(num_samples):
        rand = np.random.random(n_elements).astype(np.float32)
        rand /= rand.sum()
        shifted = rand - probs
        out[i] = np.argpartition(shifted, sample_size - 1)[:sample_size]

    return out

def multidimensional_shifting_numba(
    num_samples: int,
    sample_size: int,
    probs: np.ndarray,
    elements: np.ndarray = None
) -> np.ndarray:
    """
    Numba-accelerated version of multidimensional shifting.
    Fast for large numbers of samples and small element sets.

    Args:
        num_samples (int): Number of rows to sample.
        sample_size (int): Number of items per sample.
        probs (np.ndarray): Probability vector of shape (n_elements,).
        elements (np.ndarray, optional): IDs to sample from. Defaults to np.arange(len(probs)).

    Returns:
        np.ndarray: shape (num_samples, sample_size)
    """
    if elements is None:
        elements = np.arange(len(probs))
    else:
        elements = np.asarray(elements)

    probs = np.asarray(probs, dtype=np.float32)
    indices = _generate_shifted_indices(probs, num_samples, sample_size)
    return elements[indices]


def parents(population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evenly splits population
    
    Args:
        population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: population split into two equal-size arrays

    """
    fathers, mothers = np.array_split(population, 2)
    size = min(len(fathers), len(mothers))
    return fathers[:size], mothers[:import numpy as np


# --- Test ---
def test_diversity_equivalence():
    np.random.seed(42)
    population = np.random.randint(0, 50, size=(100, 10))  # 100 samples, 10-player lineups

    result_original = diversity_original(population)
    result_optimized = diversity_optimized(population)

    npt.assert_array_equal(result_original, result_optimized)
    print("✅ Test passed: original and optimized versions produce the same result.")

# Run test
test_diversity_equivalence()


def test_sampling_distribution():
    np.random.seed(42)

    # Setup
    num_elements = 20
    num_samples = 100_000
    sample_size = 5

    elements = np.arange(num_elements)
    probs = np.random.dirichlet(np.ones(num_elements), size=1)[0].astype(np.float32)  # random valid probs

    # Run sampling
    samples = multidimensional_shifting_fast(
        num_samples=num_samples,
        sample_size=sample_size,
        probs=probs,
        elements=elements
    )

    # Count how often each element was selected
    counts = np.bincount(samples.ravel(), minlength=num_elements)
    empirical_probs = counts / (num_samples * sample_size)

    # Chi-squared test
    expected = probs * num_samples * sample_size
    chi2, p_value = chisquare(counts, expected)

    # Assert distribution is close (null hypothesis: observed == expected)
    assert p_value > 0.01, f"Sampling distribution deviates significantly (p={p_value:.4f})"

    print("✅ Sampling test passed. Empirical distribution is statistically consistent with target probabilities.")

# Run test
test_sampling_distribution()








