# pangadfs/pangadfs/sampling.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

"""Probabilistic sampling utilities for population generation."""

from typing import Iterable, Tuple
import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ModuleNotFoundError:
    HAS_NUMBA = False


def multidimensional_shifting(
    elements: Iterable,
    num_samples: int,
    sample_size: int,
    probs: Iterable
) -> np.ndarray:
    """High-performance probabilistic sampling using random shifting.

    Based on https://medium.com/ibm-watson/incredibly-fast-random-sampling-in-python-baf154bd836a
    Uses float32 arithmetic internally. Auto-selects numba path if available.

    Args:
        elements: Iterable to sample from (DataFrame index, array, etc.)
        num_samples: Number of sample rows to generate.
        sample_size: Number of items to select per row.
        probs: Probability vector, same length as elements.

    Returns:
        np.ndarray of shape (num_samples, sample_size)
    """
    if hasattr(elements, 'to_numpy'):
        elements = elements.to_numpy()
    else:
        elements = np.asarray(elements)

    probs = np.asarray(probs, dtype=np.float32)

    # Use numba-accelerated path if available
    if HAS_NUMBA:
        indices = _generate_shifted_indices(probs, num_samples, sample_size)
        return elements[indices]

    # Vectorized numpy path (float32 for speed)
    rand = np.random.random((num_samples, len(probs))).astype(np.float32)
    rand /= rand.sum(axis=1, keepdims=True)
    shifted = rand - probs
    idx = np.argpartition(shifted, sample_size - 1, axis=1)[:, :sample_size]
    return elements[idx]


def _generate_shifted_indices_impl(probs: np.ndarray, num_samples: int, sample_size: int) -> np.ndarray:
    """Core sampling loop (plain Python fallback or numba target)."""
    n_elements = probs.size
    out = np.empty((num_samples, sample_size), dtype=np.int32)

    for i in range(num_samples):
        rand = np.random.random(n_elements).astype(np.float32)
        rand /= rand.sum()
        shifted = rand - probs
        out[i] = np.argpartition(shifted, sample_size - 1)[:sample_size]

    return out


if HAS_NUMBA:
    _generate_shifted_indices = njit(parallel=True, fastmath=True)(_generate_shifted_indices_impl)
else:
    _generate_shifted_indices = _generate_shifted_indices_impl


def parents(population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evenly splits population into two groups for crossover.

    Args:
        population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: population split into two equal-size arrays
    """
    fathers, mothers = np.array_split(population, 2)
    size = min(len(fathers), len(mothers))
    return fathers[:size], mothers[:size]
