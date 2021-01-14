# pangadfs/tests/test_crossover.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
import pytest

from pangadfs.crossover import *
from pangadfs.fitness import FitnessDefault


def test_crossover_default(pop):
    newpop = CrossoverDefault().crossover(population=pop, method='uniform')
    assert isinstance(newpop, np.ndarray)
    assert newpop.dtype == 'int64'


def test_crossover_diverse(pop):
    newpop = CrossoverDefault().crossover(population=pop, method='diverse')
    assert isinstance(newpop, np.ndarray)
    assert newpop.dtype == 'int64'


def test_crossover_single_point(pop, tprint):
    """Tests crossover single point"""
    pop2 = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [2, 1, 3, 9, 10, 11, 12, 6, 5],
        [7, 2, 4, 7, 8, 6, 7, 8, 9],
        [6, 1, 3, 13, 15, 10, 12, 6, 5]
    ])

    expected = np.array([
       [ 1,  2,  3,  7,  8,  6,  7,  8,  9],
       [ 2,  1,  3, 13, 15, 10, 12,  6,  5],
       [ 7,  2,  4,  4,  5,  6,  7,  8,  9],
       [ 6,  1,  3,  9, 10, 11, 12,  6,  5]
    ])

    newpop = CrossoverDefault().crossover(population=pop2, method='one_point')
    assert np.array_equal(newpop, expected)
    

def test_crossover_two_point(pop):
    """Tests crossover two point"""
    pop2 = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [2, 1, 3, 9, 10, 11, 12, 6, 5],
        [7, 2, 4, 7, 8, 6, 7, 11, 10],
        [6, 1, 3, 13, 15, 10, 12, 19, 20]
    ])

    expected = np.array([
       [ 1,  2,  3,  7,  8,  6,  7,  8,  9],
       [ 2,  1,  3, 13, 15, 10, 12,  6,  5],
       [ 7,  2,  4,  4,  5,  6,  7,  11,  10],
       [ 6,  1,  3,  9, 10, 11, 12,  19,  20]
    ])

    newpop = CrossoverDefault().crossover(population=pop2, method='two_point')
    assert np.array_equal(newpop, expected)


@pytest.mark.skip
def test_crossover_ox1(p, pop):
    newpop = CrossoverDefault().crossover(population=p, method='ordered')
    assert isinstance(newpop, np.ndarray)
    assert newpop.dtype == 'int64'

