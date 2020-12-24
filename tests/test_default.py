# pangadfs/tests/test_default.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import random

import numpy as np
import pandas as pd
import pytest

from pangadfs import GeneticAlgorithm
from pangadfs.default import *


@pytest.fixture
def p(test_directory):
    csvpth = test_directory / 'test_pool.csv'
    return DefaultPool().pool(csvpth)


@pytest.fixture
def pf():
    return {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6}


@pytest.fixture
def pm():
	return {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}


@pytest.fixture
def pp(p, pf):
    cm = {}
    return DefaultPospool().pospool(
      pool=p, posfilter=pf, column_mapping=cm 
    )    


@pytest.fixture
def pop(pp, pm):
    return DefaultPopulate().populate(
      pospool=pp, posmap=pm, population_size=100
    )


def test_pool_default(test_directory):
    csvpth = test_directory / 'test_pool.csv'
    pool = DefaultPool().pool(csvpth)    
    assert isinstance(pool, pd.core.api.DataFrame)
    assert not pool.empty


def test_pospool_default(p, pf):
    pospool = DefaultPospool().pospool(
      pool=p, posfilter=pf, column_mapping={} 
    )    
    assert isinstance (pospool, dict)
    key = random.choice(list(pospool.keys()))
    assert isinstance(pospool[key], pd.core.api.DataFrame)


def test_populate_default(pp, pm, tprint):
    size = 2
    population = DefaultPopulate().populate(
      pospool=pp, posmap=pm, population_size=size
    )
    assert isinstance(population, np.ndarray)
    assert len(population) == size


def test_fitness_default(p, pop, tprint):
    points_mapping = dict(zip(p.index, p['proj']))
    fitness = DefaultFitness().fitness(
      population=pop, points_mapping=points_mapping
    )
    assert isinstance(fitness, np.ndarray)
    assert fitness.dtype == 'float64'


def test_crossover_default(p, pop):
    points_mapping = dict(zip(p.index, p['proj']))
    popfit = DefaultFitness().fitness(
      population=pop, points_mapping=points_mapping
    )
    newpop = DefaultCrossover().crossover(
      population=pop, population_fitness=popfit 
    )
    assert isinstance(newpop, np.ndarray)
    assert newpop.dtype == 'int64'


def test_mutate_default(pop):
    mpop = DefaultMutate().mutate(population=pop)
    assert pop.shape == mpop.shape
    assert not np.array_equal(pop, mpop)


def test_validate_salary(p, pop):
    smap = dict(zip(p.index, p['salary']))
    scap = 50000
    vpop = SalaryValidate().validate(
      population=pop, salary_mapping=smap, salary_cap=scap
    )
    assert isinstance(vpop, np.ndarray)
    assert vpop.size > 0
    assert len(pop) >= len(vpop)


def test_validate_duplicate(p, pop):
    vpop = DuplicatesValidate().validate(population=pop)
    assert isinstance(vpop, np.ndarray)
    assert vpop.size > 0
    assert len(pop) >= len(vpop)