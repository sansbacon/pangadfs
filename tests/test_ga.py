# pangadfs/tests/test_ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
from stevedore import driver, named

from pangadfs import GeneticAlgorithm


@pytest.fixture
def dms():
    plugins = ('crossover', 'populate', 'fitness', 'mutate', 'pool', 'pospool')
    mapping = {p: f'{p}_default' for p in plugins}
    return {
        k: driver.DriverManager(namespace=f'pangadfs.{k}', name=v, invoke_on_load=True)
        for k, v in mapping.items()
    }


@pytest.fixture
def ems():
    names = ['validate_salary', 'validate_duplicates']
    mgr = named.NamedExtensionManager(namespace=f'pangadfs.validate', names=names, invoke_on_load=True, name_order=True)
    return {'validate': mgr}


@pytest.fixture
def ga(dms, ems):
    return GeneticAlgorithm(driver_managers=dms, extension_managers=ems)


@pytest.fixture
def ga_settings(test_directory):
    return {
        'n_generations': 20,
        'population_size': 5000,
        'points_column': 'proj',
        'salary_column': 'salary',
        'csvpth': test_directory / 'test_pool.csv'
    }


@pytest.fixture
def p(test_directory, ga):
    return ga.pool(csvpth=test_directory / 'test_pool.csv')


@pytest.fixture
def pf():
    return {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6}


@pytest.fixture
def pm():
	return {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}


@pytest.fixture
def pop(pp, pm, ga):
    return ga.populate(
      pospool=pp, posmap=pm, population_size=1000
    )


@pytest.fixture
def pp(p, pf, ga):
    return ga.pospool(pool=p, posfilter=pf, column_mapping={})    


@pytest.fixture
def site_settings():
    return {
        'salary_cap': 50000,
        'posmap': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}
    }


def test_init(dms, ems, tprint):
    obj = GeneticAlgorithm(driver_managers=dms, extension_managers=ems)
    assert obj is not None
    obj = GeneticAlgorithm(driver_managers=dms, extension_managers=None)
    assert obj is not None
    obj = GeneticAlgorithm(driver_managers=None, extension_managers=ems)
    assert obj is not None
    with pytest.raises(ValueError):
        obj = GeneticAlgorithm(driver_managers=None, extension_managers=None)


def test_pool(test_directory, ga):
    csvpth = test_directory / 'test_pool.csv'
    pool = ga.pool(csvpth=csvpth)    
    assert isinstance(pool, pd.core.api.DataFrame)
    assert not pool.empty


def test_pospool(p, pf, ga):
    pospool = ga.pospool(
      pool=p, posfilter=pf, column_mapping={} 
    )
    assert isinstance (pospool, dict)
    key = random.choice(list(pospool.keys()))
    assert isinstance(pospool[key], pd.core.api.DataFrame)


def test_populate(pp, pm, ga):
    size = 1000
    population = ga.populate(
      pospool=pp, posmap=pm, population_size=size
    )
    assert isinstance(population, np.ndarray)
    assert len(population) == size


def test_fitness(p, pop, ga):
    points_mapping = dict(zip(p.index, p['proj']))
    fitness = ga.fitness(
      population=pop, points_mapping=points_mapping
    )
    assert isinstance(fitness, np.ndarray)
    assert fitness.dtype == 'float64'


def test_crossover(p, pop, ga):
    points_mapping = dict(zip(p.index, p['proj']))
    popfit = ga.fitness(
      population=pop, points_mapping=points_mapping
    )
    newpop = ga.crossover(
      population=pop, population_fitness=popfit 
    )
    assert isinstance(newpop, np.ndarray)
    assert newpop.dtype == 'int64'


def test_mutate(pop, ga):
    mpop = ga.mutate(population=pop)
    assert pop.shape == mpop.shape
    assert not np.array_equal(pop, mpop)


def test_validate(p, pop, ga):
    smap = dict(zip(p.index, p['salary']))
    scap = 50000
    vpop = ga.validate(
      population=pop, salary_mapping=smap, salary_cap=scap
    )
    assert isinstance(vpop, np.ndarray)
    assert vpop.size > 0
    assert len(pop) >= len(vpop)
