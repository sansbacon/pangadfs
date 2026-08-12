# pangadfs/tests/test_ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
from stevedore import driver, named

from pangadfs.ga import GeneticAlgorithm


@pytest.fixture
def	ctx(test_directory):
    return {
		'ga_settings': {
			'n_generations': 20,
			'population_size': 5000,
			'stop_criteria': 5,
			'points_column': 'proj',
			'salary_column': 'salary',
			'position_column': 'pos',
			'csvpth': test_directory / 'test_pool.csv'
		},

		'site_settings': {
			'salary_cap': 50000,
			'posmap': {'DST': 1, 'QB': 1, 'TE': 1, 'RB': 2, 'WR': 3, 'FLEX': 7},
			'lineup_size': 9,
			'posfilter': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4, 'FLEX': 8}
		}
	}


@pytest.fixture
def dms():
    plugins = [ns for ns in GeneticAlgorithm.PLUGIN_NAMESPACES if ns != 'validate']
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
def ga(ctx, dms, ems):
    return GeneticAlgorithm(ctx=ctx, driver_managers=dms, extension_managers=ems)


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
def pp(p, pf, ga, ctx):
    return ga.pospool(ctx=ctx, pool=p, posfilter=pf, column_mapping={}, flex_positions=('RB', 'WR', 'TE'))    


@pytest.fixture
def site_settings():
    return {
        'salary_cap': 50000,
        'posmap': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}
    }


def test_init(ctx, dms, ems):
    obj = GeneticAlgorithm(ctx=ctx, driver_managers=dms, extension_managers=ems)
    assert obj is not None
    obj = GeneticAlgorithm(ctx=ctx, driver_managers=dms, extension_managers=None)
    assert obj is not None
    obj = GeneticAlgorithm(ctx=ctx, driver_managers=None, extension_managers=ems)
    assert obj is not None


def test_pool(test_directory, ga):
    csvpth = test_directory / 'test_pool.csv'
    pool = ga.pool(csvpth=csvpth)    
    assert isinstance(pool, pd.core.api.DataFrame)
    assert not pool.empty


def test_pospool(p, pf, ga):
    pospool = ga.pospool(
      pool=p, posfilter=pf, column_mapping={}, flex_positions=('RB', 'WR', 'TE')
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
    points = p['proj'].values
    fitness = ga.fitness(
      population=pop, points=points
    )
    assert isinstance(fitness, np.ndarray)
    assert fitness.dtype == 'float64'


def test_select(p, pop, ga):
    points = p['proj'].values
    fitness = ga.fitness(
      population=pop, points=points
    )

    selected = ga.select(
      population=pop,
      population_fitness=fitness,
      n=len(pop)
    )

    assert isinstance(selected, np.ndarray)
    assert selected.dtype == 'int64'


def test_crossover(p, pop, ga):
    points = p['proj'].values
    newpop = ga.crossover(population=pop, method='uniform')
    assert isinstance(newpop, np.ndarray)
    assert newpop.dtype == 'int64'


def test_mutate(pop, ga):
    mpop = ga.mutate(population=pop, mutation_rate=.05)
    assert pop.shape == mpop.shape
    assert not np.array_equal(pop, mpop)


def test_validate(p, pop, ga):
    salaries = p['salary'].values
    scap = 50000
    vpop = ga.validate(
      population=pop, salaries=salaries, salary_cap=scap
    )
    assert isinstance(vpop, np.ndarray)
    assert vpop.size > 0
    assert len(pop) >= len(vpop)


def test_di_plugins_used_for_dispatch(test_directory):
    class PoolPlugin:
        def pool(self, *, csvpth=None, **kwargs):
            return pd.read_csv(csvpth)

    ga = GeneticAlgorithm(plugins={'pool': PoolPlugin()})
    pool = ga.pool(csvpth=test_directory / 'test_pool.csv')
    assert isinstance(pool, pd.core.api.DataFrame)
    assert not pool.empty


def test_di_validate_plugin_chain():
    class DropLast:
        def validate(self, *, population=None, salaries=None, **kwargs):
            return population[:-1]

    class DropFirst:
        def validate(self, *, population=None, salaries=None, **kwargs):
            return population[1:]

    population = np.arange(20).reshape(4, 5)
    salaries = np.ones(20)
    ga = GeneticAlgorithm(plugins={'validate': [DropLast(), DropFirst()]})

    out = ga.validate(population=population, salaries=salaries)
    assert out.shape == (2, 5)


def test_di_crossover_agg():
    class AddOne:
        def crossover(self, *, population=None, **kwargs):
            return population + 1

    class AddTwo:
        def crossover(self, *, population=None, **kwargs):
            return population + 2

    population = np.array([[1, 2], [3, 4]])
    ga = GeneticAlgorithm(plugins={'crossover': [AddOne(), AddTwo()]})

    out = ga.crossover(population=population, agg=True)
    assert out.shape == (4, 2)
    assert np.array_equal(out[:2], population + 1)
    assert np.array_equal(out[2:], population + 2)
