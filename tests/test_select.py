# pangadfs/tests/test_select.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
import pytest

from pangadfs.select import SelectDefault
from pangadfs.fitness import FitnessDefault


def test_select_default(p, pop):
    points = p['proj'].values
    fitness = FitnessDefault().fitness(
      population=pop, points=points
    )

    params = {
      'population': pop,
      'population_fitness': fitness,
      'n': len(pop) // 2
    }

    for method in ('roulette', 'sus', 'scaled', 'rank', 'tournament'):
        newparams = {**params, **{'method': method}}
        newpop = SelectDefault().select(**newparams)
        assert isinstance(newpop, np.ndarray)
        assert newpop.dtype == 'int64'
        assert len(newpop) == len(pop) // 2
