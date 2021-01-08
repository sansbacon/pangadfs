# pangadfs/tests/test_fitness.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import numpy as np
import pytest

from pangadfs.fitness import *


def test_fitness_default(p, pop, tprint):
    points = p['proj'].values
    fitness = FitnessDefault().fitness(
      population=pop, points=points
    )
    assert isinstance(fitness, np.ndarray)
    assert fitness.dtype == 'float64'

