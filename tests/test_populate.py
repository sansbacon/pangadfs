# pangadfs/tests/test_populate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
import pytest

from pangadfs.populate import *


def test_populate_default(pp, pm, tprint):
    size = 5
    population = PopulateDefault().populate(
      pospool=pp, posmap=pm, population_size=size
    )
    assert isinstance(population, np.ndarray)
    assert len(population) == size