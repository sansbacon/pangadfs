# pangadfs/tests/test_mutate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
import pytest

from pangadfs.mutate import MutateDefault


def test_mutate_default(pop):
    mpop = MutateDefault().mutate(population=pop)
    assert pop.shape == mpop.shape
    assert not np.array_equal(pop, mpop)
