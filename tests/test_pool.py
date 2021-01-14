# pangadfs/tests/test_pool.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import pandas as pd
import pytest

from pangadfs.pool import *


def test_pool_default(test_directory):
    csvpth = test_directory / 'test_pool.csv'
    pool = PoolDefault().pool(csvpth=csvpth)    
    pool = PoolDefault().pool(csvpth=csvpth)    
    assert isinstance(pool, pd.core.api.DataFrame)
    assert not pool.empty

