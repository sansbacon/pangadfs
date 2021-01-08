# pangadfs/tests/test_pospool.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import random
import pandas as pd
import pytest

from pangadfs.pospool import *


def test_pospool_default(p, pf):
    pospool = PospoolDefault().pospool(
      pool=p, posfilter=pf, column_mapping={} 
    )    
    assert isinstance (pospool, dict)
    key = random.choice(list(pospool.keys()))
    assert isinstance(pospool[key], pd.core.api.DataFrame)

