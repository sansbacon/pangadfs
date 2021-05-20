# pangadfs/tests/conftest.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from pathlib import Path
import sys

import pytest

sys.path.append("../pangadfs")

from pangadfs.pool import *
from pangadfs.pospool import *
from pangadfs.populate import *


@pytest.fixture
def pf():
    return {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6}


@pytest.fixture
def pm():
	return {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}


@pytest.fixture
def pp(p, pf):
    cm = {}
    return PospoolDefault().pospool(
      pool=p, posfilter=pf, column_mapping=cm 
    )    


@pytest.fixture
def pop(pp, pm):
    return PopulateDefault().populate(
      pospool=pp, posmap=pm, population_size=100
    )


@pytest.fixture(scope="session", autouse=True)
def root_directory(request):
    """Gets root directory"""
    return Path(request.config.rootdir)


@pytest.fixture(scope="session", autouse=True)
def test_directory(request):
    """Gets root directory of tests"""
    return Path(request.config.rootdir) / "tests"


@pytest.fixture()
def tprint(request, capsys):
    """Fixture for printing info after test, not supressed by pytest stdout/stderr capture"""
    lines = []
    yield lines.append

    with capsys.disabled():
        for line in lines:
            sys.stdout.write("\n{}".format(line))


@pytest.fixture
def p(test_directory):
    csvpth = test_directory / 'test_pool.csv'
    return PoolDefault().pool(csvpth=csvpth)