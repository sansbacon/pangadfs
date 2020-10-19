# -*- coding: utf-8 -*-
# tests/test_db.py
import pandas as pd
import pytest

from pangadfs import *


@pytest.fixture
def players(test_directory):
    return pd.read_csv(test_directory / "players.csv")
    

def test_players(players):
    """Tests team_counts"""
    assert isinstance(players, pd.core.api.DataFrame)

