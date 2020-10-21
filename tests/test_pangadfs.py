# -*- coding: utf-8 -*-
# tests/test_db.py
import logging
import random
import pandas as pd
import pytest

from pangadfs import *


LOGGER = logging.getLogger(__name__)
PLAYERS_IN_LINEUP = 9


@pytest.fixture
def initial_size():
    return random.choice([10, 20, 30])


@pytest.fixture
def lu(test_directory):
    return pd.read_csv(test_directory / 'lineups.csv')


@pytest.fixture
def players(test_directory):
    return pd.read_csv(test_directory / "players.csv")
    

def test_players(players):
    """Tests team_counts"""
    assert isinstance(players, pd.core.api.DataFrame)


def test_initial_population(benchmark, players, initial_size, tprint):
    pg = PanGaDFS(player_pool=players, 
                  initial_size=initial_size,
                  roulette_method='composite')
    lineups = benchmark.pedantic(pg.initial_population, iterations=1, rounds=2)
    lineup = lineups.iloc[0:9, :]
    tprint(lineup)
    tprint((round(lineup.proj.sum(), 2), lineup.salary.sum()))
    assert len(lineups) == initial_size * PLAYERS_IN_LINEUP


def test_breed_two(lu, tprint):
    pg = PanGaDFS(player_pool=players)
    two_lu = (lu.iloc[0:9, :], lu.iloc[9:18, :])
    new_lu = pg._breed_two(two_lu)
    assert len(new_lu) == len(two_lu[0])


def test_breed(players, tprint):
    pg = PanGaDFS(player_pool=players, 
                  initial_size=10)
    lineups = pg.initial_population()
    new_lineups = pg.breed()
    assert len(new_lineups) == len(lineups)

