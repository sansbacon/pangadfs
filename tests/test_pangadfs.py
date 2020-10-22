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
    

@pytest.fixture
def results(test_directory):
    return pd.read_csv(test_directory / "results.csv")
    

def test_players(players):
    """Tests team_counts"""
    assert isinstance(players, pd.core.api.DataFrame)


def test_initial_population(benchmark, players, initial_size, tprint):
    pg = PanGaDFS(player_pool=players, 
                  initial_size=initial_size,
                  roulette_method='composite')
    lineups = pg.initial_population()
    lineup = lineups.iloc[0:9, :]
    assert len(lineups) == initial_size * PLAYERS_IN_LINEUP


def test_breed_two(lu, tprint):
    pg = PanGaDFS(player_pool=players)
    two_lu = (lu.iloc[0:9, :], lu.iloc[9:18, :])
    new_lu = pg._breed_two(two_lu)
    assert len(new_lu) == len(two_lu[0])


def test_breed(players, tprint):
    pg = PanGaDFS(player_pool=players, 
                  initial_size=10)
    _ = pg.initial_population()
    new_lineups = pg.breed()
    tprint(new_lineups.iloc[0:9, :].to_string())
    assert len(new_lineups) == len(pg.lineups)


def test_breed_generations(players, tprint):
    pg = PanGaDFS(player_pool=players, 
                  initial_size=500)
    _ = pg.initial_population()
    oldmaxfit = pg.lineups.fitness.max()
    tprint(pg.lineups.loc[pg.lineups.fitness == oldmaxfit, :].to_string())
    new_lineups = pg.breed()
    maxfit = new_lineups.fitness.max()
    tprint(new_lineups.loc[new_lineups.fitness == maxfit, :].to_string())
    assert len(new_lineups) == len(pg.lineups)

    all_lineups = pd.concat([pg.lineups, new_lineups])
    pg.lineups = all_lineups.loc[all_lineups.fitness > all_lineups.fitness.median(), :]
    new_lineups = pg.breed()
    maxfit = new_lineups.fitness.max()
    tprint(new_lineups.loc[new_lineups.fitness == maxfit, :].to_string())
    assert len(new_lineups) == len(pg.lineups)


def test_optimize_past(results, tprint):
    tprint(results)
    assert not results.empty()