# gadfs/gadfs/gasettings.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import dacite


logging.getLogger(__name__).addHandler(logging.NullHandler())


@dataclass
class GASettings:
    csvpth: str = 'pool.csv'
    crossover_method: str = 'uniform'
    elite_divisor: int = 5
    elite_method: str = 'fittest'
    mutation_rate: float = 0.05
    n_generations: int = 20
    points_column: str = 'proj'
    population_size: int = 5000
    position_column: str = 'pos'
    salary_column: str = 'salary'
    select_method: str = 'roulette'
    stop_criteria: int = 10
    verbose: bool = True


@dataclass
class PluginSettings:
    driver_managers: Dict[str, Any] = None
    extension_managers: Dict[str, Any] = None


@dataclass
class SiteSettings:
    lineup_size: int
    posfilter: Dict[str, int]
    posmap: Dict[str, int]
    salary_cap: int
    flex_positions: Iterable = ('RB', 'WR', 'TE')


@dataclass
class AppSettings:
    ga_settings: GASettings
    site_settings: SiteSettings
    plugin_settings: PluginSettings


def ctx_from_dict(raw_cfg):
    """Creates config object from dictionary"""

    # create and validate the Configuration object
    return dacite.from_dict(
        data_class=AppSettings, 
        data=raw_cfg
    )