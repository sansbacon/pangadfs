# gadfs/gadfs/gasettings.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict

import dacite


logging.getLogger(__name__).addHandler(logging.NullHandler())


@dataclass
class GASettings:
    csvpth: Path
    n_generations: int = 20
    points_column: str = 'proj'
    population_size: int = 5000
    position_column: str = 'pos'
    salary_column: str = 'salary'
    stop_criteria: int = 10


@dataclass
class SiteSettings:
    lineup_size: int
    posfilter: Dict[str, int]
    posmap: Dict[str, int]
    salary_cap: int


@dataclass
class AppSettings:
    ga_settings: GASettings
    site_settings: SiteSettings
    driver_managers: Dict[str, str] = {} 
    extension_managers: Dict[str, str] = {}


def ctx_from_dict(raw_cfg):
    """Creates config object from dictionary"""
    converters = {
        Path: Path,
    }

    # create and validate the Configuration object
    return dacite.from_dict(
        data_class=AppSettings, data=raw_cfg,
        config=dacite.Config(type_hooks=converters),
    )