# pangadfs/app/settings.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import os
from pathlib import Path

from dotenv import load_dotenv
from stevedore import driver


env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

plugin_names = {
  'crossover': os.getenv("PANGADFS_CROSSOVER_PLUGIN"),
  'populate': os.getenv("PANGADFS_POPULATE_PLUGIN"),
  'fitness': os.getenv("PANGADFS_FITNESS_PLUGIN"),
  'validate': os.getenv("PANGADFS_VALIDATE_PLUGIN"),
  'mutate': os.getenv("PANGADFS_MUTATE_PLUGIN"),
  'pool': os.getenv("PANGADFS_POOL_PLUGIN"),
  'pospool': os.getenv("PANGADFS_POSPOOL_PLUGIN"),
}

dmgrs = {
  k: driver.DriverManager(namespace=f'pangadfs.{k}', name=v, invoke_on_load=True)
  for k, v in plugin_names.items()
}

ga_settings = {
  'n_generations': 20,
	'population_size': 1000,
	'points_column': 'proj',
	'salary_column': 'salary',
	'csvpth': Path(__file__).parent / 'pool.csv'
}

site_settings = {
  'salary_cap': 50000,
	'posmap': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}
}

