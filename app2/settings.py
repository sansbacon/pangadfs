# pangadfs/app/settings.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import os
from pathlib import Path

from stevedore.named import NamedExtensionManager


ga_settings = {
  'n_generations': 20,
	'population_size': 10000,
	'points_column': 'proj',
	'salary_column': 'salary',
	'csvpth': Path(__file__).parent / 'pool.csv'
}

site_settings = {
  'salary_cap': 50000,
	'posmap': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}
}

enabled_plugins = {
	'crossover': ['crossover_default'], 
	'populate': ['populate_default'], 
	'fitness': ['fitness_default'], 
	'validate': ['validate_default'], 
    'mutate': ['mutate_default'], 
	'pool': ['pool_default'], 
	'pospool': ['pospool_default']
}

dmgrs = {
  k: NamedExtensionManager(namespace=f'pangadfs.{k}', names=v, invoke_on_load=True, name_order=True)
  for k, v in enabled_plugins.items()
}
