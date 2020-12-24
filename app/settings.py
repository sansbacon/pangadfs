# pangadfs/app/settings.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import os
from pathlib import Path

from dotenv import load_dotenv
from stevedore import driver, named


env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

ga_settings = {
  'n_generations': 20,
	'population_size': 20000,
	'points_column': 'proj',
	'salary_column': 'salary',
	'csvpth': Path(__file__).parent / 'pool.csv'
}

site_settings = {
  'salary_cap': 50000,
	'posmap': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}
}

plugins = ('crossover', 'populate', 'fitness', 'mutate', 'pool', 'pospool')
plugin_names = {p: os.getenv(f'PANGADFS_{p.upper()}_PLUGIN', f'{p}_default') for p in plugins}

dmgrs = {
  k: driver.DriverManager(namespace=f'pangadfs.{k}', name=v, invoke_on_load=True)
  for k, v in plugin_names.items()
}

emgrs = {
  'validate': named.NamedExtensionManager(namespace='pangadfs.validate', 
                                          names=['validate_salary', 'validate_duplicates'], 
                                          invoke_on_load=True, 
                                          name_order=True)
}