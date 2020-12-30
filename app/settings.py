# pangadfs/app/settings.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

from pathlib import Path

from stevedore import driver, named

ctx = {
  'ga_settings': {
    'n_generations': 20,
    'population_size': 5000,
    'stop_criteria': 5,
    'points_column': 'proj',
    'salary_column': 'salary',
    'position_column': 'pos',
    'csvpth': Path(__file__).parent / 'pool.csv'
  },

  'site_settings': {
    'salary_cap': 50000,
    'posmap': {'DST': 1, 'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 7},
    'lineup_size': 9,
    'posthresh': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 6, 'DST': 4, 'FLEX': 8}
  }
}

plugins = ('crossover', 'populate', 'fitness', 'mutate', 'pool', 'pospool')

dmgrs = {
  p: driver.DriverManager(namespace=f'pangadfs.{p}', name=f'{p}_default', invoke_on_load=True)
  for p in plugins
}

emgrs = {
  'validate': named.NamedExtensionManager(
      namespace='pangadfs.validate',
      names=['validate_salary', 'validate_duplicates'], 
      invoke_on_load=True, 
      name_order=True
  )
}