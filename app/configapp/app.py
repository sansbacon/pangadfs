# pangadfs/app/app.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from dacite import from_dict
from stevedore import driver, named

from pangadfs.optimizer import Optimizer
from pangadfs.gasettings import GASettings


def main():
	"""Example application using pangadfs"""
	logging.basicConfig(level=logging.INFO)

	data = {
		'ga_settings': {
			'n_generations': 20,
			'population_size': 30000,
			'stop_criteria': 10,
			'points_column': 'proj',
			'salary_column': 'salary',
			'position_column': 'pos',
			'csvpth': Path(__file__).parent / 'pool.csv'
		},

		'site_settings': {
			'salary_cap': 50000,
			'posmap': {'DST': 1, 'QB': 1, 'TE': 1, 'RB': 2, 'WR': 3, 'FLEX': 7},
			'lineup_size': 9,
			'posfilter': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4, 'FLEX': 8}
		}
	}

	ctx = from_dict(data_class=GASettings, data=data)

	plugins = ('crossover', 'populate', 'select', 'fitness', 'mutate', 'pool', 'pospool')
	dmgrs = {p: driver.DriverManager(namespace=f'pangadfs.{p}', name=f'{p}_default', invoke_on_load=True) for p in plugins}
	names = ['validate_salary', 'validate_duplicates']
	emgrs = {'validate': named.NamedExtensionManager(namespace='pangadfs.validate', names=names, invoke_on_load=True, name_order=True)}

	# set up GeneticAlgorithm object
	opt = Optimizer(ctx=ctx)
	population, fitness = opt.optimize(verbose=True)


if __name__ == '__main__':
	main()
