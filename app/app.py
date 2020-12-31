# pangadfs/app/app.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stevedore import driver, named

from pangadfs import GeneticAlgorithm


def main():
	"""Example application using pangadfs"""
	logging.basicConfig(level=logging.INFO)

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
	dmgrs = {p: driver.DriverManager(namespace=f'pangadfs.{p}', name=f'{p}_default', invoke_on_load=True) for p in plugins}
	names = ['validate_salary', 'validate_duplicates']
	emgrs = {'validate': named.NamedExtensionManager(namespace='pangadfs.validate', names=names, invoke_on_load=True, name_order=True)}

	# set up GeneticAlgorithm object
	ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
	ga.optimize(verbose=True)

	
if __name__ == '__main__':
	main()
