# pangadfs/app/basicapp/app.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging

from pangadfs.ga import GeneticAlgorithm
from pathlib import Path

import numpy as np
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager


def run():
	"""Example application using pangadfs"""
	logging.basicConfig(level=logging.INFO)

	ctx = {
		'ga_settings': {
			'crossover_method': 'uniform',
			'csvpth': Path(__file__).parent / 'appdata' / 'pool.csv',
			'elite_divisor': 5,
			'elite_method': 'fittest',
			'mutation_rate': .05,
			'n_generations': 20,
			'points_column': 'proj',
			'population_size': 30000,
			'position_column': 'pos',
			'salary_column': 'salary',
			'select_method': 'tournament',
			'stop_criteria': 10,
			'verbose': True,
			'enable_profiling': True
		},

		'site_settings': {
			'flex_positions': ('RB', 'WR', 'TE'),
			'lineup_size': 9,
			'posfilter': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4, 'FLEX': 8},
			'posmap': {'DST': 1, 'QB': 1, 'TE': 1, 'RB': 2, 'WR': 3, 'FLEX': 7},
			'salary_cap': 50000
		}
	}

	# set up driver managers
	dmgrs = {}
	emgrs = {}
	for ns in GeneticAlgorithm.PLUGIN_NAMESPACES:
		pns = f'pangadfs.{ns}'
		if ns == 'validate':
			emgrs['validate'] = NamedExtensionManager(
				namespace=pns, 
				names=['validate_salary', 'validate_duplicates'], 
				invoke_on_load=True, 
				name_order=True)
		else:
			dmgrs[ns] = DriverManager(
				namespace=pns, 
				name=f'{ns}_default', 
				invoke_on_load=True)
	
	
	# set up GeneticAlgorithm object
	ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
	
	# run optimizer
	results = ga.optimize()

	# show best score and lineup at conclusion
	# will break after n_generations or when stop_criteria reached
	print(results['best_lineup'])
	print(f'Lineup score: {results["best_score"]}')
	
	# Display profiling results if enabled
	if ga.profiler.enabled:
		ga.profiler.print_profiling_results()
	

if __name__ == '__main__':
	run()
