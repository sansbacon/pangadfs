# pangadfs-captain/app/app.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging
from pathlib import Path
import pandas as pd

from settings import ctx, dmgrs, emgrs
from pangadfs import GeneticAlgorithm


def main():
	"""Main script"""
	logging.basicConfig(level=logging.INFO)
	logging.debug({k: str(v.driver) for k, v in dmgrs.items()})

	# set up GeneticAlgorithm object
	ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)

	pool = ga.pool(csvpth=ctx['ga_settings']['csvpth'])
	posfilter = {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6}
	pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping={})

	# create dict of index and stat value
	# this will allow easy lookup later on
	points_mapping = dict(zip(pool.index, pool[ctx['ga_settings']['points_column']]))
	salary_mapping = dict(zip(pool.index, pool[ctx['ga_settings']['salary_column']]))
	
	# create initial population
	population = ga.populate(
		pospool=pospool, 
	    posmap=ctx['site_settings']['posmap'], 
		population_size=ctx['ga_settings']['population_size']
	)

	# apply validators
	population = ga.validate(
		population=population, 
		salary_mapping=salary_mapping, 
		salary_cap=ctx['site_settings']['salary_cap']
	)
	
	# calculate current maximum
	population_fitness = ga.fitness(population=population, points_mapping=points_mapping)
	oldmax = population_fitness.max()
	print(pool.loc[population[population_fitness.argmax()], :])
	print(oldmax)

	# CREATE NEW GENERATIONS
	for i in range(ctx['ga_settings']['n_generations']):
		try:
			print(f'Starting generation {i}')
			population = ga.crossover(
				population=population, 
				population_fitness=population_fitness
			)
			
			population = ga.validate(
				population=population, 
				salary_mapping=salary_mapping, 
				salary_cap=ctx['site_settings']['salary_cap']
			)
			
			population_fitness = ga.fitness(population=population, points_mapping=points_mapping)
			thismax = population_fitness.max()
			if thismax > oldmax:
				oldmax = thismax
				print(round(thismax, 2))
		except:
			continue

	# BEST LINEUP
	print('\n------BEST LINEUP------\n')
	print(pool.loc[population[population_fitness.argmax()], :])
	print(f'Lineup score: {oldmax}')
	
	
if __name__ == '__main__':
	main()
