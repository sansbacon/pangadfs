# Building an app with pangadfs

```
	# set up GeneticAlgorithm object
	ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
	pop_size = ctx['ga_settings']['population_size']

	pool = ga.pool(csvpth=ctx['ga_settings']['csvpth'])
	posfilter = {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6}
	pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping={})

	# create dict of index and stat value
	# this will allow easy lookup later on
	points_mapping = dict(zip(pool.index, pool[ctx['ga_settings']['points_column']]))
	salary_mapping = dict(zip(pool.index, pool[ctx['ga_settings']['salary_column']]))
	
	# CREATE NEW GENERATIONS
	stop_criteria = ctx['ga_settings']['stop_criteria']
	n_unimproved = 0
	for i in range(ctx['ga_settings']['n_generations'] + 1):
		if n_unimproved == stop_criteria:
		    break
		if i == 0:
			# create initial population
			population = ga.populate(
				pospool=pospool, 
				posmap=ctx['site_settings']['posmap'], 
				population_size=pop_size
			)

			# apply validators
			population = ga.validate(
				population=population, 
				salary_mapping=salary_mapping, 
				salary_cap=ctx['site_settings']['salary_cap']
			)

		else:
			print(f'Starting generation {i}')
			population_fitness = ga.fitness(population=population, points_mapping=points_mapping)
			oldmax = population_fitness.max()

			population = ga.crossover(
				population=population, 
				population_fitness=population_fitness
			)

			population = ga.mutate(population=population, mutation_rate=.1)

			population = ga.validate(
				population=population, 
				salary_mapping=salary_mapping, 
				salary_cap=ctx['site_settings']['salary_cap']
			)

			population_fitness = ga.fitness(population=population, points_mapping=points_mapping)
			idx = population_fitness.argmax()
			print(f'Lineup score: {population_fitness[idx]}')
			pop_len = pop_size if len(population) > pop_size else len(population)
			population = population[np.argpartition(population_fitness, -pop_len, axis=0)][-pop_len:]

			if population_fitness.max() > oldmax:
				oldmax = population_fitness.max()
				n_unimproved = 0
			else:
				n_unimproved += 1

	# BEST LINEUP
	print('\n------BEST LINEUP------\n')
	population_fitness = ga.fitness(population=population, points_mapping=points_mapping)
	idx = population_fitness.argmax()
	print(pool.loc[population[idx], :])
	print(f'Lineup score: {population_fitness[idx]}')
```