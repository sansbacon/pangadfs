# pangadfs/app/app.py

import logging
from pathlib import Path
import pandas as pd

from .settings import dmgrs
from gadfs import GeneticAlgorithm

def main():
	"""Main script"""
	logging.basicConfig(level=logging.INFO)
	df = pd.read_csv(Path(__file__).parent / 'pool.csv')
	ga = GeneticAlgorithm(
		  pool=df,
	      driver_managers=dmgrs,
		  team_size=5
	)

	pop = ga.populate()
	print(pop)
	print(ga.crossover(pop))


if __name__ == '__main__':
	main()
