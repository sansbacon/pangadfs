![pangadfs](docs/img/pangadfs.png)

pangadfs is a pandas-based (python) genetic algorithm framework for fantasy sports. It uses a plugin architecture to enable maximum flexibility while also providing a fully-functional implementation of a genetic algorithm for lineup optimization.

---

**Documentation**: <a href="https://sansbacon.github.io/pangadfs/">https://sansbacon.github.io/pangadfs/</a>

**Source Code**: <a href="https://github.com/sansbacon/pangadfs" target="_blank">https://github.com/sansbacon/pangadfs</a>

---

The key pangadfs features are as follows:

* **Fast**: takes advantage of pandas and numpy to generate thousands of lineups quickly.
* **Extensible**: any desired functionality can be added with a straightforward plugin architecture.
* **Pythonic**: library is easy to use and extend as long as you are familiar with data analysis in python (pandas and numpy). You don't also have to be an expert in linear programming.
* **Fewer bugs**: Small core means fewer bugs and easier to trace code. Unlike other optimizers, pangadfs does not generate complicated equations behind the curtain that are difficult to comprehend and debug.


## Requirements [TODO: update reqirements]

* Python 3.8+
* pandas 1.0+
* numpy 1.19+
* stevedore 3.30+
* numpy-indexed 0.3+


## Installation

<div class="termy">

```console
$ pip install pangadfs

```

</div>

## GUI Application

For users who prefer a graphical interface, a separate GUI application is available:

**PangaDFS GUI**: <a href="https://github.com/sansbacon/pangadfs-gui" target="_blank">https://github.com/sansbacon/pangadfs-gui</a>

The GUI provides an intuitive interface for lineup optimization with features like:
- Visual configuration management
- Real-time optimization monitoring  
- Comprehensive results analysis
- Multiple export formats

Install with: `pip install pangadfs-gui`

## Examples

### Single Lineup Optimization

A simple pangadfs optimizer app for single lineup optimization:

```Python
# pangadfs/app/app.py

from pangadfs.ga import GeneticAlgorithm
from pathlib import Path

import numpy as np
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager


def run():
	"""Example optimizer application using pangadfs"""
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
			'select_method': 'roulette',
			'stop_criteria': 10,
			'verbose': True
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
	

if __name__ == '__main__':
	run()

```

### Multiple Lineup Optimization

For generating multiple diverse lineups, use the `OptimizeMultilineup` approach:

```Python
from pangadfs.ga import GeneticAlgorithm
from pangadfs.optimize import OptimizeMultilineup
from pathlib import Path

def run_multilineup():
	"""Example multilineup optimizer using OptimizeMultilineup"""
	ctx = {
		'ga_settings': {
			'crossover_method': 'uniform',
			'csvpth': Path(__file__).parent / 'appdata' / 'pool.csv',
			'elite_divisor': 5,
			'elite_method': 'fittest',
			'mutation_rate': .1,
			'n_generations': 150,
			'points_column': 'proj',
			'population_size': 1000,
			'position_column': 'pos',
			'salary_column': 'salary',
			'select_method': 'roulette',
			'stop_criteria': 25,
			'verbose': True,
			# Multilineup-specific settings
			'target_lineups': 100,
			'diversity_weight': 0.25,
			'min_overlap_threshold': 0.4,
			'diversity_method': 'jaccard'
		},

		'site_settings': {
			'flex_positions': ('RB', 'WR', 'TE'),
			'lineup_size': 9,
			'posfilter': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4, 'FLEX': 8},
			'posmap': {'DST': 1, 'QB': 1, 'TE': 1, 'RB': 2, 'WR': 3, 'FLEX': 7},
			'salary_cap': 50000
		}
	}

	# Use OptimizeMultilineup for multiple diverse lineups
	optimizer = OptimizeMultilineup()
	ga = GeneticAlgorithm(ctx=ctx, optimize=optimizer)
	
	# run optimizer
	results = ga.optimize()

	# show results
	print(f'Generated {len(results["lineups"])} diverse lineups')
	print(f'Best individual score: {results["best_score"]}')
	print(f'Average diversity overlap: {results["diversity_metrics"]["avg_overlap"]:.3f}')
	
	# Display top 5 lineups
	for i, (lineup, score) in enumerate(zip(results['lineups'][:5], results['scores'][:5])):
		print(f'\nLineup {i+1} (Score: {score:.1f}):')
		print(lineup[['player', 'pos', 'salary', 'proj']])

if __name__ == '__main__':
	run_multilineup()

```

### Run it

Run the sample application with:

<div class="termy">

```console
$ python {pangadfs_directory}/app/app.py

INFO:root:Starting generation 1
INFO:root:Best lineup score 153.00000000000003
INFO:root:Lineup unimproved 1 times
INFO:root:Starting generation 2
INFO:root:Best lineup score 153.00000000000003
INFO:root:Lineup improved to 155.2
. . . 
INFO:root:Starting generation 19
INFO:root:Best lineup score 156.3
INFO:root:Lineup improved to 156.5
INFO:root:Starting generation 20
INFO:root:Best lineup score 156.5
INFO:root:Lineup unimproved 1 times

               player team  pos  salary  proj
0             Saints    NO  DST    3800   9.8
34    Patrick Mahomes   KC   QB    8000  26.6
62        Dalvin Cook  MIN   RB    9500  27.2
68       Nyheim Hines  IND   RB    4600  15.9
72         Brian Hill  ATL   RB    4000  12.8
109     Gabriel Davis  BUF   WR    3000  10.7
136   Keelan Cole Sr.  JAX   WR    3600  11.9
138     Calvin Ridley  ATL   WR    7100  21.6
142  Justin Jefferson  MIN   WR    6300  20.0
Lineup score: 156.5
```
</div>

## Extensibility

pangadfs is extensible by design and is motivated by difficulties I encountered with other optimizers, which tend to have a monolithic design and don't make it easy to swap out components. 

This flexibility is made possible by the [stevedore plugin system](https://docs.openstack.org/stevedore/latest/ "Stevedore plugins"), which allows allow applications to customize one or more of the internal components. 

As recommended by [the stevedore documentation](https://docs.openstack.org/stevedore/latest/user/tutorial/creating_plugins.html#a-plugin-base-class "Stevedore documentation"), the [base module](base-reference.md) includes base classes to define each pluggable component. Each namespace has a default implementation (crossover, fitness, mutate, select, and so forth), which, collectively, provide a fully-functional implementation of a genetic algorithm.

## License

This project is licensed under the terms of the MIT license.
