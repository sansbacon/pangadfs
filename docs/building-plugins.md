# Developing pangadfs plugins

The best way to understand how to create a pangadfs plugin is to look at the structure and code of the existing [pangadfs-showdown](https://github.com/sansbacon/pangadfs-showdown) plugin. The default plugins for pangadfs will not gracefully handle "Showdown" (a/k/a "Captain Mode") contests on DraftKings. This plugin extends pangadfs to optimizes Showdown lineups.

## Plugin Configuration via Context

Plugins can now access configuration parameters via the context (`ctx`) object passed during initialization. This allows for more flexible and configurable plugins without having to modify the plugin code or pass parameters to each method call.

### How to Use Context in Plugins

When creating a plugin, you can access the context in the `__init__` method:

```python
def __init__(self, ctx=None):
    super().__init__(ctx)
    # Access configuration parameters from context
    self.my_param = 0.5  # Default value
    if self.ctx and 'my_param' in self.ctx:
        self.my_param = self.ctx['my_param']
```

You can then use these parameters in your plugin methods:

```python
def my_method(self, *args, **kwargs):
    # Use the parameter from context
    result = some_calculation(self.my_param)
    return result
```

### Example: Configuring Mutation Rate

The `MutateDefault` plugin uses context to configure the default mutation rate:

```python
def __init__(self, ctx=None):
    super().__init__(ctx)
    # Get default mutation rate from context if provided
    self.default_mutation_rate = .05
    if self.ctx and 'mutation_rate' in self.ctx:
        self.default_mutation_rate = self.ctx['mutation_rate']
```

This allows you to set a different mutation rate for the plugin without having to pass it to each call to the `mutate` method.

### Passing Context to Plugins

When creating a `GeneticAlgorithm` instance, you can pass a context dictionary that will be passed to all plugins:

```python
ctx = {
    'mutation_rate': 0.1,
    'fitness_scale': 1.2,
    # Other configuration parameters
}

ga = GeneticAlgorithm(ctx=ctx, use_defaults=True)
```

This context will be passed to all plugins when they are loaded, allowing them to access their specific configuration parameters.

## Plugin Namespaces

The first step is to determine which plugin namespaces are required for your plugin. pangadfs-showdown provides classes for the following namespaces:

* pangadfs.pospool: ShowdownPospool

The default pospool assumes a classic contest, where players are split into position groups along with a flex. The simplest way to handle showdown contests is to assume every player is a FLEX and then handle the 1.5x multiplier later.

* pangadfs.populate: ShowdownPopulate

Populate works differently because there is no need for a separate FLEX draw. Instead, we just need to a random, unique draw of players to fill out a lineup.

* pangadfs.fitness: ShowdownFitness

This handles the 1.5x multiplier for the first player (at index 0) when summing lineup points.

* pangadfs.validate: ShowdownSalaryValidate

This handles the 1.5x multiplier for the first player (at index 0) when summing lineup salary.

## Loading Plugins

It is advisable to provide an app that shows the proper configuration for your plugin. For [pangadfs-showdown](https://github.com/sansbacon/pangadfs-showdown), the configuration is as follows:

```
    # setup context dict for configuration
	ctx = {
		'ga_settings': {
			'crossover_method': 'uniform',
			'csvpth': Path(__file__).parent / 'pool.csv',
			'elite_divisor': 10,
			'elite_method': 'fittest',
			'mutation_rate': .10,
			'n_generations': 20,
			'points_column': 'proj',
			'population_size': 5000,
			'position_column': 'pos',
			'salary_column': 'salary',
			'select_method': 'diverse',
			'stop_criteria': 10,
			'verbose': True
		},

		'site_settings': {
			'lineup_size': 6,
			'posfilter': 2.0,
			'salary_cap': 50000
		}
	}

	# setup plugins
	plugin_names = {
  	  'pool': 'pool_default',
	  'pospool': 'pospool_showdown',
	  'populate': 'populate_showdown',
	  'optimize': 'optimize_default',
      'crossover': 'crossover_default',
	  'mutate': 'mutate_default',
	  'fitness': 'fitness_showdown',
	  'select': 'select_default'
	}

	dmgrs = {ns: DriverManager(namespace=f'pangadfs.{ns}', name=pn, invoke_on_load=True)
	         for ns, pn in plugin_names.items()}

	# when have multiple validators, need to use NamedExtensionManager so they all run
    validate_names = ['salary_validate_showdown', 'validate_duplicates']
	emgrs = {'validate': NamedExtensionManager('pangadfs.validate', names=validate_names, invoke_on_load=True)}

    ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
```
