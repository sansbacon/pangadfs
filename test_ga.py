from pangadfs.ga import GeneticAlgorithm

# Create a GeneticAlgorithm instance with default plugins
ga = GeneticAlgorithm(use_defaults=True)

# Print the available plugins
for ns in ga.PLUGIN_NAMESPACES:
    try:
        plugin = ga.get_plugin(ns)
        print(f"Plugin for namespace '{ns}': {plugin}")
    except ValueError as e:
        print(f"Error getting plugin for namespace '{ns}': {str(e)}")

print("\nAvailable plugins for 'validate' namespace:")
try:
    plugins = ga.get_plugins('validate')
    for i, plugin in enumerate(plugins):
        print(f"  {i+1}. {plugin}")
except ValueError as e:
    print(f"Error getting plugins for namespace 'validate': {str(e)}")
