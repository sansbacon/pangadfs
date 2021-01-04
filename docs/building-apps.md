# Developing pangadfs plugins

The default plugins for pangadfs will not gracefully handle "Showdown" (a/k/a "Captain Mode") contests on DraftKings. This section documents the development of pangadfs-showdown, a plugin that optimizes Showdown lineups.

## Plugin Namespaces

The first step is to determine which plugin namespaces are required for your plugin. For pangadfs-showdown, 


def _showdown_sum(x: np.ndarray, mapping: Dict[int, Number]):
class ShowdownPospool(PospoolBase):
class ShowdownPopulate(PopulateDefault):
class ShowdownFitness(FitnessBase):
class ShowdownSalaryValidate(ValidateBase):
