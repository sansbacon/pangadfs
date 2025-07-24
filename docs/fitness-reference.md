::: pangadfs.fitness

## FitnessMultiOptimizerFieldOwnership

The `FitnessMultiOptimizerFieldOwnership` class is a fitness calculator for multi-objective optimization with field ownership. It calculates a composite fitness score based on three components:
-   **Score**: The projected score of the lineups.
-   **Diversity**: The uniqueness of the lineups compared to each other.
-   **Field Ownership**: The projected ownership of the players in the lineups.

The fitness function is a weighted sum of these three components. The weights can be configured in the `ga_settings` of the context object.

### `fitness`

::: pangadfs.fitness_multioptimizer_field_ownership.FitnessMultiOptimizerFieldOwnership.fitness
