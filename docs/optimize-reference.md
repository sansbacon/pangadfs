::: pangadfs.optimize

## OptimizeMultiOptimizerFieldOwnership

The `OptimizeMultiOptimizerFieldOwnership` class is an advanced multi-optimizer that balances three key objectives:
-   **Score optimization**: Prioritizes both the best-performing lineups and the overall quality of the entire lineup set.
-   **Diversity**: Ensures that the generated lineups are different from each other, reducing overlap and increasing the chances of a unique lineup winning.
-   **Field ownership differentiation**: Provides a strategic edge by considering the projected ownership of players in the field. This can be used to create contrarian lineups that avoid popular players or to leverage low-owned players with high upside.

### Configuration

The behavior of the optimizer can be controlled through the `ga_settings` in the context object.

| Setting                      | Description                                                                                                | Default        |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------- |
| `target_lineups`             | The number of lineups to generate.                                                                         | `10`           |
| `score_weight`               | The weight to give to the score component of the fitness function.                                         | `0.5`          |
| `diversity_weight`           | The weight to give to the diversity component of the fitness function.                                     | `0.3`          |
| `field_ownership_weight`     | The weight to give to the field ownership component of the fitness function.                               | `0.2`          |
| `field_ownership_strategy`   | The strategy to use for field ownership differentiation. Can be `contrarian`, `leverage`, or `balanced`.     | `'contrarian'` |
| `ownership_column`           | The name of the column in the player pool CSV that contains the ownership data.                            | `'ownership'`  |
| `top_k_focus`                | The number of top lineups to prioritize in the score component.                                            | `min(10, target_lineups)` |

### Example

```python
ctx = {
    'ga_settings': {
        'target_lineups': 20,
        'score_weight': 0.5,
        'diversity_weight': 0.3,
        'field_ownership_weight': 0.2,
        'field_ownership_strategy': 'contrarian',
        'ownership_column': 'ownership',
        'top_k_focus': 5,
        # ... other GA settings
    }
}
