# pangadfs/optimize_multioptimizer_field_ownership.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Any, Dict, List
import numpy as np
from pangadfs.base import OptimizeBase
from pangadfs.fitness_multioptimizer_field_ownership import FitnessMultiOptimizerFieldOwnership


class OptimizeMultiOptimizerFieldOwnership(OptimizeBase):
    """
    Multi-optimizer that balances score, diversity, and field ownership strategy:
    - Score optimization (top lineups + total score)
    - Diversity (lineup differentiation)
    - Field ownership differentiation (contrarian or leverage plays)
    """

    def optimize(self, ga, **kwargs) -> Dict[str, Any]:
        """
        Multi-objective optimization for lineup sets
        
        Args:
            ga: GeneticAlgorithm instance
            **kwargs: Additional parameters
            
        Returns:
            Dict containing optimization results
        """
        # Get multi-objective settings
        target_lineups = ga.ctx['ga_settings'].get('target_lineups', 10)
        top_k = ga.ctx['ga_settings'].get('top_k_lineups', min(10, target_lineups))
        diversity_method = ga.ctx['ga_settings'].get('diversity_method', 'jaccard')
        
        # Multi-objective weights: (score, diversity, ownership)
        score_weight = ga.ctx['ga_settings'].get('score_weight', 0.5)
        diversity_weight = ga.ctx['ga_settings'].get('diversity_weight', 0.3)
        field_ownership_weight = ga.ctx['ga_settings'].get('field_ownership_weight', 0.2)
        weights = (score_weight, diversity_weight, field_ownership_weight)

        field_ownership_strategy = ga.ctx['ga_settings'].get('field_ownership_strategy', 'contrarian')
        ownership_column = ga.ctx['ga_settings'].get('ownership_column', 'ownership')

        # Start profiling
        ga.profiler.start_optimization()
        
        # Create pool and pospool
        pop_size = ga.ctx['ga_settings']['population_size']
        pool = ga.pool(csvpth=ga.ctx['ga_settings']['csvpth'])
        cmap = {'points': ga.ctx['ga_settings']['points_column'],
                'position': ga.ctx['ga_settings']['position_column'],
                'salary': ga.ctx['ga_settings']['salary_column']}
        posfilter = ga.ctx['site_settings']['posfilter']
        flex_positions = ga.ctx['site_settings']['flex_positions']
        pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping=cmap, flex_positions=flex_positions)

        # Create salary, points and ownership arrays
        points = pool[ga.ctx['ga_settings']['points_column']].values
        salaries = pool[ga.ctx['ga_settings']['salary_column']].values
        ownership = pool[ownership_column].values if ownership_column in pool.columns else np.zeros_like(points)

        # Import optional set-based plugins. Fall back to built-in implementations
        # when optimized set plugins are not available in this repository.
        populate_sets = None
        crossover_sets = None
        mutate_sets = None
        try:
            from pangadfs.populate_sets_optimized import PopulateMultilineupSetsOptimized
            from pangadfs.crossover_sets_optimized import CrossoverMultilineupSetsOptimized
            from pangadfs.mutate_sets_optimized import MutateMultilineupSetsOptimized
            populate_sets = PopulateMultilineupSetsOptimized()
            crossover_sets = CrossoverMultilineupSetsOptimized()
            mutate_sets = MutateMultilineupSetsOptimized()
        except ImportError:
            logging.info('Set-optimized plugins not found. Using built-in fallback set operators.')

        fitness_mo = FitnessMultiOptimizerFieldOwnership()
        
        # Create initial population of lineup sets
        logging.info('Creating initial population for multi-objective optimization with field ownership')
        
        if populate_sets is not None:
            initial_population_sets = populate_sets.populate(
                pospool=pospool,
                posmap=ga.ctx['site_settings']['posmap'],
                population_size=pop_size,
                target_lineups=target_lineups,
                probcol='prob',
                lineup_pool_size=ga.ctx['ga_settings'].get('lineup_pool_size', 100000),
                diversity_threshold=ga.ctx['ga_settings'].get('diversity_threshold', 0.3)
            )
        else:
            initial_population_sets = self._fallback_populate_sets(
                ga=ga,
                pospool=pospool,
                pool=pool,
                salaries=salaries,
                population_size=pop_size,
                target_lineups=target_lineups,
            )

        # Validate lineup sets for salary constraints
        logging.info('Validating lineup sets for salary constraints')
        validated_population_sets = self._validate_lineup_sets(
            initial_population_sets, salaries, ga.ctx['site_settings']['salary_cap']
        )

        # Calculate multi-objective fitness
        population_fitness = fitness_mo.fitness(
            population_sets=validated_population_sets,
            points=points,
            ownership=ownership,
            top_k=top_k,
            diversity_method=diversity_method,
            weights=weights,
            strategy=field_ownership_strategy
        )

        # Find best set initially
        best_set_idx = population_fitness.argmax()
        best_fitness = population_fitness[best_set_idx]
        best_set = validated_population_sets[best_set_idx]

        # Mark setup phase complete
        ga.profiler.mark_setup_complete()
        ga.profiler.mark_best_solution(0)

        # Evolution loop
        n_unimproved = 0
        population_sets = validated_population_sets.copy()

        for i in range(1, ga.ctx['ga_settings']['n_generations'] + 1):
            ga.profiler.start_generation(i)

            # Stop if no improvement
            if n_unimproved == ga.ctx['ga_settings']['stop_criteria']:
                break

            if ga.ctx['ga_settings'].get('verbose'):
                logging.info(f'Generation {i} - Best fitness: {best_fitness:.3f}')

            # Elite selection (preserve best sets)
            elite_size = len(population_sets) // ga.ctx['ga_settings'].get('elite_divisor', 5)
            elite_indices = np.argsort(population_fitness)[-elite_size:]
            elite_sets = population_sets[elite_indices]

            # Crossover with tournament selection
            tournament_size = ga.ctx['ga_settings'].get('tournament_size', 3)
            if crossover_sets is not None:
                crossed_over_sets = crossover_sets.crossover(
                    population_sets=population_sets,
                    fitness_scores=population_fitness,
                    tournament_size=tournament_size
                )
            else:
                crossed_over_sets = self._fallback_crossover_sets(
                    population_sets=population_sets,
                    fitness_scores=population_fitness,
                    tournament_size=tournament_size,
                )

            # Mutation
            mutation_rate = ga.ctx['ga_settings'].get('mutation_rate', 0.1)
            if mutate_sets is not None:
                mutated_sets = mutate_sets.mutate(
                    population_sets=crossed_over_sets,
                    mutation_rate=mutation_rate,
                    pospool=pospool,
                    posmap=ga.ctx['site_settings']['posmap']
                )
            else:
                mutated_sets = self._fallback_mutate_sets(
                    ga=ga,
                    population_sets=crossed_over_sets,
                    mutation_rate=mutation_rate,
                    pospool=pospool,
                    posmap=ga.ctx['site_settings']['posmap'],
                    pool=pool,
                    salaries=salaries,
                    salary_cap=ga.ctx['site_settings']['salary_cap'],
                )

            # Combine elite and mutated sets
            if len(elite_sets) > 0:
                n_mutated = len(population_sets) - len(elite_sets)
                combined_sets = np.vstack([elite_sets, mutated_sets[:n_mutated]])
            else:
                combined_sets = mutated_sets

            # Validate new population
            population_sets = self._validate_lineup_sets(
                combined_sets, salaries, ga.ctx['site_settings']['salary_cap']
            )
            
            # Calculate multi-objective fitness
            population_fitness = fitness_mo.fitness(
                population_sets=population_sets,
                points=points,
                ownership=ownership,
                top_k=top_k,
                diversity_method=diversity_method,
                weights=weights,
                strategy=field_ownership_strategy
            )
            
            # Check for improvement
            generation_best_idx = population_fitness.argmax()
            generation_best_fitness = population_fitness[generation_best_idx]
        
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_set = population_sets[generation_best_idx]
                n_unimproved = 0
                ga.profiler.mark_best_solution(i)
            else:
                n_unimproved += 1
            
            ga.profiler.end_generation()

        # End profiling
        ga.profiler.end_optimization()

        # FINALIZE RESULTS
        best_lineups = [pool.loc[lineup, :] for lineup in best_set]
        best_lineup_scores = [points[lineup].sum() for lineup in best_set]
        
        # Sort lineups by score for better presentation
        sorted_indices = np.argsort(best_lineup_scores)[::-1]
        best_lineups = [best_lineups[i] for i in sorted_indices]
        best_lineup_scores = [best_lineup_scores[i] for i in sorted_indices]
        
        diversity_metrics = self._calculate_final_diversity_metrics(best_set, diversity_method)
        
        best_individual_lineup = best_lineups[0]
        best_individual_score = best_lineup_scores[0]

        results = {
            'population': population_sets,
            'fitness': population_fitness,
            'best_lineup': best_individual_lineup,
            'best_score': best_individual_score,
            'lineups': best_lineups,
            'scores': best_lineup_scores,
            'diversity_metrics': diversity_metrics,
        }
        
        if ga.profiler.enabled:
            results['profiling'] = ga.profiler.export_to_dict()
        
        return results

    @staticmethod
    def _fallback_crossover_sets(population_sets: np.ndarray,
                                 fitness_scores: np.ndarray,
                                 tournament_size: int) -> np.ndarray:
        """Fallback set crossover using tournament-selected parent sets."""
        pop_size, target_lineups, _ = population_sets.shape
        tournament_size = max(2, min(int(tournament_size), pop_size))

        children = np.empty_like(population_sets)
        for i in range(pop_size):
            p1 = OptimizeMultiOptimizerFieldOwnership._tournament_pick(fitness_scores, tournament_size)
            p2 = OptimizeMultiOptimizerFieldOwnership._tournament_pick(fitness_scores, tournament_size)
            mask = np.random.randint(0, 2, size=target_lineups).astype(bool)
            children[i] = np.where(mask[:, None], population_sets[p1], population_sets[p2])

        return children

    @staticmethod
    def _tournament_pick(fitness_scores: np.ndarray, tournament_size: int) -> int:
        contenders = np.random.choice(len(fitness_scores), size=tournament_size, replace=False)
        winner_local = np.argmax(fitness_scores[contenders])
        return int(contenders[winner_local])

    def _fallback_populate_sets(self,
                                ga,
                                pospool: Dict[str, np.ndarray],
                                pool,
                                salaries: np.ndarray,
                                population_size: int,
                                target_lineups: int) -> np.ndarray:
        """Fallback set population generator built from core GA operators."""
        sets = []
        posmap = ga.ctx['site_settings']['posmap']
        for _ in range(population_size):
            lineups = ga.populate(
                pospool=pospool,
                posmap=posmap,
                population_size=target_lineups,
            )
            lineups = ga.validate(
                population=lineups,
                salaries=salaries,
                salary_cap=ga.ctx['site_settings']['salary_cap'],
                pool=pool,
                posmap=posmap,
                position_column=ga.ctx['ga_settings']['position_column'],
                flex_positions=ga.ctx['site_settings']['flex_positions'],
            )

            if len(lineups) == 0:
                lineups = ga.populate(pospool=pospool, posmap=posmap, population_size=target_lineups)

            if len(lineups) < target_lineups:
                reps = np.random.choice(len(lineups), size=target_lineups - len(lineups), replace=True)
                lineups = np.vstack((lineups, lineups[reps]))

            sets.append(lineups[:target_lineups])

        return np.array(sets)

    def _fallback_mutate_sets(self,
                              ga,
                              population_sets: np.ndarray,
                              mutation_rate: float,
                              pospool,
                              posmap,
                              pool,
                              salaries: np.ndarray,
                              salary_cap: int) -> np.ndarray:
        """Fallback set mutation by replacing individual lineups probabilistically."""
        mutated = population_sets.copy()
        pop_size, target_lineups, _ = mutated.shape

        for i in range(pop_size):
            for j in range(target_lineups):
                if np.random.random() < mutation_rate:
                    replacement = ga.populate(pospool=pospool, posmap=posmap, population_size=1)
                    replacement = ga.validate(
                        population=replacement,
                        salaries=salaries,
                        salary_cap=salary_cap,
                        pool=pool,
                        posmap=posmap,
                        position_column=ga.ctx['ga_settings']['position_column'],
                        flex_positions=ga.ctx['site_settings']['flex_positions'],
                    )
                    if len(replacement) > 0:
                        mutated[i, j] = replacement[0]

        return mutated

    @staticmethod
    def _validate_lineup_sets(population_sets: np.ndarray, 
                            salaries: np.ndarray, 
                            salary_cap: int) -> np.ndarray:
        """Validate all lineups in all sets for salary constraints"""
        pop_size, target_lineups, lineup_size = population_sets.shape
        valid_sets = []
        
        for set_idx in range(pop_size):
            lineup_set = population_sets[set_idx]
            set_is_valid = True
            
            for lineup in lineup_set:
                lineup_salary = salaries[lineup].sum()
                if lineup_salary > salary_cap:
                    set_is_valid = False
                    break
            
            if set_is_valid:
                valid_sets.append(lineup_set)
        
        if len(valid_sets) == 0:
            logging.warning("No valid lineup sets found! Returning original population.")
            return population_sets
        
        valid_sets_array = np.array(valid_sets)
        return valid_sets_array

    @staticmethod
    def _calculate_final_diversity_metrics(lineup_set: np.ndarray, 
                                         diversity_method: str) -> Dict[str, Any]:
        """Calculate final diversity metrics for the best set"""
        n_lineups = len(lineup_set)
        if n_lineups <= 1:
            return {'avg_overlap': 0.0, 'min_overlap': 0.0, 'diversity_matrix': np.array([[1.0]])}
        
        diversity_matrix = np.zeros((n_lineups, n_lineups))
        overlaps = []
        
        for i in range(n_lineups):
            for j in range(i + 1, n_lineups):
                if diversity_method == 'jaccard':
                    set1, set2 = set(lineup_set[i]), set(lineup_set[j])
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    overlap = intersection / union if union > 0 else 0.0
                else:
                    overlap = np.mean(lineup_set[i] == lineup_set[j])
                
                diversity_matrix[i, j] = overlap
                diversity_matrix[j, i] = overlap
                overlaps.append(overlap)
        
        np.fill_diagonal(diversity_matrix, 1.0)
        
        return {
            'avg_overlap': np.mean(overlaps) if overlaps else 0.0,
            'min_overlap': np.min(overlaps) if overlaps else 0.0,
            'diversity_matrix': diversity_matrix
        }
