# pangadfs/optimize_multi_objective.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Any, Dict, List
import numpy as np
from pangadfs.base import OptimizeBase
from pangadfs.fitness_multi_objective import FitnessMultiObjectiveOptimized


class OptimizeMultiObjective(OptimizeBase):
    """Multi-objective optimizer for lineup sets
    
    Optimizes three objectives simultaneously:
    1. Sum of top K lineups (ensures best lineups are truly optimal)
    2. Sum of all lineups (maintains overall set quality)
    3. Diversity across lineups (provides controlled variation)
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
        
        # Multi-objective weights: (top_k, total, diversity)
        mo_weights = ga.ctx['ga_settings'].get('multi_objective_weights', (0.6, 0.3, 0.1))
        
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

        # Create salary and points arrays
        cmap = {'points': ga.ctx['ga_settings']['points_column'],
                'salary': ga.ctx['ga_settings']['salary_column']}
        points = pool[cmap['points']].values
        salaries = pool[cmap['salary']].values
        
        # Import set-based plugins
        from pangadfs.populate_sets_optimized import PopulateMultilineupSetsOptimized
        from pangadfs.crossover_sets_optimized import CrossoverMultilineupSetsOptimized
        from pangadfs.mutate_sets_optimized import MutateMultilineupSetsOptimized
        
        # Initialize plugins
        populate_sets = PopulateMultilineupSetsOptimized()
        fitness_mo = FitnessMultiObjectiveOptimized()
        crossover_sets = CrossoverMultilineupSetsOptimized()
        mutate_sets = MutateMultilineupSetsOptimized()
        
        # Create initial population of lineup sets
        logging.info(f'Creating initial population for multi-objective optimization')
        logging.info(f'Target: {target_lineups} lineups, Top-K: {top_k}, Weights: {mo_weights}')
        
        initial_population_sets = populate_sets.populate(
            pospool=pospool,
            posmap=ga.ctx['site_settings']['posmap'],
            population_size=pop_size,
            target_lineups=target_lineups,
            probcol='prob',
            lineup_pool_size=ga.ctx['ga_settings'].get('lineup_pool_size', 100000),
            diversity_threshold=ga.ctx['ga_settings'].get('diversity_threshold', 0.3)
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
            top_k=top_k,
            diversity_method=diversity_method,
            weights=mo_weights
        )

        # Find best set initially
        best_set_idx = population_fitness.argmax()
        best_fitness = population_fitness[best_set_idx]
        best_set = validated_population_sets[best_set_idx]

        # Track detailed metrics for the best set
        best_metrics = fitness_mo.get_detailed_metrics(
            np.array([best_set]), points, top_k, diversity_method
        )
        
        logging.info(f'Initial best set - Top-{top_k}: {best_metrics["top_k_scores"][0]:.1f}, '
                    f'Total: {best_metrics["total_scores"][0]:.1f}, '
                    f'Diversity: {best_metrics["diversity_scores"][0]:.3f}')

        # Mark setup phase complete
        ga.profiler.mark_setup_complete()
        ga.profiler.mark_best_solution(0)

        # Evolution loop
        n_unimproved = 0
        population_sets = validated_population_sets.copy()
        
        # Track best metrics over generations
        generation_metrics = []

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
            crossed_over_sets = crossover_sets.crossover(
                population_sets=population_sets,
                fitness_scores=population_fitness,
                tournament_size=tournament_size
            )

            # Mutation
            mutation_rate = ga.ctx['ga_settings'].get('mutation_rate', 0.1)
            mutated_sets = mutate_sets.mutate(
                population_sets=crossed_over_sets,
                mutation_rate=mutation_rate,
                pospool=pospool,
                posmap=ga.ctx['site_settings']['posmap']
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
                top_k=top_k,
                diversity_method=diversity_method,
                weights=mo_weights
            )
            
            # Check for improvement
            generation_best_idx = population_fitness.argmax()
            generation_best_fitness = population_fitness[generation_best_idx]
        
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_set = population_sets[generation_best_idx]
                n_unimproved = 0
                ga.profiler.mark_best_solution(i)
                
                # Log detailed improvement
                improved_metrics = fitness_mo.get_detailed_metrics(
                    np.array([best_set]), points, top_k, diversity_method
                )
                logging.info(f'Improved! Top-{top_k}: {improved_metrics["top_k_scores"][0]:.1f}, '
                           f'Total: {improved_metrics["total_scores"][0]:.1f}, '
                           f'Diversity: {improved_metrics["diversity_scores"][0]:.3f}')
            else:
                n_unimproved += 1
                if ga.ctx['ga_settings'].get('verbose'):
                    logging.info(f'No improvement for {n_unimproved} generations')
            
            # Track metrics for analysis
            gen_metrics = fitness_mo.get_detailed_metrics(
                np.array([best_set]), points, top_k, diversity_method
            )
            generation_metrics.append({
                'generation': i,
                'fitness': best_fitness,
                'top_k_score': gen_metrics["top_k_scores"][0],
                'total_score': gen_metrics["total_scores"][0],
                'diversity_score': gen_metrics["diversity_scores"][0]
            })
            
            ga.profiler.end_generation()

        # End profiling
        ga.profiler.end_optimization()

        # FINALIZE RESULTS
        # Extract best set and convert to expected format
        best_lineups = [pool.loc[lineup, :] for lineup in best_set]
        best_lineup_scores = [points[lineup].sum() for lineup in best_set]
        
        # Calculate final detailed metrics
        final_metrics = fitness_mo.get_detailed_metrics(
            np.array([best_set]), points, top_k, diversity_method
        )
        
        # Sort lineups by score for better presentation
        sorted_indices = np.argsort(best_lineup_scores)[::-1]
        best_lineups = [best_lineups[i] for i in sorted_indices]
        best_lineup_scores = [best_lineup_scores[i] for i in sorted_indices]
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_final_diversity_metrics(best_set, diversity_method)
        
        # For backward compatibility
        best_individual_lineup = best_lineups[0]  # Highest scoring lineup
        best_individual_score = best_lineup_scores[0]

        results = {
            'population': population_sets,
            'fitness': population_fitness,
            'best_lineup': best_individual_lineup,
            'best_score': best_individual_score,
            'lineups': best_lineups,
            'scores': best_lineup_scores,
            'diversity_metrics': diversity_metrics,
            'multi_objective_metrics': {
                'top_k_score': final_metrics["top_k_scores"][0],
                'total_score': final_metrics["total_scores"][0],
                'diversity_score': final_metrics["diversity_scores"][0],
                'weights_used': mo_weights,
                'top_k_used': top_k,
                'generation_history': generation_metrics
            }
        }
        
        # Add profiling data
        if ga.profiler.enabled:
            results['profiling'] = ga.profiler.export_to_dict()
        
        # Log final results
        logging.info(f'Multi-objective optimization complete!')
        logging.info(f'Generated {len(best_lineups)} lineups')
        logging.info(f'Best individual lineup: {best_individual_score:.1f} points')
        logging.info(f'Top-{top_k} total: {final_metrics["top_k_scores"][0]:.1f} points')
        logging.info(f'All lineups total: {final_metrics["total_scores"][0]:.1f} points')
        logging.info(f'Average diversity: {diversity_metrics["avg_overlap"]:.3f}')
        
        return results

    def _validate_lineup_sets(self, 
                            population_sets: np.ndarray, 
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
        logging.info(f'Validation: {len(valid_sets)}/{pop_size} sets passed salary validation')
        
        return valid_sets_array

    def _calculate_final_diversity_metrics(self, 
                                         lineup_set: np.ndarray, 
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
