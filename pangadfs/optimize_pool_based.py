# pangadfs/optimize_pool_based.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Any, Dict, List
import numpy as np
from pangadfs.base import OptimizeBase
from pangadfs.ga import GeneticAlgorithm
from pangadfs.populate_pool_based import PopulatePoolBasedSets
from pangadfs.fitness_sets import FitnessMultilineupSets
from pangadfs.crossover_sets import CrossoverMultilineupSets
from pangadfs.mutate_sets import MutateMultilineupSets


class OptimizePoolBasedSets(OptimizeBase):
    """
    Pool-based multilineup optimization with adaptive sampling.
    
    Algorithm:
    1. Generate 100K lineups, stratify into elite (10K) and general (90K) pools
    2. Sample diverse sets from pools using adaptive elite/general ratios
    3. Evolve sets using crossover and mutation (no individual lineup generation)
    4. Adapt sampling ratios based on performance over generations
    """

    def optimize(self, ga: GeneticAlgorithm, **kwargs) -> Dict[str, Any]:
        """
        Optimizes for multiple diverse lineups using pool-based approach
        
        Args:
            ga (GeneticAlgorithm): the ga instance
            **kwargs: keyword arguments for plugins
            
        Returns:
            Dict containing:
            'population': np.ndarray,
            'fitness': np.ndarray,
            'best_lineup': pd.DataFrame,  # For backward compatibility
            'best_score': float,          # For backward compatibility
            'lineups': List[pd.DataFrame], # Multiple lineups
            'scores': List[float],        # Corresponding scores
            'diversity_metrics': Dict     # Diversity statistics
        """
        # Get pool-based settings with defaults
        target_lineups = ga.ctx['ga_settings'].get('target_lineups', 1)
        diversity_weight = ga.ctx['ga_settings'].get('diversity_weight', 0.3)
        diversity_method = ga.ctx['ga_settings'].get('diversity_method', 'jaccard')
        tournament_size = ga.ctx['ga_settings'].get('tournament_size', 3)
        
        # Pool-based specific settings
        initial_pool_size = ga.ctx['ga_settings'].get('initial_pool_size', 100000)
        elite_pool_size = ga.ctx['ga_settings'].get('elite_pool_size', 10000)
        initial_elite_ratio = ga.ctx['ga_settings'].get('initial_elite_ratio', 0.7)
        adaptive_ratio_step = ga.ctx['ga_settings'].get('adaptive_ratio_step', 0.1)
        max_elite_ratio = ga.ctx['ga_settings'].get('max_elite_ratio', 0.9)
        min_elite_ratio = ga.ctx['ga_settings'].get('min_elite_ratio', 0.5)
        pool_injection_rate = ga.ctx['ga_settings'].get('pool_injection_rate', 0.1)
        memory_optimize = ga.ctx['ga_settings'].get('memory_optimize', True)
        
        # Dynamic pool optimization settings
        enable_pool_evolution = ga.ctx['ga_settings'].get('enable_pool_evolution', False)
        pool_refresh_interval = ga.ctx['ga_settings'].get('pool_refresh_interval', 10)
        pool_evolution_rate = ga.ctx['ga_settings'].get('pool_evolution_rate', 0.1)
        
        # Start profiling
        ga.profiler.start_optimization()
        
        # Create pool and pospool (same as other optimizers)
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
        
        # Initialize pool-based plugins
        populate_pool_based = PopulatePoolBasedSets()
        fitness_sets = FitnessMultilineupSets()
        crossover_sets = CrossoverMultilineupSets()
        mutate_sets = MutateMultilineupSets()
        
        # Create initial population using pool-based approach
        logging.info(f'POOL-BASED OPTIMIZATION: Creating {pop_size} sets of {target_lineups} lineups each')
        
        initial_population_sets = populate_pool_based.populate(
            pospool=pospool,
            posmap=ga.ctx['site_settings']['posmap'],
            population_size=pop_size,
            target_lineups=target_lineups,
            probcol='prob',
            initial_pool_size=initial_pool_size,
            elite_pool_size=elite_pool_size,
            initial_elite_ratio=initial_elite_ratio,
            memory_optimize=memory_optimize,
            points=points,
            salaries=salaries,
            salary_cap=ga.ctx['site_settings']['salary_cap'],
            diversity_threshold=ga.ctx['ga_settings'].get('diversity_threshold', 0.3)
        )

        # Store pools for potential injection during mutation
        self.elite_pool = getattr(populate_pool_based, 'elite_pool', None)
        self.general_pool = getattr(populate_pool_based, 'general_pool', None)
        self.elite_fitness = getattr(populate_pool_based, 'elite_fitness', None)
        self.general_fitness = getattr(populate_pool_based, 'general_fitness', None)

        # Calculate fitness for each set
        population_fitness = fitness_sets.fitness(
            population_sets=initial_population_sets,
            points=points,
            diversity_weight=diversity_weight,
            diversity_method=diversity_method
        )

        # Find best set initially
        best_set_idx = population_fitness.argmax()
        best_fitness = population_fitness[best_set_idx]
        best_set = initial_population_sets[best_set_idx]

        # Mark setup phase complete
        ga.profiler.mark_setup_complete()
        
        # Mark initial best solution (generation 0)
        ga.profiler.mark_best_solution(0)

        # Initialize adaptive sampling tracking
        current_elite_ratio = initial_elite_ratio
        performance_history = []
        
        # Create new generations using pool-based GA
        n_unimproved = 0
        population_sets = initial_population_sets.copy()

        for i in range(1, ga.ctx['ga_settings']['n_generations'] + 1):
            # Start generation timing
            ga.profiler.start_generation(i)

            # End program after n generations if not improving
            if n_unimproved == ga.ctx['ga_settings']['stop_criteria']:
                break

            # Display progress information with verbose parameter
            if ga.ctx['ga_settings'].get('verbose'):
                logging.info(f'Starting generation {i}')
                logging.info(f'Best set fitness {best_fitness:.2f}, elite ratio {current_elite_ratio:.2f}')

            # Select elite sets (top performers)
            elite_size = len(population_sets) // ga.ctx['ga_settings'].get('elite_divisor', 5)
            elite_indices = np.argsort(population_fitness)[-elite_size:]
            elite_sets = population_sets[elite_indices]

            # Crossover to create new sets
            crossed_over_sets = crossover_sets.crossover(
                population_sets=population_sets,
                tournament_size=tournament_size
            )

            # Enhanced mutation with pool injection
            mutation_rate = ga.ctx['ga_settings'].get('mutation_rate', 0.1)
            mutated_sets = self._enhanced_mutate_with_pool_injection(
                crossed_over_sets, mutation_rate, pool_injection_rate,
                pospool, ga.ctx['site_settings']['posmap']
            )

            # Combine elite and mutated sets
            if len(elite_sets) > 0:
                # Ensure we don't exceed population size
                n_mutated = len(population_sets) - len(elite_sets)
                combined_sets = np.vstack([elite_sets, mutated_sets[:n_mutated]])
            else:
                combined_sets = mutated_sets

            # Validate the new population (basic check)
            population_sets = self._validate_population_sets(
                combined_sets, salaries, ga.ctx['site_settings']['salary_cap']
            )
            
            # Calculate fitness for new population
            population_fitness = fitness_sets.fitness(
                population_sets=population_sets,
                points=points,
                diversity_weight=diversity_weight,
                diversity_method=diversity_method
            )
            
            # Check for improvement
            generation_best_idx = population_fitness.argmax()
            generation_best_fitness = population_fitness[generation_best_idx]
        
            if generation_best_fitness > best_fitness:
                logging.info(f'Set fitness improved to {generation_best_fitness:.2f}')
                best_fitness = generation_best_fitness
                best_set = population_sets[generation_best_idx]
                n_unimproved = 0
                # Mark when best solution was found
                ga.profiler.mark_best_solution(i)
            else:
                n_unimproved += 1
                logging.info(f'Set fitness unimproved {n_unimproved} times')
            
            # Adaptive elite ratio adjustment
            current_elite_ratio = self._adapt_elite_ratio(
                current_elite_ratio, generation_best_fitness, performance_history,
                adaptive_ratio_step, max_elite_ratio, min_elite_ratio
            )
            
            # Dynamic pool evolution (if enabled)
            if (enable_pool_evolution and 
                i % pool_refresh_interval == 0 and 
                self.elite_pool is not None):
                
                logging.info(f'Generation {i}: Evolving elite pool...')
                self._evolve_elite_pool(
                    population_sets, points, pool_evolution_rate, 
                    pospool, ga.ctx['site_settings']['posmap']
                )
            
            # End generation timing
            ga.profiler.end_generation()

        # End profiling
        ga.profiler.end_optimization()

        # FINALIZE RESULTS
        # Extract the best set and convert to expected format
        best_lineups = [pool.loc[lineup, :] for lineup in best_set]
        best_lineup_scores = [points[lineup].sum() for lineup in best_set]
        
        # Calculate diversity metrics for the best set
        diversity_metrics = self._calculate_set_diversity_metrics(best_set, diversity_method)
        
        # For backward compatibility, use the highest-scoring individual lineup
        best_individual_lineup_idx = np.argmax(best_lineup_scores)
        best_individual_lineup = best_lineups[best_individual_lineup_idx]
        best_individual_score = best_lineup_scores[best_individual_lineup_idx]

        results = {
            'population': population_sets,
            'fitness': population_fitness,
            'best_lineup': best_individual_lineup,  # Backward compatibility
            'best_score': best_individual_score,    # Backward compatibility
            'lineups': best_lineups,
            'scores': best_lineup_scores,
            'diversity_metrics': diversity_metrics,
            'final_elite_ratio': current_elite_ratio,  # Pool-based specific
            'pool_stats': {
                'elite_pool_size': len(self.elite_pool) if self.elite_pool is not None else 0,
                'general_pool_size': len(self.general_pool) if self.general_pool is not None else 0
            }
        }
        
        # Add profiling data to results
        if ga.profiler.enabled:
            results['profiling'] = ga.profiler.export_to_dict()
        
        logging.info(f'Pool-based optimization complete. Generated {len(best_lineups)} diverse lineups.')
        logging.info(f'Best set total fitness: {best_fitness:.2f}')
        logging.info(f'Average diversity overlap: {diversity_metrics["avg_overlap"]:.3f}')
        logging.info(f'Final elite sampling ratio: {current_elite_ratio:.2f}')
        
        return results

    def _enhanced_mutate_with_pool_injection(self, 
                                           population_sets: np.ndarray,
                                           mutation_rate: float,
                                           pool_injection_rate: float,
                                           pospool: Dict,
                                           posmap: Dict) -> np.ndarray:
        """
        Enhanced mutation that includes pool injection
        
        Args:
            population_sets: Population to mutate
            mutation_rate: Standard mutation rate
            pool_injection_rate: Rate of pool injection mutations
            pospool: Position pool for standard mutations
            posmap: Position mapping
            
        Returns:
            Mutated population
        """
        # First apply standard mutation
        mutate_sets = MutateMultilineupSets()
        mutated_population = mutate_sets.mutate(
            population_sets=population_sets,
            mutation_rate=mutation_rate,
            pospool=pospool,
            posmap=posmap
        )
        
        # Then apply pool injection mutations
        if (pool_injection_rate > 0 and 
            self.elite_pool is not None and 
            self.general_pool is not None):
            
            mutated_population = self._apply_pool_injection(
                mutated_population, pool_injection_rate
            )
        
        return mutated_population
    
    def _apply_pool_injection(self, 
                            population_sets: np.ndarray,
                            injection_rate: float) -> np.ndarray:
        """
        Apply pool injection mutations by replacing poor lineups with pool samples
        
        Args:
            population_sets: Population to inject into
            injection_rate: Rate of injection mutations
            
        Returns:
            Population with pool injections
        """
        pop_size, target_lineups, lineup_size = population_sets.shape
        injected_population = population_sets.copy()
        
        # Combine pools for sampling
        all_pool_lineups = np.vstack([self.elite_pool, self.general_pool])
        all_pool_fitness = np.concatenate([self.elite_fitness, self.general_fitness])
        
        # Create fitness-based probabilities
        min_fitness = all_pool_fitness.min()
        normalized_fitness = all_pool_fitness - min_fitness + 1e-6
        probabilities = normalized_fitness / normalized_fitness.sum()
        
        for set_idx in range(pop_size):
            if np.random.random() < injection_rate:
                # Randomly select 1-2 lineups to replace in this set
                n_replacements = np.random.randint(1, min(3, target_lineups + 1))
                replacement_indices = np.random.choice(
                    target_lineups, n_replacements, replace=False
                )
                
                for replace_idx in replacement_indices:
                    # Sample a high-quality lineup from pools
                    sampled_idx = np.random.choice(
                        len(all_pool_lineups), p=probabilities
                    )
                    injected_population[set_idx, replace_idx] = all_pool_lineups[sampled_idx]
        
        return injected_population
    
    @staticmethod
    def _adapt_elite_ratio(current_ratio: float,
                         current_fitness: float,
                         performance_history: List[float],
                         step_size: float,
                         max_ratio: float,
                         min_ratio: float) -> float:
        """
        Adapt elite sampling ratio based on performance trends
        
        Args:
            current_ratio: Current elite ratio
            current_fitness: Current generation's best fitness
            performance_history: History of fitness improvements
            step_size: Step size for ratio adjustments
            max_ratio: Maximum allowed elite ratio
            min_ratio: Minimum allowed elite ratio
            
        Returns:
            Adjusted elite ratio
        """
        performance_history.append(current_fitness)
        
        # Only start adapting after we have some history
        if len(performance_history) < 3:
            return current_ratio
        
        # Calculate recent performance trend
        recent_trend = np.mean(performance_history[-3:]) - np.mean(performance_history[-6:-3]) if len(performance_history) >= 6 else 0
        
        # Adapt based on trend
        if recent_trend > 0:
            # Performance improving - current strategy is working
            new_ratio = current_ratio
        elif recent_trend < -0.1:
            # Performance declining - try different strategy
            if current_ratio > 0.7:
                # If elite-heavy, try more diversity
                new_ratio = max(current_ratio - step_size, min_ratio)
            else:
                # If diversity-heavy, try more elite focus
                new_ratio = min(current_ratio + step_size, max_ratio)
        else:
            # Stagnant - small random adjustment
            adjustment = np.random.choice([-step_size/2, 0, step_size/2])
            new_ratio = np.clip(current_ratio + adjustment, min_ratio, max_ratio)
        
        return new_ratio
    
    @staticmethod
    def _validate_population_sets(population_sets: np.ndarray,
                                salaries: np.ndarray,
                                salary_cap: int) -> np.ndarray:
        """
        Basic validation for population sets
        
        Args:
            population_sets: Population to validate
            salaries: Salary array
            salary_cap: Salary cap
            
        Returns:
            Validated population (may be smaller if sets are invalid)
        """
        if salaries is None or salary_cap is None:
            return population_sets
        
        valid_sets = []
        
        for set_idx, lineup_set in enumerate(population_sets):
            set_is_valid = True
            
            # Check each lineup in the set
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
        
        if len(valid_sets) < len(population_sets):
            logging.info(f'Validation: {len(valid_sets)}/{len(population_sets)} sets passed salary validation')
        
        return valid_sets_array
    
    def _calculate_set_diversity_metrics(self, 
                                       lineup_set: np.ndarray, 
                                       diversity_method: str) -> Dict[str, Any]:
        """Calculate diversity metrics for a single lineup set"""
        n_lineups = len(lineup_set)
        if n_lineups <= 1:
            return {'avg_overlap': 0.0, 'min_overlap': 0.0, 'diversity_matrix': np.array([[1.0]])}
        
        diversity_matrix = np.zeros((n_lineups, n_lineups))
        overlaps = []
        
        for i in range(n_lineups):
            for j in range(i + 1, n_lineups):
                if diversity_method == 'jaccard':
                    overlap = self._jaccard_similarity(lineup_set[i], lineup_set[j])
                elif diversity_method == 'hamming':
                    overlap = self._hamming_similarity(lineup_set[i], lineup_set[j])
                else:
                    overlap = self._jaccard_similarity(lineup_set[i], lineup_set[j])
                
                diversity_matrix[i, j] = overlap
                diversity_matrix[j, i] = overlap
                overlaps.append(overlap)
        
        # Set diagonal to 1.0 (lineup compared to itself)
        np.fill_diagonal(diversity_matrix, 1.0)
        
        return {
            'avg_overlap': np.mean(overlaps) if overlaps else 0.0,
            'min_overlap': np.min(overlaps) if overlaps else 0.0,
            'diversity_matrix': diversity_matrix
        }

    @staticmethod
    def _jaccard_similarity(lineup1: np.ndarray, lineup2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two lineups"""
        set1 = set(lineup1)
        set2 = set(lineup2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _hamming_similarity(lineup1: np.ndarray, lineup2: np.ndarray) -> float:
        """Calculate Hamming similarity between two lineups"""
        return np.sum(lineup1 == lineup2) / len(lineup1)
    
    def _evolve_elite_pool(self, 
                          population_sets: np.ndarray,
                          points: np.ndarray,
                          evolution_rate: float,
                          pospool: Dict,
                          posmap: Dict) -> None:
        """
        Evolve the elite pool by incorporating high-performing lineups from current population
        
        Args:
            population_sets: Current population of lineup sets
            points: Points array for fitness calculation
            evolution_rate: Fraction of elite pool to replace
            pospool: Position pool for generating new lineups
            posmap: Position mapping
        """
        if self.elite_pool is None or len(self.elite_pool) == 0:
            return
        
        # Extract all unique lineups from current population sets
        all_current_lineups = []
        for lineup_set in population_sets:
            for lineup in lineup_set:
                all_current_lineups.append(lineup)
        
        # Remove duplicates and calculate fitness
        unique_lineups = []
        seen_lineups = set()
        
        for lineup in all_current_lineups:
            lineup_tuple = tuple(lineup)
            if lineup_tuple not in seen_lineups:
                seen_lineups.add(lineup_tuple)
                unique_lineups.append(lineup)
        
        if len(unique_lineups) == 0:
            return
        
        # Calculate fitness for unique lineups
        unique_lineups = np.array(unique_lineups)
        unique_fitness = np.array([points[lineup].sum() for lineup in unique_lineups])
        
        # Find top performers from current population that aren't already in elite pool
        elite_set = {tuple(lineup) for lineup in self.elite_pool}
        new_candidates = []
        new_candidate_fitness = []
        
        for i, lineup in enumerate(unique_lineups):
            lineup_tuple = tuple(lineup)
            if lineup_tuple not in elite_set:
                new_candidates.append(lineup)
                new_candidate_fitness.append(unique_fitness[i])
        
        if len(new_candidates) == 0:
            logging.info('No new candidates found for elite pool evolution')
            return
        
        new_candidates = np.array(new_candidates)
        new_candidate_fitness = np.array(new_candidate_fitness)
        
        # Sort new candidates by fitness (descending)
        sorted_indices = np.argsort(new_candidate_fitness)[::-1]
        sorted_candidates = new_candidates[sorted_indices]
        sorted_candidate_fitness = new_candidate_fitness[sorted_indices]
        
        # Determine how many elite lineups to replace
        n_to_replace = max(1, int(len(self.elite_pool) * evolution_rate))
        n_to_replace = min(n_to_replace, len(sorted_candidates))
        
        if n_to_replace == 0:
            return
        
        # Find worst performers in current elite pool
        worst_elite_indices = np.argsort(self.elite_fitness)[:n_to_replace]
        
        # Check if new candidates are actually better than worst elite
        min_new_fitness = sorted_candidate_fitness[n_to_replace - 1]
        max_worst_elite_fitness = self.elite_fitness[worst_elite_indices].max()
        
        if min_new_fitness <= max_worst_elite_fitness:
            logging.info(f'New candidates not better than worst elite (best new: {min_new_fitness:.1f} vs worst elite: {max_worst_elite_fitness:.1f})')
            return
        
        # Replace worst elite lineups with best new candidates
        old_elite_count = len(self.elite_pool)
        for i, elite_idx in enumerate(worst_elite_indices):
            if i < len(sorted_candidates):
                self.elite_pool[elite_idx] = sorted_candidates[i]
                self.elite_fitness[elite_idx] = sorted_candidate_fitness[i]
        
        # Log the evolution
        new_elite_fitness_range = f"{self.elite_fitness.min():.1f} - {self.elite_fitness.max():.1f}"
        logging.info(f'Elite pool evolved: replaced {n_to_replace} lineups, new fitness range: {new_elite_fitness_range}')
        
        # Optionally, generate some completely new lineups to maintain diversity
        if evolution_rate > 0.05:  # Only if evolution rate is significant
            self._inject_fresh_lineups_to_elite(
                int(n_to_replace * 0.3),  # Replace 30% with fresh lineups
                points, pospool, posmap
            )
    
    def _inject_fresh_lineups_to_elite(self, 
                                     n_fresh: int,
                                     points: np.ndarray,
                                     pospool: Dict,
                                     posmap: Dict) -> None:
        """
        Inject completely new lineups into elite pool to maintain diversity
        
        Args:
            n_fresh: Number of fresh lineups to inject
            points: Points array for fitness calculation
            pospool: Position pool
            posmap: Position mapping
        """
        if n_fresh <= 0 or self.elite_pool is None:
            return
        populate_fresh = PopulatePoolBasedSets()
        
        # Generate fresh lineups
        try:
            # Use the position data preparation method
            pos_data = populate_fresh._prepare_position_data(pospool, posmap, 'prob')
            lineup_size = sum(posmap.values())
            
            fresh_lineups = populate_fresh._generate_initial_pool(
                pos_data, n_fresh * 10, lineup_size, True  # Generate 10x to select best
            )
            
            # Calculate fitness and select best
            fresh_fitness = np.array([points[lineup].sum() for lineup in fresh_lineups])
            best_fresh_indices = np.argsort(fresh_fitness)[-n_fresh:]
            
            selected_fresh = fresh_lineups[best_fresh_indices]
            selected_fresh_fitness = fresh_fitness[best_fresh_indices]
            
            # Replace some of the lower-performing elite lineups
            worst_elite_indices = np.argsort(self.elite_fitness)[:n_fresh]
            
            for i, elite_idx in enumerate(worst_elite_indices):
                if i < len(selected_fresh):
                    self.elite_pool[elite_idx] = selected_fresh[i]
                    self.elite_fitness[elite_idx] = selected_fresh_fitness[i]
            
            logging.info(f'Injected {len(selected_fresh)} fresh lineups into elite pool')
            
        except Exception as e:
            logging.warning(f'Failed to inject fresh lineups: {str(e)}')
