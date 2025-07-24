# pangadfs/pangadfs/optimize.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Any, Dict, List

import numpy as np

from pangadfs.base import OptimizeBase
from pangadfs.ga import GeneticAlgorithm


class OptimizeDefault(OptimizeBase):

    def optimize(self, ga: GeneticAlgorithm, **kwargs) -> Dict[str, Any]:
        """Creates initial pool
        
        Args:
            ga (GeneticAlgorithm): the ga instance
            **kwargs: keyword arguments for plugins
            
        Returns:
            Dict
            'population': np.ndarray,
            'fitness': np.ndarray,
            'best_lineup': pd.DataFrame,
            'best_score': float

        """
        # Start profiling
        ga.profiler.start_optimization()
        # create pool and pospool
        # pospool used to generate initial population
        # is a dict of position_name: DataFrame
        pop_size = ga.ctx['ga_settings']['population_size']
        pool = ga.pool(csvpth=ga.ctx['ga_settings']['csvpth'])
        cmap = {'points': ga.ctx['ga_settings']['points_column'],
                'position': ga.ctx['ga_settings']['position_column'],
                'salary': ga.ctx['ga_settings']['salary_column']}
        posfilter = ga.ctx['site_settings']['posfilter']
        flex_positions = ga.ctx['site_settings']['flex_positions']
        pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping=cmap, flex_positions=flex_positions)

        # create salary and points arrays
        # these match indices of pool
        cmap = {'points': ga.ctx['ga_settings']['points_column'],
                'salary': ga.ctx['ga_settings']['salary_column']}
        points = pool[cmap['points']].values
        salaries = pool[cmap['salary']].values
        
        # create initial population
        initial_population = ga.populate(
            pospool=pospool, 
            posmap=ga.ctx['site_settings']['posmap'], 
            population_size=pop_size
        )

        # apply validators
        # default is to validate duplicates, salary, and positions
        # can add other validators as desired
        initial_population = ga.validate(
            population=initial_population, 
            salaries=salaries,
            salary_cap=ga.ctx['site_settings']['salary_cap'],
            pool=pool,
            posmap=ga.ctx['site_settings']['posmap'],
            position_column=ga.ctx['ga_settings']['position_column'],
            flex_positions=ga.ctx['site_settings']['flex_positions']
        )

        # need fitness to determine best lineup
        # and also for selection when loop starts
        population_fitness = ga.fitness(
            population=initial_population, 
            points=points
        )

        # set overall_max based on initial population
        omidx = population_fitness.argmax()
        best_fitness = population_fitness[omidx]
        best_lineup = initial_population[omidx]

        # Mark setup phase complete
        ga.profiler.mark_setup_complete()
        
        # Mark initial best solution (generation 0)
        ga.profiler.mark_best_solution(0)

        # create new generations
        n_unimproved = 0
        population = initial_population.copy()

        for i in range(1, ga.ctx['ga_settings']['n_generations'] + 1):
            # Start generation timing
            ga.profiler.start_generation(i)

            # end program after n generations if not improving
            if n_unimproved == ga.ctx['ga_settings']['stop_criteria']:
                break

            # display progress information with verbose parameter
            if ga.ctx['ga_settings'].get('verbose'):
                logging.info(f'Starting generation {i}')
                logging.info(f'Best lineup score {best_fitness}')

            # select the population
            # here, we are holding back the fittest 20% to ensure
            # that crossover and mutation do not overwrite good individuals
            elite = ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population) // ga.ctx['ga_settings'].get('elite_divisor', 5),
                method=ga.ctx['ga_settings'].get('elite_method', 'fittest')
            )

            selected = ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population),
                method=ga.ctx['ga_settings'].get('select_method', 'roulette')
            )

            # cross over the population
            # here, we use uniform crossover, which splits the population
            # and randomly exchanges 0 - all chromosomes
            crossed_over = ga.crossover(population=selected, method=ga.ctx['ga_settings'].get('crossover_method', 'uniform'))

            # mutate the crossed over population (leave elite alone)
            # can use fixed rate or variable to reduce mutation over generations
            # here we use a variable rate that increases if no improvement is found
            mutation_rate = ga.ctx['ga_settings'].get('mutation_rate', max(.05, n_unimproved / 50))
            mutated = ga.mutate(population=crossed_over, mutation_rate=mutation_rate)

            # validate the population (elite + mutated)
            population = ga.validate(
                population=np.vstack((elite, mutated)), 
                salaries=salaries, 
                salary_cap=ga.ctx['site_settings']['salary_cap'],
                pool=pool,
                posmap=ga.ctx['site_settings']['posmap'],
                position_column=ga.ctx['ga_settings']['position_column'],
                flex_positions=ga.ctx['site_settings']['flex_positions']
            )
            
            # assess fitness and get the best score
            population_fitness = ga.fitness(population=population, points=points)
            omidx = population_fitness.argmax()
            generation_max = population_fitness[omidx]
        
            # if new best score, then set n_unimproved to 0
            # and save the new best score and lineup
            # otherwise increment n_unimproved
            if generation_max > best_fitness:
                logging.info(f'Lineup improved to {generation_max}')
                best_fitness = generation_max
                best_lineup = population[omidx]
                n_unimproved = 0
                # Mark when best solution was found
                ga.profiler.mark_best_solution(i)
            else:
                n_unimproved += 1
                logging.info(f'Lineup unimproved {n_unimproved} times')
            
            # End generation timing
            ga.profiler.end_generation()

        # End profiling
        ga.profiler.end_optimization()

        # FINALIZE RESULTS
        # will break after n_generations or when stop_criteria reached
        results = {
            'population': population,
            'fitness': population_fitness,
            'best_lineup': pool.loc[best_lineup, :],
            'best_score': best_fitness
        }
        
        # Add profiling data to results
        if ga.profiler.enabled:
            results['profiling'] = ga.profiler.export_to_dict()
        
        return results


class OptimizeMultilineup(OptimizeBase):

    def optimize(self, ga: GeneticAlgorithm, **kwargs) -> Dict[str, Any]:
        """Optimizes for multiple diverse lineups
        
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
        # Get multilineup settings with defaults
        target_lineups = ga.ctx['ga_settings'].get('target_lineups', 1)
        diversity_weight = ga.ctx['ga_settings'].get('diversity_weight', 0.2)
        min_overlap_threshold = ga.ctx['ga_settings'].get('min_overlap_threshold', 0.4)  # More aggressive default
        diversity_method = ga.ctx['ga_settings'].get('diversity_method', 'jaccard')
        
        # Start profiling
        ga.profiler.start_optimization()
        
        # Create pool and pospool (same as OptimizeDefault)
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
        
        # Create initial population
        initial_population = ga.populate(
            pospool=pospool, 
            posmap=ga.ctx['site_settings']['posmap'], 
            population_size=pop_size
        )

        # Apply validators
        initial_population = ga.validate(
            population=initial_population, 
            salaries=salaries,
            salary_cap=ga.ctx['site_settings']['salary_cap'],
            pool=pool,
            posmap=ga.ctx['site_settings']['posmap'],
            position_column=ga.ctx['ga_settings']['position_column'],
            flex_positions=ga.ctx['site_settings']['flex_positions']
        )

        # Calculate fitness
        population_fitness = ga.fitness(
            population=initial_population, 
            points=points
        )

        # Set overall_max based on initial population
        omidx = population_fitness.argmax()
        best_fitness = population_fitness[omidx]
        best_lineup = initial_population[omidx]

        # Mark setup phase complete
        ga.profiler.mark_setup_complete()
        
        # Mark initial best solution (generation 0)
        ga.profiler.mark_best_solution(0)

        # Create new generations (same GA loop as OptimizeDefault)
        n_unimproved = 0
        population = initial_population.copy()

        for i in range(1, ga.ctx['ga_settings']['n_generations'] + 1):
            # Start generation timing
            ga.profiler.start_generation(i)

            # End program after n generations if not improving
            if n_unimproved == ga.ctx['ga_settings']['stop_criteria']:
                break

            # Display progress information with verbose parameter
            if ga.ctx['ga_settings'].get('verbose'):
                logging.info(f'Starting generation {i}')
                logging.info(f'Best lineup score {best_fitness}')

            # Select the population
            elite = ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population) // ga.ctx['ga_settings'].get('elite_divisor', 5),
                method=ga.ctx['ga_settings'].get('elite_method', 'fittest')
            )

            selected = ga.select(
                population=population, 
                population_fitness=population_fitness, 
                n=len(population),
                method=ga.ctx['ga_settings'].get('select_method', 'roulette')
            )

            # Cross over the population
            crossed_over = ga.crossover(population=selected, method=ga.ctx['ga_settings'].get('crossover_method', 'uniform'))

            # Mutate the crossed over population
            mutation_rate = ga.ctx['ga_settings'].get('mutation_rate', max(.05, n_unimproved / 50))
            mutated = ga.mutate(population=crossed_over, mutation_rate=mutation_rate)

            # Validate the population
            population = ga.validate(
                population=np.vstack((elite, mutated)), 
                salaries=salaries, 
                salary_cap=ga.ctx['site_settings']['salary_cap'],
                pool=pool,
                posmap=ga.ctx['site_settings']['posmap'],
                position_column=ga.ctx['ga_settings']['position_column'],
                flex_positions=ga.ctx['site_settings']['flex_positions']
            )
            
            # Assess fitness and get the best score
            population_fitness = ga.fitness(population=population, points=points)
            omidx = population_fitness.argmax()
            generation_max = population_fitness[omidx]
        
            # If new best score, then set n_unimproved to 0
            if generation_max > best_fitness:
                logging.info(f'Lineup improved to {generation_max}')
                best_fitness = generation_max
                best_lineup = population[omidx]
                n_unimproved = 0
                # Mark when best solution was found
                ga.profiler.mark_best_solution(i)
            else:
                n_unimproved += 1
                logging.info(f'Lineup unimproved {n_unimproved} times')
            
            # End generation timing
            ga.profiler.end_generation()

        # End profiling
        ga.profiler.end_optimization()

        # MULTILINEUP SELECTION
        # Select diverse lineups from final population
        if target_lineups == 1:
            # Single lineup mode - return same structure as OptimizeDefault
            selected_lineups = [population[omidx]]
            selected_scores = [best_fitness]
            diversity_metrics = {'avg_overlap': 0.0, 'min_overlap': 0.0, 'diversity_matrix': np.array([[1.0]])}
        else:
            # Multiple lineup mode - select diverse lineups
            selected_lineups, selected_scores, diversity_metrics = self._select_diverse_lineups(
                population, population_fitness, target_lineups, diversity_weight, min_overlap_threshold, diversity_method
            )

        # FINALIZE RESULTS
        results = {
            'population': population,
            'fitness': population_fitness,
            'best_lineup': pool.loc[best_lineup, :],  # Backward compatibility
            'best_score': best_fitness,               # Backward compatibility
            'lineups': [pool.loc[lineup, :] for lineup in selected_lineups],
            'scores': selected_scores,
            'diversity_metrics': diversity_metrics
        }
        
        # Add profiling data to results
        if ga.profiler.enabled:
            results['profiling'] = ga.profiler.export_to_dict()
        
        return results

    def _select_diverse_lineups(self, population: np.ndarray, fitness: np.ndarray, 
                               target_count: int, diversity_weight: float, 
                               min_overlap_threshold: float, diversity_method: str) -> tuple:
        """
        Select diverse lineups that balance high fitness with diversity
        Uses a more aggressive diversity-first approach
        
        Args:
            population: Final population from GA
            fitness: Fitness scores for population
            target_count: Number of lineups to select
            diversity_weight: Weight for diversity penalty (0-1)
            min_overlap_threshold: Minimum allowed overlap between lineups
            diversity_method: Method for calculating diversity
            
        Returns:
            Tuple of (selected_lineups, selected_scores, diversity_metrics)
        """
        if target_count > len(population):
            target_count = len(population)
            logging.warning(f'Target lineups ({target_count}) exceeds population size. Using full population.')

        selected_lineups = []
        selected_indices = []
        
        # Start with the absolute best lineup
        best_idx = fitness.argmax()
        selected_lineups.append(population[best_idx])
        selected_indices.append(best_idx)
        
        # If only one lineup requested, return early
        if target_count == 1:
            selected_scores = [fitness[best_idx]]
            diversity_metrics = self._calculate_diversity_metrics(selected_lineups, diversity_method)
            return selected_lineups, selected_scores, diversity_metrics
        
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness)[::-1]
        available_indices = [idx for idx in sorted_indices if idx != best_idx]
        
        # More aggressive diversity approach
        # Start with very strict diversity requirements and relax more aggressively
        current_threshold = max(min_overlap_threshold, 0.6)  # Start very strict (60% difference required)
        min_threshold = 0.1  # Don't go below 10% difference
        
        logging.info(f'Starting AGGRESSIVE multilineup selection: target={target_count}, initial_threshold={current_threshold:.3f}')
        
        # Force completion - guarantee we get the target number
        while len(selected_lineups) < target_count:
            if not available_indices:
                logging.warning(f'No more available lineups. Stopping at {len(selected_lineups)} lineups.')
                break
                
            found_lineup = False
            best_candidate_idx = None
            best_candidate_overlap = 1.0
            
            # Find the most diverse candidate that meets the current threshold
            for candidate_idx in available_indices:
                candidate = population[candidate_idx]
                
                # Check diversity against all selected lineups
                max_overlap = 0.0
                
                for selected in selected_lineups:
                    if diversity_method == 'jaccard':
                        overlap = self._jaccard_similarity(candidate, selected)
                    elif diversity_method == 'hamming':
                        overlap = self._hamming_similarity(candidate, selected)
                    else:
                        overlap = self._jaccard_similarity(candidate, selected)
                    
                    max_overlap = max(max_overlap, overlap)
                
                # If this candidate meets diversity threshold and is more diverse than current best
                if max_overlap <= (1.0 - current_threshold) and max_overlap < best_candidate_overlap:
                    best_candidate_idx = candidate_idx
                    best_candidate_overlap = max_overlap
                    found_lineup = True
            
            # If we found a diverse lineup, select it
            if found_lineup and best_candidate_idx is not None:
                selected_lineups.append(population[best_candidate_idx])
                selected_indices.append(best_candidate_idx)
                available_indices.remove(best_candidate_idx)
                logging.info(f'Selected lineup {len(selected_lineups)}/{target_count} with overlap={best_candidate_overlap:.3f} (threshold={current_threshold:.3f})')
            else:
                # No lineup meets current threshold
                if current_threshold > min_threshold:
                    # Relax threshold and try again
                    current_threshold *= 0.5  # Very aggressive relaxation (50% reduction)
                    logging.info(f'Relaxing diversity threshold to {current_threshold:.3f}')
                    continue  # Try again with relaxed threshold
                else:
                    # Threshold is already at minimum, just take the most diverse remaining lineup
                    if available_indices:
                        # Find the lineup with minimum overlap to any selected lineup
                        best_remaining_idx = available_indices[0]  # Default to first available
                        best_remaining_overlap = 1.0
                        
                        for candidate_idx in available_indices:
                            candidate = population[candidate_idx]
                            max_overlap = 0.0
                            
                            for selected in selected_lineups:
                                if diversity_method == 'jaccard':
                                    overlap = self._jaccard_similarity(candidate, selected)
                                elif diversity_method == 'hamming':
                                    overlap = self._hamming_similarity(candidate, selected)
                                else:
                                    overlap = self._jaccard_similarity(candidate, selected)
                                
                                max_overlap = max(max_overlap, overlap)
                            
                            if max_overlap < best_remaining_overlap:
                                best_remaining_idx = candidate_idx
                                best_remaining_overlap = max_overlap
                        
                        # Always select something to guarantee progress
                        selected_lineups.append(population[best_remaining_idx])
                        selected_indices.append(best_remaining_idx)
                        available_indices.remove(best_remaining_idx)
                        logging.info(f'Selected lineup {len(selected_lineups)}/{target_count} as most diverse remaining (overlap={best_remaining_overlap:.3f})')
                    else:
                        # This should never happen, but just in case
                        logging.error('No available indices but target not reached - this should not happen')
                        break
        
        # Calculate diversity metrics
        selected_scores = [fitness[idx] for idx in selected_indices]
        diversity_metrics = self._calculate_diversity_metrics(selected_lineups, diversity_method)
        
        logging.info(f'Successfully selected {len(selected_lineups)} lineups with avg overlap: {diversity_metrics["avg_overlap"]:.3f}')
        
        return selected_lineups, selected_scores, diversity_metrics

    def _calculate_selection_scores(self, population: np.ndarray, fitness: np.ndarray,
                                   selected_lineups: List[np.ndarray], selected_indices: List[int],
                                   diversity_weight: float, min_overlap_threshold: float, 
                                   diversity_method: str) -> np.ndarray:
        """
        Calculate scores that balance fitness and diversity from already-selected lineups
        """
        available_mask = np.ones(len(population), dtype=bool)
        available_mask[selected_indices] = False
        
        if not np.any(available_mask):
            return np.full(len(population), -np.inf)
        
        # Calculate diversity penalties for available lineups
        diversity_penalties = self._calculate_diversity_penalties(
            population[available_mask], selected_lineups, min_overlap_threshold, diversity_method
        )
        
        # Combine fitness and diversity: score = fitness - (diversity_weight * penalty)
        scores = np.full(len(population), -np.inf)
        scores[available_mask] = (
            fitness[available_mask] - diversity_weight * diversity_penalties
        )
        
        return scores

    def _calculate_diversity_penalties(self, candidates: np.ndarray, selected_lineups: List[np.ndarray],
                                     min_overlap_threshold: float, diversity_method: str) -> np.ndarray:
        """
        Calculate diversity penalties for candidate lineups against selected lineups
        """
        if len(selected_lineups) == 0:
            return np.zeros(len(candidates))
        
        penalties = np.zeros(len(candidates))
        
        for i, candidate in enumerate(candidates):
            max_overlap = 0.0
            
            for selected in selected_lineups:
                if diversity_method == 'jaccard':
                    overlap = self._jaccard_similarity(candidate, selected)
                elif diversity_method == 'hamming':
                    overlap = self._hamming_similarity(candidate, selected)
                else:  # Default to jaccard
                    overlap = self._jaccard_similarity(candidate, selected)
                
                max_overlap = max(max_overlap, overlap)
            
            # Penalty increases as overlap approaches 1.0
            # If overlap exceeds threshold, apply heavy penalty
            if max_overlap > (1.0 - min_overlap_threshold):
                penalties[i] = 1000.0  # Heavy penalty for too similar lineups
            else:
                penalties[i] = max_overlap * 100.0  # Scaled penalty
        
        return penalties

    def _calculate_diversity_metrics(self, lineups: List[np.ndarray], diversity_method: str) -> Dict[str, Any]:
        """Calculate diversity metrics for the selected lineups"""
        if len(lineups) <= 1:
            return {'avg_overlap': 0.0, 'min_overlap': 0.0, 'diversity_matrix': np.array([[1.0]])}
        
        n_lineups = len(lineups)
        diversity_matrix = np.zeros((n_lineups, n_lineups))
        
        overlaps = []
        for i in range(n_lineups):
            for j in range(i + 1, n_lineups):
                if diversity_method == 'jaccard':
                    overlap = self._jaccard_similarity(lineups[i], lineups[j])
                elif diversity_method == 'hamming':
                    overlap = self._hamming_similarity(lineups[i], lineups[j])
                else:
                    overlap = self._jaccard_similarity(lineups[i], lineups[j])
                
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
