# pangadfs/pangadfs/optimize.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Any, Dict, List

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ModuleNotFoundError:
    HAS_NUMBA = False

from pangadfs.base import OptimizeBase
from pangadfs.ga import GeneticAlgorithm


def _jaccard_similarity_sorted_py(lineup1_sorted: np.ndarray, lineup2_sorted: np.ndarray) -> float:
    """Python fallback for sorted Jaccard overlap."""
    i = 0
    j = 0
    intersection = 0

    while i < len(lineup1_sorted) and j < len(lineup2_sorted):
        left = lineup1_sorted[i]
        right = lineup2_sorted[j]
        if left == right:
            intersection += 1
            i += 1
            j += 1
        elif left < right:
            i += 1
        else:
            j += 1

    union = len(lineup1_sorted) + len(lineup2_sorted) - intersection
    return intersection / union if union > 0 else 0.0


if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _jaccard_similarity_sorted_numba(lineup1_sorted: np.ndarray, lineup2_sorted: np.ndarray) -> float:
        i = 0
        j = 0
        intersection = 0

        while i < lineup1_sorted.shape[0] and j < lineup2_sorted.shape[0]:
            left = lineup1_sorted[i]
            right = lineup2_sorted[j]
            if left == right:
                intersection += 1
                i += 1
                j += 1
            elif left < right:
                i += 1
            else:
                j += 1

        union = lineup1_sorted.shape[0] + lineup2_sorted.shape[0] - intersection
        return intersection / union if union > 0 else 0.0

    @njit(cache=True, fastmath=True)
    def _max_jaccard_against_selected_numba(candidate_sorted: np.ndarray, selected_sorted: np.ndarray) -> float:
        max_overlap = 0.0
        for i in range(selected_sorted.shape[0]):
            overlap = _jaccard_similarity_sorted_numba(candidate_sorted, selected_sorted[i])
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap
else:
    _jaccard_similarity_sorted_numba = _jaccard_similarity_sorted_py

    def _max_jaccard_against_selected_numba(candidate_sorted: np.ndarray, selected_sorted: np.ndarray) -> float:
        max_overlap = 0.0
        for i in range(selected_sorted.shape[0]):
            overlap = _jaccard_similarity_sorted_py(candidate_sorted, selected_sorted[i])
            max_overlap = max(max_overlap, overlap)
        return max_overlap


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
        settings = ga.ctx['ga_settings']
        site = ga.ctx['site_settings']

        # Start profiling
        ga.profiler.start_optimization()
        # create pool and pospool
        # pospool used to generate initial population
        # is a dict of position_name: DataFrame
        pop_size = settings['population_size']
        pool = ga.pool(csvpth=settings['csvpth'])
        cmap = {
            'points': settings['points_column'],
            'position': settings['position_column'],
            'salary': settings['salary_column']
        }
        posfilter = site['posfilter']
        flex_positions = site['flex_positions']
        pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping=cmap, flex_positions=flex_positions)

        # create salary and points arrays
        # these match indices of pool
        points = pool[settings['points_column']].to_numpy()
        salaries = pool[settings['salary_column']].to_numpy()
        
        # create initial population
        initial_population = ga.populate(
            pospool=pospool, 
            posmap=site['posmap'], 
            population_size=pop_size
        )

        # apply validators
        # default is to validate duplicates, salary, and positions
        # can add other validators as desired
        initial_population = ga.validate(
            population=initial_population, 
            salaries=salaries,
            salary_cap=site['salary_cap'],
            pool=pool,
            posmap=site['posmap'],
            position_column=settings['position_column'],
            flex_positions=site['flex_positions']
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

        for i in range(1, settings['n_generations'] + 1):
            ga.profiler.start_generation(i)

            if n_unimproved == settings['stop_criteria']:
                break

            if settings.get('verbose'):
                logging.info(f'Starting generation {i}, best={best_fitness}')

            # Evolve population: select → crossover → mutate → validate → fitness
            population, population_fitness = self._evolve_population(
                ga, population, population_fitness, salaries, pool, points, n_unimproved
            )

            # Track best solution
            omidx = population_fitness.argmax()
            generation_max = population_fitness[omidx]

            if generation_max > best_fitness:
                best_fitness = generation_max
                best_lineup = population[omidx]
                n_unimproved = 0
                ga.profiler.mark_best_solution(i)
            else:
                n_unimproved += 1

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

    def _evolve_population(self, ga, population, population_fitness, salaries, pool, points, n_unimproved):
        """Single generation: select → crossover → mutate → validate → fitness.
        
        Args:
            ga: GeneticAlgorithm instance
            population: current population array
            population_fitness: current fitness array
            salaries: salary array for validation
            pool: player pool DataFrame
            n_unimproved: generations without improvement (affects mutation rate)
            
        Returns:
            Tuple of (new_population, new_fitness)
        """
        settings = ga.ctx['ga_settings']
        site = ga.ctx['site_settings']

        # Elite preservation
        elite = ga.select(
            population=population,
            population_fitness=population_fitness,
            n=len(population) // settings.get('elite_divisor', 5),
            method=settings.get('elite_method', 'fittest')
        )

        # Selection for breeding
        selected = ga.select(
            population=population,
            population_fitness=population_fitness,
            n=len(population),
            method=settings.get('select_method', 'roulette')
        )

        # Crossover + mutation
        crossed_over = ga.crossover(
            population=selected,
            method=settings.get('crossover_method', 'uniform')
        )
        mutation_rate = settings.get('mutation_rate', max(.05, n_unimproved / 50))
        mutated = ga.mutate(population=crossed_over, mutation_rate=mutation_rate)

        # Validate elite + mutated
        population = ga.validate(
            population=np.vstack((elite, mutated)),
            salaries=salaries,
            salary_cap=site['salary_cap'],
            pool=pool,
            posmap=site['posmap'],
            position_column=settings['position_column'],
            flex_positions=site['flex_positions']
        )

        # Assess fitness using precomputed points array
        population_fitness = ga.fitness(population=population, points=points)
        return population, population_fitness


class OptimizeMultilineup(OptimizeBase):

    def __init__(self):
        super().__init__()
        self._penalty_cache: Dict[Any, np.ndarray] = {}

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
        settings = ga.ctx['ga_settings']
        site = ga.ctx['site_settings']

        # Get multilineup settings with defaults
        target_lineups = settings.get('target_lineups', 1)
        diversity_weight = settings.get('diversity_weight', 0.2)
        min_overlap_threshold = settings.get('min_overlap_threshold', 0.4)  # More aggressive default
        diversity_method = settings.get('diversity_method', 'jaccard')
        
        # Start profiling
        ga.profiler.start_optimization()
        
        # Create pool and pospool (same as OptimizeDefault)
        pop_size = settings['population_size']
        pool = ga.pool(csvpth=settings['csvpth'])
        cmap = {
            'points': settings['points_column'],
            'position': settings['position_column'],
            'salary': settings['salary_column']
        }
        posfilter = site['posfilter']
        flex_positions = site['flex_positions']
        pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping=cmap, flex_positions=flex_positions)

        # Create salary and points arrays
        points = pool[settings['points_column']].to_numpy()
        salaries = pool[settings['salary_column']].to_numpy()
        
        # Create initial population
        initial_population = ga.populate(
            pospool=pospool, 
            posmap=site['posmap'], 
            population_size=pop_size
        )

        # Apply validators
        initial_population = ga.validate(
            population=initial_population, 
            salaries=salaries,
            salary_cap=site['salary_cap'],
            pool=pool,
            posmap=site['posmap'],
            position_column=settings['position_column'],
            flex_positions=site['flex_positions']
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

        evolve_helper = OptimizeDefault()

        for i in range(1, settings['n_generations'] + 1):
            # Start generation timing
            ga.profiler.start_generation(i)

            # End program after n generations if not improving
            if n_unimproved == settings['stop_criteria']:
                break

            # Display progress information with verbose parameter
            if settings.get('verbose'):
                logging.info(f'Starting generation {i}')
                logging.info(f'Best lineup score {best_fitness}')

            # Reuse the same optimized generation evolution path as OptimizeDefault.
            population, population_fitness = evolve_helper._evolve_population(
                ga, population, population_fitness, salaries, pool, points, n_unimproved
            )
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

        if target_count <= 0:
            return [], [], {'avg_overlap': 0.0, 'min_overlap': 0.0, 'diversity_matrix': np.array([[1.0]])}

        use_jaccard = diversity_method != 'hamming'
        sorted_population = np.sort(population, axis=1)

        selected_lineups = []
        selected_indices = []
        
        # Start with the absolute best lineup
        best_idx = fitness.argmax()
        selected_lineups.append(population[best_idx])
        selected_sorted_matrix = np.empty((0, sorted_population.shape[1]), dtype=sorted_population.dtype)
        if use_jaccard:
            selected_sorted_matrix = np.vstack((selected_sorted_matrix, sorted_population[best_idx]))
        selected_indices.append(best_idx)
        
        # If only one lineup requested, return early
        if target_count == 1:
            selected_scores = [fitness[best_idx]]
            diversity_metrics = self._calculate_diversity_metrics(selected_lineups, diversity_method)
            return selected_lineups, selected_scores, diversity_metrics
        
        # Sort population by fitness (descending) and track candidate availability
        sorted_indices = np.argsort(fitness)[::-1]
        available_mask = np.ones(len(population), dtype=bool)
        available_mask[best_idx] = False
        overlap_cache = np.full(len(population), -1.0, dtype=np.float64)
        overlap_cache_valid = np.zeros(len(population), dtype=bool)

        def _candidate_overlap(candidate_idx: int) -> float:
            if overlap_cache_valid[candidate_idx]:
                return overlap_cache[candidate_idx]

            candidate = population[candidate_idx]
            candidate_sorted = sorted_population[candidate_idx]

            if use_jaccard:
                max_overlap = _max_jaccard_against_selected_numba(candidate_sorted, selected_sorted_matrix)
            else:
                max_overlap = 0.0
                for selected in selected_lineups:
                    overlap = self._hamming_similarity(candidate, selected)
                    max_overlap = max(max_overlap, overlap)

            overlap_cache[candidate_idx] = max_overlap
            overlap_cache_valid[candidate_idx] = True
            return max_overlap
        
        # More aggressive diversity approach
        # Start with very strict diversity requirements and relax more aggressively
        current_threshold = max(min_overlap_threshold, 0.6)  # Start very strict (60% difference required)
        min_threshold = 0.1  # Don't go below 10% difference
        
        logging.info(f'Starting AGGRESSIVE multilineup selection: target={target_count}, initial_threshold={current_threshold:.3f}')
        
        # Force completion - guarantee we get the target number
        while len(selected_lineups) < target_count:
            if not np.any(available_mask):
                logging.warning(f'No more available lineups. Stopping at {len(selected_lineups)} lineups.')
                break
                
            found_lineup = False
            best_candidate_idx = None
            best_candidate_overlap = 1.0
            
            # Find the most diverse candidate that meets the current threshold
            for candidate_idx in sorted_indices:
                if not available_mask[candidate_idx]:
                    continue
                max_overlap = _candidate_overlap(candidate_idx)
                
                # If this candidate meets diversity threshold and is more diverse than current best
                if max_overlap <= (1.0 - current_threshold) and max_overlap < best_candidate_overlap:
                    best_candidate_idx = candidate_idx
                    best_candidate_overlap = max_overlap
                    found_lineup = True
            
            # If we found a diverse lineup, select it
            if found_lineup and best_candidate_idx is not None:
                selected_lineups.append(population[best_candidate_idx])
                if use_jaccard:
                    selected_sorted_matrix = np.vstack((selected_sorted_matrix, sorted_population[best_candidate_idx]))
                selected_indices.append(best_candidate_idx)
                available_mask[best_candidate_idx] = False
                overlap_cache_valid[:] = False
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
                    if np.any(available_mask):
                        # Find the lineup with minimum overlap to any selected lineup
                        best_remaining_idx = None
                        best_remaining_overlap = 1.0
                        
                        for candidate_idx in sorted_indices:
                            if not available_mask[candidate_idx]:
                                continue
                            max_overlap = _candidate_overlap(candidate_idx)
                            
                            if max_overlap < best_remaining_overlap:
                                best_remaining_idx = candidate_idx
                                best_remaining_overlap = max_overlap
                        
                        # Always select something to guarantee progress
                        if best_remaining_idx is not None:
                            selected_lineups.append(population[best_remaining_idx])
                            if use_jaccard:
                                selected_sorted_matrix = np.vstack((selected_sorted_matrix, sorted_population[best_remaining_idx]))
                            selected_indices.append(best_remaining_idx)
                            available_mask[best_remaining_idx] = False
                            overlap_cache_valid[:] = False
                            logging.info(f'Selected lineup {len(selected_lineups)}/{target_count} as most diverse remaining (overlap={best_remaining_overlap:.3f})')
                        else:
                            logging.warning('No selectable lineup remained under availability mask.')
                            break
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

        # Reuse penalties for the same selected-lineup set and candidate snapshot.
        selected_key = tuple(tuple(int(v) for v in np.sort(lineup).tolist()) for lineup in selected_lineups)
        candidates_key = tuple(tuple(int(v) for v in row.tolist()) for row in candidates)
        cache_key = (selected_key, candidates_key, float(min_overlap_threshold), diversity_method)
        if cache_key in self._penalty_cache:
            return self._penalty_cache[cache_key].copy()
        
        penalties = np.zeros(len(candidates))
        use_jaccard = diversity_method != 'hamming'
        sorted_candidates = np.sort(candidates, axis=1)
        sorted_selected = [np.sort(lineup) for lineup in selected_lineups]
        selected_sorted_matrix = np.asarray(sorted_selected) if sorted_selected else np.empty((0, candidates.shape[1]), dtype=sorted_candidates.dtype)
        
        for i, candidate in enumerate(candidates):
            max_overlap = 0.0
            
            if use_jaccard:
                max_overlap = _max_jaccard_against_selected_numba(sorted_candidates[i], selected_sorted_matrix)
            else:
                for selected in selected_lineups:
                    overlap = self._hamming_similarity(candidate, selected)
                    max_overlap = max(max_overlap, overlap)
            
            # Penalty increases as overlap approaches 1.0
            # If overlap exceeds threshold, apply heavy penalty
            if max_overlap > (1.0 - min_overlap_threshold):
                penalties[i] = 1000.0  # Heavy penalty for too similar lineups
            else:
                penalties[i] = max_overlap * 100.0  # Scaled penalty
        
        self._penalty_cache[cache_key] = penalties.copy()

        # Keep memory bounded for long-running sessions.
        if len(self._penalty_cache) > 32:
            oldest_key = next(iter(self._penalty_cache))
            self._penalty_cache.pop(oldest_key, None)

        return penalties

    def _calculate_diversity_metrics(self, lineups: List[np.ndarray], diversity_method: str) -> Dict[str, Any]:
        """Calculate diversity metrics for the selected lineups"""
        if len(lineups) <= 1:
            return {'avg_overlap': 0.0, 'min_overlap': 0.0, 'diversity_matrix': np.array([[1.0]])}
        
        n_lineups = len(lineups)
        diversity_matrix = np.zeros((n_lineups, n_lineups))
        
        overlaps = []
        use_jaccard = diversity_method != 'hamming'
        sorted_lineups = [np.sort(lineup) for lineup in lineups]
        for i in range(n_lineups):
            for j in range(i + 1, n_lineups):
                if use_jaccard:
                    overlap = self._jaccard_similarity_sorted(sorted_lineups[i], sorted_lineups[j])
                else:
                    overlap = self._hamming_similarity(lineups[i], lineups[j])
                
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
    def _jaccard_similarity_sorted(lineup1_sorted: np.ndarray, lineup2_sorted: np.ndarray) -> float:
        """Calculate Jaccard similarity for pre-sorted lineups without building sets."""
        return _jaccard_similarity_sorted_numba(lineup1_sorted, lineup2_sorted)

    @staticmethod
    def _hamming_similarity(lineup1: np.ndarray, lineup2: np.ndarray) -> float:
        """Calculate Hamming similarity between two lineups"""
        return np.sum(lineup1 == lineup2) / len(lineup1)
