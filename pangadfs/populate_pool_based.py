# pangadfs/populate_pool_based.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from pangadfs.base import PopulateBase
from pangadfs.misc import multidimensional_shifting_fast

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class PopulatePoolBasedSets(PopulateBase):
    """
    Pool-based lineup set generation for multilineup optimization.
    
    Algorithm:
    1. Generate 100K individual lineups
    2. Rank by fitness and extract top 10K as elite pool
    3. Keep remaining 90K as general pool
    4. Remove duplicates and validate both pools
    5. Sample diverse sets from pools using adaptive ratios
    """

    def populate(self, 
                 pospool: Dict[str, pd.DataFrame],
                 posmap: Dict[str, int], 
                 population_size: int,
                 target_lineups: int,
                 probcol: str = 'prob',
                 diversity_threshold: float = 0.3,
                 max_attempts: int = 100,
                 initial_pool_size: int = 100000,
                 elite_pool_size: int = 10000,
                 initial_elite_ratio: float = 0.7,
                 memory_optimize: bool = True,
                 use_compression: bool = True,
                 points: np.ndarray = None,
                 salaries: np.ndarray = None,
                 salary_cap: int = None,
                 **kwargs) -> np.ndarray:
        """
        Creates initial population using pool-based approach
        
        Args:
            pospool: Dictionary of position DataFrames
            posmap: Position mapping (e.g., {'QB': 1, 'RB': 2, ...})
            population_size: Number of lineup sets to create
            target_lineups: Number of lineups per set
            probcol: Column name for probabilities
            diversity_threshold: Minimum diversity required between lineups in a set
            max_attempts: Maximum attempts to generate diverse lineup within a set
            initial_pool_size: Size of initial lineup pool (default 100K)
            elite_pool_size: Size of elite pool (default 10K)
            initial_elite_ratio: Initial ratio for sampling from elite pool
            memory_optimize: Use memory optimization techniques
            use_compression: Use compression for pool storage
            points: Points array for fitness calculation
            salaries: Salaries array for validation
            salary_cap: Salary cap for validation
            
        Returns:
            np.ndarray: Shape (population_size, target_lineups, lineup_size)
        """
        logging.info(f'POOL-BASED: Generating {population_size} sets of {target_lineups} lineups each')
        logging.info(f'Pool strategy: {initial_pool_size} total -> {elite_pool_size} elite + {initial_pool_size - elite_pool_size} general')
        
        # Calculate lineup size from posmap
        lineup_size = sum(posmap.values())
        
        # Pre-calculate position data for vectorized operations
        pos_data = self._prepare_position_data(pospool, posmap, probcol)
        
        # Step 1: Generate large pool of individual lineups
        logging.info(f'Step 1: Generating {initial_pool_size} individual lineups...')
        lineup_pool = self._generate_initial_pool(
            pos_data, initial_pool_size, lineup_size, memory_optimize
        )
        
        # Step 2: Evaluate fitness and create elite/general pools
        if points is not None:
            logging.info(f'Step 2: Evaluating fitness and creating stratified pools...')
            elite_pool, general_pool, elite_fitness, general_fitness = self._create_stratified_pools(
                lineup_pool, points, elite_pool_size, memory_optimize
            )
        else:
            # If no points provided, use random stratification
            logging.warning('No points provided - using random stratification')
            elite_pool = lineup_pool[:elite_pool_size]
            general_pool = lineup_pool[elite_pool_size:]
            elite_fitness = np.random.random(len(elite_pool))
            general_fitness = np.random.random(len(general_pool))
        
        # Step 3: Remove duplicates and validate pools
        logging.info(f'Step 3: Removing duplicates and validating pools...')
        elite_pool, elite_fitness = self._deduplicate_and_validate_pool(
            elite_pool, elite_fitness, salaries, salary_cap, "elite"
        )
        general_pool, general_fitness = self._deduplicate_and_validate_pool(
            general_pool, general_fitness, salaries, salary_cap, "general"
        )
        
        # Step 4: Sample diverse sets from pools
        logging.info(f'Step 4: Sampling {population_size} diverse sets from pools...')
        population_sets = self._sample_diverse_sets_from_pools(
            elite_pool, general_pool, elite_fitness, general_fitness,
            population_size, target_lineups, lineup_size,
            initial_elite_ratio, diversity_threshold
        )
        
        # Store pools for potential future use (optional)
        self._store_pools_metadata(elite_pool, general_pool, elite_fitness, general_fitness)
        
        logging.info(f'Pool-based generation complete: {len(elite_pool)} elite + {len(general_pool)} general lineups')
        return population_sets
    
    def _prepare_position_data(self, pospool: Dict[str, pd.DataFrame], 
                             posmap: Dict[str, int], probcol: str) -> Dict[str, Any]:
        """Prepare position data for vectorized operations"""
        pos_data = {}
        current_idx = 0
        
        for pos, count in posmap.items():
            if pos not in pospool or len(pospool[pos]) == 0:
                pos_data[pos] = {
                    'start_idx': current_idx,
                    'end_idx': current_idx + count,
                    'count': count,
                    'indices': np.array([]),
                    'probs': np.array([])
                }
                current_idx += count
                continue
            
            pos_df = pospool[pos]
            indices = pos_df.index.values
            
            if probcol in pos_df.columns:
                probs = pos_df[probcol].values
                probs = probs / np.sum(probs)  # Normalize
            else:
                probs = np.ones(len(indices)) / len(indices)
            
            pos_data[pos] = {
                'start_idx': current_idx,
                'end_idx': current_idx + count,
                'count': count,
                'indices': indices,
                'probs': probs.astype(np.float32)
            }
            current_idx += count
        
        return pos_data
    
    def _generate_initial_pool(self, pos_data: Dict[str, Any], 
                             pool_size: int, lineup_size: int,
                             memory_optimize: bool) -> np.ndarray:
        """Generate initial pool of lineups using vectorized operations"""
        
        # Use memory-optimized data type if requested
        dtype = np.int16 if memory_optimize else np.int32
        lineup_pool = np.zeros((pool_size, lineup_size), dtype=dtype)
        
        # Generate all lineups for each position at once using vectorized approach
        for pos, data in pos_data.items():
            if len(data['indices']) == 0:
                continue
            
            start_idx = data['start_idx']
            end_idx = data['end_idx']
            count = data['count']
            
            if count > 0 and len(data['indices']) >= count:
                # Use vectorized sampling for all lineups at once
                try:
                    selected_players = multidimensional_shifting_fast(
                        num_samples=pool_size,
                        sample_size=count,
                        probs=data['probs'],
                        elements=data['indices']
                    )
                    lineup_pool[:, start_idx:end_idx] = selected_players.astype(dtype)
                except:
                    # Fallback to individual generation if vectorized fails
                    logging.warning(f"Vectorized generation failed for position {pos}, using fallback")
                    for i in range(pool_size):
                        selected = np.random.choice(
                            data['indices'], size=count, replace=False, p=data['probs']
                        )
                        lineup_pool[i, start_idx:end_idx] = selected.astype(dtype)
        
        return lineup_pool
    
    def _create_stratified_pools(self, lineup_pool: np.ndarray, points: np.ndarray,
                               elite_pool_size: int, memory_optimize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create elite and general pools based on fitness ranking"""
        
        # Calculate fitness for all lineups
        logging.info('Calculating fitness for all lineups...')
        fitness_scores = np.array([points[lineup].sum() for lineup in lineup_pool])
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Split into elite and general pools
        elite_indices = sorted_indices[:elite_pool_size]
        general_indices = sorted_indices[elite_pool_size:]
        
        elite_pool = lineup_pool[elite_indices]
        general_pool = lineup_pool[general_indices]
        elite_fitness = fitness_scores[elite_indices]
        general_fitness = fitness_scores[general_indices]
        
        logging.info(f'Elite pool: {len(elite_pool)} lineups, fitness range: {elite_fitness.min():.1f} - {elite_fitness.max():.1f}')
        logging.info(f'General pool: {len(general_pool)} lineups, fitness range: {general_fitness.min():.1f} - {general_fitness.max():.1f}')
        
        return elite_pool, general_pool, elite_fitness, general_fitness
    
    def _deduplicate_and_validate_pool(self, pool: np.ndarray, fitness: np.ndarray,
                                     salaries: np.ndarray, salary_cap: int,
                                     pool_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Remove duplicates and validate salary constraints for a pool"""
        
        if salaries is None or salary_cap is None:
            logging.warning(f"No salary validation for {pool_name} pool")
            return self._deduplicate_pool(pool, fitness)
        
        # First remove duplicates
        unique_pool, unique_fitness = self._deduplicate_pool(pool, fitness)
        
        # Then validate salary constraints
        valid_indices = []
        for i, lineup in enumerate(unique_pool):
            lineup_salary = salaries[lineup].sum()
            if lineup_salary <= salary_cap:
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            logging.error(f"No valid lineups in {pool_name} pool after salary validation!")
            return unique_pool, unique_fitness
        
        valid_pool = unique_pool[valid_indices]
        valid_fitness = unique_fitness[valid_indices]
        
        removed_count = len(unique_pool) - len(valid_pool)
        logging.info(f'{pool_name.capitalize()} pool: {len(pool)} -> {len(unique_pool)} (dedup) -> {len(valid_pool)} (valid), removed {removed_count}')
        
        return valid_pool, valid_fitness
    
    def _deduplicate_pool(self, pool: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove duplicate lineups from pool"""
        
        # Convert to tuples for hashing
        lineup_tuples = [tuple(lineup) for lineup in pool]
        
        # Find unique lineups
        seen = set()
        unique_indices = []
        
        for i, lineup_tuple in enumerate(lineup_tuples):
            if lineup_tuple not in seen:
                seen.add(lineup_tuple)
                unique_indices.append(i)
        
        unique_pool = pool[unique_indices]
        unique_fitness = fitness[unique_indices]
        
        return unique_pool, unique_fitness
    
    def _sample_diverse_sets_from_pools(self, elite_pool: np.ndarray, general_pool: np.ndarray,
                                      elite_fitness: np.ndarray, general_fitness: np.ndarray,
                                      population_size: int, target_lineups: int, lineup_size: int,
                                      elite_ratio: float, diversity_threshold: float) -> np.ndarray:
        """Sample diverse lineup sets from elite and general pools"""
        
        population_sets = np.zeros((population_size, target_lineups, lineup_size), dtype=elite_pool.dtype)
        
        # Calculate sampling counts
        elite_count = int(target_lineups * elite_ratio)
        general_count = target_lineups - elite_count
        
        logging.info(f'Sampling strategy: {elite_count} from elite, {general_count} from general per set')
        
        for set_idx in range(population_size):
            if set_idx % 1000 == 0:
                logging.info(f'Sampled {set_idx}/{population_size} sets')
            
            lineup_set = self._sample_single_diverse_set(
                elite_pool, general_pool, elite_fitness, general_fitness,
                elite_count, general_count, diversity_threshold
            )
            
            # Fill the population set
            for i, lineup in enumerate(lineup_set):
                if i < target_lineups:
                    population_sets[set_idx, i] = lineup
        
        return population_sets
    
    def _sample_single_diverse_set(self, elite_pool: np.ndarray, general_pool: np.ndarray,
                                 elite_fitness: np.ndarray, general_fitness: np.ndarray,
                                 elite_count: int, general_count: int,
                                 diversity_threshold: float) -> list:
        """Sample a single diverse set from pools"""
        
        selected_lineups = []
        
        # Sample from elite pool first (fitness-weighted)
        if elite_count > 0 and len(elite_pool) > 0:
            elite_samples = self._sample_diverse_from_pool(
                elite_pool, elite_fitness, elite_count, diversity_threshold, selected_lineups
            )
            selected_lineups.extend(elite_samples)
        
        # Sample from general pool
        if general_count > 0 and len(general_pool) > 0:
            general_samples = self._sample_diverse_from_pool(
                general_pool, general_fitness, general_count, diversity_threshold, selected_lineups
            )
            selected_lineups.extend(general_samples)
        
        return selected_lineups
    
    def _sample_diverse_from_pool(self, pool: np.ndarray, fitness: np.ndarray,
                                count: int, diversity_threshold: float,
                                existing_lineups: list) -> list:
        """Sample diverse lineups from a specific pool"""
        
        if count <= 0 or len(pool) == 0:
            return []
        
        # Ensure we don't try to sample more than available
        count = min(count, len(pool))
        
        selected = []
        available_indices = list(range(len(pool)))
        
        # Create fitness-based probabilities for sampling
        if len(fitness) > 0:
            # Normalize fitness for probability calculation
            min_fitness = fitness.min()
            normalized_fitness = fitness - min_fitness + 1e-6  # Avoid zero probabilities
            probabilities = normalized_fitness / normalized_fitness.sum()
        else:
            probabilities = np.ones(len(pool)) / len(pool)
        
        for _ in range(count):
            if not available_indices:
                break
            
            best_candidate_idx = None
            best_diversity = -1.0
            
            # Sample candidates to check (fitness-weighted)
            n_candidates = min(50, len(available_indices))
            candidate_pool_indices = np.random.choice(
                available_indices, size=n_candidates, replace=False,
                p=probabilities[available_indices] / probabilities[available_indices].sum()
            )
            
            # Find most diverse candidate
            for pool_idx in candidate_pool_indices:
                candidate = pool[pool_idx]
                
                # Calculate diversity to existing lineups (both selected and from other pools)
                all_existing = existing_lineups + selected
                min_diversity = self._calculate_min_diversity_to_lineups(candidate, all_existing)
                
                if min_diversity > best_diversity:
                    best_candidate_idx = pool_idx
                    best_diversity = min_diversity
                    
                    # If diversity is good enough, use it
                    if min_diversity >= diversity_threshold:
                        break
            
            # Add the best candidate found
            if best_candidate_idx is not None:
                selected.append(pool[best_candidate_idx])
                available_indices.remove(best_candidate_idx)
            else:
                # Fallback: random selection
                random_idx = np.random.choice(available_indices)
                selected.append(pool[random_idx])
                available_indices.remove(random_idx)
        
        return selected
    
    def _calculate_min_diversity_to_lineups(self, candidate: np.ndarray, existing_lineups: list) -> float:
        """Calculate minimum diversity between candidate and existing lineups"""
        if not existing_lineups:
            return 1.0
        
        candidate_set = set(candidate)
        min_diversity = 1.0
        
        for existing_lineup in existing_lineups:
            existing_set = set(existing_lineup)
            intersection = len(candidate_set & existing_set)
            union = len(candidate_set | existing_set)
            
            if union > 0:
                similarity = intersection / union
                diversity = 1.0 - similarity
                min_diversity = min(min_diversity, diversity)
        
        return min_diversity
    
    def _store_pools_metadata(self, elite_pool: np.ndarray, general_pool: np.ndarray,
                            elite_fitness: np.ndarray, general_fitness: np.ndarray):
        """Store pool metadata for potential future use"""
        # Store as instance variables for potential access by optimizer
        self.elite_pool = elite_pool
        self.general_pool = general_pool
        self.elite_fitness = elite_fitness
        self.general_fitness = general_fitness
        
        logging.info(f'Stored pools: {len(elite_pool)} elite, {len(general_pool)} general lineups')


# Numba-optimized functions if available
if NUMBA_AVAILABLE:
    @njit
    def _calculate_fitness_numba(lineup_pool, points):
        """Numba-optimized fitness calculation"""
        fitness_scores = np.zeros(len(lineup_pool))
        for i in range(len(lineup_pool)):
            fitness_scores[i] = points[lineup_pool[i]].sum()
        return fitness_scores
    
    # Replace method if numba is available
    PopulatePoolBasedSets._calculate_fitness_fast = staticmethod(_calculate_fitness_numba)
