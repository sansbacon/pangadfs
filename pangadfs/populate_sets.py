# pangadfs/populate_sets.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from pangadfs.base import PopulateBase


class PopulateMultilineupSets(PopulateBase):
    """Creates initial population of lineup sets for multilineup optimization"""

    def populate(self, 
                 pospool: Dict[str, pd.DataFrame],
                 posmap: Dict[str, int], 
                 population_size: int,
                 target_lineups: int,
                 probcol: str = 'prob',
                 diversity_threshold: float = 0.3,
                 max_attempts: int = 100,
                 lineup_pool_size: int = 100000,
                 **kwargs) -> np.ndarray:
        """
        Creates initial population where each individual is a set of diverse lineups
        Uses efficient approach: generate large lineup pool first, then sample diverse sets
        
        Args:
            pospool: Dictionary of position DataFrames
            posmap: Position mapping (e.g., {'QB': 1, 'RB': 2, ...})
            population_size: Number of lineup sets to create
            target_lineups: Number of lineups per set
            probcol: Column name for probabilities
            diversity_threshold: Minimum diversity required between lineups in a set
            max_attempts: Maximum attempts to generate diverse lineup within a set
            lineup_pool_size: Size of initial lineup pool to generate
            
        Returns:
            np.ndarray: Shape (population_size, target_lineups, lineup_size)
        """
        logging.info(f'Generating {population_size} sets of {target_lineups} diverse lineups each')
        logging.info(f'Using efficient approach: creating pool of {lineup_pool_size} lineups first')
        
        # Calculate lineup size from posmap
        lineup_size = sum(posmap.values())
        
        # Pre-calculate position indices for efficiency
        pos_indices = self._calculate_position_indices(posmap)
        
        # Step 1: Generate large pool of lineups efficiently
        logging.info(f'Step 1: Generating pool of {lineup_pool_size} lineups...')
        lineup_pool = self._generate_lineup_pool(
            pospool, pos_indices, lineup_pool_size, probcol
        )
        
        # Step 2: Sample diverse sets from the pool
        logging.info(f'Step 2: Sampling {population_size} diverse sets from pool...')
        population_sets = self._sample_diverse_sets_from_pool(
            lineup_pool, population_size, target_lineups, diversity_threshold
        )
        
        logging.info(f'Successfully generated {population_size} diverse lineup sets')
        return population_sets
    
    @staticmethod
    def _calculate_position_indices(posmap: Dict[str, int]) -> Dict[str, tuple]:
        """Calculate start and end indices for each position in the lineup array"""
        pos_indices = {}
        current_idx = 0
        
        for pos, count in posmap.items():
            start_idx = current_idx
            end_idx = current_idx + count
            pos_indices[pos] = (start_idx, end_idx)
            current_idx = end_idx
            
        return pos_indices
    
    def _generate_diverse_lineup_set(self, 
                                   pospool: Dict[str, pd.DataFrame],
                                   pos_indices: Dict[str, tuple],
                                   target_lineups: int,
                                   probcol: str,
                                   diversity_threshold: float,
                                   max_attempts: int) -> np.ndarray:
        """Generate a set of diverse lineups"""
        lineup_size = sum(end - start for start, end in pos_indices.values())
        lineup_set = np.zeros((target_lineups, lineup_size), dtype=int)
        
        # Generate first lineup normally
        lineup_set[0] = self._generate_single_lineup(pospool, pos_indices, probcol)
        
        # Generate remaining lineups with diversity constraints
        for lineup_idx in range(1, target_lineups):
            best_lineup = None
            best_diversity = 0.0
            
            # Try multiple attempts to find a diverse lineup
            for attempt in range(max_attempts):
                candidate = self._generate_single_lineup(pospool, pos_indices, probcol)
                
                # Calculate minimum diversity to existing lineups
                min_diversity = self._calculate_min_diversity_to_set(
                    candidate, lineup_set[:lineup_idx]
                )
                
                # If this candidate is more diverse than our current best, keep it
                if min_diversity > best_diversity:
                    best_lineup = candidate
                    best_diversity = min_diversity
                    
                    # If it meets our threshold, use it immediately
                    if min_diversity >= diversity_threshold:
                        break
            
            # Use the best lineup we found (even if it doesn't meet threshold)
            if best_lineup is not None:
                lineup_set[lineup_idx] = best_lineup
            else:
                # Fallback: generate a random lineup
                lineup_set[lineup_idx] = self._generate_single_lineup(pospool, pos_indices, probcol)
        
        return lineup_set
    
    @staticmethod
    def _generate_single_lineup(pospool: Dict[str, pd.DataFrame],
                              pos_indices: Dict[str, tuple],
                              probcol: str) -> np.ndarray:
        """Generate a single lineup using the existing populate logic"""
        lineup_size = sum(end - start for start, end in pos_indices.values())
        lineup = np.zeros(lineup_size, dtype=int)
        
        for pos, (start_idx, end_idx) in pos_indices.items():
            if pos not in pospool:
                continue
                
            pos_df = pospool[pos]
            n_positions = end_idx - start_idx
            
            if len(pos_df) == 0:
                continue
                
            # Use multidimensional_shifting for consistent selection
            if probcol in pos_df.columns:
                probs = pos_df[probcol].values
            else:
                # Equal probability if no prob column
                probs = np.ones(len(pos_df)) / len(pos_df)
            
            # Ensure we don't try to select more players than available
            n_select = min(n_positions, len(pos_df))
            
            if n_select > 0:
                # Use numpy random choice with probabilities for selection
                if len(probs) > 0 and np.sum(probs) > 0:
                    # Normalize probabilities
                    probs = probs / np.sum(probs)
                    selected_indices = np.random.choice(
                        pos_df.index.values, 
                        size=n_select, 
                        replace=False, 
                        p=probs
                    )
                else:
                    # Fallback to random selection without probabilities
                    selected_indices = np.random.choice(
                        pos_df.index.values, 
                        size=n_select, 
                        replace=False
                    )
                
                # Fill the lineup positions
                for i, player_idx in enumerate(selected_indices):
                    if start_idx + i < lineup_size:
                        lineup[start_idx + i] = player_idx
        
        return lineup
    
    @staticmethod
    def _calculate_min_diversity_to_set(candidate: np.ndarray, 
                                      existing_lineups: np.ndarray) -> float:
        """Calculate minimum Jaccard diversity between candidate and existing lineups"""
        if len(existing_lineups) == 0:
            return 1.0
        
        min_diversity = 1.0
        candidate_set = set(candidate)
        
        for existing_lineup in existing_lineups:
            existing_set = set(existing_lineup)
            
            # Calculate Jaccard diversity (1 - Jaccard similarity)
            intersection = len(candidate_set.intersection(existing_set))
            union = len(candidate_set.union(existing_set))
            
            if union > 0:
                jaccard_similarity = intersection / union
                jaccard_diversity = 1.0 - jaccard_similarity
                min_diversity = min(min_diversity, jaccard_diversity)
        
        return min_diversity
    
    def _generate_lineup_pool(self, 
                            pospool: Dict[str, pd.DataFrame],
                            pos_indices: Dict[str, tuple],
                            pool_size: int,
                            probcol: str) -> np.ndarray:
        """
        Generate a large pool of lineups efficiently using vectorized operations
        
        Args:
            pospool: Dictionary of position DataFrames
            pos_indices: Position indices mapping
            pool_size: Number of lineups to generate
            probcol: Column name for probabilities
            
        Returns:
            np.ndarray: Shape (pool_size, lineup_size)
        """
        lineup_size = sum(end - start for start, end in pos_indices.values())
        lineup_pool = np.zeros((pool_size, lineup_size), dtype=int)
        
        # Generate lineups in batches for efficiency
        batch_size = min(10000, pool_size)
        
        for batch_start in range(0, pool_size, batch_size):
            batch_end = min(batch_start + batch_size, pool_size)
            current_batch_size = batch_end - batch_start
            
            if batch_start % 50000 == 0:
                logging.info(f'Generated {batch_start}/{pool_size} lineups in pool')
            
            # Generate batch of lineups
            for i in range(current_batch_size):
                lineup_pool[batch_start + i] = self._generate_single_lineup(
                    pospool, pos_indices, probcol
                )
        
        logging.info(f'Completed generating {pool_size} lineups in pool')
        return lineup_pool
    
    def _sample_diverse_sets_from_pool(self, 
                                     lineup_pool: np.ndarray,
                                     population_size: int,
                                     target_lineups: int,
                                     diversity_threshold: float) -> np.ndarray:
        """
        Sample diverse lineup sets from the pre-generated pool
        
        Args:
            lineup_pool: Pre-generated pool of lineups (pool_size, lineup_size)
            population_size: Number of lineup sets to create
            target_lineups: Number of lineups per set
            diversity_threshold: Minimum diversity required between lineups in a set
            
        Returns:
            np.ndarray: Shape (population_size, target_lineups, lineup_size)
        """
        pool_size, lineup_size = lineup_pool.shape
        population_sets = np.zeros((population_size, target_lineups, lineup_size), dtype=int)
        
        # Pre-compute all pairwise Jaccard similarities for efficiency
        logging.info('Pre-computing pairwise similarities for efficient sampling...')
        similarity_matrix = self._compute_similarity_matrix(lineup_pool)
        
        for set_idx in range(population_size):
            if set_idx % 1000 == 0:
                logging.info(f'Sampled {set_idx}/{population_size} diverse sets')
            
            # Sample a diverse set using the similarity matrix
            selected_indices = self._sample_diverse_set_indices(
                similarity_matrix, target_lineups, diversity_threshold
            )
            
            # Fill the set with selected lineups
            for lineup_idx, pool_idx in enumerate(selected_indices):
                population_sets[set_idx, lineup_idx] = lineup_pool[pool_idx]
        
        logging.info(f'Completed sampling {population_size} diverse sets')
        return population_sets
    
    def _compute_similarity_matrix(self, lineup_pool: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Jaccard similarities between all lineups in the pool
        Uses efficient vectorized operations
        
        Args:
            lineup_pool: Pool of lineups (pool_size, lineup_size)
            
        Returns:
            np.ndarray: Similarity matrix (pool_size, pool_size)
        """
        pool_size = len(lineup_pool)
        
        # For very large pools, we'll use a sampling approach
        if pool_size > 50000:
            logging.info(f'Large pool detected ({pool_size}), using sampling approach for diversity')
            return self._compute_similarity_matrix_sampled(lineup_pool)
        
        # For smaller pools, compute full similarity matrix
        similarity_matrix = np.zeros((pool_size, pool_size), dtype=np.float32)
        
        # Compute similarities in chunks to manage memory
        chunk_size = min(1000, pool_size)
        
        for i in range(0, pool_size, chunk_size):
            end_i = min(i + chunk_size, pool_size)
            
            for j in range(i, pool_size, chunk_size):
                end_j = min(j + chunk_size, pool_size)
                
                # Compute similarities for this chunk
                for ii in range(i, end_i):
                    for jj in range(j, end_j):
                        if ii <= jj:  # Only compute upper triangle
                            sim = self._jaccard_similarity(lineup_pool[ii], lineup_pool[jj])
                            similarity_matrix[ii, jj] = sim
                            similarity_matrix[jj, ii] = sim  # Symmetric
        
        return similarity_matrix
    
    @staticmethod
    def _compute_similarity_matrix_sampled(lineup_pool: np.ndarray) -> np.ndarray:
        """
        For very large pools, use a sampled approach to estimate similarities
        
        Args:
            lineup_pool: Pool of lineups (pool_size, lineup_size)
            
        Returns:
            np.ndarray: Approximate similarity matrix (pool_size, pool_size)
        """
        pool_size = len(lineup_pool)
        
        # Initialize with zeros (will compute similarities on-demand)
        similarity_matrix = np.zeros((pool_size, pool_size), dtype=np.float32)
        
        # Set diagonal to 1.0 (lineup compared to itself)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _sample_diverse_set_indices(self, 
                                   similarity_matrix: np.ndarray,
                                   target_lineups: int,
                                   diversity_threshold: float) -> list:
        """
        Sample indices for a diverse set of lineups
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            target_lineups: Number of lineups to select
            diversity_threshold: Minimum diversity required
            
        Returns:
            List of selected indices
        """
        pool_size = len(similarity_matrix)
        
        if target_lineups >= pool_size:
            return list(range(pool_size))
        
        selected_indices = []
        available_indices = list(range(pool_size))
        
        # Start with a random lineup
        first_idx = np.random.choice(available_indices)
        selected_indices.append(first_idx)
        available_indices.remove(first_idx)
        
        # For large pools with sampled similarity matrix, use greedy approach
        if pool_size > 50000:
            return self._sample_diverse_set_greedy(
                pool_size, target_lineups, selected_indices, available_indices
            )
        
        # Select remaining lineups to maximize diversity
        for _ in range(target_lineups - 1):
            if not available_indices:
                break
            
            best_candidate = None
            best_min_similarity = 1.0
            
            # Find the candidate with minimum similarity to selected lineups
            for candidate_idx in available_indices:
                max_similarity = 0.0
                
                for selected_idx in selected_indices:
                    similarity = similarity_matrix[candidate_idx, selected_idx]
                    max_similarity = max(max_similarity, similarity)
                
                # If this candidate is more diverse, keep it
                if max_similarity < best_min_similarity:
                    best_candidate = candidate_idx
                    best_min_similarity = max_similarity
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                available_indices.remove(best_candidate)
            else:
                # Fallback: random selection
                random_idx = np.random.choice(available_indices)
                selected_indices.append(random_idx)
                available_indices.remove(random_idx)
        
        return selected_indices
    
    @staticmethod
    def _sample_diverse_set_greedy(pool_size: int,
                                 target_lineups: int,
                                 selected_indices: list,
                                 available_indices: list) -> list:
        """
        Greedy sampling for large pools where we can't compute full similarity matrix
        
        Args:
            pool_size: Size of the lineup pool
            target_lineups: Number of lineups to select
            selected_indices: Already selected indices
            available_indices: Available indices to choose from
            
        Returns:
            List of selected indices
        """
        # For large pools, use a simpler random sampling with some diversity checks
        sample_size = min(target_lineups * 10, len(available_indices))
        
        while len(selected_indices) < target_lineups and available_indices:
            # Sample a subset of candidates
            candidates = np.random.choice(
                available_indices, 
                size=min(sample_size, len(available_indices)), 
                replace=False
            )
            
            # Pick the first candidate (could be enhanced with diversity checks)
            selected_idx = candidates[0]
            selected_indices.append(selected_idx)
            available_indices.remove(selected_idx)
        
        return selected_indices
    
    @staticmethod
    def _jaccard_similarity(lineup1: np.ndarray, lineup2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two lineups"""
        set1 = set(lineup1)
        set2 = set(lineup2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
