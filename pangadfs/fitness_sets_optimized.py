# pangadfs/fitness_sets_optimized.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict, Any
import numpy as np
from pangadfs.base import FitnessBase

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class FitnessMultilineupSetsOptimized(FitnessBase):
    """Ultra-fast fitness evaluation for lineup sets with vectorized operations"""

    def fitness(self, 
                population_sets: np.ndarray,
                points: np.ndarray,
                diversity_weight: float = 0.3,
                diversity_method: str = 'jaccard',
                **kwargs) -> np.ndarray:
        """
        Calculate fitness for each lineup set using vectorized operations
        
        Args:
            population_sets: Shape (pop_size, target_lineups, lineup_size)
            points: Points array for all players
            diversity_weight: Weight for diversity penalty (0-1)
            diversity_method: Method for calculating diversity ('jaccard' or 'hamming')
            
        Returns:
            np.ndarray: Fitness scores for each set (shape: pop_size,)
        """
        pop_size, target_lineups, lineup_size = population_sets.shape
        
        # Vectorized total points calculation for all sets at once
        total_points = self._calculate_total_points_vectorized(population_sets, points)
        
        # Vectorized diversity penalty calculation
        if diversity_weight > 0:
            diversity_penalties = self._calculate_diversity_penalties_vectorized(
                population_sets, diversity_method
            )
            fitness_scores = total_points - (diversity_weight * diversity_penalties)
        else:
            # Skip diversity calculation if weight is 0
            fitness_scores = total_points
        
        return fitness_scores
    
    @staticmethod
    def _calculate_total_points_vectorized(population_sets: np.ndarray, 
                                         points: np.ndarray) -> np.ndarray:
        """Vectorized calculation of total points for all sets"""
        # Use advanced indexing to get points for all lineups at once
        # Shape: (pop_size, target_lineups, lineup_size) -> (pop_size, target_lineups)
        lineup_points = points[population_sets]
        
        # Sum across lineup_size dimension, then across target_lineups dimension
        set_points = np.sum(lineup_points, axis=(1, 2))
        
        return set_points
    
    def _calculate_diversity_penalties_vectorized(self, 
                                                population_sets: np.ndarray,
                                                diversity_method: str) -> np.ndarray:
        """Vectorized diversity penalty calculation for all sets"""
        pop_size, target_lineups, lineup_size = population_sets.shape
        penalties = np.zeros(pop_size)
        
        if NUMBA_AVAILABLE and diversity_method == 'jaccard':
            # Use numba-optimized version for speed
            penalties = self._calculate_jaccard_penalties_numba(population_sets)
        else:
            # Use optimized numpy version
            for set_idx in range(pop_size):
                penalties[set_idx] = self._calculate_set_diversity_penalty_fast(
                    population_sets[set_idx], diversity_method
                )
        
        return penalties
    
    @staticmethod
    def _calculate_set_diversity_penalty_fast(lineup_set: np.ndarray, 
                                            diversity_method: str) -> float:
        """Fast diversity penalty calculation for a single set"""
        n_lineups = len(lineup_set)
        if n_lineups <= 1:
            return 0.0
        
        total_similarity = 0.0
        n_pairs = 0
        
        # Pre-convert to sets for faster intersection/union operations
        if diversity_method == 'jaccard':
            lineup_sets = [set(lineup) for lineup in lineup_set]
            
            for i in range(n_lineups):
                for j in range(i + 1, n_lineups):
                    set1, set2 = lineup_sets[i], lineup_sets[j]
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    similarity = intersection / union if union > 0 else 0.0
                    total_similarity += similarity
                    n_pairs += 1
        else:
            # Hamming similarity - can be vectorized
            for i in range(n_lineups):
                for j in range(i + 1, n_lineups):
                    similarity = np.mean(lineup_set[i] == lineup_set[j])
                    total_similarity += similarity
                    n_pairs += 1
        
        # Convert to penalty (reduced scale for quality-first approach)
        avg_similarity = total_similarity / n_pairs if n_pairs > 0 else 0.0
        penalty_scale = 50.0  # Reduced from 1000.0 for quality-first optimization
        return avg_similarity * penalty_scale
    
    @staticmethod
    def _calculate_jaccard_penalties_numba(population_sets: np.ndarray) -> np.ndarray:
        """Numba-optimized Jaccard penalty calculation"""
        if not NUMBA_AVAILABLE:
            return np.zeros(len(population_sets))
        
        return _jaccard_penalties_numba_impl(population_sets)


# Numba-optimized functions
if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _jaccard_penalties_numba_impl(population_sets):
        """Numba implementation of Jaccard penalty calculation"""
        pop_size, target_lineups, lineup_size = population_sets.shape
        penalties = np.zeros(pop_size)
        
        for set_idx in prange(pop_size):
            lineup_set = population_sets[set_idx]
            total_similarity = 0.0
            n_pairs = 0
            
            for i in range(target_lineups):
                for j in range(i + 1, target_lineups):
                    # Calculate Jaccard similarity
                    intersection = 0
                    union_size = lineup_size * 2
                    
                    # Count intersections
                    for k in range(lineup_size):
                        for l in range(lineup_size):
                            if lineup_set[i, k] == lineup_set[j, l]:
                                intersection += 1
                                union_size -= 1
                                break
                    
                    similarity = intersection / union_size if union_size > 0 else 0.0
                    total_similarity += similarity
                    n_pairs += 1
            
            avg_similarity = total_similarity / n_pairs if n_pairs > 0 else 0.0
            penalties[set_idx] = avg_similarity * 50.0  # Reduced penalty scale
        
        return penalties
