# pangadfs/fitness_multi_objective.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Dict, Any, Tuple
import numpy as np
from pangadfs.base import FitnessBase

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class FitnessMultiObjective(FitnessBase):
    """Multi-objective fitness evaluation for lineup sets
    
    Optimizes three objectives:
    1. Sum of top K lineups (quality of best lineups)
    2. Sum of all lineups (overall set quality)
    3. Diversity score (variation across lineups)
    """

    def fitness(self, 
                population_sets: np.ndarray,
                points: np.ndarray,
                top_k: int = 10,
                diversity_method: str = 'jaccard',
                weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
                **kwargs) -> np.ndarray:
        """
        Calculate multi-objective fitness for each lineup set
        
        Args:
            population_sets: Shape (pop_size, target_lineups, lineup_size)
            points: Points array for all players
            top_k: Number of top lineups to consider for first objective
            diversity_method: Method for calculating diversity ('jaccard' or 'hamming')
            weights: Weights for (top_k_sum, total_sum, diversity) objectives
            
        Returns:
            np.ndarray: Combined fitness scores for each set (shape: pop_size,)
        """
        pop_size, target_lineups, lineup_size = population_sets.shape
        
        # Ensure top_k doesn't exceed target_lineups
        top_k = min(top_k, target_lineups)
        
        # Calculate all three objectives
        top_k_scores = self._calculate_top_k_objective(population_sets, points, top_k)
        total_scores = self._calculate_total_objective(population_sets, points)
        diversity_scores = self._calculate_diversity_objective(population_sets, diversity_method)
        
        # Normalize objectives to similar scales
        top_k_norm = self._normalize_scores(top_k_scores)
        total_norm = self._normalize_scores(total_scores)
        diversity_norm = self._normalize_scores(diversity_scores)
        
        # Combine objectives with weights
        w_top, w_total, w_diversity = weights
        combined_fitness = (
            w_top * top_k_norm + 
            w_total * total_norm + 
            w_diversity * diversity_norm
        )
        
        logging.info(f'Multi-objective fitness - Top-K avg: {np.mean(top_k_scores):.1f}, '
                    f'Total avg: {np.mean(total_scores):.1f}, '
                    f'Diversity avg: {np.mean(diversity_scores):.3f}')
        
        return combined_fitness
    
    def _calculate_top_k_objective(self, 
                                  population_sets: np.ndarray, 
                                  points: np.ndarray, 
                                  top_k: int) -> np.ndarray:
        """Calculate sum of points for top K lineups in each set"""
        pop_size, target_lineups, lineup_size = population_sets.shape
        top_k_scores = np.zeros(pop_size)
        
        for set_idx in range(pop_size):
            lineup_set = population_sets[set_idx]
            
            # Calculate points for each lineup in the set
            lineup_points = np.array([points[lineup].sum() for lineup in lineup_set])
            
            # Get top K lineups
            top_k_indices = np.argsort(lineup_points)[-top_k:]
            top_k_scores[set_idx] = lineup_points[top_k_indices].sum()
        
        return top_k_scores
    
    def _calculate_total_objective(self, 
                                  population_sets: np.ndarray, 
                                  points: np.ndarray) -> np.ndarray:
        """Calculate sum of points for all lineups in each set"""
        # Vectorized calculation for efficiency
        lineup_points = points[population_sets]
        total_scores = np.sum(lineup_points, axis=(1, 2))
        return total_scores
    
    def _calculate_diversity_objective(self, 
                                     population_sets: np.ndarray,
                                     diversity_method: str) -> np.ndarray:
        """Calculate diversity score for each set (higher = more diverse)"""
        pop_size, target_lineups, lineup_size = population_sets.shape
        diversity_scores = np.zeros(pop_size)
        
        for set_idx in range(pop_size):
            lineup_set = population_sets[set_idx]
            diversity_scores[set_idx] = self._calculate_set_diversity_score(
                lineup_set, diversity_method
            )
        
        return diversity_scores
    
    def _calculate_set_diversity_score(self, 
                                     lineup_set: np.ndarray, 
                                     diversity_method: str) -> float:
        """Calculate diversity score for a single set (higher = more diverse)"""
        n_lineups = len(lineup_set)
        if n_lineups <= 1:
            return 1.0  # Maximum diversity for single lineup
        
        total_dissimilarity = 0.0
        n_pairs = 0
        
        if diversity_method == 'jaccard':
            # Convert to sets for faster operations
            lineup_sets = [set(lineup) for lineup in lineup_set]
            
            for i in range(n_lineups):
                for j in range(i + 1, n_lineups):
                    set1, set2 = lineup_sets[i], lineup_sets[j]
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    similarity = intersection / union if union > 0 else 0.0
                    dissimilarity = 1.0 - similarity  # Convert to dissimilarity
                    total_dissimilarity += dissimilarity
                    n_pairs += 1
        else:
            # Hamming dissimilarity
            for i in range(n_lineups):
                for j in range(i + 1, n_lineups):
                    similarity = np.mean(lineup_set[i] == lineup_set[j])
                    dissimilarity = 1.0 - similarity
                    total_dissimilarity += dissimilarity
                    n_pairs += 1
        
        # Average dissimilarity (0 = identical, 1 = completely different)
        avg_dissimilarity = total_dissimilarity / n_pairs if n_pairs > 0 else 0.0
        return avg_dissimilarity
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range"""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def get_detailed_metrics(self, 
                           population_sets: np.ndarray,
                           points: np.ndarray,
                           top_k: int = 10,
                           diversity_method: str = 'jaccard') -> Dict[str, np.ndarray]:
        """Get detailed metrics for analysis"""
        top_k_scores = self._calculate_top_k_objective(population_sets, points, top_k)
        total_scores = self._calculate_total_objective(population_sets, points)
        diversity_scores = self._calculate_diversity_objective(population_sets, diversity_method)
        
        return {
            'top_k_scores': top_k_scores,
            'total_scores': total_scores,
            'diversity_scores': diversity_scores,
            'top_k_normalized': self._normalize_scores(top_k_scores),
            'total_normalized': self._normalize_scores(total_scores),
            'diversity_normalized': self._normalize_scores(diversity_scores)
        }


class FitnessMultiObjectiveOptimized(FitnessMultiObjective):
    """Optimized version with numba acceleration where possible"""
    
    def _calculate_diversity_objective(self, 
                                     population_sets: np.ndarray,
                                     diversity_method: str) -> np.ndarray:
        """Optimized diversity calculation"""
        if NUMBA_AVAILABLE and diversity_method == 'jaccard':
            return self._calculate_diversity_numba(population_sets)
        else:
            return super()._calculate_diversity_objective(population_sets, diversity_method)
    
    def _calculate_diversity_numba(self, population_sets: np.ndarray) -> np.ndarray:
        """Numba-optimized diversity calculation"""
        return _diversity_scores_numba_impl(population_sets)


# Numba-optimized functions
if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _diversity_scores_numba_impl(population_sets):
        """Numba implementation of diversity score calculation"""
        pop_size, target_lineups, lineup_size = population_sets.shape
        diversity_scores = np.zeros(pop_size)
        
        for set_idx in prange(pop_size):
            lineup_set = population_sets[set_idx]
            
            if target_lineups <= 1:
                diversity_scores[set_idx] = 1.0
                continue
            
            total_dissimilarity = 0.0
            n_pairs = 0
            
            for i in range(target_lineups):
                for j in range(i + 1, target_lineups):
                    # Calculate Jaccard dissimilarity
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
                    dissimilarity = 1.0 - similarity
                    total_dissimilarity += dissimilarity
                    n_pairs += 1
            
            avg_dissimilarity = total_dissimilarity / n_pairs if n_pairs > 0 else 0.0
            diversity_scores[set_idx] = avg_dissimilarity
        
        return diversity_scores
