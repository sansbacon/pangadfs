# pangadfs/fitness_sets.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Dict, Any
import numpy as np
from pangadfs.base import FitnessBase


class FitnessMultilineupSets(FitnessBase):
    """Evaluates fitness of lineup sets with diversity penalty"""

    def fitness(self, 
                population_sets: np.ndarray,
                points: np.ndarray,
                diversity_weight: float = 0.3,
                diversity_method: str = 'jaccard',
                **kwargs) -> np.ndarray:
        """
        Calculate fitness for each lineup set
        
        Args:
            population_sets: Shape (pop_size, target_lineups, lineup_size)
            points: Points array for all players
            diversity_weight: Weight for diversity penalty (0-1)
            diversity_method: Method for calculating diversity ('jaccard' or 'hamming')
            
        Returns:
            np.ndarray: Fitness scores for each set (shape: pop_size,)
        """
        pop_size, target_lineups, lineup_size = population_sets.shape
        fitness_scores = np.zeros(pop_size)
        
        for set_idx in range(pop_size):
            lineup_set = population_sets[set_idx]
            
            # Calculate total points for all lineups in the set
            total_points = self._calculate_total_points(lineup_set, points)
            
            # Calculate diversity penalty for the set
            diversity_penalty = self._calculate_diversity_penalty(
                lineup_set, diversity_method
            )
            
            # Final fitness = total_points - (diversity_weight * penalty)
            fitness_scores[set_idx] = total_points - (diversity_weight * diversity_penalty)
        
        return fitness_scores
    
    def _calculate_total_points(self, lineup_set: np.ndarray, points: np.ndarray) -> float:
        """Calculate total points for all lineups in a set"""
        total_points = 0.0
        
        for lineup in lineup_set:
            # Sum points for all players in this lineup
            lineup_points = points[lineup].sum()
            total_points += lineup_points
            
        return total_points
    
    def _calculate_diversity_penalty(self, 
                                   lineup_set: np.ndarray, 
                                   diversity_method: str) -> float:
        """
        Calculate diversity penalty for a lineup set
        Higher penalty = less diverse lineups
        """
        n_lineups = len(lineup_set)
        if n_lineups <= 1:
            return 0.0
        
        total_similarity = 0.0
        n_pairs = 0
        
        # Calculate pairwise similarities
        for i in range(n_lineups):
            for j in range(i + 1, n_lineups):
                if diversity_method == 'jaccard':
                    similarity = self._jaccard_similarity(lineup_set[i], lineup_set[j])
                elif diversity_method == 'hamming':
                    similarity = self._hamming_similarity(lineup_set[i], lineup_set[j])
                else:
                    similarity = self._jaccard_similarity(lineup_set[i], lineup_set[j])
                
                total_similarity += similarity
                n_pairs += 1
        
        # Average similarity across all pairs
        avg_similarity = total_similarity / n_pairs if n_pairs > 0 else 0.0
        
        # Convert to penalty: higher similarity = higher penalty
        # Scale by a factor to make penalty meaningful relative to points
        penalty_scale = 1000.0  # Adjust this based on typical point ranges
        diversity_penalty = avg_similarity * penalty_scale
        
        return diversity_penalty
    
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
