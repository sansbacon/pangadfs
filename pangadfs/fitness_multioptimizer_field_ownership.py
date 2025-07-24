# pangadfs/fitness_multioptimizer_field_ownership.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
from pangadfs.base import FitnessBase


class FitnessMultiOptimizerFieldOwnership(FitnessBase):
    """
    Fitness calculator for multi-objective optimization with field ownership.
    """

    def fitness(self,
                population_sets: np.ndarray,
                points: np.ndarray,
                ownership: np.ndarray,
                top_k: int,
                diversity_method: str,
                weights: tuple,
                strategy: str) -> np.ndarray:
        """
        Calculates the multi-objective fitness for each lineup set.
        
        Args:
            population_sets (np.ndarray): The population of lineup sets.
            points (np.ndarray): The points for each player.
            ownership (np.ndarray): The ownership for each player.
            top_k (int): The number of top lineups to consider for the score component.
            diversity_method (str): The method to calculate diversity.
            weights (tuple): The weights for (score, diversity, ownership).
            strategy (str): The ownership strategy ('contrarian', 'leverage', 'balanced').

        Returns:
            np.ndarray: The fitness score for each lineup set.
        """
        score_weight, diversity_weight, ownership_weight = weights

        # Calculate score component
        lineup_scores = np.sum(points[population_sets], axis=2)
        top_k_scores = np.sum(np.sort(lineup_scores, axis=1)[:, -top_k:], axis=1)
        total_scores = np.sum(lineup_scores, axis=1)
        score_component = top_k_scores + total_scores

        # Calculate diversity component
        diversity_component = self._calculate_diversity(population_sets, diversity_method)

        # Calculate ownership component
        ownership_component = self._calculate_ownership(population_sets, ownership, strategy)

        # Normalize components
        score_norm = self._normalize(score_component)
        diversity_norm = self._normalize(diversity_component)
        ownership_norm = self._normalize(ownership_component)

        # Combined fitness
        fitness = (score_weight * score_norm +
                   diversity_weight * diversity_norm +
                   ownership_weight * ownership_norm)
        
        return fitness

    @staticmethod
    def _calculate_diversity(population_sets: np.ndarray, method: str) -> np.ndarray:
        """Calculates the diversity for each lineup set."""
        n_sets = population_sets.shape[0]
        diversity_scores = np.zeros(n_sets)
        for i in range(n_sets):
            lineup_set = population_sets[i]
            n_lineups = lineup_set.shape[0]
            if n_lineups <= 1:
                continue
            
            overlaps = []
            for j in range(n_lineups):
                for k in range(j + 1, n_lineups):
                    if method == 'jaccard':
                        set1, set2 = set(lineup_set[j]), set(lineup_set[k])
                        intersection = len(set1 & set2)
                        union = len(set1 | set2)
                        overlap = intersection / union if union > 0 else 0.0
                    else: # hamming
                        overlap = np.mean(lineup_set[j] == lineup_set[k])
                    overlaps.append(1 - overlap) # we want to maximize diversity (1 - overlap)
            diversity_scores[i] = np.mean(overlaps) if overlaps else 0
        return diversity_scores

    @staticmethod
    def _calculate_ownership(population_sets: np.ndarray, ownership: np.ndarray, strategy: str) -> np.ndarray:
        """Calculates the ownership score for each lineup set based on the strategy."""
        lineup_ownerships = np.sum(ownership[population_sets], axis=2)
        
        if strategy == 'contrarian':
            # Lower total ownership is better
            return -np.sum(lineup_ownerships, axis=1)
        elif strategy == 'leverage':
            # For now, same as contrarian. Can be expanded.
            return -np.sum(lineup_ownerships, axis=1)
        elif strategy == 'balanced':
            # Reward variance in ownership across the set
            return np.var(lineup_ownerships, axis=1)
        return np.zeros(population_sets.shape[0])

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """Min-max normalization."""
        if np.all(data == data[0]):
            return np.ones_like(data)
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
