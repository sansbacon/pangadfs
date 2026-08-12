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
        """Calculates the mean pairwise diversity for each lineup set.
        
        Uses vectorized one-hot overlap matrix instead of nested Python loops.
        """
        n_sets = population_sets.shape[0]
        n_lineups = population_sets.shape[1]
        lineup_size = population_sets.shape[2]
        diversity_scores = np.zeros(n_sets)
        
        if n_lineups <= 1:
            return diversity_scores
        
        # Number of unique pairs per set
        n_pairs = n_lineups * (n_lineups - 1) // 2
        
        for i in range(n_sets):
            lineup_set = population_sets[i]  # shape (n_lineups, lineup_size)
            
            if method == 'jaccard':
                # Build one-hot membership matrix: (n_lineups, n_unique_players)
                uniques, inverse = np.unique(lineup_set, return_inverse=True)
                n_unique = len(uniques)
                membership = np.zeros((n_lineups, n_unique), dtype=np.uint8)
                rows = np.repeat(np.arange(n_lineups), lineup_size)
                np.add.at(membership, (rows, inverse), 1)
                # Clip to binary (presence/absence for Jaccard)
                membership = (membership > 0).astype(np.uint8)
                
                # Pairwise intersection and union via dot products
                intersection_matrix = membership @ membership.T  # (n_lineups, n_lineups)
                sizes = membership.sum(axis=1)  # per-lineup unique player count
                union_matrix = sizes[:, None] + sizes[None, :] - intersection_matrix
                
                # Extract upper triangle (unique pairs)
                triu_idx = np.triu_indices(n_lineups, k=1)
                intersections = intersection_matrix[triu_idx]
                unions = union_matrix[triu_idx]
                
                similarities = np.divide(intersections, unions, 
                                        out=np.zeros_like(intersections, dtype=float),
                                        where=unions > 0)
                diversity_scores[i] = (1.0 - similarities).mean()
            else:
                # Hamming: pairwise element-wise equality
                # overlap[j,k] = mean(lineup_j == lineup_k)
                # Vectorized: broadcast comparison
                eq_matrix = (lineup_set[:, None, :] == lineup_set[None, :, :]).mean(axis=2)
                triu_idx = np.triu_indices(n_lineups, k=1)
                overlaps = eq_matrix[triu_idx]
                diversity_scores[i] = (1.0 - overlaps).mean()
        
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
