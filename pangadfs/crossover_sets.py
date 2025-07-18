# pangadfs/crossover_sets.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict, Any
import numpy as np
from pangadfs.base import CrossoverBase


class CrossoverMultilineupSets(CrossoverBase):
    """Crossover operation for lineup sets using tournament selection within sets"""

    def crossover(self, 
                  population_sets: np.ndarray,
                  method: str = 'tournament_within_sets',
                  tournament_size: int = 3,
                  crossover_rate: float = 0.8,
                  **kwargs) -> np.ndarray:
        """
        Perform crossover on lineup sets
        
        Args:
            population_sets: Shape (pop_size, target_lineups, lineup_size)
            method: Crossover method ('tournament_within_sets')
            tournament_size: Size of tournament for selecting lineups within sets
            crossover_rate: Probability of crossover occurring
            
        Returns:
            np.ndarray: New population after crossover
        """
        pop_size, target_lineups, lineup_size = population_sets.shape
        new_population = np.zeros_like(population_sets)
        
        # Process pairs of parents
        for i in range(0, pop_size - 1, 2):
            parent1 = population_sets[i]
            parent2 = population_sets[i + 1] if i + 1 < pop_size else population_sets[0]
            
            if np.random.random() < crossover_rate:
                # Perform crossover
                if method == 'tournament_within_sets':
                    child1, child2 = self._tournament_crossover(
                        parent1, parent2, tournament_size, target_lineups
                    )
                else:
                    # Default to tournament crossover
                    child1, child2 = self._tournament_crossover(
                        parent1, parent2, tournament_size, target_lineups
                    )
            else:
                # No crossover - copy parents
                child1, child2 = parent1.copy(), parent2.copy()
            
            new_population[i] = child1
            if i + 1 < pop_size:
                new_population[i + 1] = child2
        
        return new_population
    
    def _tournament_crossover(self, 
                            parent1: np.ndarray, 
                            parent2: np.ndarray,
                            tournament_size: int,
                            target_lineups: int) -> tuple:
        """
        Tournament-based crossover within sets
        
        Args:
            parent1: First parent set (target_lineups, lineup_size)
            parent2: Second parent set (target_lineups, lineup_size)
            tournament_size: Size of tournament for lineup selection
            target_lineups: Number of lineups per set
            
        Returns:
            Tuple of two child sets
        """
        lineup_size = parent1.shape[1]
        child1 = np.zeros((target_lineups, lineup_size), dtype=int)
        child2 = np.zeros((target_lineups, lineup_size), dtype=int)
        
        # For each position in the child sets
        for pos in range(target_lineups):
            # Tournament selection from parent1 for child1
            selected_lineup1 = self._tournament_select_lineup(parent1, tournament_size)
            child1[pos] = selected_lineup1
            
            # Tournament selection from parent2 for child2  
            selected_lineup2 = self._tournament_select_lineup(parent2, tournament_size)
            child2[pos] = selected_lineup2
        
        # Ensure diversity within each child set by replacing duplicates
        child1 = self._ensure_set_diversity(child1, parent1, parent2)
        child2 = self._ensure_set_diversity(child2, parent1, parent2)
        
        return child1, child2
    
    @staticmethod
    def _tournament_select_lineup(parent_set: np.ndarray, 
                                tournament_size: int) -> np.ndarray:
        """
        Select a lineup from parent set using tournament selection
        
        Args:
            parent_set: Parent lineup set (target_lineups, lineup_size)
            tournament_size: Size of tournament
            
        Returns:
            Selected lineup
        """
        n_lineups = len(parent_set)
        tournament_size = min(tournament_size, n_lineups)
        
        # Randomly select tournament participants
        tournament_indices = np.random.choice(n_lineups, tournament_size, replace=False)
        
        # For simplicity, just return a random selection from tournament
        # In a more sophisticated version, we could evaluate lineup quality
        selected_idx = np.random.choice(tournament_indices)
        
        return parent_set[selected_idx].copy()
    
    def _ensure_set_diversity(self, 
                            child_set: np.ndarray,
                            parent1: np.ndarray,
                            parent2: np.ndarray) -> np.ndarray:
        """
        Ensure diversity within a child set by replacing highly similar lineups
        
        Args:
            child_set: Child set to check for diversity
            parent1: First parent set (for replacement options)
            parent2: Second parent set (for replacement options)
            
        Returns:
            Child set with improved diversity
        """
        target_lineups, lineup_size = child_set.shape
        similarity_threshold = 0.8  # If similarity > 80%, consider replacing
        
        # Check each pair of lineups in the child set
        for i in range(target_lineups):
            for j in range(i + 1, target_lineups):
                similarity = self._jaccard_similarity(child_set[i], child_set[j])
                
                if similarity > similarity_threshold:
                    # Replace one of the similar lineups
                    # Try to find a more diverse replacement from parents
                    replacement_found = False
                    
                    # Try lineups from both parents
                    all_parent_lineups = np.vstack([parent1, parent2])
                    
                    for replacement_lineup in all_parent_lineups:
                        # Check if this replacement is more diverse
                        sim_to_i = self._jaccard_similarity(replacement_lineup, child_set[i])
                        sim_to_j = self._jaccard_similarity(replacement_lineup, child_set[j])
                        
                        if sim_to_i < similarity_threshold and sim_to_j < similarity_threshold:
                            # Replace the second lineup (j) with this more diverse option
                            child_set[j] = replacement_lineup.copy()
                            replacement_found = True
                            break
                    
                    # If no good replacement found, just use a random parent lineup
                    if not replacement_found:
                        random_parent_idx = np.random.randint(len(all_parent_lineups))
                        child_set[j] = all_parent_lineups[random_parent_idx].copy()
        
        return child_set
    
    @staticmethod
    def _jaccard_similarity(lineup1: np.ndarray, lineup2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two lineups"""
        set1 = set(lineup1)
        set2 = set(lineup2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
