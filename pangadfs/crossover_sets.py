# pangadfs/crossover_sets_optimized.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict, Any
import numpy as np
from pangadfs.base import CrossoverBase

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class CrossoverMultilineupSets(CrossoverBase):
    """Optimized crossover for lineup sets with tournament selection"""

    def crossover(self, 
                  population_sets: np.ndarray,
                  fitness_scores: np.ndarray = None,
                  tournament_size: int = 3,
                  crossover_rate: float = 0.8,
                  **kwargs) -> np.ndarray:
        """
        Perform crossover on lineup sets using tournament selection
        
        Args:
            population_sets: Shape (pop_size, target_lineups, lineup_size)
            fitness_scores: Fitness scores for tournament selection
            tournament_size: Size of tournament for parent selection
            crossover_rate: Probability of crossover vs. direct copy
            
        Returns:
            np.ndarray: New population of lineup sets
        """
        pop_size, target_lineups, lineup_size = population_sets.shape
        
        if fitness_scores is None:
            # If no fitness scores provided, use random selection
            fitness_scores = np.random.random(pop_size)
        
        # Create new population
        new_population = np.zeros_like(population_sets)
        
        for i in range(pop_size):
            if np.random.random() < crossover_rate:
                # Select two parents using tournament selection
                parent1_idx = self._tournament_selection(fitness_scores, tournament_size)
                parent2_idx = self._tournament_selection(fitness_scores, tournament_size)
                
                # Perform set-level crossover
                new_population[i] = self._crossover_sets(
                    population_sets[parent1_idx], 
                    population_sets[parent2_idx]
                )
            else:
                # Direct copy (no crossover)
                parent_idx = self._tournament_selection(fitness_scores, tournament_size)
                new_population[i] = population_sets[parent_idx].copy()
        
        return new_population
    
    @staticmethod
    def _tournament_selection(fitness_scores: np.ndarray, tournament_size: int) -> int:
        """Select parent using tournament selection"""
        tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        tournament_fitness = fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return winner_idx
    
    def _crossover_sets(self, parent1_set: np.ndarray, parent2_set: np.ndarray) -> np.ndarray:
        """
        Crossover two lineup sets
        
        Uses multiple crossover strategies:
        1. Lineup-level crossover (swap entire lineups)
        2. Player-level crossover (swap players within lineups)
        3. Hybrid approach
        """
        target_lineups, lineup_size = parent1_set.shape
        
        # Choose crossover method
        crossover_method = np.random.choice(['lineup_swap', 'player_swap', 'hybrid'], p=[0.4, 0.4, 0.2])
        
        if crossover_method == 'lineup_swap':
            return self._lineup_level_crossover(parent1_set, parent2_set)
        elif crossover_method == 'player_swap':
            return self._player_level_crossover(parent1_set, parent2_set)
        else:
            return self._hybrid_crossover(parent1_set, parent2_set)
    
    @staticmethod
    def _lineup_level_crossover(parent1_set: np.ndarray, parent2_set: np.ndarray) -> np.ndarray:
        """Crossover by swapping entire lineups between sets"""
        target_lineups, lineup_size = parent1_set.shape
        child_set = parent1_set.copy()
        
        # Randomly select lineups to swap from parent2
        n_swaps = np.random.randint(1, max(2, target_lineups // 2))
        swap_indices = np.random.choice(target_lineups, n_swaps, replace=False)
        
        for idx in swap_indices:
            child_set[idx] = parent2_set[idx]
        
        return child_set
    
    @staticmethod
    def _player_level_crossover(parent1_set: np.ndarray, parent2_set: np.ndarray) -> np.ndarray:
        """Crossover by swapping players within corresponding lineups"""
        target_lineups, lineup_size = parent1_set.shape
        child_set = parent1_set.copy()
        
        # For each lineup, potentially swap some players
        for lineup_idx in range(target_lineups):
            if np.random.random() < 0.5:  # 50% chance to crossover this lineup
                # Uniform crossover at player level
                crossover_mask = np.random.random(lineup_size) < 0.5
                child_set[lineup_idx][crossover_mask] = parent2_set[lineup_idx][crossover_mask]
        
        return child_set
    
    def _hybrid_crossover(self, parent1_set: np.ndarray, parent2_set: np.ndarray) -> np.ndarray:
        """Hybrid approach combining lineup and player level crossover"""
        target_lineups, lineup_size = parent1_set.shape
        
        # Start with lineup-level crossover
        child_set = self._lineup_level_crossover(parent1_set, parent2_set)
        
        # Then apply some player-level modifications
        n_lineups_to_modify = np.random.randint(1, max(2, target_lineups // 3))
        modify_indices = np.random.choice(target_lineups, n_lineups_to_modify, replace=False)
        
        for lineup_idx in modify_indices:
            # Swap a few players
            n_players_to_swap = np.random.randint(1, max(2, lineup_size // 3))
            player_indices = np.random.choice(lineup_size, n_players_to_swap, replace=False)
            
            # Choose which parent to take players from
            source_parent = parent2_set if np.random.random() < 0.5 else parent1_set
            child_set[lineup_idx][player_indices] = source_parent[lineup_idx][player_indices]
        
        return child_set


class CrossoverMultilineupSetsNumba(CrossoverMultilineupSets):
    """Numba-optimized version for better performance"""
    
    def crossover(self, 
                  population_sets: np.ndarray,
                  fitness_scores: np.ndarray = None,
                  tournament_size: int = 3,
                  crossover_rate: float = 0.8,
                  **kwargs) -> np.ndarray:
        """Numba-optimized crossover"""
        if not NUMBA_AVAILABLE:
            return super().crossover(population_sets, fitness_scores, tournament_size, crossover_rate, **kwargs)
        
        pop_size, target_lineups, lineup_size = population_sets.shape
        
        if fitness_scores is None:
            fitness_scores = np.random.random(pop_size)
        
        # Use numba-optimized implementation
        return _crossover_sets_numba_impl(
            population_sets, fitness_scores, tournament_size, crossover_rate
        )


# Numba-optimized functions
if NUMBA_AVAILABLE:
    @njit
    def _tournament_selection_numba(fitness_scores, tournament_size):
        """Numba-optimized tournament selection"""
        pop_size = len(fitness_scores)
        tournament_indices = np.random.choice(pop_size, tournament_size)
        
        best_idx = tournament_indices[0]
        best_fitness = fitness_scores[best_idx]
        
        for i in range(1, tournament_size):
            idx = tournament_indices[i]
            if fitness_scores[idx] > best_fitness:
                best_fitness = fitness_scores[idx]
                best_idx = idx
        
        return best_idx
    
    @njit
    def _lineup_crossover_numba(parent1_set, parent2_set):
        """Numba-optimized lineup crossover"""
        target_lineups, lineup_size = parent1_set.shape
        child_set = parent1_set.copy()
        
        # Simple single-point crossover at lineup level
        crossover_point = np.random.randint(1, target_lineups)
        
        for i in range(crossover_point, target_lineups):
            for j in range(lineup_size):
                child_set[i, j] = parent2_set[i, j]
        
        return child_set
    
    @njit(parallel=True)
    def _crossover_sets_numba_impl(population_sets, fitness_scores, tournament_size, crossover_rate):
        """Numba-optimized crossover implementation"""
        pop_size, target_lineups, lineup_size = population_sets.shape
        new_population = np.zeros_like(population_sets)
        
        for i in prange(pop_size):
            if np.random.random() < crossover_rate:
                # Select parents
                parent1_idx = _tournament_selection_numba(fitness_scores, tournament_size)
                parent2_idx = _tournament_selection_numba(fitness_scores, tournament_size)
                
                # Crossover
                new_population[i] = _lineup_crossover_numba(
                    population_sets[parent1_idx], 
                    population_sets[parent2_idx]
                )
            else:
                # Direct copy
                parent_idx = _tournament_selection_numba(fitness_scores, tournament_size)
                new_population[i] = population_sets[parent_idx].copy()
        
        return new_population
