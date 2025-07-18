# pangadfs/mutate_sets_optimized.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict, Any
import numpy as np
from pangadfs.base import MutateBase

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class MutateMultilineupSetsOptimized(MutateBase):
    """Optimized mutation for lineup sets"""

    def mutate(self, 
               population_sets: np.ndarray,
               mutation_rate: float = 0.1,
               pospool: Dict = None,
               posmap: Dict = None,
               **kwargs) -> np.ndarray:
        """
        Mutate lineup sets with multiple strategies
        
        Args:
            population_sets: Shape (pop_size, target_lineups, lineup_size)
            mutation_rate: Probability of mutation
            pospool: Position pools for replacement players
            posmap: Position mapping
            
        Returns:
            np.ndarray: Mutated population sets
        """
        pop_size, target_lineups, lineup_size = population_sets.shape
        mutated_population = population_sets.copy()
        
        # Convert pospool to arrays for faster access
        if pospool is not None:
            pos_arrays = self._prepare_position_arrays(pospool, posmap)
        else:
            pos_arrays = None
        
        for set_idx in range(pop_size):
            if np.random.random() < mutation_rate:
                mutated_population[set_idx] = self._mutate_set(
                    population_sets[set_idx], pos_arrays, posmap
                )
        
        return mutated_population
    
    @staticmethod
    def _prepare_position_arrays(pospool: Dict, posmap: Dict) -> Dict:
        """Convert pospool to numpy arrays for faster access"""
        pos_arrays = {}
        
        for pos, df in pospool.items():
            if len(df) > 0:
                pos_arrays[pos] = df.index.values
            else:
                pos_arrays[pos] = np.array([])
        
        return pos_arrays
    
    def _mutate_set(self, 
                   lineup_set: np.ndarray, 
                   pos_arrays: Dict = None,
                   posmap: Dict = None) -> np.ndarray:
        """
        Mutate a single lineup set using multiple strategies
        """
        target_lineups, lineup_size = lineup_set.shape
        mutated_set = lineup_set.copy()
        
        # Choose mutation strategy
        mutation_strategies = ['player_swap', 'lineup_shuffle', 'position_replace', 'hybrid']
        strategy = np.random.choice(mutation_strategies, p=[0.4, 0.2, 0.3, 0.1])
        
        if strategy == 'player_swap':
            mutated_set = self._player_swap_mutation(mutated_set)
        elif strategy == 'lineup_shuffle':
            mutated_set = self._lineup_shuffle_mutation(mutated_set)
        elif strategy == 'position_replace':
            mutated_set = self._position_replace_mutation(mutated_set, pos_arrays, posmap)
        else:
            mutated_set = self._hybrid_mutation(mutated_set, pos_arrays, posmap)
        
        return mutated_set
    
    @staticmethod
    def _player_swap_mutation(lineup_set: np.ndarray) -> np.ndarray:
        """Swap players between lineups in the set"""
        target_lineups, lineup_size = lineup_set.shape
        
        if target_lineups < 2:
            return lineup_set
        
        # Select two random lineups
        lineup1_idx, lineup2_idx = np.random.choice(target_lineups, 2, replace=False)
        
        # Select random positions to swap
        n_swaps = np.random.randint(1, max(2, lineup_size // 3))
        swap_positions = np.random.choice(lineup_size, n_swaps, replace=False)
        
        # Perform swaps
        for pos in swap_positions:
            temp = lineup_set[lineup1_idx, pos]
            lineup_set[lineup1_idx, pos] = lineup_set[lineup2_idx, pos]
            lineup_set[lineup2_idx, pos] = temp
        
        return lineup_set
    
    @staticmethod
    def _lineup_shuffle_mutation(lineup_set: np.ndarray) -> np.ndarray:
        """Shuffle players within individual lineups"""
        target_lineups, lineup_size = lineup_set.shape
        
        # Select random lineups to shuffle
        n_lineups_to_shuffle = np.random.randint(1, max(2, target_lineups // 2))
        lineup_indices = np.random.choice(target_lineups, n_lineups_to_shuffle, replace=False)
        
        for lineup_idx in lineup_indices:
            # Shuffle a portion of the lineup
            n_positions_to_shuffle = np.random.randint(2, lineup_size)
            positions_to_shuffle = np.random.choice(lineup_size, n_positions_to_shuffle, replace=False)
            
            # Shuffle the selected positions
            shuffled_players = lineup_set[lineup_idx, positions_to_shuffle].copy()
            np.random.shuffle(shuffled_players)
            lineup_set[lineup_idx, positions_to_shuffle] = shuffled_players
        
        return lineup_set
    
    def _position_replace_mutation(self, 
                                  lineup_set: np.ndarray, 
                                  pos_arrays: Dict = None,
                                  posmap: Dict = None) -> np.ndarray:
        """Replace players with others from the same position"""
        if pos_arrays is None or posmap is None:
            return self._player_swap_mutation(lineup_set)
        
        target_lineups, lineup_size = lineup_set.shape
        
        # Select random lineups to mutate
        n_lineups_to_mutate = np.random.randint(1, max(2, target_lineups // 2))
        lineup_indices = np.random.choice(target_lineups, n_lineups_to_mutate, replace=False)
        
        for lineup_idx in lineup_indices:
            # Select random positions to replace
            n_positions_to_replace = np.random.randint(1, max(2, lineup_size // 3))
            position_indices = np.random.choice(lineup_size, n_positions_to_replace, replace=False)
            
            for pos_idx in position_indices:
                # Determine position type for this slot
                position_type = self._get_position_type_for_slot(pos_idx, posmap)
                
                if position_type in pos_arrays and len(pos_arrays[position_type]) > 0:
                    # Replace with random player from same position
                    new_player = np.random.choice(pos_arrays[position_type])
                    lineup_set[lineup_idx, pos_idx] = new_player
        
        return lineup_set
    
    def _hybrid_mutation(self, 
                        lineup_set: np.ndarray, 
                        pos_arrays: Dict = None,
                        posmap: Dict = None) -> np.ndarray:
        """Combine multiple mutation strategies"""
        # Apply player swap first
        lineup_set = self._player_swap_mutation(lineup_set)
        
        # Then apply position replacement with lower probability
        if np.random.random() < 0.5:
            lineup_set = self._position_replace_mutation(lineup_set, pos_arrays, posmap)
        
        return lineup_set
    
    @staticmethod
    def _get_position_type_for_slot(slot_idx: int, posmap: Dict) -> str:
        """Determine position type for a given slot index"""
        if posmap is None:
            return 'FLEX'  # Default fallback
        
        current_slot = 0
        for pos, count in posmap.items():
            if slot_idx < current_slot + count:
                return pos
            current_slot += count
        
        return 'FLEX'  # Fallback


class MutateMultilineupSetsNumba(MutateMultilineupSetsOptimized):
    """Numba-optimized version for better performance"""
    
    def mutate(self, 
               population_sets: np.ndarray,
               mutation_rate: float = 0.1,
               pospool: Dict = None,
               posmap: Dict = None,
               **kwargs) -> np.ndarray:
        """Numba-optimized mutation"""
        if not NUMBA_AVAILABLE:
            return super().mutate(population_sets, mutation_rate, pospool, posmap, **kwargs)
        
        pop_size, target_lineups, lineup_size = population_sets.shape
        
        # Use numba-optimized implementation for basic mutations
        return _mutate_sets_numba_impl(population_sets, mutation_rate)


# Numba-optimized functions
if NUMBA_AVAILABLE:
    @njit
    def _player_swap_mutation_numba(lineup_set):
        """Numba-optimized player swap mutation"""
        target_lineups, lineup_size = lineup_set.shape
        
        if target_lineups < 2:
            return lineup_set
        
        # Select two random lineups
        lineup1_idx = np.random.randint(0, target_lineups)
        lineup2_idx = np.random.randint(0, target_lineups)
        while lineup2_idx == lineup1_idx:
            lineup2_idx = np.random.randint(0, target_lineups)
        
        # Select random positions to swap
        n_swaps = max(1, lineup_size // 4)
        
        for _ in range(n_swaps):
            pos = np.random.randint(0, lineup_size)
            temp = lineup_set[lineup1_idx, pos]
            lineup_set[lineup1_idx, pos] = lineup_set[lineup2_idx, pos]
            lineup_set[lineup2_idx, pos] = temp
        
        return lineup_set
    
    @njit
    def _lineup_shuffle_mutation_numba(lineup_set):
        """Numba-optimized lineup shuffle mutation"""
        target_lineups, lineup_size = lineup_set.shape
        
        # Select a random lineup to shuffle
        lineup_idx = np.random.randint(0, target_lineups)
        
        # Simple shuffle of the entire lineup
        for i in range(lineup_size):
            j = np.random.randint(0, lineup_size)
            temp = lineup_set[lineup_idx, i]
            lineup_set[lineup_idx, i] = lineup_set[lineup_idx, j]
            lineup_set[lineup_idx, j] = temp
        
        return lineup_set
    
    @njit(parallel=True)
    def _mutate_sets_numba_impl(population_sets, mutation_rate):
        """Numba-optimized mutation implementation"""
        pop_size, target_lineups, lineup_size = population_sets.shape
        mutated_population = population_sets.copy()
        
        for set_idx in prange(pop_size):
            if np.random.random() < mutation_rate:
                # Choose mutation strategy
                strategy = np.random.randint(0, 2)
                
                if strategy == 0:
                    mutated_population[set_idx] = _player_swap_mutation_numba(
                        mutated_population[set_idx]
                    )
                else:
                    mutated_population[set_idx] = _lineup_shuffle_mutation_numba(
                        mutated_population[set_idx]
                    )
        
        return mutated_population
