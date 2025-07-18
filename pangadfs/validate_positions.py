# pangadfs/validate_positions.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

from typing import Dict
import numpy as np
import pandas as pd
from pangadfs.base import ValidateBase


class PositionValidate(ValidateBase):
    """Validates that lineups meet position requirements"""

    def validate(self, *, 
                 population: np.ndarray,
                 pool: pd.DataFrame,
                 posmap: Dict[str, int],
                 position_column: str = 'pos',
                 flex_positions: tuple = ('RB', 'WR', 'TE'),
                 **kwargs) -> np.ndarray:
        """
        Validates that each lineup meets position requirements
        
        Args:
            population: Array of lineups (population_size, lineup_size)
            pool: Player pool DataFrame with position information
            posmap: Position requirements (e.g., {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1})
            position_column: Column name for positions in pool
            flex_positions: Positions that can fill FLEX slots
            
        Returns:
            np.ndarray: Filtered population with only valid lineups
        """
        if len(population) == 0:
            return population
        
        # Get position information for all players
        player_positions = pool[position_column].to_dict()
        
        valid_lineups = []
        
        for lineup in population:
            if self._is_lineup_valid(lineup, player_positions, posmap, flex_positions):
                valid_lineups.append(lineup)
        
        if len(valid_lineups) == 0:
            # If no valid lineups, return empty array with correct shape
            return np.empty((0, population.shape[1]), dtype=population.dtype)
        
        return np.array(valid_lineups)
    
    @staticmethod
    def _is_lineup_valid(lineup: np.ndarray, player_positions: Dict[int, str], 
                        posmap: Dict[str, int], flex_positions: tuple) -> bool:
        """Check if a single lineup meets position requirements"""
        
        # Count positions in the lineup
        lineup_positions = {}
        for player_id in lineup:
            if player_id in player_positions:
                pos = player_positions[player_id]
                lineup_positions[pos] = lineup_positions.get(pos, 0) + 1
        
        # Check non-FLEX positions first
        for pos, required_count in posmap.items():
            if pos == 'FLEX':
                continue
                
            actual_count = lineup_positions.get(pos, 0)
            if actual_count < required_count:
                return False
            
            # Remove the required players from the count
            lineup_positions[pos] = actual_count - required_count
        
        # Check FLEX positions
        if 'FLEX' in posmap:
            flex_required = posmap['FLEX']
            flex_available = 0
            
            # Count available flex players
            for pos in flex_positions:
                flex_available += lineup_positions.get(pos, 0)
            
            if flex_available < flex_required:
                return False
        
        return True


class PositionValidateOptimized(ValidateBase):
    """Optimized version using vectorized operations where possible"""

    def validate(self, *, 
                 population: np.ndarray,
                 pool: pd.DataFrame,
                 posmap: Dict[str, int],
                 position_column: str = 'pos',
                 flex_positions: tuple = ('RB', 'WR', 'TE'),
                 **kwargs) -> np.ndarray:
        """
        Validates that each lineup meets position requirements using optimized approach
        """
        if len(population) == 0:
            return population
        
        # Create position mapping array for fast lookup
        max_player_id = max(pool.index.max(), population.max()) + 1
        position_array = np.full(max_player_id, '', dtype='U5')
        
        for player_id, pos in zip(pool.index, pool[position_column]):
            position_array[player_id] = pos
        
        # Vectorized validation
        valid_mask = np.array([
            self._is_lineup_valid_vectorized(lineup, position_array, posmap, flex_positions)
            for lineup in population
        ])
        
        return population[valid_mask]
    
    @staticmethod
    def _is_lineup_valid_vectorized(lineup: np.ndarray, position_array: np.ndarray,
                                  posmap: Dict[str, int], flex_positions: tuple) -> bool:
        """Vectorized validation for a single lineup"""
        
        # Get positions for all players in lineup
        lineup_pos = position_array[lineup]
        
        # Count each position
        unique_pos, counts = np.unique(lineup_pos, return_counts=True)
        pos_counts = dict(zip(unique_pos, counts))
        
        # Check non-FLEX requirements
        remaining_counts = pos_counts.copy()
        for pos, required in posmap.items():
            if pos == 'FLEX':
                continue
            
            actual = remaining_counts.get(pos, 0)
            if actual < required:
                return False
            
            # Subtract required from remaining
            remaining_counts[pos] = actual - required
        
        # Check FLEX requirements
        if 'FLEX' in posmap:
            flex_required = posmap['FLEX']
            flex_available = sum(remaining_counts.get(pos, 0) for pos in flex_positions)
            
            if flex_available < flex_required:
                return False
        
        return True
