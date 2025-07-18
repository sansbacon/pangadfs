# pangadfs/mutate_sets.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from pangadfs.base import MutateBase


class MutateMultilineupSets(MutateBase):
    """Mutation operation for lineup sets"""

    def mutate(self, 
               population_sets: np.ndarray,
               mutation_rate: float = 0.1,
               pospool: Dict[str, pd.DataFrame] = None,
               posmap: Dict[str, int] = None,
               probcol: str = 'prob',
               diversity_threshold: float = 0.3,
               max_attempts: int = 50,
               **kwargs) -> np.ndarray:
        """
        Mutate lineup sets by replacing lineups within sets
        
        Args:
            population_sets: Shape (pop_size, target_lineups, lineup_size)
            mutation_rate: Probability of mutating each set
            pospool: Dictionary of position DataFrames
            posmap: Position mapping
            probcol: Column name for probabilities
            diversity_threshold: Minimum diversity for replacement lineups
            max_attempts: Maximum attempts to generate diverse replacement
            
        Returns:
            np.ndarray: Mutated population
        """
        if pospool is None or posmap is None:
            logging.warning("pospool or posmap not provided for mutation, returning unchanged population")
            return population_sets
        
        pop_size, target_lineups, lineup_size = population_sets.shape
        mutated_population = population_sets.copy()
        
        # Pre-calculate position indices for efficiency
        pos_indices = self._calculate_position_indices(posmap)
        
        # Mutate each set with probability mutation_rate
        for set_idx in range(pop_size):
            if np.random.random() < mutation_rate:
                mutated_population[set_idx] = self._mutate_lineup_set(
                    population_sets[set_idx], pospool, pos_indices, probcol,
                    diversity_threshold, max_attempts
                )
        
        return mutated_population
    
    def _mutate_lineup_set(self, 
                          lineup_set: np.ndarray,
                          pospool: Dict[str, pd.DataFrame],
                          pos_indices: Dict[str, tuple],
                          probcol: str,
                          diversity_threshold: float,
                          max_attempts: int) -> np.ndarray:
        """
        Mutate a single lineup set by replacing some lineups
        
        Args:
            lineup_set: Single set of lineups (target_lineups, lineup_size)
            pospool: Dictionary of position DataFrames
            pos_indices: Position indices mapping
            probcol: Column name for probabilities
            diversity_threshold: Minimum diversity for replacement lineups
            max_attempts: Maximum attempts to generate diverse replacement
            
        Returns:
            Mutated lineup set
        """
        target_lineups, lineup_size = lineup_set.shape
        mutated_set = lineup_set.copy()
        
        # Determine how many lineups to replace (1-3 lineups typically)
        n_replacements = np.random.randint(1, min(4, target_lineups + 1))
        
        # Randomly select which lineups to replace
        replacement_indices = np.random.choice(
            target_lineups, n_replacements, replace=False
        )
        
        for replace_idx in replacement_indices:
            # Generate a new diverse lineup to replace the selected one
            new_lineup = self._generate_diverse_replacement(
                mutated_set, replace_idx, pospool, pos_indices, probcol,
                diversity_threshold, max_attempts
            )
            
            if new_lineup is not None:
                mutated_set[replace_idx] = new_lineup
        
        return mutated_set
    
    def _generate_diverse_replacement(self,
                                    lineup_set: np.ndarray,
                                    replace_idx: int,
                                    pospool: Dict[str, pd.DataFrame],
                                    pos_indices: Dict[str, tuple],
                                    probcol: str,
                                    diversity_threshold: float,
                                    max_attempts: int) -> np.ndarray:
        """
        Generate a diverse replacement lineup for the set
        
        Args:
            lineup_set: Current lineup set
            replace_idx: Index of lineup to replace
            pospool: Dictionary of position DataFrames
            pos_indices: Position indices mapping
            probcol: Column name for probabilities
            diversity_threshold: Minimum diversity required
            max_attempts: Maximum attempts to generate diverse lineup
            
        Returns:
            New diverse lineup or None if failed
        """
        # Get other lineups in the set (excluding the one being replaced)
        other_lineups = np.delete(lineup_set, replace_idx, axis=0)
        
        best_lineup = None
        best_diversity = 0.0
        
        # Try multiple attempts to generate a diverse lineup
        for attempt in range(max_attempts):
            candidate = self._generate_single_lineup(pospool, pos_indices, probcol)
            
            # Calculate minimum diversity to other lineups in the set
            min_diversity = self._calculate_min_diversity_to_set(candidate, other_lineups)
            
            # Keep track of the most diverse candidate
            if min_diversity > best_diversity:
                best_lineup = candidate
                best_diversity = min_diversity
                
                # If it meets our threshold, use it immediately
                if min_diversity >= diversity_threshold:
                    break
        
        return best_lineup
    
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
