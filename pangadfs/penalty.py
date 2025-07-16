# pangadfs/pangadfs/penalty.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

"""
# Penalty framework

Idea is to replace optimizer rules with penalties. Penalties can be negative (bad) or positive (good).
* Advantages 
    * Doesn't throw away reasonable options (125% ownership arbitrary) and is flexible.
    * Does not require absurdly complex optimizer rules.
    * Can easily layer penalties on top of each other.
* Disadvantages
    * Takes some fiddling to get the parameters correct.

# Possible penalties

* Individual ownership penalty (global or just high-owned)
* Cumulative ownership penalty (global or just high-owned)
* Distances (too many similar lineups)
* Diversity (another way of measuring too many similar lineups)
* Position combinations (QB vs DST, WR + own DST, etc.)
* Correlation-based penalties (QB-WR stacks, game stacks)
* Variance-based penalties (high/low variance based on tournament type)
* Position-specific penalties

"""

import functools
import logging
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from pangadfs.base import PenaltyBase

# Set up logging
logger = logging.getLogger(__name__)

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    logger.info("Numba is available for JIT compilation")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba not available, using fallback implementations")
    # Fallback if Numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Try to import scipy for optimized distance calculations
try:
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Profiling decorator
def profile(func):
    """Decorator to profile function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Caching mixin
class CachedPenaltyMixin:
    """Mixin that provides caching for penalty calculations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_enabled = True
        self._reset_cache()
    
    def _reset_cache(self):
        """Reset the calculation cache"""
        self._calculate_penalty = functools.lru_cache(maxsize=10000)(self._calculate_penalty)
    
    def _calculate_penalty(self, lineup_tuple, *args, **kwargs):
        """Actual penalty calculation (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def disable_cache(self):
        """Disable caching"""
        self._cache_enabled = False
    
    def enable_cache(self):
        """Enable caching"""
        self._cache_enabled = True
        self._reset_cache()

# JIT-compiled helper functions
if NUMBA_AVAILABLE:
    @njit
    def _calculate_one_hot_encoding(population, max_id):
        """JIT-compiled one-hot encoding calculation"""
        n_lineups = len(population)
        n_players = population.shape[1]
        
        # Create one-hot encoding
        ohe = np.zeros((n_lineups, max_id + 1), dtype=np.int32)
        for i in range(n_lineups):
            for j in range(n_players):
                player_id = population[i, j]
                ohe[i, player_id] = 1
                
        return ohe
    
    @njit
    def _calculate_diversity_matrix(ohe):
        """JIT-compiled diversity matrix calculation"""
        n_lineups = ohe.shape[0]
        diversity = np.zeros((n_lineups, n_lineups), dtype=np.float32)
        
        for i in range(n_lineups):
            for j in range(n_lineups):
                overlap = 0
                for k in range(ohe.shape[1]):
                    overlap += ohe[i, k] * ohe[j, k]
                diversity[i, j] = overlap
                
        return diversity
    
    @njit(parallel=True)
    def _calculate_ownership_penalties(ownership, threshold, penalty_factor, base):
        """JIT-compiled ownership penalty calculation"""
        n_players = len(ownership)
        penalties = np.zeros(n_players, dtype=np.float32)
        
        for i in prange(n_players):
            if ownership[i] > threshold:
                excess = ownership[i] - threshold
                penalties[i] = -penalty_factor * np.log(1 + excess * 10) / np.log(base)
                
        return penalties
    
    @njit(parallel=True)
    def _calculate_correlation_penalties(population, correlation_matrix, strength):
        """JIT-compiled correlation penalty calculation"""
        n_lineups = len(population)
        n_players = population.shape[1]
        penalties = np.zeros(n_lineups, dtype=np.float32)
        
        for i in prange(n_lineups):
            lineup = population[i]
            lineup_corr = 0.0
            
            for j in range(n_players):
                for k in range(j+1, n_players):
                    player1 = lineup[j]
                    player2 = lineup[k]
                    corr = correlation_matrix[player1, player2]
                    if corr > 0:
                        lineup_corr += corr
                        
            penalties[i] = -strength * lineup_corr
            
        return penalties


class DistancePenalty(PenaltyBase):
    """Penalty based on distance between lineups."""

    @profile
    def penalty(self, *, population: np.ndarray, robust: bool = True, 
                sigmoid_scale: float = 1.0, batch_size: int = 1000, **kwargs) -> np.ndarray:
        """Calculates distance penalty for overlapping lineups
        
        Args:
            population (np.ndarray): the population
            robust (bool): Whether to use robust normalization (default: True)
            sigmoid_scale (float): Scaling factor for sigmoid transformation
            batch_size (int): Size of batches for processing large populations
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: 1D array of float penalties

        """
        n_lineups = len(population)
        
        # For small populations, process all at once
        if n_lineups <= batch_size or batch_size <= 0:
            return self._calculate_distance_penalty(
                population, robust, sigmoid_scale, **kwargs)
        
        # For large populations, process in batches to save memory
        penalties = np.zeros(n_lineups)
        for i in range(0, n_lineups, batch_size):
            batch_end = min(i + batch_size, n_lineups)
            batch = population[i:batch_end]
            
            batch_penalties = self._calculate_distance_penalty(
                batch, robust, sigmoid_scale, **kwargs)
            
            penalties[i:batch_end] = batch_penalties
            
        return penalties
    
    def _calculate_distance_penalty(self, population: np.ndarray, 
                                   robust: bool, sigmoid_scale: float, **kwargs) -> np.ndarray:
        """Helper method to calculate distance penalty for a batch"""
        if SCIPY_AVAILABLE:
            # Use scipy's optimized distance calculation
            max_id = population.max()
            
            # Create one-hot encoding
            if NUMBA_AVAILABLE:
                ohe = _calculate_one_hot_encoding(population, max_id)
            else:
                ohe = np.zeros((len(population), max_id + 1), dtype=np.int32)
                for i, lineup in enumerate(population):
                    for player_id in lineup:
                        ohe[i, player_id] = 1
            
            # Calculate pairwise distances efficiently
            distances = squareform(pdist(ohe, 'euclidean'))
            mean_distances = np.mean(distances, axis=1)
        else:
            # Fallback to original implementation
            ohe = np.sum((np.arange(population.max()) == population[...,None]-1).astype(int), axis=1)
            b = ohe.reshape(ohe.shape[0], 1, ohe.shape[1])
            dist = np.sqrt(np.einsum('ijk, ijk->ij', ohe-b, ohe-b))
            mean_distances = np.mean(dist, axis=1)
        
        if robust:
            # Apply robust normalization using quantiles
            q25, q75 = np.percentile(mean_distances, [25, 75])
            iqr = q75 - q25 or 1.0  # Avoid division by zero
            normalized_dist = (mean_distances - q25) / iqr
            
            # Apply sigmoid transformation to handle outliers better
            return -1 * (1 / (1 + np.exp(-sigmoid_scale * normalized_dist)))
        else:
            # Original z-score normalization
            return 0 - ((mean_distances - mean_distances.mean()) / (mean_distances.std() or 1.0))


class DiversityPenalty(PenaltyBase):
    """Penalty based on diversity between lineups."""

    @profile
    def penalty(self, *, population: np.ndarray, robust: bool = True, 
                sigmoid_scale: float = 1.0, batch_size: int = 1000, **kwargs) -> np.ndarray:
        """Calculates diversity penalty for overlapping lineups
        
        Args:
            population (np.ndarray): the population
            robust (bool): Whether to use robust normalization (default: True)
            sigmoid_scale (float): Scaling factor for sigmoid transformation
            batch_size (int): Size of batches for processing large populations
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: 1D array of float penalties

        """
        n_lineups = len(population)
        
        # For small populations, process all at once
        if n_lineups <= batch_size or batch_size <= 0:
            return self._calculate_diversity_penalty(
                population, robust, sigmoid_scale, **kwargs)
        
        # For large populations, process in batches to save memory
        penalties = np.zeros(n_lineups)
        for i in range(0, n_lineups, batch_size):
            batch_end = min(i + batch_size, n_lineups)
            batch = population[i:batch_end]
            
            batch_penalties = self._calculate_diversity_penalty(
                batch, robust, sigmoid_scale, **kwargs)
            
            penalties[i:batch_end] = batch_penalties
            
        return penalties
    
    def _calculate_diversity_penalty(self, population: np.ndarray, 
                                    robust: bool, sigmoid_scale: float, **kwargs) -> np.ndarray:
        """Helper method to calculate diversity penalty for a batch"""
        # Optimized diversity calculation
        max_id = population.max()
        
        # Create one-hot encoding
        if NUMBA_AVAILABLE:
            ohe = _calculate_one_hot_encoding(population, max_id)
            diversity_matrix = _calculate_diversity_matrix(ohe)
        else:
            uniques = np.unique(population)
            a = (population[..., None] == uniques).sum(1)
            diversity_matrix = np.einsum('ij,kj->ik', a, a)
            
        diversity = np.sum(diversity_matrix, axis=1) / population.size
        
        if robust:
            # Apply robust normalization using quantiles
            q25, q75 = np.percentile(diversity, [25, 75])
            iqr = q75 - q25 or 1.0  # Avoid division by zero
            normalized_diversity = (diversity - q25) / iqr
            
            # Apply sigmoid transformation to handle outliers better
            return -1 * (1 / (1 + np.exp(-sigmoid_scale * normalized_diversity)))
        else:
            # Original z-score normalization
            return 0 - ((diversity - diversity.mean()) / (diversity.std() or 1.0))


class OwnershipPenalty(CachedPenaltyMixin, PenaltyBase):
    """Penalty based on player ownership percentages."""

    def __init__(self, ctx=None):
        PenaltyBase.__init__(self, ctx)
        CachedPenaltyMixin.__init__(self)

    @profile
    def penalty(self, *, ownership: np.ndarray, base: float = 3, boost: float = 2, 
                min_penalty: float = -5.0, max_penalty: float = 0.0, **kwargs) -> np.ndarray:
        """Calculates penalties that are inverse to projected ownership
        
        Args:
            ownership (np.ndarray): 1D array of ownership percentages (0-1)
            base (float): the logarithm base, default 3
            boost (float): the constant to boost low-owned players
            min_penalty (float): Minimum penalty value
            max_penalty (float): Maximum penalty value
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
            
        """
        # Vectorized calculation without caching
        # Ensure ownership values are valid (between 0 and 1)
        ownership = np.clip(ownership, 0.01, 1.0)  # Avoid log(0)
        
        # Calculate penalties
        penalties = 0 - np.log(ownership) / np.log(base) + boost
        
        # Clip penalties to specified range
        return np.clip(penalties, min_penalty, max_penalty)
    
    def _calculate_penalty(self, player_id, ownership, base, boost, min_penalty, max_penalty):
        """Cached penalty calculation for a single player"""
        # Ensure ownership value is valid
        ownership_val = max(0.01, min(1.0, ownership[player_id]))
        
        # Calculate penalty
        penalty = 0 - np.log(ownership_val) / np.log(base) + boost
        
        # Clip penalty to specified range
        return max(min_penalty, min(max_penalty, penalty))


class HighOwnershipPenalty(PenaltyBase):
    """Penalty that only applies to players with high ownership."""

    @profile
    def penalty(self, *, ownership: np.ndarray, threshold: float = 0.15, 
                penalty_factor: float = 2.0, base: float = 3, **kwargs) -> np.ndarray:
        """Calculates penalties only for highly-owned players
        
        Args:
            ownership (np.ndarray): 1D array of ownership percentages (0-1)
            threshold (float): Ownership threshold above which penalties apply
            penalty_factor (float): Multiplier for penalty severity
            base (float): Logarithm base for penalty calculation
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
        """
        # Use JIT-compiled function if available
        if NUMBA_AVAILABLE:
            return _calculate_ownership_penalties(ownership, threshold, penalty_factor, base)
        
        # Fallback implementation
        # Initialize penalties to zero
        penalties = np.zeros_like(ownership, dtype=float)
        
        # Apply penalties only to players above the threshold
        high_owned_mask = ownership > threshold
        if np.any(high_owned_mask):
            # Calculate how much above threshold
            excess = ownership[high_owned_mask] - threshold
            
            # Apply logarithmic penalty that increases more rapidly as ownership increases
            penalties[high_owned_mask] = -penalty_factor * np.log(1 + excess * 10) / np.log(base)
            
        return penalties


class CorrelationPenalty(PenaltyBase):
    """Penalty based on statistical correlations between players."""

    @profile
    def penalty(self, *, population: np.ndarray, correlation_matrix: np.ndarray, 
                strength: float = 1.0, position_pairs: List[Tuple[str, str]] = None, 
                early_termination: bool = True, **kwargs) -> np.ndarray:
        """Applies penalties based on statistical correlations between players
        
        Args:
            population (np.ndarray): the population
            correlation_matrix (np.ndarray): Matrix of player correlations
            strength (float): Penalty strength multiplier
            position_pairs (list): List of position pairs to consider for correlation
                                  e.g. [('QB', 'WR'), ('QB', 'TE')]
            early_termination (bool): Whether to use early termination optimization
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
        """
        # Use JIT-compiled function if available
        if NUMBA_AVAILABLE:
            return _calculate_correlation_penalties(population, correlation_matrix, strength)
        
        # Fallback implementation with vectorization
        n_lineups = len(population)
        penalties = np.zeros(n_lineups)
        
        # Create a mask for upper triangle to avoid counting correlations twice
        lineup_size = population.shape[1]
        mask = np.triu(np.ones((lineup_size, lineup_size)), k=1).astype(bool)
        
        for i in range(n_lineups):
            # Get correlation submatrix for this lineup
            lineup_indices = population[i]
            
            # Early termination check - if all indices are the same, correlation is perfect
            if early_termination and len(np.unique(lineup_indices)) == 1:
                penalties[i] = -strength * correlation_matrix[lineup_indices[0], lineup_indices[0]]
                continue
                
            corr_submatrix = correlation_matrix[np.ix_(lineup_indices, lineup_indices)]
            
            # Sum only positive correlations in upper triangle
            positive_corrs = np.maximum(0, corr_submatrix[mask])
            penalties[i] = -strength * np.sum(positive_corrs)
        
        return penalties


class StackingReward(PenaltyBase):
    """Reward (negative penalty) for strategic stacking of correlated positions."""

    @profile
    def penalty(self, *, population: np.ndarray, player_teams: np.ndarray, 
                player_positions: np.ndarray, stack_configs: Dict[str, float] = None, 
                **kwargs) -> np.ndarray:
        """Provides rewards (negative penalties) for strategic position stacking
        
        Args:
            population (np.ndarray): the population
            player_teams (np.ndarray): Array mapping player indices to team indices
            player_positions (np.ndarray): Array mapping player indices to position indices
            stack_configs (dict): Configuration for different stack types and their rewards
                e.g. {'QB-WR': 2.0, 'QB-TE': 1.5, 'QB-WR-WR': 3.0}
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of rewards (negative penalties)
        """
        if stack_configs is None:
            stack_configs = {'QB-WR': 2.0, 'QB-TE': 1.5, 'QB-WR-WR': 3.0}
            
        n_lineups = len(population)
        rewards = np.zeros(n_lineups)
        
        # Pre-compute position masks for vectorization
        qb_mask = player_positions == 'QB'
        wr_mask = player_positions == 'WR'
        te_mask = player_positions == 'TE'
        
        for i in range(n_lineups):
            lineup = population[i]
            
            # Get teams and positions in this lineup
            teams = player_teams[lineup]
            
            # Find QB indices
            qb_indices = np.where(qb_mask[lineup])[0]
            
            for qb_idx in qb_indices:
                qb_team = teams[qb_idx]
                
                # Count WRs from same team (vectorized)
                same_team_wrs = np.sum((teams == qb_team) & wr_mask[lineup])
                if same_team_wrs > 0 and 'QB-WR' in stack_configs:
                    rewards[i] += stack_configs['QB-WR'] * same_team_wrs
                
                # Count TEs from same team (vectorized)
                same_team_tes = np.sum((teams == qb_team) & te_mask[lineup])
                if same_team_tes > 0 and 'QB-TE' in stack_configs:
                    rewards[i] += stack_configs['QB-TE'] * same_team_tes
                    
                # Check for QB-WR-WR stack (double stack)
                if same_team_wrs >= 2 and 'QB-WR-WR' in stack_configs:
                    rewards[i] += stack_configs['QB-WR-WR']
            
        return rewards  # Note: rewards are negative penalties


class GameTheoryPenalty(PenaltyBase):
    """Penalty based on game theory concepts for tournament optimization."""

    @profile
    def penalty(self, *, population: np.ndarray, ownership: np.ndarray, 
                field_size: int, tournament_type: str = 'gpp',
                early_termination_threshold: float = 0.01, **kwargs) -> np.ndarray:
        """Applies game theory based penalties for tournament optimization
        
        Args:
            population (np.ndarray): the population
            ownership (np.ndarray): Projected ownership percentages
            field_size (int): Number of entries in the tournament
            tournament_type (str): 'gpp', 'double_up', 'h2h', etc.
            early_termination_threshold (float): Threshold for early termination
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
        """
        n_lineups = len(population)
        penalties = np.zeros(n_lineups)
        
        # Different penalty strategies based on tournament type
        if tournament_type == 'gpp':
            # For GPPs, penalize chalk (high ownership) lineups
            for i in range(n_lineups):
                lineup = population[i]
                
                # Calculate average ownership of the lineup
                lineup_ownership = np.mean(ownership[lineup])
                
                # Early termination: if ownership is very low, uniqueness will be high
                # and penalty will be negligible
                if lineup_ownership < early_termination_threshold:
                    penalties[i] = 0  # Effectively no penalty
                    continue
                
                # Calculate uniqueness factor (probability of unique lineup)
                # Based on binomial probability of no duplicates
                uniqueness = (1 - lineup_ownership) ** field_size
                
                # Apply penalty inversely proportional to uniqueness
                penalties[i] = -5.0 * (1 - uniqueness)
                
        elif tournament_type == 'cash':
            # For cash games, reward chalk lineups (vectorized)
            for i in range(n_lineups):
                lineup = population[i]
                # Calculate average ownership of the lineup
                lineup_ownership = np.mean(ownership[lineup])
                
                # Apply reward proportional to ownership
                penalties[i] = 2.0 * lineup_ownership
                
        return penalties


class VariancePenalty(PenaltyBase):
    """Penalty based on player variance, crucial for tournament strategy."""

    @profile
    def penalty(self, *, population: np.ndarray, player_variance: np.ndarray, 
                tournament_type: str = 'gpp', strength: float = 1.0, **kwargs) -> np.ndarray:
        """Applies penalties based on lineup variance
        
        Args:
            population (np.ndarray): the population
            player_variance (np.ndarray): Variance estimates for each player
            tournament_type (str): 'gpp', 'double_up', 'h2h', etc.
            strength (float): Penalty strength multiplier
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
        """
        n_lineups = len(population)
        penalties = np.zeros(n_lineups)
        
        # Vectorized calculation
        lineup_variances = np.zeros(n_lineups)
        for i in range(n_lineups):
            # Calculate total variance of the lineup
            lineup_variances[i] = np.sum(player_variance[population[i]])
        
        if tournament_type == 'gpp':
            # For GPPs, reward high variance
            penalties = -strength * lineup_variances
        elif tournament_type == 'cash':
            # For cash games, penalize high variance
            penalties = strength * lineup_variances
                
        return penalties


class PositionSpecificPenalty(PenaltyBase):
    """Penalty that applies different weights based on player positions."""

    @profile
    def penalty(self, *, population: np.ndarray, player_positions: np.ndarray, 
                position_weights: Dict[str, float] = None, **kwargs) -> np.ndarray:
        """Applies different penalties based on player positions
        
        Args:
            population (np.ndarray): the population
            player_positions (np.ndarray): Array mapping player indices to positions
            position_weights (dict): Weights for each position, e.g. {'QB': 2.0, 'WR': 1.5}
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
        """
        if position_weights is None:
            position_weights = {'QB': 1.0, 'RB': 1.0, 'WR': 1.0, 'TE': 1.0, 'DST': 1.0}
            
        n_lineups = len(population)
        penalties = np.zeros(n_lineups)
        
        # Pre-compute position weights array for vectorization
        unique_positions = np.unique(player_positions)
        position_weight_map = np.zeros(len(player_positions))
        
        for pos in unique_positions:
            if pos in position_weights:
                position_weight_map[player_positions == pos] = position_weights[pos]
        
        # Vectorized calculation
        for i in range(n_lineups):
            lineup = population[i]
            penalties[i] = np.sum(position_weight_map[lineup])
            
        # Normalize penalties
        if len(penalties) > 0 and np.std(penalties) > 0:
            penalties = (penalties - np.mean(penalties)) / np.std(penalties)
            
        return -penalties  # Convert to penalties (negative values)


class AdaptivePenaltyScaler(PenaltyBase):
    """Scales penalties adaptively based on fitness distribution."""

    @profile
    def penalty(self, *, base_penalties: np.ndarray, fitness_scores: np.ndarray, 
                target_impact: float = 0.2, **kwargs) -> np.ndarray:
        """Scales penalties adaptively based on fitness distribution
        
        Args:
            base_penalties (np.ndarray): Raw penalty values
            fitness_scores (np.ndarray): Current fitness scores
            target_impact (float): Target impact of penalties on fitness (0-1)
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: Scaled penalties
        """
        # Calculate fitness statistics
        fitness_range = np.max(fitness_scores) - np.min(fitness_scores)
        
        # Calculate current penalty statistics
        penalty_std = np.std(base_penalties)
        
        if penalty_std == 0 or fitness_range == 0:
            return base_penalties
            
        # Calculate target scale to achieve desired impact
        target_scale = (fitness_range * target_impact) / penalty_std
        
        # Apply scaling
        return base_penalties * target_scale


class OptimizedEnsemblePenalty(PenaltyBase):
    """Optimized version that combines multiple penalties with learned weights."""

    def __init__(self, ctx=None, penalty_classes=None, weights=None):
        """Initialize with multiple penalty classes and weights
        
        Args:
            ctx: Optional context dictionary
            penalty_classes (list): List of penalty classes to use
            weights (dict): Weights for each penalty class
        """
        super().__init__(ctx)
        self.penalty_classes = penalty_classes or []
        self.weights = weights or {cls.__name__: 1.0 for cls in self.penalty_classes}
        self.penalty_instances = [cls(ctx) for cls in self.penalty_classes]
        
    @profile
    def penalty(self, **kwargs) -> np.ndarray:
        """Applies an ensemble of penalties with optimized batch processing
        
        Args:
            **kwargs: Arguments to pass to each penalty class
            
        Returns:
            np.ndarray: 1D array of combined penalties
        """
        population = kwargs.get('population', np.array([]))
        if not self.penalty_instances or len(population) == 0:
            return np.zeros(len(population))
            
        # Group penalty instances by type for batch processing
        penalty_groups = {}
        for i, instance in enumerate(self.penalty_instances):
            cls_name = self.penalty_classes[i].__name__
            if cls_name not in penalty_groups:
                penalty_groups[cls_name] = []
            penalty_groups[cls_name].append((instance, self.weights.get(cls_name, 1.0)))
        
        # Initialize penalties array
        penalties = np.zeros(len(population))
        
        # Process each penalty group
        for cls_name, instances in penalty_groups.items():
            try:
                # Calculate penalty once per group and apply weights
                if len(instances) == 1:
                    instance, weight = instances[0]
                    penalties += weight * instance.penalty(**kwargs)
                else:
                    # For multiple instances of same type, calculate once and apply different weights
                    for instance, weight in instances:
                        penalties += weight * instance.penalty(**kwargs)
                        
                logger.debug(f"Applied {cls_name} penalties")
            except Exception as e:
                logger.error(f"Error applying {cls_name} penalties: {str(e)}")
                
        return penalties


# For backward compatibility
EnsemblePenalty = OptimizedEnsemblePenalty
