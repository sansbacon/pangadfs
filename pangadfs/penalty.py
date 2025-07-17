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
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
from pangadfs.base import PenaltyBase

# Set up logging
logger = logging.getLogger(__name__)

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange, types
    from numba.typed import Dict as NumbaDict
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
    from scipy.sparse import csr_matrix
    SCIPY_AVAILABLE = True
    logger.info("SciPy is available for optimized calculations")
except ImportError:
    SCIPY_AVAILABLE = False
    logger.info("SciPy not available, using fallback implementations")

# Try to import sklearn for advanced calculations
try:
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    SKLEARN_AVAILABLE = True
    logger.info("Scikit-learn is available for advanced metrics")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.info("Scikit-learn not available, using fallback implementations")


@dataclass
class PenaltyStats:
    """Statistics for penalty calculations"""
    calculation_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    batch_count: int = 0
    memory_usage: float = 0.0


# Enhanced profiling decorator with statistics
def profile_enhanced(func):
    """Enhanced decorator to profile function execution time and collect statistics"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Update statistics if available
        if hasattr(self, '_stats'):
            self._stats.calculation_time += execution_time
        
        # Log slow operations
        if execution_time > 0.1:
            logger.warning(f"{func.__name__} took {execution_time:.4f} seconds (slow operation)")
        else:
            logger.debug(f"{func.__name__} took {execution_time:.4f} seconds")
        
        return result
    return wrapper


# Thread-safe caching mixin with enhanced features
class EnhancedCachedPenaltyMixin:
    """Enhanced mixin that provides thread-safe caching for penalty calculations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_enabled = True
        self._cache_lock = threading.RLock()
        self._stats = PenaltyStats()
        self._reset_cache()
    
    def _reset_cache(self):
        """Reset the calculation cache with thread safety"""
        with self._cache_lock:
            self._calculate_penalty = functools.lru_cache(maxsize=10000)(self._calculate_penalty_impl)
    
    def _calculate_penalty_impl(self, *args, **kwargs):
        """Actual penalty calculation (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def _calculate_penalty(self, *args, **kwargs):
        """Wrapper for cached penalty calculation with statistics"""
        if self._cache_enabled:
            with self._cache_lock:
                # Check if this is a cache hit or miss
                cache_info = self._calculate_penalty.cache_info()
                prev_hits = cache_info.hits
                
                result = self._calculate_penalty_impl(*args, **kwargs)
                
                # Update statistics
                new_cache_info = self._calculate_penalty.cache_info()
                if new_cache_info.hits > prev_hits:
                    self._stats.cache_hits += 1
                else:
                    self._stats.cache_misses += 1
                
                return result
        else:
            return self._calculate_penalty_impl(*args, **kwargs)
    
    def disable_cache(self):
        """Disable caching"""
        with self._cache_lock:
            self._cache_enabled = False
    
    def enable_cache(self):
        """Enable caching"""
        with self._cache_lock:
            self._cache_enabled = True
            self._reset_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if hasattr(self, '_calculate_penalty'):
            cache_info = self._calculate_penalty.cache_info()
            return {
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'maxsize': cache_info.maxsize,
                'currsize': cache_info.currsize,
                'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
            }
        return {}
    
    def clear_cache(self):
        """Clear the cache"""
        with self._cache_lock:
            if hasattr(self, '_calculate_penalty'):
                self._calculate_penalty.cache_clear()


# Enhanced JIT-compiled helper functions
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _calculate_one_hot_encoding_fast(population, max_id):
        """Optimized JIT-compiled one-hot encoding calculation"""
        n_lineups = len(population)
        n_players = population.shape[1]
        
        # Create one-hot encoding with better memory layout
        ohe = np.zeros((n_lineups, max_id + 1), dtype=np.int8)  # Use int8 to save memory
        
        for i in prange(n_lineups):
            for j in range(n_players):
                player_id = population[i, j]
                if 0 <= player_id <= max_id:  # Bounds checking
                    ohe[i, player_id] = 1
                
        return ohe
    
    @njit(parallel=True, fastmath=True)
    def _calculate_diversity_matrix_fast(ohe):
        """Optimized JIT-compiled diversity matrix calculation"""
        n_lineups = ohe.shape[0]
        diversity = np.zeros((n_lineups, n_lineups), dtype=np.float32)
        
        for i in prange(n_lineups):
            for j in range(i, n_lineups):  # Only calculate upper triangle
                overlap = 0
                for k in range(ohe.shape[1]):
                    overlap += ohe[i, k] * ohe[j, k]
                diversity[i, j] = overlap
                diversity[j, i] = overlap  # Mirror to lower triangle
                
        return diversity
    
    @njit(parallel=True, fastmath=True)
    def _calculate_ownership_penalties_fast(ownership, threshold, penalty_factor, base):
        """Optimized JIT-compiled ownership penalty calculation"""
        n_players = len(ownership)
        penalties = np.zeros(n_players, dtype=np.float32)
        
        for i in prange(n_players):
            if ownership[i] > threshold:
                excess = ownership[i] - threshold
                # Use fastmath for better performance
                penalties[i] = -penalty_factor * np.log(1.0 + excess * 10.0) / np.log(base)
                
        return penalties
    
    @njit(parallel=True, fastmath=True)
    def _calculate_correlation_penalties_fast(population, correlation_matrix, strength):
        """Optimized JIT-compiled correlation penalty calculation"""
        n_lineups = len(population)
        n_players = population.shape[1]
        penalties = np.zeros(n_lineups, dtype=np.float32)
        
        for i in prange(n_lineups):
            lineup = population[i]
            lineup_corr = 0.0
            
            # Only calculate upper triangle to avoid double counting
            for j in range(n_players):
                for k in range(j + 1, n_players):
                    player1 = lineup[j]
                    player2 = lineup[k]
                    if 0 <= player1 < correlation_matrix.shape[0] and 0 <= player2 < correlation_matrix.shape[1]:
                        corr = correlation_matrix[player1, player2]
                        if corr > 0:
                            lineup_corr += corr
                        
            penalties[i] = -strength * lineup_corr
            
        return penalties
    
    @njit(parallel=True, fastmath=True)
    def _calculate_distance_penalties_fast(population, max_id):
        """Optimized distance penalty calculation using Hamming distance"""
        n_lineups = len(population)
        n_players = population.shape[1]
        penalties = np.zeros(n_lineups, dtype=np.float32)
        
        # Calculate pairwise Hamming distances efficiently
        for i in prange(n_lineups):
            total_distance = 0.0
            lineup_i = population[i]
            
            for j in range(n_lineups):
                if i != j:
                    lineup_j = population[j]
                    distance = 0
                    
                    # Count different players (Hamming distance)
                    for k in range(n_players):
                        if lineup_i[k] != lineup_j[k]:
                            distance += 1
                    
                    total_distance += distance
            
            # Average distance from all other lineups
            penalties[i] = total_distance / (n_lineups - 1) if n_lineups > 1 else 0.0
            
        return penalties


class OptimizedDistancePenalty(PenaltyBase):
    """Optimized penalty based on distance between lineups with enhanced performance."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._stats = PenaltyStats()

    @profile_enhanced
    def penalty(self, *, population: np.ndarray, robust: bool = True, 
                sigmoid_scale: float = 1.0, batch_size: int = 1000, 
                distance_metric: str = 'hamming', **kwargs) -> np.ndarray:
        """Calculates optimized distance penalty for overlapping lineups
        
        Args:
            population (np.ndarray): the population
            robust (bool): Whether to use robust normalization (default: True)
            sigmoid_scale (float): Scaling factor for sigmoid transformation
            batch_size (int): Size of batches for processing large populations
            distance_metric (str): Distance metric to use ('hamming', 'euclidean', 'cosine')
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: 1D array of float penalties

        """
        n_lineups = len(population)
        
        # For small populations, process all at once
        if n_lineups <= batch_size or batch_size <= 0:
            return self._calculate_distance_penalty(
                population, robust, sigmoid_scale, distance_metric, **kwargs)
        
        # For large populations, process in batches to save memory
        penalties = np.zeros(n_lineups)
        self._stats.batch_count = 0
        
        for i in range(0, n_lineups, batch_size):
            batch_end = min(i + batch_size, n_lineups)
            batch = population[i:batch_end]
            
            batch_penalties = self._calculate_distance_penalty(
                batch, robust, sigmoid_scale, distance_metric, **kwargs)
            
            penalties[i:batch_end] = batch_penalties
            self._stats.batch_count += 1
            
        return penalties
    
    def _calculate_distance_penalty(self, population: np.ndarray, 
                                   robust: bool, sigmoid_scale: float, 
                                   distance_metric: str, **kwargs) -> np.ndarray:
        """Helper method to calculate distance penalty for a batch"""
        
        # Use optimized Numba implementation if available
        if NUMBA_AVAILABLE and distance_metric == 'hamming':
            max_id = population.max()
            mean_distances = _calculate_distance_penalties_fast(population, max_id)
        elif SCIPY_AVAILABLE and distance_metric == 'euclidean':
            # Use scipy's optimized distance calculation
            max_id = population.max()
            ohe = _calculate_one_hot_encoding_fast(population, max_id) if NUMBA_AVAILABLE else self._create_one_hot_fallback(population, max_id)
            distances = squareform(pdist(ohe, 'euclidean'))
            mean_distances = np.mean(distances, axis=1)
        elif SKLEARN_AVAILABLE and distance_metric == 'cosine':
            # Use sklearn's cosine similarity
            max_id = population.max()
            ohe = _calculate_one_hot_encoding_fast(population, max_id) if NUMBA_AVAILABLE else self._create_one_hot_fallback(population, max_id)
            similarities = cosine_similarity(ohe)
            # Convert similarities to distances
            distances = 1 - similarities
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
    
    def _create_one_hot_fallback(self, population: np.ndarray, max_id: int) -> np.ndarray:
        """Fallback one-hot encoding when Numba is not available"""
        ohe = np.zeros((len(population), max_id + 1), dtype=np.int8)
        for i, lineup in enumerate(population):
            for player_id in lineup:
                if 0 <= player_id <= max_id:
                    ohe[i, player_id] = 1
        return ohe


class OptimizedDiversityPenalty(PenaltyBase):
    """Optimized penalty based on diversity between lineups with enhanced performance."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._stats = PenaltyStats()

    @profile_enhanced
    def penalty(self, *, population: np.ndarray, robust: bool = True, 
                sigmoid_scale: float = 1.0, batch_size: int = 1000, **kwargs) -> np.ndarray:
        """Calculates optimized diversity penalty for overlapping lineups
        
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
        self._stats.batch_count = 0
        
        for i in range(0, n_lineups, batch_size):
            batch_end = min(i + batch_size, n_lineups)
            batch = population[i:batch_end]
            
            batch_penalties = self._calculate_diversity_penalty(
                batch, robust, sigmoid_scale, **kwargs)
            
            penalties[i:batch_end] = batch_penalties
            self._stats.batch_count += 1
            
        return penalties
    
    def _calculate_diversity_penalty(self, population: np.ndarray, 
                                    robust: bool, sigmoid_scale: float, **kwargs) -> np.ndarray:
        """Helper method to calculate diversity penalty for a batch"""
        # Optimized diversity calculation
        max_id = population.max()
        
        # Create one-hot encoding
        if NUMBA_AVAILABLE:
            ohe = _calculate_one_hot_encoding_fast(population, max_id)
            diversity_matrix = _calculate_diversity_matrix_fast(ohe)
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


class OptimizedOwnershipPenalty(EnhancedCachedPenaltyMixin, PenaltyBase):
    """Optimized penalty based on player ownership percentages with enhanced caching."""

    def __init__(self, ctx=None):
        PenaltyBase.__init__(self, ctx)
        EnhancedCachedPenaltyMixin.__init__(self)

    @profile_enhanced
    def penalty(self, *, ownership: np.ndarray, base: float = 3, boost: float = 2, 
                min_penalty: float = -5.0, max_penalty: float = 0.0, 
                use_vectorized: bool = True, **kwargs) -> np.ndarray:
        """Calculates optimized penalties that are inverse to projected ownership
        
        Args:
            ownership (np.ndarray): 1D array of ownership percentages (0-1)
            base (float): the logarithm base, default 3
            boost (float): the constant to boost low-owned players
            min_penalty (float): Minimum penalty value
            max_penalty (float): Maximum penalty value
            use_vectorized (bool): Whether to use vectorized calculation
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
            
        """
        if use_vectorized:
            # Vectorized calculation for better performance
            # Ensure ownership values are valid (between 0 and 1)
            ownership_clipped = np.clip(ownership, 0.01, 1.0)  # Avoid log(0)
            
            # Calculate penalties using vectorized operations
            penalties = 0 - np.log(ownership_clipped) / np.log(base) + boost
            
            # Clip penalties to specified range
            return np.clip(penalties, min_penalty, max_penalty)
        else:
            # Use cached calculation for individual players
            penalties = np.zeros_like(ownership)
            for i, own in enumerate(ownership):
                penalties[i] = self._calculate_penalty(i, ownership, base, boost, min_penalty, max_penalty)
            return penalties
    
    def _calculate_penalty_impl(self, player_id, ownership, base, boost, min_penalty, max_penalty):
        """Cached penalty calculation for a single player"""
        # Ensure ownership value is valid
        ownership_val = max(0.01, min(1.0, ownership[player_id]))
        
        # Calculate penalty
        penalty = 0 - np.log(ownership_val) / np.log(base) + boost
        
        # Clip penalty to specified range
        return max(min_penalty, min(max_penalty, penalty))


class OptimizedHighOwnershipPenalty(PenaltyBase):
    """Optimized penalty that only applies to players with high ownership."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._stats = PenaltyStats()

    @profile_enhanced
    def penalty(self, *, ownership: np.ndarray, threshold: float = 0.15, 
                penalty_factor: float = 2.0, base: float = 3, **kwargs) -> np.ndarray:
        """Calculates optimized penalties only for highly-owned players
        
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
            return _calculate_ownership_penalties_fast(ownership, threshold, penalty_factor, base)
        
        # Optimized fallback implementation using vectorization
        # Initialize penalties to zero
        penalties = np.zeros_like(ownership, dtype=np.float32)
        
        # Apply penalties only to players above the threshold
        high_owned_mask = ownership > threshold
        if np.any(high_owned_mask):
            # Calculate how much above threshold (vectorized)
            excess = ownership[high_owned_mask] - threshold
            
            # Apply logarithmic penalty that increases more rapidly as ownership increases
            penalties[high_owned_mask] = -penalty_factor * np.log(1 + excess * 10) / np.log(base)
            
        return penalties


class OptimizedCorrelationPenalty(PenaltyBase):
    """Optimized penalty based on statistical correlations between players."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._stats = PenaltyStats()

    @profile_enhanced
    def penalty(self, *, population: np.ndarray, correlation_matrix: np.ndarray, 
                strength: float = 1.0, position_pairs: List[Tuple[str, str]] = None, 
                early_termination: bool = True, use_sparse: bool = False, **kwargs) -> np.ndarray:
        """Applies optimized penalties based on statistical correlations between players
        
        Args:
            population (np.ndarray): the population
            correlation_matrix (np.ndarray): Matrix of player correlations
            strength (float): Penalty strength multiplier
            position_pairs (list): List of position pairs to consider for correlation
            early_termination (bool): Whether to use early termination optimization
            use_sparse (bool): Whether to use sparse matrix operations for large matrices
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
        """
        # Use JIT-compiled function if available
        if NUMBA_AVAILABLE:
            return _calculate_correlation_penalties_fast(population, correlation_matrix, strength)
        
        # Use sparse matrix operations for large correlation matrices
        if use_sparse and SCIPY_AVAILABLE and correlation_matrix.size > 10000:
            return self._calculate_sparse_correlation_penalties(population, correlation_matrix, strength)
        
        # Optimized fallback implementation with vectorization
        n_lineups = len(population)
        penalties = np.zeros(n_lineups, dtype=np.float32)
        
        # Pre-compute upper triangle mask for efficiency
        lineup_size = population.shape[1]
        triu_indices = np.triu_indices(lineup_size, k=1)
        
        for i in range(n_lineups):
            lineup_indices = population[i]
            
            # Early termination check
            if early_termination and len(np.unique(lineup_indices)) == 1:
                penalties[i] = -strength * correlation_matrix[lineup_indices[0], lineup_indices[0]]
                continue
                
            # Get correlation submatrix for this lineup
            corr_submatrix = correlation_matrix[np.ix_(lineup_indices, lineup_indices)]
            
            # Sum only positive correlations in upper triangle (vectorized)
            upper_triangle_values = corr_submatrix[triu_indices]
            positive_corrs = np.maximum(0, upper_triangle_values)
            penalties[i] = -strength * np.sum(positive_corrs)
        
        return penalties
    
    def _calculate_sparse_correlation_penalties(self, population: np.ndarray, 
                                               correlation_matrix: np.ndarray, 
                                               strength: float) -> np.ndarray:
        """Calculate correlation penalties using sparse matrix operations"""
        # Convert to sparse matrix if not already
        if not SCIPY_AVAILABLE:
            return self.penalty(population=population, correlation_matrix=correlation_matrix, 
                              strength=strength, use_sparse=False)
        
        sparse_corr = csr_matrix(correlation_matrix)
        n_lineups = len(population)
        penalties = np.zeros(n_lineups, dtype=np.float32)
        
        for i in range(n_lineups):
            lineup_indices = population[i]
            
            # Extract submatrix efficiently using sparse operations
            submatrix = sparse_corr[lineup_indices][:, lineup_indices]
            
            # Convert back to dense for upper triangle calculation
            dense_sub = submatrix.toarray()
            
            # Calculate upper triangle sum
            triu_mask = np.triu(np.ones_like(dense_sub), k=1).astype(bool)
            positive_corrs = np.maximum(0, dense_sub[triu_mask])
            penalties[i] = -strength * np.sum(positive_corrs)
        
        return penalties


class OptimizedStackingReward(PenaltyBase):
    """Optimized reward (negative penalty) for strategic stacking of correlated positions."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._stats = PenaltyStats()

    @profile_enhanced
    def penalty(self, *, population: np.ndarray, player_teams: np.ndarray, 
                player_positions: np.ndarray, stack_configs: Dict[str, float] = None, 
                use_vectorized: bool = True, **kwargs) -> np.ndarray:
        """Provides optimized rewards (negative penalties) for strategic position stacking
        
        Args:
            population (np.ndarray): the population
            player_teams (np.ndarray): Array mapping player indices to team indices
            player_positions (np.ndarray): Array mapping player indices to position indices
            stack_configs (dict): Configuration for different stack types and their rewards
            use_vectorized (bool): Whether to use vectorized calculations
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of rewards (negative penalties)
        """
        if stack_configs is None:
            stack_configs = {'QB-WR': 2.0, 'QB-TE': 1.5, 'QB-WR-WR': 3.0}
            
        n_lineups = len(population)
        rewards = np.zeros(n_lineups, dtype=np.float32)
        
        if use_vectorized:
            # Vectorized implementation for better performance
            for i in range(n_lineups):
                lineup = population[i]
                teams = player_teams[lineup]
                positions = player_positions[lineup]
                
                # Find QB positions (vectorized)
                qb_mask = positions == 'QB'
                qb_teams = teams[qb_mask]
                
                for qb_team in np.unique(qb_teams):
                    # Count same-team players by position (vectorized)
                    same_team_mask = teams == qb_team
                    
                    same_team_wrs = np.sum(same_team_mask & (positions == 'WR'))
                    same_team_tes = np.sum(same_team_mask & (positions == 'TE'))
                    
                    # Apply stack rewards
                    if same_team_wrs > 0 and 'QB-WR' in stack_configs:
                        rewards[i] += stack_configs['QB-WR'] * same_team_wrs
                    
                    if same_team_tes > 0 and 'QB-TE' in stack_configs:
                        rewards[i] += stack_configs['QB-TE'] * same_team_tes
                        
                    # Check for QB-WR-WR stack (double stack)
                    if same_team_wrs >= 2 and 'QB-WR-WR' in stack_configs:
                        rewards[i] += stack_configs['QB-WR-WR']
        else:
            # Original implementation
            qb_mask = player_positions == 'QB'
            wr_mask = player_positions == 'WR'
            te_mask = player_positions == 'TE'
            
            for i in range(n_lineups):
                lineup = population[i]
                teams = player_teams[lineup]
                
                qb_indices = np.where(qb_mask[lineup])[0]
                
                for qb_idx in qb_indices:
                    qb_team = teams[qb_idx]
                    
                    same_team_wrs = np.sum((teams == qb_team) & wr_mask[lineup])
                    if same_team_wrs > 0 and 'QB-WR' in stack_configs:
                        rewards[i] += stack_configs['QB-WR'] * same_team_wrs
                    
                    same_team_tes = np.sum((teams == qb_team) & te_mask[lineup])
                    if same_team_tes > 0 and 'QB-TE' in stack_configs:
                        rewards[i] += stack_configs['QB-TE'] * same_team_tes
                        
                    if same_team_wrs >= 2 and 'QB-WR-WR' in stack_configs:
                        rewards[i] += stack_configs['QB-WR-WR']
            
        return rewards


class OptimizedGameTheoryPenalty(PenaltyBase):
    """Optimized penalty based on game theory concepts for tournament optimization."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._stats = PenaltyStats()

    @profile_enhanced
    def penalty(self, *, population: np.ndarray, ownership: np.ndarray, 
                field_size: int, tournament_type: str = 'gpp',
                early_termination_threshold: float = 0.01, 
                use_vectorized: bool = True, **kwargs) -> np.ndarray:
        """Applies optimized game theory based penalties for tournament optimization
        
        Args:
            population (np.ndarray): the population
            ownership (np.ndarray): Projected ownership percentages
            field_size (int): Number of entries in the tournament
            tournament_type (str): 'gpp', 'double_up', 'h2h', etc.
            early_termination_threshold (float): Threshold for early termination
            use_vectorized (bool): Whether to use vectorized calculations
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
        """
        n_lineups = len(population)
        penalties = np.zeros(n_lineups, dtype=np.float32)
        
        if use_vectorized and tournament_type == 'gpp':
            # Vectorized GPP calculation
            lineup_ownership = np.array([np.mean(ownership[lineup]) for lineup in population])
            
            # Early termination mask
            low_ownership_mask = lineup_ownership < early_termination_threshold
            penalties[low_ownership_mask] = 0
            
            # Calculate uniqueness for remaining lineups
            high_ownership_mask = ~low_ownership_mask
            if np.any(high_ownership_mask):
                high_ownership_values = lineup_ownership[high_ownership_mask]
                uniqueness = (1 - high_ownership_values) ** field_size
                penalties[high_ownership_mask] = -5.0 * (1 - uniqueness)
                
        elif use_vectorized and tournament_type == 'cash':
            # Vectorized cash game calculation
            lineup_ownership = np.array([np.mean(ownership[lineup]) for lineup in population])
            penalties = 2.0 * lineup_ownership
            
        else:
            # Original implementation for other tournament types
            for i in range(n_lineups):
                lineup = population[i]
                lineup_ownership = np.mean(ownership[lineup])
                
                if tournament_type == 'gpp':
                    if lineup_ownership < early_termination_threshold:
                        penalties[i] = 0
                        continue
                    
                    uniqueness = (1 - lineup_ownership) ** field_size
                    penalties[i] = -5.0 * (1 - uniqueness)
                    
                elif tournament_type == 'cash':
                    penalties[i] = 2.0 * lineup_ownership
                
        return penalties


class OptimizedVariancePenalty(PenaltyBase):
    """Optimized penalty based on player variance, crucial for tournament strategy."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._stats = PenaltyStats()

    @profile_enhanced
    def penalty(self, *, population: np.ndarray, player_variance: np.ndarray, 
                tournament_type: str = 'gpp', strength: float = 1.0, 
                use_vectorized: bool = True, **kwargs) -> np.ndarray:
        """Applies optimized penalties based on lineup variance
        
        Args:
            population (np.ndarray): the population
            player_variance (np.ndarray): Variance estimates for each player
            tournament_type (str): 'gpp', 'double_up', 'h2h', etc.
            strength (float): Penalty strength multiplier
            use_vectorized (bool): Whether to use vectorized calculations
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
        """
        if use_vectorized:
            # Vectorized calculation for all lineups at once
            lineup_variances = np.array([np.sum(player_variance[lineup]) for lineup in population])
            
            if tournament_type == 'gpp':
                # For GPPs, reward high variance
                penalties = -strength * lineup_variances
            elif tournament_type == 'cash':
                # For cash games, penalize high variance
                penalties = strength * lineup_variances
            else:
                penalties = np.zeros_like(lineup_variances)
        else:
            # Original implementation
            n_lineups = len(population)
            penalties = np.zeros(n_lineups, dtype=np.float32)
            
            for i in range(n_lineups):
                lineup_variance = np.sum(player_variance[population[i]])
                
                if tournament_type == 'gpp':
                    penalties[i] = -strength * lineup_variance
                elif tournament_type == 'cash':
                    penalties[i] = strength * lineup_variance
                
        return penalties


class OptimizedPositionSpecificPenalty(PenaltyBase):
    """Optimized penalty that applies different weights based on player positions."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._stats = PenaltyStats()

    @profile_enhanced
    def penalty(self, *, population: np.ndarray, player_positions: np.ndarray, 
                position_weights: Dict[str, float] = None, 
                use_vectorized: bool = True, **kwargs) -> np.ndarray:
        """Applies optimized different penalties based on player positions
        
        Args:
            population (np.ndarray): the population
            player_positions (np.ndarray): Array mapping player indices to positions
            position_weights (dict): Weights for each position
            use_vectorized (bool): Whether to use vectorized calculations
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: 1D array of penalties
        """
        if position_weights is None:
            position_weights = {'QB': 1.0, 'RB': 1.0, 'WR': 1.0, 'TE': 1.0, 'DST': 1.0}
            
        n_lineups = len(population)
        
        if use_vectorized:
            # Pre-compute position weights array for vectorization
            position_weight_map = np.ones(len(player_positions), dtype=np.float32)
            
            for pos, weight in position_weights.items():
                position_weight_map[player_positions == pos] = weight
            
            # Vectorized calculation
            penalties = np.array([np.sum(position_weight_map[lineup]) for lineup in population])
            
            # Normalize penalties
            if len(penalties) > 0 and np.std(penalties) > 0:
                penalties = (penalties - np.mean(penalties)) / np.std(penalties)
                
            return -penalties  # Convert to penalties (negative values)
        else:
            # Original implementation
            penalties = np.zeros(n_lineups, dtype=np.float32)
            
            for i in range(n_lineups):
                lineup = population[i]
                lineup_positions = player_positions[lineup]
                
                for pos in lineup_positions:
                    if pos in position_weights:
                        penalties[i] += position_weights[pos]
                        
            # Normalize penalties
            if len(penalties) > 0 and np.std(penalties) > 0:
                penalties = (penalties - np.mean(penalties)) / np.std(penalties)
                
            return -penalties


class OptimizedAdaptivePenaltyScaler(PenaltyBase):
    """Optimized scaler that scales penalties adaptively based on fitness distribution."""

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self._stats = PenaltyStats()

    @profile_enhanced
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
        # Vectorized calculation of fitness and penalty statistics
        fitness_range = np.ptp(fitness_scores)  # Peak-to-peak (max - min)
        penalty_std = np.std(base_penalties)
        
        if penalty_std == 0 or fitness_range == 0:
            return base_penalties
            
        # Calculate target scale to achieve desired impact
        target_scale = (fitness_range * target_impact) / penalty_std
        
        # Apply scaling
        return base_penalties * target_scale


class OptimizedEnsemblePenalty(PenaltyBase):
    """Optimized version that combines multiple penalties with learned weights and parallel processing."""

    def __init__(self, ctx=None, penalty_classes=None, weights=None, use_parallel=False, max_workers=None):
        """Initialize with multiple penalty classes and weights
        
        Args:
            ctx: Optional context dictionary
            penalty_classes (list): List of penalty classes to use
            weights (dict): Weights for each penalty class
            use_parallel (bool): Whether to use parallel processing
            max_workers (int): Maximum number of worker threads
        """
        super().__init__(ctx)
        self.penalty_classes = penalty_classes or []
        self.weights = weights or {cls.__name__: 1.0 for cls in self.penalty_classes}
        self.penalty_instances = [cls(ctx) for cls in self.penalty_classes]
        self.use_parallel = use_parallel
        self.max_workers = max_workers or min(4, len(self.penalty_instances))
        self._stats = PenaltyStats()
        
    @profile_enhanced
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
        
        if self.use_parallel and len(self.penalty_instances) > 1:
            # Parallel processing for multiple penalty instances
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for instance in self.penalty_instances:
                    future = executor.submit(instance.penalty, **kwargs)
                    futures.append((instance.__class__.__name__, future))
                
                # Collect results
                penalty_results = {}
                for class_name, future in futures:
                    try:
                        penalty_results[class_name] = future.result()
                    except Exception as e:
                        logger.error(f"Error in {class_name}: {str(e)}")
                        penalty_results[class_name] = np.zeros(len(population))
        else:
            # Sequential processing
            penalty_results = {}
            for instance in self.penalty_instances:
                class_name = instance.__class__.__name__
                try:
                    penalty_results[class_name] = instance.penalty(**kwargs)
                except Exception as e:
                    logger.error(f"Error in {class_name}: {str(e)}")
                    penalty_results[class_name] = np.zeros(len(population))
        
        # Combine penalties with weights
        combined_penalties = np.zeros(len(population), dtype=np.float32)
        total_weight = 0.0
        
        for class_name, penalties in penalty_results.items():
            weight = self.weights.get(class_name, 1.0)
            combined_penalties += weight * penalties
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_penalties /= total_weight
            
        return combined_penalties
    
    def get_penalty_stats(self) -> Dict[str, Any]:
        """Get statistics for all penalty instances"""
        stats = {}
        for instance in self.penalty_instances:
            class_name = instance.__class__.__name__
            if hasattr(instance, '_stats'):
                stats[class_name] = {
                    'calculation_time': instance._stats.calculation_time,
                    'cache_hits': getattr(instance._stats, 'cache_hits', 0),
                    'cache_misses': getattr(instance._stats, 'cache_misses', 0),
                    'batch_count': getattr(instance._stats, 'batch_count', 0)
                }
            if hasattr(instance, 'get_cache_stats'):
                stats[class_name]['cache_stats'] = instance.get_cache_stats()
        return stats


# Utility functions for penalty optimization
def create_optimized_penalty_ensemble(penalty_configs: List[Dict[str, Any]], 
                                     ctx=None, use_parallel: bool = True) -> OptimizedEnsemblePenalty:
    """Factory function to create an optimized penalty ensemble
    
    Args:
        penalty_configs (list): List of penalty configurations
        ctx: Optional context dictionary
        use_parallel (bool): Whether to use parallel processing
        
    Returns:
        OptimizedEnsemblePenalty: Configured ensemble penalty
    """
    penalty_classes = []
    weights = {}
    
    # Map penalty names to classes
    penalty_class_map = {
        'distance': OptimizedDistancePenalty,
        'diversity': OptimizedDiversityPenalty,
        'ownership': OptimizedOwnershipPenalty,
        'high_ownership': OptimizedHighOwnershipPenalty,
        'correlation': OptimizedCorrelationPenalty,
        'stacking': OptimizedStackingReward,
        'game_theory': OptimizedGameTheoryPenalty,
        'variance': OptimizedVariancePenalty,
        'position_specific': OptimizedPositionSpecificPenalty,
        'adaptive_scaler': OptimizedAdaptivePenaltyScaler
    }
    
    for config in penalty_configs:
        penalty_type = config.get('type')
        weight = config.get('weight', 1.0)
        
        if penalty_type in penalty_class_map:
            penalty_class = penalty_class_map[penalty_type]
            penalty_classes.append(penalty_class)
            weights[penalty_class.__name__] = weight
        else:
            logger.warning(f"Unknown penalty type: {penalty_type}")
    
    return OptimizedEnsemblePenalty(
        ctx=ctx,
        penalty_classes=penalty_classes,
        weights=weights,
        use_parallel=use_parallel
    )


def benchmark_penalty_performance(penalty_instance, population_sizes: List[int], 
                                 iterations: int = 10) -> Dict[str, List[float]]:
    """Benchmark penalty performance across different population sizes
    
    Args:
        penalty_instance: Penalty instance to benchmark
        population_sizes (list): List of population sizes to test
        iterations (int): Number of iterations per size
        
    Returns:
        dict: Benchmark results
    """
    results = {
        'population_sizes': population_sizes,
        'execution_times': [],
        'memory_usage': []
    }
    
    for size in population_sizes:
        # Create test population
        test_population = np.random.randint(0, 1000, size=(size, 9))
        
        times = []
        for _ in range(iterations):
            start_time = time.time()
            
            # Run penalty calculation
            try:
                penalty_instance.penalty(population=test_population)
                execution_time = time.time() - start_time
                times.append(execution_time)
            except Exception as e:
                logger.error(f"Error in benchmark: {str(e)}")
                times.append(float('inf'))
        
        avg_time = np.mean(times) if times else float('inf')
        results['execution_times'].append(avg_time)
        
        # Estimate memory usage (rough approximation)
        memory_estimate = test_population.nbytes * 2  # Rough estimate
        results['memory_usage'].append(memory_estimate)
    
    return results


# Create aliases for backward compatibility - use optimized versions as defaults
DistancePenalty = OptimizedDistancePenalty
DiversityPenalty = OptimizedDiversityPenalty
OwnershipPenalty = OptimizedOwnershipPenalty
HighOwnershipPenalty = OptimizedHighOwnershipPenalty
CorrelationPenalty = OptimizedCorrelationPenalty
StackingReward = OptimizedStackingReward
GameTheoryPenalty = OptimizedGameTheoryPenalty
VariancePenalty = OptimizedVariancePenalty
PositionSpecificPenalty = OptimizedPositionSpecificPenalty
AdaptivePenaltyScaler = OptimizedAdaptivePenaltyScaler
EnsemblePenalty = OptimizedEnsemblePenalty
CachedPenaltyMixin = EnhancedCachedPenaltyMixin

# Factory function alias for backward compatibility
create_penalty_ensemble = create_optimized_penalty_ensemble

# Export both optimized and original names for maximum compatibility
__all__ = [
    # Optimized class names (new)
    'OptimizedDistancePenalty',
    'OptimizedDiversityPenalty', 
    'OptimizedOwnershipPenalty',
    'OptimizedHighOwnershipPenalty',
    'OptimizedCorrelationPenalty',
    'OptimizedStackingReward',
    'OptimizedGameTheoryPenalty',
    'OptimizedVariancePenalty',
    'OptimizedPositionSpecificPenalty',
    'OptimizedAdaptivePenaltyScaler',
    'OptimizedEnsemblePenalty',
    'EnhancedCachedPenaltyMixin',
    
    # Original class names (backward compatibility aliases)
    'DistancePenalty',
    'DiversityPenalty',
    'OwnershipPenalty',
    'HighOwnershipPenalty',
    'CorrelationPenalty',
    'StackingReward',
    'GameTheoryPenalty',
    'VariancePenalty',
    'PositionSpecificPenalty',
    'AdaptivePenaltyScaler',
    'EnsemblePenalty',
    'CachedPenaltyMixin',
    
    # Utility classes and functions
    'PenaltyStats',
    'profile_enhanced',
    'create_optimized_penalty_ensemble',
    'create_penalty_ensemble',
    'benchmark_penalty_performance'
]
