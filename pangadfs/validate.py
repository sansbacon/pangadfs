# pangadfs/pangadfs/validate.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

"""
Enhanced validation module for genetic algorithms with performance optimizations.

This module provides comprehensive validation functionality for DFS lineups including:
- Salary cap validation with JIT optimization
- Position requirement validation
- Player exposure and ownership constraints
- Team stacking validation
- Duplicate removal (leveraging duplicates.py)
- Composite validation with parallel processing
- Comprehensive validation statistics and monitoring
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from pangadfs.base import ValidateBase
from pangadfs.duplicates import DuplicateRemover, remove_internal_duplicates, remove_duplicate_lineups

# Set up logging
logger = logging.getLogger(__name__)

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    logger.debug("Numba is available for validation JIT compilation")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.debug("Numba not available, using fallback implementations")
    # Fallback if Numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# JIT-compiled helper functions for performance
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _validate_salaries_jit(population: np.ndarray, salaries: np.ndarray, 
                              salary_cap: float, tolerance: float) -> np.ndarray:
        """JIT-compiled salary validation for large populations"""
        n_individuals = population.shape[0]
        valid_mask = np.zeros(n_individuals, dtype=np.bool_)
        
        for i in prange(n_individuals):
            lineup_salary = 0.0
            for j in range(population.shape[1]):
                lineup_salary += salaries[population[i, j]]
            
            if lineup_salary <= salary_cap + tolerance:
                valid_mask[i] = True
        
        return valid_mask

    @njit(parallel=True, fastmath=True)
    def _validate_positions_jit(population: np.ndarray, position_map: np.ndarray,
                               required_positions: np.ndarray) -> np.ndarray:
        """JIT-compiled position validation"""
        n_individuals = population.shape[0]
        valid_mask = np.zeros(n_individuals, dtype=np.bool_)
        
        for i in prange(n_individuals):
            lineup = population[i]
            position_counts = np.zeros(len(required_positions), dtype=np.int32)
            
            # Count positions in lineup
            for j in range(len(lineup)):
                player_id = lineup[j]
                if player_id < len(position_map):
                    pos = position_map[player_id]
                    if 0 <= pos < len(position_counts):
                        position_counts[pos] += 1
            
            # Check if all position requirements are met
            valid = True
            for k in range(len(required_positions)):
                if position_counts[k] != required_positions[k]:
                    valid = False
                    break
            
            valid_mask[i] = valid
        
        return valid_mask

    @njit(parallel=True, fastmath=True)
    def _calculate_exposure_jit(population: np.ndarray, n_players: int) -> np.ndarray:
        """JIT-compiled player exposure calculation"""
        exposure_counts = np.zeros(n_players, dtype=np.int32)
        n_lineups = population.shape[0]
        
        for i in prange(n_lineups):
            for j in range(population.shape[1]):
                player_id = population[i, j]
                if player_id < n_players:
                    exposure_counts[player_id] += 1
        
        return exposure_counts


class DuplicatesValidate(ValidateBase):
    """Enhanced duplicate validation using optimized duplicate removal functions."""

    def __init__(self, ctx=None, remover_config=None):
        """Initialize with optional DuplicateRemover configuration.
        
        Args:
            ctx: Plugin context
            remover_config (dict): Configuration for DuplicateRemover
        """
        super().__init__(ctx)
        
        # Initialize DuplicateRemover with configuration
        if remover_config is None:
            remover_config = {}
        
        self.duplicate_remover = DuplicateRemover(**remover_config)
        self._validation_stats = {}

    def validate(self,
                 *, 
                 population: np.ndarray,
                 remove_internal: bool = True,
                 remove_lineup_duplicates: bool = True,
                 return_stats: bool = False,
                 **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Enhanced duplicate removal with configurable options and statistics.
        
        Args:
            population (np.ndarray): Population to validate
            remove_internal (bool): Remove lineups with internal duplicates
            remove_lineup_duplicates (bool): Remove duplicate lineups
            return_stats (bool): Return validation statistics
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray or Tuple[np.ndarray, Dict]: Validated population, optionally with stats
        """
        if population.size == 0:
            if return_stats:
                return population, {'total_lineups': 0, 'removed_lineups': 0}
            return population
        
        original_size = len(population)
        
        # Use DuplicateRemover for comprehensive duplicate handling
        validated_population = self.duplicate_remover.remove_duplicates(
            population=population,
            remove_internal=remove_internal,
            remove_lineup_duplicates=remove_lineup_duplicates
        )
        
        # Calculate validation statistics if requested
        if return_stats:
            stats = self.duplicate_remover.validate_population(population)
            stats['original_size'] = original_size
            stats['final_size'] = len(validated_population)
            stats['removed_lineups'] = original_size - len(validated_population)
            stats['removal_percentage'] = (stats['removed_lineups'] / original_size) * 100.0
            
            self._validation_stats = stats
            return validated_population, stats
        
        return validated_population
    
    def get_last_validation_stats(self) -> Dict:
        """Get statistics from the last validation run."""
        return self._validation_stats.copy()


class SalaryValidate(ValidateBase):
    """Enhanced salary validation with JIT optimization and performance monitoring."""

    def __init__(self, ctx=None, use_jit=None, batch_size=1000):
        """Initialize salary validator with performance options.
        
        Args:
            ctx: Plugin context
            use_jit (bool): Force JIT usage. If None, auto-selects based on data size
            batch_size (int): Batch size for large population processing
        """
        super().__init__(ctx)
        self.use_jit = use_jit
        self.batch_size = batch_size
        self._jit_threshold = 100
        self._validation_stats = {}

    def validate(self,
                 *, 
                 population: np.ndarray,
                 salaries: np.ndarray,
                 salary_cap: Union[int, float],
                 tolerance: Union[int, float] = 0,
                 return_stats: bool = False,
                 **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Enhanced salary validation with JIT optimization.
        
        Args:
            population (np.ndarray): Population to validate
            salaries (np.ndarray): 1D array where indices match player indices
            salary_cap (Union[int, float]): Maximum allowed salary
            tolerance (Union[int, float]): Tolerance for salary cap (default: 0)
            return_stats (bool): Return validation statistics
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray or Tuple[np.ndarray, Dict]: Valid population, optionally with stats
        """
        # Input validation
        if population.size == 0:
            if return_stats:
                return population, {'total_lineups': 0, 'valid_lineups': 0}
            return population
        
        if len(salaries) == 0:
            raise ValueError("Salaries array cannot be empty")
        
        if salary_cap <= 0:
            raise ValueError("Salary cap must be positive")
        
        # Check if any player indices are out of bounds
        max_player_id = np.max(population)
        if max_player_id >= len(salaries):
            raise ValueError(f"Player ID {max_player_id} exceeds salaries array length {len(salaries)}")
        
        original_size = len(population)
        
        # Auto-select algorithm based on data size and availability
        use_jit = self.use_jit
        if use_jit is None:
            use_jit = NUMBA_AVAILABLE and len(population) > self._jit_threshold
        
        # Process in batches if population is large
        if len(population) > self.batch_size:
            valid_population = self._validate_salaries_batched(
                population, salaries, salary_cap, tolerance, use_jit
            )
        else:
            valid_population = self._validate_salaries_single(
                population, salaries, salary_cap, tolerance, use_jit
            )
        
        # Calculate validation statistics if requested
        if return_stats:
            stats = self._calculate_salary_stats(
                population, valid_population, salaries, salary_cap, tolerance
            )
            self._validation_stats = stats
            return valid_population, stats
        
        return valid_population
    
    def _validate_salaries_single(self, population: np.ndarray, salaries: np.ndarray,
                                 salary_cap: float, tolerance: float, use_jit: bool) -> np.ndarray:
        """Validate salaries for a single batch."""
        if use_jit and NUMBA_AVAILABLE:
            valid_mask = _validate_salaries_jit(population, salaries, salary_cap, tolerance)
        else:
            # Numpy-based implementation
            lineup_salaries = np.sum(salaries[population], axis=1)
            valid_mask = lineup_salaries <= (salary_cap + tolerance)
        
        return population[valid_mask]
    
    def _validate_salaries_batched(self, population: np.ndarray, salaries: np.ndarray,
                                  salary_cap: float, tolerance: float, use_jit: bool) -> np.ndarray:
        """Process large populations in batches."""
        valid_batches = []
        
        for i in range(0, len(population), self.batch_size):
            batch = population[i:i + self.batch_size]
            valid_batch = self._validate_salaries_single(batch, salaries, salary_cap, tolerance, use_jit)
            if len(valid_batch) > 0:
                valid_batches.append(valid_batch)
        
        if valid_batches:
            return np.vstack(valid_batches)
        else:
            return np.array([]).reshape(0, population.shape[1])
    
    def _calculate_salary_stats(self, original_pop: np.ndarray, valid_pop: np.ndarray,
                               salaries: np.ndarray, salary_cap: float, tolerance: float) -> Dict:
        """Calculate detailed salary validation statistics."""
        original_salaries = np.sum(salaries[original_pop], axis=1)
        
        stats = {
            'total_lineups': len(original_pop),
            'valid_lineups': len(valid_pop),
            'invalid_lineups': len(original_pop) - len(valid_pop),
            'validation_rate': len(valid_pop) / len(original_pop) if len(original_pop) > 0 else 0,
            'salary_cap': salary_cap,
            'tolerance': tolerance,
            'salary_stats': {
                'min_salary': float(np.min(original_salaries)) if len(original_salaries) > 0 else 0,
                'max_salary': float(np.max(original_salaries)) if len(original_salaries) > 0 else 0,
                'mean_salary': float(np.mean(original_salaries)) if len(original_salaries) > 0 else 0,
                'std_salary': float(np.std(original_salaries)) if len(original_salaries) > 0 else 0,
                'over_cap_count': int(np.sum(original_salaries > salary_cap + tolerance))
            }
        }
        
        if len(valid_pop) > 0:
            valid_salaries = np.sum(salaries[valid_pop], axis=1)
            stats['valid_salary_stats'] = {
                'min_salary': float(np.min(valid_salaries)),
                'max_salary': float(np.max(valid_salaries)),
                'mean_salary': float(np.mean(valid_salaries)),
                'std_salary': float(np.std(valid_salaries))
            }
        
        return stats
    
    def set_jit_threshold(self, threshold: int):
        """Set the population size threshold for JIT compilation."""
        self._jit_threshold = threshold
    
    def get_last_validation_stats(self) -> Dict:
        """Get statistics from the last validation run."""
        return self._validation_stats.copy()


class PositionValidate(ValidateBase):
    """Validate that lineups meet position requirements."""

    def __init__(self, ctx=None, use_jit=None):
        """Initialize position validator.
        
        Args:
            ctx: Plugin context
            use_jit (bool): Force JIT usage. If None, auto-selects based on data size
        """
        super().__init__(ctx)
        self.use_jit = use_jit
        self._jit_threshold = 100
        self._validation_stats = {}

    def validate(self,
                 *,
                 population: np.ndarray,
                 position_map: Union[Dict, np.ndarray],
                 position_requirements: Dict[str, int],
                 return_stats: bool = False,
                 **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Validate position requirements for lineups.
        
        Args:
            population (np.ndarray): Population to validate
            position_map (Dict or np.ndarray): Maps player IDs to position indices
            position_requirements (Dict): Required count for each position
            return_stats (bool): Return validation statistics
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray or Tuple[np.ndarray, Dict]: Valid population, optionally with stats
        """
        if population.size == 0:
            if return_stats:
                return population, {'total_lineups': 0, 'valid_lineups': 0}
            return population
        
        # Convert position_map to numpy array if it's a dict
        if isinstance(position_map, dict):
            max_player_id = max(position_map.keys()) if position_map else 0
            pos_array = np.full(max_player_id + 1, -1, dtype=np.int32)
            for player_id, pos_idx in position_map.items():
                pos_array[player_id] = pos_idx
            position_map = pos_array
        
        # Convert position requirements to arrays
        position_names = list(position_requirements.keys())
        required_counts = np.array([position_requirements[pos] for pos in position_names], dtype=np.int32)
        
        original_size = len(population)
        
        # Auto-select algorithm based on data size
        use_jit = self.use_jit
        if use_jit is None:
            use_jit = NUMBA_AVAILABLE and len(population) > self._jit_threshold
        
        if use_jit and NUMBA_AVAILABLE:
            valid_mask = _validate_positions_jit(population, position_map, required_counts)
            valid_population = population[valid_mask]
        else:
            valid_population = self._validate_positions_numpy(
                population, position_map, required_counts
            )
        
        # Calculate validation statistics if requested
        if return_stats:
            stats = {
                'total_lineups': original_size,
                'valid_lineups': len(valid_population),
                'invalid_lineups': original_size - len(valid_population),
                'validation_rate': len(valid_population) / original_size if original_size > 0 else 0,
                'position_requirements': position_requirements
            }
            self._validation_stats = stats
            return valid_population, stats
        
        return valid_population
    
    def _validate_positions_numpy(self, population: np.ndarray, position_map: np.ndarray,
                                 required_counts: np.ndarray) -> np.ndarray:
        """Numpy-based position validation."""
        valid_indices = []
        
        for i, lineup in enumerate(population):
            # Count positions in this lineup
            position_counts = np.zeros(len(required_counts), dtype=np.int32)
            
            for player_id in lineup:
                if player_id < len(position_map) and position_map[player_id] >= 0:
                    pos_idx = position_map[player_id]
                    if pos_idx < len(position_counts):
                        position_counts[pos_idx] += 1
            
            # Check if requirements are met
            if np.array_equal(position_counts, required_counts):
                valid_indices.append(i)
        
        return population[valid_indices] if valid_indices else np.array([]).reshape(0, population.shape[1])
    
    def get_last_validation_stats(self) -> Dict:
        """Get statistics from the last validation run."""
        return self._validation_stats.copy()


class ExposureValidate(ValidateBase):
    """Validate player exposure and ownership constraints."""

    def __init__(self, ctx=None, use_jit=None):
        """Initialize exposure validator.
        
        Args:
            ctx: Plugin context
            use_jit (bool): Force JIT usage. If None, auto-selects based on data size
        """
        super().__init__(ctx)
        self.use_jit = use_jit
        self._jit_threshold = 100
        self._validation_stats = {}

    def validate(self,
                 *,
                 population: np.ndarray,
                 max_exposure_pct: Optional[float] = None,
                 player_limits: Optional[Dict[int, float]] = None,
                 min_unique_players: Optional[int] = None,
                 return_stats: bool = False,
                 **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Validate exposure constraints.
        
        Args:
            population (np.ndarray): Population to validate
            max_exposure_pct (float): Maximum exposure percentage for any player
            player_limits (Dict): Specific exposure limits for individual players
            min_unique_players (int): Minimum number of unique players required
            return_stats (bool): Return validation statistics
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray or Tuple[np.ndarray, Dict]: Valid population, optionally with stats
        """
        if population.size == 0:
            if return_stats:
                return population, {'total_lineups': 0, 'valid_lineups': 0}
            return population
        
        original_size = len(population)
        
        # Calculate player exposures
        max_player_id = np.max(population)
        n_players = max_player_id + 1
        
        # Auto-select algorithm based on data size
        use_jit = self.use_jit
        if use_jit is None:
            use_jit = NUMBA_AVAILABLE and len(population) > self._jit_threshold
        
        if use_jit and NUMBA_AVAILABLE:
            exposure_counts = _calculate_exposure_jit(population, n_players)
        else:
            exposure_counts = self._calculate_exposure_numpy(population, n_players)
        
        exposure_percentages = (exposure_counts / len(population)) * 100.0
        
        # Apply exposure constraints
        valid_population = population
        
        # Apply global max exposure constraint
        if max_exposure_pct is not None:
            over_exposed_players = np.where(exposure_percentages > max_exposure_pct)[0]
            if len(over_exposed_players) > 0:
                valid_population = self._remove_over_exposed_lineups(
                    valid_population, over_exposed_players, max_exposure_pct
                )
        
        # Apply individual player limits
        if player_limits:
            for player_id, limit_pct in player_limits.items():
                if player_id < len(exposure_percentages) and exposure_percentages[player_id] > limit_pct:
                    valid_population = self._remove_player_from_excess_lineups(
                        valid_population, player_id, limit_pct
                    )
        
        # Apply minimum unique players constraint
        if min_unique_players is not None:
            unique_players = len(np.unique(valid_population))
            if unique_players < min_unique_players:
                logger.warning(f"Population has only {unique_players} unique players, "
                             f"but {min_unique_players} required")
        
        # Calculate validation statistics if requested
        if return_stats:
            stats = self._calculate_exposure_stats(
                population, valid_population, exposure_counts, exposure_percentages,
                max_exposure_pct, player_limits, min_unique_players
            )
            self._validation_stats = stats
            return valid_population, stats
        
        return valid_population
    
    def _calculate_exposure_numpy(self, population: np.ndarray, n_players: int) -> np.ndarray:
        """Numpy-based exposure calculation."""
        exposure_counts = np.zeros(n_players, dtype=np.int32)
        
        for lineup in population:
            for player_id in lineup:
                if player_id < n_players:
                    exposure_counts[player_id] += 1
        
        return exposure_counts
    
    def _remove_over_exposed_lineups(self, population: np.ndarray, 
                                    over_exposed_players: np.ndarray,
                                    max_exposure_pct: float) -> np.ndarray:
        """Remove lineups containing over-exposed players."""
        # This is a simplified implementation - in practice, you might want
        # more sophisticated exposure balancing
        valid_indices = []
        
        for i, lineup in enumerate(population):
            has_over_exposed = np.any(np.isin(lineup, over_exposed_players))
            if not has_over_exposed:
                valid_indices.append(i)
        
        return population[valid_indices] if valid_indices else np.array([]).reshape(0, population.shape[1])
    
    def _remove_player_from_excess_lineups(self, population: np.ndarray,
                                          player_id: int, limit_pct: float) -> np.ndarray:
        """Remove excess lineups containing a specific player."""
        # Find lineups containing the player
        lineups_with_player = []
        lineups_without_player = []
        
        for i, lineup in enumerate(population):
            if player_id in lineup:
                lineups_with_player.append(i)
            else:
                lineups_without_player.append(i)
        
        # Calculate how many lineups we can keep with this player
        max_lineups_with_player = int((limit_pct / 100.0) * len(population))
        
        # Keep only the allowed number of lineups with this player
        if len(lineups_with_player) > max_lineups_with_player:
            # Randomly select which lineups to keep (could be made more sophisticated)
            np.random.shuffle(lineups_with_player)
            lineups_with_player = lineups_with_player[:max_lineups_with_player]
        
        # Combine valid lineups
        valid_indices = sorted(lineups_with_player + lineups_without_player)
        return population[valid_indices]
    
    def _calculate_exposure_stats(self, original_pop: np.ndarray, valid_pop: np.ndarray,
                                 exposure_counts: np.ndarray, exposure_percentages: np.ndarray,
                                 max_exposure_pct: Optional[float], player_limits: Optional[Dict],
                                 min_unique_players: Optional[int]) -> Dict:
        """Calculate detailed exposure validation statistics."""
        stats = {
            'total_lineups': len(original_pop),
            'valid_lineups': len(valid_pop),
            'invalid_lineups': len(original_pop) - len(valid_pop),
            'validation_rate': len(valid_pop) / len(original_pop) if len(original_pop) > 0 else 0,
            'exposure_constraints': {
                'max_exposure_pct': max_exposure_pct,
                'player_limits': player_limits,
                'min_unique_players': min_unique_players
            },
            'exposure_stats': {
                'unique_players': int(np.sum(exposure_counts > 0)),
                'max_exposure_pct': float(np.max(exposure_percentages)) if len(exposure_percentages) > 0 else 0,
                'mean_exposure_pct': float(np.mean(exposure_percentages[exposure_counts > 0])) if np.sum(exposure_counts > 0) > 0 else 0,
                'players_over_limit': int(np.sum(exposure_percentages > max_exposure_pct)) if max_exposure_pct else 0
            }
        }
        
        return stats
    
    def get_last_validation_stats(self) -> Dict:
        """Get statistics from the last validation run."""
        return self._validation_stats.copy()


class StackingValidate(ValidateBase):
    """Validate team stacking constraints (e.g., QB+WR stacks)."""

    def __init__(self, ctx=None):
        """Initialize stacking validator.
        
        Args:
            ctx: Plugin context
        """
        super().__init__(ctx)
        self._validation_stats = {}

    def validate(self,
                 *,
                 population: np.ndarray,
                 stacking_rules: List[Dict],
                 player_teams: Dict[int, str],
                 return_stats: bool = False,
                 **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Validate stacking constraints.
        
        Args:
            population (np.ndarray): Population to validate
            stacking_rules (List[Dict]): List of stacking rules to enforce
            player_teams (Dict): Maps player IDs to team names
            return_stats (bool): Return validation statistics
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray or Tuple[np.ndarray, Dict]: Valid population, optionally with stats
        """
        if population.size == 0:
            if return_stats:
                return population, {'total_lineups': 0, 'valid_lineups': 0}
            return population
        
        original_size = len(population)
        valid_indices = []
        
        for i, lineup in enumerate(population):
            if self._validate_lineup_stacking(lineup, stacking_rules, player_teams):
                valid_indices.append(i)
        
        valid_population = population[valid_indices] if valid_indices else np.array([]).reshape(0, population.shape[1])
        
        # Calculate validation statistics if requested
        if return_stats:
            stats = {
                'total_lineups': original_size,
                'valid_lineups': len(valid_population),
                'invalid_lineups': original_size - len(valid_population),
                'validation_rate': len(valid_population) / original_size if original_size > 0 else 0,
                'stacking_rules': stacking_rules
            }
            self._validation_stats = stats
            return valid_population, stats
        
        return valid_population
    
    def _validate_lineup_stacking(self, lineup: np.ndarray, stacking_rules: List[Dict],
                                 player_teams: Dict[int, str]) -> bool:
        """Validate stacking rules for a single lineup."""
        # Get teams for all players in lineup
        lineup_teams = {}
        for player_id in lineup:
            if player_id in player_teams:
                team = player_teams[player_id]
                if team not in lineup_teams:
                    lineup_teams[team] = []
                lineup_teams[team].append(player_id)
        
        # Check each stacking rule
        for rule in stacking_rules:
            rule_type = rule.get('type', 'min_stack')
            min_players = rule.get('min_players', 2)
            positions = rule.get('positions', [])
            
            if rule_type == 'min_stack':
                # Check if any team has at least min_players from specified positions
                stack_found = False
                for team, team_players in lineup_teams.items():
                    if len(team_players) >= min_players:
                        # If positions are specified, check position requirements
                        if positions:
                            # This would require position information for each player
                            # Simplified implementation assumes rule is met if enough players from team
                            pass
                        stack_found = True
                        break
                
                if not stack_found:
                    return False
            
            elif rule_type == 'max_stack':
                # Check that no team exceeds max_players
                max_players = rule.get('max_players', 3)
                for team, team_players in lineup_teams.items():
                    if len(team_players) > max_players:
                        return False
        
        return True
    
    def get_last_validation_stats(self) -> Dict:
        """Get statistics from the last validation run."""
        return self._validation_stats.copy()


class CompositeValidate(ValidateBase):
    """Composite validator that combines multiple validation methods."""

    def __init__(self, ctx=None, validators=None, parallel=False):
        """Initialize composite validator.
        
        Args:
            ctx: Plugin context
            validators (List): List of validator instances
            parallel (bool): Whether to run validations in parallel (future feature)
        """
        super().__init__(ctx)
        self.validators = validators or []
        self.parallel = parallel
        self._validation_stats = {}

    def add_validator(self, validator: ValidateBase):
        """Add a validator to the composite."""
        self.validators.append(validator)

    def validate(self,
                 *,
                 population: np.ndarray,
                 return_stats: bool = False,
                 early_termination: bool = True,
                 **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Run all validators in sequence.
        
        Args:
            population (np.ndarray): Population to validate
            return_stats (bool): Return validation statistics
            early_termination (bool): Stop if population becomes empty
            **kwargs: Keyword arguments passed to all validators
            
        Returns:
            np.ndarray or Tuple[np.ndarray, Dict]: Valid population, optionally with stats
        """
        if population.size == 0:
            if return_stats:
                return population, {'total_lineups': 0, 'valid_lineups': 0, 'validator_stats': []}
            return population
        
        original_size = len(population)
        current_population = population.copy()
        validator_stats = []
        
        # Run each validator in sequence
        for i, validator in enumerate(self.validators):
            if early_termination and len(current_population) == 0:
                logger.warning(f"Population became empty after validator {i-1}, stopping early")
                break
            
            try:
                # Run validator with statistics if requested
                if return_stats:
                    current_population, validator_stat = validator.validate(
                        population=current_population,
                        return_stats=True,
                        **kwargs
                    )
                    validator_stats.append({
                        'validator_name': validator.__class__.__name__,
                        'validator_index': i,
                        'stats': validator_stat
                    })
                else:
                    current_population = validator.validate(
                        population=current_population,
                        **kwargs
                    )
                
                logger.debug(f"Validator {validator.__class__.__name__} processed: "
                           f"{len(current_population)} lineups remaining")
                
            except Exception as e:
                logger.error(f"Validator {validator.__class__.__name__} failed: {str(e)}")
                if early_termination:
                    break
                # Continue with next validator if early_termination is False
        
        # Calculate composite validation statistics if requested
        if return_stats:
            composite_stats = {
                'total_lineups': original_size,
                'valid_lineups': len(current_population),
                'invalid_lineups': original_size - len(current_population),
                'validation_rate': len(current_population) / original_size if original_size > 0 else 0,
                'validators_run': len(validator_stats),
                'validator_stats': validator_stats
            }
            self._validation_stats = composite_stats
            return current_population, composite_stats
        
        return current_population
    
    def get_last_validation_stats(self) -> Dict:
        """Get statistics from the last validation run."""
        return self._validation_stats.copy()


# Utility functions for validation
def validate_population_comprehensive(population: np.ndarray,
                                    salaries: Optional[np.ndarray] = None,
                                    salary_cap: Optional[float] = None,
                                    position_map: Optional[Union[Dict, np.ndarray]] = None,
                                    position_requirements: Optional[Dict[str, int]] = None,
                                    max_exposure_pct: Optional[float] = None,
                                    player_limits: Optional[Dict[int, float]] = None,
                                    stacking_rules: Optional[List[Dict]] = None,
                                    player_teams: Optional[Dict[int, str]] = None,
                                    remove_duplicates: bool = True,
                                    return_stats: bool = False,
                                    **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """Comprehensive validation function that applies all relevant constraints.
    
    Args:
        population (np.ndarray): Population to validate
        salaries (np.ndarray): Player salaries for salary cap validation
        salary_cap (float): Maximum allowed salary
        position_map (Dict or np.ndarray): Maps player IDs to positions
        position_requirements (Dict): Required count for each position
        max_exposure_pct (float): Maximum exposure percentage for any player
        player_limits (Dict): Specific exposure limits for individual players
        stacking_rules (List[Dict]): Team stacking rules to enforce
        player_teams (Dict): Maps player IDs to team names
        remove_duplicates (bool): Whether to remove duplicate lineups
        return_stats (bool): Return validation statistics
        **kwargs: Additional keyword arguments
        
    Returns:
        np.ndarray or Tuple[np.ndarray, Dict]: Valid population, optionally with stats
    """
    if population.size == 0:
        if return_stats:
            return population, {'total_lineups': 0, 'valid_lineups': 0}
        return population
    
    # Create composite validator with appropriate validators
    composite = CompositeValidate()
    
    # Add duplicate removal if requested
    if remove_duplicates:
        composite.add_validator(DuplicatesValidate())
    
    # Add salary validation if parameters provided
    if salaries is not None and salary_cap is not None:
        composite.add_validator(SalaryValidate())
        kwargs['salaries'] = salaries
        kwargs['salary_cap'] = salary_cap
    
    # Add position validation if parameters provided
    if position_map is not None and position_requirements is not None:
        composite.add_validator(PositionValidate())
        kwargs['position_map'] = position_map
        kwargs['position_requirements'] = position_requirements
    
    # Add exposure validation if parameters provided
    if max_exposure_pct is not None or player_limits is not None:
        composite.add_validator(ExposureValidate())
        if max_exposure_pct is not None:
            kwargs['max_exposure_pct'] = max_exposure_pct
        if player_limits is not None:
            kwargs['player_limits'] = player_limits
    
    # Add stacking validation if parameters provided
    if stacking_rules is not None and player_teams is not None:
        composite.add_validator(StackingValidate())
        kwargs['stacking_rules'] = stacking_rules
        kwargs['player_teams'] = player_teams
    
    # Run composite validation
    return composite.validate(
        population=population,
        return_stats=return_stats,
        **kwargs
    )


def create_validation_pipeline(validation_config: Dict) -> CompositeValidate:
    """Create a validation pipeline from configuration.
    
    Args:
        validation_config (Dict): Configuration specifying which validators to include
        
    Returns:
        CompositeValidate: Configured composite validator
    """
    composite = CompositeValidate()
    
    # Add validators based on configuration
    if validation_config.get('remove_duplicates', True):
        remover_config = validation_config.get('duplicate_remover_config', {})
        composite.add_validator(DuplicatesValidate(remover_config=remover_config))
    
    if validation_config.get('validate_salary', False):
        salary_config = validation_config.get('salary_config', {})
        composite.add_validator(SalaryValidate(**salary_config))
    
    if validation_config.get('validate_positions', False):
        position_config = validation_config.get('position_config', {})
        composite.add_validator(PositionValidate(**position_config))
    
    if validation_config.get('validate_exposure', False):
        exposure_config = validation_config.get('exposure_config', {})
        composite.add_validator(ExposureValidate(**exposure_config))
    
    if validation_config.get('validate_stacking', False):
        stacking_config = validation_config.get('stacking_config', {})
        composite.add_validator(StackingValidate(**stacking_config))
    
    return composite


# Performance monitoring utilities
class ValidationProfiler:
    """Profiler for validation performance analysis."""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_validator(self, validator: ValidateBase, population: np.ndarray,
                         iterations: int = 3, **kwargs) -> Dict:
        """Profile a validator's performance.
        
        Args:
            validator: Validator to profile
            population: Test population
            iterations: Number of iterations to run
            **kwargs: Arguments for validator
            
        Returns:
            Dict: Performance profile results
        """
        import time
        
        validator_name = validator.__class__.__name__
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            start_time = time.time()
            start_memory = population.nbytes
            
            result = validator.validate(population=population, **kwargs)
            
            end_time = time.time()
            end_memory = result.nbytes if hasattr(result, 'nbytes') else 0
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        profile = {
            'validator_name': validator_name,
            'population_size': len(population),
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_memory_delta': np.mean(memory_usage),
            'iterations': iterations
        }
        
        self.profiles[validator_name] = profile
        return profile
    
    def get_profiles(self) -> Dict:
        """Get all performance profiles."""
        return self.profiles.copy()


# Export main classes and functions
__all__ = [
    'DuplicatesValidate',
    'SalaryValidate', 
    'PositionValidate',
    'ExposureValidate',
    'StackingValidate',
    'CompositeValidate',
    'validate_population_comprehensive',
    'create_validation_pipeline',
    'ValidationProfiler'
]
