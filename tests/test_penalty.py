# pangadfs/tests/test_penalty.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
import pytest

from pangadfs.penalty import (
    DistancePenalty, 
    DiversityPenalty, 
    OwnershipPenalty, 
    HighOwnershipPenalty,
    CorrelationPenalty,
    StackingReward,
    GameTheoryPenalty,
    VariancePenalty,
    PositionSpecificPenalty,
    AdaptivePenaltyScaler,
    OptimizedEnsemblePenalty
)


@pytest.fixture
def ownership():
    """Sample ownership percentages"""
    return np.array([0.05, 0.10, 0.20, 0.30, 0.01, 0.25, 0.15, 0.08, 0.12, 0.18])


@pytest.fixture
def correlation_matrix():
    """Sample correlation matrix"""
    n = 10
    matrix = np.zeros((n, n))
    # Fill with some sample correlations
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0  # Self-correlation is 1
            else:
                # Random correlation between -0.5 and 0.5
                matrix[i, j] = (np.random.random() - 0.5)
    
    # Make it symmetric
    matrix = (matrix + matrix.T) / 2
    
    return matrix


@pytest.fixture
def player_teams():
    """Sample player team assignments"""
    return np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])


@pytest.fixture
def player_positions():
    """Sample player position assignments"""
    return np.array(['QB', 'WR', 'RB', 'TE', 'QB', 'WR', 'RB', 'TE', 'WR', 'DST'])


@pytest.fixture
def player_variance():
    """Sample player variance values"""
    return np.array([5.0, 8.0, 6.0, 4.0, 7.0, 9.0, 3.0, 5.0, 6.0, 2.0])


def test_distance_penalty(pop):
    """Test DistancePenalty class"""
    penalty = DistancePenalty().penalty(population=pop)
    assert isinstance(penalty, np.ndarray)
    assert len(penalty) == len(pop)
    assert penalty.dtype == np.float64 or penalty.dtype == np.float32
    
    # Test with robust normalization
    penalty_robust = DistancePenalty().penalty(population=pop, robust=True)
    assert isinstance(penalty_robust, np.ndarray)
    assert len(penalty_robust) == len(pop)
    
    # Test with batch processing
    penalty_batch = DistancePenalty().penalty(population=pop, batch_size=10)
    assert isinstance(penalty_batch, np.ndarray)
    assert len(penalty_batch) == len(pop)


def test_diversity_penalty(pop):
    """Test DiversityPenalty class"""
    penalty = DiversityPenalty().penalty(population=pop)
    assert isinstance(penalty, np.ndarray)
    assert len(penalty) == len(pop)
    assert penalty.dtype == np.float64 or penalty.dtype == np.float32
    
    # Test with robust normalization
    penalty_robust = DiversityPenalty().penalty(population=pop, robust=True)
    assert isinstance(penalty_robust, np.ndarray)
    assert len(penalty_robust) == len(pop)
    
    # Test with batch processing
    penalty_batch = DiversityPenalty().penalty(population=pop, batch_size=10)
    assert isinstance(penalty_batch, np.ndarray)
    assert len(penalty_batch) == len(pop)


def test_ownership_penalty(ownership):
    """Test OwnershipPenalty class"""
    penalty = OwnershipPenalty().penalty(ownership=ownership)
    assert isinstance(penalty, np.ndarray)
    assert len(penalty) == len(ownership)
    assert penalty.dtype == np.float64 or penalty.dtype == np.float32
    
    # Test with different parameters
    penalty_params = OwnershipPenalty().penalty(
        ownership=ownership, base=2, boost=1, min_penalty=-10.0, max_penalty=-1.0
    )
    assert isinstance(penalty_params, np.ndarray)
    assert len(penalty_params) == len(ownership)
    
    # Verify min/max clipping
    assert np.all(penalty_params >= -10.0)
    assert np.all(penalty_params <= -1.0)


def test_high_ownership_penalty(ownership):
    """Test HighOwnershipPenalty class"""
    penalty = HighOwnershipPenalty().penalty(ownership=ownership)
    assert isinstance(penalty, np.ndarray)
    assert len(penalty) == len(ownership)
    assert penalty.dtype == np.float64 or penalty.dtype == np.float32
    
    # Verify that only high-owned players are penalized
    threshold = 0.15
    high_owned_mask = ownership > threshold
    low_owned_mask = ~high_owned_mask
    
    penalty_threshold = HighOwnershipPenalty().penalty(
        ownership=ownership, threshold=threshold
    )
    
    # Low-owned players should have zero penalty
    assert np.all(penalty_threshold[low_owned_mask] == 0)
    
    # High-owned players should have negative penalties
    if np.any(high_owned_mask):
        assert np.all(penalty_threshold[high_owned_mask] < 0)


def test_correlation_penalty(pop, correlation_matrix):
    """Test CorrelationPenalty class"""
    penalty = CorrelationPenalty().penalty(
        population=pop, correlation_matrix=correlation_matrix
    )
    assert isinstance(penalty, np.ndarray)
    assert len(penalty) == len(pop)
    assert penalty.dtype == np.float64 or penalty.dtype == np.float32
    
    # Test with different strength
    penalty_strength = CorrelationPenalty().penalty(
        population=pop, correlation_matrix=correlation_matrix, strength=2.0
    )
    assert isinstance(penalty_strength, np.ndarray)
    assert len(penalty_strength) == len(pop)


def test_stacking_reward(pop, player_teams, player_positions):
    """Test StackingReward class"""
    reward = StackingReward().penalty(
        population=pop, player_teams=player_teams, player_positions=player_positions
    )
    assert isinstance(reward, np.ndarray)
    assert len(reward) == len(pop)
    assert reward.dtype == np.float64 or reward.dtype == np.float32
    
    # Test with custom stack configs
    custom_configs = {'QB-WR': 3.0, 'QB-TE': 2.0, 'QB-WR-WR': 5.0}
    reward_custom = StackingReward().penalty(
        population=pop, player_teams=player_teams, 
        player_positions=player_positions, stack_configs=custom_configs
    )
    assert isinstance(reward_custom, np.ndarray)
    assert len(reward_custom) == len(pop)


def test_game_theory_penalty(pop, ownership):
    """Test GameTheoryPenalty class"""
    # Ensure ownership array is long enough for the population
    if len(ownership) < pop.max() + 1:
        extended_ownership = np.zeros(pop.max() + 1)
        extended_ownership[:len(ownership)] = ownership
        ownership = extended_ownership
    
    # Test GPP mode
    penalty_gpp = GameTheoryPenalty().penalty(
        population=pop, ownership=ownership, field_size=1000, tournament_type='gpp'
    )
    assert isinstance(penalty_gpp, np.ndarray)
    assert len(penalty_gpp) == len(pop)
    assert penalty_gpp.dtype == np.float64 or penalty_gpp.dtype == np.float32
    
    # Test cash mode
    penalty_cash = GameTheoryPenalty().penalty(
        population=pop, ownership=ownership, field_size=1000, tournament_type='cash'
    )
    assert isinstance(penalty_cash, np.ndarray)
    assert len(penalty_cash) == len(pop)


def test_variance_penalty(pop, player_variance):
    """Test VariancePenalty class"""
    # Ensure variance array is long enough for the population
    if len(player_variance) < pop.max() + 1:
        extended_variance = np.zeros(pop.max() + 1)
        extended_variance[:len(player_variance)] = player_variance
        player_variance = extended_variance
    
    # Test GPP mode
    penalty_gpp = VariancePenalty().penalty(
        population=pop, player_variance=player_variance, tournament_type='gpp'
    )
    assert isinstance(penalty_gpp, np.ndarray)
    assert len(penalty_gpp) == len(pop)
    assert penalty_gpp.dtype == np.float64 or penalty_gpp.dtype == np.float32
    
    # Test cash mode
    penalty_cash = VariancePenalty().penalty(
        population=pop, player_variance=player_variance, tournament_type='cash'
    )
    assert isinstance(penalty_cash, np.ndarray)
    assert len(penalty_cash) == len(pop)
    
    # Verify that GPP rewards high variance (negative penalties) and cash penalizes it (positive penalties)
    # This assumes the same lineup has the same variance in both tests
    if len(pop) > 0:
        high_variance_idx = np.argmax(np.sum(player_variance[pop], axis=1))
        assert penalty_gpp[high_variance_idx] < 0  # GPP rewards high variance
        assert penalty_cash[high_variance_idx] > 0  # Cash penalizes high variance


def test_position_specific_penalty(pop, player_positions):
    """Test PositionSpecificPenalty class"""
    # Ensure positions array is long enough for the population
    if len(player_positions) < pop.max() + 1:
        extended_positions = np.array(['UNKNOWN'] * (pop.max() + 1), dtype=object)
        extended_positions[:len(player_positions)] = player_positions
        player_positions = extended_positions
    
    penalty = PositionSpecificPenalty().penalty(
        population=pop, player_positions=player_positions
    )
    assert isinstance(penalty, np.ndarray)
    assert len(penalty) == len(pop)
    assert penalty.dtype == np.float64 or penalty.dtype == np.float32
    
    # Test with custom position weights
    custom_weights = {'QB': 2.0, 'RB': 1.5, 'WR': 1.0, 'TE': 0.5, 'DST': 0.0}
    penalty_custom = PositionSpecificPenalty().penalty(
        population=pop, player_positions=player_positions, position_weights=custom_weights
    )
    assert isinstance(penalty_custom, np.ndarray)
    assert len(penalty_custom) == len(pop)


def test_adaptive_penalty_scaler(pop):
    """Test AdaptivePenaltyScaler class"""
    # Create some base penalties and fitness scores
    base_penalties = np.random.normal(size=len(pop))
    fitness_scores = np.random.normal(size=len(pop))
    
    scaled_penalties = AdaptivePenaltyScaler().penalty(
        base_penalties=base_penalties, fitness_scores=fitness_scores
    )
    assert isinstance(scaled_penalties, np.ndarray)
    assert len(scaled_penalties) == len(pop)
    assert scaled_penalties.dtype == np.float64 or scaled_penalties.dtype == np.float32
    
    # Test with different target impact
    scaled_penalties_custom = AdaptivePenaltyScaler().penalty(
        base_penalties=base_penalties, fitness_scores=fitness_scores, target_impact=0.5
    )
    assert isinstance(scaled_penalties_custom, np.ndarray)
    assert len(scaled_penalties_custom) == len(pop)


def test_optimized_ensemble_penalty(pop, ownership, player_variance):
    """Test OptimizedEnsemblePenalty class"""
    # Ensure arrays are long enough for the population
    if len(ownership) < pop.max() + 1:
        extended_ownership = np.zeros(pop.max() + 1)
        extended_ownership[:len(ownership)] = ownership
        ownership = extended_ownership
    
    if len(player_variance) < pop.max() + 1:
        extended_variance = np.zeros(pop.max() + 1)
        extended_variance[:len(player_variance)] = player_variance
        player_variance = extended_variance
    
    # Create an ensemble with multiple penalty classes
    penalty_classes = [
        DistancePenalty,
        DiversityPenalty,
        VariancePenalty
    ]
    
    weights = {
        'DistancePenalty': 1.0,
        'DiversityPenalty': 0.5,
        'VariancePenalty': 2.0
    }
    
    ensemble = OptimizedEnsemblePenalty(
        penalty_classes=penalty_classes,
        weights=weights
    )
    
    # Test the ensemble penalty
    penalty = ensemble.penalty(
        population=pop,
        player_variance=player_variance,
        tournament_type='gpp'
    )
    
    assert isinstance(penalty, np.ndarray)
    assert len(penalty) == len(pop)
    assert penalty.dtype == np.float64 or penalty.dtype == np.float32


def test_ensemble_backward_compatibility(pop):
    """Test backward compatibility of EnsemblePenalty"""
    from pangadfs.penalty import EnsemblePenalty
    
    # Verify that EnsemblePenalty is the same as OptimizedEnsemblePenalty
    assert EnsemblePenalty is OptimizedEnsemblePenalty
    
    # Create a simple ensemble
    penalty_classes = [DistancePenalty]
    
    ensemble = EnsemblePenalty(penalty_classes=penalty_classes)
    
    # Test the ensemble penalty
    penalty = ensemble.penalty(population=pop)
    
    assert isinstance(penalty, np.ndarray)
    assert len(penalty) == len(pop)
