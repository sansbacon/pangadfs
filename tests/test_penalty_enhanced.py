# tests/test_penalty_enhanced.py
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import time
from unittest.mock import Mock, patch
import threading

from pangadfs.penalty import (
    OptimizedDistancePenalty,
    OptimizedDiversityPenalty,
    OptimizedOwnershipPenalty,
    OptimizedHighOwnershipPenalty,
    OptimizedCorrelationPenalty,
    OptimizedStackingReward,
    OptimizedGameTheoryPenalty,
    OptimizedVariancePenalty,
    OptimizedPositionSpecificPenalty,
    OptimizedAdaptivePenaltyScaler,
    OptimizedEnsemblePenalty,
    EnhancedCachedPenaltyMixin,
    PenaltyStats,
    create_optimized_penalty_ensemble,
    benchmark_penalty_performance
)


class TestOptimizedPenalties:
    """Test optimized penalty classes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ctx = {'test': True}
        self.population = np.random.randint(0, 100, size=(50, 9))
        self.ownership = np.random.random(100) * 0.5 + 0.01  # 0.01 to 0.51
        self.correlation_matrix = np.random.random((100, 100)) * 0.5
        
    def test_optimized_distance_penalty(self):
        """Test optimized distance penalty calculation."""
        penalty = OptimizedDistancePenalty(self.ctx)
        
        # Test basic functionality
        result = penalty.penalty(population=self.population)
        assert len(result) == len(self.population)
        assert isinstance(result, np.ndarray)
        
        # Test different distance metrics
        result_hamming = penalty.penalty(population=self.population, distance_metric='hamming')
        result_euclidean = penalty.penalty(population=self.population, distance_metric='euclidean')
        result_cosine = penalty.penalty(population=self.population, distance_metric='cosine')
        
        assert len(result_hamming) == len(self.population)
        assert len(result_euclidean) == len(self.population)
        assert len(result_cosine) == len(self.population)
        
        # Test batch processing
        large_population = np.random.randint(0, 100, size=(1000, 9))
        result_batched = penalty.penalty(population=large_population, batch_size=100)
        assert len(result_batched) == 1000
        assert penalty._stats.batch_count > 0

    def test_optimized_diversity_penalty(self):
        """Test optimized diversity penalty calculation."""
        penalty = OptimizedDiversityPenalty(self.ctx)
        
        result = penalty.penalty(population=self.population)
        assert len(result) == len(self.population)
        assert isinstance(result, np.ndarray)
        
        # Test robust vs non-robust normalization
        result_robust = penalty.penalty(population=self.population, robust=True)
        result_standard = penalty.penalty(population=self.population, robust=False)
        
        assert len(result_robust) == len(self.population)
        assert len(result_standard) == len(self.population)

    def test_optimized_ownership_penalty(self):
        """Test optimized ownership penalty calculation."""
        penalty = OptimizedOwnershipPenalty(self.ctx)
        
        # Test vectorized calculation
        result_vectorized = penalty.penalty(ownership=self.ownership, use_vectorized=True)
        assert len(result_vectorized) == len(self.ownership)
        
        # Test cached calculation
        result_cached = penalty.penalty(ownership=self.ownership, use_vectorized=False)
        assert len(result_cached) == len(self.ownership)
        
        # Test cache statistics
        cache_stats = penalty.get_cache_stats()
        assert isinstance(cache_stats, dict)

    def test_optimized_high_ownership_penalty(self):
        """Test optimized high ownership penalty calculation."""
        penalty = OptimizedHighOwnershipPenalty(self.ctx)
        
        result = penalty.penalty(ownership=self.ownership, threshold=0.15)
        assert len(result) == len(self.ownership)
        
        # Test that only high ownership players get penalties
        high_ownership = np.array([0.05, 0.20, 0.30, 0.10])
        result = penalty.penalty(ownership=high_ownership, threshold=0.15)
        
        # Only indices 1 and 2 should have penalties (ownership > 0.15)
        assert result[0] == 0  # Below threshold
        assert result[1] < 0   # Above threshold (penalty)
        assert result[2] < 0   # Above threshold (penalty)
        assert result[3] == 0  # Below threshold

    def test_optimized_correlation_penalty(self):
        """Test optimized correlation penalty calculation."""
        penalty = OptimizedCorrelationPenalty(self.ctx)
        
        result = penalty.penalty(
            population=self.population,
            correlation_matrix=self.correlation_matrix
        )
        assert len(result) == len(self.population)
        
        # Test sparse matrix operations
        result_sparse = penalty.penalty(
            population=self.population,
            correlation_matrix=self.correlation_matrix,
            use_sparse=True
        )
        assert len(result_sparse) == len(self.population)

    def test_optimized_stacking_reward(self):
        """Test optimized stacking reward calculation."""
        penalty = OptimizedStackingReward(self.ctx)
        
        # Create test data
        player_teams = np.random.randint(0, 32, 100)  # 32 teams
        player_positions = np.random.choice(['QB', 'RB', 'WR', 'TE', 'DST'], 100)
        
        result = penalty.penalty(
            population=self.population,
            player_teams=player_teams,
            player_positions=player_positions
        )
        assert len(result) == len(self.population)
        
        # Test custom stack configurations
        custom_configs = {'QB-WR': 3.0, 'QB-TE': 2.0}
        result_custom = penalty.penalty(
            population=self.population,
            player_teams=player_teams,
            player_positions=player_positions,
            stack_configs=custom_configs
        )
        assert len(result_custom) == len(self.population)

    def test_optimized_game_theory_penalty(self):
        """Test optimized game theory penalty calculation."""
        penalty = OptimizedGameTheoryPenalty(self.ctx)
        
        # Test GPP tournament
        result_gpp = penalty.penalty(
            population=self.population,
            ownership=self.ownership,
            field_size=10000,
            tournament_type='gpp'
        )
        assert len(result_gpp) == len(self.population)
        
        # Test cash game tournament
        result_cash = penalty.penalty(
            population=self.population,
            ownership=self.ownership,
            field_size=10000,
            tournament_type='cash'
        )
        assert len(result_cash) == len(self.population)

    def test_optimized_variance_penalty(self):
        """Test optimized variance penalty calculation."""
        penalty = OptimizedVariancePenalty(self.ctx)
        
        player_variance = np.random.random(100) * 10  # Variance values
        
        # Test GPP (should reward high variance)
        result_gpp = penalty.penalty(
            population=self.population,
            player_variance=player_variance,
            tournament_type='gpp'
        )
        assert len(result_gpp) == len(self.population)
        
        # Test cash game (should penalize high variance)
        result_cash = penalty.penalty(
            population=self.population,
            player_variance=player_variance,
            tournament_type='cash'
        )
        assert len(result_cash) == len(self.population)

    def test_optimized_position_specific_penalty(self):
        """Test optimized position-specific penalty calculation."""
        penalty = OptimizedPositionSpecificPenalty(self.ctx)
        
        player_positions = np.random.choice(['QB', 'RB', 'WR', 'TE', 'DST'], 100)
        
        result = penalty.penalty(
            population=self.population,
            player_positions=player_positions
        )
        assert len(result) == len(self.population)
        
        # Test custom position weights
        custom_weights = {'QB': 2.0, 'RB': 1.5, 'WR': 1.0, 'TE': 0.8, 'DST': 0.5}
        result_custom = penalty.penalty(
            population=self.population,
            player_positions=player_positions,
            position_weights=custom_weights
        )
        assert len(result_custom) == len(self.population)

    def test_optimized_adaptive_penalty_scaler(self):
        """Test optimized adaptive penalty scaler."""
        scaler = OptimizedAdaptivePenaltyScaler(self.ctx)
        
        base_penalties = np.random.random(50) * 2 - 1  # -1 to 1
        fitness_scores = np.random.random(50) * 100    # 0 to 100
        
        result = scaler.penalty(
            base_penalties=base_penalties,
            fitness_scores=fitness_scores,
            target_impact=0.2
        )
        assert len(result) == len(base_penalties)

    def test_optimized_ensemble_penalty(self):
        """Test optimized ensemble penalty calculation."""
        penalty_classes = [
            OptimizedDistancePenalty,
            OptimizedDiversityPenalty,
            OptimizedOwnershipPenalty
        ]
        
        weights = {
            'OptimizedDistancePenalty': 1.0,
            'OptimizedDiversityPenalty': 0.5,
            'OptimizedOwnershipPenalty': 2.0
        }
        
        ensemble = OptimizedEnsemblePenalty(
            ctx=self.ctx,
            penalty_classes=penalty_classes,
            weights=weights,
            use_parallel=False
        )
        
        result = ensemble.penalty(
            population=self.population,
            ownership=self.ownership
        )
        assert len(result) == len(self.population)
        
        # Test parallel processing
        ensemble_parallel = OptimizedEnsemblePenalty(
            ctx=self.ctx,
            penalty_classes=penalty_classes,
            weights=weights,
            use_parallel=True,
            max_workers=2
        )
        
        result_parallel = ensemble_parallel.penalty(
            population=self.population,
            ownership=self.ownership
        )
        assert len(result_parallel) == len(self.population)

    def test_enhanced_cached_penalty_mixin(self):
        """Test enhanced cached penalty mixin functionality."""
        
        class TestCachedPenalty(EnhancedCachedPenaltyMixin):
            def __init__(self, ctx=None):
                super().__init__()
                self.ctx = ctx
                
            def _calculate_penalty_impl(self, x):
                # Simulate expensive calculation
                time.sleep(0.001)
                return x * 2
        
        penalty = TestCachedPenalty(self.ctx)
        
        # Test caching
        start_time = time.time()
        result1 = penalty._calculate_penalty(5)
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        result2 = penalty._calculate_penalty(5)  # Should be cached
        second_call_time = time.time() - start_time
        
        assert result1 == result2 == 10
        assert second_call_time < first_call_time  # Cached call should be faster
        
        # Test cache statistics
        cache_stats = penalty.get_cache_stats()
        assert cache_stats['hits'] >= 1
        
        # Test cache management
        penalty.clear_cache()
        cache_stats_after_clear = penalty.get_cache_stats()
        assert cache_stats_after_clear['hits'] == 0

    def test_penalty_stats(self):
        """Test penalty statistics collection."""
        penalty = OptimizedDistancePenalty(self.ctx)
        
        # Run some calculations to generate stats
        penalty.penalty(population=self.population)
        penalty.penalty(population=self.population)
        
        assert penalty._stats.calculation_time > 0

    def test_create_optimized_penalty_ensemble(self):
        """Test penalty ensemble factory function."""
        penalty_configs = [
            {'type': 'distance', 'weight': 1.0},
            {'type': 'diversity', 'weight': 0.5},
            {'type': 'ownership', 'weight': 2.0},
            {'type': 'unknown_type', 'weight': 1.0}  # Should be ignored
        ]
        
        ensemble = create_optimized_penalty_ensemble(
            penalty_configs=penalty_configs,
            ctx=self.ctx,
            use_parallel=False
        )
        
        assert len(ensemble.penalty_instances) == 3  # Unknown type ignored
        assert 'OptimizedDistancePenalty' in ensemble.weights
        assert 'OptimizedDiversityPenalty' in ensemble.weights
        assert 'OptimizedOwnershipPenalty' in ensemble.weights

    def test_benchmark_penalty_performance(self):
        """Test penalty performance benchmarking."""
        penalty = OptimizedDistancePenalty(self.ctx)
        
        results = benchmark_penalty_performance(
            penalty_instance=penalty,
            population_sizes=[10, 50, 100],
            iterations=3
        )
        
        assert 'population_sizes' in results
        assert 'execution_times' in results
        assert 'memory_usage' in results
        assert len(results['execution_times']) == 3
        assert len(results['memory_usage']) == 3

    def test_performance_optimizations(self):
        """Test that performance optimizations work correctly."""
        penalty = OptimizedDistancePenalty(self.ctx)
        
        # Test with small population (no batching)
        small_pop = np.random.randint(0, 100, size=(10, 9))
        result_small = penalty.penalty(population=small_pop, batch_size=1000)
        assert len(result_small) == 10
        assert penalty._stats.batch_count == 0
        
        # Test with large population (batching)
        large_pop = np.random.randint(0, 100, size=(500, 9))
        penalty._stats.batch_count = 0  # Reset
        result_large = penalty.penalty(population=large_pop, batch_size=100)
        assert len(result_large) == 500
        assert penalty._stats.batch_count > 0

    def test_error_handling(self):
        """Test error handling in penalty calculations."""
        ensemble = OptimizedEnsemblePenalty(
            ctx=self.ctx,
            penalty_classes=[OptimizedDistancePenalty],
            use_parallel=False
        )
        
        # Test with invalid population (should handle gracefully)
        result = ensemble.penalty(population=np.array([]))
        assert len(result) == 0

    def test_thread_safety(self):
        """Test thread safety of cached penalty calculations."""
        penalty = OptimizedOwnershipPenalty(self.ctx)
        
        results = []
        errors = []
        
        def worker():
            try:
                result = penalty.penalty(ownership=self.ownership, use_vectorized=False)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert len(results) == 5
        
        # All results should be the same
        for result in results[1:]:
            np.testing.assert_array_almost_equal(results[0], result)


class TestPenaltyIntegration:
    """Integration tests for penalty system."""
    
    def test_full_penalty_workflow(self):
        """Test complete penalty calculation workflow."""
        # Create test data
        population = np.random.randint(0, 100, size=(100, 9))
        ownership = np.random.random(100) * 0.5 + 0.01
        correlation_matrix = np.random.random((100, 100)) * 0.5
        player_teams = np.random.randint(0, 32, 100)
        player_positions = np.random.choice(['QB', 'RB', 'WR', 'TE', 'DST'], 100)
        player_variance = np.random.random(100) * 10
        
        # Create ensemble with multiple penalty types
        penalty_configs = [
            {'type': 'distance', 'weight': 1.0},
            {'type': 'diversity', 'weight': 0.8},
            {'type': 'ownership', 'weight': 1.5},
            {'type': 'correlation', 'weight': 0.5},
            {'type': 'stacking', 'weight': 2.0},
            {'type': 'variance', 'weight': 1.2}
        ]
        
        ensemble = create_optimized_penalty_ensemble(
            penalty_configs=penalty_configs,
            use_parallel=True
        )
        
        # Calculate combined penalties
        result = ensemble.penalty(
            population=population,
            ownership=ownership,
            correlation_matrix=correlation_matrix,
            player_teams=player_teams,
            player_positions=player_positions,
            player_variance=player_variance,
            tournament_type='gpp'
        )
        
        assert len(result) == len(population)
        assert isinstance(result, np.ndarray)
        
        # Get penalty statistics
        stats = ensemble.get_penalty_stats()
        assert isinstance(stats, dict)
        assert len(stats) > 0


if __name__ == "__main__":
    # Run basic functionality tests
    test_penalties = TestOptimizedPenalties()
    test_penalties.setup_method()
    
    print("Testing optimized distance penalty...")
    test_penalties.test_optimized_distance_penalty()
    print("✅ Distance penalty tests passed")
    
    print("Testing optimized diversity penalty...")
    test_penalties.test_optimized_diversity_penalty()
    print("✅ Diversity penalty tests passed")
    
    print("Testing optimized ownership penalty...")
    test_penalties.test_optimized_ownership_penalty()
    print("✅ Ownership penalty tests passed")
    
    print("Testing optimized high ownership penalty...")
    test_penalties.test_optimized_high_ownership_penalty()
    print("✅ High ownership penalty tests passed")
    
    print("Testing optimized ensemble penalty...")
    test_penalties.test_optimized_ensemble_penalty()
    print("✅ Ensemble penalty tests passed")
    
    print("Testing enhanced caching...")
    test_penalties.test_enhanced_cached_penalty_mixin()
    print("✅ Enhanced caching tests passed")
    
    print("Testing penalty factory...")
    test_penalties.test_create_optimized_penalty_ensemble()
    print("✅ Penalty factory tests passed")
    
    print("Testing performance benchmarking...")
    test_penalties.test_benchmark_penalty_performance()
    print("✅ Performance benchmarking tests passed")
    
    print("Testing integration workflow...")
    test_integration = TestPenaltyIntegration()
    test_integration.test_full_penalty_workflow()
    print("✅ Integration workflow tests passed")
    
    print("✅ All penalty enhancement tests passed!")
