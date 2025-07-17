# tests/test_optimize_enhanced.py
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import time
from unittest.mock import Mock, MagicMock
import pandas as pd

from pangadfs.optimize import OptimizeDefault, OptimizationStats


class TestOptimizeEnhancements:
    """Test enhanced optimization features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = OptimizeDefault()
        
        # Create mock GA instance
        self.mock_ga = Mock()
        self.mock_ga.ctx = {
            'ga_settings': {
                'population_size': 20,
                'csvpth': 'test.csv',
                'points_column': 'points',
                'position_column': 'position',
                'salary_column': 'salary',
                'n_generations': 10,
                'stop_criteria': 5,
                'elite_divisor': 5,
                'mutation_rate': 0.05,
                'verbose': False,
                'select_method': 'roulette',
                'crossover_method': 'uniform'
            },
            'site_settings': {
                'posfilter': {},
                'flex_positions': [],
                'posmap': {0: 'QB', 1: 'RB', 2: 'WR'},
                'salary_cap': 50000
            }
        }
        
        # Mock the GA methods
        self.setup_mock_ga_methods()

    def setup_mock_ga_methods(self):
        """Set up mock GA methods with realistic behavior."""
        # Mock pool data
        pool_data = pd.DataFrame({
            'points': np.random.uniform(5, 25, 100),
            'salary': np.random.randint(3000, 12000, 100),
            'position': np.random.choice(['QB', 'RB', 'WR'], 100)
        })
        self.mock_ga.pool.return_value = pool_data
        
        # Mock pospool
        pospool_mock = {
            'QB': pool_data[pool_data['position'] == 'QB'],
            'RB': pool_data[pool_data['position'] == 'RB'],
            'WR': pool_data[pool_data['position'] == 'WR']
        }
        self.mock_ga.pospool.return_value = pospool_mock
        
        # Mock population generation
        def mock_populate(**kwargs):
            pop_size = kwargs.get('population_size', 20)
            return np.random.randint(0, 100, size=(pop_size, 3))
        self.mock_ga.populate.side_effect = mock_populate
        
        # Mock validation (return unchanged)
        self.mock_ga.validate.side_effect = lambda population, **kwargs: population
        
        # Mock fitness calculation
        def mock_fitness(population, points):
            return np.array([points[individual].sum() for individual in population])
        self.mock_ga.fitness.side_effect = mock_fitness
        
        # Mock selection
        def mock_select(population, population_fitness, n, method='roulette'):
            # Simple selection - return random subset
            indices = np.random.choice(len(population), size=n, replace=True)
            return population[indices]
        self.mock_ga.select.side_effect = mock_select
        
        # Mock crossover
        def mock_crossover(population, method='uniform'):
            # Simple crossover - return slightly modified population
            result = population.copy()
            for i in range(0, len(result)-1, 2):
                if np.random.random() < 0.5:
                    # Swap some genes
                    swap_point = np.random.randint(1, result.shape[1])
                    temp = result[i, swap_point:].copy()
                    result[i, swap_point:] = result[i+1, swap_point:]
                    result[i+1, swap_point:] = temp
            return result
        self.mock_ga.crossover.side_effect = mock_crossover
        
        # Mock mutation
        def mock_mutate(population, mutation_rate):
            result = population.copy()
            mask = np.random.random(population.shape) < mutation_rate
            if np.any(mask):
                result[mask] = np.random.randint(0, 100, size=np.sum(mask))
            return result
        self.mock_ga.mutate.side_effect = mock_mutate

    def test_optimization_stats_dataclass(self):
        """Test OptimizationStats dataclass functionality."""
        stats = OptimizationStats()
        assert stats.generation == 0
        assert stats.best_fitness == 0.0
        assert stats.diversity_score == 0.0
        
        # Test with custom values
        stats = OptimizationStats(generation=5, best_fitness=100.5, diversity_score=0.8)
        assert stats.generation == 5
        assert stats.best_fitness == 100.5
        assert stats.diversity_score == 0.8

    def test_diversity_calculation_fast(self):
        """Test fast diversity calculation."""
        # Create population with known diversity
        population = np.array([
            [1, 2, 3],
            [1, 2, 3],  # Duplicate
            [4, 5, 6],
            [7, 8, 9]
        ])
        
        diversity = self.optimizer._calculate_diversity_fast(population)
        expected_diversity = 3 / 4  # 3 unique out of 4 total
        assert abs(diversity - expected_diversity) < 0.01

    def test_adaptive_mutation_rate(self):
        """Test adaptive mutation rate calculation."""
        stats = OptimizationStats(
            diversity_score=0.5,
            avg_fitness=100.0,
            fitness_std=10.0
        )
        
        # Test with no improvement
        rate = self.optimizer._calculate_adaptive_mutation_rate(0.05, stats, 0)
        assert 0.01 <= rate <= 0.3
        
        # Test with many generations without improvement
        rate_high = self.optimizer._calculate_adaptive_mutation_rate(0.05, stats, 20)
        assert rate_high > rate  # Should be higher

    def test_elite_selection(self):
        """Test elite selection functionality."""
        fitness = np.array([10, 50, 30, 80, 20, 90, 40])
        elite_indices = self.optimizer._get_elite_indices(fitness, 3)
        
        # Should select top 3 fitness values (indices 5, 3, 1)
        elite_fitness = fitness[elite_indices]
        assert len(elite_indices) == 3
        assert np.all(elite_fitness >= 50)  # All should be high fitness

    def test_population_combination(self):
        """Test population combination logic."""
        elite = np.array([[1, 2, 3], [4, 5, 6]])
        mutated = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
        
        combined = self.optimizer._combine_populations(elite, mutated, 4)
        assert combined.shape[0] == 4
        assert combined.shape[1] == 3

    def test_advanced_termination_criteria(self):
        """Test advanced termination criteria."""
        # Test low diversity termination
        stats_low_diversity = OptimizationStats(
            diversity_score=0.03,
            n_unimproved=15
        )
        assert self.optimizer._check_advanced_termination(stats_low_diversity, 30)
        
        # Test convergence rate termination
        stats_low_convergence = OptimizationStats(
            convergence_rate=1e-7
        )
        assert self.optimizer._check_advanced_termination(stats_low_convergence, 60)
        
        # Test normal conditions (should not terminate)
        stats_normal = OptimizationStats(
            diversity_score=0.5,
            n_unimproved=3,
            convergence_rate=0.1
        )
        assert not self.optimizer._check_advanced_termination(stats_normal, 30)

    def test_optimization_summary(self):
        """Test optimization summary generation."""
        # Add some mock stats
        self.optimizer.stats_history = [
            OptimizationStats(generation=1, best_fitness=100, elapsed_time=1.0),
            OptimizationStats(generation=2, best_fitness=110, elapsed_time=2.0),
            OptimizationStats(generation=3, best_fitness=110, elapsed_time=3.0)
        ]
        
        summary = self.optimizer.get_optimization_summary()
        
        assert summary['total_generations'] == 3
        assert summary['final_best_fitness'] == 110
        assert summary['total_time'] == 3.0
        assert summary['avg_time_per_generation'] == 1.0
        assert summary['improvement_generations'] == 1

    def test_setup_optimization(self):
        """Test optimization setup phase."""
        setup_data = self.optimizer._setup_optimization(self.mock_ga)
        
        # Verify all required keys are present
        required_keys = ['pool', 'pospool', 'points', 'salaries', 'salary_cap', 
                        'initial_population', 'initial_fitness']
        for key in required_keys:
            assert key in setup_data
        
        # Verify data types
        assert isinstance(setup_data['points'], np.ndarray)
        assert isinstance(setup_data['salaries'], np.ndarray)
        assert isinstance(setup_data['initial_population'], np.ndarray)
        assert isinstance(setup_data['initial_fitness'], np.ndarray)

    def test_full_optimization_run(self):
        """Test a complete optimization run."""
        result = self.optimizer.optimize(self.mock_ga)
        
        # Verify result structure
        required_keys = ['population', 'fitness', 'best_lineup', 'best_score', 
                        'generations_run', 'final_diversity', 'stats_history', 'total_time']
        for key in required_keys:
            assert key in result
        
        # Verify data types and shapes
        assert isinstance(result['population'], np.ndarray)
        assert isinstance(result['fitness'], np.ndarray)
        assert isinstance(result['best_score'], (int, float, np.number))
        assert isinstance(result['stats_history'], list)
        assert result['generations_run'] > 0
        assert result['total_time'] > 0

    def test_performance_improvement(self):
        """Test that optimized version performs reasonably."""
        start_time = time.time()
        result = self.optimizer.optimize(self.mock_ga)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max for small test
        
        # Should have reasonable number of generations
        assert result['generations_run'] <= self.mock_ga.ctx['ga_settings']['n_generations']

    def test_stats_tracking(self):
        """Test that statistics are properly tracked."""
        result = self.optimizer.optimize(self.mock_ga)
        
        stats_history = result['stats_history']
        assert len(stats_history) > 0
        
        # Check that stats are properly populated
        for stats in stats_history:
            assert isinstance(stats, OptimizationStats)
            assert stats.generation > 0
            assert stats.best_fitness >= 0
            assert 0 <= stats.diversity_score <= 1
            assert stats.mutation_rate > 0

    def test_empty_stats_summary(self):
        """Test summary with empty stats history."""
        optimizer = OptimizeDefault()
        summary = optimizer.get_optimization_summary()
        assert summary == {}


class TestOptimizeMemoryEfficiency:
    """Test memory efficiency improvements."""
    
    def test_array_reuse(self):
        """Test that arrays are reused efficiently."""
        optimizer = OptimizeDefault()
        
        # Test elite indices calculation doesn't create unnecessary arrays
        fitness = np.random.random(1000)
        elite_indices = optimizer._get_elite_indices(fitness, 50)
        
        assert len(elite_indices) == 50
        assert np.all(elite_indices < len(fitness))

    def test_large_population_handling(self):
        """Test handling of larger populations."""
        optimizer = OptimizeDefault()
        
        # Test diversity calculation with larger population
        large_pop = np.random.randint(0, 100, size=(1000, 10))
        diversity = optimizer._calculate_diversity_fast(large_pop)
        
        assert 0 <= diversity <= 1
        assert isinstance(diversity, float)


if __name__ == "__main__":
    # Run basic functionality tests
    test_opt = TestOptimizeEnhancements()
    test_opt.setup_method()
    
    print("Testing optimization statistics...")
    test_opt.test_optimization_stats_dataclass()
    test_opt.test_diversity_calculation_fast()
    test_opt.test_adaptive_mutation_rate()
    print("✅ Statistics tests passed")
    
    print("Testing optimization components...")
    test_opt.test_elite_selection()
    test_opt.test_population_combination()
    test_opt.test_advanced_termination_criteria()
    print("✅ Component tests passed")
    
    print("Testing full optimization...")
    test_opt.test_setup_optimization()
    test_opt.test_full_optimization_run()
    test_opt.test_stats_tracking()
    print("✅ Full optimization tests passed")
    
    print("Testing performance...")
    test_opt.test_performance_improvement()
    print("✅ Performance tests passed")
    
    print("✅ All optimization enhancement tests passed!")
