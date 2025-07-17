# tests/test_ga_enhanced.py
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import time
from unittest.mock import Mock, patch
import pandas as pd

from pangadfs.ga import GeneticAlgorithm


class TestGAEnhancements:
    """Test enhanced GA functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ctx = {
            'test': True,
            'ga_settings': {
                'population_size': 50,
                'n_generations': 10
            }
        }
        
    def test_ga_initialization_with_performance_features(self):
        """Test GA initialization with performance features."""
        # Test with performance monitoring enabled
        ga = GeneticAlgorithm(
            ctx=self.ctx,
            enable_caching=True,
            performance_monitoring=True
        )
        
        assert ga.enable_caching is True
        assert ga.performance_monitoring is True
        assert ga._cache_hits == 0
        assert ga._cache_misses == 0
        assert ga._operation_times == {}
        
        # Test with performance monitoring disabled
        ga_no_perf = GeneticAlgorithm(
            ctx=self.ctx,
            enable_caching=False,
            performance_monitoring=False
        )
        
        assert ga_no_perf.enable_caching is False
        assert ga_no_perf.performance_monitoring is False
        assert ga_no_perf._operation_times is None

    def test_plugin_caching(self):
        """Test plugin caching functionality."""
        ga = GeneticAlgorithm(ctx=self.ctx, enable_caching=True)
        
        # Mock a plugin manager
        mock_manager = Mock()
        mock_plugin = Mock()
        mock_manager.driver = mock_plugin
        ga.driver_managers['test'] = mock_manager
        
        # First call should miss cache
        plugin1 = ga.get_plugin('test')
        assert ga._cache_misses == 1
        assert ga._cache_hits == 0
        
        # Second call should hit cache
        plugin2 = ga.get_plugin('test')
        assert ga._cache_hits == 1
        assert plugin1 is plugin2

    def test_array_contiguity_optimization(self):
        """Test that arrays are made contiguous for performance."""
        ga = GeneticAlgorithm(ctx=self.ctx, performance_monitoring=True)
        
        # Create non-contiguous array
        arr = np.random.random((10, 5))
        non_contiguous = arr[::2, ::2]  # Create non-contiguous view
        assert not non_contiguous.flags['C_CONTIGUOUS']
        
        # Mock plugin
        mock_plugin = Mock()
        mock_plugin.fitness.return_value = np.random.random(5)
        ga._plugin_cache['fitness'] = mock_plugin
        
        # Call fitness method
        points = np.random.random(25)
        ga.fitness(population=non_contiguous, points=points)
        
        # Check that the plugin received a contiguous array
        called_args = mock_plugin.fitness.call_args
        called_population = called_args[1]['population']
        assert called_population.flags['C_CONTIGUOUS']

    def test_input_validation(self):
        """Test input validation in GA methods."""
        ga = GeneticAlgorithm(ctx=self.ctx)
        
        # Test crossover validation
        with pytest.raises(ValueError, match="Population cannot be None or empty"):
            ga.crossover(population=None)
        
        with pytest.raises(ValueError, match="Population cannot be None or empty"):
            ga.crossover(population=np.array([]))
        
        # Test fitness validation
        with pytest.raises(ValueError, match="Population and points cannot be None"):
            ga.fitness(population=None, points=None)
        
        # Test mutate validation
        with pytest.raises(ValueError, match="Population cannot be None"):
            ga.mutate(population=None)
        
        # Test populate validation
        with pytest.raises(ValueError, match="pospool, posmap, and population_size are required"):
            ga.populate(pospool=None, posmap=None, population_size=None)

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        ga = GeneticAlgorithm(ctx=self.ctx, performance_monitoring=True)
        
        # Mock plugin
        mock_plugin = Mock()
        mock_plugin.fitness.return_value = np.random.random(10)
        ga._plugin_cache['fitness'] = mock_plugin
        
        # Call method multiple times
        population = np.random.randint(0, 100, (10, 5))
        points = np.random.random(500)
        
        for _ in range(3):
            ga.fitness(population=population, points=points)
        
        # Check performance stats
        stats = ga.get_performance_stats()
        assert stats['performance_monitoring'] is True
        assert 'fitness' in stats['operation_stats']
        assert stats['operation_stats']['fitness']['count'] == 3

    def test_cache_management(self):
        """Test cache management functionality."""
        ga = GeneticAlgorithm(ctx=self.ctx, enable_caching=True)
        
        # Add some items to cache
        ga._plugin_cache['test'] = Mock()
        ga._array_cache['test'] = np.array([1, 2, 3])
        ga._cache_hits = 5
        ga._cache_misses = 2
        
        # Clear cache
        ga.clear_cache()
        
        assert len(ga._plugin_cache) == 0
        assert len(ga._array_cache) == 0
        assert ga._cache_hits == 0
        assert ga._cache_misses == 0

    def test_default_mutation_rate(self):
        """Test default mutation rate setting."""
        ga = GeneticAlgorithm(ctx=self.ctx)
        
        # Mock plugin
        mock_plugin = Mock()
        mock_plugin.mutate.return_value = np.random.randint(0, 100, (10, 5))
        ga._plugin_cache['mutate'] = mock_plugin
        
        population = np.random.randint(0, 100, (10, 5))
        
        # Call without mutation_rate
        ga.mutate(population=population)
        
        # Check that default rate was used
        called_args = mock_plugin.mutate.call_args
        assert called_args[1]['mutation_rate'] == 0.05

    def test_pool_caching(self):
        """Test pool result caching."""
        ga = GeneticAlgorithm(ctx=self.ctx, enable_caching=True)
        
        # Mock plugin
        mock_plugin = Mock()
        test_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        mock_plugin.pool.return_value = test_df
        ga._plugin_cache['pool'] = mock_plugin
        
        # First call
        result1 = ga.pool(csvpth='test.csv')
        assert ga._cache_misses == 1
        
        # Second call with same parameters should hit cache
        result2 = ga.pool(csvpth='test.csv')
        assert ga._cache_hits == 1
        
        # Results should be the same
        pd.testing.assert_frame_equal(result1, result2)

    def test_performance_stats_disabled(self):
        """Test performance stats when monitoring is disabled."""
        ga = GeneticAlgorithm(ctx=self.ctx, performance_monitoring=False)
        
        stats = ga.get_performance_stats()
        assert stats == {"performance_monitoring": False}

    def test_benchmark_operations(self):
        """Test benchmarking functionality."""
        ga = GeneticAlgorithm(ctx=self.ctx, performance_monitoring=True)
        
        # Mock plugin
        mock_plugin = Mock()
        ga._plugin_cache['fitness'] = mock_plugin
        
        # Run benchmark
        results = ga.benchmark_operations(iterations=10)
        
        assert 'plugin_retrieval' in results
        assert 'array_contiguous' in results
        assert isinstance(results['plugin_retrieval'], float)
        assert isinstance(results['array_contiguous'], float)

    def test_benchmark_without_monitoring(self):
        """Test benchmark when performance monitoring is disabled."""
        ga = GeneticAlgorithm(ctx=self.ctx, performance_monitoring=False)
        
        results = ga.benchmark_operations()
        assert results == {}

    def test_list_available_plugins(self):
        """Test listing available plugins."""
        # This is a static method, so we can test it directly
        with patch('pangadfs.ga.ExtensionManager') as mock_em:
            mock_ext = Mock()
            mock_ext.name = 'test_plugin'
            mock_em.return_value.extensions = [mock_ext]
            
            plugins = GeneticAlgorithm.list_available_plugins('test.namespace')
            assert plugins == ['test_plugin']

    def test_list_plugins_error_handling(self):
        """Test error handling in list_available_plugins."""
        with patch('pangadfs.ga.ExtensionManager') as mock_em:
            mock_em.side_effect = Exception("Test error")
            
            plugins = GeneticAlgorithm.list_available_plugins('test.namespace')
            assert plugins == []

    def test_operation_time_logging(self):
        """Test operation time logging."""
        ga = GeneticAlgorithm(ctx=self.ctx, performance_monitoring=True)
        
        # Log some operation times
        ga._log_operation_time('test_op', 0.1)
        ga._log_operation_time('test_op', 0.2)
        ga._log_operation_time('other_op', 0.05)
        
        assert 'test_op' in ga._operation_times
        assert 'other_op' in ga._operation_times
        assert len(ga._operation_times['test_op']) == 2
        assert len(ga._operation_times['other_op']) == 1

    def test_enhanced_error_handling(self):
        """Test enhanced error handling in GA methods."""
        ga = GeneticAlgorithm(ctx=self.ctx)
        
        # Mock plugin that raises an error
        mock_plugin = Mock()
        mock_plugin.fitness.side_effect = Exception("Test error")
        ga._plugin_cache['fitness'] = mock_plugin
        
        population = np.random.randint(0, 100, (10, 5))
        points = np.random.random(500)
        
        with pytest.raises(Exception, match="Test error"):
            ga.fitness(population=population, points=points)


class TestGAPerformanceIntegration:
    """Integration tests for GA performance features."""
    
    def test_full_workflow_with_monitoring(self):
        """Test full GA workflow with performance monitoring."""
        ga = GeneticAlgorithm(
            ctx={'test': True},
            enable_caching=True,
            performance_monitoring=True
        )
        
        # Mock all required plugins
        plugins = {}
        for namespace in ['fitness', 'mutate', 'crossover', 'select']:
            mock_plugin = Mock()
            plugins[namespace] = mock_plugin
            ga._plugin_cache[namespace] = mock_plugin
        
        # Configure mock returns
        plugins['fitness'].fitness.return_value = np.random.random(10)
        plugins['mutate'].mutate.return_value = np.random.randint(0, 100, (10, 5))
        plugins['crossover'].crossover.return_value = np.random.randint(0, 100, (10, 5))
        plugins['select'].select.return_value = np.random.randint(0, 100, (10, 5))
        
        # Run operations
        population = np.random.randint(0, 100, (10, 5))
        points = np.random.random(500)
        fitness_scores = np.random.random(10)
        
        ga.fitness(population=population, points=points)
        ga.mutate(population=population, mutation_rate=0.1)
        ga.crossover(population=population)
        ga.select(population=population, population_fitness=fitness_scores, n=5)
        
        # Check performance stats
        stats = ga.get_performance_stats()
        assert stats['performance_monitoring'] is True
        assert len(stats['operation_stats']) >= 4


if __name__ == "__main__":
    # Run basic functionality tests
    test_ga = TestGAEnhancements()
    test_ga.setup_method()
    
    print("Testing GA initialization...")
    test_ga.test_ga_initialization_with_performance_features()
    print("✅ GA initialization tests passed")
    
    print("Testing caching...")
    test_ga.test_plugin_caching()
    test_ga.test_cache_management()
    print("✅ Caching tests passed")
    
    print("Testing performance optimizations...")
    test_ga.test_array_contiguity_optimization()
    test_ga.test_performance_monitoring()
    print("✅ Performance optimization tests passed")
    
    print("Testing input validation...")
    test_ga.test_input_validation()
    print("✅ Input validation tests passed")
    
    print("Testing benchmarking...")
    test_ga.test_benchmark_operations()
    print("✅ Benchmarking tests passed")
    
    print("✅ All GA enhancement tests passed!")
