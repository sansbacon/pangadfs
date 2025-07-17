# tests/test_mutate_optimized.py
# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as npt
import pytest
import time

from pangadfs.mutate import MutateDefault


class TestMutateOptimizations:
    """Test optimized mutation methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mutator = MutateDefault()
        np.random.seed(42)
        self.population = np.random.randint(0, 100, size=(1000, 10))
        self.small_population = np.random.randint(0, 20, size=(10, 5))

    def test_swap_mutation_basic(self):
        """Test basic swap mutation functionality."""
        result = self.mutator.mutate(population=self.small_population, 
                                   mutation_rate=0.1, method='swap')
        
        # Check shape preservation
        assert result.shape == self.small_population.shape
        
        # Check that some mutations occurred (with high probability)
        # Note: with mutation_rate=0.1 and 50 genes, we expect ~5 mutations
        assert not np.array_equal(result, self.small_population)

    def test_uniform_mutation(self):
        """Test uniform mutation method."""
        result = self.mutator.mutate(population=self.small_population,
                                   mutation_rate=0.2, method='uniform',
                                   min_val=0, max_val=50)
        
        assert result.shape == self.small_population.shape
        # Check that mutated values are within specified range
        assert np.all(result >= 0) and np.all(result < 50)

    def test_gaussian_mutation(self):
        """Test Gaussian mutation method."""
        float_pop = self.small_population.astype(float)
        result = self.mutator.mutate(population=float_pop,
                                   mutation_rate=0.1, method='gaussian',
                                   sigma=2.0)
        
        assert result.shape == float_pop.shape
        # Some values should have changed
        assert not np.array_equal(result, float_pop)

    def test_position_swap_mutation(self):
        """Test position-aware swap mutation."""
        result = self.mutator.mutate(population=self.small_population,
                                   mutation_rate=0.3, method='position_swap')
        
        assert result.shape == self.small_population.shape
        
        # Check that each column contains the same elements (just reordered)
        for col in range(self.small_population.shape[1]):
            original_col = np.sort(self.small_population[:, col])
            result_col = np.sort(result[:, col])
            npt.assert_array_equal(original_col, result_col)

    def test_adaptive_mutation(self):
        """Test adaptive mutation functionality."""
        # Create fitness array
        fitness = np.random.random(len(self.small_population))
        
        result = self.mutator.adaptive_mutate(population=self.small_population,
                                            fitness=fitness,
                                            base_rate=0.05)
        
        assert result.shape == self.small_population.shape

    def test_multi_method_mutation(self):
        """Test multi-method mutation."""
        result = self.mutator.multi_method_mutate(
            population=self.small_population,
            mutation_rate=0.1,
            methods=['swap', 'uniform'],
            weights=[0.7, 0.3]
        )
        
        assert result.shape == self.small_population.shape

    def test_zero_mutation_rate(self):
        """Test that zero mutation rate returns unchanged population."""
        result = self.mutator.mutate(population=self.small_population,
                                   mutation_rate=0.0, method='swap')
        
        npt.assert_array_equal(result, self.small_population)

    def test_performance_comparison(self):
        """Compare performance of optimized vs original implementation."""
        large_pop = np.random.randint(0, 1000, size=(5000, 20))
        
        # Test optimized version
        start_time = time.time()
        for _ in range(10):
            result_opt = self.mutator._mutate_swap_fast(large_pop, 0.05)
        opt_time = time.time() - start_time
        
        # Test original-style implementation
        start_time = time.time()
        for _ in range(10):
            mutate = (np.random.binomial(n=1, p=0.05, size=large_pop.size)
                     .reshape(large_pop.shape).astype(bool))
            swap = large_pop[np.random.choice(len(large_pop), size=len(large_pop), replace=False)]
            result_orig = np.where(mutate, swap, large_pop)
        orig_time = time.time() - start_time
        
        print(f"Optimized time: {opt_time:.4f}s, Original time: {orig_time:.4f}s")
        # Optimized version should be faster or at least comparable
        assert opt_time <= orig_time * 1.5  # Allow 50% tolerance

    def test_mutation_rate_bounds(self):
        """Test behavior with extreme mutation rates."""
        # Very low mutation rate
        result_low = self.mutator.mutate(population=self.small_population,
                                       mutation_rate=0.001, method='swap')
        assert result_low.shape == self.small_population.shape
        
        # High mutation rate
        result_high = self.mutator.mutate(population=self.small_population,
                                        mutation_rate=0.9, method='swap')
        assert result_high.shape == self.small_population.shape
        # Should be very different from original
        assert np.sum(result_high != self.small_population) > self.small_population.size * 0.5

    def test_method_dispatch(self):
        """Test that method dispatch works correctly."""
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown mutation method"):
            self.mutator.mutate(population=self.small_population,
                              mutation_rate=0.1, method='invalid_method')

    def test_context_initialization(self):
        """Test initialization with context."""
        ctx = {'mutation_rate': 0.15}
        mutator_ctx = MutateDefault(ctx)
        
        assert mutator_ctx.default_mutation_rate == 0.15
        
        # Test that default is used when no rate is provided
        result = mutator_ctx.mutate(population=self.small_population, method='swap')
        assert result.shape == self.small_population.shape


class TestMutateMemoryOptimization:
    """Test memory optimization aspects."""
    
    def test_memory_efficiency(self):
        """Test that optimized version uses less memory for large populations."""
        # This is more of a conceptual test - the optimized version should
        # avoid creating unnecessary large arrays
        large_pop = np.random.randint(0, 1000, size=(10000, 50))
        mutator = MutateDefault()
        
        # Low mutation rate should be very memory efficient
        result = mutator._mutate_swap_fast(large_pop, 0.001)
        assert result.shape == large_pop.shape
        
        # The optimized version should handle this without memory issues
        # (compared to creating full swap arrays)

    def test_early_return_optimization(self):
        """Test that zero mutations return early."""
        mutator = MutateDefault()
        pop = np.array([[1, 2, 3], [4, 5, 6]])
        
        # With mutation_rate=0, should return copy without processing
        result = mutator._mutate_swap_fast(pop, 0.0)
        npt.assert_array_equal(result, pop)
        assert result is not pop  # Should be a copy


if __name__ == "__main__":
    # Run basic functionality tests
    test_opt = TestMutateOptimizations()
    test_opt.setup_method()
    
    print("Testing basic mutation methods...")
    test_opt.test_swap_mutation_basic()
    test_opt.test_uniform_mutation()
    test_opt.test_gaussian_mutation()
    test_opt.test_position_swap_mutation()
    print("✅ Basic mutation tests passed")
    
    print("Testing advanced features...")
    test_opt.test_adaptive_mutation()
    test_opt.test_multi_method_mutation()
    print("✅ Advanced feature tests passed")
    
    print("Testing edge cases...")
    test_opt.test_zero_mutation_rate()
    test_opt.test_mutation_rate_bounds()
    print("✅ Edge case tests passed")
    
    print("Testing performance...")
    test_opt.test_performance_comparison()
    print("✅ Performance tests passed")
    
    print("✅ All mutation optimization tests passed!")
