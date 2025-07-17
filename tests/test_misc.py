# tests/test_misc.py
# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import chisquare

from pangadfs.misc import diversity, multidimensional_shifting, exposure, parents


class TestDiversity:
    """Test diversity function and its implementations."""
    
    def test_diversity_basic(self):
        """Test basic diversity calculation."""
        np.random.seed(42)
        population = np.random.randint(0, 50, size=(10, 5))
        
        result = diversity(population)
        
        # Check shape
        assert result.shape == (10, 10)
        
        # Check symmetry
        npt.assert_array_equal(result, result.T)
        
        # Check diagonal (self-overlap should be maximum)
        diagonal = np.diag(result)
        assert np.all(diagonal >= result.max(axis=1) - diagonal)

    def test_diversity_equivalence(self):
        """Test that different implementations produce the same result."""
        np.random.seed(42)
        population = np.random.randint(0, 50, size=(100, 10))
        
        # Test with original simple implementation for comparison
        uniques = np.unique(population)
        a = (population[..., None] == uniques).sum(1)
        expected = np.einsum('ij,kj->ik', a, a)
        
        result = diversity(population)
        
        npt.assert_array_equal(result, expected)

    def test_diversity_small_case(self):
        """Test diversity with a small, known case."""
        # Simple case where we can manually verify
        population = np.array([
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4]
        ])
        
        result = diversity(population)
        
        # Expected overlaps:
        # [0,0] vs [0,0]: 3 (all same)
        # [0,0] vs [1,1]: 2 (1,2 in common)
        # [0,0] vs [2,2]: 2 (1,3 in common)
        # [1,1] vs [2,2]: 2 (1,4 in common)
        expected = np.array([
            [3, 2, 2],
            [2, 3, 2],
            [2, 2, 3]
        ])
        
        npt.assert_array_equal(result, expected)


class TestMultidimensionalShifting:
    """Test multidimensional_shifting function and its implementations."""
    
    def test_multidimensional_shifting_basic(self):
        """Test basic functionality of multidimensional shifting."""
        np.random.seed(42)
        
        elements = np.arange(20)
        num_samples = 100
        sample_size = 5
        probs = np.random.dirichlet(np.ones(20), size=1)[0]
        
        result = multidimensional_shifting(elements, num_samples, sample_size, probs)
        
        # Check shape
        assert result.shape == (num_samples, sample_size)
        
        # Check all elements are from the original set
        assert np.all(np.isin(result, elements))
        
        # Check no duplicates within each sample
        for i in range(num_samples):
            assert len(np.unique(result[i])) == sample_size

    def test_multidimensional_shifting_with_pandas_index(self):
        """Test with pandas-like object that has to_numpy method."""
        class MockPandasIndex:
            def __init__(self, data):
                self.data = np.array(data)
            
            def to_numpy(self):
                return self.data
        
        np.random.seed(42)
        elements = MockPandasIndex([10, 20, 30, 40, 50])
        num_samples = 10
        sample_size = 3
        probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        result = multidimensional_shifting(elements, num_samples, sample_size, probs)
        
        assert result.shape == (num_samples, sample_size)
        assert np.all(np.isin(result, [10, 20, 30, 40, 50]))

    def test_sampling_distribution(self):
        """Test that sampling follows the probability distribution."""
        np.random.seed(42)

        # Setup
        num_elements = 20
        num_samples = 100_000
        sample_size = 5

        elements = np.arange(num_elements)
        probs = np.random.dirichlet(np.ones(num_elements), size=1)[0].astype(np.float32)

        # Run sampling
        samples = multidimensional_shifting(elements, num_samples, sample_size, probs)

        # Count how often each element was selected
        counts = np.bincount(samples.ravel(), minlength=num_elements)
        
        # Chi-squared test
        expected = probs * num_samples * sample_size
        chi2, p_value = chisquare(counts, expected)

        # Assert distribution is close (null hypothesis: observed == expected)
        assert p_value > 0.01, f"Sampling distribution deviates significantly (p={p_value:.4f})"


class TestExposure:
    """Test exposure function."""
    
    def test_exposure_basic(self):
        """Test basic exposure calculation."""
        population = np.array([
            [1, 2, 3],
            [1, 2, 4],
            [2, 3, 4],
            [1, 3, 4]
        ])
        
        result = exposure(population)
        
        # Expected counts: 1 appears 3 times, 2 appears 3 times, 3 appears 3 times, 4 appears 3 times
        expected = {1: 3, 2: 3, 3: 3, 4: 3}
        
        assert result == expected

    def test_exposure_empty(self):
        """Test exposure with empty population."""
        population = np.array([]).reshape(0, 3)
        
        result = exposure(population)
        
        assert result == {}


class TestParents:
    """Test parents function."""
    
    def test_parents_even_split(self):
        """Test parents function with even population."""
        population = np.arange(20).reshape(10, 2)
        
        fathers, mothers = parents(population)
        
        assert len(fathers) == len(mothers) == 5
        npt.assert_array_equal(fathers, population[:5])
        npt.assert_array_equal(mothers, population[5:])

    def test_parents_odd_split(self):
        """Test parents function with odd population."""
        population = np.arange(18).reshape(9, 2)
        
        fathers, mothers = parents(population)
        
        # Should take minimum size
        assert len(fathers) == len(mothers) == 4
        npt.assert_array_equal(fathers, population[:4])
        npt.assert_array_equal(mothers, population[5:9])

    def test_parents_single_individual(self):
        """Test parents function with single individual."""
        population = np.array([[1, 2]])
        
        fathers, mothers = parents(population)
        
        # Both should be empty since min(1, 0) = 0
        assert len(fathers) == len(mothers) == 0


if __name__ == "__main__":
    # Run basic tests
    test_diversity = TestDiversity()
    test_diversity.test_diversity_basic()
    test_diversity.test_diversity_equivalence()
    test_diversity.test_diversity_small_case()
    print("✅ Diversity tests passed")
    
    test_shifting = TestMultidimensionalShifting()
    test_shifting.test_multidimensional_shifting_basic()
    test_shifting.test_multidimensional_shifting_with_pandas_index()
    test_shifting.test_sampling_distribution()
    print("✅ Multidimensional shifting tests passed")
    
    test_exp = TestExposure()
    test_exp.test_exposure_basic()
    test_exp.test_exposure_empty()
    print("✅ Exposure tests passed")
    
    test_par = TestParents()
    test_par.test_parents_even_split()
    test_par.test_parents_odd_split()
    test_par.test_parents_single_individual()
    print("✅ Parents tests passed")
    
    print("✅ All tests passed!")
