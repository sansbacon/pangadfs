# pangadfs/tests/test_duplicates.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
import pytest

from pangadfs.duplicates import (
    find_non_duplicate_flex,
    remove_internal_duplicates,
    remove_duplicate_lineups,
    find_duplicate_indices,
    validate_flex_positions,
    DuplicateRemover,
    benchmark_duplicate_removal
)


class TestFlexDuplicateRemoval:
    """Test FLEX duplicate removal functionality."""
    
    def test_find_non_duplicate_flex_basic(self):
        """Test basic FLEX duplicate removal."""
        main_lineups = np.array([[1, 2, 3], [4, 5, 6]])
        flex_options = np.array([[1, 7, 8], [4, 9, 10]])  # 1 and 4 are duplicates
        
        result = find_non_duplicate_flex(main_lineups, flex_options)
        
        assert len(result) == 2
        assert result[0] == 7  # First non-duplicate for lineup [1, 2, 3]
        assert result[1] == 9  # First non-duplicate for lineup [4, 5, 6]
    
    def test_find_non_duplicate_flex_all_duplicates(self):
        """Test FLEX removal when all options are duplicates."""
        main_lineups = np.array([[1, 2, 3]])
        flex_options = np.array([[1, 2, 3]])  # All are duplicates
        
        result = find_non_duplicate_flex(main_lineups, flex_options)
        
        assert len(result) == 1
        assert result[0] == 1  # Should use first option when all are duplicates
    
    def test_find_non_duplicate_flex_no_duplicates(self):
        """Test FLEX removal when no options are duplicates."""
        main_lineups = np.array([[1, 2, 3]])
        flex_options = np.array([[4, 5, 6]])  # No duplicates
        
        result = find_non_duplicate_flex(main_lineups, flex_options)
        
        assert len(result) == 1
        assert result[0] == 4  # Should use first option
    
    def test_find_non_duplicate_flex_mismatched_shapes(self):
        """Test error handling for mismatched shapes."""
        main_lineups = np.array([[1, 2, 3]])
        flex_options = np.array([[1, 7], [4, 9]])  # Different number of lineups
        
        with pytest.raises(ValueError, match="Number of lineups must match"):
            find_non_duplicate_flex(main_lineups, flex_options)
    
    def test_find_non_duplicate_flex_force_jit(self):
        """Test forcing JIT compilation."""
        main_lineups = np.array([[1, 2, 3], [4, 5, 6]])
        flex_options = np.array([[1, 7, 8], [4, 9, 10]])
        
        # Test with JIT forced on and off
        result_jit = find_non_duplicate_flex(main_lineups, flex_options, use_jit=True)
        result_numpy = find_non_duplicate_flex(main_lineups, flex_options, use_jit=False)
        
        np.testing.assert_array_equal(result_jit, result_numpy)


class TestInternalDuplicateRemoval:
    """Test internal duplicate removal functionality."""
    
    def test_remove_internal_duplicates_basic(self):
        """Test basic internal duplicate removal."""
        population = np.array([
            [1, 2, 3, 4],
            [1, 1, 3, 4],  # Internal duplicate
            [5, 6, 7, 8]
        ])
        
        result = remove_internal_duplicates(population)
        
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [1, 2, 3, 4])
        np.testing.assert_array_equal(result[1], [5, 6, 7, 8])
    
    def test_remove_internal_duplicates_no_duplicates(self):
        """Test when no internal duplicates exist."""
        population = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        
        result = remove_internal_duplicates(population)
        
        assert len(result) == 2
        np.testing.assert_array_equal(result, population)
    
    def test_remove_internal_duplicates_empty(self):
        """Test with empty population."""
        population = np.array([]).reshape(0, 4)
        
        result = remove_internal_duplicates(population)
        
        assert result.shape == (0, 4)
    
    def test_remove_internal_duplicates_all_duplicates(self):
        """Test when all lineups have internal duplicates."""
        population = np.array([
            [1, 1, 3, 4],
            [5, 5, 7, 8]
        ])
        
        result = remove_internal_duplicates(population)
        
        assert len(result) == 0


class TestLineupDuplicateRemoval:
    """Test lineup duplicate removal functionality."""
    
    def test_remove_duplicate_lineups_basic(self):
        """Test basic lineup duplicate removal."""
        population = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 3, 4]  # Duplicate
        ])
        
        result = remove_duplicate_lineups(population)
        
        assert len(result) == 2
        # Should contain both unique lineups
        unique_lineups = [tuple(row) for row in result]
        assert (1, 2, 3, 4) in unique_lineups
        assert (5, 6, 7, 8) in unique_lineups
    
    def test_remove_duplicate_lineups_with_indices(self):
        """Test lineup duplicate removal with index return."""
        population = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 3, 4]  # Duplicate
        ])
        
        result, indices = remove_duplicate_lineups(population, return_indices=True)
        
        assert len(result) == 2
        assert len(indices) == 2
        # Indices should point to original unique lineups
        np.testing.assert_array_equal(result, population[indices])
    
    def test_remove_duplicate_lineups_no_duplicates(self):
        """Test when no lineup duplicates exist."""
        population = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        
        result = remove_duplicate_lineups(population)
        
        assert len(result) == 2
        np.testing.assert_array_equal(result, population)
    
    def test_remove_duplicate_lineups_empty(self):
        """Test with empty population."""
        population = np.array([]).reshape(0, 4)
        
        result = remove_duplicate_lineups(population)
        
        assert result.shape == (0, 4)


class TestDuplicateIndices:
    """Test duplicate index finding functionality."""
    
    def test_find_duplicate_indices_basic(self):
        """Test basic duplicate index finding."""
        population = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 3, 4]  # Duplicate
        ])
        
        duplicate_mask = find_duplicate_indices(population)
        
        assert len(duplicate_mask) == 3
        assert duplicate_mask[0] == False  # First occurrence is not marked as duplicate
        assert duplicate_mask[1] == False  # Unique lineup
        assert duplicate_mask[2] == True   # Duplicate lineup
    
    def test_find_duplicate_indices_no_duplicates(self):
        """Test when no duplicates exist."""
        population = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        
        duplicate_mask = find_duplicate_indices(population)
        
        assert len(duplicate_mask) == 2
        assert not duplicate_mask.any()  # No duplicates


class TestFlexValidation:
    """Test FLEX position validation functionality."""
    
    def test_validate_flex_positions_basic(self):
        """Test basic FLEX position validation."""
        population = np.array([
            [1, 2, 3, 4],  # FLEX (4) is unique
            [5, 6, 7, 5],  # FLEX (5) conflicts with position 0
            [8, 9, 10, 11] # FLEX (11) is unique
        ])
        
        valid_mask = validate_flex_positions(population, flex_column_index=3)
        
        assert len(valid_mask) == 3
        assert valid_mask[0] == True   # Valid
        assert valid_mask[1] == False  # Invalid (FLEX conflicts)
        assert valid_mask[2] == True   # Valid
    
    def test_validate_flex_positions_specific_indices(self):
        """Test FLEX validation against specific position indices."""
        population = np.array([
            [1, 2, 3, 4],  # FLEX (4) doesn't conflict with positions 0,1
            [5, 6, 7, 5],  # FLEX (5) conflicts with position 0
        ])
        
        valid_mask = validate_flex_positions(
            population, 
            flex_column_index=3, 
            main_position_indices=[0, 1]
        )
        
        assert len(valid_mask) == 2
        assert valid_mask[0] == True   # Valid
        assert valid_mask[1] == False  # Invalid


class TestDuplicateRemoverClass:
    """Test DuplicateRemover class functionality."""
    
    def test_duplicate_remover_basic(self):
        """Test basic DuplicateRemover functionality."""
        remover = DuplicateRemover()
        
        population = np.array([
            [1, 2, 3, 4],
            [1, 1, 3, 4],  # Internal duplicate
            [5, 6, 7, 8],
            [1, 2, 3, 4]   # Lineup duplicate
        ])
        
        result = remover.remove_duplicates(population)
        
        assert len(result) == 2  # Should have 2 unique, valid lineups
    
    def test_duplicate_remover_configuration(self):
        """Test DuplicateRemover configuration options."""
        remover = DuplicateRemover(batch_size=2, use_jit=False)
        
        population = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [1, 2, 3, 4]   # Duplicate
        ])
        
        result = remover.remove_duplicates(population)
        
        assert len(result) == 3  # Should have 3 unique lineups
    
    def test_duplicate_remover_validation_stats(self):
        """Test DuplicateRemover validation statistics."""
        remover = DuplicateRemover()
        
        population = np.array([
            [1, 2, 3, 4],
            [1, 1, 3, 4],  # Internal duplicate
            [5, 6, 7, 8],
            [1, 2, 3, 4]   # Lineup duplicate
        ])
        
        stats = remover.validate_population(population)
        
        assert stats['total_lineups'] == 4
        assert stats['unique_lineups'] == 3
        assert stats['duplicate_lineups'] == 1
        assert stats['internal_duplicates'] == 1
        assert stats['duplicate_percentage'] == 25.0
    
    def test_duplicate_remover_selective_removal(self):
        """Test selective duplicate removal options."""
        remover = DuplicateRemover()
        
        population = np.array([
            [1, 2, 3, 4],
            [1, 1, 3, 4],  # Internal duplicate
            [5, 6, 7, 8],
            [1, 2, 3, 4]   # Lineup duplicate
        ])
        
        # Remove only internal duplicates
        result1 = remover.remove_duplicates(
            population, 
            remove_internal=True, 
            remove_lineup_duplicates=False
        )
        assert len(result1) == 3  # Should remove 1 internal duplicate
        
        # Remove only lineup duplicates
        result2 = remover.remove_duplicates(
            population, 
            remove_internal=False, 
            remove_lineup_duplicates=True
        )
        assert len(result2) == 3  # Should remove 1 lineup duplicate


class TestBenchmarking:
    """Test benchmarking functionality."""
    
    def test_benchmark_duplicate_removal(self):
        """Test duplicate removal benchmarking."""
        population_sizes = [10, 50]
        
        results = benchmark_duplicate_removal(
            population_sizes=population_sizes,
            iterations=1,
            duplicate_rate=0.1
        )
        
        assert 'population_sizes' in results
        assert 'numpy_times' in results
        assert 'memory_estimates' in results
        assert len(results['numpy_times']) == len(population_sizes)
        assert len(results['memory_estimates']) == len(population_sizes)
        
        # All times should be positive
        assert all(t >= 0 for t in results['numpy_times'])


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_populations(self):
        """Test handling of empty populations."""
        empty_pop = np.array([]).reshape(0, 4)
        
        # All functions should handle empty populations gracefully
        assert remove_internal_duplicates(empty_pop).shape == (0, 4)
        assert remove_duplicate_lineups(empty_pop).shape == (0, 4)
        assert find_duplicate_indices(empty_pop).shape == (0,)
        assert validate_flex_positions(empty_pop, 0).shape == (0,)
    
    def test_single_lineup(self):
        """Test handling of single lineup populations."""
        single_pop = np.array([[1, 2, 3, 4]])
        
        result1 = remove_internal_duplicates(single_pop)
        result2 = remove_duplicate_lineups(single_pop)
        
        np.testing.assert_array_equal(result1, single_pop)
        np.testing.assert_array_equal(result2, single_pop)
    
    def test_large_population_auto_selection(self):
        """Test automatic algorithm selection for large populations."""
        # Create a larger population to trigger JIT selection
        large_pop = np.random.randint(0, 100, size=(300, 9))
        
        # Should work without errors and produce consistent results
        result1 = remove_duplicate_lineups(large_pop, use_jit=True)
        result2 = remove_duplicate_lineups(large_pop, use_jit=False)
        
        # Results should have same number of unique lineups
        assert len(result1) == len(result2)
