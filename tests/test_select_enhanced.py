# pangadfs/tests/test_select_enhanced.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
import pytest

from pangadfs.select import SelectDefault
from pangadfs.fitness import FitnessDefault


class TestSelectEnhanced:
    """Test enhanced selection functionality."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data for selection tests."""
        # Create a diverse population for testing
        np.random.seed(42)
        population = np.random.randint(1, 100, size=(50, 9))
        
        # Create fitness values with some variation
        fitness = np.random.exponential(scale=10, size=50)
        fitness = fitness + np.random.normal(0, 2, size=50)  # Add some noise
        
        selector = SelectDefault()
        
        return {
            'population': population,
            'fitness': fitness,
            'selector': selector,
            'n_select': 20
        }
    
    def test_enhanced_fittest_selection(self, setup_data):
        """Test enhanced fittest selection with performance optimizations."""
        data = setup_data
        
        # Test small selection (< half population)
        selected = data['selector']._fittest(
            population=data['population'],
            population_fitness=data['fitness'],
            n=10
        )
        
        assert len(selected) == 10
        assert isinstance(selected, np.ndarray)
        
        # Verify selected individuals have highest fitness
        selected_indices = []
        for sel in selected:
            matches = np.where((data['population'] == sel).all(axis=1))[0]
            if len(matches) > 0:
                selected_indices.append(matches[0])
        
        selected_fitness = data['fitness'][selected_indices]
        top_10_fitness = np.sort(data['fitness'])[-10:]
        
        # Selected fitness should match top fitness values
        np.testing.assert_array_almost_equal(
            np.sort(selected_fitness), 
            np.sort(top_10_fitness)
        )
    
    def test_enhanced_rank_selection(self, setup_data):
        """Test enhanced rank selection with selection pressure."""
        data = setup_data
        
        # Test with default pressure
        selected1 = data['selector']._rank(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select']
        )
        
        # Test with high pressure
        selected2 = data['selector']._rank(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select'],
            selection_pressure=2.0
        )
        
        assert len(selected1) == data['n_select']
        assert len(selected2) == data['n_select']
        
        # High pressure should select fitter individuals on average
        # (This is probabilistic, so we test multiple times)
        fitness_improvements = []
        for _ in range(10):
            sel1 = data['selector']._rank(
                population=data['population'],
                population_fitness=data['fitness'],
                n=data['n_select'],
                selection_pressure=1.0
            )
            sel2 = data['selector']._rank(
                population=data['population'],
                population_fitness=data['fitness'],
                n=data['n_select'],
                selection_pressure=3.0
            )
            
            # Calculate average fitness of selections
            indices1 = [np.where((data['population'] == s).all(axis=1))[0][0] for s in sel1]
            indices2 = [np.where((data['population'] == s).all(axis=1))[0][0] for s in sel2]
            
            avg_fitness1 = data['fitness'][indices1].mean()
            avg_fitness2 = data['fitness'][indices2].mean()
            
            fitness_improvements.append(avg_fitness2 >= avg_fitness1)
        
        # High pressure should generally produce better fitness
        assert sum(fitness_improvements) >= 6  # At least 60% of the time
    
    def test_enhanced_roulette_wheel(self, setup_data):
        """Test enhanced roulette wheel with numerical stability."""
        data = setup_data
        
        # Test with positive fitness
        selected = data['selector']._roulette_wheel(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select']
        )
        
        assert len(selected) == data['n_select']
        
        # Test with negative fitness values
        negative_fitness = data['fitness'] - data['fitness'].max() - 10
        selected_neg = data['selector']._roulette_wheel(
            population=data['population'],
            population_fitness=negative_fitness,
            n=data['n_select']
        )
        
        assert len(selected_neg) == data['n_select']
    
    def test_enhanced_scaled_selection(self, setup_data):
        """Test enhanced scaled selection with scaling factor."""
        data = setup_data
        
        # Test with different scaling factors
        selected1 = data['selector']._scaled(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select'],
            scaling_factor=1.0
        )
        
        selected2 = data['selector']._scaled(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select'],
            scaling_factor=2.0
        )
        
        assert len(selected1) == data['n_select']
        assert len(selected2) == data['n_select']
        
        # Test with uniform fitness (edge case)
        uniform_fitness = np.ones(len(data['population']))
        selected_uniform = data['selector']._scaled(
            population=data['population'],
            population_fitness=uniform_fitness,
            n=data['n_select']
        )
        
        assert len(selected_uniform) == data['n_select']
    
    def test_enhanced_sus_selection(self, setup_data):
        """Test enhanced SUS with JIT optimization."""
        data = setup_data
        
        # Test normal case
        selected = data['selector']._sus(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select']
        )
        
        assert len(selected) == data['n_select']
        
        # Test with large population to trigger JIT
        large_pop = np.random.randint(1, 100, size=(200, 9))
        large_fitness = np.random.exponential(scale=10, size=200)
        
        selected_large = data['selector']._sus(
            population=large_pop,
            population_fitness=large_fitness,
            n=50
        )
        
        assert len(selected_large) == 50
        
        # Test with negative fitness
        negative_fitness = data['fitness'] - data['fitness'].max() - 5
        selected_neg = data['selector']._sus(
            population=data['population'],
            population_fitness=negative_fitness,
            n=data['n_select']
        )
        
        assert len(selected_neg) == data['n_select']
    
    def test_enhanced_tournament_selection(self, setup_data):
        """Test enhanced tournament selection with JIT optimization."""
        data = setup_data
        
        # Test with default parameters
        selected = data['selector']._tournament(
            population=data['population'],
            population_fitness=data['fitness'],
            tournament_size=3,
            n=data['n_select']
        )
        
        assert len(selected) == data['n_select']
        
        # Test with large population to trigger JIT
        large_pop = np.random.randint(1, 100, size=(200, 9))
        large_fitness = np.random.exponential(scale=10, size=200)
        
        selected_large = data['selector']._tournament(
            population=large_pop,
            population_fitness=large_fitness,
            tournament_size=5,
            n=60
        )
        
        assert len(selected_large) == 60
        
        # Test with different tournament sizes
        for tournament_size in [2, 4, 8]:
            selected_ts = data['selector']._tournament(
                population=data['population'],
                population_fitness=data['fitness'],
                tournament_size=tournament_size,
                n=10
            )
            assert len(selected_ts) == 10
    
    def test_diversity_tournament_selection(self, setup_data):
        """Test diversity-aware tournament selection."""
        data = setup_data
        
        # Test with different diversity weights
        selected_fitness = data['selector']._diversity_tournament(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select'],
            diversity_weight=0.0  # Pure fitness
        )
        
        selected_diversity = data['selector']._diversity_tournament(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select'],
            diversity_weight=1.0  # Pure diversity
        )
        
        assert len(selected_fitness) == data['n_select']
        assert len(selected_diversity) == data['n_select']
        
        # Test balanced selection
        selected_balanced = data['selector']._diversity_tournament(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select'],
            diversity_weight=0.5,
            tournament_size=4
        )
        
        assert len(selected_balanced) == data['n_select']
    
    def test_adaptive_pressure_selection(self, setup_data):
        """Test adaptive selection pressure."""
        data = setup_data
        
        # Test with different base methods
        for base_method in ['tournament', 'rank', 'roulette']:
            selected = data['selector']._adaptive_pressure(
                population=data['population'],
                population_fitness=data['fitness'],
                n=data['n_select'],
                base_method=base_method
            )
            
            assert len(selected) == data['n_select']
        
        # Test with low diversity population (should reduce pressure)
        low_diversity_fitness = np.ones(len(data['population'])) + np.random.normal(0, 0.01, len(data['population']))
        selected_low = data['selector']._adaptive_pressure(
            population=data['population'],
            population_fitness=low_diversity_fitness,
            n=data['n_select'],
            base_method='tournament'
        )
        
        assert len(selected_low) == data['n_select']
        
        # Test with high diversity population (should increase pressure)
        high_diversity_fitness = np.random.exponential(scale=50, size=len(data['population']))
        selected_high = data['selector']._adaptive_pressure(
            population=data['population'],
            population_fitness=high_diversity_fitness,
            n=data['n_select'],
            base_method='tournament'
        )
        
        assert len(selected_high) == data['n_select']
    
    def test_elite_preserving_selection(self, setup_data):
        """Test elite-preserving selection."""
        data = setup_data
        
        # Test with different elite ratios
        for elite_ratio in [0.1, 0.2, 0.5]:
            selected = data['selector']._elite_preserving(
                population=data['population'],
                population_fitness=data['fitness'],
                n=data['n_select'],
                elite_ratio=elite_ratio,
                base_method='tournament'
            )
            
            assert len(selected) == data['n_select']
            
            # Verify elite individuals are included
            n_elite = max(1, int(data['n_select'] * elite_ratio))
            top_fitness_indices = np.argpartition(data['fitness'], -n_elite)[-n_elite:]
            top_fitness_values = data['fitness'][top_fitness_indices]
            
            # Find fitness values of selected individuals
            selected_indices = []
            for sel in selected:
                matches = np.where((data['population'] == sel).all(axis=1))[0]
                if len(matches) > 0:
                    selected_indices.append(matches[0])
            
            selected_fitness_values = data['fitness'][selected_indices]
            
            # Check that top fitness values are represented in selection
            for top_val in top_fitness_values:
                assert np.any(np.abs(selected_fitness_values - top_val) < 1e-10)
    
    def test_input_validation(self, setup_data):
        """Test input validation for all selection methods."""
        data = setup_data
        
        # Test empty population
        with pytest.raises(ValueError, match="Population cannot be empty"):
            data['selector']._fittest(
                population=np.array([]).reshape(0, 9),
                population_fitness=np.array([]),
                n=1
            )
        
        # Test mismatched lengths
        with pytest.raises(ValueError, match="Population and fitness arrays must have same length"):
            data['selector']._fittest(
                population=data['population'],
                population_fitness=data['fitness'][:-1],
                n=10
            )
        
        # Test invalid n
        with pytest.raises(ValueError, match="Number of individuals to select must be positive"):
            data['selector']._fittest(
                population=data['population'],
                population_fitness=data['fitness'],
                n=0
            )
        
        # Test n > population size
        with pytest.raises(ValueError, match="Cannot select more individuals than population size"):
            data['selector']._fittest(
                population=data['population'],
                population_fitness=data['fitness'],
                n=len(data['population']) + 1
            )
        
        # Test non-finite fitness
        bad_fitness = data['fitness'].copy()
        bad_fitness[0] = np.inf
        with pytest.raises(ValueError, match="Fitness values must be finite"):
            data['selector']._fittest(
                population=data['population'],
                population_fitness=bad_fitness,
                n=10
            )
    
    def test_tournament_validation(self, setup_data):
        """Test tournament-specific validation."""
        data = setup_data
        
        # Test invalid tournament size
        with pytest.raises(ValueError, match="Tournament size must be positive"):
            data['selector']._tournament(
                population=data['population'],
                population_fitness=data['fitness'],
                tournament_size=0,
                n=10
            )
        
        # Test tournament size > population
        with pytest.raises(ValueError, match="Tournament size cannot exceed population size"):
            data['selector']._tournament(
                population=data['population'],
                population_fitness=data['fitness'],
                tournament_size=len(data['population']) + 1,
                n=10
            )
    
    def test_select_method_dispatch(self, setup_data):
        """Test the main select method with all available methods."""
        data = setup_data
        
        methods = [
            'roulette', 'sus', 'rank', 'tournament', 'scaled', 'fittest',
            'diversity_tournament', 'adaptive', 'elite'
        ]
        
        for method in methods:
            selected = data['selector'].select(
                population=data['population'],
                population_fitness=data['fitness'],
                n=data['n_select'],
                method=method
            )
            
            assert len(selected) == data['n_select']
            assert isinstance(selected, np.ndarray)
        
        # Test unknown method (should fallback to roulette)
        selected_unknown = data['selector'].select(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select'],
            method='unknown_method'
        )
        
        assert len(selected_unknown) == data['n_select']
    
    def test_utility_methods(self, setup_data):
        """Test utility methods like cache management and statistics."""
        data = setup_data
        
        # Test cache clearing
        data['selector'].clear_cache()
        assert len(data['selector']._diversity_cache) == 0
        
        # Test JIT threshold setting
        data['selector'].set_jit_threshold(200)
        assert data['selector']._use_jit_threshold == 200
        
        # Test selection statistics
        selected = data['selector'].select(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select'],
            method='fittest'
        )
        
        stats = data['selector'].get_selection_stats(
            population=data['population'],
            population_fitness=data['fitness'],
            selected=selected
        )
        
        assert 'selection_ratio' in stats
        assert 'fitness_improvement' in stats
        assert 'diversity_metrics' in stats
        assert 'elite_capture' in stats
        
        # Verify statistics make sense
        assert 0 < stats['selection_ratio'] <= 1
        assert stats['fitness_improvement']['improvement_ratio'] >= 1  # Fittest should improve fitness
    
    def test_error_handling(self, setup_data):
        """Test error handling and fallback mechanisms."""
        data = setup_data
        
        # Create a selector that will fail (by mocking a method to raise an exception)
        selector = SelectDefault()
        
        # Mock a method to raise an exception
        original_method = selector._roulette_wheel
        def failing_method(*args, **kwargs):
            raise RuntimeError("Simulated failure")
        
        selector._roulette_wheel = failing_method
        
        # Should fallback to fittest selection
        selected = selector.select(
            population=data['population'],
            population_fitness=data['fitness'],
            n=data['n_select'],
            method='roulette'
        )
        
        assert len(selected) == data['n_select']
        
        # Restore original method
        selector._roulette_wheel = original_method
    
    def test_performance_with_large_populations(self):
        """Test performance optimizations with large populations."""
        # Create large population to test JIT compilation paths
        np.random.seed(42)
        large_pop = np.random.randint(1, 1000, size=(500, 9))
        large_fitness = np.random.exponential(scale=20, size=500)
        
        selector = SelectDefault()
        selector.set_jit_threshold(100)  # Ensure JIT is used
        
        # Test all methods with large population
        methods = ['tournament', 'sus', 'rank']
        
        for method in methods:
            selected = selector.select(
                population=large_pop,
                population_fitness=large_fitness,
                n=100,
                method=method
            )
            
            assert len(selected) == 100
            assert isinstance(selected, np.ndarray)
    
    def test_backward_compatibility(self, p, pop):
        """Test backward compatibility with original test."""
        # This should work exactly like the original test
        points = p['proj'].values
        fitness = FitnessDefault().fitness(
            population=pop, points=points
        )

        params = {
            'population': pop,
            'population_fitness': fitness,
            'n': len(pop) // 2
        }

        # Test original methods
        for method in ('roulette', 'sus', 'scaled', 'rank', 'tournament'):
            newparams = {**params, **{'method': method}}
            newpop = SelectDefault().select(**newparams)
            assert isinstance(newpop, np.ndarray)
            assert newpop.dtype == 'int64'
            assert len(newpop) == len(pop) // 2
        
        # Test fittest method (was missing from original)
        fittest_params = {**params, **{'method': 'fittest'}}
        fittest_pop = SelectDefault().select(**fittest_params)
        assert isinstance(fittest_pop, np.ndarray)
        assert fittest_pop.dtype == 'int64'
        assert len(fittest_pop) == len(pop) // 2
