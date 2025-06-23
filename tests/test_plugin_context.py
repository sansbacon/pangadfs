# tests/test_plugin_context.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
import pytest

from pangadfs.ga import GeneticAlgorithm
from pangadfs.mutate import MutateDefault
from pangadfs.fitness import FitnessDefault


def test_mutate_plugin_with_context():
    """Test that the MutateDefault plugin uses the mutation rate from context."""
    # Create a context with a custom mutation rate
    ctx = {'mutation_rate': 0.2}
    
    # Create the plugin with the context
    plugin = MutateDefault(ctx)
    
    # Check that the plugin has the correct mutation rate
    assert plugin.default_mutation_rate == 0.2
    
    # Create a population to mutate
    population = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Mutate the population without specifying a mutation rate
    # This should use the rate from context
    mutated = plugin.mutate(population=population)
    
    # The mutated population should be different from the original
    # but have the same shape
    assert mutated.shape == population.shape
    
    # We can't directly test the mutation rate since it's random,
    # but we can check that the plugin used the correct default rate
    assert plugin.default_mutation_rate == 0.2


def test_fitness_plugin_with_context():
    """Test that the FitnessDefault plugin uses the fitness scale from context."""
    # Create a context with a custom fitness scale
    ctx = {'fitness_scale': 1.5}
    
    # Create the plugin with the context
    plugin = FitnessDefault(ctx)
    
    # Check that the plugin has the correct fitness scale
    assert plugin.fitness_scale == 1.5
    
    # Create a population and points array
    population = np.array([[0, 1], [2, 3]])
    points = np.array([10, 20, 30, 40])
    
    # Calculate fitness without specifying a scale
    # This should use the scale from context
    fitness = plugin.fitness(population=population, points=points)
    
    # The fitness should be calculated correctly with the scale applied
    expected = np.array([(10 + 20) * 1.5, (30 + 40) * 1.5])
    np.testing.assert_array_equal(fitness, expected)
    
    # Calculate fitness with a specific scale
    # This should override the scale from context
    fitness = plugin.fitness(population=population, points=points, scale=2.0)
    
    # The fitness should be calculated with the specified scale
    expected = np.array([(10 + 20) * 2.0, (30 + 40) * 2.0])
    np.testing.assert_array_equal(fitness, expected)


def test_ga_with_plugin_context():
    """Test that the GeneticAlgorithm passes context to plugins."""
    # Create a context with plugin configuration
    ctx = {
        'mutation_rate': 0.25,
        'fitness_scale': 1.75
    }
    
    # Create a GA instance with the context
    ga = GeneticAlgorithm(ctx=ctx, use_defaults=True)
    
    # Get the mutate plugin and check its configuration
    mutate_plugin = ga.get_plugin('mutate')
    assert mutate_plugin.default_mutation_rate == 0.25
    
    # Get the fitness plugin and check its configuration
    fitness_plugin = ga.get_plugin('fitness')
    assert fitness_plugin.fitness_scale == 1.75
    
    # Test that the GA methods use the configured plugins
    population = np.array([[0, 1], [2, 3]])
    points = np.array([10, 20, 30, 40])
    
    # Calculate fitness
    fitness = ga.fitness(population=population, points=points)
    expected = np.array([(10 + 20) * 1.75, (30 + 40) * 1.75])
    np.testing.assert_array_equal(fitness, expected)
