#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example script demonstrating how to use plugin configuration via context

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pangadfs.ga import GeneticAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Example of using plugin configuration via context."""
    
    # Setup context dict for configuration
    # This demonstrates how to configure plugins via context
    ctx = {
        # General GA settings
        'csvpth': Path(__file__).parent / '../tests/test_pool.csv',
        'population_size': 1000,
        'n_generations': 10,
        
        # Plugin-specific settings
        'mutation_rate': 0.15,  # Higher mutation rate than default (0.05)
        'fitness_scale': 1.2,   # Scale fitness values by 1.2
        
        # Site-specific settings
        'lineup_size': 9,
        'salary_cap': 50000
    }
    
    logger.info("Creating GeneticAlgorithm instance with context")
    ga = GeneticAlgorithm(ctx=ctx, use_defaults=True)
    
    # Load player pool
    logger.info("Loading player pool")
    pool = ga.pool(csvpth=ctx['csvpth'])
    
    # Create position pools
    logger.info("Creating position pools")
    posfilter = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0, 'FLEX': 0}
    column_mapping = {'player': 'player', 'position': 'pos', 'salary': 'salary', 'projection': 'proj'}
    flex_positions = ['RB', 'WR', 'TE']
    pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping=column_mapping, flex_positions=flex_positions)
    
    # Create position map
    posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1}
    
    # Create initial population
    logger.info("Creating initial population")
    population = ga.populate(pospool=pospool, posmap=posmap, population_size=ctx['population_size'])
    
    # Calculate fitness
    logger.info("Calculating fitness")
    points = np.array(pool['proj'].values)
    fitness = ga.fitness(population=population, points=points)
    
    # Log average fitness
    logger.info(f"Average fitness: {fitness.mean():.2f}")
    
    # Demonstrate that the mutation rate from context is being used
    logger.info("Mutating population")
    mutated = ga.mutate(population=population)
    
    # Calculate how many individuals were mutated
    diff = (population != mutated).any(axis=1).sum()
    logger.info(f"Number of individuals mutated: {diff} out of {len(population)}")
    
    # Expected number with default rate (0.05): ~50 out of 1000
    # Expected number with context rate (0.15): ~150 out of 1000
    expected = ctx['population_size'] * ctx['mutation_rate']
    logger.info(f"Expected number with mutation_rate={ctx['mutation_rate']}: ~{expected:.0f}")
    
    # Show top 5 lineups
    top_indices = np.argsort(fitness)[-5:][::-1]
    logger.info("Top 5 lineups:")
    
    for i, idx in enumerate(top_indices):
        lineup = population[idx]
        players = [pool.iloc[p]['player'] for p in lineup]
        total_salary = sum(pool.iloc[p]['salary'] for p in lineup)
        total_points = fitness[idx]
        
        logger.info(f"Lineup {i+1}: Points={total_points:.2f}, Salary=${total_salary}")
        logger.info(f"  Players: {', '.join(players)}")
    
    logger.info("Example completed successfully")

if __name__ == "__main__":
    main()
