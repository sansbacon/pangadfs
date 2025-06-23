#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Advanced example demonstrating how to use nested plugin configuration via context

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pangadfs.ga import GeneticAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPluginContextExample:
    """Advanced example of using plugin configuration via context with nested dictionaries."""
    
    def __init__(self):
        """Initialize the example."""
        # Setup context dict with nested configuration
        self.ctx = {
            # General GA settings
            'ga_settings': {
                'csvpth': Path(__file__).parent / '../tests/test_pool.csv',
                'population_size': 1000,
                'n_generations': 15,
                'stop_criteria': 5,  # Stop if no improvement after 5 generations
                'verbose': True
            },
            
            # Plugin-specific settings in their own namespaces
            'plugin_settings': {
                'mutate': {
                    'mutation_rate': 0.15,  # Higher mutation rate than default (0.05)
                    'mutation_strategy': 'random'  # Example of additional parameter
                },
                'fitness': {
                    'fitness_scale': 1.2,   # Scale fitness values by 1.2
                    'use_weighted_scoring': True  # Example of additional parameter
                },
                'select': {
                    'select_method': 'tournament',
                    'tournament_size': 3
                },
                'crossover': {
                    'crossover_method': 'uniform',
                    'uniform_rate': 0.5
                }
            },
            
            # Site-specific settings
            'site_settings': {
                'lineup_size': 9,
                'salary_cap': 50000,
                'posfilter': {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0, 'FLEX': 0},
                'posmap': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1},
                'flex_positions': ['RB', 'WR', 'TE']
            }
        }
        
        # Create a custom plugin loader function that extracts plugin-specific settings
        self.plugin_loader = lambda plugin_name, ctx: {
            **ctx.get('ga_settings', {}),
            **ctx.get('site_settings', {}),
            **(ctx.get('plugin_settings', {}).get(plugin_name, {}))
        }
        
    def run(self):
        """Run the example."""
        logger.info("Creating GeneticAlgorithm instance with nested context")
        
        # Process the context to extract plugin-specific settings
        processed_ctx = {}
        for namespace in GeneticAlgorithm.PLUGIN_NAMESPACES:
            # Extract settings for this plugin namespace
            plugin_ctx = self.plugin_loader(namespace, self.ctx)
            # Add to processed context
            processed_ctx.update(plugin_ctx)
        
        # Create GA instance with processed context
        ga = GeneticAlgorithm(ctx=processed_ctx, use_defaults=True)
        
        # Load player pool
        logger.info("Loading player pool")
        pool = ga.pool(csvpth=self.ctx['ga_settings']['csvpth'])
        
        # Create position pools
        logger.info("Creating position pools")
        posfilter = self.ctx['site_settings']['posfilter']
        column_mapping = {'player': 'player', 'position': 'pos', 'salary': 'salary', 'projection': 'proj'}
        flex_positions = self.ctx['site_settings']['flex_positions']
        pospool = ga.pospool(pool=pool, posfilter=posfilter, column_mapping=column_mapping, flex_positions=flex_positions)
        
        # Create initial population
        logger.info("Creating initial population")
        posmap = self.ctx['site_settings']['posmap']
        population_size = self.ctx['ga_settings']['population_size']
        population = ga.populate(pospool=pospool, posmap=posmap, population_size=population_size)
        
        # Run optimization for multiple generations
        logger.info(f"Running optimization for {self.ctx['ga_settings']['n_generations']} generations")
        
        points = np.array(pool['proj'].values)
        best_fitness = 0
        stagnant_generations = 0
        
        for gen in range(self.ctx['ga_settings']['n_generations']):
            # Calculate fitness
            fitness = ga.fitness(population=population, points=points)
            
            # Track best fitness
            current_best = fitness.max()
            if current_best > best_fitness:
                best_fitness = current_best
                stagnant_generations = 0
                logger.info(f"Generation {gen+1}: New best fitness: {best_fitness:.2f}")
            else:
                stagnant_generations += 1
                logger.info(f"Generation {gen+1}: No improvement. Best fitness: {best_fitness:.2f}")
            
            # Check stop criteria
            if stagnant_generations >= self.ctx['ga_settings']['stop_criteria']:
                logger.info(f"Stopping early: No improvement for {stagnant_generations} generations")
                break
            
            # Select parents
            selected = ga.select(population=population, population_fitness=fitness, 
                                n=population_size // 2)
            
            # Crossover
            offspring = ga.crossover(population=selected)
            
            # Mutate
            population = ga.mutate(population=offspring)
        
        # Calculate final fitness
        fitness = ga.fitness(population=population, points=points)
        
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
        
        logger.info("Advanced example completed successfully")

if __name__ == "__main__":
    example = AdvancedPluginContextExample()
    example.run()
