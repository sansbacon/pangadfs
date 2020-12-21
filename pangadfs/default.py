# pangadfs/pangadfs/default.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .base import *


class DefaultCrossover(CrossoverBase):
    """Default crossover technique
    
    The default approach is to take the top {pctl} of the population and divide in half.
    Then generate boolean array of the same size as each individual (e.g. 9 for DK)
    Use the boolean to generate two children
    Child 1 takes from father on True and mother on False
    Child 2 takes from mother on True and father on False

    Example:
        c = DefaultCrossover()
        newpop = c.crossover(population=oldpop, population_fitness=oldpopfit, pctl=50)

    """

    def crossover(self, 
                  *, 
                  population: np.ndarray, 
                  population_fitness: np.ndarray=None,
                  pctl: int=50):
        """Crosses over individuals in population
        
        Args:
            population (np.ndarray): the population to crossover. Shape is n_individuals x n_chromosomes.
            population_fitness (np.ndarray): the population fitness. Is a 1D array same length as population.
            pctl (int): percentile fitness to sample from, integer value from 1 - 99

        Returns:
            np.ndarray
            concatenation of fathers, mothers, offspring1, offspring2
            does not recalculate fitness / sort or trim / validate population

        """
        logging.info('using default crossover')      

        # split top pctl of population into parents
        if population_fitness is not None:
            fittest = np.argwhere(population_fitness > np.percentile(population_fitness, pctl)).ravel()
            fathers, mothers = np.array_split(population[fittest], 2)
        else:
            fathers, mothers = np.array_split(population, 2)
        if len(fathers) != len(mothers):
            mothers = mothers[:-1]
      
        # choice is a 9-element array of True and False
        choice = np.random.randint(2, size=fathers.size).reshape(fathers.shape).astype(bool)   
        return np.vstack((fathers, 
                          mothers, 
                          np.where(choice, fathers, mothers), 
                          np.where(choice, mothers, fathers)))
       
class DefaultMutate(MutateBase):
    """Default mutate technique"""

    def mutate(self, *, population: np.ndarray, mutation_rate: float=.05):
        """Mutates individuals in population
        
        Args:

        Returns:
            np.ndarray of same shape as population

        """
        logging.info('using default mutate')
        mutate = (
            np.random.binomial(n=1, p=mutation_rate, size=population.size)
            .reshape(population.shape)
            .astype(bool)
        )
        return np.where(mutate, np.random.permutation(population), population)      


class DefaultFitness(FitnessBase):
    """Default fitness technique."""

    def fitness(self, 
                *, 
                population: np.ndarray, 
                points_mapping: Dict[int, float]):
        """Assesses population fitness"""
        logging.info('using default fitness')
        return np.apply_along_axis(lambda x: sum([points_mapping[i] for i in x]), axis=1, arr=population)


class DefaultPool(PoolBase):
    """Default pool technique
    
        Reads in a csv file with the following columns:
        player, team, pos, salary, proj

        Sorts players by position so indexes are sequential by position

    """

    @staticmethod
    def pool(csvpth):
        """Creates initial pool"""
        logging.info('using default pool')
        pool = pd.read_csv(csvpth)
        assert set(pool.columns) == set(['player', 'team', 'pos', 'salary', 'proj'])
        return pool.sort_values(['pos']).reset_index(drop=True)


class DefaultPospool(PospoolBase):
    """Default pospool technique
    
       Creates prob (probabilities) column for weighted random sampling.
       Uses points per dollar to nudge initial selection to best options.
       While there are good reasons not to optimize solely on ppd, it is 
       effective at creating a good initial population.    

    """

    @staticmethod
    def pospool(*,
                pool: pd.DataFrame,
                posfilter: Dict[str, int], 
                column_mapping: Dict[str, str],
                flex_positions: Iterable[str] = ('RB', 'WR', 'TE'),
                ):
        """Creates initial position pool
        
        Args:
            pool (pd.DataFrame):
            posfilter (Dict[str, int]): position name and points threshold
            column_mapping (Dict[str, str]): column names for player, position, salary, projection
            flex_positions (Iterable[str]): e.g. (WR, RB, TE)
 
        Returns:
            Dict[str, pd.DataFrame] where keys == posfilter.keys

        """
        d = {}
        poscol = column_mapping.get('position', 'pos')
        pointscol = column_mapping.get('points', 'proj')
        salcol = column_mapping.get('salary', 'salary')
        wanted = column_mapping.values()
        for position, thresh in posfilter.items():
            if position == 'FLEX':
                tmp = pool.loc[(pool[poscol].isin(flex_positions)) & (pool[pointscol] >= thresh), wanted]      
            else:           
                tmp = pool.loc[(pool[poscol] == position) & (pool[pointscol] >= thresh), wanted]
            prob_ = (pool[pointscol] / pool[salcol]) * 1000
            prob_ = prob_ / prob_.sum()
            d[position] = tmp.assign(prob=prob_)
        return d


class DefaultPopulate(PopulateBase):
    """Default populate technique"""

    @staticmethod
    def multidimensional_shifting(elements, num_samples, sample_size, probs):
        """Based on https://medium.com/ibm-watson/incredibly-fast-random-sampling-in-python-baf154bd836a
        
        Args:
            elements (iterable): iterable to sample from, typically a dataframe index
            num_samples (int): the number of rows (e.g. initial population size)
            sample_size (int): the number of columns (e.g. team size)
            probs (iterable): is same size as elements

        Returns:
            ndarray with shape (num_samples, sample_size)
            
        """
        replicated_probabilities = np.tile(probs, (num_samples, 1))
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        shifted_probabilities = random_shifts - replicated_probabilities
        return elements[np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]]

    def populate(self, 
                 *, 
                 pospool, 
                 posmap: Dict[str, int], 
                 population_size: int, 
                 probcol: str='prob'):
        """Creates individuals in population
        
        Args:
            pospool (Dict[str, DataFrame]): pool split into positions
            posmap (Dict[str, int]): positions & accompanying roster slots
            population_size (int): number of individuals to create
            probcol (str): the dataframe column with probabilities

        Returns:
            ndarray of size (population_size, sum(posmap.values()))

        """
        logging.info('using default populate')

        pos_samples = {
            pos: self.multidimensional_shifting(pospool[pos].index, population_size, n, pospool[pos][probcol])
            for pos, n in posmap.items()
        }

        # concatenate positions into single row
        pop = np.concatenate([pos_samples[pos] for pos in posmap if pos != 'FLEX'], axis=1)

        # add flex and filter out duplicates
        flex = np.array([np.random.choice(np.setdiff1d(pos_samples['FLEX'][i], pop[i])) 
                         for i in range(population_size)])
        
        # return aggregated array
        return np.column_stack((pop, flex))


class DefaultValidate(ValidateBase):
    """Default validate technique"""

    @staticmethod
    def bincount(population: np.ndarray, valid_size: int= 9):
        n = population.max()+1
        population_off = population + (np.arange(population.shape[0])[:,None]) * n
        M = population.shape[0]*n
        return np.where((np.bincount(population_off.ravel(), minlength=M).reshape(-1,n)!=0).sum(1) == valid_size, True, False)

    def validate(self, 
                 *, 
                 population: np.ndarray, 
                 salary_mapping: Dict[int, int],
                 salary_cap: int):
        """Ensures valid individuals in population"""
        logging.info('using default validate')
        popsal = np.apply_along_axis(lambda x: sum([salary_mapping[i] for i in x]), axis=1, arr=population)

        # duplicates validation
        population = population[popsal <= salary_cap]
        return population[self.bincount(population)]


if __name__ == '__main__':
    pass

