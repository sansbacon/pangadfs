# pangadfs/pangadfs/default.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from pangadfs.base import *


class CrossoverDefault(CrossoverBase):

    def crossover(self,
                  *, 
                  population: np.ndarray, 
                  population_fitness: np.ndarray = None,
                  pctl: int = 50):
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
        # split top pctl of population into parents
        if population_fitness is not None:
            fittest = np.argwhere(population_fitness > np.percentile(population_fitness, pctl)).ravel()
            fathers, mothers = np.array_split(population[fittest], 2)
        else:
            fathers, mothers = np.array_split(population, 2)
        if len(fathers) > len(mothers):
            fathers = fathers[:-1]
      
        # choice is a 9-element array of True and False
        choice = np.random.randint(2, size=fathers.size).reshape(fathers.shape).astype(bool)   
        return np.vstack((population, 
                          np.where(choice, fathers, mothers), 
                          np.where(choice, mothers, fathers)))

       
class MutateDefault(MutateBase):

    def mutate(self, *, population: np.ndarray, mutation_rate: float = .05):
        """Mutates individuals in population
        
        Args:
            population (np.ndarray): the population to mutate. Shape is n_individuals x n_chromosomes.
            mutation_rate (float): decimal value from 0 to 1, default .05

        Returns:
            np.ndarray of same shape as population

        """
        mutate = (
            np.random.binomial(n=1, p=mutation_rate, size=population.size)
            .reshape(population.shape)
            .astype(bool)
        )
        return np.where(mutate, np.random.permutation(population), population)      


class FitnessDefault(FitnessBase):

    def fitness(self,
                *, 
                population: np.ndarray, 
                points_mapping: Dict[int, float]):
        """Assesses population fitness using supplied mapping
        
        Args:
            population (np.ndarray): the population to assess fitness
            points_mapping (Dict[int, float]): the array index: projected points

        Returns:
            np.ndarray: 1D array of float

        """
        return np.apply_along_axis(lambda x: sum([points_mapping[i] for i in x]), axis=1, arr=population)


class PoolDefault(PoolBase):

    def pool(self, csvpth):
        """Creates initial pool"""
        pool = pd.read_csv(csvpth)
        assert set(pool.columns) == set(['player', 'team', 'pos', 'salary', 'proj'])
        return pool.sort_values(['pos']).reset_index(drop=True)


class PospoolDefault(PospoolBase):

    def pospool(self,
                *,
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
        for position, thresh in posfilter.items():
            if position == 'FLEX':
                tmp = pool.loc[(pool[poscol].isin(flex_positions)) & (pool[pointscol] >= thresh), [pointscol, salcol]]      
            else:           
                tmp = pool.loc[(pool[poscol] == position) & (pool[pointscol] >= thresh), [pointscol, salcol]]
            prob_ = (tmp[pointscol] / tmp[salcol]) * 1000
            prob_ = prob_ / prob_.sum()
            d[position] = tmp.assign(prob=prob_)
        return d


class PopulateDefault(PopulateBase):

    def multidimensional_shifting(self, elements, num_samples, sample_size, probs):
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
        samples = np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]
        return elements.to_numpy()[samples]

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
        pos_samples = {
            pos: self.multidimensional_shifting(pospool[pos].index, population_size, n, pospool[pos][probcol])
            for pos, n in posmap.items()
        }

        # concatenate positions into single row
        pop = np.concatenate([pos_samples[pos] for pos in posmap if pos != 'FLEX'], axis=1)

        # find non-duplicate FLEX and aggregate with other positions
        # https://stackoverflow.com/questions/65473095
        # https://stackoverflow.com/questions/54155844/
        dups = (pos_samples['FLEX'][..., None] == pop[:, None, :]).any(-1)
        return np.column_stack((pop, pos_samples['FLEX'][np.invert(dups).cumsum(axis=1).cumsum(axis=1) == 1]))


class SalaryValidate(ValidateBase):

    def validate(self,
                 *, 
                 population: np.ndarray, 
                 salary_mapping: Dict[int, int],
                 salary_cap: int,
                 **kwargs):
        """Ensures valid individuals in population"""
        popsal = np.apply_along_axis(lambda x: sum([salary_mapping[i] for i in x]), axis=1, arr=population)
        return population[popsal <= salary_cap]


class DuplicatesValidate(ValidateBase):

    def validate(self,
                 *, 
                 population: np.ndarray, 
                 valid_size: int = 9,
                 **kwargs):
        """Ensures valid individuals in population"""
        n = population.max() + 1
        population_off = population + (np.arange(population.shape[0])[:,None]) * n
        M = population.shape[0]*n
        idx = np.where((np.bincount(population_off.ravel(), minlength=M).reshape(-1,n)!=0).sum(1) == valid_size, True, False)
        return population[idx]


if __name__ == '__main__':
    pass
