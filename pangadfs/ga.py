# gadfs/gadfs/ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License


class GeneticAlgorithm:

    def __init__(self, pool, driver_managers, initial_size=10, team_size=5):
        self.pool = pool
        self.initial_size = initial_size
        self.team_size = team_size
        self.driver_managers = driver_managers

    def crossover(self, population):
        mgr = self.driver_managers.get('crossover')
        return mgr.driver.crossover(population=population, pool=self.pool)

    def fitness(self, population):
        mgr = self.driver_managers.get('fitness')
        return mgr.driver.fitness(population=population)

    def mutate(self, population, mutation_rate):
        mgr = self.driver_managers.get('mutate')
        return mgr.driver.mutate(population=population, 
                                 pool=self.pool, 
                                 mutation_rate=mutation_rate)

    def populate(self):
        mgr = self.driver_managers.get('populate')
        return mgr.driver.populate(pool=self.pool, 
                                   initial_size=self.initial_size, 
                                   n_chromosomes=self.team_size)

    def validate(self, population):
        mgr = self.driver_managers.get('validate')
        return mgr.driver.validate(population=population)


if __name__ == '__main__':
    pass