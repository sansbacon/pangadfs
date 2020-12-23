# gadfs/gadfs/ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging


class GeneticAlgorithm:

    def __init__(self, driver_managers=None, extension_managers=None):
        logging.getLogger(__name__).addHandler(logging.NullHandler())
        self._pool = None
        self._pospool = None
        self.driver_managers = driver_managers
        self.extension_managers = extension_managers
        
    def crossover(self, **kwargs):
        if mgr := self.driver_managers.get('crossover'):
            return mgr.driver.crossover(**kwargs)
        population = kwargs['population']
        for ext in self.extension_managers.get('crossover'):
            kwargs['population'] = population
            population = ext.obj.crossover(**kwargs)
        return population

    def fitness(self, **kwargs):
        if mgr := self.driver_managers.get('fitness'):
            return mgr.driver.fitness(**kwargs)

    def mutate(self, **kwargs):
        if mgr := self.driver_managers.get('mutate'):
            return mgr.driver.mutate(**kwargs)

    def pool(self, **kwargs):
        if mgr := self.driver_managers.get('pool'):
            return mgr.driver.pool(**kwargs)           
        
    def populate(self, **kwargs):
        if mgr := self.driver_managers.get('populate'):
            return mgr.driver.populate(**kwargs)

    def pospool(self, **kwargs):
        if mgr := self.driver_managers.get('pospool'): 
            return mgr.driver.pospool(**kwargs)
        
    def validate(self, **kwargs):
        if mgr := self.driver_managers.get('validate'):
            return mgr.driver.validate(**kwargs)


if __name__ == '__main__':
    pass