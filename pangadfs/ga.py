# gadfs/gadfs/ga.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import logging


class GeneticAlgorithm:

    def __init__(self, driver_managers):
        logging.getLogger(__name__).addHandler(logging.NullHandler())
        self._pool = None
        self._pospool = None
        self.driver_managers = driver_managers

    def crossover(self, **kwargs):
        mgr = self.driver_managers.get('crossover')
        return mgr.driver.crossover(**kwargs)

    def fitness(self, **kwargs):
        mgr = self.driver_managers.get('fitness')
        return mgr.driver.fitness(**kwargs)

    def mutate(self, **kwargs):
        mgr = self.driver_managers.get('mutate')
        return mgr.driver.mutate(**kwargs)

    def pool(self, **kwargs):
        if not self._pool:
            mgr = self.driver_managers.get('pool')
            return mgr.driver.pool(**kwargs)           
        return self._pool
        
    def populate(self, **kwargs):
        mgr = self.driver_managers.get('populate')
        return mgr.driver.populate(**kwargs)

    def pospool(self, **kwargs):
        if not self._pospool:
            mgr = self.driver_managers.get('pospool')
            return mgr.driver.pospool(**kwargs)           
        return self._pool
        
    def validate(self, **kwargs):
        mgr = self.driver_managers.get('validate')
        return mgr.driver.validate(**kwargs)


if __name__ == '__main__':
    pass