# pangadfs/tests/test_base.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the Apache 2.0 License

import pytest
from pangadfs.base import *


class Crossover(CrossoverBase):
    def t(self):
        return None


class Fitness(FitnessBase):
    def t(self):
        return None


class Mutate(MutateBase):
    def t(self):
        return None


class Pool(PoolBase):
    def t(self):
        return None


class Populate(PopulateBase):
    def t(self):
        return None


class Pospool(PospoolBase):
    def t(self):
        return None


class Validate(ValidateBase):
    def t(self):
        return None


def test_crossover_base():
    with pytest.raises(TypeError):
        obj = Crossover()    


def test_fitness_base():
    with pytest.raises(TypeError):
        obj = Fitness()


def test_mutate_base():
    with pytest.raises(TypeError):
        obj = Mutate()


def test_populate_base():
    with pytest.raises(TypeError):
        obj = Populate()


def test_pool_base():
    with pytest.raises(TypeError):
        obj = Pool()


def test_pospool_base():
    with pytest.raises(TypeError):
        obj = Pospool()


def test_validate_base():
    with pytest.raises(TypeError):
        obj = Validate()
